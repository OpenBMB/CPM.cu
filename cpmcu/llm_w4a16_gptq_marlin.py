from . import C

import os, json, glob
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from safetensors.torch import load_file
import time, math
import torch.nn.functional as F
from .common.logging import logger

dtype_map = {
    torch.float16: 0,
    torch.bfloat16: 1,
}

def dtype_to_int(dtype):
    ret = dtype_map.get(dtype, -1)
    if ret == -1:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return ret

class W4A16GPTQMarlinLLM(torch.nn.Module):
    def __init__(self,
                 path: str, # hf model path
                 memory_limit: float = 0.8,
                 chunk_length: int = 1024,
                 dtype: torch.dtype = None,
                 cuda_graph: bool = False,
                 apply_sparse: bool = False,
                 sink_window_size: int = 1,
                 block_window_size: int = 32,
                 sparse_topk_k: int = 32,
                 sparse_switch: int = 8192,
                 use_compress_lse: bool = False,
                 use_qk_norm: bool = False,
                 use_attn_bias: bool = False,
                 temperature: float = 0.0,
                 random_seed = None,
    ):
        super().__init__()

        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        self.dtype = dtype if dtype is not None else self.config.torch_dtype
        self.dtype_int = dtype_to_int(self.dtype)
        self.cuda_graph = cuda_graph
        self.temperature = temperature
        self.chunk_length = chunk_length
        
        # Initialize random generator if random_seed is provided
        if random_seed is not None:
            self.generator = torch.Generator(device="cuda")
            self.generator.manual_seed(random_seed)
        else:
            self.generator = None
        
        # For Qwen3, head_dim is explicitly specified in config and may not equal hidden_size // num_attention_heads
        if not hasattr(self.config, "head_dim"):
            self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        else:
            # Qwen3 models have explicit head_dim that might be different
            logger.info(f"Using explicit head_dim from config: {self.config.head_dim}")
        
        self.group_size = self.config.quantization_config['group_size']
        scale_embed = self.config.scale_emb if hasattr(self.config, "scale_emb") else 1.0
        scale_lmhead = (self.config.dim_model_base / self.config.hidden_size) if hasattr(self.config, "dim_model_base") else 1.0
        scale_residual = self.config.scale_depth / math.sqrt(self.config.num_hidden_layers) if hasattr(self.config, "scale_depth") else 1.0

        if apply_sparse:
            C.init_w4a16_gptq_marlin_minicpm4_model(
                memory_limit,
                self.config.vocab_size,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rms_norm_eps,
                self.group_size,
                self.dtype_int,
                self.chunk_length,
                scale_embed,
                scale_lmhead,
                scale_residual,
                sink_window_size,
                block_window_size,
                sparse_topk_k,
                sparse_switch,
                use_compress_lse,
            )
        else:
            C.init_w4a16_gptq_marlin_base_model(
                memory_limit,
                self.config.vocab_size,
                self.config.num_hidden_layers,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rms_norm_eps,
                self.group_size,
                self.dtype_int,
                self.chunk_length,
                scale_embed,
                scale_lmhead,
                scale_residual,
                use_qk_norm,
                use_attn_bias,
            )

        self.logits = torch.empty((64, self.config.vocab_size), dtype=self.dtype, device="cuda")

    def init_storage(self):
        self.max_total_length = C.init_storage()

    def _load(self, name, param, dtype=None, cls=None):
        if dtype is None:
            if 'rotary_emb' in name:
                dtype = torch.float32
            else:
                dtype = self.dtype

        # if 'gate_up_proj' in name:
        #     self._load(name.replace("gate_up_proj", "gate_proj"), param[:param.shape[0]//2], dtype)
        #     self._load(name.replace("gate_up_proj", "up_proj"), param[param.shape[0]//2:])
        # elif 'qkv_proj' in name:
        #     self._load(name.replace("qkv_proj", "q_proj"), param[:self.config.num_attention_heads * self.config.head_dim])
        #     self._load(name.replace("qkv_proj", "k_proj"), param[self.config.num_attention_heads * self.config.head_dim:(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim])
        #     self._load(name.replace("qkv_proj", "v_proj"), param[(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim:])
        # else:
        param = param.contiguous()
        if param.dtype not in [torch.int8, torch.int16, torch.int32]:
            param = param.to(dtype)
        C.load_model(name, param.data_ptr())

        if "embed_tokens" in name and hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
            self._load("lm_head.weight", param)

    def _load_from_ckpt(self, path, cls=None):
        supported_suffix_1 = ["bin.index.json", "safetensors.index.json"]
        supported_suffix_2 = ["bin", "safetensors", "pt"]
        file = None
        for suffix in supported_suffix_1:
            files = glob.glob(os.path.join(path, f"*.{suffix}"))
            if len(files) > 1:
                raise ValueError(f"Multiple files with suffix {suffix} found in {path}")
            elif len(files) == 1:
                file = files[0]
                break
        else:
            for suffix in supported_suffix_2:
                files = glob.glob(os.path.join(path, f"*.{suffix}"))
                if len(files) > 1:
                    logger.info(f"Found files: {files}")
                    if path + "/model_gptq_marlin.safetensors" in files:
                            file = path + "/model_gptq_marlin.safetensors"
                    else:
                        raise ValueError(f"Autogptq models not found in {path}")
                    break
                elif len(files) == 1:
                    file = files[0]
                    break
            else:
                raise ValueError(f"No supported checkpoint file found in {path}, supported suffixes: {supported_suffix_1 + supported_suffix_2}")

        if file.endswith(".index.json"):
            with open(file, "r") as f:
                file_list = set(json.load(f)["weight_map"].values())
            file_list = [os.path.join(path, file) for file in file_list]
        else:
            file_list = [file]

        for file in file_list:
            logger.info(f"load from {file}")
            if file.endswith(".bin") or file.endswith(".pt"):
                ckpt = torch.load(file, map_location="cpu")
            elif file.endswith(".safetensors"):
                ckpt = load_file(file)
            for name, param in ckpt.items():
                self._load(name, param, cls=cls)

    def load_from_hf(self):
        with torch.no_grad():
            self._load_from_ckpt(self.path)

            # rope
            if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
                rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
                if rope_type == "longrope" and not hasattr(self.config.rope_scaling, "factor"):
                    self.config.rope_scaling["factor"] = 1.0
            else:
                rope_type = "default"
            # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, "cpu", seq_len=self.max_total_length)
            # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
            self._load("model.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
            # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)

    def prefill(self, input_ids, position_ids, progress_callback=None):
        assert input_ids.dtype == torch.int32
        # Check if input length exceeds maximum supported length
        if input_ids.numel() > self.max_total_length:
            raise ValueError(f"Input token count ({input_ids.numel()}) exceeds maximum supported length ({self.max_total_length}) under current memory limit")
        
        total_length = input_ids.numel()
        num_chunks = (total_length + self.chunk_length - 1) // self.chunk_length
        
        actual_prefill_start = time.time()
        
        # Initialize progress callback if provided
        if progress_callback:
            progress_callback('begin', {'total_tokens': total_length})
        
        for chunk_idx, i in enumerate(range(0, input_ids.numel(), self.chunk_length)):
            # torch.cuda.nvtx.range_push(f"chunk from {i}")
            C.prefill(
                min(input_ids.numel() - i, self.chunk_length), i,
                input_ids.view(-1)[i:].data_ptr(), position_ids.view(-1)[i:].data_ptr(),
                self.logits.data_ptr()
            )
            # torch.cuda.nvtx.range_pop()
            
            # Update progress via callback
            if progress_callback:
                current_tokens = min(i + self.chunk_length, total_length)
                progress_callback('advance', {'current_tokens': current_tokens})
        
        # Calculate actual prefill time
        actual_prefill_time = time.time() - actual_prefill_start
        
        # Complete progress via callback
        if progress_callback:
            progress_callback('finish', {'total_time': actual_prefill_time})
        
        # Store the actual prefill time for use in generate method
        self._last_prefill_time = actual_prefill_time
        
        return self.logits[:1].clone()

    def decode(self, input_ids, position_ids, cache_length, mask_2d = None):
        assert input_ids.dtype == torch.int32
        assert position_ids.dtype == torch.int32
        assert cache_length.dtype == torch.int32
        if mask_2d is not None:
            # assert mask_2d.dtype == torch.int64
            assert input_ids.numel() == mask_2d.shape[0]

        # torch.cuda.nvtx.range_push(f"decode")
        cache_length += input_ids.numel() # temparary add for convinience in flash_attn
        padded_length = (cache_length[0].item() + 128 - 1) // 128 * 128
        C.decode(
            input_ids.numel(), padded_length,
            input_ids.data_ptr(), position_ids.data_ptr(), cache_length.data_ptr(),
            mask_2d.data_ptr() if mask_2d is not None else 0,
            self.logits.data_ptr(),
            self.cuda_graph
        )
        cache_length -= input_ids.numel()
        # torch.cuda.nvtx.range_pop()
        return self.logits[:input_ids.numel()].clone()

    def generate(self, input_ids, generation_length=100, teminators=[], use_stream=False, progress_callback=None):
        """
        Generate text with optional streaming output.
        Returns (tokens, decode_time, prefill_time) if use_stream=False, or generator yielding {'token', 'text', 'is_finished', 'prefill_time', 'decode_time'} if use_stream=True.
        """
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        
        # Measure prefill time
        torch.cuda.synchronize()
        prefill_start = time.time()
        logits = self.prefill(input_ids, position_ids, progress_callback)
        torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
        
        if self.temperature > 0.0:
            token = torch.multinomial(F.softmax(logits[0]/self.temperature, dim=-1), num_samples=1, generator=self.generator)[0].item()
        else:   
            token = logits[0].argmax(dim=-1).item()

        if not hasattr(self, "input_ids"):
            self.input_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.position_ids = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")

        if use_stream:
            # Stream generation (optimized)
            def _stream_generator():
                nonlocal token
                # Keep minimal context for correct spacing
                prev_token = token
                
                # yield first token
                text = self.tokenizer.decode([token], skip_special_tokens=True)
                
                yield {
                    'token': token,
                    'text': text,
                    'is_finished': token in teminators,
                    'prefill_time': prefill_time,
                    'decode_time': 0.0  # First token comes from prefill
                }
                
                if token in teminators:
                    return

                decode_start_time = time.time()
                
                for i in range(generation_length-1):
                    self.input_ids[0] = token
                    self.position_ids[0] = prefix_length + i
                    self.cache_length[0] = prefix_length + i

                    logits = self.decode(self.input_ids, self.position_ids, self.cache_length)
                    if self.temperature > 0.0:
                        token = torch.multinomial(F.softmax(logits[0]/self.temperature, dim=-1), num_samples=1, generator=self.generator)[0].item()
                    else:   
                        token = logits[0].argmax(dim=-1).item()
                    
                    # For correct spacing, decode with previous token context
                    if prev_token is not None:
                        context_tokens = [prev_token, token]
                        context_text = self.tokenizer.decode(context_tokens, skip_special_tokens=True)
                        prev_text = self.tokenizer.decode([prev_token], skip_special_tokens=True)
                        text = context_text[len(prev_text):]
                    else:
                        text = self.tokenizer.decode([token], skip_special_tokens=True)
                    
                    is_finished = token in teminators or i == generation_length - 2
                    
                    # Calculate time only when needed to reduce overhead
                    decode_time = time.time() - decode_start_time
                        
                    yield {
                        'token': token,
                        'text': text,
                        'is_finished': is_finished,
                        'prefill_time': 0.0,  # Only report prefill_time for first token
                        'decode_time': decode_time
                    }
                    
                    if token in teminators:
                        break
                    
                    # Update prev_token
                    prev_token = token
            
            return _stream_generator()
        else:
            # Original batch generation
            tokens = [token]
            torch.cuda.synchronize()
            decode_start = time.time()
            for i in range(generation_length-1):
                self.input_ids[0] = token
                self.position_ids[0] = prefix_length + i
                self.cache_length[0] = prefix_length + i

                logits = self.decode(self.input_ids, self.position_ids, self.cache_length)
                if self.temperature > 0.0:
                    token = torch.multinomial(F.softmax(logits[0]/self.temperature, dim=-1), num_samples=1, generator=self.generator)[0].item()
                else:
                    token = logits[0].argmax(dim=-1).item()
                tokens.append(token)
                if token in teminators:
                    break
            torch.cuda.synchronize()
            decode_time = time.time() - decode_start
            return tokens, decode_time, prefill_time

    def print_perf_summary(self):
        C.print_perf_summary()