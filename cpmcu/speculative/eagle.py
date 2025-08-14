from .. import C
from .tree_drafter import LLM_with_tree_drafter
import math, torch
from transformers import PretrainedConfig
from ..common.logging import logger

class EagleConfig(PretrainedConfig):
    def __init__(
        self,
        num_hidden_layers=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eagle_num_layers = num_hidden_layers

class LLM_with_eagle(LLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 eagle_window_size=0,
                 frspec_vocab_size=0,
                 apply_eagle_quant: bool=False,
                 use_rope: bool=False,
                 use_input_norm: bool=False,
                 use_attn_norm: bool=False,
                 use_eagle3: bool=False,
                 **kwargs):
        
        drafter_type = "eagle3" if use_eagle3 else "eagle"

        super().__init__(
            drafter_type, eagle_path, base_path,
            tree_size = tree_size,
            **kwargs
        )

        self.eagle_path = eagle_path
        self.eagle_config = EagleConfig.from_pretrained(eagle_path)
        
        # For Qwen3, head_dim is explicitly specified in config and may not equal hidden_size // num_attention_heads
        if not hasattr(self.eagle_config, "head_dim"):
            self.eagle_config.head_dim = self.eagle_config.hidden_size // self.eagle_config.num_attention_heads
        else:
            # Qwen3 models have explicit head_dim that might be different
            logger.info(f"Using explicit head_dim from eagle config: {self.eagle_config.head_dim}")
        
        if not use_eagle3:
            # Ensure presence consistency and equality for scale_depth, dim_model_base, and scale_emb
            for attr in ("scale_depth", "dim_model_base", "scale_emb"):
                base_has = hasattr(self.config, attr)
                eagle_has = hasattr(self.eagle_config, attr)
                assert base_has == eagle_has, f"{attr} presence mismatch between base and eagle config"
                if base_has:
                    assert getattr(self.config, attr) == getattr(self.eagle_config, attr), f"{attr} in base config and eagle config should be the same"
        
        scale_residual = self.config.scale_depth / math.sqrt(self.config.num_hidden_layers + 1) if hasattr(self.config, "scale_depth") else 1.0
        self.apply_eagle_quant = apply_eagle_quant
        if apply_eagle_quant and hasattr(self.eagle_config, "quantization_config"):
            self.group_size = self.eagle_config.quantization_config.get('group_size', 0)
        else:
            self.group_size = 0
        assert self.group_size == 128 or self.group_size == 0, "only group_size 128 is supported in quantization mode"

        if use_eagle3:
            if not use_rope:
                scale_residual = 1.0
            
            C.init_minicpm4_eagle3_model(
                self.eagle_config.eagle_num_layers,
                self.eagle_config.intermediate_size,
                self.eagle_config.num_attention_heads,
                self.eagle_config.num_key_value_heads,
                self.eagle_config.head_dim,
                self.eagle_config.rms_norm_eps,
                num_iter,
                topk_per_iter,
                self.tree_size,
                self.dtype_int,
                apply_eagle_quant,
                self.group_size,
                eagle_window_size,
                self.eagle_config.draft_vocab_size,
                scale_residual
            )
        else:
            if not use_rope and not use_input_norm and not use_attn_norm and not apply_eagle_quant:
                C.init_eagle_model(
                    self.eagle_config.eagle_num_layers,
                    self.eagle_config.intermediate_size,
                    self.eagle_config.num_attention_heads,
                    self.eagle_config.num_key_value_heads,
                    self.eagle_config.head_dim,
                    self.eagle_config.rms_norm_eps,
                    num_iter,
                    topk_per_iter,
                    self.tree_size,
                    self.dtype_int
                )
            else:
                C.init_minicpm4_eagle_model(
                    self.eagle_config.eagle_num_layers,
                    self.eagle_config.intermediate_size,
                    self.eagle_config.num_attention_heads,
                    self.eagle_config.num_key_value_heads,
                    self.eagle_config.head_dim,
                    self.eagle_config.rms_norm_eps,
                    num_iter,
                    topk_per_iter,
                    self.tree_size,
                    self.dtype_int,
                    apply_eagle_quant,
                    self.group_size,
                    eagle_window_size,
                    frspec_vocab_size,
                    scale_residual,
                    use_input_norm, 
                    use_attn_norm
                )

    def _load(self, name, param, dtype=None, cls=None):
        if cls == "eagle":
            if name == "token_id_remap":
                C.load_model(f"{cls}.{name}", param.data_ptr())
                return
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous()
            if not self.apply_eagle_quant:
                param = param.to(dtype)
            if 'embed_tokens' in name:
                return
            if 'fc' in name:
                if 'weight' in name or "scales" in name:
                    param1 = param[..., :param.shape[-1] // 2].contiguous()
                    param2 = param[..., param.shape[-1] // 2:].contiguous()
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param1.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc2')}", param2.data_ptr())
                else: # bias
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param.data_ptr())
            else:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        elif cls == "eagle3":
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous()
            if 'd2t' in name:
                param = param.to(torch.int32)
                C.load_model(f"{cls}.{name}", param.data_ptr())
                return
            if not self.apply_eagle_quant:
                param = param.to(dtype)
            if 'embed_tokens' in name:
                return
            C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)
    
    def load_from_hf(self):
        super().load_from_hf()

        if self.drafter_type == "eagle3":
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.eagle_config.head_dim, 2).float() / self.eagle_config.head_dim))
            self._load(f"{self.drafter_type}.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
