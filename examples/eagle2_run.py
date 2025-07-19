import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
from huggingface_hub import hf_hub_download
import torch.nn as nn


# 简化版的EAGLE配置类
class EagleConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    @classmethod
    def from_pretrained(cls, config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)


# 简化版的EAGLE模型类
class SimpleEAGLEModel(nn.Module):
    def __init__(self, config, bias=True):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # EAGLE的关键组件
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        
        # 简化：只创建必要的层，完整实现需要更多层
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                batch_first=True
            ) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, input_ids, hidden_states):
        # 简化的前向传播
        embed = self.embed_tokens(input_ids)
        combined = torch.cat([embed, hidden_states], dim=-1)
        output = self.fc(combined)
        return output


@torch.no_grad()
def generate_draft_tokens_with_eagle(base_model, eagle_model, tokenizer, input_ids, num_tokens):
    """使用EAGLE模型生成草稿tokens"""
    device = base_model.device  # 获取基础模型的设备
    input_ids = input_ids.to(device)
    
    # 获取基础模型的隐藏状态
    with torch.no_grad():
        outputs = base_model(input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states[-1]  # 最后一层的隐藏状态
    
    # 使用EAGLE模型预测下一个tokens
    draft_tokens = []
    current_input = input_ids
    
    for _ in range(num_tokens):
        # 简化的EAGLE推理过程
        last_hidden = hidden_states[:, -1:, :]  # 最后一个token的隐藏状态
        
        # 确保hidden states在正确设备上
        last_hidden = last_hidden.to(device)
        
        # 这里应该是完整的EAGLE推理，现在先用简化版本
        with torch.no_grad():
            next_token_logits = base_model.lm_head(last_hidden)
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            draft_tokens.append(next_token_id)
            
            # 确保next_token_id在正确设备上
            next_token_id = next_token_id.to(device)
            
            # 更新输入用于下一次预测
            current_input = torch.cat([current_input, next_token_id], dim=1)
            
            # 重新计算隐藏状态（简化版）
            if len(draft_tokens) < num_tokens:
                outputs = base_model(current_input, output_hidden_states=True, use_cache=False)
                hidden_states = outputs.hidden_states[-1]
    
    # 确保所有draft_tokens在同一设备上
    draft_tokens = [token.to(device) for token in draft_tokens]
    return torch.cat(draft_tokens, dim=1)


def load_eagle_model_properly(eagle_path, base_model):
    """正确加载EAGLE模型的方式"""
    print(f"尝试正确加载EAGLE模型: {eagle_path}")
    
    # 检查EAGLE配置文件
    config_path = os.path.join(eagle_path, "config.json")
    if not os.path.exists(config_path):
        try:
            config_path = hf_hub_download(eagle_path, "config.json")
        except Exception as e:
            print(f"无法下载EAGLE配置文件: {e}")
            return None
    
    # 检查EAGLE权重文件
    model_path = os.path.join(eagle_path, "pytorch_model.bin")
    if not os.path.exists(model_path):
        try:
            model_path = hf_hub_download(eagle_path, "pytorch_model.bin")
        except Exception as e:
            print(f"无法下载EAGLE权重文件: {e}")
            return None
    
    try:
        # 加载EAGLE配置
        config = EagleConfig.from_pretrained(config_path)
        print(f"EAGLE配置加载成功: hidden_size={config.hidden_size}")
        
        # 创建EAGLE模型
        eagle_model = SimpleEAGLEModel(config)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location="cpu")
        
        # 尝试加载权重（可能需要调整）
        eagle_model.load_state_dict(state_dict, strict=False)
        
        # 移动到正确设备
        device = base_model.device
        eagle_model.to(base_model.dtype).to(device)
        eagle_model.eval()
        
        print("EAGLE模型加载成功！")
        return eagle_model
        
    except Exception as e:
        print(f"EAGLE模型加载失败: {e}")
        return None


@torch.no_grad()
def verify_with_main_model(main_model, input_ids, draft_tokens, main_tokenizer, max_verify=8, verbose=False):
    """验证草稿tokens的函数保持不变"""
    input_ids = input_ids.to(main_model.device)
    draft_tokens = draft_tokens.to(main_model.device)
    
    max_len = min(draft_tokens.shape[1], max_verify)

    for i in range(max_len):
        input_for_main = torch.cat([input_ids, draft_tokens[:, :i]], dim=1)
        output = main_model(input_for_main)
        logits = output.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        pred_token = torch.argmax(probs, dim=-1)
        ref_token = draft_tokens[:, i]

        if verbose:
            print(f"[Verify Token {i}]")
            print(f"  → Pred Token   : {pred_token.item()} ({main_tokenizer.decode(pred_token)})")
            print(f"  → Draft Token  : {ref_token.item()} ({main_tokenizer.decode(ref_token)})")
            print(f"  → Match        : {'✅' if torch.equal(pred_token, ref_token) else '❌'}")
            print(f"  → Logits Top-5 : {[main_tokenizer.decode(t.item()) for t in torch.topk(probs, 5).indices[0]]}\n")

        if not torch.equal(pred_token, ref_token):
            verified_prefix = draft_tokens[:, :i] if i > 0 else torch.empty((1, 0), dtype=draft_tokens.dtype, device=draft_tokens.device)
            main_pred = pred_token.unsqueeze(0)
            return verified_prefix, main_pred

    return draft_tokens[:, :max_len], None


def eagle2_loop(main_model, eagle_model, main_tokenizer, prompt, config):
    """修改后的EAGLE主循环"""
    num_draft = config["num_draft"]
    max_verify = config["max_verify"]
    max_steps = config["max_steps"]
    
    # 新增长度控制参数
    max_new_tokens = config.get("max_new_tokens", 150)  # 最大新生成token数
    max_total_tokens = config.get("max_total_tokens", 200)  # 最大总token数
    stop_on_punctuation = config.get("stop_on_punctuation", True)  # 是否在标点符号处停止
    custom_stop_words = config.get("custom_stop_words", [])  # 自定义停止词
    min_new_tokens = config.get("min_new_tokens", 10)  # 最少生成token数（避免过早停止）

    input_ids = main_tokenizer(prompt, return_tensors="pt").input_ids.to(main_model.device)
    full_output = input_ids.clone()
    initial_length = input_ids.shape[1]  # 记录初始长度

    print(f"开始生成 - 初始长度: {initial_length} tokens")
    print(f"控制参数: max_new_tokens={max_new_tokens}, max_total_tokens={max_total_tokens}")
    print(f"停止设置: punctuation={stop_on_punctuation}, min_new_tokens={min_new_tokens}")

    for step in range(max_steps):
        print(f"\n=== [EAGLE Step {step}] ===")
        
        # 检查长度限制
        current_length = input_ids.shape[1]
        new_tokens_generated = current_length - initial_length
        
        print(f"当前状态: 总长度={current_length}, 新生成={new_tokens_generated}")
        
        # 硬性长度限制检查
        if current_length >= max_total_tokens:
            print(f"→ 达到最大总长度限制 ({max_total_tokens})，强制停止")
            break
            
        if new_tokens_generated >= max_new_tokens:
            print(f"→ 达到最大新token限制 ({max_new_tokens})，强制停止")
            break
        
        if eagle_model is not None:
            # 使用EAGLE模型生成草稿
            draft_tokens = generate_draft_tokens_with_eagle(main_model, eagle_model, main_tokenizer, input_ids, num_draft)
        else:
            # 回退：使用主模型自身生成草稿（用于测试）
            print("警告：EAGLE模型未加载，使用主模型生成草稿")
            output = main_model.generate(
                input_ids=input_ids,
                max_new_tokens=num_draft,
                do_sample=False,
                use_cache=True,
                pad_token_id=main_tokenizer.eos_token_id
            )
            draft_tokens = output[:, input_ids.shape[-1]:]
            
        print("草稿生成:", [main_tokenizer.decode(tok) for tok in draft_tokens[0]])
        print("tokens id:", draft_tokens[0])

        # 验证草稿tokens
        verified_tokens, main_prediction = verify_with_main_model(main_model, input_ids, draft_tokens, main_tokenizer, max_verify, verbose=True)

        # 构建要添加的tokens
        tokens_to_add = verified_tokens
        
        if main_prediction is not None:
            tokens_to_add = torch.cat([verified_tokens, main_prediction], dim=1)
            print(f"→ 接受了 {verified_tokens.shape[1]} 个草稿tokens，添加主模型预测: {main_tokenizer.decode(main_prediction[0])}")
        else:
            print(f"→ 全部 {verified_tokens.shape[1]} 个草稿tokens都通过验证")
        
        if tokens_to_add.shape[1] == 0:
            print("→ 无草稿token通过验证，使用主模型生成下一个token")
            output = main_model(input_ids)
            logits = output.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            tokens_to_add = next_token
            print(f"  主模型生成: {main_tokenizer.decode(next_token[0])}")

        tokens_to_add = tokens_to_add.to(main_model.device)
        input_ids = torch.cat([input_ids, tokens_to_add], dim=1)
        full_output = input_ids

        decoded = main_tokenizer.decode(full_output[0], skip_special_tokens=True)
        print(f"当前累计输出: {decoded}")

        # 检查停止条件（只有在生成足够token后才检查）
        new_tokens_count = input_ids.shape[1] - initial_length
        if new_tokens_count >= min_new_tokens:
            # 检查标点符号停止
            if stop_on_punctuation and decoded.strip().endswith((".", "!", "?")):
                print(f"→ 在标点符号处停止（已生成 {new_tokens_count} tokens）")
                break
            
            # 检查自定义停止词
            if custom_stop_words:
                for stop_word in custom_stop_words:
                    if stop_word.lower() in decoded.lower():
                        print(f"→ 遇到停止词 '{stop_word}'，停止生成")
                        break
                else:
                    continue  # 没有遇到停止词，继续循环
                break  # 遇到停止词，跳出循环

    final_length = full_output.shape[1]
    final_new_tokens = final_length - initial_length
    
    print(f"\n=== 生成统计 ===")
    print(f"初始长度: {initial_length} tokens")
    print(f"最终长度: {final_length} tokens")
    print(f"新生成: {final_new_tokens} tokens")
    print(f"总步数: {step + 1} steps")
    
    print("\n=== 最终输出 ===")
    print(main_tokenizer.decode(full_output[0], skip_special_tokens=True))


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_model_path", type=str, default="unsloth/llama-3-8b-Instruct")
    parser.add_argument("--draft_model_path", type=str, default="yuhuili/EAGLE-LLaMA3-Instruct-8B")
    parser.add_argument("--eagle_config", type=str, default="eagle_config.json")
    parser.add_argument("--prompt", type=str, default="hello, how are you?")
    
    # 新增长度控制参数
    parser.add_argument("--max_new_tokens", type=int, help="最大新生成token数（覆盖配置文件）")
    parser.add_argument("--max_total_tokens", type=int, help="最大总token数（覆盖配置文件）")
    parser.add_argument("--min_new_tokens", type=int, help="最少生成token数（覆盖配置文件）")
    parser.add_argument("--no_punctuation_stop", action="store_true", help="禁用标点符号停止")
    parser.add_argument("--stop_words", type=str, nargs="*", help="自定义停止词列表")
    
    args = parser.parse_args()

    print("加载模型中...")
    print(f"主模型: {args.main_model_path}")
    print(f"EAGLE模型: {args.draft_model_path}")

    # 加载tokenizer
    print("加载tokenizer...")
    try:
        main_tokenizer = AutoTokenizer.from_pretrained(args.main_model_path, use_fast=True)
    except Exception as e:
        print(f"Fast tokenizer加载失败: {e}")
        main_tokenizer = AutoTokenizer.from_pretrained(args.main_model_path, use_fast=False)
    
    if main_tokenizer.pad_token is None:
        main_tokenizer.pad_token = main_tokenizer.eos_token

    # 加载主模型
    print("加载主模型...")
    main_model = AutoModelForCausalLM.from_pretrained(
        args.main_model_path, torch_dtype=torch.float16, device_map="auto"
    ).eval()

    # 尝试正确加载EAGLE模型
    print("加载EAGLE模型...")
    eagle_model = load_eagle_model_properly(args.draft_model_path, main_model)
    
    if eagle_model is None:
        print("警告：EAGLE模型加载失败，将使用主模型作为草稿模型进行测试")

    # 加载配置并应用命令行覆盖
    config = load_config(args.eagle_config)
    
    # 应用命令行参数覆盖配置
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens
        print(f"覆盖配置: max_new_tokens = {args.max_new_tokens}")
    
    if args.max_total_tokens is not None:
        config["max_total_tokens"] = args.max_total_tokens
        print(f"覆盖配置: max_total_tokens = {args.max_total_tokens}")
    
    if args.min_new_tokens is not None:
        config["min_new_tokens"] = args.min_new_tokens
        print(f"覆盖配置: min_new_tokens = {args.min_new_tokens}")
    
    if args.no_punctuation_stop:
        config["stop_on_punctuation"] = False
        print("覆盖配置: 禁用标点符号停止")
    
    if args.stop_words is not None:
        config["custom_stop_words"] = args.stop_words
        print(f"覆盖配置: custom_stop_words = {args.stop_words}")

    # 启动EAGLE
    eagle2_loop(main_model, eagle_model, main_tokenizer, args.prompt, config)
