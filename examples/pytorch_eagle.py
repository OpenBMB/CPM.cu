from eagle.model.ea_model import EaModel
from transformers import AutoTokenizer
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="/cache/lizhen/repos/temp-cpm/CPM.cu/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--eagle_model", type=str, default="/cache/lizhen/repos/temp-cpm/CPM.cu/models/EAGLE-LLaMA3.1-Instruct-8B")
    parser.add_argument("--prompt", type=str, default="who are you?")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)  # 改为0.0获得更确定的输出
    
    # ===== 🎯 Prompt格式选项 =====
    parser.add_argument("--use_chat_template", action="store_true", default=True,
                       help="使用LLaMA的chat template格式")
    parser.add_argument("--use_direct_prompt", action="store_true", 
                       help="使用直接的指令格式")
    parser.add_argument("--no_chat_template", action="store_true",
                       help="使用原生prompt格式（不使用chat template）")
    
    # ===== 🎯 EAGLE Speculative Decoding参数 =====
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=32, 
                       help="总的draft token数量 (对应total_token)")
    parser.add_argument("--speculative-num-steps", type=int, default=5,
                       help="生成树的深度/步数 (对应depth)")
    parser.add_argument("--speculative-eagle-topk", type=int, default=8,
                       help="每层生成的候选token数量 (对应top_k)")
    parser.add_argument("--threshold", type=float, default=1.0,
                       help="接受阈值")
    args = parser.parse_args()

    print(f"加载 base 模型: {args.base_model}")
    print(f"加载 EAGLE 草稿模型: {args.eagle_model}")
    print(f"🎯 EAGLE参数设置:")
    print(f"  - Draft Tokens数量: {args.speculative_num_draft_tokens}")
    print(f"  - 生成步数: {args.speculative_num_steps}")  
    print(f"  - TopK: {args.speculative_eagle_topk}")
    print(f"  - 阈值: {args.threshold}")
    print(f"  - Temperature: {args.temperature}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = EaModel.from_pretrained(
        use_eagle3=False,  # 对于yuhuili/EAGLE-LLaMA3.1-Instruct-8B模型使用False
        base_model_path=args.base_model,
        ea_model_path=args.eagle_model,
        # ===== 🎯 EAGLE核心参数设置 =====
        total_token=args.speculative_num_draft_tokens,
        depth=args.speculative_num_steps,
        top_k=args.speculative_eagle_topk,
        threshold=args.threshold,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    # 使用外部tokenizer替换模型内部的tokenizer
    model.tokenizer = tokenizer

    # ===== 🎯 改进Prompt格式 =====
    if args.use_direct_prompt:
        # 使用更直接的指令格式
        formatted_prompt = f"Please answer this question directly and concisely: {args.prompt}\nAnswer:"
        print(f"📝 使用直接指令格式")
    elif args.no_chat_template:
        # 使用原生prompt格式
        formatted_prompt = args.prompt
        print(f"📝 使用原生prompt格式")
    elif args.use_chat_template:
        # 使用LLaMA的chat template
        messages = [
            {"role": "user", "content": args.prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"📝 使用Chat Template格式")
    else:
        # 默认使用原生prompt格式
        formatted_prompt = args.prompt
        print(f"📝 使用原生prompt格式（默认）")
    
    print(f"🔤 最终prompt: {repr(formatted_prompt)}")

    input_ids = tokenizer([formatted_prompt]).input_ids
    input_ids = torch.tensor(input_ids).cuda()
    
    # 记录原始输入长度
    input_length = input_ids.shape[1]

    output_ids = model.eagenerate(
        input_ids,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )

    # 只解码新生成的部分
    generated_tokens = output_ids[0, input_length:]  # 跳过原始输入部分
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 同时显示完整输出和仅生成部分
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("📊 生成结果:")
    print("="*60)
    print("✅ 完整输出（包含prompt）：", full_output)
    print("\n🎯 仅生成内容：", generated_text)
