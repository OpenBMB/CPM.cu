#!/usr/bin/env python3
"""
Direct Logits Comparison Script
Runs the same generation logic as test_generate.py but captures logits
"""

import sys
import os
import pickle
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from transformers import AutoTokenizer
from cpmcu.common.logging import logger
from cpmcu.common.utils import (
    setup_model_paths,
    create_model,
    setup_frspec_vocab,
    apply_minicpm4_yarn_config
)
from cpmcu.common.display import Display
from cpmcu.common.args import create_cli_parser


class LogitsCapture:
    """Class to capture logits during generation"""
    
    def __init__(self, config_name: str):
        self.config_name = config_name
        self.captured_logits = []  # List of (step_id, logits_tensor, step_type)
        self.captured_tokens = []  # List of (step_id, token_id)
        self.step_counter = 0
        self.generation_info = []
        self.temperature = 0.0
        
    def save_data(self, filename: str):
        """Save captured data"""
        data = {
            'config_name': self.config_name,
            'captured_logits': self.captured_logits,
            'captured_tokens': self.captured_tokens,
            'total_steps': len(self.captured_logits)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self.captured_logits)} logits steps to {filename}")


def patch_model_for_logits_capture(model, capture: LogitsCapture):
    """Patch the model to capture logits during generation"""
    
    original_generate = model.generate
    
    def patched_generate(*args, **kwargs):
        print(f"[{capture.config_name}] Generate called")
        
        # Store original methods
        original_prefill = model.prefill
        original_decode = model.decode
        
        decode_step_counter = [0]
        stored_logits = []  # Store all decode logits
        prefill_logits = [None]
        
        def patched_prefill(*prefill_args, **prefill_kwargs):
            print(f"[{capture.config_name}] Prefill called")
            print(f"[{capture.config_name}] Prefill args[0] length: {len(prefill_args[0])}")
            logits = original_prefill(*prefill_args, **prefill_kwargs)
            prefill_logits[0] = logits.detach().cpu().numpy() if logits is not None else None
            return logits
        
        def patched_decode(*decode_args, **decode_kwargs):
            decode_step_counter[0] += 1
            print(f"[{capture.config_name}] Decode step {decode_step_counter[0]}")
            logits = original_decode(*decode_args, **decode_kwargs)
            if logits is not None:
                stored_logits.append(logits.detach().cpu().numpy())
            return logits
        
        # Temporarily patch methods
        model.prefill = patched_prefill
        model.decode = patched_decode
        
        try:
            # Call original generate
            results = original_generate(*args, **kwargs)
            
            # Process results to extract accepted tokens and their logits
            if isinstance(results, tuple) and len(results) >= 2:
                tokens, accept_lengths = results[:2]
                
                # Capture prefill token and its logits
                if len(tokens) > 0 and prefill_logits[0] is not None:
                    capture.step_counter += 1
                    step_id = capture.step_counter
                    first_token = tokens[0]
                    
                    # For prefill, take the logits from the last position
                    if len(prefill_logits[0].shape) > 1:
                        # Multi-dimensional logits, take last position
                        token_logits = prefill_logits[0][-1] if prefill_logits[0].shape[0] > 1 else prefill_logits[0][0]
                    else:
                        token_logits = prefill_logits[0]
                    
                    capture.captured_logits.append((step_id, token_logits.copy(), "accepted_prefill"))
                    capture.captured_tokens.append((step_id, first_token))
                    print(f"[{capture.config_name}] Step {step_id} (accepted_prefill): Token {first_token}, Logits preview: {token_logits[:5].tolist()}")
                
                # Process decode steps - only capture accepted tokens
                token_idx = 1  # Start from second token (first is from prefill)
                
                for step_idx, accept_length in enumerate(accept_lengths):
                    if step_idx >= len(stored_logits):
                        break
                    if token_idx >= len(tokens):
                        break
                    
                    step_logits = stored_logits[step_idx]
                    
                    # Only capture the accepted tokens from this decode step
                    for j in range(accept_length):
                        if token_idx + j >= len(tokens):
                            break
                        if j >= len(step_logits):
                            break
                        
                        capture.step_counter += 1
                        step_id = capture.step_counter
                        token = tokens[token_idx + j]
                        token_logits = step_logits[j]  # j-th token's logits in this step
                        
                        capture.captured_logits.append((step_id, token_logits.copy(), f"accepted_decode_{step_idx}_{j}"))
                        capture.captured_tokens.append((step_id, token))
                        print(f"[{capture.config_name}] Step {step_id} (accepted_decode_{step_idx}_{j}): Token {token}, Logits preview: {token_logits[:5].tolist()}")
                    
                    token_idx += accept_length
            
            return results
            
        finally:
            # Restore original methods
            model.prefill = original_prefill
            model.decode = original_decode
    
    # Replace generate method
    model.generate = patched_generate
    
    def restore():
        model.generate = original_generate
    
    return restore


def run_generation_with_config(spec_num_iter: int, spec_tree_size: int, config_name: str, comparison_steps: int = 20, use_chat_template: bool = False):
    """Run generation with specific configuration and capture logits"""
    
    # Calculate appropriate topk_per_iter to satisfy constraint: topk_per_iter <= tree_size - 1
    max_topk_per_iter = spec_tree_size - 1
    topk_per_iter = min(8, max_topk_per_iter)  # Default is 8, but ensure it's valid
    
    if topk_per_iter <= 0:
        raise ValueError(f"tree_size ({spec_tree_size}) too small, must be at least 2")
    
    print(f"Using topk_per_iter={topk_per_iter} (max allowed: {max_topk_per_iter})")
    
    # Parse arguments similar to test_generate.py
    parser = create_cli_parser()
    
    # Build argument list
    args_list = [
        # "--model-path", "openbmb/MiniCPM4-8B-marlin-cpmcu",
        # "--draft-model-path", "openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu",
        "--model-path", "unsloth/Meta-Llama-3.1-8B-Instruct",
        "--draft-model-path", "jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B",
        # "--frspec-path", "openbmb/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu",
        "--prompt-file", "../prompt.txt",
        "--spec-num-iter", str(spec_num_iter),
        "--spec-topk-per-iter", str(topk_per_iter),  # Use calculated value
        "--spec-tree-size", str(spec_tree_size),
        "--num-generate", str(comparison_steps),  # Use comparison_steps for generation length
        "--minicpm4-yarn",
        "--memory-limit", "0.8",  # Set very conservative memory limit to prevent OOM
        "--chunk-length", "1024",  # Reduce chunk length to save memory
        "--cuda-graph", "false",  # Disable CUDA graph to save memory
    ]
    
    # Add chat template argument based on parameter
    if use_chat_template:
        args_list.extend(["--use-chat-template", "true"])
    else:
        args_list.extend(["--use-chat-template", "false"])
    
    args = parser.parse_args(args_list)
    
    print(f"\n{'='*60}")
    print(f"Running {config_name}")
    print(f"spec_num_iter: {spec_num_iter}, spec_tree_size: {spec_tree_size}")
    print(f"{'='*60}")
    
    # Configure display
    Display.configure(use_plain_mode=args.plain_output)
    
    # Setup model paths
    config = vars(args)
    model_path, draft_model_path, frspec_path = setup_model_paths(config)
    
    # Clear GPU memory before model creation
    torch.cuda.empty_cache()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Create model with error handling
    try:
        llm = create_model(model_path, draft_model_path, config)
    except Exception as e:
        print(f"Error creating model: {e}")
        # Try to free memory
        torch.cuda.empty_cache()
        return None
    
    # Load prompt
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read().strip()
    
    # Apply chat template
    if args.use_chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_content}], 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        prompt = prompt_content
    
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda", dtype=torch.int32)
    print(f"Input tokens: {input_ids.shape[1]}")
    
    # Initialize model storage
    llm.init_storage()
    
    # Apply MiniCPM4 YARN configuration
    if args.minicpm4_yarn:
        apply_minicpm4_yarn_config(llm)
    
    # Load frequency speculative vocabulary
    has_speculative = draft_model_path is not None
    if has_speculative and (frspec_path is not None) and (args.frspec_vocab_size > 0):
        setup_frspec_vocab(llm, frspec_path, args.frspec_vocab_size)
    
    # Load model weights
    logger.info("Loading model weights...")
    llm.load_from_hf()
    logger.success("Model loading completed!")
    
    # Create logits capture
    capture = LogitsCapture(config_name)
    
    # Patch model for logits capture
    restore_func = patch_model_for_logits_capture(llm, capture)
    
    try:
        # Setup terminators
        terminators = [tokenizer.eos_token_id]
        
        # Run generation (similar to test_generate.py)
        results = llm.generate(
            input_ids=input_ids.view(-1),
            generation_length=args.num_generate,
            teminators=terminators,
            use_stream=False
        )
        
        # Process results
        if isinstance(results, tuple):
            if len(results) >= 4:
                tokens, accept_lengths, decode_time, prefill_time = results[:4]
            elif len(results) >= 3:
                tokens, decode_time, prefill_time = results[:3]
                accept_lengths = []
            else:
                tokens = results[0]
                decode_time = prefill_time = 0.0
                accept_lengths = []
        else:
            tokens = results
            decode_time = prefill_time = 0.0
            accept_lengths = []
        
        # Decode generated text
        if isinstance(tokens, (list, tuple, torch.Tensor)):
            generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            generated_text = "Unable to decode tokens"
        
        print(f"\n{config_name} Results:")
        print(f"Generated text: {generated_text}")
        print(f"Prefill time: {prefill_time:.4f}s")
        print(f"Decode time: {decode_time:.4f}s")
        print(f"Total tokens: {len(tokens) if hasattr(tokens, '__len__') else 'N/A'}")
        print(f"Accept lengths: {accept_lengths}")
        print(f"Captured steps: {len(capture.captured_logits)}")
        
        # Save data
        capture.save_data(f"logits_capture_{config_name}.pkl")
        
        return capture
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return capture
        
    finally:
        restore_func()
        # Free GPU memory after generation
        del llm
        torch.cuda.empty_cache()


def compare_logits_data(capture1: LogitsCapture, capture2: LogitsCapture, comparison_steps: int = 20):
    """Compare captured logits data"""
    
    print(f"\n{'='*80}")
    print(f"DETAILED LOGITS COMPARISON: {capture1.config_name} vs {capture2.config_name}")
    print(f"{'='*80}")
    
    logits1 = capture1.captured_logits
    logits2 = capture2.captured_logits
    tokens1 = capture1.captured_tokens
    tokens2 = capture2.captured_tokens
    
    print(f"{capture1.config_name}: {len(logits1)} steps")
    print(f"{capture2.config_name}: {len(logits2)} steps")
    
    if len(logits1) == 0 or len(logits2) == 0:
        print("⚠️  One or both captures are empty. Cannot compare.")
        return
    
    min_steps = min(len(logits1), len(logits2))
    comparison_results = []
    
    # Use the smaller of min_steps and comparison_steps for comparison
    steps_to_show = min(min_steps, comparison_steps)
    print(f"\nDetailed step-by-step comparison (first {steps_to_show} steps):")
    
    for i in range(steps_to_show):  # Use steps_to_show instead of hardcoded 20
        step_id1, logits_data1, step_type1 = logits1[i]
        step_id2, logits_data2, step_type2 = logits2[i]
        
        token_id1, token1 = tokens1[i]
        token_id2, token2 = tokens2[i]
        
        # Calculate metrics
        mse = np.mean((logits_data1 - logits_data2) ** 2)
        cos_sim = np.dot(logits_data1, logits_data2) / (np.linalg.norm(logits_data1) * np.linalg.norm(logits_data2))
        
        # Top-5 tokens for each config
        top5_indices1 = np.argsort(logits_data1)[-5:][::-1]
        top5_indices2 = np.argsort(logits_data2)[-5:][::-1]
        top5_values1 = logits_data1[top5_indices1]
        top5_values2 = logits_data2[top5_indices2]
        
        comparison_result = {
            'step_index': i,
            'step_id1': step_id1,
            'step_id2': step_id2,
            'step_type1': step_type1,
            'step_type2': step_type2,
            'token1': token1,
            'token2': token2,
            'mse': mse,
            'cosine_similarity': cos_sim,
            'top5_tokens1': top5_indices1.tolist(),
            'top5_values1': top5_values1.tolist(),
            'top5_tokens2': top5_indices2.tolist(),
            'top5_values2': top5_values2.tolist()
        }
        
        comparison_results.append(comparison_result)
        
        print(f"\n" + "="*60)
        print(f"Step {i+1}:")
        print(f"  {capture1.config_name}: Step ID {step_id1}, Type {step_type1}, Token {token1}")
        print(f"  {capture2.config_name}: Step ID {step_id2}, Type {step_type2}, Token {token2}")
        print(f"  MSE: {mse:.8f}")
        print(f"  Cosine Similarity: {cos_sim:.8f}")
        
        print(f"  Top-5 logits {capture1.config_name}:")
        for j, (idx, val) in enumerate(zip(top5_indices1, top5_values1)):
            print(f"    {j+1}. Token {idx}: {val:.6f}")
        
        print(f"  Top-5 logits {capture2.config_name}:")
        for j, (idx, val) in enumerate(zip(top5_indices2, top5_values2)):
            print(f"    {j+1}. Token {idx}: {val:.6f}")
        
        if token1 != token2:
            print(f"  ⚠️  TOKEN MISMATCH: {token1} vs {token2}")
        
        if mse > 1e-6:
            print(f"  ⚠️  LARGE MSE DIFFERENCE: {mse:.8f}")
    
    # Summary statistics
    mse_values = [r['mse'] for r in comparison_results]
    cos_sim_values = [r['cosine_similarity'] for r in comparison_results]
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total steps compared: {min_steps}")
    print(f"Steps shown: {len(comparison_results)}")
    print(f"Average MSE: {np.mean(mse_values):.8f}")
    print(f"Max MSE: {np.max(mse_values):.8f}")
    print(f"Min MSE: {np.min(mse_values):.8f}")
    print(f"Average Cosine Similarity: {np.mean(cos_sim_values):.8f}")
    print(f"Min Cosine Similarity: {np.min(cos_sim_values):.8f}")
    
    token_mismatches = sum(1 for r in comparison_results if r['token1'] != r['token2'])
    print(f"Token sequence mismatches: {token_mismatches}/{len(comparison_results)}")
    
    # Save comparison results
    comparison_summary = {
        'config1': capture1.config_name,
        'config2': capture2.config_name,
        'steps_compared': min_steps,
        'comparison_results': [
            {
                'step_index': r['step_index'],
                'step_id1': r['step_id1'],
                'step_id2': r['step_id2'],
                'step_type1': r['step_type1'],
                'step_type2': r['step_type2'],
                'token1': r['token1'],
                'token2': r['token2'],
                'mse': float(r['mse']),
                'cosine_similarity': float(r['cosine_similarity']) if not np.isnan(r['cosine_similarity']) else None,
                'top5_tokens1': r['top5_tokens1'],
                'top5_values1': [float(v) for v in r['top5_values1']],
                'top5_tokens2': r['top5_tokens2'],
                'top5_values2': [float(v) for v in r['top5_values2']]
            } for r in comparison_results
        ],
        'summary_stats': {
            'avg_mse': float(np.mean(mse_values)),
            'max_mse': float(np.max(mse_values)),
            'min_mse': float(np.min(mse_values)),
            'avg_cosine_similarity': float(np.nanmean(cos_sim_values)) if not np.all(np.isnan(cos_sim_values)) else None,
            'min_cosine_similarity': float(np.nanmin(cos_sim_values)) if not np.all(np.isnan(cos_sim_values)) else None,
            'token_mismatches': token_mismatches
        }
    }
    
    with open('detailed_logits_comparison_results.json', 'w') as f:
        json.dump(comparison_summary, f, indent=2)
    
    print(f"\nDetailed comparison results saved to detailed_logits_comparison_results.json")


# def main():
#     """Main function"""
    
#     # Parse command line arguments for comparison configuration
#     parser = argparse.ArgumentParser(description='Compare logits between different speculative decoding configurations')
#     parser.add_argument('--comparison-steps', '--comparison_steps', type=int, default=20,
#                        help='Number of tokens to compare between configurations (default: 20)')
#     parser.add_argument('--config1-iter', type=int, default=5,
#                        help='spec_num_iter for configuration 1 (default: 5)')
#     parser.add_argument('--config1-tree-size', type=int, default=32,
#                        help='spec_tree_size for configuration 1 (default: 32)')
#     parser.add_argument('--config2-iter', type=int, default=2,
#                        help='spec_num_iter for configuration 2 (default: 2)')
#     parser.add_argument('--config2-tree-size', type=int, default=12,
#                        help='spec_tree_size for configuration 2 (default: 12)')
    
#     # Parse only known args to avoid conflicts with other argument parsing in the script
#     comparison_args, _ = parser.parse_known_args()
    
#     comparison_steps = comparison_args.comparison_steps
#     config1_iter = comparison_args.config1_iter
#     config1_tree_size = comparison_args.config1_tree_size
#     config2_iter = comparison_args.config2_iter
#     config2_tree_size = comparison_args.config2_tree_size
    
#     print(f"🚀 Starting Direct Logits Comparison...")
#     print(f"📊 Comparison settings: {comparison_steps} tokens per configuration")
    
#     # Configuration 1
#     print(f"\n" + "="*60)
#     print(f"CONFIGURATION 1: spec_num_iter={config1_iter}, spec_tree_size={config1_tree_size}")
#     print("="*60)
    
#     capture1 = run_generation_with_config(config1_iter, config1_tree_size, 
#                                          f"config1_iter{config1_iter}_tree{config1_tree_size}", 
#                                          comparison_steps)
    
#     # Clear memory between configurations
#     torch.cuda.empty_cache()
#     print(f"\n📄 Memory cleared after Configuration 1")
    
#     # Configuration 2
#     print(f"\n" + "="*60)
#     print(f"CONFIGURATION 2: spec_num_iter={config2_iter}, spec_tree_size={config2_tree_size}")  
#     print("="*60)
    
#     capture2 = run_generation_with_config(config2_iter, config2_tree_size, 
#                                          f"config2_iter{config2_iter}_tree{config2_tree_size}", 
#                                          comparison_steps)
    
#     # Compare the results
#     if capture1 and capture2:
#         compare_logits_data(capture1, capture2, comparison_steps)
#     else:
#         print("⚠️  One or both configurations failed to run successfully.")
#         if capture1:
#             print(f"✅ Configuration 1 completed successfully with {len(capture1.captured_logits)} steps")
#         else:
#             print("❌ Configuration 1 failed")
#         if capture2:
#             print(f"✅ Configuration 2 completed successfully with {len(capture2.captured_logits)} steps")
#         else:
#             print("❌ Configuration 2 failed")
    
#     print(f"\n✅ Direct logits comparison completed!")

def main():
    """Main function"""
    
    # Parse command line arguments for comparison configuration
    parser = argparse.ArgumentParser(description='Compare logits between different speculative decoding configurations')
    parser.add_argument('--comparison-steps', '--comparison_steps', type=int, default=20,
                       help='Number of tokens to compare between configurations (default: 20)')
    parser.add_argument('--config1-iter', type=int, default=5,
                       help='spec_num_iter for configuration 1 (default: 5)')
    parser.add_argument('--config1-tree-size', type=int, default=32,
                       help='spec_tree_size for configuration 1 (default: 32)')
    parser.add_argument('--use-chat-template', action='store_true', default=False,
                       help='Use chat template for prompt formatting (default: False)')

    
    # Parse only known args to avoid conflicts with other argument parsing in the script
    comparison_args, _ = parser.parse_known_args()
    
    comparison_steps = comparison_args.comparison_steps
    config1_iter = comparison_args.config1_iter
    config1_tree_size = comparison_args.config1_tree_size
    use_chat_template = comparison_args.use_chat_template
    
    print(f"🚀 Starting Direct Logits Comparison...")
    print(f"📊 Comparison settings: {comparison_steps} tokens per configuration")
    print(f"📝 Chat template: {'Enabled' if use_chat_template else 'Disabled'}")
    
    # Configuration 1
    print(f"\n" + "="*60)
    print(f"CONFIGURATION 1: spec_num_iter={config1_iter}, spec_tree_size={config1_tree_size}")
    print("="*60)
    
    capture1 = run_generation_with_config(config1_iter, config1_tree_size, 
                                         f"config1_iter{config1_iter}_tree{config1_tree_size}", 
                                         comparison_steps, use_chat_template)
    
    # Clear memory between configurations
    torch.cuda.empty_cache()
    print(f"\n📄 Memory cleared after Configuration 1")
    

if __name__ == "__main__":
    main() 