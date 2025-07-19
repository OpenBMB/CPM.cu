#!/usr/bin/env python3
"""
CPM.cu Unified Argument Processing

Unified argument processing module for generic model support
"""

import argparse

def str2bool(v):
    """Convert string to boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_model_config_args(parser: argparse.ArgumentParser):
    """Add common model configuration arguments"""
    
    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model-path', '--model_path', '--model', type=str, required=True,
                           help='Path to the main model (local path or HuggingFace repo)')
    model_group.add_argument('--draft-model-path', '--draft_model_path', type=str, default=None,
                           help='Path to draft model for speculative decoding')
    model_group.add_argument('--frspec-path', '--frspec_path', type=str, default=None,
                           help='Path to frequency speculative vocabulary file (.pt file)')
    model_group.add_argument('--model-type', '--model_type', type=str, default='auto',
                           choices=['auto', 'llama', 'minicpm', 'minicpm4'],
                           help='Model type (default: auto-detect)')
    model_group.add_argument('--dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                            help='Model dtype (default: float16)')
    model_group.add_argument('--minicpm4-yarn', '--minicpm4_yarn', default=False,
                            type=str2bool, nargs='?', const=True,
                            help='Enable MiniCPM4 YARN for long context support (default: False)')

    # System Configuration
    system_group = parser.add_argument_group('System Configuration')
    system_group.add_argument('--cuda-graph', '--cuda_graph', default=True,
                            type=str2bool, nargs='?', const=True,
                            help='Use CUDA graph optimization (default: True)')
    system_group.add_argument('--memory-limit', '--memory_limit', type=float, default=0.9,
                            help='Memory limit (default: 0.9)')
    system_group.add_argument('--chunk-length', '--chunk_length', type=int, default=2048,
                            help='Chunked prefill size (default: 2048)')
    system_group.add_argument('--plain-output', '--plain_output', default=False,
                            type=str2bool, nargs='?', const=True,
                            help='Use plain text output (no colors/formatting) for maximum compatibility (default: False)')

    # Speculative Decoding
    spec_group = parser.add_argument_group('Speculative Decoding')
    spec_group.add_argument('--spec-window-size', '--spec_window_size', type=int, default=1024,
                           help='Speculative decoding slidingwindow size (default: 1024)')
    spec_group.add_argument('--spec-num-iter', '--spec_num_iter', type=int, default=2,
                           help='Speculative decoding number of iterations (default: 2)')
    spec_group.add_argument('--spec-topk-per-iter', '--spec_topk_per_iter', type=int, default=10,
                           help='Speculative decoding top-k per iteration (default: 10)')
    spec_group.add_argument('--spec-tree-size', '--spec_tree_size', type=int, default=12,
                           help='Speculative decoding tree size (default: 12)')
    spec_group.add_argument('--frspec-vocab-size', '--frspec_vocab_size', type=int, default=32768, # TODO: fix frspec vocab size display without frspec_path
                           help='Frequent speculation vocab size (default: 32768)')

    # Sparse Attention
    sparse_group = parser.add_argument_group('Sparse Attention')
    sparse_group.add_argument('--sink-window-size', '--sink_window_size', type=int, default=1,
                            help='Sink window size (default: 1)')
    sparse_group.add_argument('--block-window-size', '--block_window_size', type=int, default=8,
                            help='Block window size (default: 8)')
    sparse_group.add_argument('--sparse-topk-k', '--sparse_topk_k', type=int, default=64,
                            help='Sparse attention top-k (default: 64)')
    sparse_group.add_argument('--sparse-switch', '--sparse_switch', type=int, default=0,
                            help='Sparse switch threshold (default: 0)')
    sparse_group.add_argument('--use-compress-lse', '--use_compress_lse', default=True,
                            type=str2bool, nargs='?', const=True,
                            help='Use LSE compression (default: True)')


def create_server_parser() -> argparse.ArgumentParser:
    """Create server argument parser"""
    parser = argparse.ArgumentParser(description='CPM.cu Server')
    
    # Server Configuration
    server_group = parser.add_argument_group('Server Configuration')
    server_group.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    server_group.add_argument('--port', type=int, default=8000, help='Server port')
    
    add_model_config_args(parser)
    return parser


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(description='CPM.cu CLI')
    
    # Prompt Configuration
    prompt_group = parser.add_argument_group('Prompt Configuration')
    prompt_group.add_argument('--prompt-file', '--prompt_file', type=str, default=None,
                       help='Path to prompt file')
    prompt_group.add_argument('--prompt-text', '--prompt_text', type=str, default=None,
                       help='Direct prompt text')
    prompt_group.add_argument('--use-chat-template', '--use_chat_template', default=True,
                       type=str2bool, nargs='?', const=True,
                       help='Use chat template for prompt formatting (default: True)')
    
    # Generation Configuration
    generation_group = parser.add_argument_group('Generation Configuration')
    generation_group.add_argument('--use-stream', '--use_stream', default=True,
                       type=str2bool, nargs='?', const=True,
                       help='Use stream generation (default: True)')
    generation_group.add_argument('--num-generate', '--num_generate', type=int, default=1024,
                       help='Number of tokens to generate (default: 1024)')
    generation_group.add_argument('--temperature', '--temp', type=float, default=0.0,
                             help='Temperature (default: 0.0)')
    generation_group.add_argument('--random-seed', '--random_seed', type=int, default=None,
                            help='Random seed')
    generation_group.add_argument('--ignore-eos', '--ignore_eos', default=False,
                       type=str2bool, nargs='?', const=True,
                       help='Ignore EOS tokens during generation (default: False)')
    
    # 添加数据集相关参数到Generation Configuration组
    generation_group.add_argument("--dataset", type=str, 
                            choices=['mtbench', 'specbench', 'gsm8k', 'qa', 'wmt14', 'rag', 'summarization'], 
                            help="Dataset type to evaluate on")
    generation_group.add_argument("--dataset-path", "--dataset_path", type=str, 
                            help="Path to dataset file (for mtbench path, default uses benchmark/datasets/mtbench.jsonl)")
    generation_group.add_argument("--output-dir", "--output_dir", type=str, default="benchmark/results/logs",
                            help="Directory to save benchmark datasets evaluation results, only enable when dataset is used")
    generation_group.add_argument("--batch-size", "--batch_size", type=int, default=1,
                            help="Batch size for dataset evaluation")

    add_model_config_args(parser)
    return parser


def parse_server_args() -> argparse.Namespace:
    """Parse server arguments"""
    parser = create_server_parser()
    args = parser.parse_args()
    return args


def parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments"""
    parser = create_cli_parser()
    args = parser.parse_args()
    return args
