#!/usr/bin/env python3
"""
CPM.cu Utility Functions

Common utility functions for generic model support.
"""

import os
import json
import torch
from huggingface_hub import snapshot_download
from .logging import logger


def check_or_download_model(path):
    """Check if model exists locally, otherwise download from HuggingFace"""
    if os.path.exists(path):
        return path
    else:
        cache_dir = snapshot_download(path)
        return cache_dir


def detect_quantization_from_path(model_path):
    """Auto-detect quantization from model path"""
    if not model_path:
        return False
    path_lower = model_path.lower()
    quant_keywords = ['marlin', 'gptq', 'quant', 'awq', 'int4', 'int8', 'w4a16', 'qat']
    return any(keyword in path_lower for keyword in quant_keywords)


def detect_model_type(model_path):
    """Auto-detect model type based on config.json"""
    try:
        config_path = os.path.join(model_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            arch = config.get('architectures', [''])[0].lower()
            model_type = config.get('model_type', '').lower()
            
            if 'qwen2' in arch or 'qwen2' in model_type:
                return 'qwen2'
            elif 'qwen3' in arch or 'qwen3' in model_type:
                return 'qwen3'
            elif 'minicpm' in arch or 'minicpm' in model_type:
                # Use architecture parameters to distinguish MiniCPM4 from MiniCPM
                num_hidden_layers = config.get('num_hidden_layers', 0)
                num_key_value_heads = config.get('num_key_value_heads', 0)
                
                # Check if the ratio is 16 (MiniCPM4 characteristic)
                if num_key_value_heads > 0 and num_hidden_layers / num_key_value_heads == 16:
                    return 'minicpm4'
                else:
                    return 'minicpm'
            elif 'llama' in arch or 'llama' in model_type:
                return 'llama'
            else:
                return 'unknown'
    except Exception as e:
        logger.warning(f"Could not detect model type from {model_path}: {e}")
        return 'unknown'


def setup_model_paths(config):
    """Setup and validate model paths with automatic feature detection"""
    
    with logger.stage_context("Setting up model paths"):
        # Setup main model
        model_path = check_or_download_model(config['model_path'])
        
        # Auto-detect model type and quantization
        if config.get('model_type', 'auto') == 'auto':
            config['model_type'] = detect_model_type(model_path)
            logger.info(f"Auto-detected model type: [cyan]{config['model_type']}[/cyan]")
        
        # Setup draft model for speculative decoding
        draft_model_path = None
        if config.get('draft_model_path'):
            draft_model_path = check_or_download_model(config['draft_model_path'])
            logger.info("Draft model specified, enabling [green]speculative decoding[/green]")
        
        # Setup FRSpec path
        frspec_path = None
        if config.get('frspec_path'):
            frspec_path = check_or_download_model(config['frspec_path'])
            vocab_size = config.get('frspec_vocab_size', 0)
            
            if os.path.exists(frspec_path) and os.path.isdir(frspec_path):
                freq_file = os.path.join(frspec_path, f"freq_{vocab_size}.pt")
                if os.path.exists(freq_file):
                    frspec_path = freq_file
                    logger.info(f"Found FRSpec file: [cyan]{freq_file}[/cyan]")
                else:
                    frspec_path, config['frspec_vocab_size'] = None, 0
                    logger.warning(f"FRSpec file freq_{vocab_size}.pt not found in directory: {frspec_path}")
        else:
            config['frspec_vocab_size'] = 0
    
    return model_path, draft_model_path, frspec_path


def create_model(model_path, draft_model_path, config):
    """Create model instance based on configuration"""
    from ..llm import LLM
    from ..llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
    from ..speculative import LLM_with_eagle
    from ..speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
    
    with logger.stage_context("Creating model instance"):
        # Auto-detect model features
        base_model_quantized = detect_quantization_from_path(model_path)
        has_draft_model = draft_model_path is not None
        draft_model_quantized = detect_quantization_from_path(draft_model_path) if has_draft_model else False
        
        # Build common arguments
        common_kwargs = {
            'dtype': torch.float16 if config['dtype'] == 'float16' else torch.bfloat16,
            'chunk_length': config['chunk_length'],
            'cuda_graph': config['cuda_graph'],
            'apply_sparse': config.get('model_type') == 'minicpm4',
            'use_qk_norm': config.get('model_type') == 'qwen3',
            'use_attn_bias': config.get('model_type') == 'qwen2',
            'sink_window_size': config['sink_window_size'],
            'block_window_size': config['block_window_size'],
            'sparse_topk_k': config['sparse_topk_k'],
            'sparse_switch': config['sparse_switch'],
            'use_compress_lse': config['use_compress_lse'],
            'memory_limit': config['memory_limit'],
            'temperature': config.get('temperature', 0.0),
            'random_seed': config.get('random_seed', None),
        }
        spec_kwargs = {
            'num_iter': config.get('spec_num_iter', 2),
            'topk_per_iter': config.get('spec_topk_per_iter', 10),
            'tree_size': config.get('spec_tree_size', 12),
            'eagle_window_size': config.get('spec_window_size', 1024),
            'frspec_vocab_size': config.get('frspec_vocab_size', 0),
            'apply_eagle_quant': draft_model_quantized,
            'use_rope': config.get('model_type') in ['minicpm', 'minicpm4'],
            'use_input_norm': config.get('model_type') in ['minicpm', 'minicpm4'],
            'use_attn_norm': config.get('model_type') in ['minicpm', 'minicpm4'],
            'use_eagle3': config.get('use_eagle3', False)
        }
        
        # Create model based on configuration
        if base_model_quantized:
            if has_draft_model:
                logger.info("Creating [yellow]quantized model[/yellow] with [green]Eagle speculative decoding[/green]")
                return W4A16GPTQMarlinLLM_with_eagle(draft_model_path, model_path, **common_kwargs, **spec_kwargs)
            else:
                logger.info("Creating [yellow]quantized model[/yellow]")
                return W4A16GPTQMarlinLLM(model_path, **common_kwargs)
        else:
            if has_draft_model:
                logger.info("Creating model with [green]Eagle speculative decoding[/green]")
                return LLM_with_eagle(draft_model_path, model_path, **common_kwargs, **spec_kwargs)
            else:
                logger.info("Creating [cyan]standard model[/cyan]")
                return LLM(model_path, **common_kwargs)


def setup_frspec_vocab(llm, frspec_path, frspec_vocab_size):
    """Setup frequency speculative vocabulary for speculative models"""
    if not frspec_path:
        return "not_specified"
    
    if os.path.exists(frspec_path):
        logger.info(f"Loading frequency vocabulary from: {frspec_path}")
        with open(frspec_path, 'rb') as f:
            token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
        llm._load("token_id_remap", token_id_remap, cls="eagle")
        return True
    else:
        logger.error(f"FRSpec file not found: {frspec_path}")
        return "not_found"


def apply_minicpm4_yarn_config(llm):
    """Apply MiniCPM4 YARN configuration to model"""
    yarn_factors = [
        0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193,
        1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284,
        1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191,
        1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756,
        2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632,
        4.048719380066383, 4.752651957515948, 5.590913044973868, 6.584005926629993,
        7.7532214876576155, 9.119754865903639, 10.704443927019176, 12.524994176518703,
        14.59739595363613, 16.93214476166354, 19.53823297353041, 22.417131025031697,
        25.568260840911098, 28.991144156566317, 32.68408069090375, 36.65174474170465,
        40.90396065611201, 45.4664008671033, 50.37147343433591, 55.6804490772103,
        61.470816952306556, 67.8622707390618, 75.00516023410414, 83.11898235973767,
        92.50044360202462, 103.57086856690864, 116.9492274587385, 118.16074567836519,
        119.18497548708795, 120.04810876261652, 120.77352815196981, 121.38182790207875,
        121.89094985353891, 122.31638758099915, 122.6714244963338, 122.9673822552567,
        123.21386397019609, 123.41898278254268, 123.58957065488238, 123.73136519024158,
        123.84917421274221, 123.94701903496814, 124.02825801299717, 124.09569231686116
    ]
    
    if not hasattr(llm.config, 'rope_scaling') or llm.config.rope_scaling is None:
        llm.config.rope_scaling = {}
    
    llm.config.rope_scaling['rope_type'] = 'longrope'
    llm.config.rope_scaling['long_factor'] = yarn_factors
    llm.config.rope_scaling['short_factor'] = yarn_factors
    logger.info("Applied MiniCPM4 YARN rope_scaling parameters") 