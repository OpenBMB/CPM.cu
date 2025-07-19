#!/usr/bin/env python3
"""
CPM.cu Benchmark Module

Dataset loading and evaluation functionality for CPM.cu benchmark evaluation.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from .logging import logger


def load_questions(filename: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file"""
    questions = []
    with open(filename, "r", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                obj = json.loads(line)
                questions.append(obj)
    return questions


def load_dataset(dataset_type: str, dataset_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Load dataset based on type and path"""
    
    # Determine dataset file path
    if dataset_path:
        dataset_file = dataset_path
    else:
        # Default paths in benchmark/datasets/
        default_paths = {
            "mtbench": "benchmark/datasets/mtbench.jsonl",
            "specbench": "benchmark/datasets/specbench.jsonl", 
            "gsm8k": "benchmark/datasets/gsm8k.jsonl",
            "qa": "benchmark/datasets/qa.jsonl",
            "wmt14": "benchmark/datasets/wmt14.jsonl",
            "rag": "benchmark/datasets/rag.jsonl",
            "summarization": "benchmark/datasets/summarization.jsonl",
        }
        
        if dataset_type not in default_paths:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Supported types: {list(default_paths.keys())}")
        
        dataset_file = default_paths[dataset_type]
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    
    # Load raw questions
    raw_questions = load_questions(dataset_file)
    
    # Parse questions based on dataset type
    questions = []
    
    if dataset_type in ["mtbench", "specbench"]:
        # MTBench/SpecBench format: use both turns for two-round conversation
        for data in raw_questions:
            turns = data.get('turns', [])
            if len(turns) >= 1:
                questions.append({
                    'id': data.get('question_id', len(questions)),
                    'question': turns[0],  # First turn
                    'follow_up': turns[1] if len(turns) > 1 else None,  # Second turn (optional)
                    'category': data.get('category', 'general'),
                    'turns': turns  # Keep original turns for reference
                })
    
    elif dataset_type == "gsm8k":
        # GSM8K format: single turn with reference answer
        for data in raw_questions:
            turns = data.get('turns', [])
            if len(turns) >= 1:
                questions.append({
                    'id': data.get('question_id', len(questions)),
                    'question': turns[0],
                    'category': data.get('category', 'math_reasoning'),
                    'reference': data.get('reference', []),
                    'turns': turns
                })
    
    else:
        # Generic format: try to extract question from turns or other fields
        for data in raw_questions:
            turns = data.get('turns', [])
            question_text = ""
            
            if turns and len(turns) > 0:
                question_text = turns[0]
            else:
                # Fallback to other possible field names
                question_text = (data.get('question') or 
                               data.get('prompt') or 
                               data.get('text') or 
                               str(data))
            
            if question_text:
                questions.append({
                    'id': data.get('question_id') or data.get('id', len(questions)),
                    'question': question_text,
                    'category': data.get('category', 'general'),
                    'turns': turns
                })
    
    question_count = len(questions)
    logger.info(f"Loaded {question_count} questions from {dataset_file} ({dataset_type})")
    return questions, question_count


def save_results(results: List[Dict[str, Any]], output_dir: str, 
                dataset_type: str, model_name: str) -> str:
    """Save evaluation results to JSON file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = model_name.replace('/', '_').replace('\\', '_')
    filename = f"{dataset_type}_{model_safe}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Calculate summary statistics
    successful = len([r for r in results if not r.get('error', False)])
    total_time = sum(r.get('timing', {}).get('total_time', 0) for r in results if not r.get('error', False))
    total_tokens = sum(r.get('tokens', {}).get('output_length', 0) for r in results if not r.get('error', False))
    
    output_data = {
        'dataset_type': dataset_type,
        'model_name': model_name,
        'timestamp': timestamp,
        'total_questions': len(results),
        'successful_questions': successful,
        'success_rate': successful / len(results) if results else 0,
        'summary_stats': {
            'total_time': round(total_time, 2),
            'avg_time_per_question': round(total_time / successful, 2) if successful > 0 else 0,
            'total_output_tokens': total_tokens,
            'avg_tokens_per_question': round(total_tokens / successful, 2) if successful > 0 else 0,
            'throughput_tokens_per_sec': round(total_tokens / total_time, 2) if total_time > 0 else 0
        },
        'results': results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.success(f"Results saved to: {filepath}")
    logger.info(f"Summary: {successful}/{len(results)} questions successful, "
                f"{output_data['summary_stats']['throughput_tokens_per_sec']:.2f} tokens/s")
    
    return filepath


def get_available_datasets() -> List[str]:
    """Get list of available dataset types"""
    return ["mtbench", "specbench", "gsm8k", "qa", "wmt14", "rag", "summarization"]


def validate_dataset_exists(dataset_type: str, dataset_path: Optional[str] = None) -> bool:
    """Check if dataset file exists"""
    try:
        load_dataset(dataset_type, dataset_path)
        return True
    except (FileNotFoundError, ValueError):
        return False 