#!/usr/bin/env python3
import argparse
import json
import torch
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from asyncio import Lock
from fastapi.middleware.cors import CORSMiddleware

from .common.openai_api import (
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    ErrorResponse,
    HealthResponse
)
from .common.utils import (
    setup_model_paths,
    create_model,
    setup_frspec_vocab,
    apply_minicpm4_yarn_config
)
from .common.args import parse_server_args
from .common.display import display
from .common.logging import logger

# Global model instance
model_instance = None
model_config = {}


def initialize_model(config):
    """Initialize model with given configuration"""
    logger.info("Loading model with configuration:")
    
    # Setup model paths
    model_path, draft_model_path, frspec_path = setup_model_paths(config)
    
    logger.info(f"Base model path: {model_path}")
    if draft_model_path:
        logger.info(f"Draft model path: {draft_model_path}")
    
    # Create model instance
    model_instance = create_model(model_path, draft_model_path, config)
    
    # Initialize model storage
    model_instance.init_storage()
    
    # Display GPU memory information and max supported length adjacently
    memory_limit = config.get('memory_limit', 0.8)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_memory / (1024**3)
    limit_gb = total_gb * memory_limit
    logger.info(f"GPU Memory: {total_gb:.1f}GB total, {limit_gb:.1f}GB allocated ({memory_limit:.0%})")
    logger.info(f"Maximum context length under current memory limit: {model_instance.max_total_length} tokens")
    
    # Apply MiniCPM4 YARN configuration if enabled
    if config.get('minicpm4_yarn', False):
        try:
            apply_minicpm4_yarn_config(model_instance)
        except Exception as e:
            logger.warning(f"MiniCPM4 YARN configuration failed: {e}")
    
    # Load frequency speculative vocabulary if enabled
    if draft_model_path and (frspec_path is not None) and (config.get('frspec_vocab_size', 0) > 0):
        logger.info(f"Loading frequency vocabulary from {frspec_path}")
        frspec_result = setup_frspec_vocab(model_instance, frspec_path, config.get('frspec_vocab_size', 0))
        if frspec_result is True:
            logger.success("Frequency vocabulary loaded successfully")
        else:
            logger.warning("Could not load frequency vocabulary")
    
    # Load model weights
    logger.info("Loading model weights...")
    model_instance.load_from_hf()
    logger.success("Model loaded successfully!")
    
    return model_instance


def normalize_content_to_text(content) -> str:
    """Normalize OpenAI message content to plain text.
    Supports both string and array-of-parts with type fields, e.g. [{"type": "text", "text": "..."}].
    Non-text parts are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            # Handle dict-like parts
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text":
                    text_value = part.get("text", "")
                    parts.append(text_value if isinstance(text_value, str) else str(text_value))
                else:
                    # Ignore non-text types for now
                    continue
            else:
                # Handle model-like objects with attributes (e.g., pydantic model)
                part_type = getattr(part, "type", None)
                if part_type == "text":
                    text_value = getattr(part, "text", "")
                    parts.append(text_value if isinstance(text_value, str) else str(text_value))
        return "".join(parts)
    # Fallback to string conversion for unexpected types
    return str(content)


def format_messages_to_prompt(messages: list, tokenizer) -> str:
    """Convert OpenAI messages format to a single prompt string using chat template"""
    
    # Convert OpenAI messages format to the format expected by chat template
    chat_messages = []
    
    for message in messages:
        chat_messages.append({
            "role": message.role,
            "content": normalize_content_to_text(message.content)
        })
    
    # Use tokenizer's chat template to format the prompt
    try:
        prompt = tokenizer.apply_chat_template(
            chat_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt
    except Exception as e:
        # Fallback to simple formatting if chat template fails
        logger.warning(f"Chat template failed ({e}), falling back to simple formatting")
        return simple_format_fallback(messages)


def simple_format_fallback(messages: list) -> str:
    """Simple formatting fallback solution"""
    prompt_parts = []
    
    for message in messages:
        role = message.role
        content = normalize_content_to_text(message.content)
        
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    # Add final assistant prompt
    prompt_parts.append("Assistant:")
    
    return "\n".join(prompt_parts)


def get_stop_tokens(stop_param: Optional[Union[str, list]] = None) -> list:
    """Convert stop parameter to token IDs"""
    if not stop_param:
        return []
    
    if isinstance(stop_param, str):
        stop_list = [stop_param]
    else:
        stop_list = stop_param
    
    # Convert to token IDs using tokenizer
    stop_token_ids = []
    for stop_str in stop_list:
        try:
            tokens = model_instance.tokenizer.encode(stop_str, add_special_tokens=False)
            stop_token_ids.extend(tokens)
        except:
            pass
    
    return stop_token_ids


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load model on startup"""
    global model_instance, model_config
    
    try:
        config = model_config['config']
        model_instance = initialize_model(config)
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    model_instance = None


# Create FastAPI app
app = FastAPI(
    title="CPM.cu OpenAI API Server",
    description="OpenAI API compatible server for CPM.cu models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        model_loaded=model_instance is not None,
        memory_usage=f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else None
    )


# Global async lock to serialize generation to avoid CUDA context interference
_generation_lock: Lock = Lock()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI Chat Completions API endpoint"""
    global model_instance
    
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get tokenizer from model instance first
        tokenizer = getattr(model_instance, 'tokenizer', None)
        if hasattr(model_instance, 'base_model') and hasattr(model_instance.base_model, 'tokenizer'):
            tokenizer = model_instance.base_model.tokenizer
        
        if tokenizer is None:
            raise HTTPException(status_code=500, detail="Tokenizer not available")
        
        # Format messages to prompt using chat template
        prompt = format_messages_to_prompt(request.messages, tokenizer)
        
        # Tokenize input
        input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = input_ids.to(torch.int32).cuda()
        
        # Get stop token IDs and add EOS token if ignore_eos is disabled
        stop_tokens = get_stop_tokens(request.stop)
        config = model_config['config']
        if not config.get('ignore_eos', False):
            if tokenizer.eos_token_id not in stop_tokens:
                stop_tokens.append(tokenizer.eos_token_id)
        
        if request.stream:
            async def _locked_stream():
                async with _generation_lock:
                    async for chunk in stream_chat_completion(
                        input_ids,
                        request.max_tokens or 100,
                        stop_tokens,
                        request.model,
                        request,
                        tokenizer,
                    ):
                        yield chunk

            return StreamingResponse(
                _locked_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                }
            )
        else:
            async with _generation_lock:
                return await generate_chat_completion(
                    input_ids,
                    request.max_tokens or 100, 
                    stop_tokens,
                    request.model,
                    request,
                    tokenizer
                )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


async def generate_chat_completion(
    input_ids: torch.Tensor, 
    max_tokens: int, 
    stop_tokens: list,
    model_name: str,
    request: ChatCompletionRequest,
    tokenizer
) -> ChatCompletionResponse:
    """Generate non-streaming chat completion"""
    
    # Generate tokens using the same interface as test_generate.py
    gen_result = model_instance.generate(
        input_ids.view(-1), 
        generation_length=max_tokens,
        teminators=stop_tokens,
        use_stream=False,
        temperature=request.temperature
    )
    
    # Handle different return formats based on model type
    config = model_config['config']
    has_speculative = config.get('draft_model_path') is not None
    if has_speculative:
        # Speculative models return: (tokens, accept_lengths, decode_time, prefill_time)
        tokens, accept_lengths, decode_time, prefill_time = gen_result
    else:
        # Base models return: (tokens, decode_time, prefill_time)
        tokens, decode_time, prefill_time = gen_result
    
    # Decode response
    generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
    
    # Extract only the assistant's response (after the last "Assistant:")
    if "Assistant:" in generated_text:
        response_text = generated_text.split("Assistant:")[-1].strip()
    else:
        response_text = generated_text.strip()
    
    # Determine finish reason
    finish_reason = "stop"
    if len(tokens) >= max_tokens:
        finish_reason = "length"
    elif any(token in tokens for token in stop_tokens):
        finish_reason = "stop"
    
    return ChatCompletionResponse(
        model=model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason=finish_reason
            )
        ],
        usage={
            "prompt_tokens": input_ids.numel(),
            "completion_tokens": len(tokens),
            "total_tokens": input_ids.numel() + len(tokens)
        }
    )


async def stream_chat_completion(
    input_ids: torch.Tensor,
    max_tokens: int,
    stop_tokens: list, 
    model_name: str,
    request: ChatCompletionRequest,
    tokenizer
) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion"""
    
    import time
    import uuid
    import asyncio
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    # Start generation stream
    stream_gen = model_instance.generate(
        input_ids.view(-1),
        generation_length=max_tokens,
        teminators=stop_tokens,
        use_stream=True,
        temperature=request.temperature
    )
    
    accumulated_text = ""
    
    try:
        for chunk in stream_gen:
            token = chunk['token']
            text = chunk['text']
            is_finished = chunk['is_finished']
            
            # Append accumulated text (optional) and send raw token text
            accumulated_text += text

            # Create streaming response using the model-provided token text
            if not is_finished:
                response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": text},
                            finish_reason=None
                        )
                    ]
                )
            else:
                # Final chunk
                finish_reason = "stop"
                if token in stop_tokens:
                    finish_reason = "stop"
                elif len(accumulated_text) >= max_tokens:
                    finish_reason = "length"
                
                response = ChatCompletionStreamResponse(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[
                        ChatCompletionStreamChoice(
                            index=0,
                            delta={},
                            finish_reason=finish_reason
                        )
                    ]
                )
            
            # Yield with immediate flush to ensure real-time streaming
            yield f"data: {response.model_dump_json()}\n\n"
            
            # Add a small async yield to allow the event loop to process
            await asyncio.sleep(0)
            
            if is_finished:
                break
                
    except Exception as e:
        error_response = {
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "generation_failed"
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
        await asyncio.sleep(0)
    
    yield "data: [DONE]\n\n"
    await asyncio.sleep(0)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error={
                "message": str(exc),
                "type": "internal_error", 
                "code": "server_error"
            }
        ).model_dump()
    )


def server(args: argparse.Namespace):
    """Launch server with given configuration"""
    
    # Display configuration summary
    display.render_config(args, "Server Configuration")
    
    # Set global model config - convert args to dict for compatibility
    global model_config
    model_config = {"config": vars(args)}
    
    logger.info(f"Starting CPM.cu OpenAI API Server on {getattr(args, 'host', '0.0.0.0')}:{getattr(args, 'port', 8000)}")
    
    # Configure external loggers to use the same handler with colors
    logger.configure_external_loggers(['uvicorn', 'uvicorn.access', 'uvicorn.error'])
    
    uvicorn.run(
        app,
        host=getattr(args, 'host', '0.0.0.0'),
        port=getattr(args, 'port', 8000),
        log_config=None,  # Disable Uvicorn's default logging config
        access_log=True
    )


def main():
    """Server entry point using unified argument processing"""
    
    # Use unified argument parsing
    args = parse_server_args()

    # Configure display and logger mode before first use
    use_plain_mode = getattr(args, 'plain_output', False)
    from .common.display import Display
    from .common.logging import Logger
    Display.configure(use_plain_mode=use_plain_mode)
    Logger.configure(use_plain_mode=use_plain_mode)
    
    # Launch server
    server(args)


if __name__ == "__main__":
    main() 