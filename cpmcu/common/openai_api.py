from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
import time
import uuid

class ContentPartText(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[ContentPartText, Dict[str, Any]]]]

class ChatCompletionRequest(BaseModel):
    model: str = "model"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=65536)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, str]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    memory_usage: Optional[str] = None 