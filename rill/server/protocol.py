"""OpenAI-compatible request/response schemas (the subset rill implements)."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingFields(BaseModel):
    model: str | None = None
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)  # extension (vLLM-style): -1 = disabled
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    ignore_eos: bool = False  # extension used by benchmarks
    n: int = Field(default=1, ge=1, le=1)  # only n=1 is supported

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class CompletionRequest(_SamplingFields):
    prompt: str | list[int]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage] = Field(min_length=1)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def now() -> int:
    return int(time.time())
