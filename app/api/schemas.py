"""OpenAI-compatible request and admin control-plane schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: Any = None
    name: str | None = None

    @field_validator("role")
    @classmethod
    def _role(cls, value: str) -> str:
        if value not in ("system", "developer", "user", "assistant", "tool", "function"):
            raise ValueError(f"unsupported message role '{value}'")
        return value


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(default="auto", max_length=200)
    messages: list[Message] = Field(min_length=1, max_length=2000)
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=131072)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    n: int = Field(default=1, ge=1, le=8)
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    user: str | None = Field(default=None, max_length=256)

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_tokens or self.max_completion_tokens

    @model_validator(mode="after")
    def _stream_options(self) -> "ChatCompletionRequest":
        if self.stream_options and not self.stream:
            # OpenAI ignores this, so do we, but keep the payload clean.
            self.stream_options = None
        return self


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(default="auto", max_length=200)
    prompt: Any = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = Field(default=16, ge=1, le=131072)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    n: int = Field(default=1, ge=1, le=8)
    stop: str | list[str] | None = None

    @field_validator("prompt")
    @classmethod
    def _prompt(cls, value: Any) -> Any:
        if value is None:
            raise ValueError("prompt is required")
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(
            isinstance(item, (str, int)) or (
                isinstance(item, list) and all(isinstance(t, int) for t in item)
            )
            for item in value
        ):
            return value
        raise ValueError("prompt must be a string, token list, or list of prompts")
    @model_validator(mode="after")
    def _stream_options(self) -> "CompletionRequest":
        if self.stream_options and not self.stream:
            self.stream_options = None
        return self


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(default="auto", max_length=200)
    input: Any
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = Field(default=None, ge=1)

    @field_validator("input")
    @classmethod
    def _input(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value and all(
            isinstance(v, str) or (isinstance(v, list) and all(isinstance(t, int) for t in v))
            for v in value
        ):
            return value
        raise ValueError("input must be a string or a non-empty array of strings/tokens")
    @model_validator(mode="after")
    def _validate(self) -> "EmbeddingRequest":
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("input must not be empty")
        return self


class ModelCreate(BaseModel):
    path: str = Field(min_length=1, max_length=4096)
    name: str | None = Field(default=None, max_length=120)
    aliases: list[str] | None = None
    capabilities: list[str] | None = None
    context_length: int | None = Field(default=None, ge=256, le=1_048_576)
    priority: int = Field(default=50, ge=0, le=100)
    residency: Literal["on_demand", "keep_loaded"] = "on_demand"
    max_concurrency: int | None = Field(default=None, ge=1, le=16)
    n_gpu_layers: int | None = Field(default=None, ge=-1, le=999)
    threads: int | None = Field(default=None, ge=1, le=256)
    description: str = Field(default="", max_length=500)
    enabled: bool = True
    extra_args: list[str] | None = None


class ModelUpdate(BaseModel):
    name: str | None = Field(default=None, max_length=120)
    aliases: list[str] | None = None
    capabilities: list[str] | None = None
    context_length: int | None = Field(default=None, ge=256, le=1_048_576)
    priority: int | None = Field(default=None, ge=0, le=100)
    residency: Literal["on_demand", "keep_loaded"] | None = None
    max_concurrency: int | None = Field(default=None, ge=1, le=16)
    n_gpu_layers: int | None = Field(default=None, ge=-1, le=999)
    threads: int | None = Field(default=None, ge=1, le=256)
    description: str | None = Field(default=None, max_length=500)
    enabled: bool | None = None
    extra_args: list[str] | None = None


class KeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    role: Literal["admin", "inference"] = "inference"
    permissions: dict[str, Any] = Field(default_factory=dict)
    rate_limit_rpm: int | None = Field(default=None, ge=1, le=100_000)
    rate_limit_tpm: int | None = Field(default=None, ge=100, le=100_000_000)
    expires_at: str | None = None


class KeyUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    permissions: dict[str, Any] | None = None
    rate_limit_rpm: int | None = Field(default=None, ge=1, le=100_000)
    rate_limit_tpm: int | None = Field(default=None, ge=100, le=100_000_000)
    expires_at: str | None = None


class RouterConfigUpdate(BaseModel):
    strategy: Literal["balanced", "quality", "latency", "local"] | None = None
    llm_enabled: bool | None = None
    llm_model: str | None = None
    llm_timeout_ms: int | None = Field(default=None, ge=200, le=10_000)
    ambiguity_threshold: float | None = Field(default=None, ge=0, le=1)
    prioritize_loaded: bool | None = None
    fallback_enabled: bool | None = None
    allow_general_fallback: bool | None = None
    max_alternatives: int | None = Field(default=None, ge=1, le=10)


class TunnelStart(BaseModel):
    mode: Literal["quick", "token"] = "quick"
    token: str | None = Field(default=None, max_length=4096)
    public_url: str | None = Field(default=None, max_length=2048)
    tunnel_name: str | None = Field(default=None, max_length=128)


class FeedbackRequest(BaseModel):
    request_id: str
    helpful: bool
    comment: str | None = Field(default=None, max_length=1000)
