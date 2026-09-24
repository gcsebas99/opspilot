from typing import Any, Literal, Protocol

from pydantic import BaseModel


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]


ContentBlock = TextBlock | ToolUseBlock


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ModelResponse(BaseModel):
    """Normalized shape every backend (real, scripted, replay) returns."""

    content: list[ContentBlock]
    stop_reason: str
    usage: Usage
    latency_ms: float


class ModelClient(Protocol):
    """Every model backend implements this.

    [HARNESS:ORCH] One normalized response shape across three backends.
    WHY: the loop (1.6) only ever talks to this Protocol -- it doesn't know
    or care whether it's driving a live Anthropic call, a hand-scripted
    sequence for a deterministic unit test, or (Day 3) a recorded replay.
    Swapping backends is a constructor change at the call site, never a
    change to the loop itself.
    INTERVIEW: "How do you test an agent loop without hitting the real
    API?" -> the loop depends on this Protocol, not a concrete client;
    tests inject ScriptedModel, which returns a scripted list of
    ModelResponse and drives every exit condition deterministically.
    """

    async def create(
        self,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse: ...
