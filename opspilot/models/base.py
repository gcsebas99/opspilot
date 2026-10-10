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

    [HARNESS:ORCH] One normalized response shape across every model backend.
    WHY: the loop depends on this Protocol, not a client, so live API, scripted
    tests and recorded replays are interchangeable -- swapping backends is a
    constructor change, never a change to the loop.

    `tool_choice` is optional and only the LLM judge (evals/graders/judge.py)
    sets it, to force a structured answer; the agent loops never do.
    """

    async def create(
        self,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
    ) -> ModelResponse: ...
