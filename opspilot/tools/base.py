from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from opspilot.env.sandbox import Sandbox

# [HARNESS:TOOLS] Every tool declares a risk class: read, destructive or terminal.
# WHY: the permission policy keys off risk, not tool names, so a new tool is
# governed the moment it's registered -- no policy edit to forget.
Risk = Literal["read", "destructive", "terminal"]


class ToolResult(BaseModel):
    """What a tool call returns to the loop, regardless of success/failure."""

    ok: bool
    content: str
    data: dict[str, Any] | None = None
    truncated: bool = False


# [HARNESS:TOOLS] The tool contract: name, description, input model, risk, function.
# WHY: one pydantic model is both the argument validator and the JSON schema the
# model sees, so what a tool accepts can never drift from what it advertises.
@dataclass(frozen=True)
class Tool:
    """A single callable capability the model can invoke.

    `input_model` is the pydantic model used both to validate incoming
    arguments and (via `model_json_schema()`) to generate the Anthropic tool
    schema -- one source of truth for "what does this tool accept."
    """

    name: str
    description: str
    input_model: type[BaseModel]
    risk: Risk
    fn: Callable[[Sandbox, BaseModel], ToolResult]


def _truncate(content: str, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False

    lines = content.splitlines()
    kept_text = ""
    cutoff_index = len(lines)
    for i, line in enumerate(lines):
        candidate = f"{kept_text}\n{line}" if kept_text else line
        if len(candidate) > max_chars:
            cutoff_index = i
            break
        kept_text = candidate

    remaining = len(lines) - cutoff_index
    hint = f"...[{remaining} more lines truncated, narrow your pattern]"
    truncated_content = f"{kept_text}\n{hint}" if kept_text else hint
    return truncated_content, True


class ToolRegistry:
    def __init__(self, max_output_chars: int = 4000) -> None:
        self._tools: dict[str, Tool] = {}
        self.max_output_chars = max_output_chars

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def to_anthropic_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_model.model_json_schema(),
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, raw_args: dict[str, Any], sandbox: Sandbox) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(ok=False, content=f"unknown tool {name!r}")

        # [HARNESS:TOOLS] Schema-driven self-correction: bad args become a tool result.
        # WHY: tool args are model-generated JSON. A validation error returned as
        # ok=False lets the next turn read it and retry, instead of crashing the run.
        try:
            args = tool.input_model.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult(ok=False, content=f"invalid arguments for {name!r}: {exc}")

        # [HARNESS:ORCH] Tool exceptions never crash the loop.
        # WHY: some tool bug is inevitable across a dozen tools; turning the exception
        # into ok=False content lets the model route around it (or escalate).
        try:
            result = tool.fn(sandbox, args)
        except Exception as exc:
            return ToolResult(ok=False, content=f"tool {name!r} raised {type(exc).__name__}: {exc}")

        # [HARNESS:CONTEXT] Bound tool output before it enters the context window.
        # WHY: one log dump can blow the token budget. The explicit truncation hint tells
        # the model it saw a partial result and should narrow its query.
        content, truncated = _truncate(result.content, self.max_output_chars)
        if truncated:
            return result.model_copy(update={"content": content, "truncated": True})
        return result
