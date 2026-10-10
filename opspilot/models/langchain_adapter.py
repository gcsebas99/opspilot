from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.ai import InputTokenDetails, UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from opspilot.models.base import ModelClient, ModelResponse, TextBlock, ToolUseBlock


def _as_blocks(content: str | list[Any]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [b if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in content]


def _ai_to_wire(message: AIMessage) -> dict[str, Any]:
    # Text comes from `content`; tool_use comes from `tool_calls`, which is
    # LangChain's authoritative field (ChatAnthropic also mirrors tool_use
    # blocks into `content` -- skip those so they aren't sent twice).
    blocks = [b for b in _as_blocks(message.content) if b.get("type") == "text" and b.get("text")]
    blocks += [
        {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]}
        for tc in message.tool_calls
    ]
    return {"role": "assistant", "content": blocks}


def to_anthropic_messages(
    messages: Sequence[BaseMessage],
) -> tuple[list[dict[str, Any]] | str, list[dict[str, Any]]]:
    """LangChain messages -> (system, messages) in Anthropic wire format.

    Produces exactly the shape the raw loop builds by hand, so a request
    from either strategy hashes the same way (see models/cassette.py).
    """
    system: list[dict[str, Any]] | str = []
    wire: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system = cast("list[dict[str, Any]] | str", message.content)
            continue
        if isinstance(message, AIMessage):
            wire.append(_ai_to_wire(message))
            continue
        if isinstance(message, ToolMessage):
            content: str | list[Any] = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                    "is_error": message.status == "error",
                }
            ]
        elif isinstance(message, HumanMessage):
            content = message.content
        else:
            raise TypeError(f"unsupported message type {type(message).__name__}")

        # The API wants one user turn per tool round: parallel tool results
        # (and a stuck-nudge the graph emits alongside them) are separate
        # LangChain messages but must merge into one user message -- the
        # same thing the raw loop does by collecting `result_blocks`.
        if wire and wire[-1]["role"] == "user":
            wire[-1]["content"] = _as_blocks(wire[-1]["content"]) + _as_blocks(content)
        else:
            wire.append({"role": "user", "content": content})
    return system, wire


def to_ai_message(response: ModelResponse) -> AIMessage:
    blocks: list[str | dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if isinstance(block, TextBlock):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            blocks.append(
                {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
            )
            tool_calls.append(ToolCall(name=block.name, args=block.input, id=block.id))
    usage = response.usage
    # LangChain convention (matches ChatAnthropic): input_tokens is the TOTAL
    # prompt size including cache reads/writes; the split lives in details.
    total_input = (
        usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
    )
    usage_metadata = UsageMetadata(
        input_tokens=total_input,
        output_tokens=usage.output_tokens,
        total_tokens=total_input + usage.output_tokens,
        input_token_details=InputTokenDetails(
            cache_read=usage.cache_read_input_tokens,
            cache_creation=usage.cache_creation_input_tokens,
        ),
    )
    return AIMessage(
        content=blocks,
        tool_calls=tool_calls,
        usage_metadata=usage_metadata,
        response_metadata={"stop_reason": response.stop_reason, "latency_ms": response.latency_ms},
    )


class ModelClientChatModel(BaseChatModel):
    """A LangChain chat model backed by any opspilot ModelClient.

    [HARNESS:ORCH] One seam for record/replay across both loop strategies.
    WHY: adapting LangChain -> ModelClient means one recording/replay implementation
    and one cassette format, with no change to the graph's nodes. Tradeoff: in
    record/replay the graph skips ChatAnthropic's own message conversion.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any  # ModelClient -- a Protocol, which pydantic can't validate

    @property
    def _llm_type(self) -> str:
        return "opspilot-model-client"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        # Only Anthropic-format dicts (ToolRegistry.to_anthropic_schema()) --
        # converting pydantic classes/functions to schemas is ChatAnthropic's
        # job, and nothing in this project needs it here.
        if not all(isinstance(t, dict) for t in tools):
            raise TypeError("ModelClientChatModel.bind_tools accepts Anthropic tool dicts only")
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("ModelClientChatModel is async-only; use ainvoke()")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        client: ModelClient = self.client
        system, wire = to_anthropic_messages(messages)
        response = await client.create(system=system, messages=wire, tools=kwargs.get("tools", []))
        return ChatResult(generations=[ChatGeneration(message=to_ai_message(response))])
