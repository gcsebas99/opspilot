import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage

from opspilot.models.base import ModelResponse, TextBlock, ToolUseBlock, Usage
from opspilot.models.langchain_adapter import (
    ModelClientChatModel,
    to_ai_message,
    to_anthropic_messages,
)
from opspilot.models.scripted import ScriptedModel

TOOLS = [{"name": "grep_logs", "description": "d", "input_schema": {"type": "object"}}]


def test_parallel_tool_results_and_nudge_merge_into_one_user_turn() -> None:
    messages = [
        SystemMessage(content=[{"type": "text", "text": "sys"}]),
        HumanMessage(content="New alert:\nx"),
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="a", args={}, id="t1"),
                ToolCall(name="b", args={"s": 1}, id="t2"),
            ],
        ),
        ToolMessage(content="out a", tool_call_id="t1"),
        ToolMessage(content="boom", tool_call_id="t2", status="error"),
        HumanMessage(content="stuck nudge"),
    ]
    system, wire = to_anthropic_messages(messages)

    assert system == [{"type": "text", "text": "sys"}]
    assert wire == [
        {"role": "user", "content": "New alert:\nx"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "b", "input": {"s": 1}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "out a", "is_error": False},
                {"type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": True},
                {"type": "text", "text": "stuck nudge"},
            ],
        },
    ]


def test_chat_anthropic_style_message_does_not_duplicate_tool_use() -> None:
    # ChatAnthropic mirrors tool_use into `content` as well as `tool_calls`.
    message = AIMessage(
        content=[
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
        ],
        tool_calls=[ToolCall(name="a", args={}, id="t1")],
    )
    _, wire = to_anthropic_messages([message])
    assert wire[0]["content"] == [
        {"type": "text", "text": "checking"},
        {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
    ]


def test_to_ai_message_uses_langchain_total_input_convention() -> None:
    response = ModelResponse(
        content=[TextBlock(text="hm"), ToolUseBlock(id="t1", name="a", input={"x": 1})],
        stop_reason="tool_use",
        usage=Usage(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=1000,
            cache_creation_input_tokens=50,
        ),
        latency_ms=5.0,
    )
    message = to_ai_message(response)

    assert message.tool_calls == [ToolCall(name="a", args={"x": 1}, id="t1", type="tool_call")]
    assert message.usage_metadata is not None
    assert message.usage_metadata["input_tokens"] == 1150  # total, like ChatAnthropic
    assert message.usage_metadata["input_token_details"] == {
        "cache_read": 1000,
        "cache_creation": 50,
    }
    assert message.response_metadata["stop_reason"] == "tool_use"


async def test_bound_tools_reach_the_model_client() -> None:
    inner = ScriptedModel(
        [
            ModelResponse(
                content=[TextBlock(text="hi")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=1, output_tokens=1),
                latency_ms=1.0,
            )
        ]
    )
    model = ModelClientChatModel(client=inner).bind_tools(TOOLS)
    reply = await model.ainvoke([SystemMessage(content="sys"), HumanMessage(content="hello")])

    assert isinstance(reply, AIMessage)
    assert inner.calls[0]["tools"] == TOOLS
    assert inner.calls[0]["system"] == "sys"


def test_bind_tools_rejects_non_dict_tools() -> None:
    def some_function() -> None: ...

    with pytest.raises(TypeError, match="Anthropic tool dicts only"):
        ModelClientChatModel(client=None).bind_tools([some_function])
