import anthropic
import httpx2
import pytest

from opspilot.models.anthropic_model import AnthropicModel
from opspilot.models.base import TextBlock

_MESSAGE_JSON = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 5},
}

_OVERLOADED_JSON = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
_BAD_REQUEST_JSON = {
    "type": "error",
    "error": {"type": "invalid_request_error", "message": "bad request"},
}


def _client(handler: object) -> anthropic.AsyncAnthropic:
    http_client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    return anthropic.AsyncAnthropic(api_key="test-key", http_client=http_client, max_retries=0)


async def test_retries_on_529_twice_then_succeeds() -> None:
    call_count = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx2.Response(529, json=_OVERLOADED_JSON)
        return httpx2.Response(200, json=_MESSAGE_JSON)

    model = AnthropicModel(client=_client(handler), model="claude-opus-5")

    response = await model.create(
        system="be helpful", messages=[{"role": "user", "content": "hi"}], tools=[]
    )

    assert call_count == 3
    assert response.stop_reason == "end_turn"
    assert response.content == [TextBlock(text="ok")]
    assert response.usage.input_tokens == 10
    assert response.latency_ms >= 0


async def test_non_retryable_error_surfaces_immediately() -> None:
    call_count = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal call_count
        call_count += 1
        return httpx2.Response(400, json=_BAD_REQUEST_JSON)

    model = AnthropicModel(client=_client(handler), model="claude-opus-5")

    with pytest.raises(anthropic.BadRequestError):
        await model.create(system="x", messages=[{"role": "user", "content": "hi"}], tools=[])

    assert call_count == 1


async def test_retries_on_plain_500_too() -> None:
    # 529 "overloaded" and generic >=500 are sibling exception classes in the
    # SDK (anthropic.OverloadedError, anthropic.InternalServerError), not a
    # subclass relationship -- cover both so a future edit that narrows the
    # retryable tuple to just one of them fails a test.
    call_count = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx2.Response(
                500, json={"type": "error", "error": {"type": "api_error", "message": "oops"}}
            )
        return httpx2.Response(200, json=_MESSAGE_JSON)

    model = AnthropicModel(client=_client(handler), model="claude-opus-5")

    response = await model.create(
        system="x", messages=[{"role": "user", "content": "hi"}], tools=[]
    )

    assert call_count == 2
    assert response.stop_reason == "end_turn"


async def test_gives_up_after_max_attempts() -> None:
    call_count = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal call_count
        call_count += 1
        return httpx2.Response(529, json=_OVERLOADED_JSON)

    model = AnthropicModel(client=_client(handler), model="claude-opus-5", max_attempts=2)

    with pytest.raises(anthropic.OverloadedError):
        await model.create(system="x", messages=[{"role": "user", "content": "hi"}], tools=[])

    assert call_count == 2


async def test_normalizes_tool_use_blocks() -> None:
    message = {
        **_MESSAGE_JSON,
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "list_services", "input": {}}],
        "stop_reason": "tool_use",
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=message)

    model = AnthropicModel(client=_client(handler), model="claude-opus-5")

    response = await model.create(
        system="x", messages=[{"role": "user", "content": "hi"}], tools=[]
    )

    assert response.stop_reason == "tool_use"
    assert len(response.content) == 1
    block = response.content[0]
    assert block.type == "tool_use"
    assert block.id == "toolu_1"
    assert block.name == "list_services"
