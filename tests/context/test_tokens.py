from typing import Any

from opspilot.context.tokens import count_tokens


class _FakeCountTokensResponse:
    def __init__(self, input_tokens: int) -> None:
        self.input_tokens = input_tokens


class _FakeMessages:
    def __init__(self, input_tokens: int) -> None:
        self._input_tokens = input_tokens
        self.calls: list[dict[str, Any]] = []

    def count_tokens(self, **kwargs: Any) -> _FakeCountTokensResponse:
        self.calls.append(kwargs)
        return _FakeCountTokensResponse(self._input_tokens)


class _FakeClient:
    def __init__(self, input_tokens: int) -> None:
        self.messages = _FakeMessages(input_tokens)


def test_count_tokens_returns_input_tokens() -> None:
    client = _FakeClient(input_tokens=123)

    result = count_tokens(
        client, model="claude-opus-5", messages=[{"role": "user", "content": "hi"}]
    )

    assert result == 123


def test_count_tokens_passes_system_and_tools_when_given() -> None:
    client = _FakeClient(input_tokens=1)

    count_tokens(
        client,
        model="claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        system="be terse",
        tools=[{"name": "x"}],
    )

    assert client.messages.calls[0]["system"] == "be terse"
    assert client.messages.calls[0]["tools"] == [{"name": "x"}]


def test_count_tokens_omits_system_and_tools_when_not_given() -> None:
    client = _FakeClient(input_tokens=1)

    count_tokens(client, model="claude-opus-5", messages=[{"role": "user", "content": "hi"}])

    assert "system" not in client.messages.calls[0]
    assert "tools" not in client.messages.calls[0]
