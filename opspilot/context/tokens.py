from typing import Any, Protocol


class _MessagesNamespace(Protocol):
    def count_tokens(self, **kwargs: Any) -> Any: ...


class TokenCountingClient(Protocol):
    """The slice of `anthropic.Anthropic` this module needs.

    A Protocol, not `anthropic.Anthropic` itself, so tests can pass a tiny
    fake with no API key and no network access -- unit tests never call the
    real Anthropic API (CLAUDE.md).
    """

    messages: _MessagesNamespace


def count_tokens(
    client: TokenCountingClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Count input tokens for a would-be request via the Anthropic count_tokens endpoint.

    Used for Day 4 compaction (deciding when to summarize) and for reporting
    estimated cost before a run. `system`/`tools` are omitted from the call
    entirely when not given, rather than passed as None -- the SDK
    distinguishes "not provided" from an explicit null.
    """
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if system is not None:
        kwargs["system"] = system
    if tools is not None:
        kwargs["tools"] = tools
    response = client.messages.count_tokens(**kwargs)
    return int(response.input_tokens)
