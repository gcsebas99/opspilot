import time
from typing import Any, cast

import anthropic
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from opspilot.models.base import ContentBlock, ModelResponse, TextBlock, ToolUseBlock, Usage

# 429, >=500, and 529 "overloaded" (its own sibling class in the installed
# SDK, not a subclass of InternalServerError -- confirmed against the
# installed anthropic package rather than assumed from docs).
_RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
)


def _normalize(response: Any, latency_ms: float) -> ModelResponse:
    blocks: list[ContentBlock] = []
    for block in response.content:
        if block.type == "text":
            blocks.append(TextBlock(text=block.text))
        elif block.type == "tool_use":
            blocks.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))
        # thinking/redacted_thinking blocks are dropped -- no extended-thinking
        # config is sent yet at Day 1.

    usage = Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
    )
    return ModelResponse(
        content=blocks,
        stop_reason=response.stop_reason or "end_turn",
        usage=usage,
        latency_ms=latency_ms,
    )


class AnthropicModel:
    """Real Anthropic backend.

    Owns its own retry policy rather than the SDK's built-in one -- pass a
    client constructed with `max_retries=0` so the two don't double-retry
    with conflicting backoff schedules.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        model: str,
        max_tokens: int = 8192,
        max_attempts: int = 5,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts

    # [HARNESS:ORCH] Exponential backoff + jitter on retryable errors only.
    # WHY: 429/5xx/overloaded are transient server-side conditions -- back
    # off and retry, with jitter so many concurrent runs don't retry in
    # lockstep and re-trigger the same rate limit. Everything else (400 bad
    # request, 401 auth, 404 unknown model) is a bug in our own request, not
    # a transient condition; retrying it would just waste time and tokens,
    # so it's excluded from the retry predicate and surfaces immediately.
    # INTERVIEW: "What do you retry, and what fails fast?" -> RateLimitError
    # (429), InternalServerError (>=500), and OverloadedError (529 -- a
    # sibling class, not a subclass of InternalServerError) retry with
    # backoff+jitter, capped at max_attempts; every other APIError is not
    # retryable and propagates as-is (reraise=True, no RetryError wrapping).
    async def create(
        self,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            wait=wait_random_exponential(multiplier=1, max=20),
            stop=stop_after_attempt(self._max_attempts),
            reraise=True,
        )

        # tenacity's AsyncRetrying auto-detects whether `fn` is a coroutine
        # function to decide whether to await it -- that detection doesn't
        # reliably see through the SDK's bound `client.messages.create`, so
        # wrap it in an unambiguous `async def` instead of passing it directly.
        # The SDK's create() wants precise TypedDicts (TextBlockParam,
        # MessageParam, ToolParam, ...); ModelClient deliberately takes plain
        # dicts so every backend (real/scripted/replay) shares one loose
        # Protocol. We know these dicts are shaped correctly at runtime --
        # cast at this one boundary rather than tighten the Protocol.
        async def _call() -> Any:
            return await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=cast(Any, system),
                messages=cast(Any, messages),
                tools=cast(Any, tools),
            )

        start = time.monotonic()
        response: Any = await retrying(_call)
        latency_ms = (time.monotonic() - start) * 1000
        return _normalize(response, latency_ms)
