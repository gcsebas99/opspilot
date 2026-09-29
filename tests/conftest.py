from collections.abc import Iterator
from typing import Any

import httpx2
import pytest


class NetworkDisabledInTests(RuntimeError):
    pass


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise NetworkDisabledInTests(
        "a unit test tried to make a real HTTP request -- inject a ScriptedModel / "
        "cassette instead (CLAUDE.md: unit tests never call the real Anthropic API)"
    )


# [HARNESS:EVAL] Test-suite network guard -- "no real API in unit tests",
# enforced rather than hoped for.
# WHY: a default argument (a live judge client) once made runner tests send
# real requests; they only "passed" because auth failed and the error was
# recorded as data. Patching the SDK's real transport makes any such leak a
# loud failure. httpx2.MockTransport (used to test AnthropicModel's retry
# logic) is a different transport, so those tests are unaffected.
@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", _refuse)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", _refuse)
    yield
