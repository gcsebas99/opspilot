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


# [HARNESS:EVAL] Network guard: "no real API in unit tests", enforced.
# WHY: a default argument once made tests send real (auth-failing) requests that
# still "passed". Patching the SDK's real transport turns any leak into a loud
# failure; httpx2.MockTransport (used by retry tests) is unaffected.
@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", _refuse)
    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", _refuse)
    yield
