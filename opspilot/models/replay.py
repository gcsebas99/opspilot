from typing import Any

from opspilot.models.base import ModelResponse
from opspilot.models.cassette import Cassette, CassetteMiss, normalize_request, request_key


class ReplayModel:
    """Serves recorded responses from a cassette; never touches the network.

    [HARNESS:EVAL] Replay mode -- a cache miss fails loudly, never falls back.
    WHY: a miss means the request changed (prompt, tools, model, or loop
    logic). Silently calling the live API would make CI cost money and go
    flaky; returning some "nearest" response would test a conversation that
    never happened. Failing with the first differing field says what to re-record.
    INTERVIEW: "What does replay actually test?" -> the harness (loop,
    tools, policy, graders) deterministically; live runs test model+prompt.
    """

    def __init__(self, cassette: Cassette, model: str) -> None:
        self._cassette = cassette
        self._model = model

    async def create(
        self,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        request = normalize_request(self._model, system, messages, tools)
        entry = self._cassette.get(request_key(request))
        if entry is None:
            raise CassetteMiss(self._cassette.explain_miss(request))
        return entry.response
