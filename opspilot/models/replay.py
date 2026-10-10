from typing import Any

from opspilot.models.base import ModelResponse
from opspilot.models.cassette import Cassette, CassetteMiss, normalize_request, request_key


class ReplayModel:
    """Serves recorded responses from a cassette; never touches the network.

    [HARNESS:EVAL] Replay mode: a cache miss fails loudly, never falls back.
    WHY: a miss means the prompt, tools, model or loop changed. Calling the API would
    make CI cost money and flake; a "nearest" answer would test a conversation that
    never happened. The error names the first differing field to re-record.
    """

    def __init__(self, cassette: Cassette, model: str) -> None:
        self._cassette = cassette
        self._model = model

    async def create(
        self,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | None = None,
    ) -> ModelResponse:
        request = normalize_request(self._model, system, messages, tools, tool_choice)
        entry = self._cassette.get(request_key(request))
        if entry is None:
            raise CassetteMiss(self._cassette.explain_miss(request))
        return entry.response
