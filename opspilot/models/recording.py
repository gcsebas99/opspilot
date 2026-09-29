from typing import Any

from opspilot.models.base import ModelClient, ModelResponse
from opspilot.models.cassette import Cassette, normalize_request, request_key


class RecordingModel:
    """Wraps a real ModelClient and writes every call to a cassette.

    [HARNESS:EVAL] Record mode -- pay for each distinct request once.
    WHY: a hit is served from the cassette instead of re-calling the API,
    so re-recording a run with an extra branch (e.g. the reject path after
    already recording approve) only pays for the new requests.
    INTERVIEW: "How do you make evals cheap?" -> record real responses
    once, replay them for free; record mode only pays for cache misses.
    """

    def __init__(self, inner: ModelClient, cassette: Cassette, model: str) -> None:
        self._inner = inner
        self._cassette = cassette
        self._model = model
        self.hits = 0
        self.misses = 0

    async def create(
        self,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        request = normalize_request(self._model, system, messages, tools)
        entry = self._cassette.get(request_key(request))
        if entry is not None:
            self.hits += 1
            return entry.response
        self.misses += 1
        response = await self._inner.create(system=system, messages=messages, tools=tools)
        self._cassette.append(request, response)
        return response
