from typing import Any

from opspilot.models.base import ModelResponse


class ScriptedModel:
    """Returns a predefined list of ModelResponse in sequence.

    Drives the loop (1.6) deterministically in tests -- no network, no
    randomness. Day 3 adds ReplayModel for recorded real-call cassettes;
    this is for hand-written exit-condition tests where you want to state
    exactly what the model "said" on each turn.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls: list[dict[str, Any]] = []

    async def create(
        self,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        # Snapshot (shallow-copy) the list arguments -- the loop keeps
        # mutating its `messages` list via .append() on every turn, so
        # storing the live reference would make every past call's recorded
        # "messages" silently grow to match the CURRENT state by the time a
        # test inspects it after the run finishes.
        self.calls.append(
            {
                "system": list(system) if isinstance(system, list) else system,
                "messages": list(messages),
                "tools": list(tools),
            }
        )
        if self._index >= len(self._responses):
            raise IndexError(
                f"ScriptedModel exhausted: only {len(self._responses)} response(s) scripted, "
                f"but call {self._index + 1} was requested"
            )
        response = self._responses[self._index]
        self._index += 1
        return response
