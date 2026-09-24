import pytest

from opspilot.models.base import ModelResponse, TextBlock, Usage
from opspilot.models.scripted import ScriptedModel


def _response(text: str, stop_reason: str = "end_turn") -> ModelResponse:
    return ModelResponse(
        content=[TextBlock(text=text)],
        stop_reason=stop_reason,
        usage=Usage(input_tokens=10, output_tokens=5),
        latency_ms=1.0,
    )


async def test_returns_responses_in_order() -> None:
    model = ScriptedModel([_response("first"), _response("second")])

    r1 = await model.create(system="x", messages=[], tools=[])
    r2 = await model.create(system="x", messages=[], tools=[])

    assert r1.content == [TextBlock(text="first")]
    assert r2.content == [TextBlock(text="second")]


async def test_records_calls() -> None:
    model = ScriptedModel([_response("first")])

    await model.create(
        system="sys", messages=[{"role": "user", "content": "hi"}], tools=[{"name": "x"}]
    )

    assert model.calls == [
        {"system": "sys", "messages": [{"role": "user", "content": "hi"}], "tools": [{"name": "x"}]}
    ]


async def test_raises_clearly_when_exhausted() -> None:
    model = ScriptedModel([_response("only one")])

    await model.create(system="x", messages=[], tools=[])

    with pytest.raises(IndexError, match="exhausted"):
        await model.create(system="x", messages=[], tools=[])
