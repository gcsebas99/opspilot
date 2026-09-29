import anthropic
import pytest

from opspilot.models.anthropic_model import AnthropicModel
from tests.conftest import NetworkDisabledInTests


async def test_real_api_calls_are_blocked_in_tests() -> None:
    client = anthropic.AsyncAnthropic(api_key="sk-test", max_retries=0)
    model = AnthropicModel(client=client, model="claude-haiku-4-5", max_attempts=1)

    with pytest.raises(Exception) as excinfo:
        await model.create(system="s", messages=[{"role": "user", "content": "hi"}], tools=[])

    chain: list[BaseException] = []
    exc: BaseException | None = excinfo.value
    while exc is not None:
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    assert any(isinstance(e, NetworkDisabledInTests) for e in chain), chain
