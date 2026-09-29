from pathlib import Path

import pytest

from opspilot.config import Settings
from opspilot.models.factory import (
    LiveCallRefused,
    build_model_client,
    demo_cassette_path,
    resolve_mode,
)
from opspilot.models.replay import ReplayModel
from opspilot.models.scripted import ScriptedModel


def _settings(api_key: str = "", mode: str = "replay") -> Settings:
    return Settings.model_validate(
        {"ANTHROPIC_API_KEY": api_key, "OPSPILOT_MODEL": "m", "OPSPILOT_MODEL_MODE": mode}
    )


def test_default_mode_is_replay_and_needs_no_key() -> None:
    assert Settings.model_fields["opspilot_model_mode"].default == "replay"
    assert resolve_mode(None, _settings()) == "replay"


@pytest.mark.parametrize("mode", ["live", "record"])
def test_paid_modes_refused_without_api_key(mode: str) -> None:
    with pytest.raises(LiveCallRefused, match="ANTHROPIC_API_KEY is empty"):
        resolve_mode(mode, _settings())


def test_paid_mode_from_env_is_also_guarded() -> None:
    with pytest.raises(LiveCallRefused):
        resolve_mode(None, _settings(mode="live"))


def test_explicit_mode_overrides_env_default() -> None:
    assert resolve_mode("live", _settings(api_key="sk-test")) == "live"
    assert resolve_mode("replay", _settings(api_key="sk-test", mode="live")) == "replay"


def test_unknown_mode_rejected() -> None:
    with pytest.raises(ValueError, match="got 'cheap'"):
        resolve_mode("cheap", _settings())


def test_replay_never_builds_a_live_client(tmp_path: Path) -> None:
    def explode(_settings: Settings) -> ScriptedModel:
        raise AssertionError("live client constructed in replay mode")

    client = build_model_client(_settings(), "replay", tmp_path / "c.jsonl", live_client=explode)
    assert isinstance(client, ReplayModel)


def test_record_and_replay_require_a_cassette_path() -> None:
    with pytest.raises(ValueError, match="needs a cassette path"):
        build_model_client(_settings(), "replay", None)


def test_demo_cassette_path_has_no_strategy_component() -> None:
    assert demo_cassette_path("false_alarm", 42, "viewer") == Path(
        "evals/cassettes/demo/false_alarm-s42-viewer.jsonl"
    )
