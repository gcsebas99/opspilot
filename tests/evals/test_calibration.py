from typing import Any

import pytest
from typer.testing import CliRunner

from evals.calibration import (
    CalibrationItem,
    ItemResult,
    agreement,
    build_trial,
    load_calibration,
    run_judge_check,
)
from evals.graders.judge import CRITERIA
from evals.loader import load_cases
from opspilot.cli import app
from opspilot.config import Settings, get_settings
from opspilot.models.base import ModelResponse, ToolUseBlock, Usage
from opspilot.models.scripted import ScriptedModel

_SETTINGS = Settings.model_validate(
    {"ANTHROPIC_API_KEY": "", "OPSPILOT_MODEL": "m", "OPSPILOT_JUDGE_MODEL": "claude-sonnet-5"}
)


def _verdict(score: int) -> ModelResponse:
    return ModelResponse(
        content=[
            ToolUseBlock(
                id="j",
                name="submit_grades",
                input={c: {"score": score, "rationale": "r"} for c in CRITERIA},
            )
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50),
        latency_ms=1.0,
    )


def _result(expected: list[int | None], judged: list[int]) -> ItemResult:
    return ItemResult(
        id="i",
        tier="good",
        expected=dict(zip(CRITERIA, expected, strict=True)),
        judged=dict(zip(CRITERIA, judged, strict=True)),
    )


# --- the dataset itself ---


def test_calibration_set_covers_every_tier_and_a_hallucination() -> None:
    items = load_calibration()
    assert len(items) == 6
    assert sorted(i.tier for i in items) == ["bad", "bad", "good", "good", "mediocre", "mediocre"]
    assert any("hallucinat" in i.id for i in items)
    golden = {c.id for c in load_cases("golden")}
    assert {i.case for i in items} <= golden


def test_every_calibration_trajectory_runs_for_real() -> None:
    cases = {c.id: c for c in load_cases("golden")}
    for item in load_calibration():
        trial = build_trial(item, cases[item.case], _SETTINGS)
        assert len(trial.tool_calls) == len(item.tool_calls)
        assert all(call["content"] for call in trial.tool_calls)
        assert trial.sandbox_facts


def test_build_trial_rejects_a_tool_call_that_fails() -> None:
    item = CalibrationItem(
        id="broken",
        case="pool-exhaustion-operator-approve",
        tier="bad",
        why="x",
        outcome="completed",
        tool_calls=[("rollback_config", {"service": "checkout", "version": 99})],
        report={},
    )
    case = {c.id: c for c in load_cases("golden")}[item.case]
    with pytest.raises(ValueError, match="broken"):
        build_trial(item, case, _SETTINGS)


# --- agreement math ---


def test_perfect_agreement() -> None:
    within, mae, pass_agree, per = agreement([_result([5, 5, 4, 4, 5], [5, 5, 4, 4, 5])])
    assert (within, mae, pass_agree) == (1.0, 0.0, 1.0)
    assert set(per.values()) == {0.0}


def test_off_by_one_counts_as_agreement_but_not_off_by_two() -> None:
    within, mae, _, per = agreement([_result([5, 5, 5, 5, 1], [4, 5, 5, 5, 3])])
    assert within == pytest.approx(4 / 5)
    assert mae == pytest.approx(3 / 5)
    assert per["no_hallucination"] == 2.0


def test_pass_fail_disagreement_is_detected() -> None:
    # Human: avg 2.0 (fail). Judge: avg 4.0 (pass) -- e.g. missed a hallucination.
    _, _, pass_agree, _ = agreement(
        [_result([5, 5, 5, 5, 5], [5, 5, 5, 5, 5]), _result([2, 2, 2, 2, 2], [4, 4, 4, 4, 4])]
    )
    assert pass_agree == 0.5


def test_unscored_items_are_excluded_from_agreement() -> None:
    within, _, _, _ = agreement(
        [_result([5, 5, 5, 5, 5], [5, 5, 5, 5, 5]), _result([None] * 5, [1, 1, 1, 1, 1])]
    )
    assert within == 1.0


async def test_run_judge_check_trusts_a_judge_that_matches_humans() -> None:
    items = [
        i.model_copy(update={"expected": dict.fromkeys(CRITERIA, 4)}) for i in load_calibration()
    ]
    report = await run_judge_check(
        items, settings=_SETTINGS, judge_client_for=lambda _i: ScriptedModel([_verdict(4)])
    )
    assert report.trusted
    assert report.within_one == 1.0 and report.pass_agreement == 1.0


# --- CLI blinding ---


def _patch_judge(monkeypatch: pytest.MonkeyPatch, score: int) -> None:
    def fake_client(*_args: Any, **_kwargs: Any) -> ScriptedModel:
        return ScriptedModel([_verdict(score)])

    monkeypatch.setattr("opspilot.cli.build_model_client", fake_client)
    # Don't depend on the developer's .env (CI has none).
    monkeypatch.setenv("OPSPILOT_JUDGE_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("OPSPILOT_MODEL_MODE", "replay")
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_settings() -> Any:
    yield
    get_settings.cache_clear()


def test_judge_check_hides_judge_scores_until_humans_have_scored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_judge(monkeypatch, 3)
    result = CliRunner().invoke(app, ["eval", "judge-check", "--mode", "replay"])

    assert result.exit_code == 1
    assert "have no human scores yet" in result.stdout
    assert "judge=" not in result.stdout  # no judge averages leaked


def test_judge_check_reveal_shows_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_judge(monkeypatch, 3)
    result = CliRunner().invoke(app, ["eval", "judge-check", "--mode", "replay", "--reveal"])

    assert "judge=3.0" in result.stdout
