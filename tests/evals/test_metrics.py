from datetime import UTC, datetime
from typing import Any

import pytest

from evals.metrics import case_verdict, compute_metrics, percentile, rates, trial_passed
from evals.models import EvalCase
from opspilot.store.models import EvalTrialDoc


def _case(case_id: str, tags: list[str]) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": case_id,
            "scenario": "false_alarm",
            "seed": 42,
            "role": "viewer",
            "tags": tags,
            "expect": {"outcome": "completed"},
            "budgets": {"max_steps": 1, "max_tokens": 1, "max_cost_usd": 1, "max_latency_s": 1},
        }
    )


def _trial(
    case_id: str,
    trial: int,
    passed: bool,
    *,
    mode: str = "live",
    cost: float = 0.02,
    latency: float = 10.0,
    judge: float | None = 4.0,
    **extra: Any,
) -> EvalTrialDoc:
    grades: dict[str, Any] = {
        "trajectory": {"passed": passed, "score": 1.0 if passed else 0.5, "details": {}},
        "outcome": {"passed": True, "score": 1.0, "details": {}},
        "budget": {"passed": True, "score": 1.0, "details": {}},
    }
    if judge is not None:
        grades["judge"] = {
            "passed": judge >= 3.5,
            "score": (judge - 1) / 4,
            "details": {"average": judge, "cost_usd": 0.01},
        }
    return EvalTrialDoc(
        trial_id=f"{case_id}:{trial}",
        sweep_id="s",
        suite="golden",
        case_id=case_id,
        trial=trial,
        run_id="r",
        git_sha="x",
        prompt_version="v",
        model="m",
        strategy="graph",
        mode=mode,  # type: ignore[arg-type]
        started_at=datetime.now(UTC),
        passed=passed,
        grades=grades,
        cost_usd=cost,
        latency_s=latency,
        tokens={"input_tokens": 900, "output_tokens": 100},
        **extra,
    )


# a: always passes, b: flaky (1 of 3), c: always fails
def _sweep(mode: str = "live") -> list[EvalTrialDoc]:
    return [
        *[_trial("a", i, True, mode=mode) for i in range(3)],
        _trial("b", 0, True, mode=mode),
        _trial("b", 1, False, mode=mode),
        _trial("b", 2, False, mode=mode),
        *[_trial("c", i, False, mode=mode) for i in range(3)],
    ]


_CASES = {
    "a": _case("a", ["smoke", "config"]),
    "b": _case("b", ["adversarial"]),
    "c": _case("c", ["adversarial", "config"]),
}


def test_pass_at_k_vs_pass_hat_k_separate_capability_from_reliability() -> None:
    m = compute_metrics(_sweep(), _CASES)

    assert m.k == 3
    assert m.overall.pass_at_1 == pytest.approx(4 / 9)  # 4 of 9 trials
    assert m.overall.pass_at_k == pytest.approx(2 / 3)  # a and b solved at least once
    assert m.overall.pass_hat_k == pytest.approx(1 / 3)  # only a solved every time
    assert m.case_verdicts == {"a": "pass", "b": "flaky", "c": "fail"}


def test_k_equals_one_collapses_all_three_rates() -> None:
    r = rates({"a": [_trial("a", 0, True)], "c": [_trial("c", 0, False)]})
    assert r.pass_at_1 == r.pass_at_k == r.pass_hat_k == 0.5


def test_per_tag_breakdown() -> None:
    m = compute_metrics(_sweep(), _CASES)

    assert set(m.per_tag) == {"adversarial", "config", "smoke"}
    assert m.per_tag["adversarial"].cases == 2
    assert m.per_tag["adversarial"].pass_at_k == pytest.approx(1 / 2)  # b yes, c no
    assert m.per_tag["adversarial"].pass_hat_k == 0.0
    assert m.per_tag["config"].pass_hat_k == pytest.approx(1 / 2)  # a yes, c no


def test_per_grader_rates_show_which_layer_fails() -> None:
    m = compute_metrics(_sweep(), _CASES)
    assert m.per_grader["trajectory"] == pytest.approx(4 / 9)
    assert m.per_grader["outcome"] == 1.0
    assert m.per_grader["judge"] == 1.0


def test_errored_and_grading_errored_trials_count_as_failures() -> None:
    crashed = _trial("a", 0, True, error="RuntimeError: boom")
    ungraded = _trial("a", 1, True, grading_error="CassetteMiss: ...")
    assert not trial_passed(crashed) and not trial_passed(ungraded)
    assert case_verdict([crashed, ungraded, _trial("a", 2, True)]) == "flaky"

    m = compute_metrics([crashed, ungraded], _CASES)
    assert m.errored == 2
    assert m.overall.pass_at_1 == 0.0


def test_latency_only_from_live_trials_but_cost_kept_in_replay() -> None:
    replay = compute_metrics(_sweep(mode="replay"), _CASES)
    assert replay.latency_s is None
    assert replay.agent_cost is not None and replay.agent_cost.total == pytest.approx(0.18)

    live = compute_metrics(_sweep(mode="live"), _CASES)
    assert live.latency_s is not None and live.latency_s.mean == 10.0


def test_cost_tokens_and_judge_aggregates() -> None:
    m = compute_metrics(_sweep(), _CASES)
    assert m.judge_cost is not None and m.judge_cost.total == pytest.approx(0.09)
    assert m.tokens is not None and m.tokens.mean == 1000
    assert m.judge_avg == 4.0

    no_judge = compute_metrics([_trial("a", 0, True, judge=None)], _CASES)
    assert no_judge.judge_avg is None and no_judge.judge_cost is None


def test_nearest_rank_percentile() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.95) == 3.0  # few samples -> p95 is the max
    assert percentile([float(i) for i in range(1, 101)], 0.95) == 95.0
    assert percentile([7.0], 0.5) == 7.0


def test_empty_sweep_is_all_zero_not_an_error() -> None:
    m = compute_metrics([], {})
    assert m.k == 0 and m.overall.cases == 0 and m.agent_cost is None
