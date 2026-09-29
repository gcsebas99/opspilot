from datetime import UTC, datetime
from typing import Any

import pytest

from evals.graders.budget import grade_budget
from evals.graders.outcome import grade_outcome
from evals.models import EvalCase
from opspilot.store.models import EvalTrialDoc


def _case(root_cause: str | None = "config_change:checkout:db_pool_size") -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "c",
            "scenario": "checkout_pool_exhaustion",
            "seed": 42,
            "role": "operator",
            "expect": {"outcome": "completed", "root_cause": root_cause},
            "budgets": {
                "max_steps": 10,
                "max_tokens": 1000,
                "max_cost_usd": 0.10,
                "max_latency_s": 30,
            },
        }
    )


def _trial(**fields: Any) -> EvalTrialDoc:
    base: dict[str, Any] = {
        "trial_id": "t",
        "sweep_id": "s",
        "suite": "golden",
        "case_id": "c",
        "trial": 0,
        "run_id": "r",
        "git_sha": "x",
        "prompt_version": "v",
        "model": "m",
        "strategy": "graph",
        "started_at": datetime.now(UTC),
    }
    return EvalTrialDoc(**{**base, **fields})


# --- outcome ---


@pytest.mark.parametrize(
    ("reported", "score", "match"),
    [
        ("config_change:checkout:db_pool_size", 1.0, "exact"),
        ("config_change:checkout:timeout_ms", 0.5, "category+service"),
        ("config_change:checkout", 0.5, "category+service"),
        ("config_change:payments:db_pool_size", 0.0, "wrong category or service"),
        ("bad_deploy:checkout", 0.0, "wrong category or service"),
        ("no_incident", 0.0, "incident vs no_incident mismatch"),
    ],
)
def test_root_cause_partial_credit(reported: str, score: float, match: str) -> None:
    grade = grade_outcome(_case(), _trial(report={"root_cause": reported}))
    assert (grade.score, grade.details["match"]) == (score, match)
    assert grade.passed is (score >= 0.5)


def test_escalation_is_graded_on_suspected_root_cause() -> None:
    trial = _trial(
        report={
            "reason": "disk full",
            "suspected_root_cause": "config_change:checkout:db_pool_size",
        }
    )
    assert grade_outcome(_case(), trial).score == 1.0


def test_escalation_without_a_hypothesis_scores_zero() -> None:
    grade = grade_outcome(_case(), _trial(report={"reason": "no idea"}))
    assert (grade.score, grade.passed) == (0.0, False)
    assert grade.details["match"] == "no root cause reported"


def test_no_incident_is_all_or_nothing() -> None:
    case = _case("no_incident")
    assert grade_outcome(case, _trial(report={"root_cause": "no_incident"})).score == 1.0
    assert grade_outcome(case, _trial(report={"root_cause": "no_incident_x"})).score == 0.0


def test_no_root_cause_expectation_passes() -> None:
    assert grade_outcome(_case(None), _trial()).passed


# --- budget ---


def test_within_budget_passes() -> None:
    trial = _trial(
        steps=5,
        tokens={"input_tokens": 500, "output_tokens": 100},
        cost_usd=0.05,
        latency_s=10.0,
    )
    grade = grade_budget(_case(), trial)
    assert grade.passed and grade.score == 1.0


def test_tokens_include_cache_and_each_overrun_is_named() -> None:
    trial = _trial(
        steps=11,
        tokens={"input_tokens": 100, "output_tokens": 100, "cache_read_input_tokens": 900},
        cost_usd=0.05,
        latency_s=10.0,
    )
    grade = grade_budget(_case(), trial)
    failed = [c["name"] for c in grade.details["checks"] if not c["passed"]]
    assert failed == ["steps", "tokens"]
    assert grade.score == 0.5


def test_replay_skips_latency_but_not_cost() -> None:
    trial = _trial(
        mode="replay",
        steps=1,
        tokens={"input_tokens": 1},
        cost_usd=0.50,
        latency_s=999.0,
    )
    checks = {c["name"]: c for c in grade_budget(_case(), trial).details["checks"]}
    assert checks["latency_s"]["passed"] and "skipped" in checks["latency_s"]["detail"]
    assert not checks["cost_usd"]["passed"]


def test_errored_trial_fails_budget_rather_than_passing_by_default() -> None:
    grade = grade_budget(_case(), _trial(error="boom"))
    assert not grade.passed
