import pytest
from pydantic import ValidationError

from evals.models import Budgets, EvalCase, Expect


def _case(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "test-case",
        "scenario": "checkout_pool_exhaustion",
        "seed": 42,
        "role": "operator",
        "expect": {"outcome": "completed"},
        "budgets": {
            "max_steps": 12,
            "max_tokens": 50_000,
            "max_cost_usd": 0.15,
            "max_latency_s": 90,
        },
    }
    base.update(overrides)
    return base


def test_minimal_case_parses_with_defaults() -> None:
    case = EvalCase.model_validate(_case())

    assert case.approval_policy == "approve_all"
    assert case.tags == []
    assert case.expect.must_call == []
    assert case.expect.requires_approval is False
    assert case.rubric_notes == ""


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown scenario"):
        EvalCase.model_validate(_case(scenario="does_not_exist"))


def test_approval_policy_accepts_scripted_dict() -> None:
    case = EvalCase.model_validate(
        _case(approval_policy={"rollback_config": "approve", "restart_service": "reject"})
    )

    assert case.approval_policy == {"rollback_config": "approve", "restart_service": "reject"}


def test_approval_policy_rejects_unknown_string() -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(_case(approval_policy="approve_some"))


def test_expect_order_pairs_parse_as_tuples() -> None:
    expect = Expect.model_validate(
        {"outcome": "completed", "order": [["config_history", "rollback_config"]]}
    )

    assert expect.order == [("config_history", "rollback_config")]


def test_budgets_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        Budgets.model_validate({"max_steps": 12})
