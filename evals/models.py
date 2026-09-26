from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from opspilot.env.scenarios import get_scenario
from opspilot.policy.permissions import Role

# [HARNESS:EVAL] approval_policy is a case-level fixture, not something the
# agent or a human decides at run time -- the runner (3.2) auto-resolves
# every RequireApproval pause according to this field so a k-trial eval run
# never blocks on real human input. "approve_all"/"reject_all" cover the
# common cases; the dict form scripts a specific decision per tool name for
# cases that need finer control (e.g. approve the rollback but reject a
# restart) without inventing a whole new mini-language.
ApprovalPolicy = Literal["approve_all", "reject_all"] | dict[str, Literal["approve", "reject"]]


class Expect(BaseModel):
    """What a passing trial must look like. Every field is optional except
    `outcome` -- an empty list/None means "no constraint", not "must be
    empty" (an empty `must_call`, for instance, says nothing about which
    tools were or weren't called)."""

    outcome: Literal["completed", "escalated"]
    root_cause: str | None = None
    must_call: list[str] = Field(default_factory=list)
    must_not_call: list[str] = Field(default_factory=list)
    # Each pair [A, B] means "A's call must happen before B's call" --
    # checked by the trajectory grader (3.4) against the tool_calls list.
    order: list[tuple[str, str]] = Field(default_factory=list)
    # Interpreted against the run's sandbox_snapshot by the trajectory
    # grader (3.4); keys are a grader-defined dotted path (e.g.
    # "checkout.config_version"), not a literal snapshot key.
    final_state: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False


class Budgets(BaseModel):
    max_steps: int
    max_tokens: int
    max_cost_usd: float
    max_latency_s: float


class EvalCase(BaseModel):
    """One row of the golden dataset -- see docs/specs/day3.md 3.1 for the
    YAML shape this mirrors field-for-field."""

    id: str
    scenario: str
    seed: int
    role: Role
    approval_policy: ApprovalPolicy = "approve_all"
    tags: list[str] = Field(default_factory=list)
    expect: Expect
    budgets: Budgets
    rubric_notes: str = ""

    @field_validator("scenario")
    @classmethod
    def _scenario_must_exist(cls, value: str) -> str:
        # Fail loudly at load time, not at run time three steps into a
        # trial -- a typo'd scenario name in a YAML file is a dataset bug,
        # not something a grader should have to discover.
        try:
            get_scenario(value)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        return value
