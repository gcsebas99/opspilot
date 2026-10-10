from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from opspilot.env.scenarios import get_scenario
from opspilot.policy.permissions import Role

# [HARNESS:EVAL] approval_policy: the case decides approvals, not a live human.
# WHY: k-trial sweeps must never block on input. approve_all / reject_all cover
# most cases; a per-tool dict scripts finer ones (approve rollback, reject restart).
ApprovalPolicy = Literal["approve_all", "reject_all"] | dict[str, Literal["approve", "reject"]]


class ToolMatcher(BaseModel):
    """Matches a tool call by name and (optionally) a subset of its args.

    YAML accepts a bare tool name (`restart_service`) or the full form
    (`{tool: restart_service, args: {service: web}}`) -- the bare form is
    normalized into this model at load time.
    """

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)

    def matches(self, call: dict[str, Any]) -> bool:
        call_args = call.get("input") or {}
        return call.get("name") == self.tool and all(
            call_args.get(k) == v for k, v in self.args.items()
        )

    def __str__(self) -> str:
        if not self.args:
            return self.tool
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.tool}({rendered})"


def _to_matchers(value: Any) -> Any:
    if isinstance(value, list):
        return [{"tool": v} if isinstance(v, str) else v for v in value]
    return value


class Expect(BaseModel):
    """What a passing trial must look like. Every field is optional except
    `outcome` -- an empty list/None means "no constraint", not "must be
    empty" (an empty `must_call`, for instance, says nothing about which
    tools were or weren't called)."""

    outcome: Literal["completed", "escalated"]
    root_cause: str | None = None
    must_call: list[ToolMatcher] = Field(default_factory=list)
    # Counts *attempts*, including calls the policy then blocked: for an
    # injection case, trying to restart an unrelated service is the failure
    # even though the harness stopped it. "Recommend, don't execute" is
    # expressed as unchanged final_state instead -- the agent isn't told its
    # role, so attempt -> denied -> recommend is the intended viewer flow.
    must_not_call: list[ToolMatcher] = Field(default_factory=list)
    # Each pair [A, B] means "A's call must happen before B's call" --
    # checked by the trajectory grader (3.4) against the tool_calls list.
    order: list[tuple[str, str]] = Field(default_factory=list)
    # Interpreted against the run's sandbox_snapshot by the trajectory
    # grader (3.4); keys are a grader-defined dotted path (e.g.
    # "checkout.config_version"), not a literal snapshot key.
    # Keys are Sandbox.facts() keys; a `null` value means "fact absent"
    # (e.g. `inventory.restarted: null` -- inventory was never restarted).
    final_state: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False

    @field_validator("must_call", "must_not_call", mode="before")
    @classmethod
    def _normalize_matchers(cls, value: Any) -> Any:
        return _to_matchers(value)


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
