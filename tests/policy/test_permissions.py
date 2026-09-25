import pytest
from pydantic import BaseModel

from opspilot.env.sandbox import Sandbox
from opspilot.policy.permissions import Allow, Deny, RequireApproval, Role, decide
from opspilot.tools.base import Risk, Tool, ToolResult


class _EmptyInput(BaseModel):
    pass


def _noop(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    return ToolResult(ok=True, content="")


def _tool(risk: Risk, name: str = "some_tool") -> Tool:
    return Tool(name=name, description="test tool", input_model=_EmptyInput, risk=risk, fn=_noop)


_ALL_ROLES: list[Role] = ["viewer", "operator", "admin", "system"]
_ALL_RISKS: list[Risk] = ["read", "destructive", "terminal"]

# The full permission matrix, per docs/PLAN.md's role table. "100% tested"
# means every (role, risk) cell has an explicit expected decision here --
# test_full_matrix_is_covered guards against silently adding a role or
# risk without adding a corresponding entry.
_EXPECTED: dict[tuple[Role, Risk], type] = {
    ("viewer", "read"): Allow,
    ("viewer", "destructive"): Deny,
    ("viewer", "terminal"): Allow,
    ("operator", "read"): Allow,
    ("operator", "destructive"): RequireApproval,
    ("operator", "terminal"): Allow,
    ("admin", "read"): Allow,
    ("admin", "destructive"): Allow,
    ("admin", "terminal"): Allow,
    ("system", "read"): Allow,
    ("system", "destructive"): RequireApproval,
    ("system", "terminal"): Allow,
}


def test_full_matrix_is_covered() -> None:
    assert set(_EXPECTED) == {(role, risk) for role in _ALL_ROLES for risk in _ALL_RISKS}


@pytest.mark.parametrize(("role", "risk"), sorted(_EXPECTED))
def test_decide_matches_plan_matrix(role: Role, risk: Risk) -> None:
    decision = decide(role, _tool(risk))

    assert type(decision) is _EXPECTED[(role, risk)]


def test_deny_reason_names_role_and_tool() -> None:
    decision = decide("viewer", _tool("destructive", name="restart_service"))

    assert isinstance(decision, Deny)
    assert "viewer" in decision.reason
    assert "restart_service" in decision.reason


def test_require_approval_reason_names_tool() -> None:
    decision = decide("operator", _tool("destructive", name="rollback_config"))

    assert isinstance(decision, RequireApproval)
    assert "rollback_config" in decision.reason


def test_allow_carries_no_reason() -> None:
    decision = decide("admin", _tool("destructive"))

    assert decision == Allow()


def test_out_of_scope_destructive_downgrades_admin_allow_to_require_approval() -> None:
    decision = decide("admin", _tool("destructive", name="restart_service"), in_scope=False)

    assert isinstance(decision, RequireApproval)
    assert "restart_service" in decision.reason


def test_in_scope_destructive_stays_allowed_for_admin() -> None:
    decision = decide("admin", _tool("destructive"), in_scope=True)

    assert decision == Allow()


@pytest.mark.parametrize(("role", "risk"), sorted(_EXPECTED))
def test_out_of_scope_never_loosens_the_base_matrix(role: Role, risk: Risk) -> None:
    # Scope check only ever makes a decision stricter -- never looser than
    # what the base role x risk matrix already grants.
    in_scope_decision = decide(role, _tool(risk), in_scope=True)
    out_of_scope_decision = decide(role, _tool(risk), in_scope=False)

    if type(in_scope_decision) is Allow:
        assert type(out_of_scope_decision) is not Deny
    else:
        assert type(out_of_scope_decision) is type(in_scope_decision)
