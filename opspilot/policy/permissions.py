from dataclasses import dataclass
from typing import Literal

from opspilot.tools.base import Risk, Tool

Role = Literal["viewer", "operator", "admin", "system"]

_Outcome = Literal["allow", "deny", "require_approval"]


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class Deny:
    reason: str


@dataclass(frozen=True)
class RequireApproval:
    reason: str


Decision = Allow | Deny | RequireApproval

# Role x risk -> outcome, per docs/PLAN.md's permission matrix.
_MATRIX: dict[Role, dict[Risk, _Outcome]] = {
    "viewer": {"read": "allow", "destructive": "deny", "terminal": "allow"},
    "operator": {"read": "allow", "destructive": "require_approval", "terminal": "allow"},
    "admin": {"read": "allow", "destructive": "allow", "terminal": "allow"},
    "system": {"read": "allow", "destructive": "require_approval", "terminal": "allow"},
}


# [HARNESS:PERM] Table-driven policy: role x tool risk -> allow / deny / approve.
# WHY: a prompt is advisory (a steered model can ignore it); a dict lookup the
# model never sees is not. AGENTS.md says what a good agent does; this table
# enforces what it *can* do. Pure function, testable without a model.
def decide(role: Role, tool: Tool, *, in_scope: bool = True) -> Decision:
    outcome = _MATRIX[role][tool.risk]

    # [HARNESS:GUARD] Scope check: out-of-scope destructive actions need approval.
    # WHY: injection *detection* can miss a new phrasing; this can't be talked past,
    # because it only sees which service is targeted. It only ever tightens
    # (allow -> require_approval), even for admin.
    if tool.risk == "destructive" and not in_scope and outcome == "allow":
        return RequireApproval(
            f"{tool.name!r} targets a service not named in the alert or investigated yet "
            "this run; out-of-scope destructive actions require approval even for admin."
        )

    if outcome == "allow":
        return Allow()
    if outcome == "deny":
        return Deny(f"{role} role cannot run {tool.name!r}; recommend it in your report instead.")
    return RequireApproval(f"{tool.name!r} requires human approval before it can run.")


# Which *human* roles may decide a pending approval. Separate from _MATRIX on
# purpose: that table governs what the *agent* may do; this one governs what
# the person clicking "approve" may do.
_CAN_DECIDE_APPROVAL: dict[Role, bool] = {
    "viewer": False,
    "operator": True,
    "admin": True,
    "system": False,  # automation never approves its own actions
}


# [HARNESS:PERM] Approving is itself a permission, enforced server-side.
# WHY: hiding the button is UX, not security -- anyone can POST to
# /approvals/{id}. The server checks this table before recording any decision.
def can_decide_approval(role: Role) -> bool:
    return _CAN_DECIDE_APPROVAL[role]
