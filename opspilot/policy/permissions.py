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


# [HARNESS:PERM] Table-driven, pure function -- the whole policy in one
# place, testable without a model, a sandbox, or a tool call.
# WHY: permissions live in the harness (this table), not the system prompt,
# because a prompt is advisory -- a sufficiently confused or adversarially
# steered model can be talked into ignoring instructions (see the
# prompt_injection scenario, where a log line tries exactly this), but it
# cannot talk its way past a Python dict lookup it never sees and never
# controls. AGENTS.md tells the model what a good agent does; this table
# enforces what's actually possible for it to do.
# INTERVIEW: "Why not just tell the model 'viewers can't restart services'
# in the system prompt?" -> because that's a request, not a constraint --
# the model is a probabilistic text generator, not a security boundary.
# Enforcement has to live in code the model's output can't reach.
def decide(role: Role, tool: Tool) -> Decision:
    outcome = _MATRIX[role][tool.risk]
    if outcome == "allow":
        return Allow()
    if outcome == "deny":
        return Deny(f"{role} role cannot run {tool.name!r}; recommend it in your report instead.")
    return RequireApproval(f"{tool.name!r} requires human approval before it can run.")
