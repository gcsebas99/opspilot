from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunDoc(BaseModel):
    """One document per agent run."""

    run_id: str
    created_at: datetime
    scenario: str
    seed: int
    role: str
    model: str
    prompt_version: str
    strategy: Literal["raw", "graph"]
    outcome: str | None = None
    steps: int | None = None
    tokens: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    finished_at: datetime | None = None


class SpanDoc(BaseModel):
    """One trace span.

    Its own collection, not embedded in RunDoc: a single run can produce far
    more spans than fit under Mongo's 16MB document limit once tool/model
    calls accumulate across many steps, and spans grow independently of the
    run document's own (small, fixed-shape) fields.
    """

    span_id: str
    run_id: str
    parent_id: str | None = None
    kind: Literal[
        "run", "model_call", "tool_call", "policy_check", "approval_wait", "compaction", "guardrail"
    ]
    name: str
    start: datetime
    end: datetime | None = None
    duration_ms: float | None = None
    status: Literal["ok", "error"] = "ok"
    attrs: dict[str, Any] = Field(default_factory=dict)


class AuditDoc(BaseModel):
    """Append-only, hash-chained record. The hash-chaining logic itself
    lands in 2.7 -- prev_hash/hash exist on the model now so the schema
    doesn't need a migration when that logic arrives.
    """

    ts: datetime
    actor: str
    action: str
    target: str
    decision: str
    run_id: str
    prompt_version: str
    model: str
    prev_hash: str | None = None
    hash: str | None = None


class ApprovalDoc(BaseModel):
    """A pending/decided HITL approval request (HITL flow itself lands in 2.5)."""

    approval_id: str
    run_id: str
    tool: str
    args: dict[str, Any]
    reason: str
    status: Literal["pending", "approved", "rejected"] = "pending"
    requested_at: datetime
    decided_at: datetime | None = None
    approver: str | None = None
