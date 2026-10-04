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
    # Stored so `opspilot approve` resumes a paused run with the same
    # backend it started with (a replayed demo must not resume live), and so
    # a replayed run's cost_usd (recorded cost, $0 actual) can be told apart.
    mode: Literal["live", "record", "replay"] = "live"
    cassette: str | None = None
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


class EvalTrialDoc(BaseModel):
    """One (case x trial) execution from an `opspilot eval` sweep.

    Deliberately self-contained -- everything the graders (3.4) need
    (tool_calls, report, sandbox_snapshot, outcome) is denormalized onto
    this document rather than requiring a join against `spans`, so grading
    a trial is a single read. `run_id` still links back to a normal RunDoc
    and its spans for anyone who wants the full trace (`opspilot trace`
    works on an eval trial's run_id exactly like a manual run's).
    """

    trial_id: str
    sweep_id: str
    suite: str
    case_id: str
    trial: int
    run_id: str
    git_sha: str
    prompt_version: str
    model: str
    strategy: Literal["raw", "graph"]
    # In replay, cost_usd is what the recorded calls cost when captured (the
    # actual spend is $0) and latency_s is near-zero wall-clock -- graders
    # and reports need this to avoid reading replay numbers as live ones.
    mode: Literal["live", "record", "replay"] = "live"
    started_at: datetime
    outcome: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    report: dict[str, Any] | None = None
    # File hashes -- detects *that* state changed, but includes wall-clock
    # values (restart_service's restarted_at), so it is NOT stable across
    # runs. Compare sandbox_facts for "same end state".
    sandbox_snapshot: dict[str, str] = Field(default_factory=dict)
    # Everything below (through latency_s) exists so graders (3.4) are pure
    # functions of (case, trial) -- no sandbox or store access needed, so a
    # stored sweep can be re-graded later, even from an in-memory store run.
    sandbox_facts: dict[str, Any] = Field(default_factory=dict)
    # Policy-relevant audit entries for this run (approval decisions,
    # permission denials, executed destructive tools), as {action, target,
    # decision} -- the system's own record, not the runner's bookkeeping.
    policy_events: list[dict[str, str]] = Field(default_factory=list)
    steps: int | None = None
    tokens: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    latency_s: float | None = None
    # Populated by the graders (3.4) in a separate pass -- left empty here
    # on purpose; 3.2 only runs trials and persists raw results.
    grades: dict[str, Any] | None = None
    # The case pass rule's verdict (evals/grading.py) -- None until graded.
    passed: bool | None = None
    # Separate from `error` (the agent run crashed): grading can fail on its
    # own (e.g. judge CassetteMiss) and must stay re-gradable without a re-run.
    grading_error: str | None = None
    error: str | None = None
