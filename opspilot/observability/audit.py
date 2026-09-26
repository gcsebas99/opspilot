import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from opspilot.loops.react_raw import LoopEvent
from opspilot.store.base import Store
from opspilot.store.models import AuditDoc


def _truncate_to_milliseconds(ts: datetime) -> datetime:
    return ts.replace(microsecond=(ts.microsecond // 1000) * 1000)


def _compute_hash(entry: AuditDoc) -> str:
    """sha256 over every field except `hash` itself -- `prev_hash` is
    included, which is what actually chains records together: changing
    ts/actor/action/target/decision/run_id/prompt_version/model on THIS
    record changes its own hash, and changing prev_hash without also
    updating every later record's hash breaks the chain from that point
    on. json.dumps(sort_keys=True) makes the serialization deterministic
    regardless of pydantic's field-declaration order.

    `ts` is truncated to millisecond precision before hashing -- BSON only
    has millisecond precision, so MongoStore silently drops microseconds on
    insert. Hashing the untruncated value computed by record_audit() (before
    insert) would never match the value verify_chain() recomputes from what
    list_audit() reads back (after Mongo's round-trip), making every entry
    look tampered even when nothing touched it. Caught live against real
    Mongo -- MemoryStore never round-trips through BSON, so it never lost
    the precision that made this mismatch visible.
    """
    payload = entry.model_copy(update={"ts": _truncate_to_milliseconds(entry.ts)}).model_dump(
        exclude={"hash"}, mode="json"
    )
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# [HARNESS:AUDIT] Tamper-evident hash chain, not just an append-only table.
# WHY: append-only stops accidental loss, not deliberate tampering -- anyone
# with write access to Mongo can edit a historical document in place, and a
# plain append-only collection has no way to detect that after the fact.
# Chaining each record's hash into the next means editing (or deleting, or
# reordering) any single record breaks every hash computed from it forward
# -- verify_chain() below recomputes the whole chain and reports exactly
# where it diverges from what's stored.
# INTERVIEW: "How would you prove nobody tampered with the audit log?" ->
# recompute each record's hash from its own fields plus the previous
# record's hash and compare to what's stored; the first mismatch is the
# tampered (or truncated/reordered) record -- not just "some record is
# wrong somewhere."
async def record_audit(
    store: Store,
    *,
    actor: str,
    action: str,
    target: str,
    decision: str,
    run_id: str,
    prompt_version: str,
    model: str,
    ts: datetime,
) -> AuditDoc:
    previous = await store.get_last_audit()
    entry = AuditDoc(
        ts=ts,
        actor=actor,
        action=action,
        target=target,
        decision=decision,
        run_id=run_id,
        prompt_version=prompt_version,
        model=model,
        prev_hash=previous.hash if previous is not None else None,
    )
    entry = entry.model_copy(update={"hash": _compute_hash(entry)})
    await store.insert_audit(entry)
    return entry


@dataclass(frozen=True)
class ChainVerification:
    ok: bool
    total_entries: int
    broken_at_index: int | None = None
    reason: str | None = None


def verify_chain(entries: Sequence[AuditDoc]) -> ChainVerification:
    """`entries` must be in chronological (insertion) order -- exactly what
    `store.list_audit()` already returns."""
    expected_prev: str | None = None
    for index, entry in enumerate(entries):
        if entry.prev_hash != expected_prev:
            return ChainVerification(
                ok=False,
                total_entries=len(entries),
                broken_at_index=index,
                reason="prev_hash does not match the previous record's actual hash",
            )
        recomputed = _compute_hash(entry.model_copy(update={"hash": None}))
        if entry.hash != recomputed:
            return ChainVerification(
                ok=False,
                total_entries=len(entries),
                broken_at_index=index,
                reason="stored hash does not match recomputed hash -- record was modified",
            )
        expected_prev = entry.hash
    return ChainVerification(ok=True, total_entries=len(entries))


# [HARNESS:AUDIT] Reconstructing audit entries from events, same reasoning
# as instrumentation.py's record_loop_spans: the raw loop has no live store
# access (its `on_event` callback is sync), so this is a second, post-hoc
# consumer of the same event stream, called once the run completes.
# RequireApproval is treated identically to Deny here -- the raw loop
# permanently blocks both (see react_raw.py's [HARNESS:PERM] note), so from
# an audit standpoint a blocked call is a blocked call regardless of which
# policy outcome produced it.
async def record_loop_audit(
    store: Store,
    run_start: datetime,
    events: Sequence[LoopEvent],
    *,
    run_id: str,
    role: str,
    prompt_version: str,
    model: str,
) -> None:
    cursor = run_start
    for event in events:
        if event.type != "tool_call":
            continue
        duration_ms = event.data["duration_ms"]
        policy_decision = event.data.get("policy_decision")
        name = event.data["name"]
        if policy_decision in ("deny", "require_approval"):
            await record_audit(
                store,
                actor=role,
                action="permission_denied",
                target=name,
                decision=event.data.get("denial_reason") or "denied",
                run_id=run_id,
                prompt_version=prompt_version,
                model=model,
                ts=cursor,
            )
        elif event.data.get("risk") == "destructive" and policy_decision == "allow":
            outcome = "executed" if event.data["ok"] else "execution failed"
            await record_audit(
                store,
                actor=role,
                action="destructive_tool_executed",
                target=name,
                decision=outcome,
                run_id=run_id,
                prompt_version=prompt_version,
                model=model,
                ts=cursor,
            )
        cursor += timedelta(milliseconds=duration_ms)


# [HARNESS:AUDIT] Config-change tracking. WHY: AGENTS.md, the runbook
# index, or a tool schema changing between two runs is exactly the kind of
# silent behavior shift an incident postmortem needs to rule in or out --
# "did the agent's instructions change right before it started doing this?"
# prompt_version (2.1's hash of exactly that content) makes the comparison
# a string equality check instead of a diff nobody remembers to run.
async def record_prompt_version_change_if_needed(
    store: Store, *, run_id: str, prompt_version: str, model: str, ts: datetime
) -> None:
    previous_runs = await store.list_runs()  # newest first; called before this run's insert_run
    if not previous_runs or previous_runs[0].prompt_version == prompt_version:
        return
    await record_audit(
        store,
        actor="system",
        action="prompt_version_changed",
        target="prompt",
        decision=f"{previous_runs[0].prompt_version} -> {prompt_version}",
        run_id=run_id,
        prompt_version=prompt_version,
        model=model,
        ts=ts,
    )
