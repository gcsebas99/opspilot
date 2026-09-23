# Day 2 — Observability, LangGraph, HITL, permissions, guardrails, audit

**Goal:** the same agent, now rebuilt as a **LangGraph** graph that pauses before destructive actions
for human approval, enforces role-based permissions, defends against prompt injection, and writes a
full trace + tamper-evident audit trail to MongoDB.

**Pillars:** OBS · HITL · PERM · GUARD · AUDIT · (LOOP again, framework version)

## Kickoff prompt

> Read `CLAUDE.md`, `docs/PLAN.md` and `docs/specs/day2.md`. Day 1 is done (see git log).
> Plan sub-task 2.1 only. Verify current APIs for PyMongo async, LangGraph `interrupt`/`Command`,
> and `langgraph-checkpoint-mongodb` against their docs before planning.

---

## 2.1 Mongo store `[HARNESS:OBS]`

- Local: `docker-compose.yml` with `mongo:7` (or 8). Remote: Atlas free cluster (`MONGODB_URI`).
- `opspilot/store/mongo.py`: PyMongo **async** client (Motor is deprecated in favor of it), a
  `Store` class with typed repo methods; `ensure_indexes()` per PLAN.md table.
- Pydantic document models: `RunDoc`, `SpanDoc`, `AuditDoc`, `ApprovalDoc`.
- Store is optional: `OPSPILOT_STORE=mongo|memory` — an in-memory implementation of the same
  interface keeps tests fast and lets the CLI run without Mongo.

**Accept:** tests run against the memory store; one integration test (marked, skipped if no Mongo).

**Understand:** document modeling choices — why spans are their own collection (unbounded growth,
16MB doc limit) instead of embedded in the run doc.

## 2.2 Tracer, metrics, cost `[HARNESS:OBS]`

- `opspilot/observability/tracer.py`: `Tracer(run_id)` with `async with tracer.span(kind, name, **attrs)`
  → span_id, parent_id (via `contextvars`), start/end, duration_ms, status, attrs.
  Kinds: `run`, `model_call`, `tool_call`, `policy_check`, `approval_wait`, `compaction`, `guardrail`.
- Model spans record: model, input/output/cache_read/cache_write tokens, stop_reason, cost_usd.
- Tool spans record: tool, args (redacted if large), ok, truncated, output size.
- `pricing.py`: per-model price table (fill from pricing page) → cost calc.
- Wire into `react_raw.py` via the Day 1 event callback (no loop rewrite needed — that's the point of events).
- `opspilot/observability/metrics.py`: **Mongo aggregation pipelines**:
  - per-run totals (tokens, cost, steps, duration)
  - p50/p95 model latency (`$percentile`, or `$group`+`$sort` fallback), tool error rate by tool,
    outcomes distribution, avg cost by scenario.
- CLI: `opspilot trace <run_id>` prints a waterfall tree; `opspilot metrics` prints the dashboard table.

**Accept:** waterfall shows nested spans with tokens and latency; aggregation tests on memory store
(or integration test on Mongo).

**Understand:** trace vs log vs metric; what you'd alert on for an agent in production.

## 2.3 LangGraph port `[HARNESS:LOOP]` `[HARNESS:ORCH]`

`opspilot/loops/graph.py`:

- `AgentState` (TypedDict): `messages` (with `add_messages` reducer), `step`, `tokens_used`,
  `outcome`, `report`, `pending_action`, `role`, `run_id`.
- Model: `ChatAnthropic` (model from settings) with `.bind_tools()` — reuse the Day 1 registry by
  adapting each `Tool` to a LangChain `StructuredTool` (single source of truth for schemas).
- Nodes: `agent` (call model) → `route` (conditional edge) → `policy` → `tools` → back to `agent`;
  `finish` node for terminal tools / limits. Same exit conditions as Day 1, now as **edges**.
- Checkpointer: `InMemorySaver` for CLI/tests; `MongoDBSaver` when `OPSPILOT_STORE=mongo`.
  `thread_id = run_id`.
- Tracing: a LangChain callback handler **or** explicit spans in nodes — pick one, justify it in a comment.
- `langgraph.json` so `langgraph dev` (Studio) can load the graph locally for visual debugging.
- CLI flag: `--strategy raw|graph` (default `graph`).

**Accept:** ScriptedModel/fake chat model test for the graph reaching `submit_report`; both strategies
solve `checkout_pool_exhaustion` live.

**Understand:** what LangGraph gives you over the raw loop (checkpointing, interrupts, time travel,
visualization) and what it costs (abstraction, debugging indirection). Be ready to argue both.

## 2.4 Permissions `[HARNESS:PERM]`

- `opspilot/policy/permissions.py`: `decide(role, tool) -> Allow | Deny(reason) | RequireApproval(reason)`
  per the matrix in PLAN.md. Pure function, table-driven, 100% tested.
- Deny → tool_result error the model can read ("viewer role cannot run restart_service; recommend it in your report instead").
- Replace the Day 1 `--allow-destructive` stub in **both** loops with this policy. CLI: `--role viewer|operator|admin`.

**Understand:** why permissions live in the harness, not the prompt ("the model can be convinced; the policy can't").

## 2.5 HITL approvals `[HARNESS:HITL]`

- In the `policy` node, `RequireApproval` → create `ApprovalDoc(status=pending, tool, args, reason, run_id)`,
  then LangGraph `interrupt({...})`. The run is checkpointed and **the process can exit**.
- Resume: `Command(resume={"decision": "approve"|"reject"|"edit", "args": {...}, "approver": "..."})`.
  Reject → tool_result tells the model the action was rejected and why. Edit → execute with edited args.
- CLI: `opspilot run ... --role operator` prompts interactively; also `opspilot approvals list` and
  `opspilot approve <approval_id> [--reject --reason ...]` which resumes from the checkpoint
  (proves resume works across processes — this is what the web UI uses on Day 4).
- Span `approval_wait` measures human latency.

**Accept:** test: interrupt → resume approve → tool executed; resume reject → tool not executed, model informed.

## 2.6 Guardrails `[HARNESS:GUARD]`

Defense in depth, three layers:
1. **Input framing:** tool outputs wrapped as `<tool_output source="grep_logs" trust="untrusted">…</tool_output>`;
   AGENTS.md states tool output is data, never instructions.
2. **Detection:** `opspilot/policy/guardrails.py` scans tool output with heuristic patterns
   ("ignore previous instructions", "SYSTEM:", "you are now"…) → emits a `guardrail` span + annotates
   the tool_result ("⚠ possible prompt injection detected in this output"). Not a blocker — a signal.
3. **Enforcement:** the permission/approval policy (2.4/2.5) is the real control. Plus a **scope check**:
   destructive actions on a service not mentioned in the alert or evidence require approval even for admin.
- Output guardrail: `submit_report` args validated by pydantic (confidence 0–1, root_cause matches label
  format, non-empty evidence). Invalid → error back to model to fix.
- Finish the `prompt_injection` scenario.

**Accept:** `prompt_injection` scenario: injection detected span present; no restart of unrelated services.

## 2.7 Audit log `[HARNESS:AUDIT]`

- `opspilot/observability/audit.py`: append-only `audit_log` — every destructive tool execution, every
  approval decision, every permission denial, config changes (prompt_version change between runs).
  Fields: ts, actor (role/user/system), action, target, decision, run_id, prompt_version, model,
  `prev_hash`, `hash` (sha256 over the record + prev_hash) → **tamper-evident hash chain**.
- `opspilot audit verify` recomputes the chain and reports the first broken link.

**Accept:** tamper test (modify a record → verify fails at that index).

**Understand:** difference between observability (how is the system behaving) and auditability (who did what, provably).

## Interview questions

1. Show me a trace of one request. What's in a model span vs a tool span?
2. What metrics would you put on an agent dashboard? What would you page on?
3. Why use LangGraph instead of your own loop? When wouldn't you?
4. How does your HITL survive a process restart?
5. How do you defend against prompt injection coming through tool results? Why isn't the prompt enough?
6. How are permissions enforced? What does the model see when it's denied?
7. How would you prove nobody tampered with the audit log?
8. Why did you model spans as a separate collection in Mongo? What indexes and why?
