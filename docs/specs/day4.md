# Day 4 — Web UI, outer loops, deploy, compaction, long-term memory

**Goal:** a public Render URL where anyone can trigger an incident, watch the trace stream in, and approve
or reject destructive actions — plus context compaction and long-term incident memory, both evaluated.

**Pillars:** MEMORY · CONTEXT · outer loops · deployment

Order matters: do 4.1–4.4 first (deployable demo). 4.5–4.6 are cut first if short on time.

## Kickoff prompt

> Read `CLAUDE.md`, `docs/PLAN.md` and `docs/specs/day4.md`. Days 1–3 are done.
> Plan sub-task 4.1 only. Keep the UI server-rendered (FastAPI + Jinja2 + HTMX), no JS build step.

---

## 4.1 FastAPI app `[HARNESS:HITL]` `[HARNESS:OBS]`

`opspilot/web/`:
- `POST /runs` (scenario, seed, role, strategy) → creates run, starts it as a background task, redirects to run page.
- `GET /runs` list; `GET /runs/{id}` run page: header (outcome, cost, tokens) + **trace waterfall** partial
  refreshed by HTMX polling (`hx-trigger="every 2s"` until finished).
- `GET /approvals` pending list; `POST /approvals/{id}` approve/reject/edit → resumes the graph from the
  **Mongo checkpointer** (this is why Day 2 made resume work across processes).
- `GET /metrics` dashboard (Day 2 aggregations), `GET /evals` latest eval report, `GET /healthz`.
- "Act as" role selector (demo auth — clearly labeled). Approver identity recorded in audit log.
- Server-side guards for the public demo: per-IP rate limit, **global daily run cap**, hard token budget per run
  `[HARNESS:GUARD]` — your API key is behind this.

**Accept:** local end-to-end: start run as operator → pause → approve in browser → run completes; trace visible.

## 4.2 Outer loops `[HARNESS:LOOP]`

- **Event/hook:** `POST /webhooks/alert` — verifies **HMAC signature** (`X-Signature`, shared secret),
  **idempotency key** (same alert twice ⇒ one run), maps alert → scenario, runs as `system` role.
- **Cron/heartbeat:** `.github/workflows/canary.yml` scheduled daily: sends a signed `false_alarm`
  alert to the deployed webhook and asserts the run ends `completed` with `no_incident` — a production
  canary that doubles as an online eval.
- Document in README: goal-driven (the inner loop), event (webhook), time-based (cron), heartbeat (canary)
  — and where Ralph-style brute-force retry loops would fit (outer "retry until evals pass" loop — explain, don't build).

## 4.3 Deploy to Render

- `Dockerfile` (slim, uv, non-root user), `render.yaml` blueprint (web service, free plan, env vars:
  `ANTHROPIC_API_KEY`, `MONGODB_URI`, `OPSPILOT_MODEL`, `WEBHOOK_SECRET`, `DAILY_RUN_CAP`).
- Atlas: free cluster, DB user, network access (Render free has no static IPs → allowlist 0.0.0.0/0,
  mitigated by strong credentials — **write this tradeoff in the README**).
- Sandbox on Render = temp dir + path-guarded tools (no Docker-in-Docker). Note it in README.
- Cold start on free tier is expected; the UI shows a friendly "waking up" note.

**Accept:** public URL works end-to-end; canary workflow passes against it.

## 4.4 README + concept index

- README: pitch (3 lines), architecture mermaid (from PLAN.md), **concept → file map table**,
  screenshots/GIF of the approval flow, eval results table + ablation highlights, how to run locally,
  tradeoffs & what I'd do next.
- `scripts/concepts.py`: greps `[HARNESS:*]` tags → generates `CONCEPTS.md` grouped by pillar with
  file:line links. Run it in CI and fail if it's stale.

## 4.5 Context compaction `[HARNESS:CONTEXT]` (cut 2nd)

- `opspilot/context/compaction.py`: strategy interface `HistoryStrategy` with `FullReplay` and `Compaction`.
- Compaction triggers when estimated context tokens > threshold (e.g. 70% of a configured window, set low
  for demo): keep system + original alert + last N turns; summarize the middle with the model into a
  structured "investigation so far" note (hypotheses, evidence, ruled-out causes, actions taken).
- **Invariant:** never split a `tool_use` from its `tool_result`. Test it.
- `compaction` span with before/after token counts. Ablation: compaction on vs off (accuracy, tokens).

## 4.6 Long-term memory `[HARNESS:MEMORY]` (cut 1st)

- On a run that completes **and** was approved/verified: write to `memories` — symptoms (alert + key log
  signatures), root cause, fix, outcome, run_id. Only verified outcomes are stored (memory hygiene —
  don't learn from your mistakes as if they were truths).
- Retrieval at context assembly: Mongo **text index** search on symptoms → top 3 → injected as
  "Similar past incidents (may not apply)". Traced as a span.
- Short-term memory = LangGraph checkpointed state; long-term = this collection. Document the difference.
- Ablation: memory on vs off on a second pass over `golden` (steps/cost should drop; accuracy must not).
- Mention (don't build): vector search (Atlas Vector Search + embeddings) as the next step, and memory
  poisoning risks.

## Interview questions

1. How does your UI resume a paused agent after a human approves? What if the server restarted in between?
2. How do you secure a webhook that triggers an agent? What about duplicate alerts?
3. Explain inner vs outer loops using your project.
4. Full replay vs compaction: tradeoffs? What can go wrong with compaction?
5. Short-term vs long-term memory in your system. How do you prevent bad memories?
6. What protects your API bill on a public demo?
7. What would you change to run this for a real company? (auth, multi-tenant sandboxes, vector memory,
   OpenTelemetry export, eval dataset from prod traces)
