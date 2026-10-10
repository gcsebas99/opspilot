# Day 4 — Web UI, outer loops, deploy, compaction, long-term memory

**Goal:** a public Render URL where anyone can trigger an incident, watch the trace stream in, and approve
or reject destructive actions — plus context compaction and long-term incident memory, both evaluated.

## Demo mode: replay by default (decided 2026-10-04)

The public demo makes **zero Anthropic API calls**. It reuses Day 3's record/replay:

- **Model responses are replayed** from recorded demo cassettes (`evals/cassettes/demo/`, see
  `opspilot run --mode record`). **Everything else runs for real**: tools against a real sandbox,
  permission policy, guardrails, HITL pause/resume, trace, audit log.
- **Banner on every page:** "Model responses are recordings of real Claude runs; tools, permissions,
  approvals and traces execute live. Clone the repo and set `ANTHROPIC_API_KEY` to run against the live model."
- **Only recorded paths are offered.** The UI lists scenario × role × seed combinations from the demo
  cassettes that exist. Approve *and* reject branches are both recorded (one cassette, shared prefix).
  "Approve with edits" produces unrecorded arguments → disabled in replay, or a cassette miss shows a
  friendly "this path wasn't recorded — clone to run it live" instead of an error.
- **Live mode is owner-only and off by default** (`OPSPILOT_DEMO_LIVE=false`). The cost guards below
  (rate limit, daily cap, token budget) protect that optional live path. The public page's strongest
  protection is simply never holding a usable key in replay deployments.
- **Not offered:** "bring your own key" on the site — handling visitors' secrets server-side is a
  security/privacy burden a demo shouldn't carry. Clone-and-run gives the same thing safely.
- Demo cassettes are recorded **late** (after 4.5 compaction, which changes prompts), haiku agent,
  no judge: ~$0.04 per recorded path. Ask before recording.

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
- Model mode per the demo decision above: replay by default, choices limited to recorded paths, friendly
  cassette-miss page, banner.
- Server-side guards for the **optional live mode**: per-IP rate limit, **global daily run cap**, hard token
  budget per run `[HARNESS:GUARD]` — your API key is behind this when `OPSPILOT_DEMO_LIVE=true`.

**Accept:** local end-to-end **in replay ($0)**: start run as operator → pause → approve in browser → run
completes; trace visible.

## 4.2 Outer loops `[HARNESS:LOOP]`

- **Event/hook:** `POST /webhooks/alert` — verifies **HMAC signature** (`X-Signature`, shared secret),
  **idempotency key** (same alert twice ⇒ one run), maps alert → scenario, runs as `system` role.
- **Cron/heartbeat:** `.github/workflows/canary.yml` scheduled **weekly** (decided 2026-10-05: a demo showcase, not a 24/7 service): sends a signed `false_alarm`
  alert to the deployed webhook and asserts the run ends `completed` with `no_incident` — a production
  canary that doubles as an online eval. Runs against the **replay** deployment, so it verifies the whole
  path (signature, idempotency, run, outcome) for $0.
- Document in README: goal-driven (the inner loop), event (webhook), time-based (cron), heartbeat (canary)
  — and where Ralph-style brute-force retry loops would fit (outer "retry until evals pass" loop — explain, don't build).

## 4.3 Deploy to Render

- `Dockerfile` (slim, uv, non-root user), `render.yaml` blueprint (web service, free plan, env vars:
  `OPSPILOT_MODEL_MODE=replay`, `OPSPILOT_DEMO_LIVE=false`, `MONGODB_URI`, `OPSPILOT_MODEL`,
  `OPSPILOT_WEBHOOK_SECRET` (Render-generated), `OPSPILOT_DAILY_RUN_CAP` (live mode only); `ANTHROPIC_API_KEY` left unset unless live mode is deliberately enabled).
- Record the demo cassettes (see "Demo mode") before deploying — the only paid step of Day 4's demo.
- Atlas: free cluster, DB user, network access (Render free has no static IPs → allowlist 0.0.0.0/0,
  mitigated by strong credentials — **write this tradeoff in the README**).
- Sandbox on Render = temp dir + path-guarded tools (no Docker-in-Docker). Note it in README.
- Cold start on free tier is expected; the UI shows a friendly "waking up" note.

**Accept:** public URL works end-to-end; canary workflow passes against it.

## 4.4 Learning guide + README + concept index (expanded 2026-10-10)

The project has two jobs: teach harness engineering to its author, and demonstrate it to others.
So 4.4 is a **guided, learn-oriented documentation set**, not just a README.
Plain Markdown + Mermaid in the repo (GitHub renders both natively — no docs dependency, no build).

**Shape** — `docs/learn/`:
- `README.md` — how to use the guide, suggested reading order, map of paths.
- `00-big-picture.md` — what a harness is; the anatomy of one run end-to-end (sequence diagram:
  alert → context → loop → tools → policy → HITL → report → trace/audit → evals).
- `glossary.md` — every term in one or two lines, each linking to the path section that explains it.
- `paths/` — one guided path per concern, ordered like a run flows:
  1. **Loop** (LOOP) — inner ReAct loop, exit conditions, raw vs graph, outer loops.
  2. **Tools & environment** (TOOLS, ENV) — tool contracts, risk tags, registry, sandbox, scenarios.
  3. **Context** (CONTEXT) — prompt assembly, AGENTS.md, runbooks (progressive disclosure),
     prompt versioning, caching. (Compaction: placeholder until 4.5.)
  4. **Safety & control** (PERM, GUARD, HITL) — permission matrix, guardrails/injection, approvals.
  5. **Observability & audit** (OBS, AUDIT) — spans/trace, metrics, tamper-evident audit log.
  6. **Reliability** (ORCH) — normalized model client, retries, record/replay, idempotency, races.
  7. **Evals** (EVAL) — dataset, runner, graders (3 layers), judge calibration, pass@k/pass^k,
     reports, regressions, CI gates.
  8. **Production** — web app, cost guards, webhook, canary, deploy (Docker/Render/Atlas).
  9. **Memory** (MEMORY) — short-term (checkpointer) now; long-term placeholder until 4.6.

**Every path follows one template** (consistency is what makes it learnable):
in one sentence · why it matters (the failure modes it prevents) · key terms · a diagram ·
walkthrough top→down where each idea points to *where it lives* (file + symbol, not line numbers,
which rot) · design decisions & the alternative not taken · "see it yourself" ($0 replay commands) ·
check your understanding (questions with collapsible answers) ·
further reading (well-known articles, every link verified).

**Keeping it true (docs as code):**
- `scripts/concepts.py` → `CONCEPTS.md`: the exhaustive `[HARNESS:*]` index, grouped by pillar,
  file:line links. The paths are the curated route; CONCEPTS.md is the complete map.
- `scripts/check_docs.py`: every relative link in `docs/` and the README resolves, and every
  `file::symbol` reference still exists in the code. Both run in CI (fail on stale/broken).
- Tag audit: tags are uneven today (EVAL 24, TOOLS 1, CONTEXT 3, MEMORY 0); add tags where a
  concept really lives so the index and the paths agree.

**README rewrite** (the front door, ~1 screen): 3-line pitch, live demo link + replay note,
architecture Mermaid, "start here" → `docs/learn/`, concept → path table, a screenshot/GIF of the
approval flow, eval results (from the committed baseline), how to run locally, tradeoffs & next steps.

**Comment pass (first):** every `[HARNESS:*]` block rewritten to 3–5 lines (detail moves to the
paths), and Q&A-style notes rephrased as plain explanation across code, docs, specs and
CLAUDE.md. The codebase reads strictly as a learning/demo project.

**Order:** comment pass → concepts.py + check_docs.py + tag audit → big picture + glossary + guide skeleton →
paths, one or two per step (reviewed as they land) → README last (it summarizes everything).

## 4.5 Context compaction `[HARNESS:CONTEXT]` (cut 2nd)

- `opspilot/context/compaction.py`: strategy interface `HistoryStrategy` with `FullReplay` and `Compaction`.
- Compaction triggers when estimated context tokens > threshold (e.g. 70% of a configured window, set low
  for demo): keep system + original alert + last N turns; summarize the middle with the model into a
  structured "investigation so far" note (hypotheses, evidence, ruled-out causes, actions taken).
- **Invariant:** never split a `tool_use` from its `tool_result`. Test it.
- `compaction` span with before/after token counts. Ablation: compaction on vs off (accuracy, tokens).
- Compaction changes the prompts → existing cassettes miss when it's on. Re-record whatever it affects.

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

## Combined ablation session (after 4.6)

Day 3's 3.6 was deferred to here: run all ablations together, with one cost plan — model A vs B
(haiku-4-5 vs sonnet-5), raw vs graph, compaction on vs off, memory on vs off — and write
`docs/ablations.md`. Prerequisite: the human scores in `evals/judge_calibration.yaml` are filled in
(blind) and `opspilot eval judge-check` reports the judge as trusted.

## Questions to test yourself

1. How does your UI resume a paused agent after a human approves? What if the server restarted in between?
2. How do you secure a webhook that triggers an agent? What about duplicate alerts?
3. Explain inner vs outer loops using your project.
4. Full replay vs compaction: tradeoffs? What can go wrong with compaction?
5. Short-term vs long-term memory in your system. How do you prevent bad memories?
6. What protects your API bill on a public demo? (replay by default — no usable key on the public
   deployment; live mode owner-only behind rate limit, daily cap and per-run token budget)
7. What would you change to run this for a real company? (auth, multi-tenant sandboxes, vector memory,
   OpenTelemetry export, eval dataset from prod traces)
