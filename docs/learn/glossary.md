# Glossary

> Part of the [learning guide](README.md). Short definitions; each points to the path that
> explains it and, where it helps, the code that implements it.

[A](#a) · [B](#b) · [C](#c) · [D](#d) · [E](#e) · [G](#g) · [H](#h) · [I](#i) · [J](#j) ·
[L](#l) · [M](#m) · [O](#o) · [P](#p) · [R](#r) · [S](#s) · [T](#t) · [W](#w)

## A

**Agent** — a model plus a harness, running in a loop: it chooses tools, sees their results, and
decides when it's done. → [Big picture](00-big-picture.md)

**AGENTS.md** — the agent's standing instructions (how to investigate, when to escalate, that tool
output is data). Part of every system prompt. `opspilot/context/AGENTS.md` → [Context](paths/03-context.md)

**Approval (HITL)** — a pause where a human approves or rejects a destructive action before it
runs. `opspilot/loops/graph.py::resume_react_graph` → [Safety & control](paths/04-safety-and-control.md)

**Audit log** — an append-only record of security-relevant events (denials, decisions, destructive
actions), hash-chained so tampering is detectable. `opspilot/observability/audit.py::verify_chain`
→ [Observability & audit](paths/05-observability-and-audit.md)

## B

**Backoff with jitter** — retrying a transient failure after a growing, slightly randomized delay,
so many clients don't retry in lockstep. `opspilot/models/anthropic_model.py::AnthropicModel`
→ [Reliability](paths/06-reliability.md)

**Baseline** — a deliberately committed eval report that CI compares new results against.
`evals/baselines/smoke.json` → [Evals](paths/07-evals.md)

**Budget** — a cap on steps, tokens, cost or latency. The loop enforces step and token budgets;
evals grade all four. → [The loop](paths/01-loop.md), [Evals](paths/07-evals.md)

## C

**Canary** — a scheduled synthetic request with a known right answer, sent to the deployed system
to prove it still works end to end. `scripts/canary.py::run_canary` → [Production](paths/08-production.md)

**Cassette** — a file of recorded model requests and responses, keyed by a hash of the request.
`opspilot/models/cassette.py::Cassette` → [Reliability](paths/06-reliability.md), [Evals](paths/07-evals.md)

**Checkpointer** — LangGraph's persistence for a run's state after every step; what makes a paused
run resumable, even from another process. `opspilot/loops/graph.py::build_checkpointer`
→ [Memory](paths/09-memory.md)

**Compaction** — summarizing older turns to keep a long conversation inside the context window.
Planned, not built yet. → [Context](paths/03-context.md)

**Context window** — everything the model sees on one call: system prompt, tool schemas, and the
whole conversation so far. → [Context](paths/03-context.md)

## D

**Destructive tool** — a tool that changes the environment (`restart_service`, `rollback_config`,
`rollback_deploy`). Gated by the permission policy. → [Tools & environment](paths/02-tools-and-environment.md)

## E

**Escalate** — the terminal tool for "a human needs to take this"; still names a suspected root
cause. `opspilot/tools/terminal.py::EscalateInput` → [The loop](paths/01-loop.md)

**Eval** — running the agent on known cases and grading the results, many times, to measure it.
`evals/runner.py::run_suite` → [Evals](paths/07-evals.md)

**Exit condition** — any rule that ends the loop: terminal tool, step cap, token budget, stuck
detection, or no tool call twice. → [The loop](paths/01-loop.md)

## G

**Golden dataset** — the hand-written eval cases, each with expected outcome, calls and budgets.
`evals/golden/` → [Evals](paths/07-evals.md)

**Grader** — code (or a model) that scores one trial on one dimension. Here: trajectory, outcome,
budget, judge. `evals/grading.py::grade_trial` → [Evals](paths/07-evals.md)

**Guardrail** — a check that constrains inputs or outputs: untrusted-data framing, injection
detection, scope check, report validation. `opspilot/policy/guardrails.py`
→ [Safety & control](paths/04-safety-and-control.md)

## H

**Harness** — all the code around the model that makes it an agent: loop, tools, context, policy,
memory, observability, evals. → [Big picture](00-big-picture.md)

**HMAC signature** — a keyed hash proving a webhook came from someone who knows the shared secret
and wasn't altered. `opspilot/web/webhooks.py::verify` → [Production](paths/08-production.md)

## I

**Idempotency** — doing something twice has the same effect as once; a redelivered alert returns
the existing run. `opspilot/web/service.py::WebRunService` → [Reliability](paths/06-reliability.md)

**Interrupt** — LangGraph's way to pause a graph mid-node and wait for input (here, a human
decision). → [Safety & control](paths/04-safety-and-control.md)

## J

**Judge (LLM-as-judge)** — a model that grades another model's output against a rubric. Needs
calibration against human scores. `evals/graders/judge.py::grade_judge` → [Evals](paths/07-evals.md)

**Judge calibration** — checking the judge against human scores on reports of known quality.
`evals/calibration.py::run_judge_check` → [Evals](paths/07-evals.md)

## L

**LangGraph** — a library for building agents as graphs of nodes with checkpointed state; used for
the graph loop. `opspilot/loops/graph.py::build_graph` → [The loop](paths/01-loop.md)

## M

**Model client** — the one interface every model backend implements (live, scripted, recorded,
replayed). `opspilot/models/base.py::ModelClient` → [Reliability](paths/06-reliability.md)

## O

**Outer loop** — what starts or re-runs the agent: an event (webhook), a schedule (cron), a
heartbeat (canary). → [The loop](paths/01-loop.md)

## P

**pass@k / pass^k** — share of cases solved in *at least one* of k trials (capability) vs in *all*
k (reliability). `evals/metrics.py::rates` → [Evals](paths/07-evals.md)

**Permission policy** — the table of what each role may do with each risk class: allow, deny, or
require approval. `opspilot/policy/permissions.py::decide` → [Safety & control](paths/04-safety-and-control.md)

**Progressive disclosure** — giving the model an index first and the details only on request
(runbooks load via a tool). → [Context](paths/03-context.md)

**Prompt caching** — the API reusing an unchanged prompt prefix across calls, cheaper and faster.
Breaks if anything before the cache marker changes. → [Context](paths/03-context.md)

**Prompt injection** — instructions hidden in data (here, a log line) trying to steer the agent.
→ [Safety & control](paths/04-safety-and-control.md)

**Prompt version** — a hash of everything the agent is told (AGENTS.md, runbook index, tool
schemas), stored on every run. `opspilot/context/assembler.py::compute_prompt_version`
→ [Context](paths/03-context.md)

## R

**ReAct** — the reason → act → observe loop: the model thinks, calls a tool, reads the result,
repeats. → [The loop](paths/01-loop.md)

**Record / replay** — recording real model responses once, then replaying them for free and
deterministically. → [Reliability](paths/06-reliability.md)

**Regression** — a case that reliably passed before and no longer does. `evals/compare.py::classify`
→ [Evals](paths/07-evals.md)

**Risk class** — every tool is `read`, `destructive` or `terminal`; the policy keys off it.
`opspilot/tools/base.py::Risk` → [Tools & environment](paths/02-tools-and-environment.md)

**Role** — whose permissions a run (or a human) has: `viewer`, `operator`, `admin`, `system`.
→ [Safety & control](paths/04-safety-and-control.md)

**Runbook** — a markdown guide for one failure mode, loaded on demand by the `load_runbook` tool.
`opspilot/context/runbooks/` → [Context](paths/03-context.md)

## S

**Sandbox** — the run's private, seeded copy of ShopStack; every file access is confined to it.
`opspilot/env/sandbox.py::Sandbox` → [Tools & environment](paths/02-tools-and-environment.md)

**Scenario** — a fault to inject (and its hidden ground truth) plus the alert the agent sees.
`opspilot/env/scenarios.py::Scenario` → [Tools & environment](paths/02-tools-and-environment.md)

**Scope check** — destructive actions on services unrelated to the alert need approval, even for
admin. `opspilot/policy/guardrails.py::is_in_scope` → [Safety & control](paths/04-safety-and-control.md)

**Short-term memory** — what the agent remembers within one run: the conversation and state,
checkpointed after every step. → [Memory](paths/09-memory.md)

**Span** — one timed unit of work in a trace (model call, tool call, policy check, approval wait),
with a parent. `opspilot/observability/tracer.py::Tracer` → [Observability & audit](paths/05-observability-and-audit.md)

**Stuck detection** — ending a run that repeats the same tool call with the same arguments.
→ [The loop](paths/01-loop.md)

## T

**Terminal tool** — a tool that ends the run: `submit_report` or `escalate`. → [The loop](paths/01-loop.md)

**Tool** — a function the model can call, described by a name, description and JSON schema.
`opspilot/tools/base.py::Tool` → [Tools & environment](paths/02-tools-and-environment.md)

**Tool result** — what a tool returns to the model; errors are results too, so the model can
correct itself. `opspilot/tools/base.py::ToolResult` → [Tools & environment](paths/02-tools-and-environment.md)

**Trace** — all the spans of one run, nested and timed; shown as a waterfall in the web UI.
→ [Observability & audit](paths/05-observability-and-audit.md)

**Trajectory** — the sequence of tool calls a run made; graded separately from the final answer.
`evals/graders/trajectory.py::grade_trajectory` → [Evals](paths/07-evals.md)

**Trial** — one run of one eval case; a case runs k trials. → [Evals](paths/07-evals.md)

## W

**Webhook** — an HTTP endpoint another system calls when something happens; here, an alert
starts a run. `opspilot/web/app.py::create_app` → [Production](paths/08-production.md)
