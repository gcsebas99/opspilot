# CLAUDE.md — OpsPilot

OpsPilot is an **incident triage agent** for a fake e-commerce stack ("ShopStack").
Its real purpose is to be a **learning + portfolio project for harness engineering**:
every harness concept (loop, tools, context, environment, memory, observability,
guardrails, HITL, permissions, audit, evals) must be implemented *visibly* and *correctly*.
What the agent accomplishes matters less than how well the harness around it is built.

The owner (Sebas) is building this to learn. He must understand every line.

## Working agreement (read first, every session)

1. **Always start in plan mode.** Read the relevant spec in `docs/specs/dayN.md`,
   propose a plan listing files + concepts touched, and wait for approval.
2. **Work in small steps** (one sub-task of the spec at a time). After each step:
   summarize what changed, which `[HARNESS:*]` concepts were implemented, and how to run/verify it.
3. **Explain, don't just write.** When a design choice has a tradeoff, state it in 1–2 lines
   and name the alternative you didn't pick.
4. **Ask before adding a dependency** not already listed in `docs/PLAN.md`.
5. **Verify library APIs against current docs** (LangGraph, langchain-anthropic, anthropic SDK,
   PyMongo async). Do not rely on memory for APIs that change between versions.
6. Never commit secrets. Config comes from env vars via `pydantic-settings` (`.env` is gitignored).
7. Stop at the end of a spec's sub-task list. Do not start the next day's spec unprompted.

## Learning-comment convention (MANDATORY)

Mark every place where a harness concept is *actually implemented* with a tagged comment:

```python
# [HARNESS:LOOP] Exit condition #2 — hard step cap.
# WHY: a confused model can loop forever and burn tokens.
# INTERVIEW: "How do you stop runaway agents?" → step cap, token budget,
#            stuck detection (same tool+args repeated), explicit final_answer tool.
```

Rules:
- Format: first line `# [HARNESS:<TAG>] <what this is>`, then `# WHY:` (required),
  then `# INTERVIEW:` (optional: likely question → short answer). Max ~6 lines.
- Allowed tags: `LOOP`, `TOOLS`, `CONTEXT`, `ENV`, `MEMORY`, `OBS`, `EVAL`, `GUARD`,
  `HITL`, `PERM`, `AUDIT`, `ORCH` (orchestration / error handling / retries).
- Only on concept-bearing code. No tags on boilerplate, getters, imports.
- Prefer one good tag per concept location over many shallow ones.
- `scripts/concepts.py` builds `CONCEPTS.md` by grepping these tags — keep the first line
  self-explanatory because it becomes the index entry.
- Ordinary comments stay ordinary: explain *why*, never restate the code.

## Tech & conventions

- Python 3.12, `uv` for env/deps, `ruff` (lint+format), `mypy --strict` on `opspilot/`.
- Pydantic v2 models for all tool inputs/outputs, config, and persisted documents.
- Async for I/O (Anthropic calls, Mongo, FastAPI). Sync is fine for pure logic.
- Package layout:
  ```
  opspilot/
    env/            # ShopStack fake system, scenarios, sandbox
    tools/          # tool registry, tool implementations, risk tags
    context/        # prompt assembly, AGENTS.md, runbooks (skills), compaction
    loops/          # react_raw.py (hand-written) and graph.py (LangGraph)
    models/         # ModelClient protocol, Anthropic client, Scripted/Replay clients
    policy/         # permissions, guardrails
    observability/  # tracer, spans, metrics, cost
    store/          # Mongo access layer
    memory/         # long-term incident memory
    web/            # FastAPI app + Jinja/HTMX templates
    cli.py
  evals/            # golden dataset, graders, runner, cassettes, reports
  tests/
  docs/             # PLAN.md, specs/
  ```
- The model ID is **never hardcoded**: `OPSPILOT_MODEL`, `OPSPILOT_JUDGE_MODEL` env vars.

## Testing rules

- **Unit tests never call the real Anthropic API.** Use `ScriptedModel` (predefined responses)
  or `ReplayModel` (cassettes).
- Every tool has at least one test. Every exit condition of the loop has a test.
- Commands:
  - `uv run pytest -q`
  - `uv run ruff check . && uv run ruff format --check .`
  - `uv run mypy opspilot`
  - `uv run opspilot run --scenario <name> --seed 42`
  - `uv run opspilot eval --suite smoke --mode replay`

## Definition of done (per sub-task)

Tests pass, lint/type checks pass, HARNESS tags present where concepts live,
and a 3–5 line "what you should understand now" note printed at the end of the step
so Sebas can put it into `LEARNING.md` in his own words.
