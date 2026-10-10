# Learning guide: how OpsPilot's harness works

OpsPilot is a small incident-triage agent built to make **harness engineering** visible: every
part of the code around the model — loop, tools, context, permissions, approvals, traces, audit,
evals — is implemented plainly, tagged, and explained here.

This guide is the route through it. It assumes you know Python and have called an LLM API; it
doesn't assume you've built an agent.

## How to read it

1. **[The big picture](00-big-picture.md)** (10 min) — what a harness is, the pieces, and one
   run end to end. Read this first.
2. **The paths** — one concern each, in the order a run flows. Each takes 10–15 minutes and can
   be read on its own once you've seen the big picture.
3. **[The glossary](glossary.md)** — keep it open; every term links back to the path that explains it.

Every path has the same shape: *in one sentence* · *why it matters* · *key terms* · *a diagram* ·
*the walkthrough* (each idea points to where it lives in the code) · *design decisions* ·
*see it yourself* · *check your understanding* · *further reading*.

## The paths

| # | Path | What it answers |
|---|---|---|
| 1 | [The loop](paths/01-loop.md) | How does the agent decide what to do next — and when to stop? What starts it? |
| 2 | [Tools & environment](paths/02-tools-and-environment.md) | What can it call, and what world do those calls act on? |
| 3 | [Context](paths/03-context.md) | What does the model see on every call, and how do we keep that small and stable? |
| 4 | [Safety & control](paths/04-safety-and-control.md) | What may it do? What stops a bad action? When does a human decide? |
| 5 | [Observability & audit](paths/05-observability-and-audit.md) | How do we see what happened — and prove the record wasn't changed? |
| 6 | [Reliability](paths/06-reliability.md) | What happens with flaky APIs, swapped backends, duplicate requests, races? |
| 7 | [Evals](paths/07-evals.md) | How do you measure an agent whose output changes every run? |
| 8 | [Production](paths/08-production.md) | What does it take to run it for others — safely and for $0? |
| 9 | [Memory](paths/09-memory.md) | What does it remember within a run, and across runs? |

## Conventions

- **Code references** look like `opspilot/runs.py::open_run` — a file and a function, class or
  variable in it. Line numbers aren't used in the guide; they change too often.
- **Tags in the code.** Every place a concept is implemented has a comment like
  `# [HARNESS:LOOP] Exit condition #2 -- hard step cap.` followed by a one-line *WHY*. The
  complete list, with line links, is [`CONCEPTS.md`](../../CONCEPTS.md).
- **"See it yourself" commands cost $0.** They replay recorded model responses; no API key needed.

## How this guide stays true

Docs that aren't checked drift from the code. Two scripts run in CI on every push:

- `scripts/concepts.py` regenerates `CONCEPTS.md` from the tags, and fails if it's stale or a
  tag breaks the convention.
- `scripts/check_docs.py` fails if a link in this guide is broken, or a code reference names a
  file or function that no longer exists.

Rename a function, and CI names the page that needs updating.
