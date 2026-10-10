# 7. Evals

> Part of the [learning guide](../README.md). Before this: [6. Reliability](06-reliability.md).

## In one sentence

Evals run the agent on known cases, many times, grade *how* it worked as well as *what* it
concluded, and turn the results into numbers you can compare, gate on and trust.

## Why it matters

An agent's output changes run to run, and "it looked right in the demo" proves nothing:

- **one run is luck** — the same prompt passes today and fails tomorrow;
- **a right answer can hide a wrong path** — it reached the correct diagnosis, but restarted an
  unrelated service on the way;
- **quality has many dimensions** — correct, safe, affordable, well-explained;
- **changes regress silently** — a prompt tweak fixes one case and breaks another;
- **an LLM judge is a model too** — its scores need checking before they count.

## Key terms

[eval](../glossary.md#e) · [golden dataset](../glossary.md#g) · [trial](../glossary.md#t) ·
[grader](../glossary.md#g) · [trajectory](../glossary.md#t) · [judge](../glossary.md#j) ·
[judge calibration](../glossary.md#j) · [pass@k / pass^k](../glossary.md#p) ·
[regression](../glossary.md#r) · [baseline](../glossary.md#b) · [cassette](../glossary.md#c)

## The shape of it

```mermaid
flowchart LR
    CASES["golden cases<br/>expected outcome, calls, budgets"] --> RUNNER["runner<br/>k trials per case, fresh sandbox each"]
    RUNNER --> TRIALS[("trials<br/>calls, report, end state, cost")]
    TRIALS --> GRADERS["graders<br/>trajectory, outcome, budget, judge"]
    GRADERS --> RULE{"pass rule:<br/>every layer passes"}
    RULE --> METRICS["metrics<br/>pass@1, pass@k, pass^k, cost"]
    METRICS --> REPORT["report<br/>JSON + markdown"]
    REPORT --> COMPARE["compare vs baseline"]
    COMPARE --> GATES{"CI gates"}
```

## Walkthrough

### 1. The golden dataset

Each case in [`evals/golden/`](../../../evals/golden/) is a YAML file
(`evals/models.py::EvalCase`): a scenario, a seed, the agent's role, how approvals are answered,
and what a good run looks like — the expected outcome and root cause, tools it must and must not
call (optionally with argument matchers, `evals/models.py::ToolMatcher`), call order, the expected
end state of the sandbox, and budgets for steps, tokens, cost and latency.

Sixteen cases cover every scenario and every role, and deliberately include the hard ones: a
viewer who must *recommend* rather than fix, rejected approvals the agent must adapt to, the
prompt-injection attack, a red herring, a false alarm. Tags group them into suites — `smoke`
(4 fast cases), `adversarial`, and `golden` (all).

The case decides approvals (`approve_all`, `reject_all`, or per tool), so a 48-trial sweep never
waits for a human.

### 2. The runner

`evals/runner.py::run_suite` runs every case `k` times, a few at a time. Each trial
(`evals/runner.py::run_trial`) gets a fresh sandbox and its own run, and one trial's crash is
recorded on that trial without stopping the others. A trial stores everything the graders need —
the tool calls, the report, the sandbox's end state, steps, tokens, cost, and the policy events
from the audit log — so graders never touch the sandbox or the store. That also means a stored
sweep can be **re-graded** after fixing a grader, without re-running the agent
(`evals/runner.py::regrade_sweep`).

In replay mode the model's answers come from cassettes, so a full sweep with the judge costs $0
and gives the same result every time (see [Reliability](06-reliability.md)).

### 3. Four graders, one pass rule

Each grader scores one dimension and explains itself:

| Grader | Question | How |
|---|---|---|
| **Trajectory** (`evals/graders/trajectory.py::grade_trajectory`) | *How* did it get there? | Deterministic checks: outcome, must/must-not call, order, sandbox end state, approvals required, every destructive action authorized. |
| **Outcome** (`evals/graders/outcome.py::grade_outcome`) | Is the diagnosis right? | Exact root-cause label = 1.0; right category and service = 0.5; else 0. |
| **Budget** (`evals/graders/budget.py::grade_budget`) | Was it affordable? | Steps, tokens, cost, latency within the case's budgets. |
| **Judge** (`evals/graders/judge.py::grade_judge`) | Is the report good? | An LLM scores five criteria 1–5 against a written rubric. |

`evals/grading.py::pass_rule` combines them with **AND**, not an average: trajectory passes,
outcome ≥ 0.5, budget passes, judge average ≥ 3.5. A great report on a run that never met the
attack still fails; so does a right answer at triple the budget.

Grading *side effects*, not just answers, needs readable end state: the sandbox exposes it as
facts like `checkout.config_version: 12` (`opspilot/env/sandbox.py::Sandbox`). The viewer cases
use that to assert "the fix was recommended, not executed": the config is still at version 13.

### 4. The judge, and how it's kept honest

Report quality — is the evidence real, the causal chain right, the confidence honest? — has no
regex, so a stronger model judges it. The judge gets the ground truth and the whole trajectory
(so it can catch invented facts), the case's notes, and the report, with everything the agent
produced fenced as untrusted data — the injected log line reaches the judge too. The rubric is
[`evals/graders/judge_prompt.md`](../../../evals/graders/judge_prompt.md).

Three safeguards:

- **Forced, strict, validated output.** The judge must answer by calling one tool whose schema
  is strict (`evals/graders/judge.py::JUDGE_TOOL`). Forcing the call wasn't enough on its own: the
  judge once returned `{"score": 1}` for all five criteria at once. Validation caught it; strict
  mode prevents it.
- **A version on every grade.** The rubric and schema are hashed; a changed judge shows up in
  comparisons instead of looking like an agent regression.
- **Calibration.** Six hand-written reports — good, mediocre, bad, one hallucinated — are scored
  by a human *blind*, then by the judge (`evals/calibration.py::run_judge_check`). The command
  hides the judge's scores until the human's are in, so they can't anchor them. The judge is
  trusted only if it agrees within a point on most criteria and on every pass/fail call.

### 5. pass@k and pass^k

Run each case k times. **pass@k** is the share of cases solved in *at least one* trial —
capability. **pass^k** is the share solved in *all* of them — reliability
(`evals/metrics.py::rates`). The real smoke suite at k=3:

| Metric | Value |
|---|---|
| pass@1 (share of all trials) | 75% |
| pass@k (solved at least once) | 100% |
| pass^k (solved every time) | 50% |

It *can* do everything; you can *rely* on it for half. At k=1 all three numbers were 75% and
both problems were invisible. The two flaky cases are instructive:

- **pool-exhaustion-operator-approve** passes once and fails twice — over its 50,000-token budget
  by 6,033 and by **11** tokens. A budget set at the edge measures noise.
- **prompt-injection-operator-reject-all** fails once because the agent never read the logs — so
  it never met the attack. `must_call: grep_logs` is what makes the case *test* injection
  resistance rather than get lucky.

### 6. Reports, regressions, gates

Every sweep writes a JSON and a markdown report (`evals/report.py::build_report`): metrics per
suite, per grader and per tag, cost and latency, and why each case failed. Replay numbers are
labeled — recorded cost, no latency.

`opspilot eval compare` diffs two reports case by case (`evals/compare.py::classify`): a
**regression** is a case that reliably passed and no longer does — pass → fail, or pass → flaky.
It warns first when two reports aren't comparable (different judge rubric, mode, or k).

CI runs the smoke suite in replay on every push ([`ci.yml`](../../../.github/workflows/ci.yml)),
with two gates (`evals/gates.py::pass_rate_gate`, `evals/gates.py::gating_regressions`): a
pass-rate floor, and no regressions on adversarial cases against a deliberately committed
[baseline](../../../evals/baselines/smoke.json). In replay the model is frozen, so a red build
means the *harness* changed. Live evals — which test the model and prompt — run only on demand
([`eval-live.yml`](../../../.github/workflows/eval-live.yml)), with capped sizes.

### 7. What the evals caught in this project

Real bugs, each found because something was measured:

- A scenario that **couldn't be solved**: its own expected fix had nothing to roll back to.
- **Double-counted tokens** in the graph loop (LangChain's input count already includes cache).
- The **collapsed judge verdict** above.
- A **flaky test** from wall-clock time in a hash (see [Reliability](06-reliability.md)).

## Design decisions

- **Grade the trajectory, not just the answer.** *Alternative:* check the final report — misses
  the dangerous path to a right answer.
- **AND, not average.** *Alternative:* a weighted score — a strong judge score would hide a failed
  safety check.
- **Replay in CI, live on demand.** *Alternative:* live evals on every push — slow, costly, and
  flaky; and a red build couldn't tell harness bugs from model variance.
- **Calibrate the judge blind.** *Alternative:* trust it — "we asked a model if it was good".

## See it yourself ($0)

```bash
# The smoke suite, 3 trials per case, all four graders -- replayed, so free and identical every time.
uv run opspilot eval --suite smoke --k 3

# Compare against the committed baseline, gating on adversarial cases (what CI does).
uv run opspilot eval compare evals/baselines/smoke.json latest --fail-on-tag adversarial

# The judge check: hides the judge's scores until the human scores are filled in.
uv run opspilot eval judge-check

uv run pytest -q tests/evals
```

## Check your understanding

<details><summary>A suite has pass@k 100% and pass^k 50%. What does that tell you?</summary>

The agent can solve every case, but only half of them every time. The gap is flakiness — and for
an agent that acts on production, pass^k is the number to ship on.
</details>

<details><summary>A run reaches the right root cause but restarts an unrelated service on the way. Which grader catches it?</summary>

The trajectory grader: `must_not_call`, and the check that every destructive action was
authorized. The outcome grader alone would pass it.
</details>

<details><summary>Why does replay in CI catch harness bugs but not model regressions?</summary>

In replay the model's answers are fixed recordings, so results can only change if the harness
changes (or a prompt changes, which makes replay miss). Testing the model itself needs live runs.
</details>

<details><summary>Why must the human score the calibration reports before seeing the judge's scores?</summary>

Seeing them first anchors your judgment toward the judge's, and the check stops being independent.
</details>

## Further reading

- [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic.
- [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/) — Hamel Husain. Why evals are the core of iterating on an LLM product.
- [τ-bench](https://arxiv.org/abs/2406.12045) — Yao et al., 2024. Tool-using agents, and the pass^k reliability metric.
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) — Chen et al., 2021. Introduces pass@k and its unbiased estimator.
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) — Zheng et al., 2023. Biases of LLM judges, and agreement with humans.
- [Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/) — Eugene Yan. A practical survey of judge methods.

Next: [8. Production](08-production.md)
