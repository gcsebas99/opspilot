# Eval baselines

Reports committed **on purpose** as the reference CI compares against
(`evals/reports/` itself is gitignored — those are per-run artifacts).

| file | suite | k | what CI does with it |
|---|---|---|---|
| `smoke.json` | smoke | 3 | `opspilot eval compare evals/baselines/smoke.json latest --fail-on-tag adversarial` — fails the build if any adversarial case regresses |

In replay mode the model's answers are frozen, so a result can only change
when the **harness** changes (graders, policy, tools, loop) or a cassette is
re-recorded. A regression against this file is therefore something *we* did.

## Updating a baseline

Only when a change is intended (e.g. a grader fix, new cassettes, a case
edit) — never just to make CI green:

```bash
uv run opspilot eval --suite smoke --k 3            # replay, $0
cp "$(ls -t evals/reports/*.json | head -1)" evals/baselines/smoke.json
git add evals/baselines/smoke.json
git commit -m "chore(evals): update smoke baseline -- <why the results changed>"
```

The commit message is the audit trail: it should say *why* the expected
results moved. Run `opspilot eval compare evals/baselines/smoke.json latest`
first and paste the regressions/fixes it lists.
