from evals.graders.base import Check, Grade, grade_from_checks
from evals.models import EvalCase
from opspilot.store.models import EvalTrialDoc


def _within(name: str, actual: float | None, limit: float) -> Check:
    if actual is None:
        return Check(name=name, passed=False, detail="not recorded")
    return Check(name=name, passed=actual <= limit, detail=f"{actual:g} / {limit:g}")


# [HARNESS:EVAL] Budget grading -- a correct answer that cost too much fails.
# WHY: in production, cost and latency are requirements, not nice-to-haves;
# a prompt change that fixes one case but doubles tokens should show up as
# a failure here, not be discovered on the invoice. Tokens count cache reads
# and writes too, matching the loop's own token budget.
# INTERVIEW: "+5% accuracy for +40% cost -- ship it?" -> make cost a graded,
# per-case budget so the tradeoff is explicit in the eval report.
def grade_budget(case: EvalCase, trial: EvalTrialDoc) -> Grade:
    budgets = case.budgets
    tokens = sum(trial.tokens.values()) if trial.tokens else None
    checks = [
        _within("steps", trial.steps, budgets.max_steps),
        _within("tokens", tokens, budgets.max_tokens),
        _within("cost_usd", trial.cost_usd, budgets.max_cost_usd),
    ]
    if trial.mode == "replay":
        # Replay latency is disk reads, not model calls -- grading it would
        # pass everything and mean nothing. Cost/tokens are the recorded
        # values, so those checks still hold.
        checks.append(Check(name="latency_s", passed=True, detail="skipped in replay"))
    else:
        checks.append(_within("latency_s", trial.latency_s, budgets.max_latency_s))
    return grade_from_checks("budget", checks)
