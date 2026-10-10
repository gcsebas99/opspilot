from evals.compare import CaseDiff, Comparison
from evals.metrics import SuiteMetrics


# [HARNESS:EVAL] CI gates: what makes a red build.
# WHY: in replay the model's answers are frozen, so a changed result means the
# harness changed (or a prompt drifted). Two gates: a pass-rate floor, and no
# regressions on critical tags vs a committed baseline.
def pass_rate_gate(metrics: SuiteMetrics, min_pass_rate: float) -> str | None:
    """Failure message if pass@1 is below the floor, else None."""
    actual = metrics.overall.pass_at_1
    if actual < min_pass_rate:
        return f"pass@1 {actual:.0%} is below the required {min_pass_rate:.0%}"
    return None


def gating_regressions(comparison: Comparison, fail_on_tags: list[str]) -> list[CaseDiff]:
    """Regressions that should fail the build: all of them, or -- if tags
    are given -- only those on cases carrying at least one of those tags."""
    regressions = comparison.of_kind("regression")
    if not fail_on_tags:
        return regressions
    return [diff for diff in regressions if set(diff.tags) & set(fail_on_tags)]
