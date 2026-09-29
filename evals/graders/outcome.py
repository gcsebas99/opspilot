from evals.graders.base import Grade
from evals.models import EvalCase
from opspilot.store.models import EvalTrialDoc

NO_INCIDENT = "no_incident"


def reported_root_cause(trial: EvalTrialDoc) -> str | None:
    """submit_report's root_cause, or escalate's suspected_root_cause."""
    report = trial.report or {}
    value = report.get("root_cause") or report.get("suspected_root_cause")
    return str(value) if value else None


# [HARNESS:EVAL] Outcome grading with partial credit on the diagnosis.
# WHY: labels are category:service[:detail]. "config_change:checkout:timeout"
# for a db_pool_size change found the right service and kind of fault -- a
# human on call would be pointed at the right place -- so it earns 0.5, not
# the same 0 as blaming the wrong service. Exact label -> 1.0.
# INTERVIEW: "Exact match or fuzzy grading?" -> structured partial credit
# on a label schema; deterministic, explainable, no LLM needed.
def grade_outcome(case: EvalCase, trial: EvalTrialDoc) -> Grade:
    expected = case.expect.root_cause
    actual = reported_root_cause(trial)
    if expected is None:
        return Grade(name="outcome", passed=True, score=1.0, details={"note": "no expectation"})

    details = {"expected": expected, "actual": actual}
    if actual == expected:
        score, match = 1.0, "exact"
    elif actual is None:
        score, match = 0.0, "no root cause reported"
    elif NO_INCIDENT in (expected, actual):
        # "no incident" has no category/service -- it's all or nothing.
        score, match = 0.0, "incident vs no_incident mismatch"
    elif actual.split(":")[:2] == expected.split(":")[:2]:
        score, match = 0.5, "category+service"
    else:
        score, match = 0.0, "wrong category or service"
    return Grade(
        name="outcome", passed=score >= 0.5, score=score, details={**details, "match": match}
    )
