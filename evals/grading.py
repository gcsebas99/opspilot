from pydantic import BaseModel

from evals.graders.base import Grade
from evals.graders.budget import grade_budget
from evals.graders.judge import grade_judge
from evals.graders.outcome import grade_outcome
from evals.graders.trajectory import grade_trajectory
from evals.models import EvalCase
from opspilot.models.base import ModelClient
from opspilot.store.models import EvalTrialDoc

OUTCOME_MIN_SCORE = 0.5


class GradingResult(BaseModel):
    grades: dict[str, Grade]
    passed: bool
    # False when run with --no-judge: `passed` then reflects the
    # deterministic layers only, and reports must say so.
    judged: bool


# [HARNESS:EVAL] The case pass rule: every hard layer must pass (AND, not average).
# WHY: each layer catches what the others can't -- a perfect report on a run that
# never met the attack, a clean trajectory with a wrong diagnosis, a right answer
# at 3x budget. One strong layer must never cover for a failed one.
def pass_rule(grades: dict[str, Grade]) -> bool:
    hard = (
        grades["trajectory"].passed
        and grades["outcome"].score >= OUTCOME_MIN_SCORE
        and grades["budget"].passed
    )
    return hard and ("judge" not in grades or grades["judge"].passed)


async def grade_trial(
    case: EvalCase,
    trial: EvalTrialDoc,
    *,
    judge: ModelClient | None,
    judge_model: str,
) -> GradingResult:
    """Run every grader layer over one finished trial. Pure over (case,
    trial) except the judge call, which goes through a ModelClient -- so
    replay grades for free and deterministically."""
    grades = {
        "trajectory": grade_trajectory(case, trial),
        "outcome": grade_outcome(case, trial),
        "budget": grade_budget(case, trial),
    }
    if judge is not None:
        grades["judge"] = await grade_judge(case, trial, judge, judge_model)
    return GradingResult(grades=grades, passed=pass_rule(grades), judged=judge is not None)
