import math
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel

from evals.models import EvalCase
from opspilot.store.models import EvalTrialDoc

CaseVerdict = Literal["pass", "fail", "flaky"]


class Rates(BaseModel):
    cases: int
    trials: int
    pass_at_1: float
    pass_at_k: float
    pass_hat_k: float


class Stats(BaseModel):
    n: int
    mean: float
    p95: float
    total: float


class SuiteMetrics(BaseModel):
    k: int
    overall: Rates
    per_tag: dict[str, Rates]
    # Fraction of *graded* trials each grader passed -- which layer fails most.
    per_grader: dict[str, float]
    errored: int
    agent_cost: Stats | None
    judge_cost: Stats | None
    tokens: Stats | None
    # Live trials only -- see compute_metrics.
    latency_s: Stats | None
    judge_avg: float | None
    case_verdicts: dict[str, CaseVerdict]


def trial_passed(trial: EvalTrialDoc) -> bool:
    """A trial passes only if it ran, graded, and met the pass rule --
    an errored run or a grading error is a failure, never "skipped"."""
    return trial.passed is True and trial.error is None and trial.grading_error is None


def case_verdict(trials: list[EvalTrialDoc]) -> CaseVerdict:
    passed = sum(trial_passed(t) for t in trials)
    if passed == len(trials):
        return "pass"
    return "fail" if passed == 0 else "flaky"


# [HARNESS:EVAL] pass@k vs pass^k -- capability vs reliability.
# WHY: one trial per case measures luck. pass@k: solved in at least one of k
# trials; pass^k: solved in all k. The gap is flakiness. (n == k here, so these
# are exact; with n > k use the unbiased estimator from Chen et al., 2021.)
def rates(by_case: dict[str, list[EvalTrialDoc]]) -> Rates:
    trials = [t for ts in by_case.values() for t in ts]
    if not by_case:
        return Rates(cases=0, trials=0, pass_at_1=0.0, pass_at_k=0.0, pass_hat_k=0.0)
    verdicts = [case_verdict(ts) for ts in by_case.values()]
    return Rates(
        cases=len(by_case),
        trials=len(trials),
        pass_at_1=sum(trial_passed(t) for t in trials) / len(trials),
        pass_at_k=sum(v != "fail" for v in verdicts) / len(verdicts),
        pass_hat_k=sum(v == "pass" for v in verdicts) / len(verdicts),
    )


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. With a handful of samples p95 is simply the
    max -- that's honest, not a bug; interpolating would invent a value no
    trial actually had."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def stats(values: list[float]) -> Stats | None:
    if not values:
        return None
    return Stats(
        n=len(values),
        mean=sum(values) / len(values),
        p95=percentile(values, 0.95),
        total=sum(values),
    )


def compute_metrics(trials: list[EvalTrialDoc], cases: dict[str, EvalCase]) -> SuiteMetrics:
    by_case: dict[str, list[EvalTrialDoc]] = defaultdict(list)
    for trial in trials:
        by_case[trial.case_id].append(trial)

    by_tag: dict[str, dict[str, list[EvalTrialDoc]]] = defaultdict(dict)
    for case_id, case_trials in by_case.items():
        case = cases.get(case_id)
        for tag in case.tags if case is not None else []:
            by_tag[tag][case_id] = case_trials

    graded = [t for t in trials if t.grades]
    grader_names = sorted({name for t in graded for name in t.grades or {}})
    per_grader = {
        name: sum(
            bool((t.grades or {})[name]["passed"]) for t in graded if name in (t.grades or {})
        )
        / sum(1 for t in graded if name in (t.grades or {}))
        for name in grader_names
    }

    judge_grades = [(t.grades or {}).get("judge") for t in graded]
    judge_details = [g["details"] for g in judge_grades if g is not None]
    judge_averages = [d["average"] for d in judge_details if d.get("average") is not None]

    # [HARNESS:EVAL] Only live trials report latency.
    # WHY: replayed (or cassette-served) calls measure disk reads, not the model.
    # Cost stays: in replay it's the recorded cost of real calls, labeled as such.
    live = [t.latency_s for t in trials if t.mode == "live" and t.latency_s is not None]

    return SuiteMetrics(
        k=max((len(ts) for ts in by_case.values()), default=0),
        overall=rates(by_case),
        per_tag={tag: rates(tag_cases) for tag, tag_cases in sorted(by_tag.items())},
        per_grader=per_grader,
        errored=sum(t.error is not None or t.grading_error is not None for t in trials),
        agent_cost=stats([t.cost_usd for t in trials if t.cost_usd is not None]),
        judge_cost=stats([d["cost_usd"] for d in judge_details if d.get("cost_usd") is not None]),
        tokens=stats([float(sum(t.tokens.values())) for t in trials if t.tokens]),
        latency_s=stats(live),
        judge_avg=sum(judge_averages) / len(judge_averages) if judge_averages else None,
        case_verdicts={case_id: case_verdict(ts) for case_id, ts in sorted(by_case.items())},
    )
