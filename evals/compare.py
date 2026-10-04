from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from evals.metrics import CaseVerdict
from evals.report import REPORTS_DIR, EvalReport, ReportMeta, load_report

Transition = Literal["regression", "fix", "changed", "unchanged", "added", "removed"]


class CaseDiff(BaseModel):
    case_id: str
    kind: Transition
    before: CaseVerdict | None
    after: CaseVerdict | None
    before_rate: float | None
    after_rate: float | None
    tags: list[str] = []


Unit = Literal["rate", "usd", "count", "score", "seconds"]


class Delta(BaseModel):
    name: str
    unit: Unit
    before: float | None
    after: float | None

    def fmt(self, value: float | None, signed: bool = False) -> str:
        """Render a value (or, with signed=True, a change) in this delta's unit."""
        if value is None:
            return "n/a"
        sign = "+" if signed and value >= 0 else ("-" if value < 0 else "")
        magnitude = abs(value)
        if self.unit == "rate":
            # Changes in a rate are percentage *points*, not percent.
            return f"{sign}{magnitude * 100:.0f}{'pp' if signed else '%'}"
        if self.unit == "usd":
            return f"{sign}${magnitude:.4f}"
        if self.unit == "count":
            return f"{sign}{magnitude:,.0f}"
        return f"{sign}{magnitude:.2f}"

    @property
    def change(self) -> float | None:
        if self.before is None or self.after is None:
            return None
        return self.after - self.before


class Comparison(BaseModel):
    a: ReportMeta
    b: ReportMeta
    cases: list[CaseDiff]
    deltas: list[Delta]
    # Differences that make A vs B not apples-to-apples.
    warnings: list[str]
    # Differences that are probably the point (an ablation, a prompt change).
    notes: list[str]

    def of_kind(self, kind: Transition) -> list[CaseDiff]:
        return [c for c in self.cases if c.kind == kind]


# [HARNESS:EVAL] Regression = a case that was reliably passing no longer is.
# WHY: an aggregate pass rate can stay flat while one case breaks and
# another gets fixed -- per-case transitions are what a reviewer acts on.
# pass -> fail and pass -> flaky are both regressions: "sometimes works" is
# a loss of reliability even if pass@k didn't move.
# INTERVIEW: "How do you catch regressions in an agent?" -> diff per-case
# verdicts between a baseline sweep and the candidate, flag pass->not-pass,
# and gate CI on it (especially for adversarial/security cases).
def classify(before: CaseVerdict | None, after: CaseVerdict | None) -> Transition:
    if before is None:
        return "added"
    if after is None:
        return "removed"
    if before == after:
        return "unchanged"
    if before == "pass":
        return "regression"
    if after == "pass":
        return "fix"
    return "changed"  # fail <-> flaky: movement, but not across the "reliable" line


def _case_rates(report: EvalReport) -> dict[str, float]:
    totals: dict[str, list[bool]] = {}
    for trial in report.trials:
        totals.setdefault(trial.case_id, []).append(trial.passed)
    return {case_id: sum(passed) / len(passed) for case_id, passed in totals.items()}


def _comparability(a: ReportMeta, b: ReportMeta) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    notes: list[str] = []
    if a.suite != b.suite:
        warnings.append(f"different suites ({a.suite} vs {b.suite}) -- only shared cases compared")
    if a.judged != b.judged:
        warnings.append("one sweep ran without the judge -- pass rates aren't comparable")
    elif a.judged and a.judge_prompt_versions != b.judge_prompt_versions:
        # The measuring instrument changed: a score shift may be the judge's.
        warnings.append(
            f"judge rubric changed ({', '.join(a.judge_prompt_versions)} -> "
            f"{', '.join(b.judge_prompt_versions)}) -- judge-driven changes may not be the agent's"
        )
    if a.judge_model != b.judge_model and a.judged and b.judged:
        warnings.append(f"judge model changed ({a.judge_model} -> {b.judge_model})")
    if a.mode != b.mode:
        warnings.append(f"different modes ({a.mode} vs {b.mode}) -- latency isn't comparable")
    if a.k != b.k:
        warnings.append(f"different k ({a.k} vs {b.k}) -- pass@k / pass^k aren't comparable")
    if a.model != b.model:
        notes.append(f"model: {a.model} -> {b.model}")
    if a.strategy != b.strategy:
        notes.append(f"strategy: {a.strategy} -> {b.strategy}")
    if a.prompt_versions != b.prompt_versions:
        notes.append(f"prompt: {', '.join(a.prompt_versions)} -> {', '.join(b.prompt_versions)}")
    if a.git_sha != b.git_sha:
        notes.append(f"git: {a.git_sha} -> {b.git_sha}")
    return warnings, notes


def compare(a: EvalReport, b: EvalReport) -> Comparison:
    rates_a, rates_b = _case_rates(a), _case_rates(b)
    verdicts_a, verdicts_b = a.metrics.case_verdicts, b.metrics.case_verdicts
    cases = [
        CaseDiff(
            case_id=case_id,
            kind=classify(verdicts_a.get(case_id), verdicts_b.get(case_id)),
            before=verdicts_a.get(case_id),
            after=verdicts_b.get(case_id),
            before_rate=rates_a.get(case_id),
            after_rate=rates_b.get(case_id),
            # The candidate's tags win (a case may have been re-tagged).
            tags=b.case_tags.get(case_id) or a.case_tags.get(case_id) or [],
        )
        for case_id in sorted(set(verdicts_a) | set(verdicts_b))
    ]
    ma, mb = a.metrics, b.metrics

    def stat(m: object, name: str, field: str) -> float | None:
        value = getattr(m, name)
        return None if value is None else float(getattr(value, field))

    deltas = [
        Delta(name="pass@1", unit="rate", before=ma.overall.pass_at_1, after=mb.overall.pass_at_1),
        Delta(name="pass@k", unit="rate", before=ma.overall.pass_at_k, after=mb.overall.pass_at_k),
        Delta(
            name="pass^k", unit="rate", before=ma.overall.pass_hat_k, after=mb.overall.pass_hat_k
        ),
        Delta(name="judge avg", unit="score", before=ma.judge_avg, after=mb.judge_avg),
        Delta(
            name="agent cost mean",
            unit="usd",
            before=stat(ma, "agent_cost", "mean"),
            after=stat(mb, "agent_cost", "mean"),
        ),
        Delta(
            name="agent cost p95",
            unit="usd",
            before=stat(ma, "agent_cost", "p95"),
            after=stat(mb, "agent_cost", "p95"),
        ),
        Delta(
            name="judge cost mean",
            unit="usd",
            before=stat(ma, "judge_cost", "mean"),
            after=stat(mb, "judge_cost", "mean"),
        ),
        Delta(
            name="tokens mean",
            unit="count",
            before=stat(ma, "tokens", "mean"),
            after=stat(mb, "tokens", "mean"),
        ),
        # None on either side (replay/record) -> shown as n/a, never as a delta.
        Delta(
            name="latency mean (s)",
            unit="seconds",
            before=stat(ma, "latency_s", "mean"),
            after=stat(mb, "latency_s", "mean"),
        ),
    ]
    warnings, notes = _comparability(a.meta, b.meta)
    for label, report in (("A", a), ("B", b)):
        if not report.case_tags:
            warnings.append(
                f"report {label} has no case tags (written before tags were stored) -- "
                "tag filters only see the other report's tags"
            )
    return Comparison(
        a=a.meta, b=b.meta, cases=cases, deltas=deltas, warnings=warnings, notes=notes
    )


def resolve_report(ref: str, reports_dir: Path = REPORTS_DIR) -> Path:
    """A report path, a sweep id (or prefix, >= 4 chars), `latest`, or `previous`.

    A sweep can have several reports (each `eval grade` writes one) -- the
    newest wins, since that's the current grading of that sweep.
    """
    path = Path(ref)
    if path.suffix == ".json" and path.exists():
        return path
    reports = sorted(reports_dir.glob("*.json"))  # timestamp-prefixed -> chronological
    if ref in ("latest", "previous"):
        index = -1 if ref == "latest" else -2
        if len(reports) < -index:
            raise FileNotFoundError(f"not enough reports in {reports_dir} for {ref!r}")
        return reports[index]
    if len(ref) < 4:
        raise ValueError(f"sweep id prefix {ref!r} is too short (use at least 4 characters)")
    matches = [r for r in reports if r.stem.rsplit("-", 1)[-1].startswith(ref[:8])]
    if not matches:
        raise FileNotFoundError(f"no report for {ref!r} in {reports_dir}")
    return matches[-1]


def load_comparison(ref_a: str, ref_b: str, reports_dir: Path = REPORTS_DIR) -> Comparison:
    return compare(
        load_report(resolve_report(ref_a, reports_dir)),
        load_report(resolve_report(ref_b, reports_dir)),
    )
