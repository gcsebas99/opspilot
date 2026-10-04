from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evals.metrics import Stats, SuiteMetrics, compute_metrics, trial_passed
from evals.models import EvalCase
from opspilot.store.models import EvalTrialDoc

REPORTS_DIR = Path("evals/reports")
SCHEMA_VERSION = 1


class ReportMeta(BaseModel):
    sweep_id: str
    suite: str
    created_at: datetime
    git_sha: str
    model: str
    strategy: str
    mode: str
    k: int
    judged: bool
    judge_model: str | None
    # Lists, not single values: a re-graded sweep or a mixed sweep could
    # carry several, and `compare` must see that rather than a silent pick.
    judge_prompt_versions: list[str]
    prompt_versions: list[str]


class TrialSummary(BaseModel):
    case_id: str
    trial: int
    passed: bool
    outcome: str | None
    error: str | None
    grading_error: str | None
    scores: dict[str, float]
    judge_avg: float | None
    cost_usd: float | None
    judge_cost_usd: float | None
    latency_s: float | None
    tokens: int | None
    failed_checks: list[str]


class EvalReport(BaseModel):
    """The durable record of one sweep -- what `eval compare` reads.

    Self-contained on purpose: the default store is in-memory, so a sweep's
    trials die with the process. This file is what outlives it.
    """

    schema_version: int = SCHEMA_VERSION
    meta: ReportMeta
    metrics: SuiteMetrics
    trials: list[TrialSummary]
    # Each case's tags at run time, so `compare --fail-on-tag adversarial`
    # works from the report alone. Defaults to {} so reports written before
    # this field existed still load (compare warns about them).
    case_tags: dict[str, list[str]] = {}


def failed_checks(grades: dict[str, Any] | None) -> list[str]:
    """Human-readable reasons a trial's grades failed, one line each."""
    failed: list[str] = []
    for name, grade in (grades or {}).items():
        if grade["passed"]:
            continue
        checks = [c for c in grade["details"].get("checks", []) if not c["passed"]]
        if checks:
            failed += [
                f"{name}: {c['name']}" + (f" ({c['detail']})" if c["detail"] else "")
                for c in checks
            ]
        else:
            detail = grade["details"].get("match") or grade["details"].get("error") or ""
            failed.append(f"{name}: {detail}".rstrip(": "))
    return failed


def _distinct(values: list[str | None]) -> list[str]:
    return sorted({v for v in values if v})


def _summarize(trial: EvalTrialDoc) -> TrialSummary:
    grades = trial.grades or {}
    judge = grades.get("judge", {}).get("details", {})
    return TrialSummary(
        case_id=trial.case_id,
        trial=trial.trial,
        passed=trial_passed(trial),
        outcome=trial.outcome,
        error=trial.error,
        grading_error=trial.grading_error,
        scores={name: grade["score"] for name, grade in grades.items()},
        judge_avg=judge.get("average"),
        cost_usd=trial.cost_usd,
        judge_cost_usd=judge.get("cost_usd"),
        latency_s=trial.latency_s,
        tokens=sum(trial.tokens.values()) if trial.tokens else None,
        failed_checks=failed_checks(trial.grades),
    )


def build_report(
    trials: list[EvalTrialDoc], cases: dict[str, EvalCase], *, suite: str, judged: bool
) -> EvalReport:
    if not trials:
        raise ValueError("cannot build a report from zero trials")
    first = trials[0]
    judge_details = [
        (t.grades or {})["judge"]["details"] for t in trials if "judge" in (t.grades or {})
    ]
    metrics = compute_metrics(trials, cases)
    return EvalReport(
        meta=ReportMeta(
            sweep_id=first.sweep_id,
            suite=suite,
            created_at=datetime.now(UTC),
            git_sha=first.git_sha,
            model=", ".join(_distinct([t.model for t in trials])),
            strategy=", ".join(_distinct([t.strategy for t in trials])),
            mode=", ".join(_distinct([t.mode for t in trials])),
            k=metrics.k,
            judged=judged,
            judge_model=", ".join(_distinct([d.get("judge_model") for d in judge_details])) or None,
            judge_prompt_versions=_distinct([d.get("judge_prompt_version") for d in judge_details]),
            prompt_versions=_distinct([t.prompt_version for t in trials]),
        ),
        metrics=metrics,
        trials=[_summarize(t) for t in sorted(trials, key=lambda t: (t.case_id, t.trial))],
        case_tags={
            case_id: list(cases[case_id].tags)
            for case_id in sorted({t.case_id for t in trials})
            if case_id in cases
        },
    )


# --- markdown ---


def _cell(text: object) -> str:
    # A literal "|" (e.g. inside a regex in a failed check) would split the cell.
    return str(text).replace("|", "\\|").replace("\n", " ")


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _money(stats: Stats | None, field: str) -> str:
    return "n/a" if stats is None else f"${getattr(stats, field):.4f}"


def _num(stats: Stats | None, field: str, fmt: str = ".0f") -> str:
    return "n/a" if stats is None else format(getattr(stats, field), fmt)


def render_markdown(report: EvalReport) -> str:
    meta, m = report.meta, report.metrics
    replay = meta.mode == "replay"
    lines = [
        f"# Eval report -- {meta.suite} ({meta.sweep_id[:8]})",
        "",
        f"- **created:** {meta.created_at:%Y-%m-%d %H:%M:%S} UTC  ",
        f"- **git:** `{meta.git_sha}`  **prompt:** `{', '.join(meta.prompt_versions)}`",
        f"- **model:** `{meta.model}`  **strategy:** {meta.strategy}  **mode:** {meta.mode}",
        "- **judge:** "
        + (
            f"`{meta.judge_model}` (rubric `{', '.join(meta.judge_prompt_versions)}`)"
            if meta.judged
            else "off -- pass/fail reflects deterministic graders only"
        ),
        f"- **sweep:** `{meta.sweep_id}`  **k:** {meta.k}",
        "",
        "## Summary",
        "",
        "| pass@1 | pass@k | pass^k | cases | trials | errored |",
        "|---|---|---|---|---|---|",
        f"| {_pct(m.overall.pass_at_1)} | {_pct(m.overall.pass_at_k)} | "
        f"{_pct(m.overall.pass_hat_k)} | {m.overall.cases} | {m.overall.trials} | {m.errored} |",
        "",
    ]
    if m.k == 1:
        lines += [
            "_k=1: pass@1, pass@k and pass^k are the same number. Use `--k 3` for variance._",
            "",
        ]

    lines += [
        "## Cost, tokens, latency",
        "",
        "| | mean | p95 | total |",
        "|---|---|---|---|",
        f"| agent cost{' (recorded)¹' if replay else ''} | {_money(m.agent_cost, 'mean')} | "
        f"{_money(m.agent_cost, 'p95')} | {_money(m.agent_cost, 'total')} |",
        f"| judge cost{' (recorded)¹' if replay else ''} | {_money(m.judge_cost, 'mean')} | "
        f"{_money(m.judge_cost, 'p95')} | {_money(m.judge_cost, 'total')} |",
        f"| tokens / trial | {_num(m.tokens, 'mean')} | {_num(m.tokens, 'p95')} | "
        f"{_num(m.tokens, 'total')} |",
        f"| latency (s)² | {_num(m.latency_s, 'mean', '.1f')} | "
        f"{_num(m.latency_s, 'p95', '.1f')} | -- |",
        "",
    ]
    if m.judge_avg is not None:
        lines += [f"Mean judge score: **{m.judge_avg:.2f}** / 5", ""]

    lines += ["## By grader", "", "| grader | pass rate |", "|---|---|"]
    lines += [f"| {name} | {_pct(rate)} |" for name, rate in m.per_grader.items()]
    lines += [
        "",
        "## By tag",
        "",
        "| tag | cases | pass@1 | pass@k | pass^k |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {tag} | {r.cases} | {_pct(r.pass_at_1)} | {_pct(r.pass_at_k)} | {_pct(r.pass_hat_k)} |"
        for tag, r in m.per_tag.items()
    ]

    lines += [
        "",
        "## Cases",
        "",
        "| case | verdict | passed | judge | cost | why it failed |",
        "|---|---|---|---|---|---|",
    ]
    by_case: dict[str, list[TrialSummary]] = {}
    for trial in report.trials:
        by_case.setdefault(trial.case_id, []).append(trial)
    for case_id, trials in by_case.items():
        judge = [t.judge_avg for t in trials if t.judge_avg is not None]
        costs = [t.cost_usd for t in trials if t.cost_usd is not None]
        reasons: list[str] = []
        for t in trials:
            for reason in [t.error, t.grading_error, *t.failed_checks]:
                if reason and reason not in reasons:
                    reasons.append(reason)
        lines.append(
            f"| {_cell(case_id)} | {m.case_verdicts.get(case_id, '?')} | "
            f"{sum(t.passed for t in trials)}/{len(trials)} | "
            f"{f'{sum(judge) / len(judge):.1f}' if judge else '--'} | "
            f"{f'${sum(costs) / len(costs):.4f}' if costs else '--'} | "
            f"{_cell('; '.join(reasons)) or ''} |"
        )

    lines += [
        "",
        "---",
        "pass@1 = fraction of trials passed. pass@k = fraction of cases passed in at least one "
        "of k trials (capability). pass^k = fraction of cases passed in all k (reliability). "
        "A case is `flaky` when some but not all trials passed. Errored trials count as failures.",
        "",
    ]
    if replay:
        lines.append(
            "¹ Replay: the cost of the calls when they were recorded -- this sweep spent $0.  "
        )
    lines.append(
        "² Live trials only. Replayed or cassette-served calls measure disk reads, not the model."
    )
    return "\n".join(lines) + "\n"


def write_report(report: EvalReport, reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.meta.created_at:%Y%m%dT%H%M%SZ}-{report.meta.suite}-{report.meta.sweep_id[:8]}"
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    json_path.write_text(report.model_dump_json(indent=2) + "\n")
    md_path.write_text(render_markdown(report))
    return json_path, md_path


def load_report(path: Path) -> EvalReport:
    return EvalReport.model_validate_json(path.read_text())
