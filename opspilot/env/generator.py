import json
import random
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from opspilot.env.sandbox import Sandbox
from opspilot.env.scenarios import Scenario

SERVICES = ["web", "checkout", "payments", "inventory", "db"]
METRICS = ["latency_p95_ms", "error_rate", "cpu_pct", "mem_mb", "db_pool_in_use", "disk_pct"]
WINDOW_MINUTES = 60

# Fixed epoch, not wall-clock time -- see [HARNESS:ENV] note on build_sandbox below.
EPOCH = datetime(2025, 1, 1, tzinfo=UTC)

BASELINE: dict[str, float] = {
    "latency_p95_ms": 150.0,
    "error_rate": 0.002,
    "cpu_pct": 35.0,
    "mem_mb": 512.0,
    "db_pool_in_use": 8.0,
    "disk_pct": 45.0,
}
NOISE_FRACTION: dict[str, float] = {
    "latency_p95_ms": 0.10,
    "error_rate": 0.20,
    "cpu_pct": 0.15,
    "mem_mb": 0.05,
    "db_pool_in_use": 0.15,
    "disk_pct": 0.03,
}

SERVICE_DEFAULT_CONFIG: dict[str, dict[str, int]] = {
    "web": {"timeout_ms": 5000, "max_connections": 200},
    "checkout": {"db_pool_size": 50, "timeout_ms": 3000},
    "payments": {"retry_count": 3, "timeout_ms": 4000},
    "inventory": {"cache_ttl_s": 60, "db_pool_size": 20},
    "db": {"max_connections": 100, "disk_quota_gb": 500},
}

NOISE_LOG_TEMPLATES = [
    "INFO request completed in {ms}ms",
    "INFO health check ok",
    "DEBUG cache hit ratio {pct}%",
]

# {service: {metric: [value_at_minute_0, value_at_minute_1, ...]}}, WINDOW_MINUTES long.
# e.g. {"checkout": {"latency_p95_ms": [148.2, 151.0, ..., 3400.1], "error_rate": [0.002, ...]}}
MetricSeries = dict[str, dict[str, list[float]]]

# {service: [raw log line, ...]}, sorted by timestamp before being written to disk.
# e.g. {"checkout": ["2025-01-01T00:57:36Z checkout ERROR connection pool timeout: ..."]}
LogLines = dict[str, list[str]]

# {service: [(version, config_dict), ...]}, ordered oldest -> newest; last entry is current.
# e.g. {"checkout": [(12, {"db_pool_size": 50, "timeout_ms": 3000}),
#                     (13, {"db_pool_size": 5, "timeout_ms": 3000})]}
ConfigVersions = dict[str, list[tuple[int, dict[str, int]]]]

# [{"service": ..., "version": ..., "ts": ..., "author": ...}, ...] -- matches deploys.json.
# e.g. [{"service": "payments", "version": "2.3.1", "ts": "2025-01-01T00:35:00Z",
#         "author": "alice"}]
Deploys = list[dict[str, str]]

# {service: {field: value}}, currently just {"status": "healthy"|...} -- matches state.json.
# e.g. {"checkout": {"status": "healthy"}, "payments": {"status": "healthy"}}
State = dict[str, dict[str, str]]

IMPLEMENTED_SCENARIOS = {"checkout_pool_exhaustion", "payments_bad_deploy", "false_alarm"}


def _fmt_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _jitter(rng: random.Random, base: float, frac: float) -> float:
    return round(max(0.0, base * (1 + rng.uniform(-frac, frac))), 3)


def _baseline_series(rng: random.Random) -> dict[str, list[float]]:
    return {
        metric: [
            _jitter(rng, BASELINE[metric], NOISE_FRACTION[metric]) for _ in range(WINDOW_MINUTES)
        ]
        for metric in METRICS
    }


def _noise_log_lines(rng: random.Random, service: str) -> list[str]:
    lines = []
    for t in range(WINDOW_MINUTES):
        if rng.random() < 0.5:
            continue
        ts = _fmt_ts(EPOCH + timedelta(minutes=t, seconds=rng.randint(0, 59)))
        template = rng.choice(NOISE_LOG_TEMPLATES)
        msg = template.format(ms=rng.randint(20, 180), pct=rng.randint(70, 99))
        lines.append(f"{ts} {service} {msg}")
    return lines


def _noise_deploys(rng: random.Random) -> Deploys:
    boring_versions = {"web": "4.1.0", "inventory": "1.9.2", "db": "9.0.1"}
    authors = ["alice", "bob", "carol"]
    deploys: Deploys = []
    for service, version in boring_versions.items():
        minute = rng.randint(0, 20)
        deploys.append(
            {
                "service": service,
                "version": version,
                "ts": _fmt_ts(EPOCH + timedelta(minutes=minute)),
                "author": rng.choice(authors),
            }
        )
    return deploys


def _build_checkout_pool_exhaustion(
    rng: random.Random,
    metric_series: MetricSeries,
    log_lines: LogLines,
    config_versions: ConfigVersions,
    deploys: Deploys,
    state: State,
) -> None:
    spike_start = 40
    for t in range(spike_start, WINDOW_MINUTES):
        ramp = (t - spike_start) / max(1, WINDOW_MINUTES - spike_start)
        metric_series["checkout"]["latency_p95_ms"][t] = _jitter(rng, 150 + ramp * 3500, 0.05)
        metric_series["checkout"]["db_pool_in_use"][t] = 5.0
        ts = _fmt_ts(EPOCH + timedelta(minutes=t, seconds=rng.randint(0, 59)))
        log_lines["checkout"].append(
            f"{ts} checkout ERROR connection pool timeout: could not acquire connection"
        )
    config_versions["checkout"] = [
        (12, {"db_pool_size": 50, "timeout_ms": 3000}),
        (13, {"db_pool_size": 5, "timeout_ms": 3000}),
    ]


def _build_payments_bad_deploy(
    rng: random.Random,
    metric_series: MetricSeries,
    log_lines: LogLines,
    config_versions: ConfigVersions,
    deploys: Deploys,
    state: State,
) -> None:
    deploy_minute = 35
    deploys.append(
        {
            "service": "payments",
            "version": "2.3.1",
            "ts": _fmt_ts(EPOCH + timedelta(minutes=deploy_minute)),
            "author": "alice",
        }
    )
    for t in range(deploy_minute, WINDOW_MINUTES):
        metric_series["payments"]["error_rate"][t] = _jitter(rng, 0.18, 0.05)
        if rng.random() < 0.6:
            ts = _fmt_ts(EPOCH + timedelta(minutes=t, seconds=rng.randint(0, 59)))
            log_lines["payments"].append(
                f"{ts} payments ERROR payment gateway timeout after deploy 2.3.1"
            )


def _build_false_alarm(
    rng: random.Random,
    metric_series: MetricSeries,
    log_lines: LogLines,
    config_versions: ConfigVersions,
    deploys: Deploys,
    state: State,
) -> None:
    # A single-minute blip that's already resolved by the time the agent looks --
    # no config/deploy/resource change, nothing to roll back or restart.
    blip_minute = 10
    metric_series["checkout"]["error_rate"][blip_minute] = _jitter(rng, 0.03, 0.1)


_SCENARIO_BUILDERS: dict[
    str, Callable[[random.Random, MetricSeries, LogLines, ConfigVersions, Deploys, State], None]
] = {
    "checkout_pool_exhaustion": _build_checkout_pool_exhaustion,
    "payments_bad_deploy": _build_payments_bad_deploy,
    "false_alarm": _build_false_alarm,
}


def _write_logs(sandbox: Sandbox, log_lines: LogLines) -> None:
    for service, lines in log_lines.items():
        sandbox.path(f"logs/{service}.log").write_text("\n".join(sorted(lines)) + "\n")


def _write_configs(sandbox: Sandbox, config_versions: ConfigVersions) -> None:
    for service, history in config_versions.items():
        hist_dir = sandbox.path(f"config/history/{service}")
        hist_dir.mkdir(parents=True, exist_ok=True)
        for version, cfg in history:
            content = yaml.safe_dump({"version": version, **cfg}, sort_keys=False)
            (hist_dir / f"v{version}.yaml").write_text(content)
        current_version, current_cfg = history[-1]
        current_content = yaml.safe_dump(
            {"version": current_version, **current_cfg}, sort_keys=False
        )
        sandbox.path(f"config/{service}.yaml").write_text(current_content)


def _write_metrics_db(sandbox: Sandbox, metric_series: MetricSeries) -> None:
    db_path = sandbox.path("metrics.db")
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE metrics (ts TEXT NOT NULL, service TEXT NOT NULL, "
            "metric TEXT NOT NULL, value REAL NOT NULL)"
        )
        rows = [
            (_fmt_ts(EPOCH + timedelta(minutes=t)), service, metric, value)
            for service, series in metric_series.items()
            for metric, values in series.items()
            for t, value in enumerate(values)
        ]
        conn.executemany(
            "INSERT INTO metrics (ts, service, metric, value) VALUES (?, ?, ?, ?)", rows
        )
        conn.commit()
    finally:
        conn.close()


def _write_deploys(sandbox: Sandbox, deploys: Deploys) -> None:
    ordered = sorted(deploys, key=lambda d: d["ts"])
    sandbox.path("deploys.json").write_text(json.dumps(ordered, indent=2) + "\n")


def _write_state(sandbox: Sandbox, state: State) -> None:
    sandbox.path("state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


# [HARNESS:ENV] Reproducible -- a single seeded RNG instance and a fixed epoch,
# no global `random` state and no wall-clock time anywhere in this module.
# WHY: evals (Day 3) replay the exact same sandbox bytes across runs and across
# machines; a live clock or `random.seed()` (global, mutable, easy to leak
# between scenarios run in the same process) would make runs non-reproducible
# and make "same seed -> same file hashes" tests flaky.
# INTERVIEW: "How do you make a generated test environment reproducible?" ->
# thread one `random.Random(seed)` through every call site, never `import
# random; random.choice(...)`, and never derive content from `datetime.now()`.
def build_sandbox(scenario: Scenario, seed: int, root: Path) -> Sandbox:
    if scenario.name not in IMPLEMENTED_SCENARIOS:
        raise NotImplementedError(
            f"scenario {scenario.name!r} is not implemented yet "
            "(see docs/specs/day1.md 1.2 -- finish it Day 3)"
        )

    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(root=root)
    for rel in ("logs", "config", "config/history"):
        sandbox.path(rel).mkdir(parents=True, exist_ok=True)

    metric_series: MetricSeries = {service: _baseline_series(rng) for service in SERVICES}
    log_lines: LogLines = {service: _noise_log_lines(rng, service) for service in SERVICES}
    config_versions: ConfigVersions = {
        service: [(1, dict(SERVICE_DEFAULT_CONFIG[service]))] for service in SERVICES
    }
    deploys = _noise_deploys(rng)
    state: State = {service: {"status": "healthy"} for service in SERVICES}

    _SCENARIO_BUILDERS[scenario.name](
        rng, metric_series, log_lines, config_versions, deploys, state
    )

    _write_logs(sandbox, log_lines)
    _write_configs(sandbox, config_versions)
    _write_metrics_db(sandbox, metric_series)
    _write_deploys(sandbox, deploys)
    _write_state(sandbox, state)

    return sandbox
