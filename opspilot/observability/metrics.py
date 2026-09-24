from pydantic import BaseModel

from opspilot.store.base import Store


class DashboardData(BaseModel):
    model_call_latency_percentiles: dict[str, float]
    tool_error_rates: dict[str, float]
    outcomes_distribution: dict[str, int]
    avg_cost_by_scenario: dict[str, float]


async def build_dashboard(store: Store) -> DashboardData:
    return DashboardData(
        model_call_latency_percentiles=await store.model_call_latency_percentiles(),
        tool_error_rates=await store.tool_error_rates(),
        outcomes_distribution=await store.outcomes_distribution(),
        avg_cost_by_scenario=await store.avg_cost_by_scenario(),
    )


class RunSummary(BaseModel):
    run_id: str
    scenario: str
    outcome: str | None
    steps: int | None
    tokens: dict[str, int]
    cost_usd: float | None
    duration_s: float | None


async def run_summary(store: Store, run_id: str) -> RunSummary | None:
    """Per-run totals for `opspilot trace <run_id>`'s header."""
    run = await store.get_run(run_id)
    if run is None:
        return None
    duration_s = None
    if run.finished_at is not None:
        duration_s = (run.finished_at - run.created_at).total_seconds()
    return RunSummary(
        run_id=run.run_id,
        scenario=run.scenario,
        outcome=run.outcome,
        steps=run.steps,
        tokens=run.tokens,
        cost_usd=run.cost_usd,
        duration_s=duration_s,
    )
