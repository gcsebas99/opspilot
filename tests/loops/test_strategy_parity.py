"""Raw loop and graph (via ModelClientChatModel) must send identical requests.

This is what lets one cassette serve both strategies, and what makes the
raw-vs-graph ablation (3.6) compare loops rather than message plumbing.
"""

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from opspilot.config import Settings
from opspilot.env.generator import build_sandbox
from opspilot.env.scenarios import get_scenario
from opspilot.loops.graph import run_react_graph
from opspilot.loops.react_raw import run_react_loop
from opspilot.models.base import ModelResponse, TextBlock, ToolUseBlock, Usage
from opspilot.models.cassette import normalize_request
from opspilot.models.langchain_adapter import ModelClientChatModel
from opspilot.models.scripted import ScriptedModel
from opspilot.observability.tracer import Tracer
from opspilot.store.memory import MemoryStore
from opspilot.tools.base import ToolRegistry

_USAGE = Usage(
    input_tokens=100, output_tokens=10, cache_read_input_tokens=900, cache_creation_input_tokens=5
)


def _turn(*blocks: TextBlock | ToolUseBlock) -> ModelResponse:
    stop = "tool_use" if any(isinstance(b, ToolUseBlock) for b in blocks) else "end_turn"
    return ModelResponse(content=list(blocks), stop_reason=stop, usage=_USAGE, latency_ms=1.0)


def _script() -> list[ModelResponse]:
    read_config = {"service": "checkout"}
    return [
        _turn(
            TextBlock(text="Let me look."), ToolUseBlock(id="t1", name="list_services", input={})
        ),
        # parallel tool calls -> one merged user turn
        _turn(
            ToolUseBlock(id="t2", name="config_history", input={"service": "checkout"}),
            ToolUseBlock(
                id="t3", name="grep_logs", input={"service": "checkout", "pattern": "timeout"}
            ),
        ),
        _turn(TextBlock(text="Thinking out loud, no tool.")),  # -> no-tool-call nudge
        _turn(ToolUseBlock(id="t4", name="read_config", input=read_config)),
        _turn(ToolUseBlock(id="t5", name="read_config", input=read_config)),
        _turn(ToolUseBlock(id="t6", name="read_config", input=read_config)),  # -> stuck nudge
        _turn(
            ToolUseBlock(
                id="t7",
                name="submit_report",
                input={
                    "root_cause": "config_change:checkout:db_pool_size",
                    "evidence": ["pool size dropped"],
                    "actions_taken": [],
                    "confidence": 0.8,
                    "recommendation": "roll back config",
                },
            )
        ),
    ]


def _normalized(model: str, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_request(model, c["system"], c["messages"], c["tools"]) for c in calls]


async def test_raw_and_graph_send_identical_requests(
    tmp_path: Path, registry: ToolRegistry, settings: Settings
) -> None:
    scenario = get_scenario("checkout_pool_exhaustion")

    raw_client = ScriptedModel(_script())
    raw_result = await run_react_loop(
        model=raw_client,
        registry=registry,
        sandbox=build_sandbox(scenario, seed=42, root=tmp_path / "raw"),
        alert=scenario.alert_text,
        settings=settings,
    )

    graph_client = ScriptedModel(_script())
    store = MemoryStore()
    graph_result = await run_react_graph(
        model=ModelClientChatModel(client=graph_client).bind_tools(registry.to_anthropic_schema()),
        registry=registry,
        sandbox=build_sandbox(scenario, seed=42, root=tmp_path / "graph"),
        alert=scenario.alert_text,
        settings=settings,
        tracer=Tracer(store, run_id="parity"),
        store=store,
        checkpointer=InMemorySaver(),
        run_id="parity",
        prompt_version="v1",
    )

    assert raw_result.outcome == graph_result.outcome == "completed"
    assert len(raw_client.calls) == len(graph_client.calls) == 7
    # Compare full requests (not just hashes) so a failure shows the diff.
    assert _normalized("m", graph_client.calls) == _normalized("m", raw_client.calls)
    # Same usage in -> same token accounting out (guards the graph's
    # LangChain total-vs-uncached input_tokens conversion).
    assert graph_result.tokens == raw_result.tokens
