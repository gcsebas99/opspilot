from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolCall
from langgraph.checkpoint.memory import InMemorySaver

from opspilot.config import Settings
from opspilot.env.sandbox import Sandbox
from opspilot.loops.graph import run_react_graph
from opspilot.observability.tracer import Tracer
from opspilot.store.memory import MemoryStore
from opspilot.tools.base import ToolRegistry

_REPORT_INPUT = {
    "root_cause": "config_change:checkout:db_pool_size",
    "evidence": ["checkout latency_p95_ms spiked after config v13"],
    "actions_taken": ["rollback_config(checkout, 12)"],
    "confidence": 0.9,
    "recommendation": "monitor for 30 minutes",
}


def _tool_call_message(
    name: str, args: dict, call_id: str = "call_1", usage: dict | None = None
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[ToolCall(name=name, args=args, id=call_id)],
        usage_metadata=usage or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def _parallel_tool_call_message(calls: list[tuple[str, dict, str]]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[ToolCall(name=name, args=args, id=call_id) for name, args, call_id in calls],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def _plain_text_message(text: str) -> AIMessage:
    return AIMessage(
        content=text, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )


def _tracer(run_id: str = "test-run") -> tuple[Tracer, MemoryStore]:
    store = MemoryStore()
    return Tracer(store, run_id=run_id), store


async def test_graph_reaches_submit_report(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = FakeMessagesListChatModel(
        responses=[_tool_call_message("submit_report", _REPORT_INPUT)]
    )
    tracer, store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="checkout latency spiking",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
    )

    assert result.outcome == "completed"
    assert result.steps == 1
    assert result.report is not None
    assert result.report["root_cause"] == "config_change:checkout:db_pool_size"

    spans = await store.list_spans("test-run")
    kinds = [s.kind for s in spans]
    assert "model_call" in kinds
    assert "policy_check" in kinds
    assert "tool_call" in kinds


async def test_graph_reaches_escalate(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = FakeMessagesListChatModel(
        responses=[_tool_call_message("escalate", {"reason": "disk full, no safe tool fixes it"})]
    )
    tracer, _store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="db disk full",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
    )

    assert result.outcome == "escalated"
    assert result.report == {"reason": "disk full, no safe tool fixes it"}


async def test_graph_max_steps(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    settings = settings.model_copy(update={"opspilot_max_steps": 2})
    # Fresh AIMessage per entry -- add_messages dedups by message .id, so
    # reusing one object across responses silently collapses repeated turns.
    model = FakeMessagesListChatModel(
        responses=[_tool_call_message("list_services", {}) for _ in range(3)]
    )
    tracer, _store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
    )

    assert result.outcome == "max_steps"
    assert result.steps == 2


async def test_graph_budget_exceeded(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    settings = settings.model_copy(update={"opspilot_token_budget": 100})
    expensive = AIMessage(
        content="thinking...",
        usage_metadata={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
    )
    model = FakeMessagesListChatModel(responses=[expensive])
    tracer, _store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
    )

    assert result.outcome == "budget_exceeded"
    assert result.steps == 1


async def test_graph_no_report_after_two_plain_text_turns(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = FakeMessagesListChatModel(
        responses=[_plain_text_message("I'm not sure what to do next.") for _ in range(2)]
    )
    tracer, _store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
    )

    assert result.outcome == "no_report"
    assert result.steps == 2


async def test_graph_stuck_detection(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = FakeMessagesListChatModel(
        responses=[
            _tool_call_message("grep_logs", {"service": "checkout", "pattern": "ERROR"})
            for _ in range(4)
        ]
    )
    tracer, _store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
    )

    assert result.outcome == "stuck"
    assert result.steps == 4


async def test_graph_parallel_tool_calls(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    parallel = _parallel_tool_call_message(
        [("list_services", {}, "a"), ("read_config", {"service": "checkout"}, "b")]
    )
    submit = _tool_call_message("submit_report", _REPORT_INPUT, call_id="c")
    model = FakeMessagesListChatModel(responses=[parallel, submit])
    tracer, _store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
    )

    assert result.outcome == "completed"
    assert [tc.name for tc in result.tool_calls[:2]] == ["list_services", "read_config"]


async def test_graph_destructive_tool_denied_without_allow_destructive(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    restart = _tool_call_message("restart_service", {"service": "checkout"}, call_id="a")
    escalate = _tool_call_message("escalate", {"reason": "denied"}, call_id="b")
    model = FakeMessagesListChatModel(responses=[restart, escalate])
    tracer, _store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
        allow_destructive=False,
    )

    assert result.tool_calls[0].name == "restart_service"
    assert result.tool_calls[0].ok is False
    assert "--allow-destructive" in result.tool_calls[0].content
    assert result.outcome == "escalated"


async def test_graph_destructive_tool_allowed_with_flag(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    restart = _tool_call_message("restart_service", {"service": "checkout"}, call_id="a")
    submit = _tool_call_message("submit_report", _REPORT_INPUT, call_id="b")
    model = FakeMessagesListChatModel(responses=[restart, submit])
    tracer, _store = _tracer()

    result = await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        tracer=tracer,
        checkpointer=InMemorySaver(),
        run_id="test-run",
        allow_destructive=True,
    )

    assert result.tool_calls[0].ok is True
    assert result.outcome == "completed"


async def test_graph_checkpointer_persists_thread_state(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = FakeMessagesListChatModel(
        responses=[_tool_call_message("submit_report", _REPORT_INPUT)]
    )
    tracer, _store = _tracer()
    checkpointer = InMemorySaver()

    await run_react_graph(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        tracer=tracer,
        checkpointer=checkpointer,
        run_id="thread-42",
    )

    checkpoints = list(checkpointer.list({"configurable": {"thread_id": "thread-42"}}))
    assert len(checkpoints) > 0
