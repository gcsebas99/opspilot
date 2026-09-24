from opspilot.config import Settings
from opspilot.env.sandbox import Sandbox
from opspilot.loops.react_raw import run_react_loop
from opspilot.models.base import ModelResponse, TextBlock, ToolUseBlock, Usage
from opspilot.models.scripted import ScriptedModel
from opspilot.tools.base import ToolRegistry

_USAGE = Usage(input_tokens=10, output_tokens=5)


def _tool_use(name: str, input_: dict, tool_use_id: str = "t1") -> ModelResponse:
    return ModelResponse(
        content=[ToolUseBlock(id=tool_use_id, name=name, input=input_)],
        stop_reason="tool_use",
        usage=_USAGE,
        latency_ms=1.0,
    )


def _plain_text(text: str) -> ModelResponse:
    return ModelResponse(
        content=[TextBlock(text=text)], stop_reason="end_turn", usage=_USAGE, latency_ms=1.0
    )


_REPORT_INPUT = {
    "root_cause": "config_change:checkout:db_pool_size",
    "evidence": ["checkout latency_p95_ms spiked after config v13"],
    "actions_taken": ["rollback_config(checkout, 12)"],
    "confidence": 0.9,
    "recommendation": "monitor for 30 minutes",
}


# --- Exit condition #1: terminal tool called ---


async def test_submit_report_completes(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = ScriptedModel([_tool_use("submit_report", _REPORT_INPUT)])

    result = await run_react_loop(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="checkout latency spiking",
        settings=settings,
    )

    assert result.outcome == "completed"
    assert result.steps == 1
    assert result.report is not None
    assert result.report["root_cause"] == "config_change:checkout:db_pool_size"


async def test_escalate_escalates(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = ScriptedModel([_tool_use("escalate", {"reason": "disk full, no safe tool fixes it"})])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="db disk full", settings=settings
    )

    assert result.outcome == "escalated"
    assert result.steps == 1
    assert result.report == {"reason": "disk full, no safe tool fixes it"}


async def test_terminal_tool_with_invalid_args_does_not_end_the_run(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    # confidence=1.5 fails pydantic validation -> ok=False -> not terminal.
    bad_report = _tool_use("submit_report", {**_REPORT_INPUT, "confidence": 1.5})
    good_report = _tool_use("submit_report", _REPORT_INPUT)
    model = ScriptedModel([bad_report, good_report])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="x", settings=settings
    )

    assert result.outcome == "completed"
    assert result.steps == 2
    assert result.tool_calls[0].ok is False
    assert result.tool_calls[1].ok is True


# --- Exit condition #2: max_steps ---


async def test_max_steps_reached(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    settings = settings.model_copy(update={"opspilot_max_steps": 2})
    response = _tool_use("list_services", {})
    # Script more responses than max_steps allows -- the extras must never be consumed.
    model = ScriptedModel([response, response, response])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="x", settings=settings
    )

    assert result.outcome == "max_steps"
    assert result.steps == 2


# --- Exit condition #3: cumulative token budget ---


async def test_budget_exceeded(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    settings = settings.model_copy(update={"opspilot_token_budget": 100})
    expensive = ModelResponse(
        content=[TextBlock(text="thinking...")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1000, output_tokens=500),
        latency_ms=1.0,
    )
    model = ScriptedModel([expensive])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="x", settings=settings
    )

    assert result.outcome == "budget_exceeded"
    assert result.steps == 1


# --- Exit condition #4: stuck detection ---


async def test_stuck_detection_after_repeated_identical_calls(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    same_call = _tool_use("grep_logs", {"service": "checkout", "pattern": "ERROR"})
    model = ScriptedModel([same_call, same_call, same_call, same_call])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="x", settings=settings
    )

    assert result.outcome == "stuck"
    assert result.steps == 4

    # the nudge should have been injected before the 4th (final) call was made
    final_call_messages = model.calls[-1]["messages"]
    assert any("3 times in a row" in str(message) for message in final_call_messages)


async def test_stuck_detection_resets_on_different_args(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    call_a = _tool_use("grep_logs", {"service": "checkout", "pattern": "ERROR"})
    call_b = _tool_use("grep_logs", {"service": "checkout", "pattern": "WARN"})
    submit = _tool_use("submit_report", _REPORT_INPUT)
    # Alternating args should never trip stuck detection, however many turns.
    model = ScriptedModel([call_a, call_b, call_a, call_b, submit])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="x", settings=settings
    )

    assert result.outcome == "completed"
    assert result.steps == 5


# --- Exit condition #5: plain text, no tool call ---


async def test_no_report_after_two_consecutive_plain_text_turns(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    plain = _plain_text("I'm not sure what to do next.")
    model = ScriptedModel([plain, plain])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="x", settings=settings
    )

    assert result.outcome == "no_report"
    assert result.steps == 2
    # the nudge from turn 1 should be visible in turn 2's request
    second_call_messages = model.calls[1]["messages"]
    assert any("did not call a tool" in str(message) for message in second_call_messages)


async def test_plain_text_nudge_does_not_fire_if_tool_call_follows(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    plain = _plain_text("Let me think about this.")
    submit = _tool_use("submit_report", _REPORT_INPUT)
    model = ScriptedModel([plain, submit])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="x", settings=settings
    )

    assert result.outcome == "completed"
    assert result.steps == 2


# --- Parallel tool calls ---


async def test_parallel_tool_calls_all_execute_and_share_one_message(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    parallel = ModelResponse(
        content=[
            ToolUseBlock(id="a", name="list_services", input={}),
            ToolUseBlock(id="b", name="read_config", input={"service": "checkout"}),
        ],
        stop_reason="tool_use",
        usage=_USAGE,
        latency_ms=1.0,
    )
    submit = _tool_use("submit_report", _REPORT_INPUT, tool_use_id="c")
    model = ScriptedModel([parallel, submit])

    result = await run_react_loop(
        model=model, registry=registry, sandbox=sandbox, alert="x", settings=settings
    )

    assert result.outcome == "completed"
    assert [tc.name for tc in result.tool_calls[:2]] == ["list_services", "read_config"]

    second_call_messages = model.calls[1]["messages"]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    tool_use_ids = {block["tool_use_id"] for block in tool_result_message["content"]}
    assert tool_use_ids == {"a", "b"}


# --- Permission stub ---


async def test_destructive_tool_denied_without_allow_destructive(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    restart = _tool_use("restart_service", {"service": "checkout"}, tool_use_id="a")
    escalate = _tool_use("escalate", {"reason": "denied"}, tool_use_id="b")
    model = ScriptedModel([restart, escalate])

    result = await run_react_loop(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        allow_destructive=False,
    )

    assert result.tool_calls[0].name == "restart_service"
    assert result.tool_calls[0].ok is False
    assert "--allow-destructive" in result.tool_calls[0].content
    assert result.outcome == "escalated"


async def test_destructive_tool_allowed_with_flag(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    restart = _tool_use("restart_service", {"service": "checkout"}, tool_use_id="a")
    submit = _tool_use("submit_report", _REPORT_INPUT, tool_use_id="b")
    model = ScriptedModel([restart, submit])

    result = await run_react_loop(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        allow_destructive=True,
    )

    assert result.tool_calls[0].name == "restart_service"
    assert result.tool_calls[0].ok is True
    assert result.outcome == "completed"


# --- Events ---


async def test_events_are_emitted_for_step_model_call_tool_call_and_exit(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = ScriptedModel([_tool_use("submit_report", _REPORT_INPUT)])
    events = []

    await run_react_loop(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        on_event=events.append,
    )

    types = [event.type for event in events]
    assert types == ["step_start", "model_call", "tool_call", "exit"]


async def test_tool_call_event_carries_result_and_timing(
    sandbox: Sandbox, registry: ToolRegistry, settings: Settings
) -> None:
    model = ScriptedModel(
        [_tool_use("list_services", {}), _tool_use("submit_report", _REPORT_INPUT)]
    )
    events = []

    await run_react_loop(
        model=model,
        registry=registry,
        sandbox=sandbox,
        alert="x",
        settings=settings,
        on_event=events.append,
    )

    tool_call_events = [
        e for e in events if e.type == "tool_call" and e.data["name"] == "list_services"
    ]
    assert len(tool_call_events) == 1
    data = tool_call_events[0].data
    assert data["ok"] is True
    assert data["truncated"] is False
    assert data["output_size"] > 0
    assert data["duration_ms"] >= 0
