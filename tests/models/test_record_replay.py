from pathlib import Path
from typing import Any

import pytest

from opspilot.models.base import ModelResponse, TextBlock, ToolUseBlock, Usage
from opspilot.models.cassette import Cassette, CassetteMiss, normalize_request, request_key
from opspilot.models.recording import RecordingModel
from opspilot.models.replay import ReplayModel
from opspilot.models.scripted import ScriptedModel

MODEL = "claude-test"
TOOLS = [{"name": "get_logs", "description": "d", "input_schema": {"type": "object"}}]


def _text(text: str) -> ModelResponse:
    return ModelResponse(
        content=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
        latency_ms=123.0,
    )


def _conversation(tool_id: str, tool_output: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "alert: checkout 5xx"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": tool_id, "name": "get_logs", "input": {"s": "x"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": tool_output}],
        },
    ]


def _key(messages: list[dict[str, Any]], system: str = "sys") -> str:
    return request_key(normalize_request(MODEL, system, messages, TOOLS))


def test_key_ignores_tool_use_ids_uuids_and_restart_timestamps() -> None:
    a = _conversation(
        "toolu_AAA",
        'run 1b4e28ba-2fa1-11d2-883f-0016d3cca427 {"restarted_at": "2026-09-28T10:00:00Z"}',
    )
    b = _conversation(
        "toolu_BBB",
        'run 6fa459ea-ee8a-3ca4-894e-db77e160355e {"restarted_at": "2026-09-29T11:11:11Z"}',
    )
    assert _key(a) == _key(b)


def test_key_changes_when_prompt_or_content_changes() -> None:
    base = _conversation("toolu_A", "line 1")
    assert _key(base) != _key(base, system="different system prompt")
    assert _key(base) != _key(_conversation("toolu_A", "line 2"))
    # Seeded log timestamps are real content, not noise -- they must not be masked.
    assert _key(_conversation("t", "2026-01-01T00:00:00Z err")) != _key(
        _conversation("t", "2026-01-01T00:05:00Z err")
    )


async def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "case" / "0.jsonl"
    messages = _conversation("toolu_1", "pool exhausted")
    recorder = RecordingModel(ScriptedModel([_text("root cause: pool")]), Cassette(path), MODEL)
    recorded = await recorder.create(system="sys", messages=messages, tools=TOOLS)

    replay = ReplayModel(Cassette(path), MODEL)  # fresh load from disk
    replayed = await replay.create(system="sys", messages=messages, tools=TOOLS)

    assert replayed == recorded
    assert replayed.latency_ms == 123.0  # recorded latency, not replay wall-clock


async def test_recording_serves_hits_without_calling_inner(tmp_path: Path) -> None:
    path = tmp_path / "0.jsonl"
    inner = ScriptedModel([_text("only one scripted response")])
    recorder = RecordingModel(inner, Cassette(path), MODEL)
    messages = [{"role": "user", "content": "hi"}]

    await recorder.create(system="sys", messages=messages, tools=TOOLS)
    await recorder.create(system="sys", messages=messages, tools=TOOLS)  # would exhaust inner

    assert (recorder.hits, recorder.misses) == (1, 1)
    assert len(inner.calls) == 1
    assert len(path.read_text().splitlines()) == 1


async def test_replay_miss_fails_loudly_and_names_first_diff(tmp_path: Path) -> None:
    path = tmp_path / "0.jsonl"
    recorder = RecordingModel(ScriptedModel([_text("ok")]), Cassette(path), MODEL)
    await recorder.create(system="sys", messages=_conversation("t", "line 1"), tools=TOOLS)

    replay = ReplayModel(Cassette(path), MODEL)
    with pytest.raises(CassetteMiss, match=r"messages\[2\]\.content\[0\]\.content"):
        await replay.create(system="sys", messages=_conversation("t", "line 2"), tools=TOOLS)


async def test_replay_miss_on_model_change(tmp_path: Path) -> None:
    path = tmp_path / "0.jsonl"
    messages = [{"role": "user", "content": "hi"}]
    await RecordingModel(ScriptedModel([_text("ok")]), Cassette(path), MODEL).create(
        system="sys", messages=messages, tools=TOOLS
    )
    with pytest.raises(CassetteMiss, match=r"\.model"):
        await ReplayModel(Cassette(path), "claude-other").create(
            system="sys", messages=messages, tools=TOOLS
        )


async def test_replay_missing_cassette_says_record_first(tmp_path: Path) -> None:
    replay = ReplayModel(Cassette(tmp_path / "nope.jsonl"), MODEL)
    with pytest.raises(CassetteMiss, match="record it first"):
        await replay.create(system="sys", messages=[], tools=TOOLS)


async def test_one_cassette_holds_two_branches(tmp_path: Path) -> None:
    """Approve and reject paths share a prefix then diverge -- both replay."""
    path = tmp_path / "0.jsonl"
    approve = [*_conversation("t", "awaiting approval"), {"role": "user", "content": "approved"}]
    reject = [*_conversation("t", "awaiting approval"), {"role": "user", "content": "rejected"}]
    recorder = RecordingModel(
        ScriptedModel([_text("rolled back"), _text("escalating")]), Cassette(path), MODEL
    )
    await recorder.create(system="sys", messages=approve, tools=TOOLS)
    await recorder.create(system="sys", messages=reject, tools=TOOLS)

    replay = ReplayModel(Cassette(path), MODEL)
    assert (await replay.create(system="sys", messages=approve, tools=TOOLS)).content == [
        TextBlock(text="rolled back")
    ]
    assert (await replay.create(system="sys", messages=reject, tools=TOOLS)).content == [
        TextBlock(text="escalating")
    ]


async def test_tool_use_blocks_survive_serialization(tmp_path: Path) -> None:
    path = tmp_path / "0.jsonl"
    response = ModelResponse(
        content=[ToolUseBlock(id="toolu_x", name="get_logs", input={"service": "checkout"})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1, cache_read_input_tokens=7),
        latency_ms=1.0,
    )
    messages = [{"role": "user", "content": "hi"}]
    await RecordingModel(ScriptedModel([response]), Cassette(path), MODEL).create(
        system="sys", messages=messages, tools=TOOLS
    )
    replayed = await ReplayModel(Cassette(path), MODEL).create(
        system="sys", messages=messages, tools=TOOLS
    )
    assert replayed == response


def test_tool_choice_only_keys_the_hash_when_set() -> None:
    messages = [{"role": "user", "content": "hi"}]
    plain = request_key(normalize_request(MODEL, "sys", messages, TOOLS))
    explicit_none = request_key(normalize_request(MODEL, "sys", messages, TOOLS, None))
    forced = request_key(
        normalize_request(MODEL, "sys", messages, TOOLS, {"type": "tool", "name": "get_logs"})
    )
    # Pre-existing cassettes (recorded without tool_choice) keep their keys.
    assert plain == explicit_none
    assert forced != plain


async def test_tool_choice_reaches_inner_client_and_replays(tmp_path: Path) -> None:
    path = tmp_path / "0.jsonl"
    choice = {"type": "tool", "name": "get_logs"}
    inner = ScriptedModel([_text("forced")])
    await RecordingModel(inner, Cassette(path), MODEL).create(
        system="sys", messages=[], tools=TOOLS, tool_choice=choice
    )
    assert inner.calls[0]["tool_choice"] == choice

    replay = ReplayModel(Cassette(path), MODEL)
    assert (await replay.create(system="sys", messages=[], tools=TOOLS, tool_choice=choice)).content
    with pytest.raises(CassetteMiss, match=r"\.tool_choice"):
        await replay.create(system="sys", messages=[], tools=TOOLS)
