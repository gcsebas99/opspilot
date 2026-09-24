from pathlib import Path

import pytest
from pydantic import BaseModel

from opspilot.env.sandbox import Sandbox
from opspilot.tools.base import Tool, ToolRegistry, ToolResult


class _EchoInput(BaseModel):
    message: str


def _echo(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    assert isinstance(args, _EchoInput)
    return ToolResult(ok=True, content=args.message)


def _boom(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    raise RuntimeError("kaboom")


def _long_output(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    lines = [f"line {i}" for i in range(200)]
    return ToolResult(ok=True, content="\n".join(lines))


ECHO_TOOL = Tool(
    name="echo",
    description="Echoes the message back.",
    input_model=_EchoInput,
    risk="read",
    fn=_echo,
)
BOOM_TOOL = Tool(
    name="boom",
    description="Always raises.",
    input_model=_EchoInput,
    risk="read",
    fn=_boom,
)
LONG_TOOL = Tool(
    name="long_output",
    description="Returns 200 short lines.",
    input_model=_EchoInput,
    risk="read",
    fn=_long_output,
)


def _sandbox(tmp_path: Path) -> Sandbox:
    root = tmp_path / "sbx"
    root.mkdir()
    return Sandbox(root=root)


def test_execute_unknown_tool_returns_error_not_exception(tmp_path: Path) -> None:
    registry = ToolRegistry()
    result = registry.execute("does_not_exist", {}, _sandbox(tmp_path))
    assert result.ok is False
    assert "does_not_exist" in result.content


def test_execute_invalid_args_returns_model_readable_error(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ECHO_TOOL)

    result = registry.execute("echo", {"wrong_field": "x"}, _sandbox(tmp_path))

    assert result.ok is False
    assert "echo" in result.content
    assert "message" in result.content  # names the missing field


def test_execute_valid_args_runs_tool(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(ECHO_TOOL)

    result = registry.execute("echo", {"message": "hi"}, _sandbox(tmp_path))

    assert result.ok is True
    assert result.content == "hi"


def test_execute_tool_exception_becomes_error_result(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(BOOM_TOOL)

    result = registry.execute("boom", {"message": "irrelevant"}, _sandbox(tmp_path))

    assert result.ok is False
    assert "kaboom" in result.content
    assert "RuntimeError" in result.content


def test_execute_truncates_long_output_with_hint(tmp_path: Path) -> None:
    registry = ToolRegistry(max_output_chars=50)
    registry.register(LONG_TOOL)

    result = registry.execute("long_output", {"message": "irrelevant"}, _sandbox(tmp_path))

    assert result.ok is True
    assert result.truncated is True
    assert "more lines truncated" in result.content
    assert len(result.content) < 200  # far shorter than the untruncated 200-line output


def test_execute_short_output_is_not_marked_truncated(tmp_path: Path) -> None:
    registry = ToolRegistry(max_output_chars=4000)
    registry.register(ECHO_TOOL)

    result = registry.execute("echo", {"message": "hi"}, _sandbox(tmp_path))

    assert result.truncated is False


def test_register_duplicate_name_raises() -> None:
    registry = ToolRegistry()
    registry.register(ECHO_TOOL)
    with pytest.raises(ValueError, match="echo"):
        registry.register(ECHO_TOOL)


def test_to_anthropic_schema_shape() -> None:
    registry = ToolRegistry()
    registry.register(ECHO_TOOL)

    schema = registry.to_anthropic_schema()

    assert schema == [
        {
            "name": "echo",
            "description": "Echoes the message back.",
            "input_schema": _EchoInput.model_json_schema(),
        }
    ]
