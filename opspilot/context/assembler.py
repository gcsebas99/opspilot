import hashlib
import json
from pathlib import Path
from typing import Any

from opspilot.context.runbooks import RUNBOOK_INDEX

CONTEXT_DIR = Path(__file__).parent


def _read_agents_md(context_dir: Path) -> str:
    return (context_dir / "AGENTS.md").read_text()


def _render_runbook_index(runbook_index: dict[str, str]) -> str:
    lines = [f"- {name}: {description}" for name, description in sorted(runbook_index.items())]
    return "Available runbooks (load full text with load_runbook):\n" + "\n".join(lines)


def build_system_prompt(
    context_dir: Path = CONTEXT_DIR, runbook_index: dict[str, str] | None = None
) -> str:
    """Plain-text system prompt: AGENTS.md + the runbook index.

    [HARNESS:CONTEXT] Progressive disclosure of skills (runbooks).
    WHY: a runbook's full text only earns its place once the model has a hypothesis.
    The system prompt carries just the index (name + one line); the full text loads
    on demand via the load_runbook tool, so unused runbooks cost no tokens.
    """
    index = RUNBOOK_INDEX if runbook_index is None else runbook_index
    agents_md = _read_agents_md(context_dir)
    return f"{agents_md}\n\n{_render_runbook_index(index)}"


def build_system_blocks(
    context_dir: Path = CONTEXT_DIR, runbook_index: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """System prompt wrapped for the API with a prompt-caching breakpoint.

    [HARNESS:CONTEXT] Prompt-caching breakpoint on the last system block.
    WHY: requests render tools -> system -> messages, so one marker here caches the
    tool schemas and system prompt together. Caching is a prefix match: any byte
    change before the marker (timestamps, a reordered tool set) silently breaks it.
    """
    text = build_system_prompt(context_dir, runbook_index)
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def build_initial_messages(alert: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": f"New alert:\n{alert}"}]


# [HARNESS:CONTEXT] Prompt version: a hash of everything the agent is told.
# WHY: AGENTS.md, the runbook index and tool schemas define behavior; hashing them
# tags every run, audit entry and eval, so "did the prompt change?" is a string compare.
def compute_prompt_version(
    agents_md: str, runbook_index_text: str, tool_schemas: list[dict[str, Any]]
) -> str:
    """Pure hash function -- no file I/O, so it's trivial to test for sensitivity."""
    payload = agents_md + runbook_index_text + json.dumps(tool_schemas, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def prompt_version(
    context_dir: Path,
    tool_schemas: list[dict[str, Any]],
    runbook_index: dict[str, str] | None = None,
) -> str:
    """Reads the real on-disk context, then hashes it via compute_prompt_version."""
    index = RUNBOOK_INDEX if runbook_index is None else runbook_index
    return compute_prompt_version(
        _read_agents_md(context_dir), _render_runbook_index(index), tool_schemas
    )
