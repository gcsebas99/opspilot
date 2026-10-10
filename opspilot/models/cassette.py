import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from opspilot.models.base import ModelResponse

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# restart_service stamps wall-clock time into state.json; a later status read
# carries it back into the conversation. Seeded log timestamps are NOT masked
# -- they're deterministic and part of what the model is reasoning about.
_RESTARTED_AT_RE = re.compile(r'("restarted_at":\s*)"[^"]*"')


class CassetteMiss(LookupError):
    """A request had no recorded response -- the prompt/loop changed since recording."""


class CassetteEntry(BaseModel):
    key: str
    request: dict[str, Any]
    response: ModelResponse
    recorded_at: datetime


def _collect_tool_use_ids(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map every tool_use id to a positional placeholder, in first-seen order."""
    mapping: dict[str, str] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                mapping.setdefault(block["id"], f"toolu_{len(mapping)}")
    return mapping


def _normalize_value(value: Any, ids: dict[str, str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in ("id", "tool_use_id") and isinstance(v, str) and v in ids:
                out[k] = ids[v]
            else:
                out[k] = _normalize_value(v, ids)
        return out
    if isinstance(value, list):
        return [_normalize_value(v, ids) for v in value]
    if isinstance(value, str):
        value = _UUID_RE.sub("<uuid>", value)
        return _RESTARTED_AT_RE.sub(r'\1"<ts>"', value)
    return value


# [HARNESS:EVAL] Request normalization: mask volatile fields before hashing.
# WHY: the key must change when the *prompt* changes, never for noise (random
# tool_use ids, UUIDs, wall-clock restart times). Too little masking -> spurious
# misses; too much -> real prompt changes slip through.
def normalize_request(
    model: str,
    system: list[dict[str, Any]] | str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = _collect_tool_use_ids(messages)
    request: dict[str, Any] = {
        "model": model,
        "system": system,
        "messages": messages,
        "tools": tools,
    }
    # Only keyed when set, so every cassette recorded before tool_choice
    # existed (all agent calls) keeps its hash.
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    normalized: dict[str, Any] = _normalize_value(request, ids)
    return normalized


def request_key(normalized: dict[str, Any]) -> str:
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _first_diff(a: Any, b: Any, path: str = "") -> str | None:
    """Path of the first field where `a` and `b` differ, or None if equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}"
            found = _first_diff(a[k], b[k], f"{path}.{k}")
            if found is not None:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            found = _first_diff(x, y, f"{path}[{i}]")
            if found is not None:
                return found
        return f"{path}[{min(len(a), len(b))}]" if len(a) != len(b) else None
    return None if a == b else path or "<root>"


def _common_prefix_len(a: list[Any], b: list[Any]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


class Cassette:
    """A JSONL file of recorded model calls, looked up by request hash.

    Hash lookup (not call order) means one cassette can hold several
    branches of the same run -- e.g. the approve and the reject path of a
    HITL checkpoint share every request up to the decision, then diverge
    into distinct keys.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, CassetteEntry] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    entry = CassetteEntry.model_validate_json(line)
                    # First write wins: an identical request is an identical
                    # agent state, so serving the first response is exactly
                    # what "deterministic replay" means.
                    self._entries.setdefault(entry.key, entry)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> CassetteEntry | None:
        return self._entries.get(key)

    def append(self, request: dict[str, Any], response: ModelResponse) -> CassetteEntry:
        key = request_key(request)
        entry = CassetteEntry(
            key=key, request=request, response=response, recorded_at=datetime.now(UTC)
        )
        if key not in self._entries:
            self._entries[key] = entry
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(entry.model_dump_json() + "\n")
        return self._entries[key]

    def explain_miss(self, request: dict[str, Any]) -> str:
        """Human-readable reason a request missed: the closest recorded
        request (longest shared message prefix) and the first field that differs."""
        if not self._entries:
            return f"cassette {self.path} is empty or missing -- record it first"
        messages = request.get("messages", [])
        closest = max(
            self._entries.values(),
            key=lambda e: _common_prefix_len(e.request.get("messages", []), messages),
        )
        diff = _first_diff(closest.request, request)
        return (
            f"no recorded response in {self.path} ({len(self)} entries); "
            f"closest recorded request first differs at {diff} -- "
            "the prompt, tools, model, or loop changed since recording: re-record"
        )
