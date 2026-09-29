import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class SandboxPathError(ValueError):
    """A requested path would escape the sandbox root."""


class Sandbox(BaseModel):
    """A materialized ShopStack instance rooted at `root`.

    All tool implementations read/write files through `path()` rather than
    touching `root` directly, so the traversal guard below is the single
    enforcement point.
    """

    root: Path

    # [HARNESS:ENV] Bounded — every file access is confined to the sandbox root.
    # WHY: tool arguments (e.g. a service name used to build a log path) come from
    # model output. A model that's confused or adversarially prompted could send
    # "../../../etc/passwd" as a "service name" — this must never escape `root`,
    # on this machine or in eval/CI sandboxes running many scenarios side by side.
    # INTERVIEW: "How do you sandbox an agent's file access?" → reject absolute
    # paths and ".." components up front, then resolve() and re-check containment
    # so symlinks can't be used to escape either.
    def path(self, rel: str | Path) -> Path:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            raise SandboxPathError(f"absolute paths are not allowed: {rel}")
        if ".." in rel_path.parts:
            raise SandboxPathError(f"path traversal is not allowed: {rel}")

        root_resolved = self.root.resolve()
        candidate = (self.root / rel_path).resolve()
        if candidate != root_resolved and root_resolved not in candidate.parents:
            raise SandboxPathError(f"path escapes sandbox root: {rel}")
        return candidate

    def snapshot(self) -> dict[str, str]:
        """sha256 of state.json + every current config file, for inspection/evals."""
        hashes: dict[str, str] = {}
        targets = [self.root / "state.json", *sorted((self.root / "config").glob("*.yaml"))]
        for target in targets:
            if target.exists():
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                hashes[str(target.relative_to(self.root))] = digest
        return hashes

    # [HARNESS:EVAL] Readable end state, not just hashes -- what graders check.
    # WHY: snapshot() says *whether* files changed; a grader asserting
    # "checkout ended on config v12" needs *what* they say. Flat dotted keys
    # ("checkout.config_version") keep final_state expectations in YAML
    # trivial, and capturing this at trial end means stored trials can be
    # re-graded later without the (deleted) sandbox.
    # INTERVIEW: "How do you grade side effects, not just the answer?" ->
    # diff the environment's final state against an expected state.
    def facts(self) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        state_path = self.root / "state.json"
        if state_path.exists():
            for service, fields in json.loads(state_path.read_text()).items():
                for key, value in fields.items():
                    # restarted_at is wall-clock -- only "was it restarted" is a fact.
                    if key == "restarted_at":
                        facts[f"{service}.restarted"] = True
                    else:
                        facts[f"{service}.{key}"] = value
        for config_path in sorted((self.root / "config").glob("*.yaml")):
            config = yaml.safe_load(config_path.read_text()) or {}
            service = config_path.stem
            facts[f"{service}.config_version"] = config.get("version")
            for key, value in config.items():
                if key != "version":
                    facts[f"{service}.config.{key}"] = value
        deploys_path = self.root / "deploys.json"
        if deploys_path.exists():
            # Append order, not ts: rollback_deploy appends the entry that
            # becomes current, so the last entry per service is live.
            for deploy in json.loads(deploys_path.read_text()):
                facts[f"{deploy['service']}.deploy_version"] = deploy["version"]
        return facts

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
