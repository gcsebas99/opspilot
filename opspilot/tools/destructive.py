import json
from datetime import UTC, datetime

import yaml
from pydantic import BaseModel, Field

from opspilot.env.sandbox import Sandbox
from opspilot.tools.base import Tool, ToolResult


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class RestartServiceInput(BaseModel):
    service: str = Field(description="Service name, e.g. 'inventory'.")


def _restart_service(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    assert isinstance(args, RestartServiceInput)
    state_path = sandbox.path("state.json")
    if not state_path.exists():
        return ToolResult(ok=False, content="state.json not found in sandbox")

    state = json.loads(state_path.read_text())
    if args.service not in state:
        return ToolResult(ok=False, content=f"unknown service {args.service!r}")

    state[args.service] = {"status": "healthy", "restarted_at": _now_iso()}
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return ToolResult(ok=True, content=f"restarted {args.service}", data=state[args.service])


RESTART_SERVICE = Tool(
    name="restart_service",
    description=(
        "Restart a service, resetting its status to healthy. Use for transient issues "
        "(e.g. a memory leak) where a restart is a real fix or a reasonable mitigation "
        "while escalating. This is destructive and affects live traffic."
    ),
    input_model=RestartServiceInput,
    risk="destructive",
    fn=_restart_service,
)


class RollbackConfigInput(BaseModel):
    service: str = Field(description="Service name, e.g. 'checkout'.")
    version: int = Field(description="Version number to roll back to, from config_history.")


def _rollback_config(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    assert isinstance(args, RollbackConfigInput)
    version_path = sandbox.path(f"config/history/{args.service}/v{args.version}.yaml")
    if not version_path.exists():
        return ToolResult(
            ok=False,
            content=(
                f"no config version {args.version} for service {args.service!r} "
                "(check config_history for available versions)"
            ),
        )

    content = version_path.read_text()
    sandbox.path(f"config/{args.service}.yaml").write_text(content)
    return ToolResult(
        ok=True,
        content=f"rolled back {args.service} config to v{args.version}",
        data=yaml.safe_load(content),
    )


ROLLBACK_CONFIG = Tool(
    name="rollback_config",
    description=(
        "Revert a service's current config to a previous version (see config_history "
        "for available version numbers and what changed). Use when a config change is "
        "the root cause. This is destructive and affects live traffic."
    ),
    input_model=RollbackConfigInput,
    risk="destructive",
    fn=_rollback_config,
)


class RollbackDeployInput(BaseModel):
    service: str = Field(description="Service name, e.g. 'payments'.")


def _rollback_deploy(sandbox: Sandbox, args: BaseModel) -> ToolResult:
    assert isinstance(args, RollbackDeployInput)
    deploys_path = sandbox.path("deploys.json")
    if not deploys_path.exists():
        return ToolResult(ok=False, content="deploys.json not found in sandbox")

    deploys = json.loads(deploys_path.read_text())
    service_deploys = sorted(
        (d for d in deploys if d["service"] == args.service), key=lambda d: d["ts"]
    )
    if len(service_deploys) < 2:
        return ToolResult(
            ok=False, content=f"no earlier deploy to roll back to for service {args.service!r}"
        )

    previous = service_deploys[-2]
    rollback_entry = {
        "service": args.service,
        "version": previous["version"],
        "ts": _now_iso(),
        "author": "opspilot-agent",
    }
    deploys.append(rollback_entry)
    deploys_path.write_text(json.dumps(deploys, indent=2) + "\n")
    return ToolResult(
        ok=True,
        content=f"rolled back {args.service} to v{previous['version']}",
        data=rollback_entry,
    )


ROLLBACK_DEPLOY = Tool(
    name="rollback_deploy",
    description=(
        "Redeploy the previous version of a service, undoing its most recent deploy. "
        "Use when a deploy is the root cause (e.g. error_rate jumped right after a "
        "deploy). This is destructive and affects live traffic."
    ),
    input_model=RollbackDeployInput,
    risk="destructive",
    fn=_rollback_deploy,
)

DESTRUCTIVE_TOOLS = [RESTART_SERVICE, ROLLBACK_CONFIG, ROLLBACK_DEPLOY]
