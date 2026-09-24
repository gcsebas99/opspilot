import hashlib
from pathlib import Path

import pytest

from opspilot.env.sandbox import Sandbox, SandboxPathError


def _make_sandbox(tmp_path: Path) -> Sandbox:
    root = tmp_path / "sbx"
    root.mkdir()
    return Sandbox(root=root)


def test_path_allows_files_inside_root(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    resolved = sandbox.path("config/checkout.yaml")
    assert resolved == (sandbox.root / "config" / "checkout.yaml").resolve()


def test_path_rejects_dotdot_traversal(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    with pytest.raises(SandboxPathError):
        sandbox.path("../evil.txt")


def test_path_rejects_dotdot_in_middle(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    with pytest.raises(SandboxPathError):
        sandbox.path("config/../../evil.txt")


def test_path_rejects_absolute_paths(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    with pytest.raises(SandboxPathError):
        sandbox.path("/etc/passwd")


def test_path_rejects_symlink_escape(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (sandbox.root / "escape_link").symlink_to(outside)

    with pytest.raises(SandboxPathError):
        sandbox.path("escape_link/evil.txt")


def test_snapshot_hashes_state_and_config(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    (sandbox.root / "config").mkdir()
    (sandbox.root / "state.json").write_text('{"web": "healthy"}')
    (sandbox.root / "config" / "checkout.yaml").write_text("db_pool_size: 5\n")

    snapshot = sandbox.snapshot()

    assert snapshot["state.json"] == hashlib.sha256(b'{"web": "healthy"}').hexdigest()
    assert snapshot["config/checkout.yaml"] == hashlib.sha256(b"db_pool_size: 5\n").hexdigest()


def test_cleanup_removes_root(tmp_path: Path) -> None:
    sandbox = _make_sandbox(tmp_path)
    (sandbox.root / "marker.txt").write_text("hi")

    sandbox.cleanup()

    assert not sandbox.root.exists()
