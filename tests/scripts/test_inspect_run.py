import pytest

from scripts.inspect_run import main


def test_default_run_prints_trace_audit_and_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "outcome: completed" in out
    assert "TRACE" in out and "model_call" in out
    assert "chain verification: OK" in out


def test_operator_run_approve_and_tamper_shows_detection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["--scenario", "checkout_pool_exhaustion", "--role", "operator", "--approve", "--tamper"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "paused for approval" in out
    assert "approval_decision -> rollback_config: approve" in out
    assert "chain verification: OK (2 entries)" in out
    assert "verification: BROKEN at entry 1" in out


def test_unrecorded_path_is_reported(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--scenario", "db_disk_full", "--role", "admin"]) == 2
    assert "no recorded demo path" in capsys.readouterr().out


def test_state_flag_shows_the_paused_checkpoint(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--scenario", "checkout_pool_exhaustion", "--role", "operator", "--state"]) == 0
    out = capsys.readouterr().out
    assert "CHECKPOINTS saved for this run:" in out
    assert "messages (the conversation): 14" in out
    assert "waiting on: rollback_config" in out
