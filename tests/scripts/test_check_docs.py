from pathlib import Path

from scripts.check_docs import check, main


def _repo(tmp_path: Path, doc: str) -> Path:
    (tmp_path / "opspilot").mkdir()
    (tmp_path / "opspilot" / "runs.py").write_text(
        "LIMIT = 3\n\n\nasync def open_run():\n    ...\n\n\nclass RunHandle:\n    ...\n"
    )
    (tmp_path / "docs" / "learn").mkdir(parents=True)
    (tmp_path / "docs" / "learn" / "page.md").write_text(doc)
    return tmp_path


def _messages(root: Path) -> list[str]:
    problems, _ = check(root)
    return [p.message for p in problems]


def test_valid_links_and_code_refs_pass(tmp_path: Path) -> None:
    doc = (
        "See [open_run](../../opspilot/runs.py#L4) and `opspilot/runs.py::open_run`,\n"
        "`opspilot/runs.py::RunHandle`, `opspilot/runs.py::LIMIT`, [top](#intro),\n"
        "and [the docs](../learn/).\n"
    )
    assert _messages(_repo(tmp_path, doc)) == []


def test_broken_link_missing_file_and_renamed_symbol_are_caught(tmp_path: Path) -> None:
    doc = (
        "[gone](../../opspilot/old.py)\n"
        "`opspilot/missing.py`\n"
        "`opspilot/runs.py::resume_run`\n"
        "[past the end](../../opspilot/runs.py#L99)\n"
    )
    messages = _messages(_repo(tmp_path, doc))
    assert "broken link: ../../opspilot/old.py" in messages
    assert "code reference to missing file: opspilot/missing.py" in messages
    assert "opspilot/runs.py no longer defines `resume_run`" in messages
    assert any("line anchor past end of file" in m for m in messages)


def test_fenced_code_blocks_are_ignored_and_external_links_collected(tmp_path: Path) -> None:
    doc = "```bash\ncat `opspilot/nope.py` [x](nope.md)\n```\n[a](https://example.com/x)\n"
    root = _repo(tmp_path, doc)
    problems, external = check(root)
    assert problems == []
    assert external == [("docs/learn/page.md", 4, "https://example.com/x")]


def test_real_repo_docs_are_clean() -> None:
    assert main([]) == 0


def test_backticked_link_text_is_not_a_second_reference(tmp_path: Path) -> None:
    doc = "The prompt is [`AGENTS.md`](../../opspilot/runs.py), not a root file.\n"
    assert _messages(_repo(tmp_path, doc)) == []
