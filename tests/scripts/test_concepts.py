from pathlib import Path

from scripts.concepts import ROOT, main, render, scan, violations

# Built at runtime so this file's own fixtures never appear as real tags when
# the repo is scanned.
T = "[" + "HARNESS:"


def _repo(tmp_path: Path, source: str) -> Path:
    (tmp_path / "opspilot").mkdir()
    (tmp_path / "opspilot" / "mod.py").write_text(source)
    return tmp_path


GOOD = f'''
# {T}LOOP] Exit condition #2 -- hard step cap.
# WHY: a confused model can loop forever.
def f() -> None: ...


def g() -> None:
    """Docstring tag.

    {T}EVAL] Replay fails loudly on a miss.
    WHY: a silent fallback would test a conversation that never happened.
    """
'''


def test_scan_finds_comment_and_docstring_tags(tmp_path: Path) -> None:
    tags = scan(_repo(tmp_path, GOOD))
    assert [(t.pillar, t.title, t.line, t.block_lines) for t in tags] == [
        ("LOOP", "Exit condition #2 -- hard step cap.", 2, 2),
        ("EVAL", "Replay fails loudly on a miss.", 10, 2),
    ]
    assert violations(tags) == []


def test_rules_catch_long_blocks_missing_why_unknown_tags_and_empty_titles(
    tmp_path: Path,
) -> None:
    source = (
        f"# {T}LOOP] Too long.\n# WHY: x\n" + "# more\n" * 4 + "\n"
        f"# {T}LOOP] No reason given.\n# just words\n\n"
        f"# {T}MAGIC] Not a pillar.\n# WHY: x\n\n"
        f"# {T}EVAL]\n# WHY: no title\n"
    )
    problems = violations(scan(_repo(tmp_path, source)))
    assert any("6 lines (max 5)" in p for p in problems)
    assert any("no WHY" in p for p in problems)
    assert any(f"unknown tag {T}MAGIC]" in p for p in problems)
    assert any("empty first line" in p for p in problems)


def test_render_groups_by_pillar_with_line_links(tmp_path: Path) -> None:
    out = render(scan(_repo(tmp_path, GOOD)))
    assert "| [LOOP](#loop) |" in out
    assert "- [Exit condition #2 -- hard step cap.](opspilot/mod.py#L2)" in out
    assert "_No tags yet._" in out  # pillars without tags still get a section
    assert "do not edit by hand" in out


def test_committed_concepts_md_is_fresh_and_every_tag_follows_the_rules() -> None:
    """The real repo: the same check CI runs."""
    assert violations(scan(ROOT)) == []
    assert main(["--check"]) == 0
