"""Check that the docs still match the code.

    uv run python scripts/check_docs.py              # CI: relative links + code references
    uv run python scripts/check_docs.py --external   # also fetch every http(s) link (network)

Docs rot silently: a function gets renamed, a file moves, a link 404s. Two
conventions make that checkable:
  - relative links, e.g. [open_run](../../opspilot/runs.py) or ...#L120 --
    the file must exist (and a #L<n> anchor must be within it);
  - code references in backticks, e.g. `opspilot/runs.py::open_run` -- the file
    must exist and define that function, class or variable.
Fenced code blocks are skipped: they hold commands and examples, not references.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The curated docs. Specs and PLAN.md are plans: they name files before they exist.
DOC_GLOBS = ("README.md", "CONCEPTS.md", "docs/learn/**/*.md")

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
CODE_REF_RE = re.compile(r"`([\w./-]+\.(?:py|ya?ml|md|toml|html|json|jsonl|txt))(?:::(\w+))?`")
FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Problem:
    doc: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.doc}:{self.line}: {self.message}"


def doc_files(root: Path = ROOT) -> list[Path]:
    files: set[Path] = set()
    for pattern in DOC_GLOBS:
        files.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(files)


def prose_lines(text: str) -> list[tuple[int, str]]:
    """(line number, line) for every line outside fenced code blocks."""
    out, in_fence = [], False
    for number, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append((number, line))
    return out


def defines(path: Path, symbol: str) -> bool:
    pattern = re.compile(
        rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(symbol)}\b"
        rf"|^\s*{re.escape(symbol)}\s*(?::[^=]*)?="
        rf"|^\s*{re.escape(symbol)}:",
        re.MULTILINE,
    )
    return bool(pattern.search(path.read_text()))


def check_link(doc: Path, target: str, root: Path) -> str | None:
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    path_part, _, anchor = target.partition("#")
    resolved = (doc.parent / path_part).resolve()
    if not resolved.exists():
        return f"broken link: {target}"
    if not resolved.is_relative_to(root):
        return f"link leaves the repo: {target}"
    line_anchor = re.fullmatch(r"L(\d+)(?:-L\d+)?", anchor)
    if line_anchor and resolved.is_file():
        line_count = len(resolved.read_text().splitlines())
        if int(line_anchor[1]) > line_count:
            return f"line anchor past end of file ({line_count} lines): {target}"
    return None


def check_code_ref(path_text: str, symbol: str | None, root: Path) -> str | None:
    path = root / path_text
    if not path.is_file():
        return f"code reference to missing file: {path_text}"
    if symbol and not defines(path, symbol):
        return f"{path_text} no longer defines `{symbol}`"
    return None


def check(root: Path = ROOT) -> tuple[list[Problem], list[tuple[str, int, str]]]:
    """Return (problems, external links found) across the curated docs."""
    problems: list[Problem] = []
    external: list[tuple[str, int, str]] = []
    for doc in doc_files(root):
        rel = doc.relative_to(root).as_posix()
        for number, line in prose_lines(doc.read_text()):
            for target in LINK_RE.findall(line):
                if target.startswith(("http://", "https://")):
                    external.append((rel, number, target))
                elif (message := check_link(doc, target, root)) is not None:
                    problems.append(Problem(rel, number, message))
            # A link's text (e.g. [`AGENTS.md`](../x/AGENTS.md)) is already
            # verified by its target -- only bare backticked references count.
            for path_text, symbol in CODE_REF_RE.findall(LINK_RE.sub("", line)):
                if (message := check_code_ref(path_text, symbol or None, root)) is not None:
                    problems.append(Problem(rel, number, message))
    return problems, external


def check_external(links: list[tuple[str, int, str]]) -> list[Problem]:
    import httpx  # dev dependency; only needed for this opt-in, networked check

    problems = []
    headers = {"User-Agent": "opspilot-docs-check/1.0"}
    with httpx.Client(follow_redirects=True, timeout=20, headers=headers) as client:
        for doc, line, url in sorted(set(links)):
            try:
                status = client.get(url).status_code
            except httpx.HTTPError as exc:
                problems.append(Problem(doc, line, f"unreachable: {url} ({type(exc).__name__})"))
                continue
            if status >= 400:
                problems.append(Problem(doc, line, f"HTTP {status}: {url}"))
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--external", action="store_true", help="also fetch http(s) links")
    args = parser.parse_args(argv)

    problems, external = check()
    if args.external:
        problems += check_external(external)
    for problem in problems:
        print(problem)
    docs = len(doc_files())
    status = "OK" if not problems else f"{len(problems)} problem(s)"
    print(f"checked {docs} doc(s), {len(external)} external link(s): {status}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
