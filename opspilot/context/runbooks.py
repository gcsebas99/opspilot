from pathlib import Path

RUNBOOKS_DIR = Path(__file__).parent / "runbooks"

# name -> one-line description. This is the ONLY thing that goes into the
# system prompt; full runbook text is loaded on demand via load_runbook.
# Keep this in sync by hand when adding/renaming a runbook file.
RUNBOOK_INDEX: dict[str, str] = {
    "db_pool_issues": (
        "Diagnosing latency spikes and connection pool timeouts caused by a "
        "db_pool_size config change."
    ),
    "bad_deploys": (
        "Correlating an error_rate spike with a recent deploy and when to roll it back."
    ),
    "memory_leaks": (
        "Recognizing linear memory growth and OOMKilled restarts, and why restart isn't a real fix."
    ),
    "disk_exhaustion": ("Recognizing disk-full symptoms and why no destructive tool can fix them."),
}


def runbook_path(name: str) -> Path:
    return RUNBOOKS_DIR / f"{name}.md"
