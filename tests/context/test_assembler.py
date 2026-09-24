from opspilot.context.assembler import (
    CONTEXT_DIR,
    build_initial_messages,
    build_system_blocks,
    build_system_prompt,
    compute_prompt_version,
    prompt_version,
)


def test_build_system_prompt_snapshot() -> None:
    agents_md = (CONTEXT_DIR / "AGENTS.md").read_text()
    expected_index = (
        "Available runbooks (load full text with load_runbook):\n"
        "- bad_deploys: Correlating an error_rate spike with a recent deploy and when to "
        "roll it back.\n"
        "- db_pool_issues: Diagnosing latency spikes and connection pool timeouts caused "
        "by a db_pool_size config change.\n"
        "- disk_exhaustion: Recognizing disk-full symptoms and why no destructive tool "
        "can fix them.\n"
        "- memory_leaks: Recognizing linear memory growth and OOMKilled restarts, and "
        "why restart isn't a real fix."
    )

    assert build_system_prompt() == f"{agents_md}\n\n{expected_index}"


def test_full_runbook_bodies_are_not_in_the_system_prompt() -> None:
    prompt = build_system_prompt()

    # Progressive disclosure: only the index line should be present, not the
    # runbook's full diagnostic steps / fix sections.
    assert "db_pool_issues" in prompt
    assert "Diagnostic steps" not in prompt
    assert "rollback_config(service, <previous_version>)" not in prompt


def test_build_system_blocks_has_one_cache_control_breakpoint() -> None:
    blocks = build_system_blocks()

    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == build_system_prompt()
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_build_initial_messages() -> None:
    messages = build_initial_messages("checkout p95 latency elevated")

    assert messages == [{"role": "user", "content": "New alert:\ncheckout p95 latency elevated"}]


def test_compute_prompt_version_is_deterministic_and_short() -> None:
    schemas = [{"name": "list_services"}]

    v1 = compute_prompt_version("agents text", "index text", schemas)
    v2 = compute_prompt_version("agents text", "index text", schemas)

    assert v1 == v2
    assert len(v1) == 12


def test_prompt_version_changes_when_runbook_index_changes() -> None:
    schemas = [{"name": "list_services"}]

    v1 = prompt_version(CONTEXT_DIR, schemas, runbook_index={"a": "one description"})
    v2 = prompt_version(CONTEXT_DIR, schemas, runbook_index={"a": "a different description"})

    assert v1 != v2


def test_prompt_version_changes_when_tool_schemas_change() -> None:
    v1 = prompt_version(CONTEXT_DIR, [{"name": "a"}])
    v2 = prompt_version(CONTEXT_DIR, [{"name": "a"}, {"name": "b"}])

    assert v1 != v2


def test_prompt_version_stable_for_identical_inputs() -> None:
    schemas = [{"name": "list_services"}]

    v1 = prompt_version(CONTEXT_DIR, schemas)
    v2 = prompt_version(CONTEXT_DIR, schemas)

    assert v1 == v2
