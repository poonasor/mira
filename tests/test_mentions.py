"""Bot mention matching — respond to either the configured name or real identity."""

from __future__ import annotations

from mira.platforms.mentions import (
    author_is_filtered,
    command_after_mention,
    has_mention,
    mention_names,
    strip_mentions,
)


def test_mention_names_dedupes():
    assert mention_names("mira", None) == ["mira"]
    assert mention_names("mira", "mira") == ["mira"]
    assert mention_names("mira", "project_1_bot_x") == ["mira", "project_1_bot_x"]


def test_has_mention_matches_either():
    names = mention_names("mira", "project_1_bot_x")
    assert has_mention("hey @mira review", names)
    assert has_mention("hey @project_1_bot_x review", names)  # autocompleted real user
    assert not has_mention("no mention here", names)
    assert has_mention("@MIRA help", names)  # case-insensitive


def test_strip_mentions_removes_all_forms():
    names = mention_names("mira", "project_1_bot_x")
    assert strip_mentions("@mira review this", names) == "review this"
    assert strip_mentions("@project_1_bot_x review this", names) == "review this"


def test_command_after_mention():
    names = mention_names("mira", "project_1_bot_x")
    assert command_after_mention("@mira review", names) == "review"
    assert command_after_mention("@project_1_bot_x Reject", names) == "reject"
    assert command_after_mention("@mira", names) == ""  # no command word
    assert command_after_mention("nothing", names) == ""


def test_author_is_filtered_blocked_by_raw_login():
    assert author_is_filtered("alice", [], ["alice"]) is True


def test_author_is_filtered_blocked_by_stripped_bot_suffix():
    assert author_is_filtered("dependabot[bot]", [], ["dependabot"]) is True


def test_author_is_filtered_allowlist_empty_not_filtered():
    assert author_is_filtered("alice", [], []) is False


def test_author_is_filtered_allowlist_set_and_on_list_not_filtered():
    assert author_is_filtered("alice", ["alice"], []) is False


def test_author_is_filtered_allowlist_set_and_off_list_filtered():
    assert author_is_filtered("bob", ["alice"], []) is True


def test_author_is_filtered_blocked_preempts_allowlist():
    assert author_is_filtered("alice", ["alice"], ["alice"]) is True


def test_author_is_filtered_empty_login_not_filtered():
    assert author_is_filtered("", ["alice"], []) is False


def test_configured_name_that_prefixes_the_app_slug():
    """bot_name "mira" + GitHub App slug "mira-pm": "@mira-pm review" must be
    the review command, not a Q&A about "-pm review"."""
    names = mention_names("mira", "mira-pm")
    assert strip_mentions("@mira-pm review", names) == "review"
    assert command_after_mention("@mira-pm review", names) == "review"
    assert strip_mentions("@mira review", names) == "review"
    assert strip_mentions("@mira, can you @mira-pm review", names) == ", can you review"
    assert has_mention("@mira-pm review", names)
    assert has_mention("@mira review", names)
    assert not has_mention("@mirabelle review", names)
    assert not has_mention("@mira-pmx review", names)
