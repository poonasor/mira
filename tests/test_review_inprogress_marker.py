"""The per-PR "review in progress" marker must not outlive its run.

Regression for a PR whose first review hung: the marker stayed "reviewing",
so the next push (a new head SHA) logged "Review already in progress" and
was dropped, and nothing reviewed it until the process restarted.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mira.config import MiraConfig, ReviewConfig
from mira.core.review_status import ReviewTracker
from mira.models import ReviewResult
from mira.platforms.github.webhook import handle_pull_request
from mira.platforms.handlers import run_pr_review

REPO = "o/r"
URL = "https://github.com/o/r/pull/7"


# ── ReviewTracker ────────────────────────────────────────────────────────────


def _age(tracker: ReviewTracker, seconds: float) -> None:
    """Pretend every running marker started `seconds` ago."""
    for job in tracker._jobs.values():
        job.started_at -= seconds


def test_live_marker_blocks_duplicate_trigger_for_same_head() -> None:
    t = ReviewTracker()
    assert t.try_start(REPO, 7, "t", URL, head_sha="aaa", ttl_seconds=1200)
    assert t.try_start(REPO, 7, "t", URL, head_sha="aaa", ttl_seconds=1200) is None


def test_marker_without_head_blocks_until_ttl() -> None:
    t = ReviewTracker()
    assert t.try_start(REPO, 7, "t", URL, ttl_seconds=1200)
    assert t.try_start(REPO, 7, "t", URL, ttl_seconds=1200) is None
    # Unknown head on either side can't prove it's a different head.
    assert t.try_start(REPO, 7, "t", URL, head_sha="bbb", ttl_seconds=1200) is None


def test_stale_marker_expires_after_ttl() -> None:
    t = ReviewTracker()
    assert t.try_start(REPO, 7, "t", URL, head_sha="aaa", ttl_seconds=1200)
    _age(t, 1201)
    assert t.try_start(REPO, 7, "t", URL, head_sha="aaa", ttl_seconds=1200)


def test_marker_is_not_stale_inside_ttl() -> None:
    t = ReviewTracker()
    assert t.try_start(REPO, 7, "t", URL, head_sha="aaa", ttl_seconds=1200)
    _age(t, 1100)
    assert t.try_start(REPO, 7, "t", URL, head_sha="aaa", ttl_seconds=1200) is None


def test_new_head_is_not_blocked_by_older_heads_marker() -> None:
    t = ReviewTracker()
    assert t.try_start(REPO, 7, "t", URL, head_sha="c441919", ttl_seconds=1200)
    new_run = t.try_start(REPO, 7, "t", URL, head_sha="bb66900", ttl_seconds=1200)
    assert new_run
    (active,) = t.get_active()
    assert active.head_sha == "bb66900"
    # ...and a duplicate delivery for the new head is still deduped.
    assert t.try_start(REPO, 7, "t", URL, head_sha="bb66900", ttl_seconds=1200) is None


def test_superseded_run_cannot_clear_successors_marker() -> None:
    t = ReviewTracker()
    old = t.try_start(REPO, 7, "t", URL, head_sha="aaa", ttl_seconds=1200)
    new = t.try_start(REPO, 7, "t", URL, head_sha="bbb", ttl_seconds=1200)
    assert old and new and old != new
    t.complete(REPO, 7, run_id=old)
    t.fail(REPO, 7, "late failure", run_id=old)
    (active,) = t.get_active()
    assert active.run_id == new
    t.complete(REPO, 7, run_id=new)
    assert t.get_active() == []


def test_other_prs_are_independent() -> None:
    t = ReviewTracker()
    assert t.try_start(REPO, 7, "t", URL, head_sha="aaa")
    assert t.try_start(REPO, 8, "t", URL, head_sha="aaa")


def test_claim_marks_completed_on_clean_exit() -> None:
    t = ReviewTracker()
    with t.claim(REPO, 7, "t", URL, head_sha="aaa") as run_id:
        assert run_id
        assert len(t.get_active()) == 1
    assert t.get_active() == []
    assert t.get_all()[0].status == "completed"


@pytest.mark.parametrize(
    "exc", [RuntimeError("boom"), asyncio.CancelledError(), KeyboardInterrupt()]
)
def test_claim_clears_marker_on_any_exception(exc: BaseException) -> None:
    t = ReviewTracker()
    with pytest.raises(type(exc)), t.claim(REPO, 7, "t", URL, head_sha="aaa"):
        raise exc
    assert t.get_active() == []
    job = t.get_all()[0]
    assert job.status == "failed"
    assert job.error
    # The slot is free again for the same head.
    assert t.try_start(REPO, 7, "t", URL, head_sha="aaa")


def test_claim_that_is_refused_leaves_the_live_marker_alone() -> None:
    t = ReviewTracker()
    holder = t.try_start(REPO, 7, "t", URL, head_sha="aaa")
    with t.claim(REPO, 7, "t", URL, head_sha="aaa") as run_id:
        assert run_id is None
    (active,) = t.get_active()
    assert active.run_id == holder


# ── Config ───────────────────────────────────────────────────────────────────


def test_ttl_default_is_20_minutes_and_configurable() -> None:
    assert ReviewConfig().in_progress_ttl_minutes == 20
    cfg = MiraConfig.model_validate({"review": {"in_progress_ttl_minutes": 5}})
    assert cfg.review.in_progress_ttl_minutes == 5
    with pytest.raises(ValueError):
        ReviewConfig(in_progress_ttl_minutes=0)


# ── run_pr_review wiring ─────────────────────────────────────────────────────


def _config(ttl_minutes: float = 20) -> MiraConfig:
    return MiraConfig.model_validate({"review": {"in_progress_ttl_minutes": ttl_minutes}})


async def _review(tracker: ReviewTracker, engine: Any, *, head_sha: str, config=None) -> None:
    with (
        patch("mira.platforms.handlers.review_tracker", tracker),
        patch("mira.platforms.handlers.load_config", return_value=config or _config()),
        patch("mira.platforms.handlers.create_llm"),
        patch("mira.platforms.handlers.ReviewEngine", return_value=engine),
        patch("mira.dashboard.api._app_db") as db,
        patch("mira.outbound_webhooks.dispatch_event", new_callable=AsyncMock),
    ):
        db.get_repo.return_value = MagicMock(status="ready")
        await run_pr_review(MagicMock(), "o", "r", 7, URL, False, "mira", head_sha=head_sha)


async def test_hung_review_does_not_block_push_for_new_head() -> None:
    """c441919's review hangs; the bb66900 push must still get reviewed."""
    tracker = ReviewTracker()
    hang = asyncio.Event()
    hung_engine = MagicMock()

    async def _hang(_url: str) -> None:
        await hang.wait()

    hung_engine.review_pr = AsyncMock(side_effect=_hang)
    hung = asyncio.create_task(_review(tracker, hung_engine, head_sha="c441919"))
    await asyncio.sleep(0)
    assert [j.head_sha for j in tracker.get_active()] == ["c441919"]

    engine = MagicMock()
    engine.review_pr = AsyncMock(return_value=ReviewResult(summary="ok"))
    await _review(tracker, engine, head_sha="bb66900")
    engine.review_pr.assert_awaited_once_with(URL)

    hung.cancel()
    with pytest.raises(asyncio.CancelledError):
        await hung
    assert tracker.get_active() == []


async def test_hung_review_for_same_head_unblocks_after_ttl() -> None:
    tracker = ReviewTracker()
    tracker.try_start("o/r", 7, "", URL, head_sha="aaa")  # left behind by a hung run

    engine = MagicMock()
    engine.review_pr = AsyncMock(return_value=ReviewResult(summary="ok"))
    await _review(tracker, engine, head_sha="aaa")
    engine.review_pr.assert_not_awaited()  # still fresh: deduped

    _age(tracker, 20 * 60 + 1)
    await _review(tracker, engine, head_sha="aaa")
    engine.review_pr.assert_awaited_once_with(URL)


async def test_ttl_comes_from_config() -> None:
    tracker = ReviewTracker()
    tracker.try_start("o/r", 7, "", URL, head_sha="aaa")
    _age(tracker, 6 * 60)
    engine = MagicMock()
    engine.review_pr = AsyncMock(return_value=ReviewResult(summary="ok"))
    await _review(tracker, engine, head_sha="aaa", config=_config(ttl_minutes=5))
    engine.review_pr.assert_awaited_once()


async def test_failed_review_clears_marker() -> None:
    tracker = ReviewTracker()
    engine = MagicMock()
    engine.review_pr = AsyncMock(side_effect=RuntimeError("claude-cli exploded"))
    with pytest.raises(RuntimeError):
        await _review(tracker, engine, head_sha="aaa")
    assert tracker.get_active() == []
    assert tracker.get_all()[0].status == "failed"


async def test_setup_failure_before_review_clears_marker() -> None:
    """A crash between claiming the slot and calling the engine (LLM setup,
    DB lookups) used to skip the except block and strand the marker."""
    tracker = ReviewTracker()
    with (
        patch("mira.platforms.handlers.review_tracker", tracker),
        patch("mira.platforms.handlers.load_config", return_value=_config()),
        patch("mira.platforms.handlers.create_llm", side_effect=RuntimeError("bad model")),
        pytest.raises(RuntimeError),
    ):
        await run_pr_review(MagicMock(), "o", "r", 7, URL, False, "mira", head_sha="aaa")
    assert tracker.get_active() == []


async def test_github_synchronize_passes_head_sha() -> None:
    payload = {
        "installation": {"id": 1},
        "action": "synchronize",
        "pull_request": {"number": 65, "title": "t", "head": {"sha": "bb66900"}},
        "repository": {"owner": {"login": "o"}, "name": "r", "private": True},
    }
    with (
        patch("mira.platforms.github.webhook.run_pr_review", new=AsyncMock()) as rpr,
        patch("mira.platforms.github.webhook.create_provider"),
        patch("mira.platforms.github.webhook._record_pr_contribution"),
    ):
        await handle_pull_request(payload, AsyncMock(), "mira")
    assert rpr.await_args.kwargs["head_sha"] == "bb66900"
