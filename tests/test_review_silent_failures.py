"""Reviews must never fail silently: review.failed on every exit path,
first_review_at stamping, the stale-review sweep, partial-review scoring and
exit logging."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mira.config import MiraConfig
from mira.models import (
    ReviewResult,
    WalkthroughConfidenceScore,
    WalkthroughResult,
)
from mira.platforms.github.webhook import handle_pull_request

HEAD = "a" * 40


def _pr_payload() -> dict[str, Any]:
    return {
        "installation": {"id": 1},
        "action": "opened",
        "pull_request": {"number": 42, "title": "t", "head": {"sha": HEAD}},
        "repository": {"owner": {"login": "o"}, "name": "r"},
    }


def _events(dispatch: AsyncMock) -> list[str]:
    return [c.args[0] for c in dispatch.await_args_list]


def _data(dispatch: AsyncMock, event: str) -> dict:
    return next(c.args[1] for c in dispatch.await_args_list if c.args[0] == event)


async def _run_pr(review_effect: Any) -> tuple[AsyncMock, MagicMock]:
    engine = AsyncMock()
    if isinstance(review_effect, BaseException):
        engine.review_pr = AsyncMock(side_effect=review_effect)
    else:
        engine.review_pr = AsyncMock(return_value=review_effect)
    auth = AsyncMock()
    with (
        patch("mira.platforms.handlers.ReviewEngine", return_value=engine),
        patch("mira.platforms.handlers.create_llm"),
        patch("mira.platforms.handlers.load_config", return_value=MagicMock()),
        patch("mira.platforms.github.webhook.create_provider"),
        patch("mira.dashboard.api._app_db") as db,
        patch("mira.outbound_webhooks.dispatch_event", new_callable=AsyncMock) as dispatch,
    ):
        db.get_repo.return_value = MagicMock(status="ready")
        await handle_pull_request(_pr_payload(), auth, "mira-bot")
    return dispatch, db


# ── Item 1: review.failed on every failure path ─────────────────────────────


async def test_review_exception_emits_one_review_failed_with_context(caplog) -> None:
    caplog.set_level(logging.ERROR)
    dispatch, _ = await _run_pr(RuntimeError("boom"))

    assert _events(dispatch) == ["review.failed"]  # exactly once, no completed
    data = _data(dispatch, "review.failed")
    assert data["repo"] == "o/r"
    assert data["pr"] == 42 and data["number"] == 42
    assert data["head_sha"] == HEAD
    assert data["reason"] == "RuntimeError: boom"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and all(r.exc_info for r in errors)
    assert len(errors) == 1  # the handler doesn't log the same failure twice


async def test_cancelled_review_emits_review_failed_and_reraises(caplog) -> None:
    caplog.set_level(logging.ERROR)
    captured = AsyncMock()
    with (
        patch("mira.platforms.handlers.ReviewEngine") as engine_cls,
        patch("mira.platforms.handlers.create_llm"),
        patch("mira.platforms.handlers.load_config", return_value=MagicMock()),
        patch("mira.platforms.github.webhook.create_provider"),
        patch("mira.dashboard.api._app_db"),
        patch("mira.outbound_webhooks.dispatch_event", captured),
        pytest.raises(asyncio.CancelledError),
    ):
        engine_cls.return_value.review_pr = AsyncMock(side_effect=asyncio.CancelledError())
        await handle_pull_request(_pr_payload(), AsyncMock(), "mira-bot")

    assert _events(captured) == ["review.failed"]
    assert "cancelled" in _data(captured, "review.failed")["reason"]
    assert any(r.levelno == logging.ERROR and r.exc_info for r in caplog.records)


async def test_failure_before_review_starts_emits_review_failed() -> None:
    with (
        patch(
            "mira.platforms.github.webhook.create_provider",
            side_effect=RuntimeError("no token"),
        ),
        patch("mira.outbound_webhooks.dispatch_event", new_callable=AsyncMock) as dispatch,
    ):
        await handle_pull_request(_pr_payload(), AsyncMock(), "mira-bot")

    assert _events(dispatch) == ["review.failed"]
    assert _data(dispatch, "review.failed")["head_sha"] == HEAD


async def test_review_command_failure_emits_review_failed() -> None:
    from mira.platforms.handlers import run_pr_command

    engine = AsyncMock()
    engine.review_pr = AsyncMock(side_effect=RuntimeError("boom"))
    provider = AsyncMock()
    provider.get_repo_permission = AsyncMock(return_value="write")
    with (
        patch("mira.platforms.handlers.ReviewEngine", return_value=engine),
        patch("mira.platforms.handlers.create_llm"),
        patch("mira.platforms.handlers.load_config", return_value=MagicMock()),
        patch("mira.dashboard.api._app_db"),
        patch("mira.outbound_webhooks.dispatch_event", new_callable=AsyncMock) as dispatch,
        pytest.raises(RuntimeError),
    ):
        await run_pr_command(
            provider, "o", "r", 7, "https://x/o/r/pull/7", "review", "alice", "mira-bot"
        )
    assert _events(dispatch) == ["review.failed"]
    assert _data(dispatch, "review.failed")["reason"] == "RuntimeError: boom"


async def test_review_failed_webhook_errors_never_mask_the_failure() -> None:
    from mira.platforms.handlers import emit_review_failed

    with patch(
        "mira.outbound_webhooks.dispatch_event",
        new_callable=AsyncMock,
        side_effect=RuntimeError("endpoint down"),
    ):
        await emit_review_failed("o/r", "u", 1, HEAD, "x")  # must not raise


def test_review_completed_payload_flags_partial_reviews() -> None:
    from mira.platforms.handlers import review_completed_payload

    full = review_completed_payload("o/r", "u", 1, ReviewResult())
    assert full["partial"] is False
    partial = review_completed_payload("o/r", "u", 1, ReviewResult(unreviewed_paths=["a.py"]))
    assert partial["partial"] is True and partial["unreviewed_paths"] == ["a.py"]


# ── Item 2: first_review_at ──────────────────────────────────────────────────


async def test_successful_review_stamps_first_review_at() -> None:
    _, db = await _run_pr(ReviewResult(summary="ok"))
    db.set_pr_first_review.assert_called_once()
    owner, repo, number, ts = db.set_pr_first_review.call_args.args
    assert (owner, repo, number) == ("o", "r", 42)
    assert abs(ts - time.time()) < 60


async def test_skipped_or_failed_review_does_not_stamp_first_review_at() -> None:
    _, db = await _run_pr(ReviewResult(skipped_reason="All files matched exclusion rules"))
    db.set_pr_first_review.assert_not_called()
    _, db = await _run_pr(RuntimeError("boom"))
    db.set_pr_first_review.assert_not_called()


def test_first_review_only_recorded_for_github() -> None:
    from mira.platforms.handlers import mark_first_review

    with patch("mira.dashboard.api._app_db") as db:
        mark_first_review("gitlab", "o", "r", 1, ReviewResult())
        db.set_pr_first_review.assert_not_called()
        db.set_pr_first_review.side_effect = RuntimeError("db down")
        mark_first_review("github", "o", "r", 1, ReviewResult())  # never raises


# ── Item 3: stale-review sweep ───────────────────────────────────────────────


@pytest.fixture
def sweep():
    from mira.platforms import stale_sweep

    stale_sweep._requeued.clear()
    stale_sweep._settled.clear()
    yield stale_sweep
    stale_sweep._requeued.clear()
    stale_sweep._settled.clear()


def _open_pr(number: int, age: float, now: float, **extra: Any) -> dict[str, Any]:
    return {
        "owner": "o",
        "repo": "r",
        "number": number,
        "url": f"https://github.com/o/r/pull/{number}",
        "created_at": now - age * 60,
        "first_review_at": 0.0,
        **extra,
    }


def test_find_stale_prs_window_and_review_record(sweep) -> None:
    now = 1_000_000.0
    prs = [
        _open_pr(1, 5, now),  # too new: a normal review may still be running
        _open_pr(2, 60, now),  # stale
        _open_pr(3, 60, now),  # has a completed review
        _open_pr(4, 60 * 24 * 10, now),  # older than the max age: never mass-review a backlog
        _open_pr(5, 60, now),  # review in flight
    ]
    active = [MagicMock(repo="o/r", pr_number=5)]
    with (
        patch("mira.dashboard.api._app_db") as db,
        patch.object(sweep, "_has_completed_review", side_effect=lambda o, r, n: n == 3),
        patch.object(sweep.review_tracker, "get_active", return_value=active),
    ):
        db.get_open_pull_requests.return_value = prs
        stale = sweep.find_stale_prs(now=now, config=MiraConfig())
    assert [p["number"] for p in stale] == [2]
    assert stale[0]["age_minutes"] == 60


async def test_sweep_requeues_once_then_reports_once(sweep) -> None:
    pr = _open_pr(2, 60, time.time(), age_minutes=60)
    requeue = AsyncMock(return_value="processing")
    with (
        patch.object(sweep, "find_stale_prs", return_value=[pr]),
        patch.object(sweep, "_requeue", requeue),
        patch("mira.outbound_webhooks.dispatch_event", new_callable=AsyncMock) as dispatch,
    ):
        assert await sweep.sweep_once(MagicMock(), "mira-bot") == 1
        assert await sweep.sweep_once(MagicMock(), "mira-bot") == 0
        assert await sweep.sweep_once(MagicMock(), "mira-bot") == 0

    requeue.assert_awaited_once()
    assert _events(dispatch) == ["review.stale", "review.stale"]
    first, second = (c.args[1] for c in dispatch.await_args_list)
    assert first["requeued"] is True and first["pr"] == 2
    assert second["requeued"] is False and "after one re-queue" in second["reason"]


async def test_sweep_without_app_auth_only_reports(sweep) -> None:
    pr = _open_pr(2, 60, time.time(), age_minutes=60)
    requeue = AsyncMock()
    with (
        patch.object(sweep, "find_stale_prs", return_value=[pr]),
        patch.object(sweep, "_requeue", requeue),
        patch("mira.outbound_webhooks.dispatch_event", new_callable=AsyncMock) as dispatch,
    ):
        await sweep.sweep_once(None, "mira-bot")
        await sweep.sweep_once(None, "mira-bot")
    requeue.assert_not_awaited()
    assert _events(dispatch) == ["review.stale"]


async def test_sweep_leaves_paused_or_ignored_prs_alone(sweep) -> None:
    pr = _open_pr(2, 60, time.time(), age_minutes=60)
    requeue = AsyncMock(return_value="paused")
    with (
        patch.object(sweep, "find_stale_prs", return_value=[pr]),
        patch.object(sweep, "_requeue", requeue),
        patch("mira.outbound_webhooks.dispatch_event", new_callable=AsyncMock) as dispatch,
    ):
        await sweep.sweep_once(MagicMock(), "mira-bot")
        await sweep.sweep_once(MagicMock(), "mira-bot")
    requeue.assert_awaited_once()
    assert _events(dispatch) == []


async def test_requeue_replays_opened_event_through_normal_dispatch(sweep) -> None:
    pr_json = {
        "number": 2,
        "state": "open",
        "user": {"login": "alice"},
        "head": {"sha": HEAD},
        "base": {"repo": {"name": "r", "owner": {"login": "o"}}},
    }
    auth = AsyncMock()
    auth.get_installation_token = AsyncMock(return_value="tok")
    seen: dict[str, Any] = {}

    async def fake_dispatch(event, payload, app_auth, bot_name, tasks):
        seen.update(event=event, payload=payload)
        return "processing"

    with (
        patch("mira.dashboard.api._app_db") as db,
        patch.object(sweep, "_fetch_pr", AsyncMock(return_value=pr_json)),
        patch("mira.platforms.github.webhook.dispatch_github_event", fake_dispatch),
    ):
        db.get_repo.return_value = MagicMock(installation_id=9)
        outcome = await sweep._requeue(_open_pr(2, 60, time.time()), auth, "mira-bot")

    assert outcome == "processing"
    auth.get_installation_token.assert_awaited_once_with(9)
    assert seen["event"] == "pull_request"
    assert seen["payload"]["action"] == "opened"
    assert seen["payload"]["installation"] == {"id": 9}
    assert seen["payload"]["sender"] == {"login": "alice"}
    assert seen["payload"]["pull_request"]["head"]["sha"] == HEAD


async def test_requeue_without_installation_does_nothing(sweep) -> None:
    with patch("mira.dashboard.api._app_db") as db:
        db.get_repo.return_value = None
        outcome = await sweep._requeue(_open_pr(2, 60, time.time()), AsyncMock(), "mira-bot")
    assert outcome == "no installation id on record"


async def test_sweep_loop_respects_off_switch(sweep) -> None:
    cfg = MiraConfig()
    cfg.review.stale_sweep_enabled = False
    with (
        patch.object(sweep, "load_config", return_value=cfg),
        patch.object(sweep, "sweep_once", AsyncMock()) as once,
    ):
        await asyncio.wait_for(sweep.run_forever(MagicMock(), "mira-bot"), 1)
    once.assert_not_awaited()


def test_sweep_config_defaults() -> None:
    review = MiraConfig().review
    assert review.stale_sweep_enabled is True
    assert review.stale_sweep_interval_minutes == 30


# ── #13: no score without reviewed code ──────────────────────────────────────


def _scored_result(**kw: Any) -> ReviewResult:
    wt = WalkthroughResult(
        summary="Adds things.",
        confidence_score=WalkthroughConfidenceScore(score=4, label="Safe", reason="r"),
    )
    return ReviewResult(walkthrough=wt, **kw)


def test_score_withheld_when_every_chunk_failed() -> None:
    from mira.core.engine import _mark_partial_review

    result = _scored_result(failed_chunks=[{"chunk": 1, "files": ["a.py"]}], total_chunks=1)
    _mark_partial_review(result, [])
    assert result.confidence_score is None


def test_score_withheld_when_no_changed_file_had_a_patch() -> None:
    from mira.core.engine import _mark_partial_review

    result = _scored_result(total_paths=["a.py", "b.py"])
    _mark_partial_review(result, ["a.py", "b.py"])
    assert result.confidence_score is None
    assert result.unreviewed_paths == ["a.py", "b.py"]
    assert "Partial review" in result.walkthrough.summary


def test_partial_review_keeps_score_but_says_so() -> None:
    from mira.core.engine import _mark_partial_review

    result = _scored_result(total_paths=["a.py", "b.py"], total_chunks=1)
    _mark_partial_review(result, ["b.py"])
    assert result.confidence_score == 4
    assert result.walkthrough.summary.startswith("> ⚠️ **Partial review:**")
    assert "`b.py`" in result.walkthrough.summary
    assert result.walkthrough.summary.endswith("Adds things.")


def test_full_review_is_untouched() -> None:
    from mira.core.engine import _mark_partial_review

    result = _scored_result(total_paths=["a.py"], total_chunks=1)
    _mark_partial_review(result, [])
    assert result.confidence_score == 4
    assert result.walkthrough.summary == "Adds things."


def test_origin_missing_patch_paths() -> None:
    from mira.providers.origin import _missing_patch_paths

    files = [
        {"filename": "ok.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"},
        {"filename": "nopatch.py", "status": "modified", "patch": ""},
        {"filename": "added.py", "status": "added"},
        {"filename": "gone.py", "status": "removed"},
        {"filename": "moved.py", "status": "renamed", "previousFilename": "old.py"},
        {"filename": "counted.py", "status": "modified", "additions": 3, "deletions": 0},
        {"filename": "mode.sh", "status": "modified", "additions": 0, "deletions": 0},
    ]
    assert _missing_patch_paths(files) == ["nopatch.py", "added.py", "counted.py"]


async def test_origin_get_pr_diff_records_missing_patches(caplog) -> None:
    from mira.models import PRInfo
    from mira.providers.origin import OriginProvider

    provider = OriginProvider("tok")
    files = [
        {"filename": "a.py", "status": "modified", "patch": "@@ -1 +1 @@\n-a\n+b"},
        {"filename": "b.py", "status": "modified", "patch": ""},
    ]
    pr = PRInfo(
        title="t",
        description="",
        base_branch="main",
        head_branch="f",
        url="https://origin/o/r/pull/13",
        number=13,
        owner="o",
        repo="r",
    )
    caplog.set_level(logging.WARNING)
    with patch.object(provider, "_paginate", AsyncMock(return_value=files)):
        diff = await provider.get_pr_diff(pr)
    assert "a.py" in diff
    assert provider.missing_patch_paths == ["b.py"]
    assert any("without a patch" in r.getMessage() for r in caplog.records)


# ── Item 4: exit logging ─────────────────────────────────────────────────────


def test_signal_is_logged_with_reviews_in_flight(caplog) -> None:
    import signal

    import uvicorn

    from mira.platforms import lifecycle

    server = lifecycle.LoggingServer(uvicorn.Config(MagicMock()))
    caplog.set_level(logging.WARNING, logger="mira.lifecycle")
    with patch.object(lifecycle, "active_reviews_summary", return_value="o/r#1 (head abc, 3s)"):
        server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit
    assert lifecycle.exit_reason() == "received SIGTERM"
    assert any("SIGTERM" in r.getMessage() and "o/r#1" in r.getMessage() for r in caplog.records)


def test_escaping_system_exit_is_logged_and_reraised(caplog) -> None:
    from mira.platforms import lifecycle

    caplog.set_level(logging.CRITICAL, logger="mira.lifecycle")
    with (
        patch.object(lifecycle.LoggingServer, "run", side_effect=SystemExit(0)),
        patch.object(lifecycle, "install_exit_logging"),
        pytest.raises(SystemExit),
    ):
        lifecycle.run_server(MagicMock(), "127.0.0.1", 0)
    assert lifecycle.exit_reason() == "SystemExit escaped the event loop"
    crit = [r for r in caplog.records if r.levelno == logging.CRITICAL]
    assert crit and crit[0].exc_info


def test_atexit_hook_logs_reason(caplog) -> None:
    from mira.platforms import lifecycle

    lifecycle.set_exit_reason("received SIGTERM")
    caplog.set_level(logging.WARNING, logger="mira.lifecycle")
    with patch.object(lifecycle, "active_reviews_summary", return_value="none"):
        lifecycle._log_at_exit()
    assert "Mira process exiting: received SIGTERM" in caplog.text


def test_active_reviews_summary_lists_in_flight_reviews() -> None:
    from mira.core.review_status import tracker
    from mira.platforms.lifecycle import active_reviews_summary

    with tracker.claim("o/r", 77, "t", "https://x/o/r/pull/77", head_sha=HEAD) as run_id:
        assert run_id
        assert "o/r#77 (head aaaaaaa" in active_reviews_summary()
