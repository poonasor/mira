"""Find open PRs Mira received but never finished reviewing, and re-queue them.

A review runs as a background task after the webhook has already answered
200, so a process restart (or a crash) mid-review loses it without a trace:
the PR sits in ``pull_requests`` as open, with no review record. This sweep
runs on startup and every ``review.stale_sweep_interval_minutes``:

- candidates are open PRs in ``pull_requests`` (only the GitHub path fills that
  table) created between ``stale_sweep_min_age_minutes`` and
  ``stale_sweep_max_age_hours`` ago, with no completed Mira review in the repo's
  ``review_events`` and no live in-progress marker;
- each is logged and re-queued **once per process** by replaying a synthetic
  ``pull_request`` ``opened`` event for the PR's current state through the
  normal webhook dispatch, so the author filter, ``@bot ignore``, the pause
  label and the in-progress marker all still apply;
- a PR that can't be re-queued (no GitHub App, no installation id, PR fetch
  failed) or is still unreviewed after its one re-queue gets a
  ``review.stale`` event instead.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from starlette.background import BackgroundTasks

from mira.config import load_config
from mira.core.review_status import tracker as review_tracker

logger = logging.getLogger(__name__)

_GITHUB_API_URL = "https://api.github.com"

# PRs re-queued by this process; each is re-queued at most once.
_requeued: set[tuple[str, str, int]] = set()
# PRs the sweep is done with (reported, or deliberately not reviewed).
_settled: set[tuple[str, str, int]] = set()


def _has_completed_review(owner: str, repo: str, number: int) -> bool:
    """Whether the repo's review store has a finished Mira review for the PR."""
    from mira.index.store import IndexStore

    store = IndexStore.open(owner, repo, platform="github")
    try:
        return bool(store.list_review_events_for_pr(number))
    finally:
        store.close()


def _review_in_progress(owner: str, repo: str, number: int) -> bool:
    repo_full = f"{owner}/{repo}"
    return any(j.repo == repo_full and j.pr_number == number for j in review_tracker.get_active())


def find_stale_prs(now: float | None = None, config: Any = None) -> list[dict[str, Any]]:
    """Open PRs received in the sweep window with no completed review."""
    from mira.dashboard.api import _app_db

    cfg = (config or load_config()).review
    now = time.time() if now is None else now
    newest = now - cfg.stale_sweep_min_age_minutes * 60
    oldest = now - cfg.stale_sweep_max_age_hours * 3600
    stale: list[dict[str, Any]] = []
    for pr in _app_db.get_open_pull_requests():
        created = pr.get("created_at") or 0.0
        if not created or created > newest or created < oldest:
            continue
        owner, repo, number = pr["owner"], pr["repo"], int(pr["number"])
        if _review_in_progress(owner, repo, number):
            continue
        try:
            if _has_completed_review(owner, repo, number):
                continue
        except Exception as exc:
            logger.warning(
                "Stale sweep: can't read reviews for %s/%s#%d: %s", owner, repo, number, exc
            )
            continue
        stale.append({**pr, "age_minutes": int((now - created) // 60)})
    return stale


async def _emit_stale(pr: dict[str, Any], requeued: bool, reason: str) -> None:
    from mira.outbound_webhooks import REVIEW_STALE, dispatch_event

    try:
        await dispatch_event(
            REVIEW_STALE,
            {
                "repo": f"{pr['owner']}/{pr['repo']}",
                "pr_url": pr.get("url") or "",
                "number": pr["number"],
                "pr": pr["number"],
                "age_minutes": pr.get("age_minutes"),
                "requeued": requeued,
                "reason": reason,
            },
        )
    except Exception as exc:
        logger.warning("Stale sweep: could not send review.stale: %s", exc)


async def _fetch_pr(token: str, owner: str, repo: str, number: int) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(
            f"{_GITHUB_API_URL}/repos/{owner}/{repo}/pulls/{number}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data


async def _requeue(pr: dict[str, Any], app_auth: Any, bot_name: str) -> str:
    """Replay the PR's ``opened`` event; returns a short outcome string."""
    from mira.dashboard.api import _app_db
    from mira.platforms.github.webhook import dispatch_github_event

    owner, repo, number = pr["owner"], pr["repo"], int(pr["number"])
    record = _app_db.get_repo(owner, repo)
    installation_id = int(record.installation_id) if record else 0
    if not installation_id:
        return "no installation id on record"
    token = await app_auth.get_installation_token(installation_id)
    pr_json = await _fetch_pr(token, owner, repo, number)
    payload = {
        "action": "opened",
        "installation": {"id": installation_id},
        "pull_request": pr_json,
        "repository": (pr_json.get("base") or {}).get("repo") or {},
        "sender": pr_json.get("user") or {},
    }
    tasks = BackgroundTasks()
    if pr_json.get("state") != "open":
        # Closed since; the lifecycle handler (queued by dispatch for every
        # pull_request event) records that, nothing to review.
        payload["action"] = "closed"
    status = await dispatch_github_event("pull_request", payload, app_auth, bot_name, tasks)
    await tasks()
    return status if pr_json.get("state") == "open" else "closed since"


async def sweep_once(app_auth: Any, bot_name: str, now: float | None = None) -> int:
    """One sweep pass. Returns how many PRs were re-queued."""
    requeued = 0
    for pr in find_stale_prs(now):
        key = (pr["owner"], pr["repo"], int(pr["number"]))
        label = f"{key[0]}/{key[1]}#{key[2]}"
        if key in _settled:
            continue
        if key in _requeued:
            # Re-queued once already and still nothing recorded: report once, stop.
            _settled.add(key)
            logger.warning(
                "Stale sweep: %s still has no completed review %d min after it was "
                "opened, even after a re-queue; not re-queueing again",
                label,
                pr["age_minutes"],
            )
            await _emit_stale(pr, False, "still unreviewed after one re-queue")
            continue
        logger.warning(
            "Stale sweep: %s was received %d min ago and has no completed review",
            label,
            pr["age_minutes"],
        )
        if app_auth is None:
            _settled.add(key)
            await _emit_stale(pr, False, "GitHub App not configured; cannot re-queue")
            continue
        _requeued.add(key)
        try:
            outcome = await _requeue(pr, app_auth, bot_name)
        except Exception as exc:
            _settled.add(key)
            logger.exception("Stale sweep: re-queue failed for %s", label)
            await _emit_stale(pr, False, f"re-queue failed: {exc}")
            continue
        if outcome == "processing":
            requeued += 1
            logger.info("Stale sweep: re-queued %s", label)
            await _emit_stale(pr, True, "re-queued")
            continue
        # Paused, ignored, author-filtered or closed: not stale, don't revisit.
        _settled.add(key)
        logger.info("Stale sweep: did not re-queue %s (%s)", label, outcome)
        if outcome == "no installation id on record":
            await _emit_stale(pr, False, outcome)
    return requeued


async def run_forever(app_auth: Any, bot_name: str) -> None:
    """Sweep on startup, then every ``review.stale_sweep_interval_minutes``."""
    while True:
        cfg = load_config().review
        if not cfg.stale_sweep_enabled:
            logger.info("Stale review sweep disabled (review.stale_sweep_enabled=false)")
            return
        try:
            await sweep_once(app_auth, bot_name)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Stale review sweep failed")
        await asyncio.sleep(cfg.stale_sweep_interval_minutes * 60)
