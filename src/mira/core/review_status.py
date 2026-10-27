"""In-memory review status tracker for active/in-flight reviews.

The tracker doubles as the per-PR "in progress" marker that stops two webhooks
for the same PR from stacking redundant review runs. A marker that outlives its
run would block every later trigger for that PR until the process restarts, so
it is bounded three ways:

- it expires after a TTL (``review.in_progress_ttl_minutes``, default 20), so a
  hung run can't hold the slot forever;
- it records the head SHA it's reviewing, so a push to a new head isn't skipped
  just because an older head's review is still running;
- :meth:`ReviewTracker.claim` always finishes the marker on the way out,
  including on cancellation, and only the run that owns the marker can finish
  it, so a superseded run can't clear its successor's marker.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_IN_PROGRESS_TTL_SECONDS = 20 * 60


@dataclass
class ReviewStatus:
    repo: str  # "owner/repo"
    pr_number: int
    pr_title: str
    pr_url: str
    status: str  # "reviewing", "completed", "failed"
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    head_sha: str = ""  # head commit under review; "" when the trigger didn't say
    run_id: str = ""  # owner token, so only this run can finish the marker


class ReviewTracker:
    """Thread-safe tracker for active review jobs.

    Completed/failed jobs older than _TTL_SECONDS are evicted from memory
    since they are already persisted in the per-repo review events DB.
    """

    _TTL_SECONDS = 3600  # 1 hour

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, ReviewStatus] = {}

    def _key(self, repo: str, pr_number: int) -> str:
        return f"{repo}#{pr_number}"

    def start(
        self, repo: str, pr_number: int, pr_title: str, pr_url: str, head_sha: str = ""
    ) -> str:
        """Unconditionally register a running review. Returns its run id."""
        with self._lock:
            return self._register(repo, pr_number, pr_title, pr_url, head_sha)

    def _register(
        self, repo: str, pr_number: int, pr_title: str, pr_url: str, head_sha: str
    ) -> str:
        run_id = uuid.uuid4().hex
        self._jobs[self._key(repo, pr_number)] = ReviewStatus(
            repo=repo,
            pr_number=pr_number,
            pr_title=pr_title,
            pr_url=pr_url,
            status="reviewing",
            started_at=time.time(),
            head_sha=head_sha or "",
            run_id=run_id,
        )
        return run_id

    def try_start(
        self,
        repo: str,
        pr_number: int,
        pr_title: str,
        pr_url: str,
        head_sha: str = "",
        ttl_seconds: float = DEFAULT_IN_PROGRESS_TTL_SECONDS,
    ) -> str | None:
        """Atomically check and register.

        Returns the new run id (truthy) when the slot was claimed, or None
        when a live review already covers this trigger. An existing
        "reviewing" marker does NOT block when:

        - it is older than ``ttl_seconds`` (the run hung or died without
          finishing its marker), or
        - both triggers name a head SHA and they differ (a newer push must
          not be dropped because an older head is still under review).
        """
        with self._lock:
            self._evict_old()
            key = self._key(repo, pr_number)
            existing = self._jobs.get(key)
            if existing and existing.status == "reviewing":
                age = time.time() - existing.started_at
                if ttl_seconds > 0 and age > ttl_seconds:
                    logger.warning(
                        "In-progress marker for %s (head %s) is %.0fs old, past the %.0fs "
                        "TTL; treating it as stale and starting a new review",
                        pr_url,
                        existing.head_sha[:7] or "?",
                        age,
                        ttl_seconds,
                    )
                elif head_sha and existing.head_sha and head_sha != existing.head_sha:
                    logger.info(
                        "New head %s for %s while head %s is still under review; "
                        "starting a review for the new head",
                        head_sha[:7],
                        pr_url,
                        existing.head_sha[:7],
                    )
                else:
                    return None
            return self._register(repo, pr_number, pr_title, pr_url, head_sha)

    def _owned(self, key: str, run_id: str | None) -> ReviewStatus | None:
        job = self._jobs.get(key)
        if job is None:
            return None
        # A run that was superseded (new head, or TTL takeover) must not
        # finish the marker that now belongs to its successor.
        if run_id is not None and job.run_id != run_id:
            return None
        return job

    def complete(self, repo: str, pr_number: int, run_id: str | None = None) -> None:
        with self._lock:
            job = self._owned(self._key(repo, pr_number), run_id)
            if job:
                job.status = "completed"
                job.finished_at = time.time()

    def fail(self, repo: str, pr_number: int, error: str = "", run_id: str | None = None) -> None:
        with self._lock:
            job = self._owned(self._key(repo, pr_number), run_id)
            if job:
                job.status = "failed"
                job.error = error
                job.finished_at = time.time()

    @contextmanager
    def claim(
        self,
        repo: str,
        pr_number: int,
        pr_title: str,
        pr_url: str,
        head_sha: str = "",
        ttl_seconds: float = DEFAULT_IN_PROGRESS_TTL_SECONDS,
    ) -> Iterator[str | None]:
        """Claim the PR's review slot for the duration of a ``with`` block.

        Yields the run id, or None when another live review holds the slot
        (the caller should skip). On exit the marker is always finished:
        "completed" on a clean exit, "failed" on any exception, including
        ``asyncio.CancelledError`` and other ``BaseException``\\ s that a plain
        ``except Exception`` would miss.
        """
        run_id = self.try_start(repo, pr_number, pr_title, pr_url, head_sha, ttl_seconds)
        if run_id is None:
            yield None
            return
        error: str | None = None
        try:
            yield run_id
        except BaseException as exc:
            error = str(exc) or type(exc).__name__
            raise
        finally:
            if error is None:
                self.complete(repo, pr_number, run_id=run_id)
            else:
                self.fail(repo, pr_number, error, run_id=run_id)

    def get_active(self) -> list[ReviewStatus]:
        with self._lock:
            self._evict_old()
            return [j for j in self._jobs.values() if j.status == "reviewing"]

    def get_all(self) -> list[ReviewStatus]:
        with self._lock:
            self._evict_old()
            return list(self._jobs.values())

    def _evict_old(self) -> None:
        now = time.time()
        self._jobs = {
            k: j
            for k, j in self._jobs.items()
            if j.status == "reviewing"
            or j.finished_at == 0
            or (now - j.finished_at) <= self._TTL_SECONDS
        }


tracker = ReviewTracker()
