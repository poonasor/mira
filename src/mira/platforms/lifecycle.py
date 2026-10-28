"""Process-exit logging for ``mira serve``.

A container that stops with ExitCode 0 and nothing in the logs is
undiagnosable. This module records *why* the server is going away and logs
it, along with any reviews still in flight:

- a signal (SIGTERM/SIGINT) is logged the moment uvicorn receives it;
- a ``SystemExit``/``KeyboardInterrupt`` that escapes the event loop (which
  ``uvicorn.run`` would swallow or turn into a silent exit 0) is logged with
  its traceback and re-raised;
- an ``atexit`` hook logs the final reason on every interpreter exit that
  runs Python cleanup.
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
from types import FrameType
from typing import Any

import uvicorn

logger = logging.getLogger("mira.lifecycle")

_UNKNOWN = "no signal or exception recorded (interpreter exited on its own)"
_exit_reason = _UNKNOWN
_registered = False


def set_exit_reason(reason: str) -> None:
    global _exit_reason
    _exit_reason = reason


def exit_reason() -> str:
    return _exit_reason


def active_reviews_summary() -> str:
    """``owner/repo#N (head abc1234, 42s)`` for each review still in flight."""
    try:
        import time

        from mira.core.review_status import tracker

        jobs = tracker.get_active()
    except Exception as exc:  # never let diagnostics break shutdown
        return f"unknown ({exc})"
    if not jobs:
        return "none"
    now = time.time()
    return ", ".join(
        f"{j.repo}#{j.pr_number} (head {j.head_sha[:7] or '?'}, {now - j.started_at:.0f}s)"
        for j in jobs
    )


def _log_at_exit() -> None:
    logger.warning(
        "Mira process exiting: %s; reviews in flight: %s", _exit_reason, active_reviews_summary()
    )


def install_exit_logging() -> None:
    """Register the atexit hook once (it runs before logging's own shutdown)."""
    global _registered
    if not _registered:
        atexit.register(_log_at_exit)
        _registered = True


class LoggingServer(uvicorn.Server):
    """uvicorn server that logs the signal that stops it."""

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = str(sig)
        set_exit_reason(f"received {name}")
        logger.warning(
            "Received %s; shutting down. Reviews in flight: %s", name, active_reviews_summary()
        )
        super().handle_exit(sig, frame)


def run_server(app: Any, host: str, port: int) -> None:
    """``uvicorn.run`` equivalent that never exits without saying why."""
    install_exit_logging()
    server = LoggingServer(uvicorn.Config(app, host=host, port=port))
    try:
        server.run()
    except BaseException as exc:
        set_exit_reason(f"{type(exc).__name__} escaped the event loop")
        logger.critical(
            "Mira server stopped by %s escaping the event loop; reviews in flight: %s",
            type(exc).__name__,
            active_reviews_summary(),
            exc_info=True,
        )
        raise
    if exit_reason() == _UNKNOWN:
        set_exit_reason("server loop returned without a signal")
    if not server.started:
        # Same contract as uvicorn.run: a failed startup exits non-zero.
        sys.exit(3)
