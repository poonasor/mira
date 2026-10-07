"""Save Codex OAuth refreshes made inside Mira's throwaway Codex homes.

Each Codex CLI run gets a private temporary ``CODEX_HOME`` holding a copy of the
operator's ``auth.json`` and nothing else. ChatGPT logins use rotating refresh
tokens: when Codex refreshes during a run it writes the new tokens to that copy,
and the refresh token still in the operator's file is spent. Throwing the copy
away leaves a login that fails with "refresh token was already used".

``CodexAuthSync`` writes a refreshed copy back to the operator's file, and makes
runs that may refresh take turns so one refresh token is never spent twice.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Lock files created next to the operator's auth.json. The write lock guards the
# read-compare-replace of auth.json; the refresh lock is held by the one run that
# may refresh the login.
WRITE_LOCK_FILE = ".mira-auth.lock"
REFRESH_LOCK_FILE = ".mira-auth-refresh.lock"

# Codex CLI refreshes a ChatGPT login 5 minutes before its access token expires,
# or 8 days after last_refresh when the expiry can't be read
# (codex-rs/login/src/auth/manager.rs, should_refresh_proactively).
CODEX_REFRESH_WINDOW = timedelta(minutes=5)
CODEX_REFRESH_INTERVAL = timedelta(days=8)

WATCH_INTERVAL_SECONDS = 1.0
LEASE_POLL_SECONDS = 0.5
_MAX_AUTH_BYTES = 1024 * 1024

_warned: set[tuple[str, str]] = set()


def _warn_once(path: Path, kind: str, message: str, *args: object) -> None:
    key = (str(path), kind)
    if key in _warned:
        logger.debug(message, *args)
        return
    _warned.add(key)
    logger.warning(message, *args)


def load_login(data: bytes | None) -> dict[str, Any] | None:
    """Parse ``auth.json`` bytes; None when missing, oversized, or not a JSON object."""
    if data is None or len(data) > _MAX_AUTH_BYTES:
        return None
    try:
        value = json.loads(data)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _tokens(auth: dict[str, Any] | None) -> dict[str, Any]:
    tokens = auth.get("tokens") if auth else None
    return tokens if isinstance(tokens, dict) else {}


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _jwt_expiry(token: object) -> datetime | None:
    """Read a JWT's ``exp`` claim without verifying it, as Codex does."""
    if not isinstance(token, str) or token.count(".") != 2:
        return None
    payload = token.split(".")[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return datetime.fromtimestamp(claims["exp"], UTC)
    except (ValueError, TypeError, KeyError, OverflowError, OSError):
        return None


def may_refresh_within(
    auth: dict[str, Any] | None, horizon: timedelta, now: datetime | None = None
) -> bool:
    """Whether Codex could refresh this login before ``horizon`` has passed."""
    tokens = _tokens(auth)
    if auth is None or not tokens.get("refresh_token"):
        return False
    now = now or datetime.now(UTC)
    expires_at = _jwt_expiry(tokens.get("access_token"))
    if expires_at is not None:
        return expires_at <= now + CODEX_REFRESH_WINDOW + horizon
    last_refresh = _parse_timestamp(auth.get("last_refresh"))
    return last_refresh is not None and last_refresh + CODEX_REFRESH_INTERVAL <= now + horizon


def is_newer_login(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    """Whether ``candidate`` is a later refresh of the same login as ``current``."""
    if current is None:
        return False
    new_tokens, old_tokens = _tokens(candidate), _tokens(current)
    if not (
        new_tokens.get("refresh_token")
        and new_tokens.get("access_token")
        and old_tokens.get("refresh_token")
    ):
        return False
    if new_tokens.get("account_id") != old_tokens.get("account_id"):
        return False
    new_refresh = _parse_timestamp(candidate.get("last_refresh"))
    old_refresh = _parse_timestamp(current.get("last_refresh"))
    return new_refresh is not None and (old_refresh is None or new_refresh > old_refresh)


def _read(path: Path) -> bytes | None:
    try:
        with path.open("rb") as file:
            return file.read(_MAX_AUTH_BYTES + 1)
    except OSError:
        return None


def write_private(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with an owner-only file holding ``data``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def _open_lock(path: Path) -> int | None:
    """Open (creating if needed) a lock file; None when locking is unavailable."""
    if fcntl is None:
        return None
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return None


class CodexAuthSync:
    """Keeps one Codex run's private ``auth.json`` copy and the operator's file in step.

    Construct it after copying the source into the runtime Codex home, call
    :meth:`start` before launching Codex, run :meth:`watch` alongside it, and
    call :meth:`close` once Codex has exited, however the run ended.
    """

    def __init__(self, source: Path, runtime: Path) -> None:
        # Write through a symlinked auth.json instead of replacing the link.
        self.source = source.resolve()
        self.runtime = runtime
        self._write_lock = self.source.with_name(WRITE_LOCK_FILE)
        self._refresh_lock = self.source.with_name(REFRESH_LOCK_FILE)
        self._seen = _read(runtime)
        self._lease_fd: int | None = None

    @property
    def holds_refresh_lease(self) -> bool:
        return self._lease_fd is not None

    async def start(self, horizon: timedelta) -> None:
        """Wait for the refresh lease if Codex may refresh the login within ``horizon``."""
        if not may_refresh_within(load_login(self._seen), horizon):
            return
        deadline = time.monotonic() + (horizon + CODEX_REFRESH_WINDOW).total_seconds()
        self._lease_fd = await self._acquire_lease(deadline)
        if self._lease_fd is None:
            return
        current = _read(self.source)
        if current is not None and current != self._seen and load_login(current) is not None:
            # A run that held the lease before us saved a refreshed login; start from it.
            write_private(self.runtime, current)
            self._seen = current
        if not may_refresh_within(load_login(self._seen), horizon):
            self._release_lease()

    async def watch(self) -> None:
        """Save refreshes as they happen, so runs waiting on the lease start sooner."""
        while True:
            await asyncio.sleep(WATCH_INTERVAL_SECONDS)
            self.sync()

    def sync(self) -> None:
        """Save the private copy back to the source if Codex refreshed it."""
        data = _read(self.runtime)
        if data is None or data == self._seen:
            return
        candidate = load_login(data)
        if candidate is None:
            # Codex rewrites auth.json in place; check again once the write is complete.
            return
        self._seen = data
        self._persist(data, candidate)
        # The refresh has happened whether or not it could be saved; let the next run go.
        self._release_lease()

    def close(self) -> None:
        try:
            self.sync()
        finally:
            self._release_lease()

    def _persist(self, data: bytes, candidate: dict[str, Any]) -> bool:
        lock_fd = _open_lock(self._write_lock)
        try:
            if lock_fd is not None:
                # Held only for a read, a compare, and a rename.
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            elif fcntl is not None:
                raise PermissionError(f"cannot create {self._write_lock.name}")
            if not is_newer_login(candidate, load_login(_read(self.source))):
                logger.info(
                    "Not saving the Codex login refreshed during this run: %s holds a newer "
                    "login or a different account",
                    self.source,
                )
                return False
            write_private(self.source, data)
        except OSError as exc:
            _warn_once(
                self.source,
                "write",
                "Codex refreshed its login but Mira could not save it to %s (%s). Later Codex "
                "runs will fail until `codex login` is run again. Mount the Codex home "
                "directory writable.",
                self.source,
                exc.strerror or exc,
            )
            return False
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
        logger.info("Saved the Codex login refreshed during this run to %s", self.source)
        return True

    async def _acquire_lease(self, deadline: float) -> int | None:
        fd = _open_lock(self._refresh_lock)
        if fd is None:
            if fcntl is not None:
                _warn_once(
                    self.source,
                    "lease",
                    "Cannot create %s; concurrent Codex runs may spend the same refresh "
                    "token. Mount the Codex home directory writable.",
                    self._refresh_lock,
                )
            return None
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(LEASE_POLL_SECONDS)
        except BaseException:
            os.close(fd)
            raise
        os.close(fd)
        logger.warning(
            "Timed out waiting for another Codex run to refresh the login in %s; "
            "continuing without the refresh lock",
            self.source,
        )
        return None

    def _release_lease(self) -> None:
        if self._lease_fd is not None:
            os.close(self._lease_fd)
            self._lease_fd = None
