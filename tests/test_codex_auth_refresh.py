from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import stat
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from tenacity import stop_after_attempt

from mira.config import LLMConfig
from mira.exceptions import LLMError
from mira.llm import codex_auth
from mira.llm.codex_auth import (
    REFRESH_LOCK_FILE,
    WRITE_LOCK_FILE,
    CodexAuthSync,
    is_newer_login,
    may_refresh_within,
)
from mira.llm.codex_cli import CodexCLIProvider

HORIZON = timedelta(minutes=15)
OLD_REFRESH = "2026-09-01T08:00:00.000000Z"
NEW_REFRESH = "2026-09-14T08:00:00.123456789Z"


def _jwt(expires_at: datetime) -> str:
    def part(value: dict[str, Any]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'none'})}.{part({'exp': int(expires_at.timestamp())})}.signature"


def _login(
    refresh_token: str,
    *,
    last_refresh: str = OLD_REFRESH,
    expires_in: timedelta = timedelta(days=10),
    account_id: str = "account-1",
) -> bytes:
    auth = {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": "id-token",
            "access_token": _jwt(datetime.now(UTC) + expires_in),
            "refresh_token": refresh_token,
            "account_id": account_id,
        },
        "last_refresh": last_refresh,
    }
    return json.dumps(auth, indent=2).encode()


def _auth(refresh_token: str, **kwargs: Any) -> dict[str, Any]:
    return json.loads(_login(refresh_token, **kwargs))


def _source(tmp_path: Path, data: bytes) -> Path:
    home = tmp_path / "mounted-codex-home"
    home.mkdir()
    auth = home / "auth.json"
    auth.write_bytes(data)
    auth.chmod(0o600)
    return auth


def _run_copy(source: Path, root: Path) -> CodexAuthSync:
    """Mimic one Codex invocation: a private copy of the source auth.json."""
    runtime_home = root / "codex-home"
    runtime_home.mkdir(parents=True)
    runtime = runtime_home / "auth.json"
    runtime.write_bytes(source.read_bytes())
    return CodexAuthSync(source, runtime)


class TestCodexLoginRefreshRules:
    def test_refresh_expected_when_access_token_expires_before_run_can_end(self):
        now = datetime.now(UTC)

        assert may_refresh_within(_auth("r", expires_in=timedelta(minutes=10)), HORIZON, now)
        assert not may_refresh_within(_auth("r", expires_in=timedelta(hours=2)), HORIZON, now)

    def test_refresh_falls_back_to_codex_interval_without_readable_expiry(self):
        auth = {
            "tokens": {"access_token": "opaque", "refresh_token": "r"},
            "last_refresh": OLD_REFRESH,
        }
        refreshed_at = datetime(2026, 9, 1, 8, tzinfo=UTC)

        assert not may_refresh_within(auth, HORIZON, refreshed_at + timedelta(days=7))
        assert may_refresh_within(auth, HORIZON, refreshed_at + timedelta(days=8, minutes=-10))

    def test_api_key_login_never_refreshes(self):
        assert not may_refresh_within({"OPENAI_API_KEY": "sk-test"}, HORIZON)

    def test_only_a_later_refresh_of_the_same_account_is_newer(self):
        current = _auth("r0")

        assert is_newer_login(_auth("r1", last_refresh=NEW_REFRESH), current)
        assert not is_newer_login(_auth("r1", last_refresh=OLD_REFRESH), current)
        assert not is_newer_login(
            _auth("r1", last_refresh=NEW_REFRESH, account_id="account-2"), current
        )
        assert not is_newer_login(_auth("r1", last_refresh=NEW_REFRESH), {"OPENAI_API_KEY": "k"})
        assert not is_newer_login(_auth("r1", last_refresh=NEW_REFRESH), None)


class TestCodexLoginWriteBack:
    def test_saves_refreshed_login_atomically_with_private_permissions(self, tmp_path):
        source = _source(tmp_path, _login("refresh-token-0"))
        run = _run_copy(source, tmp_path / "run")
        refreshed = _login("refresh-token-1", last_refresh=NEW_REFRESH)
        run.runtime.write_bytes(refreshed)
        original_inode = source.stat().st_ino

        run.close()

        assert source.read_bytes() == refreshed
        assert stat.S_IMODE(source.stat().st_mode) == 0o600
        assert source.stat().st_ino != original_inode
        assert sorted(path.name for path in source.parent.iterdir()) == [
            WRITE_LOCK_FILE,
            "auth.json",
        ]

    def test_unchanged_copy_is_not_written_back(self, tmp_path):
        source = _source(tmp_path, _login("refresh-token-0"))
        run = _run_copy(source, tmp_path / "run")
        before = source.stat()

        run.close()

        after = source.stat()
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        assert not (source.parent / WRITE_LOCK_FILE).exists()

    def test_does_not_overwrite_newer_login_saved_by_another_run(self, tmp_path):
        source = _source(tmp_path, _login("refresh-token-0"))
        first = _run_copy(source, tmp_path / "first")
        second = _run_copy(source, tmp_path / "second")
        newest = _login("refresh-token-2", last_refresh="2026-09-14T09:00:00Z")
        first.runtime.write_bytes(newest)
        first.sync()

        second.runtime.write_bytes(_login("refresh-token-1", last_refresh=NEW_REFRESH))
        second.close()

        assert source.read_bytes() == newest

    def test_does_not_overwrite_login_for_a_different_account(self, tmp_path):
        source = _source(tmp_path, _login("refresh-token-0"))
        run = _run_copy(source, tmp_path / "run")
        relogin = _login("other-account-token", account_id="account-2")
        source.write_bytes(relogin)

        run.runtime.write_bytes(_login("refresh-token-1", last_refresh=NEW_REFRESH))
        run.close()

        assert source.read_bytes() == relogin

    def test_waits_for_codex_to_finish_rewriting_the_copy(self, tmp_path):
        original = _login("refresh-token-0")
        source = _source(tmp_path, original)
        run = _run_copy(source, tmp_path / "run")
        refreshed = _login("refresh-token-1", last_refresh=NEW_REFRESH)

        run.runtime.write_bytes(refreshed[: len(refreshed) // 2])
        run.sync()
        assert source.read_bytes() == original

        run.runtime.write_bytes(refreshed)
        run.sync()
        assert source.read_bytes() == refreshed

    def test_write_back_waits_for_the_login_lock(self, tmp_path):
        fcntl = pytest.importorskip("fcntl")
        source = _source(tmp_path, _login("refresh-token-0"))
        run = _run_copy(source, tmp_path / "run")
        refreshed = _login("refresh-token-1", last_refresh=NEW_REFRESH)
        run.runtime.write_bytes(refreshed)
        holder = os.open(source.parent / WRITE_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(holder, fcntl.LOCK_EX)
        release = threading.Timer(0.2, os.close, args=(holder,))
        started = time.monotonic()
        release.start()

        run.sync()

        release.join()
        assert time.monotonic() - started >= 0.15
        assert source.read_bytes() == refreshed

    def test_read_only_codex_home_warns_without_failing_or_logging_tokens(self, tmp_path, caplog):
        if os.name != "posix" or os.geteuid() == 0:
            pytest.skip("needs a non-root POSIX user for directory permissions")
        original = _login("refresh-token-0")
        source = _source(tmp_path, original)
        run = _run_copy(source, tmp_path / "run")
        run.runtime.write_bytes(_login("refresh-token-1", last_refresh=NEW_REFRESH))
        source.parent.chmod(0o500)
        try:
            with caplog.at_level(logging.WARNING, logger="mira.llm.codex_auth"):
                run.close()
        finally:
            source.parent.chmod(0o700)

        assert source.read_bytes() == original
        assert "could not save it" in caplog.text
        assert "refresh-token-1" not in caplog.text


class TestCodexRefreshLease:
    @pytest.mark.asyncio
    async def test_runs_that_may_refresh_take_turns(self, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_auth, "LEASE_POLL_SECONDS", 0.01)
        source = _source(tmp_path, _login("refresh-token-0", expires_in=timedelta(minutes=1)))
        first = _run_copy(source, tmp_path / "first")
        second = _run_copy(source, tmp_path / "second")

        await first.start(HORIZON)
        waiting = asyncio.create_task(second.start(HORIZON))
        await asyncio.sleep(0.05)

        assert first.holds_refresh_lease
        assert not waiting.done()

        refreshed = _login("refresh-token-1", last_refresh=NEW_REFRESH)
        first.runtime.write_bytes(refreshed)
        first.sync()
        await asyncio.wait_for(waiting, timeout=1)

        assert not first.holds_refresh_lease
        assert not second.holds_refresh_lease
        assert second.runtime.read_bytes() == refreshed

    @pytest.mark.asyncio
    async def test_lease_passes_on_when_a_run_ends_without_refreshing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_auth, "LEASE_POLL_SECONDS", 0.01)
        source = _source(tmp_path, _login("refresh-token-0", expires_in=timedelta(minutes=10)))
        first = _run_copy(source, tmp_path / "first")
        second = _run_copy(source, tmp_path / "second")

        await first.start(HORIZON)
        first.close()
        await asyncio.wait_for(second.start(HORIZON), timeout=1)

        assert not first.holds_refresh_lease
        assert second.holds_refresh_lease
        second.close()

    @pytest.mark.asyncio
    async def test_run_with_a_fresh_token_does_not_take_the_lease(self, tmp_path):
        source = _source(tmp_path, _login("refresh-token-0"))
        run = _run_copy(source, tmp_path / "run")

        await run.start(HORIZON)

        assert not run.holds_refresh_lease
        assert not (source.parent / REFRESH_LOCK_FILE).exists()


def _fake_codex(monkeypatch: pytest.MonkeyPatch, *, returncode: int = 0) -> list[str]:
    """Stand in for `codex exec`, refreshing its CODEX_HOME login like Codex would.

    Returns the refresh tokens spent, in order.
    """
    spent: list[str] = []

    async def spawn(*cmd: str, **kwargs: Any) -> AsyncMock:
        auth_path = Path(kwargs["env"]["CODEX_HOME"]) / "auth.json"

        async def communicate(_stdin: bytes) -> tuple[bytes, bytes]:
            auth = json.loads(auth_path.read_bytes())
            if may_refresh_within(auth, timedelta(0)):
                spent.append(auth["tokens"]["refresh_token"])
                auth_path.write_bytes(
                    _login(
                        f"refresh-token-{len(spent)}",
                        last_refresh=datetime.now(UTC).isoformat(),
                    )
                )
            await asyncio.sleep(0.2)
            return (b'{"ok": true}', b"codex failed" if returncode else b"")

        proc = AsyncMock()
        proc.returncode = returncode
        proc.communicate.side_effect = communicate
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    return spent


class TestCodexCLIProviderLoginRefresh:
    @pytest.mark.asyncio
    async def test_run_saves_login_refreshed_by_codex(self, monkeypatch, tmp_path):
        source = _source(tmp_path, _login("refresh-token-0", expires_in=timedelta(minutes=1)))
        spent = _fake_codex(monkeypatch)
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli", codex_home=str(source.parent)))

        assert await provider._run_codex("return JSON") == '{"ok": true}'

        assert spent == ["refresh-token-0"]
        assert json.loads(source.read_bytes())["tokens"]["refresh_token"] == "refresh-token-1"

    @pytest.mark.asyncio
    async def test_failed_run_still_saves_refreshed_login(self, monkeypatch, tmp_path):
        source = _source(tmp_path, _login("refresh-token-0", expires_in=timedelta(minutes=1)))
        _fake_codex(monkeypatch, returncode=1)
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli", codex_home=str(source.parent)))
        run_once = CodexCLIProvider._run_codex.retry_with(stop=stop_after_attempt(1))  # type: ignore[attr-defined]

        with pytest.raises(LLMError):
            await run_once(provider, "return JSON")

        assert json.loads(source.read_bytes())["tokens"]["refresh_token"] == "refresh-token-1"

    @pytest.mark.asyncio
    async def test_concurrent_runs_spend_the_refresh_token_once(self, monkeypatch, tmp_path):
        monkeypatch.setattr(codex_auth, "LEASE_POLL_SECONDS", 0.01)
        monkeypatch.setattr(codex_auth, "WATCH_INTERVAL_SECONDS", 0.01)
        source = _source(tmp_path, _login("refresh-token-0", expires_in=timedelta(minutes=1)))
        spent = _fake_codex(monkeypatch)
        provider = CodexCLIProvider(LLMConfig(provider="codex-cli", codex_home=str(source.parent)))

        results = await asyncio.gather(*(provider._run_codex("return JSON") for _ in range(3)))

        assert results == ['{"ok": true}'] * 3
        assert spent == ["refresh-token-0"]
        assert json.loads(source.read_bytes())["tokens"]["refresh_token"] == "refresh-token-1"
