"""Tests for the read-only failover status endpoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mira.config import LLMConfig, MiraConfig
from mira.dashboard.db import AppDatabase
from mira.dashboard.routers.admin import get_failover_status
from mira.llm import tiered
from mira.llm.tiered import tier_key

PRIMARY = LLMConfig(
    base_url="https://api.z.ai/api/paas/v4",
    api_key_env="ZAI_API_KEY",
    model="glm-5.2",
)
WITH_FAILOVER = PRIMARY.model_copy(
    update={
        "failover_cooldown_seconds": 600,
        "failover_primary_max_retries": 1,
        "failover": LLMConfig(provider="claude-cli", model="sonnet"),
    }
)


def _request(is_admin: bool = True) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(user=SimpleNamespace(is_admin=is_admin)))


@pytest.fixture
def in_memory_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AppDatabase:
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    db = AppDatabase(url="", admin_password="admin")
    monkeypatch.setattr("mira.dashboard.api._app_db", db)
    return db


@pytest.fixture(autouse=True)
def clean_cooldowns():
    tiered._cooldowns.clear()
    yield
    tiered._cooldowns.clear()


def _use_config(monkeypatch: pytest.MonkeyPatch, llm: LLMConfig) -> None:
    monkeypatch.setattr("mira.config.load_config", lambda *a, **kw: MiraConfig(llm=llm))


class TestFailoverStatus:
    def test_requires_admin(self, in_memory_db: AppDatabase):
        with pytest.raises(HTTPException) as info:
            get_failover_status(_request(is_admin=False))

        assert info.value.status_code == 403

    def test_without_failover_lists_only_the_primary(
        self, in_memory_db: AppDatabase, monkeypatch: pytest.MonkeyPatch
    ):
        _use_config(monkeypatch, PRIMARY)

        status = get_failover_status(_request())

        assert status.enabled is False
        assert [tier.tier for tier in status.tiers] == [1]
        assert status.tiers[0].backend == "zai"
        assert status.tiers[0].cooldown_remaining_seconds == 0

    def test_dashboard_model_override_shows_on_the_primary_only(
        self, in_memory_db: AppDatabase, monkeypatch: pytest.MonkeyPatch
    ):
        in_memory_db.set_setting("review_model", "glm-5.3")
        _use_config(monkeypatch, WITH_FAILOVER)

        status = get_failover_status(_request())

        assert status.enabled is True
        assert status.cooldown_seconds == 600
        assert status.primary_max_retries == 1
        primary, claude = status.tiers
        assert (primary.tier, primary.backend, primary.review_model, primary.indexing_model) == (
            1,
            "zai",
            "glm-5.3",
            "glm-5.2",
        )
        assert (claude.tier, claude.backend, claude.review_model, claude.indexing_model) == (
            2,
            "claude-cli",
            "sonnet",
            "sonnet",
        )

    def test_reports_live_cooldown_rounded_up(
        self, in_memory_db: AppDatabase, monkeypatch: pytest.MonkeyPatch
    ):
        _use_config(monkeypatch, WITH_FAILOVER)
        monkeypatch.setattr(tiered, "_now", lambda: 1000.0)
        tiered._cooldowns[tier_key(PRIMARY)] = 1000.0 + 299.2

        primary, claude = get_failover_status(_request()).tiers

        assert primary.cooldown_remaining_seconds == 300
        assert claude.cooldown_remaining_seconds == 0

    def test_response_exposes_no_endpoints_credentials_or_commands(
        self, in_memory_db: AppDatabase, monkeypatch: pytest.MonkeyPatch
    ):
        _use_config(monkeypatch, WITH_FAILOVER)

        body = get_failover_status(_request()).model_dump_json()

        for leaked in (
            "base_url",
            "api.z.ai",
            "api_key_env",
            "ZAI_API_KEY",
            "claude_oauth_token_env",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "command",
        ):
            assert leaked not in body
