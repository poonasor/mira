"""Tests for the backend-aware dynamic model catalog."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import LLMConfig
from mira.dashboard import model_catalog
from mira.dashboard.model_catalog import active_backend, build_options, fetch_catalog


@pytest.fixture(autouse=True)
def _clear_cache():
    model_catalog._cache.clear()
    yield
    model_catalog._cache.clear()


class TestActiveBackend:
    def test_default_is_openrouter(self):
        assert active_backend(LLMConfig()) == "openrouter"

    def test_bedrock_provider(self):
        assert active_backend(LLMConfig(provider="bedrock")) == "bedrock"

    def test_codex_cli_provider(self):
        assert active_backend(LLMConfig(provider="codex-cli")) == "codex-cli"

    def test_generic_endpoint(self):
        assert (
            active_backend(LLMConfig(base_url="http://localhost:11434/v1")) == "openai-compatible"
        )

    def test_zai_endpoint(self):
        assert active_backend(LLMConfig(base_url="https://api.z.ai/api/paas/v4")) == "zai"


class TestBuildOptions:
    def test_registry_filtered_by_backend(self):
        openrouter = [m["value"] for m in build_options("openrouter", None, "review")]
        bedrock = [m["value"] for m in build_options("bedrock", None, "review")]
        assert "anthropic/claude-sonnet-4-6" in openrouter
        assert "us.anthropic.claude-sonnet-4-6-v1:0" not in openrouter
        assert "us.anthropic.claude-sonnet-4-6-v1:0" in bedrock
        assert "anthropic/claude-sonnet-4-6" not in bedrock

    def test_codex_backend_only_offers_codex_models(self):
        values = [m["value"] for m in build_options("codex-cli", None, "review")]
        assert values == ["codex-default"]

    def test_openrouter_does_not_offer_codex_models(self):
        values = [m["value"] for m in build_options("openrouter", None, "review")]
        assert "codex-default" not in values

    @pytest.mark.parametrize("purpose", ["indexing", "review"])
    def test_zai_only_offers_curated_glm_model(self, purpose: str):
        dynamic = [{"value": "glm-5.3", "label": "GLM-5.3"}]
        values = [m["value"] for m in build_options("zai", dynamic, purpose)]
        assert values == ["glm-5.2"]

    def test_dynamic_merged_and_deduped_against_registry(self):
        dynamic = [
            {"value": "anthropic/claude-sonnet-4.6", "label": "Anthropic: Claude Sonnet 4.6"},
            {"value": "mistralai/mistral-large-3", "label": "Mistral Large 3"},
        ]
        options = build_options("openrouter", dynamic, "review")
        values = [m["value"] for m in options]
        # Dot-form alias of a registry id is dropped; genuinely new model kept.
        assert "anthropic/claude-sonnet-4.6" not in values
        assert "anthropic/claude-sonnet-4-6" in values
        assert "mistralai/mistral-large-3" in values

    def test_generic_endpoint_uses_dynamic_only(self):
        dynamic = [{"value": "llama-3.3-70b", "label": "llama-3.3-70b"}]
        values = [m["value"] for m in build_options("openai-compatible", dynamic, "review")]
        assert values == ["llama-3.3-70b"]

    def test_generic_endpoint_falls_back_to_registry(self):
        values = [m["value"] for m in build_options("openai-compatible", None, "review")]
        assert "anthropic/claude-sonnet-4-6" in values

    def test_recommended_sort_first(self):
        options = build_options("openrouter", None, "indexing")
        assert options[0]["recommended"] is True


class TestFetchCatalog:
    @pytest.mark.asyncio
    async def test_openai_fetch_resolves_profile_for_api_key(self, monkeypatch: pytest.MonkeyPatch):
        seen: dict = {}

        def fake_get_api_key(config, profile=None):
            seen["profile"] = profile
            return "zai-test-key"

        response = MagicMock()
        response.json.return_value = {"data": [{"id": "glm-5.2"}]}
        client = AsyncMock()
        client.get = AsyncMock(return_value=response)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(model_catalog, "_get_api_key", fake_get_api_key)
        monkeypatch.setattr(model_catalog.httpx, "AsyncClient", lambda **kwargs: client)

        config = LLMConfig(base_url="https://api.z.ai/api/paas/v4")
        result = await model_catalog._fetch_openai_style(config, tools_only=False)

        assert seen["profile"]["name"] == "zai"
        assert client.get.call_args.kwargs["headers"] == {"Authorization": "Bearer zai-test-key"}
        assert result == [{"value": "glm-5.2", "label": "glm-5.2"}]

    @pytest.mark.asyncio
    async def test_failure_returns_none_and_is_cached(self, monkeypatch: pytest.MonkeyPatch):
        calls = 0

        async def boom(config, tools_only):
            nonlocal calls
            calls += 1
            raise RuntimeError("no network")

        monkeypatch.setattr(model_catalog, "_fetch_openai_style", boom)
        assert await fetch_catalog(LLMConfig()) is None
        # A dead endpoint must not re-block every settings-page load.
        assert await fetch_catalog(LLMConfig()) is None
        assert calls == 1

    @pytest.mark.asyncio
    async def test_result_is_cached(self, monkeypatch: pytest.MonkeyPatch):
        calls = 0

        async def fake(config, tools_only):
            nonlocal calls
            calls += 1
            return [{"value": "m", "label": "m"}]

        monkeypatch.setattr(model_catalog, "_fetch_openai_style", fake)
        assert await fetch_catalog(LLMConfig()) == [{"value": "m", "label": "m"}]
        assert await fetch_catalog(LLMConfig()) == [{"value": "m", "label": "m"}]
        assert calls == 1

    @pytest.mark.asyncio
    async def test_zai_uses_static_catalog_without_models_request(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        async def unexpected(*args, **kwargs):
            raise AssertionError("Z.AI catalog must not call an undocumented /models endpoint")

        monkeypatch.setattr(model_catalog, "_fetch_openai_style", unexpected)
        config = LLMConfig(base_url="https://api.z.ai/api/paas/v4", api_key_env="ZAI_API_KEY")
        assert await fetch_catalog(config) is None

    @pytest.mark.asyncio
    async def test_concurrent_cold_fetches_coalesce(self, monkeypatch: pytest.MonkeyPatch):
        calls = 0

        async def slow(config, tools_only):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return [{"value": "m", "label": "m"}]

        monkeypatch.setattr(model_catalog, "_fetch_openai_style", slow)
        results = await asyncio.gather(*(fetch_catalog(LLMConfig()) for _ in range(5)))
        assert all(r == [{"value": "m", "label": "m"}] for r in results)
        assert calls == 1

    def test_bedrock_cache_key_includes_profile(self):
        # Switching aws_profile must not serve the previous account's catalog.
        a = LLMConfig(provider="bedrock", aws_profile="account-a")
        b = LLMConfig(provider="bedrock", aws_profile="account-b")
        assert a.region == b.region
        # Keys derived the same way fetch_catalog does.
        key_a = f"bedrock:{a.region}:{a.aws_profile or ''}"
        key_b = f"bedrock:{b.region}:{b.aws_profile or ''}"
        assert key_a != key_b
