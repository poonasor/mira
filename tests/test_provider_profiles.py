"""Tests for the provider-profile registry (mira.llm.provider_profiles)."""

from __future__ import annotations

import json

import pytest

from mira.llm import provider_profiles as profiles

ZAI_GENERAL = "https://api.z.ai/api/paas/v4"
ZAI_CODING = "https://api.z.ai/api/coding/paas/v4"


@pytest.fixture
def providers_override(tmp_path, monkeypatch):
    """Point MIRA_PROVIDERS_JSON_PATH at a providers.json built from a dict."""

    def install(data: dict) -> None:
        custom = tmp_path / "providers.json"
        custom.write_text(json.dumps(data))
        monkeypatch.setenv("MIRA_PROVIDERS_JSON_PATH", str(custom))
        profiles._load.cache_clear()

    yield install
    profiles._load.cache_clear()


class TestResolve:
    def test_matches_openrouter_by_base_url(self):
        p = profiles.resolve("https://openrouter.ai/api/v1")
        assert p["name"] == "openrouter"
        assert p["model_prefix"] == "keep"
        assert p["extra_headers"]["X-Title"] == "Mira Code Reviewer"
        assert p["reasoning_effort_map"] == {"max": "xhigh"}
        assert p["api_styles"] == ["chat", "responses"]
        assert p["reasoning_style"] == "nested"
        assert p["supports_forced_tool_choice"] is True

    def test_matches_zai_by_base_url(self):
        p = profiles.resolve(ZAI_GENERAL)
        assert p["name"] == "zai"
        assert p["api_key_env"] == "ZAI_API_KEY"
        assert p["api_styles"] == ["chat"]
        assert p["model_prefix"] == "strip"
        assert p["reasoning_style"] == "zai"
        assert p["supports_forced_tool_choice"] is False

    @pytest.mark.parametrize("base_url", [ZAI_CODING, f"{ZAI_CODING}/"])
    def test_matches_zai_coding_plan_endpoint(self, base_url: str):
        # The Coding Plan endpoint shares the general API's wire format, so it
        # gets the identical Z.AI profile rather than the portable default.
        p = profiles.resolve(base_url)
        assert p["name"] == "zai"
        assert p == profiles.resolve(ZAI_GENERAL)

    @pytest.mark.parametrize(
        "base_url", ["https://api.z.ai/api/coding/paas", "https://api.z.ai/api/anthropic"]
    )
    def test_other_zai_paths_do_not_match(self, base_url: str):
        assert profiles.resolve(base_url)["name"] == ""

    def test_trailing_slash_insensitive(self):
        assert profiles.resolve("https://openrouter.ai/api/v1/")["name"] == "openrouter"

    def test_unknown_url_returns_portable_default(self):
        p = profiles.resolve("https://some-new-llm.example/v1")
        assert p["name"] == ""
        assert p["model_prefix"] == "strip"
        assert p["extra_headers"] == {}
        assert p["reasoning_effort_map"] == {}
        assert p["api_styles"] == ["chat", "responses"]
        assert p["reasoning_style"] == "nested"
        assert p["supports_forced_tool_choice"] is True

    def test_sparse_profile_fills_from_default(self, tmp_path, monkeypatch):
        # A profile with only base_url + api_key_env still resolves with every
        # field, filled in from DEFAULT_PROFILE.
        custom = tmp_path / "providers.json"
        custom.write_text(
            json.dumps({"sparse": {"base_url": "https://sparse.test/v1", "api_key_env": "K"}})
        )
        monkeypatch.setenv("MIRA_PROVIDERS_JSON_PATH", str(custom))
        profiles._load.cache_clear()
        try:
            p = profiles.resolve("https://sparse.test/v1")
            assert p["name"] == "sparse"
            assert p["model_prefix"] == "strip"
            assert p["extra_headers"] == {}
            assert p["reasoning_effort_map"] == {}
            assert p["api_styles"] == ["chat", "responses"]
            assert p["reasoning_style"] == "nested"
            assert p["supports_forced_tool_choice"] is True
        finally:
            profiles._load.cache_clear()


class TestGet:
    def test_injects_name(self):
        assert profiles.get("openrouter")["name"] == "openrouter"

    def test_missing_returns_none(self):
        assert profiles.get("not-a-provider") is None


class TestRuntimeOverride:
    """A user can add or override a provider at runtime, no code change."""

    def test_override_file_adds_provider(self, tmp_path, monkeypatch):
        custom = tmp_path / "providers.json"
        custom.write_text(
            json.dumps(
                {
                    "acme": {
                        "base_url": "https://llm.acme.test/v1",
                        "api_key_env": "ACME_API_KEY",
                        "model_prefix": "keep",
                        "extra_headers": {"X-Acme": "1"},
                    }
                }
            )
        )
        monkeypatch.setenv("MIRA_PROVIDERS_JSON_PATH", str(custom))
        profiles._load.cache_clear()
        try:
            p = profiles.resolve("https://llm.acme.test/v1")
            assert p["name"] == "acme"
            assert p["model_prefix"] == "keep"
            assert p["extra_headers"] == {"X-Acme": "1"}
            # Bundled profiles still resolve alongside the override.
            assert profiles.resolve("https://openrouter.ai/api/v1")["name"] == "openrouter"
        finally:
            profiles._load.cache_clear()

    def test_missing_override_file_falls_back(self, monkeypatch):
        monkeypatch.setenv("MIRA_PROVIDERS_JSON_PATH", "/no/such/providers.json")
        profiles._load.cache_clear()
        try:
            assert profiles.resolve("https://openrouter.ai/api/v1")["name"] == "openrouter"
        finally:
            profiles._load.cache_clear()

    def test_override_profile_can_list_alternate_base_urls(self, providers_override):
        providers_override(
            {
                "acme": {
                    "base_url": "https://llm.acme.test/v1",
                    "base_urls": ["https://eu.llm.acme.test/v1"],
                    "model_prefix": "keep",
                }
            }
        )
        assert profiles.resolve("https://llm.acme.test/v1")["name"] == "acme"
        p = profiles.resolve("https://eu.llm.acme.test/v1/")
        assert p["name"] == "acme"
        assert p["model_prefix"] == "keep"
        # Bundled aliases keep resolving alongside the override.
        assert profiles.resolve(ZAI_CODING)["name"] == "zai"

    def test_single_string_base_urls_is_accepted(self, providers_override):
        providers_override(
            {
                "acme": {
                    "base_url": "https://llm.acme.test/v1",
                    "base_urls": "https://eu.llm.acme.test/v1",
                }
            }
        )
        assert profiles.resolve("https://eu.llm.acme.test/v1")["name"] == "acme"
        # The string is one URL, not an iterable of one-character URLs.
        assert profiles.resolve("h")["name"] == ""

    def test_same_name_override_replaces_bundled_aliases(self, providers_override):
        # An override replaces the bundled profile wholesale, so redefining
        # "zai" without base_urls drops the Coding Plan alias.
        providers_override({"zai": {"base_url": ZAI_GENERAL, "supports_forced_tool_choice": True}})
        assert profiles.resolve(ZAI_GENERAL)["supports_forced_tool_choice"] is True
        assert profiles.resolve(ZAI_CODING)["name"] == ""

    def test_override_profile_wins_over_bundled_alias(self, providers_override):
        # An operator profile for the Coding Plan URL (e.g. a workaround written
        # before the bundled alias existed) keeps applying.
        providers_override({"zai-coding": {"base_url": ZAI_CODING, "reasoning_style": "nested"}})
        p = profiles.resolve(ZAI_CODING)
        assert p["name"] == "zai-coding"
        assert p["reasoning_style"] == "nested"
        assert profiles.resolve(ZAI_GENERAL)["name"] == "zai"

    def test_override_profile_wins_over_bundled_base_url(self, providers_override):
        providers_override(
            {"my-router": {"base_url": "https://openrouter.ai/api/v1", "model_prefix": "strip"}}
        )
        p = profiles.resolve("https://openrouter.ai/api/v1")
        assert p["name"] == "my-router"
        assert p["model_prefix"] == "strip"
