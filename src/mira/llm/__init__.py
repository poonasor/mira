"""LLM provider package — factory entry point."""

from __future__ import annotations

from mira.config import LLMConfig
from mira.llm import provider_profiles as profiles
from mira.llm.base import LLMProviderProtocol


def create_llm(config: LLMConfig) -> LLMProviderProtocol:
    """Create the appropriate LLM provider based on config.provider.

    Returns an instance satisfying LLMProviderProtocol. When ``config.failover``
    is set, returns a ``TieredProvider`` that fails over from this provider to it.
    """
    if config.failover is not None:
        from mira.llm.tiered import TieredProvider

        primary = config.model_copy(
            update={
                "failover": None,
                "max_retries": min(config.max_retries, config.failover_primary_max_retries),
            }
        )
        return TieredProvider(
            [(primary, create_llm(primary)), (config.failover, create_llm(config.failover))],
            cooldown_seconds=config.failover_cooldown_seconds,
        )

    if config.provider == "bedrock":
        from mira.llm.bedrock import BedrockProvider

        return BedrockProvider(config)

    if config.provider in {"codex-cli", "codex_cli", "codex"}:
        from mira.llm.codex_cli import CodexCLIProvider

        return CodexCLIProvider(config)

    if config.provider in {"claude-cli", "claude_cli", "claude"}:
        from mira.llm.claude_cli import ClaudeCLIProvider

        return ClaudeCLIProvider(config)

    profile = profiles.resolve(config.base_url)
    if config.api_style == "responses" and "responses" in profile.get("api_styles", []):
        from mira.llm.responses import ResponsesProvider

        return ResponsesProvider(config)

    # Default: OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, etc.)
    from mira.llm.provider import LLMProvider

    return LLMProvider(config)
