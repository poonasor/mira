"""Cross-provider failover: serve each call from the first provider tier that can.

A tier that fails because of its own health (rate limit, outage, auth, timeout)
is cooled down process-wide, so later calls skip it until the cooldown expires.
A failure specific to one request fails over without a cooldown, since a
different provider may well accept that request.
"""

from __future__ import annotations

import logging
import time

import httpx

from mira.config import LLMConfig
from mira.exceptions import LLMError, MiraError
from mira.llm.base import LLMProviderProtocol

logger = logging.getLogger(__name__)

_COOLING_CODES = frozenset(
    {
        "no_api_key",
        "claude_token_missing",
        "claude_auth_failed",
        "claude_usage_limit",
        "claude_timeout",
        "claude_command_not_found",
        "codex_auth_file_missing",
        "codex_command_not_found",
        "codex_timeout",
    }
)
_COOLING_STATUSES = frozenset({401, 403, 404, 429})

# Tier key → monotonic expiry. Module-level because providers are rebuilt per
# request, and `mira serve` runs every review on one event loop.
_cooldowns: dict[tuple[str, ...], float] = {}


def _now() -> float:
    return time.monotonic()


def tier_key(config: LLMConfig) -> tuple[str, ...]:
    """Identify the account behind a tier; limits apply per credential, not model."""
    from mira.llm.claude_cli import PROVIDER_NAMES as CLAUDE_PROVIDER_NAMES

    if config.provider in CLAUDE_PROVIDER_NAMES:
        return ("claude-cli", config.claude_oauth_token_env)
    if config.provider in {"codex-cli", "codex_cli", "codex"}:
        return ("codex-cli", config.codex_home or "")
    if config.provider == "bedrock":
        return ("bedrock", config.region, config.aws_profile or "")
    return (config.provider, config.base_url, config.api_key_env)


def is_health_failure(exc: BaseException) -> bool:
    """Whether a failure reflects the provider's state rather than this request."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (httpx.TimeoutException, httpx.NetworkError)):
            return True
        if isinstance(current, LLMError):
            if current.code in _COOLING_CODES:
                return True
            status = current.details.get("status")
            if isinstance(status, int) and (status in _COOLING_STATUSES or status >= 500):
                return True
        current = current.__cause__ or current.__context__
    return False


def _prompt_text(messages: list) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages if isinstance(m, dict))


class TieredProvider:
    """An LLM provider whose calls fail over across an ordered list of providers."""

    supports_json_mode = True

    def __init__(
        self,
        tiers: list[tuple[LLMConfig, LLMProviderProtocol]],
        cooldown_seconds: int,
    ) -> None:
        self.tiers = tiers
        self.cooldown_seconds = cooldown_seconds
        self.config = tiers[0][0]
        self.last_tier: str | None = None
        self.supports_tool_calling = any(
            getattr(provider, "supports_tool_calling", False) for _, provider in tiers
        )
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._sync_usage()

    @property
    def supports_temperature(self) -> bool:
        _, _, provider = self._candidates()[0]
        return getattr(provider, "supports_temperature", True)

    def _sync_usage(self) -> None:
        self.total_prompt_tokens = sum(provider.total_prompt_tokens for _, provider in self.tiers)
        self.total_completion_tokens = sum(
            provider.total_completion_tokens for _, provider in self.tiers
        )

    @property
    def usage(self) -> dict[str, int]:
        self._sync_usage()
        prompt, completion = self.total_prompt_tokens, self.total_completion_tokens
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def count_tokens(self, text: str) -> int:
        return self.tiers[0][1].count_tokens(text)

    def _candidates(self) -> list[tuple[int, LLMConfig, LLMProviderProtocol]]:
        """Tiers that aren't cooling down, in order; every tier if all are cooling."""
        now = _now()
        indexed = [(i, cfg, provider) for i, (cfg, provider) in enumerate(self.tiers)]
        ready = [tier for tier in indexed if _cooldowns.get(tier_key(tier[1]), 0.0) <= now]
        return ready or indexed

    async def _dispatch(
        self, method: str, messages: list, *args: object, agentic: bool = False, **kwargs: object
    ) -> object:
        prompt = _prompt_text(messages)
        errors: list[str] = []
        last_exc: Exception | None = None
        for index, cfg, provider in self._candidates():
            name = f"tier {index + 1} ({cfg.provider}:{cfg.model})"
            if agentic and not getattr(provider, "supports_tool_calling", False):
                continue
            if provider.count_tokens(prompt) > cfg.max_context_tokens:
                logger.warning(
                    "Skipping %s: prompt exceeds its %d-token context", name, cfg.max_context_tokens
                )
                errors.append(f"{name}: prompt exceeds context window")
                continue
            try:
                result = await getattr(provider, method)(messages, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                errors.append(
                    f"{name}: {exc.safe_message if isinstance(exc, MiraError) else type(exc).__name__}"
                )
                if is_health_failure(exc):
                    _cooldowns[tier_key(cfg)] = _now() + self.cooldown_seconds
                    logger.warning(
                        "%s failed (%s); cooling it down for %ds and failing over",
                        name,
                        exc,
                        self.cooldown_seconds,
                    )
                else:
                    logger.warning("%s failed (%s); failing over", name, exc)
                continue
            finally:
                self._sync_usage()
            self.last_tier = name
            logger.info("LLM %s served by %s", method, name)
            return result

        if agentic and not errors:
            return {"content": "", "tool_calls": []}
        raise LLMError(
            "all_providers_failed", errors="; ".join(errors) or "no eligible provider"
        ) from last_exc

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        result = await self._dispatch(
            "complete",
            messages,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return str(result)

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        result = await self._dispatch(
            "complete_with_tools", messages, tools=tools, temperature=temperature
        )
        return str(result)

    async def complete_agentic(
        self,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        result = await self._dispatch(
            "complete_agentic", messages, tools=tools, temperature=temperature, agentic=True
        )
        return result if isinstance(result, dict) else {"content": "", "tool_calls": []}

    async def review(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        return str(await self._dispatch("review", messages, temperature=temperature))

    async def walkthrough(self, messages: list[dict[str, str]]) -> str:
        return str(await self._dispatch("walkthrough", messages))
