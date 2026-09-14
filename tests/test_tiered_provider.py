from __future__ import annotations

import asyncio

import httpx
import pytest

from mira.config import LLMConfig
from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm import create_llm, tiered
from mira.llm.claude_cli import ClaudeCLIProvider
from mira.llm.tiered import TieredProvider, is_health_failure, tier_key

ZAI = LLMConfig(
    provider="openai",
    base_url="https://api.z.ai/api/coding/paas/v4",
    api_key_env="ZAI_API_KEY",
    model="glm-5.2",
    max_context_tokens=1_000_000,
)
CLAUDE = LLMConfig(provider="claude-cli", model="sonnet", max_context_tokens=200_000)
MESSAGES = [{"role": "user", "content": "review this"}]
DEFERRED = {"content": "", "tool_calls": []}


class FakeProvider:
    supports_json_mode = True

    def __init__(
        self,
        result: object = "ok",
        error: BaseException | None = None,
        *,
        tools: bool = True,
        temperature: bool = True,
        usage: tuple[int, int] = (0, 0),
    ) -> None:
        self.result = result
        self.error = error
        self.supports_tool_calling = tools
        self.supports_temperature = temperature
        self.total_prompt_tokens, self.total_completion_tokens = usage
        self.calls = 0

    async def _call(self, *args: object, **kwargs: object) -> object:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result

    complete = complete_with_tools = complete_agentic = review = walkthrough = _call

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }


def wrapped(inner: BaseException) -> LLMError:
    """The shape OpenAI-compatible providers raise after exhausting retries."""
    outer = LLMError("tool_call_failed", model="glm-5.2", error=inner)
    outer.__cause__ = inner
    return outer


def http_failure(status: int) -> LLMError:
    cls = NonRetriableLLMError if 400 <= status < 500 and status != 429 else LLMError
    return wrapped(cls("api_error", status=status, body="upstream refused"))


def pair(primary: FakeProvider, secondary: FakeProvider, cooldown: int = 600) -> TieredProvider:
    return TieredProvider([(ZAI, primary), (CLAUDE, secondary)], cooldown_seconds=cooldown)


@pytest.fixture(autouse=True)
def clean_cooldowns():
    tiered._cooldowns.clear()
    yield
    tiered._cooldowns.clear()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(tiered, "_now", lambda: now[0])
    return now


class TestFailover:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            http_failure(429),
            http_failure(500),
            http_failure(503),
            http_failure(401),
            http_failure(403),
            wrapped(httpx.ReadTimeout("slow")),
            wrapped(httpx.ConnectError("down")),
            LLMError("no_api_key", api_key_env="ZAI_API_KEY"),
        ],
        ids=["429", "500", "503", "401", "403", "timeout", "network", "no-key"],
    )
    async def test_health_failure_fails_over_and_cools_the_tier(
        self, error: LLMError, clock: list[float]
    ):
        primary, secondary = FakeProvider(error=error), FakeProvider(result="from claude")
        provider = pair(primary, secondary)

        assert await provider.review(MESSAGES) == "from claude"
        assert await provider.review(MESSAGES) == "from claude"

        assert primary.calls == 1
        assert secondary.calls == 2
        assert provider.last_tier == "tier 2 (claude-cli:sonnet)"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [http_failure(400), http_failure(422), LLMError("no_tool_call")],
        ids=["400", "422", "no-tool-call"],
    )
    async def test_request_failure_fails_over_without_cooldown(
        self, error: LLMError, clock: list[float]
    ):
        primary, secondary = FakeProvider(error=error), FakeProvider(result="from claude")
        provider = pair(primary, secondary)

        assert await provider.review(MESSAGES) == "from claude"
        assert await provider.review(MESSAGES) == "from claude"

        assert primary.calls == 2

    @pytest.mark.asyncio
    async def test_primary_serves_while_healthy(self, clock: list[float]):
        primary, secondary = FakeProvider(result="from zai"), FakeProvider(result="from claude")
        provider = pair(primary, secondary)

        assert await provider.complete(MESSAGES) == "from zai"
        assert secondary.calls == 0
        assert provider.last_tier == "tier 1 (openai:glm-5.2)"

    @pytest.mark.asyncio
    async def test_all_tiers_failing_raises_all_providers_failed(self, clock: list[float]):
        provider = pair(
            FakeProvider(error=http_failure(503)),
            FakeProvider(error=LLMError("claude_usage_limit", status=429, detail="limit")),
        )

        with pytest.raises(LLMError) as info:
            await provider.review(MESSAGES)

        assert info.value.code == "all_providers_failed"
        assert "tier 1" in str(info.value)
        assert "tier 2" in str(info.value)

    @pytest.mark.asyncio
    async def test_cancellation_propagates_without_failover_or_cooldown(self, clock: list[float]):
        secondary = FakeProvider(result="from claude")
        provider = pair(FakeProvider(error=asyncio.CancelledError()), secondary)

        with pytest.raises(asyncio.CancelledError):
            await provider.review(MESSAGES)

        assert secondary.calls == 0
        assert tiered._cooldowns == {}

    @pytest.mark.asyncio
    async def test_prompt_larger_than_a_tier_context_skips_that_tier(self, clock: list[float]):
        small_claude = CLAUDE.model_copy(update={"max_context_tokens": 1})
        secondary = FakeProvider(result="from claude")
        provider = TieredProvider(
            [(ZAI, FakeProvider(error=http_failure(503))), (small_claude, secondary)],
            cooldown_seconds=600,
        )

        with pytest.raises(LLMError) as info:
            await provider.review(MESSAGES)

        assert info.value.code == "all_providers_failed"
        assert "exceeds context" in str(info.value)
        assert secondary.calls == 0


class TestCooldown:
    @pytest.mark.asyncio
    async def test_cooldown_expires(self, clock: list[float]):
        primary, secondary = FakeProvider(error=http_failure(429)), FakeProvider(result="claude")
        provider = pair(primary, secondary, cooldown=600)
        await provider.review(MESSAGES)

        clock[0] += 599
        await provider.review(MESSAGES)
        assert primary.calls == 1

        clock[0] += 2
        primary.error, primary.result = None, "zai"
        assert await provider.review(MESSAGES) == "zai"
        assert primary.calls == 2

    @pytest.mark.asyncio
    async def test_cooldown_is_shared_by_providers_built_later(self, clock: list[float]):
        await pair(FakeProvider(error=http_failure(429)), FakeProvider()).review(MESSAGES)

        primary = FakeProvider(result="zai")
        assert await pair(primary, FakeProvider(result="claude")).review(MESSAGES) == "claude"
        assert primary.calls == 0

    @pytest.mark.asyncio
    async def test_every_tier_cooling_tries_them_in_order(self, clock: list[float]):
        tiered._cooldowns[tier_key(ZAI)] = clock[0] + 100
        tiered._cooldowns[tier_key(CLAUDE)] = clock[0] + 100
        primary, secondary = FakeProvider(result="zai"), FakeProvider(result="claude")

        assert await pair(primary, secondary).review(MESSAGES) == "zai"
        assert secondary.calls == 0

    def test_tier_key_is_per_account_not_per_model(self):
        assert tier_key(ZAI) == tier_key(ZAI.model_copy(update={"model": "glm-4.6"}))
        assert tier_key(CLAUDE) == ("claude-cli", "CLAUDE_CODE_OAUTH_TOKEN")

    def test_cooldown_remaining_counts_down_to_zero(self, clock: list[float]):
        assert tiered.cooldown_remaining(ZAI) == 0.0

        tiered._cooldowns[tier_key(ZAI)] = clock[0] + 600
        clock[0] += 150
        assert tiered.cooldown_remaining(ZAI) == 450.0
        assert tiered.cooldown_remaining(CLAUDE) == 0.0

        clock[0] += 451
        assert tiered.cooldown_remaining(ZAI) == 0.0

    def test_health_classification_reads_status_through_the_cause_chain(self):
        assert is_health_failure(http_failure(429))
        assert not is_health_failure(http_failure(400))
        assert is_health_failure(LLMError("claude_auth_failed", status=None, detail="x"))


class TestAgentic:
    @pytest.mark.asyncio
    async def test_agentic_never_routes_to_a_tier_without_tool_calling(self, clock: list[float]):
        secondary = FakeProvider(result={"content": "x"}, tools=False)
        provider = pair(FakeProvider(error=http_failure(503)), secondary)

        with pytest.raises(LLMError):
            await provider.complete_agentic(MESSAGES, tools=[])
        # Primary is now cooling and no tool-capable tier is left: defer to a
        # single forced review instead of failing.
        assert await provider.complete_agentic(MESSAGES, tools=[]) == DEFERRED
        assert secondary.calls == 0


class TestProtocolSurface:
    def test_usage_sums_every_tier(self):
        provider = pair(FakeProvider(usage=(10, 2)), FakeProvider(usage=(5, 1)))

        assert provider.usage == {"prompt_tokens": 15, "completion_tokens": 3, "total_tokens": 18}
        assert provider.config is ZAI

    def test_supports_temperature_follows_the_first_ready_tier(self, clock: list[float]):
        provider = pair(FakeProvider(temperature=True), FakeProvider(temperature=False))
        assert provider.supports_temperature is True

        tiered._cooldowns[tier_key(ZAI)] = clock[0] + 60
        assert provider.supports_temperature is False

    def test_create_llm_builds_tiers_with_capped_primary_retries(self):
        config = ZAI.model_copy(
            update={"max_retries": 3, "failover": CLAUDE, "failover_cooldown_seconds": 300}
        )

        provider = create_llm(config)

        assert isinstance(provider, TieredProvider)
        (primary_cfg, _), (failover_cfg, failover) = provider.tiers
        assert primary_cfg.failover is None
        assert primary_cfg.max_retries == 1
        assert failover_cfg is CLAUDE
        assert isinstance(failover, ClaudeCLIProvider)
        assert provider.cooldown_seconds == 300
