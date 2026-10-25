"""A malformed chunk answer is salvaged, retried, or reported, never silently lost."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import LLMConfig, MiraConfig
from mira.core.engine import ReviewEngine
from mira.exceptions import ResponseParseError
from mira.llm.provider import LLMProvider
from mira.llm.response_parser import parse_llm_response
from mira.llm.tiered import TieredProvider
from mira.models import ReviewResult, WalkthroughResult

GOOD = {
    "path": "src/utils.py",
    "line": 9,
    "severity": "warning",
    "category": "security",
    "title": "Shell injection",
    "body": "Using shell=True is dangerous.",
    "confidence": 0.95,
}
PATHLESS = {
    "line": 12,
    "severity": "warning",
    "title": "Missing await",
    "body": "The promise is dropped.",
    "agent_prompt": "In packages/api/src/client.ts, await the fetch call on line 12.",
    "confidence": 0.9,
}
KNOWN = ["packages/api/src/client.ts", "packages/web/src/app.ts"]


class TestSalvage:
    def test_missing_path_is_recovered_from_agent_prompt(self):
        parsed = parse_llm_response(json.dumps({"comments": [PATHLESS]}), known_paths=KNOWN)
        assert [c.path for c in parsed.comments] == ["packages/api/src/client.ts"]
        assert parsed.dropped_comments == 0

    def test_unrecoverable_finding_is_dropped_not_the_whole_chunk(self):
        raw = json.dumps({"comments": [GOOD, PATHLESS], "summary": "s"})
        parsed = parse_llm_response(raw, known_paths=["src/utils.py"])
        assert [c.path for c in parsed.comments] == ["src/utils.py"]
        assert parsed.dropped_comments == 1
        assert parsed.summary == "s"

    def test_ambiguous_path_is_not_guessed(self):
        both = {**PATHLESS, "body": "Also see packages/web/src/app.ts."}
        parsed = parse_llm_response(json.dumps({"comments": [both]}), known_paths=KNOWN)
        assert parsed.comments == [] and parsed.dropped_comments == 1

    def test_path_must_match_whole(self):
        item = {**PATHLESS, "agent_prompt": "In packages/api/src/client.tsx fix it"}
        parsed = parse_llm_response(json.dumps({"comments": [item]}), known_paths=KNOWN)
        assert parsed.comments == []

    def test_non_object_still_raises(self):
        with pytest.raises(ResponseParseError):
            parse_llm_response("[1, 2]")


def _tier(name: str, review: AsyncMock) -> tuple[LLMConfig, MagicMock]:
    provider = MagicMock()
    provider.review = review
    provider.count_tokens = MagicMock(return_value=1)
    provider.total_prompt_tokens = 0
    provider.total_completion_tokens = 0
    return LLMConfig(provider="openai", model=name), provider


class TestRetryOnAnotherTier:
    @pytest.mark.asyncio
    async def test_retry_skips_the_tier_that_served_the_bad_answer(self):
        first = AsyncMock(side_effect=RuntimeError("down"))
        second = AsyncMock(return_value="glm-answer")
        tiered = TieredProvider([_tier("a", first), _tier("b", second)], cooldown_seconds=0)
        assert await tiered.review([{"role": "user", "content": "x"}]) == "glm-answer"

        first.side_effect = None
        first.return_value = "claude-answer"
        second.reset_mock()
        assert await tiered.review_on_another_tier([{"role": "user", "content": "x"}]) == (
            "claude-answer"
        )
        second.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_tier_retries_itself(self):
        only = AsyncMock(return_value="again")
        tiered = TieredProvider([_tier("a", only)], cooldown_seconds=0)
        await tiered.review([{"role": "user", "content": "x"}])
        assert await tiered.review_on_another_tier([{"role": "user", "content": "x"}]) == "again"


def _llm(review_side_effect) -> MagicMock:
    llm = MagicMock(spec=LLMProvider)
    llm.complete_agentic = AsyncMock(return_value={"content": "", "tool_calls": []})
    llm.count_tokens = MagicMock(return_value=50)
    llm.walkthrough = AsyncMock(
        return_value=json.dumps({"summary": "walkthrough", "change_groups": []})
    )
    llm.review = AsyncMock(side_effect=review_side_effect)
    llm.complete = AsyncMock(return_value=json.dumps({"comments": [], "summary": "s"}))
    llm.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return llm


def _config() -> MiraConfig:
    config = MiraConfig()
    config.filter.confidence_threshold = 0.0
    config.review.self_critique = False
    return config


GOOD_RESPONSE = json.dumps({"comments": [GOOD], "summary": "Found issues."})


class TestEngine:
    @pytest.mark.asyncio
    async def test_unparseable_answer_is_retried_once(self, sample_diff_text: str):
        llm = _llm(["NOT JSON {{{", GOOD_RESPONSE])
        result = await ReviewEngine(config=_config(), llm=llm).review_diff(sample_diff_text)

        assert llm.review.await_count == 2
        assert [c.path for c in result.comments] == ["src/utils.py"]
        assert result.failed_chunks == []
        assert any(a.get("stage") == "parse_retry" for a in result.audit)

    @pytest.mark.asyncio
    async def test_chunk_failing_twice_is_reported(self, sample_diff_text: str):
        llm = _llm(["NOT JSON {{{", "still {{{ not json"])
        result = await ReviewEngine(config=_config(), llm=llm).review_diff(sample_diff_text)

        assert llm.review.await_count == 2
        assert result.comments == []
        assert len(result.failed_chunks) == 1
        failed = result.failed_chunks[0]
        assert failed["chunk"] == 1 and "src/utils.py" in failed["files"]
        assert result.total_chunks == 1
        assert "1 of 1 review chunk couldn't be reviewed" in result.summary

    @pytest.mark.asyncio
    async def test_retry_error_marks_the_chunk_failed(self, sample_diff_text: str):
        llm = _llm(["NOT JSON {{{", RuntimeError("provider down")])
        result = await ReviewEngine(config=_config(), llm=llm).review_diff(sample_diff_text)

        assert len(result.failed_chunks) == 1
        assert "provider down" in result.failed_chunks[0]["error"]


class TestVisibility:
    def test_walkthrough_says_chunks_could_not_be_reviewed(self):
        walkthrough = WalkthroughResult(summary="Adds a client.")
        failed = [{"chunk": 2, "files": ["a.py", "b.py"], "error": "x"}]
        markdown = walkthrough.to_markdown(failed_chunks=failed, total_chunks=3)
        assert "1 of 3 review chunks couldn't be reviewed" in markdown
        assert "`a.py`, `b.py`" in markdown
        assert "couldn't be reviewed" not in walkthrough.to_markdown()

    def test_review_completed_payload_counts_failed_chunks(self):
        from mira.platforms.handlers import review_completed_payload

        failed = [{"chunk": 1, "files": ["a.py"], "error": "bad json"}]
        result = ReviewResult(failed_chunks=failed, total_chunks=2)
        payload = review_completed_payload("o/r", "u", 5, result)
        assert payload["comments"] == 0
        assert payload["failed_chunks"] == 1
        assert payload["failed_chunk_list"] == failed
        assert review_completed_payload("o/r", "u", 5, ReviewResult())["failed_chunks"] == 0

    def test_slack_message_flags_failed_chunks(self):
        from mira.outbound_webhooks import REVIEW_COMPLETED, _message

        data = {"repo": "o/r", "pr_url": "u", "number": 5, "comments": 0, "failed_chunks": 2}
        _, body, color = _message(REVIEW_COMPLETED, data)
        assert "2 chunk(s) couldn't be reviewed" in body
        assert color != "2EB67D"
