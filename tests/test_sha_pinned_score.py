"""A completed review always leaves a formal review whose score is pinned to
the head SHA it was given for, so merge gates can tie a score to a commit."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mira.config import MiraConfig
from mira.core.engine import ReviewEngine, commit_web_url
from mira.llm.provider import LLMProvider
from mira.models import (
    PRInfo,
    ReviewComment,
    ReviewResult,
    Severity,
    WalkthroughConfidenceScore,
    WalkthroughResult,
    score_block,
)
from mira.platforms.handlers import review_completed_payload
from mira.providers.forgejo import ForgejoProvider
from mira.providers.github import GitHubProvider

SHA = "0123456789abcdef0123456789abcdef01234567"
NEWER_SHA = "fedcba9876543210fedcba9876543210fedcba98"
SCORE_LINE = f"Confidence: 4/5 \u00b7 head {SHA}"
SCORE_MARKER = f"<!-- mira-score sha={SHA} score=4 -->"

_DIFF = """\
diff --git a/src/utils.py b/src/utils.py
new file mode 100644
--- /dev/null
+++ b/src/utils.py
@@ -0,0 +1,3 @@
+def add(a, b):
+    return a + b
+
"""

_WALKTHROUGH = json.dumps(
    {
        "summary": "Adds a helper.",
        "change_groups": [
            {
                "label": "Core",
                "files": [
                    {"path": "src/utils.py", "change_type": "added", "description": "helper"}
                ],
            }
        ],
        "confidence_score": {"score": 4, "label": "Safe to merge", "reason": "Small change."},
        "sequence_diagram": None,
    }
)

_NO_COMMENTS = json.dumps(
    {"comments": [], "summary": "All good!", "metadata": {"reviewed_files": 1}}
)


def _pr_info(head_sha: str = SHA, platform: str = "github") -> PRInfo:
    return PRInfo(
        title="Test PR",
        description="d",
        base_branch="main",
        head_branch="feature",
        url="https://github.com/test/repo/pull/1",
        number=1,
        owner="test",
        repo="repo",
        head_sha=head_sha,
        platform=platform,
    )


def _clean_llm() -> MagicMock:
    llm = MagicMock(spec=LLMProvider)
    llm.complete_agentic = AsyncMock(return_value={"content": "", "tool_calls": []})
    llm.walkthrough = AsyncMock(return_value=_WALKTHROUGH)
    llm.review = AsyncMock(return_value=_NO_COMMENTS)
    llm.complete = AsyncMock(return_value=_NO_COMMENTS)
    llm.count_tokens = MagicMock(return_value=100)
    llm.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return llm


def _provider(pr_info: PRInfo) -> AsyncMock:
    provider = AsyncMock()
    provider.get_pr_info.return_value = pr_info
    provider.get_pr_diff.return_value = _DIFF
    provider.post_review = AsyncMock(return_value=[])
    provider.post_comment = AsyncMock()
    provider.find_bot_comment = AsyncMock(return_value=None)
    provider.update_comment = AsyncMock()
    provider.get_unresolved_bot_threads = AsyncMock(return_value=[])
    provider.get_all_bot_threads = AsyncMock(return_value=[])
    return provider


def _comment() -> ReviewComment:
    return ReviewComment(
        path="a.py",
        line=1,
        end_line=None,
        severity=Severity.WARNING,
        category="bug",
        title="Issue",
        body="desc",
        confidence=0.9,
    )


def _scored(comments: list[ReviewComment], head_sha: str = SHA) -> ReviewResult:
    return ReviewResult(
        comments=comments,
        summary="Found issues" if comments else "",
        walkthrough=WalkthroughResult(
            summary="s",
            confidence_score=WalkthroughConfidenceScore(score=4, label="Safe", reason="r"),
        ),
        head_sha=head_sha,
    )


# ── engine ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_zero_findings_run_still_submits_scored_review():
    provider = _provider(_pr_info())
    engine = ReviewEngine(config=MiraConfig(), llm=_clean_llm(), provider=provider)

    result = await engine.review_pr("https://github.com/test/repo/pull/1")

    assert result.comments == []
    assert result.head_sha == SHA
    assert result.confidence_score == 4
    provider.post_review.assert_awaited_once()
    posted = provider.post_review.await_args.args[1]
    assert posted.head_sha == SHA
    assert posted.comments == []


@pytest.mark.asyncio
async def test_head_sha_is_the_one_fetched_at_review_start():
    """A later head (e.g. a push mid-review) never replaces the reviewed SHA."""
    provider = _provider(_pr_info(SHA))
    engine = ReviewEngine(config=MiraConfig(), llm=_clean_llm(), provider=provider)
    provider.get_pr_info.side_effect = [_pr_info(SHA), _pr_info(NEWER_SHA)]

    result = await engine.review_pr("https://github.com/test/repo/pull/1")

    assert result.head_sha == SHA
    assert provider.post_review.await_args.args[1].head_sha == SHA


@pytest.mark.asyncio
async def test_walkthrough_shows_reviewed_head_sha():
    provider = _provider(_pr_info())
    engine = ReviewEngine(config=MiraConfig(), llm=_clean_llm(), provider=provider)

    await engine.review_pr("https://github.com/test/repo/pull/1")

    bodies = [c.args[1] for c in provider.post_comment.await_args_list]
    bodies += [c.args[2] for c in provider.update_comment.await_args_list]
    final = [b for b in bodies if "Confidence: 4/5" in b]
    assert final, "walkthrough with a confidence score was never posted"
    url = f"https://github.com/test/repo/pull/1/commits/{SHA}"
    assert f'head <a href="{url}"><code>{SHA[:7]}</code></a>' in final[-1]


@pytest.mark.asyncio
async def test_unknown_head_sha_posts_nothing_for_zero_findings():
    provider = _provider(_pr_info(head_sha=""))
    engine = ReviewEngine(config=MiraConfig(), llm=_clean_llm(), provider=provider)

    await engine.review_pr("https://github.com/test/repo/pull/1")

    provider.post_review.assert_not_called()


@pytest.mark.asyncio
async def test_logs_head_sha_at_review_start(caplog):
    provider = _provider(_pr_info())
    engine = ReviewEngine(config=MiraConfig(), llm=_clean_llm(), provider=provider)

    with caplog.at_level("INFO", logger="mira.core.engine"):
        await engine.review_pr("https://github.com/test/repo/pull/1")

    assert f"Reviewing test/repo#1 at head {SHA}" in caplog.text


def test_commit_web_url_per_platform():
    assert commit_web_url(_pr_info(), SHA).endswith(f"/pull/1/commits/{SHA}")
    gl = _pr_info(platform="gitlab")
    assert commit_web_url(gl, SHA).endswith(f"/diffs?commit_id={SHA}")
    assert commit_web_url(_pr_info(platform="origin"), SHA) == ""
    assert commit_web_url(_pr_info(), "") == ""


# ── score format ────────────────────────────────────────────────────


def test_score_block_format():
    assert score_block(4, SHA) == f"{SCORE_LINE}\n{SCORE_MARKER}"
    assert score_block(None, SHA).endswith(f"<!-- mira-score sha={SHA} score=none -->")


# ── GitHub provider ─────────────────────────────────────────────────


def _github(mock_pr: MagicMock, pinned: MagicMock) -> tuple[GitHubProvider, MagicMock]:
    async def _token() -> str:
        return "t"

    provider = GitHubProvider.__new__(GitHubProvider)
    provider._token_supplier = _token
    repo = MagicMock()
    repo.get_pull.return_value = mock_pr
    repo.get_commit.return_value = pinned
    gh = MagicMock()
    gh.get_repo.return_value = repo
    return provider, gh


@pytest.mark.asyncio
async def test_github_zero_findings_submits_comment_review_pinned_to_sha():
    pinned = MagicMock(sha=SHA)
    mock_pr = MagicMock()
    mock_pr.get_commits.return_value = [MagicMock(sha=NEWER_SHA)]
    provider, gh = _github(mock_pr, pinned)

    with patch("mira.providers.github.Github", return_value=gh):
        await provider.post_review(_pr_info(), _scored([]))

    gh.get_repo.return_value.get_commit.assert_called_once_with(SHA)
    mock_pr.create_review.assert_called_once()
    kwargs = mock_pr.create_review.call_args.kwargs
    assert kwargs["commit"] is pinned
    assert kwargs["event"] == "COMMENT"
    assert kwargs["comments"] == []
    assert SCORE_LINE in kwargs["body"]
    assert SCORE_MARKER in kwargs["body"]


@pytest.mark.asyncio
async def test_github_findings_fold_score_into_the_single_review():
    pinned = MagicMock(sha=SHA)
    mock_pr = MagicMock()
    provider, gh = _github(mock_pr, pinned)

    with patch("mira.providers.github.Github", return_value=gh):
        await provider.post_review(_pr_info(), _scored([_comment()]))

    mock_pr.create_review.assert_called_once()
    kwargs = mock_pr.create_review.call_args.kwargs
    assert kwargs["commit"] is pinned
    assert kwargs["event"] == "COMMENT"
    assert len(kwargs["comments"]) == 1
    assert "Mira Review Summary" in kwargs["body"]
    assert kwargs["body"].rstrip().endswith(SCORE_MARKER)


# ── Forgejo provider ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forgejo_zero_findings_submits_scored_review():
    provider = ForgejoProvider("tok")
    provider._request = AsyncMock()  # type: ignore[method-assign]

    await provider.post_review(_pr_info(platform="forgejo"), _scored([]))

    provider._request.assert_awaited_once()
    method, url = provider._request.await_args.args
    body = provider._request.await_args.kwargs["json"]
    assert (method, url.endswith("/reviews")) == ("POST", True)
    assert body["event"] == "COMMENT"
    assert body["commit_id"] == SHA
    assert body["comments"] == []
    assert SCORE_MARKER in body["body"]


# ── review.completed payload ────────────────────────────────────────


def test_review_completed_payload_has_score_and_head_sha():
    data = review_completed_payload("o/r", "u", 1, _scored([]))
    assert data["confidence_score"] == 4
    assert data["head_sha"] == SHA


def test_review_completed_payload_without_walkthrough():
    data = review_completed_payload("o/r", "u", 1, ReviewResult(head_sha=SHA))
    assert data["confidence_score"] is None
    assert data["head_sha"] == SHA
