"""Origin per-file patches → unified diff, for files whose patch is empty.

Pure renames, empty added files (``.gitkeep``) and empty deleted files come
back from Origin's List Pull Request Files with ``patch: ""``. The rebuilt diff
must still parse through the real diff parser (unidiff), with a header-only
section and no stray blank line — a bare ``---``/``+++`` pair followed by a
blank line used to raise "Unexpected trailing newline character".
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import MiraConfig
from mira.core.diff_parser import parse_diff
from mira.core.engine import ReviewEngine
from mira.llm.provider import LLMProvider
from mira.models import FileChangeType, PRInfo
from mira.providers.origin import _patches_to_diff

RENAME = {
    "filename": "src/lib/new_name.ts",
    "previousFilename": "src/lib/old_name.ts",
    "status": "renamed",
    "additions": 0,
    "deletions": 0,
    "patch": "",
}
GITKEEP = {"filename": "public/uploads/.gitkeep", "status": "added", "patch": ""}
DELETED_EMPTY = {"filename": "tmp/.gitkeep", "status": "removed", "patch": ""}
MODIFIED = {
    "filename": "src/app.ts",
    "status": "modified",
    "additions": 1,
    "deletions": 1,
    "patch": "@@ -1,3 +1,3 @@\n import a from 'a';\n-const x = 1;\n+const x = 2;\n export {};",
}
RENAME_WITH_EDIT = {
    "filename": "src/moved.ts",
    "previousFilename": "src/orig.ts",
    "status": "renamed",
    "patch": "@@ -1,2 +1,2 @@\n-old\n+new\n same\n",
}


def _parsed(files):
    diff = _patches_to_diff(files)
    assert "\n\n" not in diff, f"stray blank line in rebuilt diff:\n{diff!r}"
    return diff, {f.path: f for f in parse_diff(diff).files}


def test_rename_only() -> None:
    diff, files = _parsed([RENAME])
    assert "rename from src/lib/old_name.ts\nrename to src/lib/new_name.ts" in diff
    assert "---" not in diff and "+++" not in diff
    f = files["src/lib/new_name.ts"]
    assert f.change_type == FileChangeType.RENAMED
    assert f.old_path == "src/lib/old_name.ts"
    assert f.hunks == []


def test_empty_added_file() -> None:
    diff, files = _parsed([GITKEEP])
    assert diff == (
        "diff --git a/public/uploads/.gitkeep b/public/uploads/.gitkeep\nnew file mode 100644\n"
    )
    f = files["public/uploads/.gitkeep"]
    assert f.change_type == FileChangeType.ADDED
    assert f.hunks == []


def test_deleted_empty_file() -> None:
    diff, files = _parsed([DELETED_EMPTY])
    assert "deleted file mode 100644" in diff
    f = files["tmp/.gitkeep"]
    assert f.change_type == FileChangeType.DELETED
    assert f.hunks == []


def test_mix_with_normal_patches_in_every_order() -> None:
    """The real failing shape (ricky-poon/intercommerce-nextjs#10): renames and
    .gitkeep files interleaved with normal patches. Empty sections first,
    middle and last must all parse."""
    other_rename = {**RENAME, "filename": "b.ts", "previousFilename": "a.ts"}
    other_keep = {**GITKEEP, "filename": "public/img/.gitkeep"}
    files_in = [
        RENAME,
        MODIFIED,
        GITKEEP,
        other_rename,
        RENAME_WITH_EDIT,
        other_keep,
        DELETED_EMPTY,
    ]
    _diff, files = _parsed(files_in)
    assert list(files) == [
        "src/lib/new_name.ts",
        "src/app.ts",
        "public/uploads/.gitkeep",
        "b.ts",
        "src/moved.ts",
        "public/img/.gitkeep",
        "tmp/.gitkeep",
    ]
    mod = files["src/app.ts"]
    assert mod.change_type == FileChangeType.MODIFIED
    assert (mod.added_lines, mod.deleted_lines, len(mod.hunks)) == (1, 1, 1)
    # A patch without a trailing newline doesn't swallow the next header.
    assert mod.hunks[0].content.rstrip().endswith("export {};")
    moved = files["src/moved.ts"]
    assert moved.change_type == FileChangeType.RENAMED
    assert moved.old_path == "src/orig.ts"
    assert (moved.added_lines, moved.deleted_lines) == (1, 1)
    assert files["b.ts"].hunks == [] and files["public/img/.gitkeep"].hunks == []


def test_empty_patch_modified_file_is_header_only() -> None:
    """Binary / mode-only changes: Origin sends an empty patch for those too."""
    _diff, files = _parsed([{"filename": "logo.png", "status": "modified", "patch": ""}, MODIFIED])
    assert files["logo.png"].hunks == []
    assert len(files["src/app.ts"].hunks) == 1


def test_no_files_is_empty_diff() -> None:
    assert _patches_to_diff([]) == ""
    assert parse_diff(_patches_to_diff([])).files == []


# ── engine: a PR whose files carry no hunks ─────────────────────────────────


def _llm() -> LLMProvider:
    llm = MagicMock(spec=LLMProvider)
    llm.complete_agentic = AsyncMock(return_value={"content": "", "tool_calls": []})
    llm.walkthrough = AsyncMock(return_value='{"summary": "x", "change_groups": []}')
    llm.review = AsyncMock(return_value='{"comments": [], "summary": "ok"}')
    llm.complete = AsyncMock(return_value="{}")
    llm.count_tokens = MagicMock(return_value=10)
    llm.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return llm


def _provider(diff: str) -> AsyncMock:
    provider = AsyncMock()
    provider.get_pr_info.return_value = PRInfo(
        title="Rename things",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://cursor.com/codebase/acme/shop/pull/10",
        number=10,
        owner="acme",
        repo="shop",
        head_sha="abc123",
        platform="origin",
    )
    provider.get_pr_diff.return_value = diff
    provider.find_bot_comment = AsyncMock(return_value=None)
    provider.get_all_bot_threads = AsyncMock(return_value=[])
    return provider


@pytest.mark.asyncio
async def test_engine_skips_hunkless_pr_without_scoring() -> None:
    """Only renames and .gitkeep files: no crash, no LLM call, no score."""
    llm = _llm()
    diff = _patches_to_diff([RENAME, GITKEEP, DELETED_EMPTY])
    engine = ReviewEngine(config=MiraConfig(), llm=llm, provider=_provider(diff), dry_run=True)
    result = await engine.review_pr("https://cursor.com/codebase/acme/shop/pull/10")

    assert result.skipped_reason and "no line changes" in result.skipped_reason.lower()
    assert result.confidence_score is None
    assert result.comments == []
    llm.walkthrough.assert_not_called()
    llm.review.assert_not_called()


@pytest.mark.asyncio
async def test_engine_reviews_mixed_pr() -> None:
    """Renames/.gitkeep next to a real patch: the review runs normally."""
    llm = _llm()
    diff = _patches_to_diff([RENAME, MODIFIED, GITKEEP])
    engine = ReviewEngine(config=MiraConfig(), llm=llm, provider=_provider(diff), dry_run=True)
    result = await engine.review_pr("https://cursor.com/codebase/acme/shop/pull/10")

    assert result.skipped_reason is None
    llm.walkthrough.assert_called()
    llm.review.assert_called()
