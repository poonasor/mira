"""Tests for the agentic tool executor used on unindexed-repo reviews."""

from __future__ import annotations

import pytest

from mira.llm.agentic_tools import (
    AGENTIC_TOOLS,
    GREP_REPO_TOOL,
    READ_FILE_TOOL,
    AgenticToolExecutor,
)


class _FakeFetcher:
    def __init__(self, sources: dict[str, str | None]):
        self._sources = sources

    async def fetch(self, path: str) -> str | None:
        return self._sources.get(path)


def _executor(sources: dict[str, str | None], tree: list[str]) -> AgenticToolExecutor:
    return AgenticToolExecutor(source_fetcher=_FakeFetcher(sources), repo_tree=tree)


class TestSchemas:
    def test_tool_set_exposes_both_helpers(self):
        names = [t["function"]["name"] for t in AGENTIC_TOOLS]
        assert "read_file" in names
        assert "grep_repo" in names

    def test_read_file_requires_path(self):
        assert READ_FILE_TOOL["function"]["parameters"]["required"] == ["path"]

    def test_grep_repo_requires_pattern(self):
        assert GREP_REPO_TOOL["function"]["parameters"]["required"] == ["pattern"]


class TestReadFile:
    @pytest.mark.asyncio
    async def test_returns_numbered_content(self):
        ex = _executor({"src/a.py": "alpha\nbeta\n"}, ["src/a.py"])
        out = await ex.execute("read_file", {"path": "src/a.py"})
        assert "src/a.py" in out
        assert "1  alpha" in out
        assert "2  beta" in out

    @pytest.mark.asyncio
    async def test_truncates_huge_files(self):
        big = "x" * 20_000
        ex = _executor({"src/big.py": big}, ["src/big.py"])
        out = await ex.execute("read_file", {"path": "src/big.py"})
        assert "truncated" in out

    @pytest.mark.asyncio
    async def test_missing_path_returns_error_string(self):
        ex = _executor({}, [])
        out = await ex.execute("read_file", {})
        assert out.startswith("[error")

    @pytest.mark.asyncio
    async def test_unknown_path_in_tree_suggests_close_match(self):
        ex = _executor({}, ["src/auth/middleware.py", "src/util.py"])
        out = await ex.execute("read_file", {"path": "AUTH/middleware.py"})
        assert "not found" in out
        assert "src/auth/middleware.py" in out

    @pytest.mark.asyncio
    async def test_caches_repeated_reads(self):
        seen: list[str] = []

        class _Counting:
            async def fetch(self, path: str) -> str | None:
                seen.append(path)
                return "hello"

        ex = AgenticToolExecutor(source_fetcher=_Counting(), repo_tree=["src/a.py"])
        await ex.execute("read_file", {"path": "src/a.py"})
        await ex.execute("read_file", {"path": "src/a.py"})
        assert seen == ["src/a.py"]  # only fetched once


class TestGrepRepo:
    @pytest.mark.asyncio
    async def test_path_only_returns_matching_paths(self):
        ex = _executor({}, ["src/auth.py", "src/util.py", "tests/test_auth.py"])
        out = await ex.execute("grep_repo", {"pattern": "auth", "path_only": True})
        assert "src/auth.py" in out
        assert "tests/test_auth.py" in out
        assert "src/util.py" not in out

    @pytest.mark.asyncio
    async def test_content_search_returns_line_hits(self):
        sources = {
            "src/a.py": "def foo():\n    return BAR\n",
            "src/b.py": "import os\nBAR = 1\n",
        }
        ex = _executor(sources, list(sources))
        out = await ex.execute("grep_repo", {"pattern": r"\bBAR\b"})
        assert "src/a.py:2" in out
        assert "src/b.py:2" in out

    @pytest.mark.asyncio
    async def test_path_glob_filters_candidates(self):
        sources = {
            "src/a.py": "needle\n",
            "src/a.go": "needle\n",
        }
        ex = _executor(sources, list(sources))
        out = await ex.execute("grep_repo", {"pattern": "needle", "path_glob": "**/*.go"})
        assert "src/a.go" in out
        assert "src/a.py" not in out

    @pytest.mark.asyncio
    async def test_invalid_regex_returns_error_string(self):
        ex = _executor({"a.py": "x"}, ["a.py"])
        out = await ex.execute("grep_repo", {"pattern": "[unclosed"})
        assert out.startswith("[invalid regex")


class TestExecutorBudget:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        ex = _executor({}, [])
        out = await ex.execute("delete_repo", {})
        assert "unknown tool" in out

    @pytest.mark.asyncio
    async def test_exhausted_budget_blocks_further_calls(self):
        ex = _executor({"a.py": "hi"}, ["a.py"])
        ex.bytes_used = 1_000_000  # simulate exhaustion
        out = await ex.execute("read_file", {"path": "a.py"})
        assert "budget exhausted" in out


class TestOutputCaps:
    @pytest.mark.asyncio
    async def test_line_ranges_and_continuation_hint(self):
        source = "\n".join(f"line{n}" for n in range(1, 101))
        ex = AgenticToolExecutor(
            source_fetcher=_FakeFetcher({"a.py": source}), repo_tree=["a.py"], max_file_chars=1_000
        )
        part = await ex.execute("read_file", {"path": "a.py", "start_line": 10, "end_line": 12})
        assert "lines 10-12 of 100" in part
        assert "10  line10" in part and "12  line12" in part and "line13" not in part

        whole = await ex.execute("read_file", {"path": "a.py"})
        assert "truncated at line" in whole and "start_line=" in whole
        assert len(whole) < 1_200

    @pytest.mark.asyncio
    async def test_repeated_identical_calls_return_a_pointer(self):
        ex = _executor({"a.py": "x = 1\n"}, ["a.py"])
        first = await ex.execute("read_file", {"path": "a.py"})
        again = await ex.execute("read_file", {"path": "a.py"})
        assert "x = 1" in first
        assert "already returned" in again and "x = 1" not in again
        g1 = await ex.execute("grep_repo", {"pattern": "x"})
        g2 = await ex.execute("grep_repo", {"pattern": "x"})
        assert "already answered" in g2 and len(g2) < len(g1) + 60

    @pytest.mark.asyncio
    async def test_total_budget_truncates_then_refuses(self):
        files = {f"f{i}.py": "y" * 900 for i in range(5)}
        ex = AgenticToolExecutor(
            source_fetcher=_FakeFetcher(files),
            repo_tree=list(files),
            max_file_chars=1_000,
            max_total_chars=2_000,
        )
        outs = [await ex.execute("read_file", {"path": p}) for p in files]
        assert "tool output budget used up" in "".join(outs)
        assert sum(len(o) for o in outs) < 2_000 + 5 * 200
        assert ex.budget_exhausted

    def test_limits_come_from_review_config(self):
        from mira.config import ReviewConfig

        review = ReviewConfig(
            agentic_max_hops=3,
            agentic_max_file_chars=4_000,
            agentic_max_total_chars=9_000,
            agentic_max_grep_hits=5,
        )
        ex = AgenticToolExecutor.from_review_config(review, _FakeFetcher({}), [])
        assert (ex.max_hops, ex.max_file_chars, ex.max_total_chars, ex.max_grep_hits) == (
            3,
            4_000,
            9_000,
            5,
        )
        defaults = AgenticToolExecutor.from_review_config(ReviewConfig(), _FakeFetcher({}), [])
        assert defaults.max_hops == 6

    def test_config_rejects_out_of_range_limits(self):
        from pydantic import ValidationError

        from mira.config import ReviewConfig

        with pytest.raises(ValidationError):
            ReviewConfig(agentic_max_hops=0)
        with pytest.raises(ValidationError):
            ReviewConfig(agentic_max_file_chars=10)


class TestLoopHopBudget:
    @pytest.mark.asyncio
    async def test_configured_hops_bound_the_loop_and_last_hop_says_submit(self):
        from unittest.mock import AsyncMock, MagicMock

        from mira.core.passes import agentic_review_loop

        read = {
            "content": "",
            "tool_calls": [
                {"id": "c", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ],
        }
        llm = MagicMock()
        last_messages: list[str] = []

        async def hop(convo, tools):
            last_messages.append(str(convo[-1].get("content")))
            return read

        llm.complete_agentic = AsyncMock(side_effect=hop)
        llm.close_agentic = AsyncMock()
        ex = _executor({"a.py": "x\n"}, ["a.py"])
        ex.max_hops = 3

        assert await agentic_review_loop(llm, [{"role": "system", "content": "s"}], ex) == ""

        assert llm.complete_agentic.await_count == 3
        assert "must call submit_review" in last_messages[2]
        assert not any("must call submit_review" in m for m in last_messages[:2])
        llm.close_agentic.assert_awaited_once()
