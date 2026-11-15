"""Regression tests for issue #381: agentic chunk failures must stay visible."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import MiraConfig
from mira.core import engine as engine_mod
from mira.core.engine import ReviewEngine, _response_excerpt
from mira.core.passes import AgenticLoopResult, agentic_review_loop
from mira.exceptions import StructuredOutputError
from mira.llm.base import _retriable
from mira.llm.provider import LLMProvider
from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL
from mira.models import PRInfo, ReviewChunk, WalkthroughResult


def _submit(arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_submit",
                "type": "function",
                "function": {
                    "name": "submit_review",
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def _read(path: str, call_id: str = "call_read") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
            }
        ],
    }


def _grep(pattern: str, call_id: str = "call_grep") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "grep_repo",
                    "arguments": json.dumps({"pattern": pattern}),
                },
            }
        ],
    }


class _FakeFetcher:
    async def fetch(self, path: str) -> str | None:
        return f"contents of {path}\nsecond line\n"


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def execute(self, name: str, args: dict) -> str:
        self.calls.append((name, args))
        return f"READ:{args.get('path') or args.get('pattern')}"


class _LoopLLM:
    def __init__(self, responses: list[dict], forced: list[object]) -> None:
        self._responses = list(responses)
        self._forced = list(forced)
        self.agentic_calls: list[list[dict]] = []
        self.forced_calls: list[dict] = []

    async def complete_agentic(self, messages, tools):
        self.agentic_calls.append([dict(m) for m in messages])
        return self._responses.pop(0)

    async def complete_with_tools(self, messages, tools, temperature=None, force_tool=None):
        self.forced_calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "force_tool": force_tool}
        )
        value = self._forced.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


class TestAgenticLoopCap:
    @pytest.mark.asyncio
    async def test_cap_forces_submit_on_existing_conversation(self):
        responses = []
        for i in range(6):
            responses.append(
                _read(f"src/f{i}.py", f"call_{i}") if i % 2 == 0 else _grep("needle", f"call_{i}")
            )
        llm = _LoopLLM(responses, ['{"comments": []}'])
        executor = _RecordingExecutor()

        result = await agentic_review_loop(
            llm, [{"role": "user", "content": "review src/a.py"}], executor, max_hops=6
        )

        assert isinstance(result, AgenticLoopResult)
        assert result.hit_hop_cap is True
        assert result.hops == 6
        assert result.raw_arguments == '{"comments": []}'
        assert len(result.tool_calls) == 6

        forced = llm.forced_calls[0]
        assert forced["force_tool"] == "submit_review"
        tool_names = [t["function"]["name"] for t in forced["tools"]]
        assert tool_names == ["read_file", "grep_repo", "submit_review"]
        prompt = "\n".join(str(m.get("content") or "") for m in forced["messages"])
        assert "You have reached the tool-call limit" in prompt
        assert "READ:src/f0.py" in prompt
        assert "READ:needle" in prompt

    @pytest.mark.asyncio
    async def test_valid_forced_submit_returns_arguments(self):
        llm = _LoopLLM([_read("src/a.py")], ['{"comments": []}'])
        # Only one hop is used before the cap would be hit; force the cap path by
        # making every hop a tool call.
        llm._responses = [_read(f"src/a{i}.py", f"c{i}") for i in range(6)]

        result = await agentic_review_loop(
            llm, [{"role": "user", "content": "review"}], _RecordingExecutor()
        )

        assert result.raw_arguments == '{"comments": []}'
        assert result.error is None

    @pytest.mark.asyncio
    async def test_prose_gets_one_corrective_retry(self):
        llm = _LoopLLM(
            [_read(f"src/a{i}.py", f"c{i}") for i in range(6)],
            ["I reviewed the code and found nothing.", '{"comments": []}'],
        )

        result = await agentic_review_loop(
            llm, [{"role": "user", "content": "review"}], _RecordingExecutor()
        )

        assert result.raw_arguments == '{"comments": []}'
        assert result.error is None
        assert len(llm.forced_calls) == 2
        corrective = "\n".join(str(m.get("content") or "") for m in llm.forced_calls[1]["messages"])
        assert "previous response was prose" in corrective

    @pytest.mark.asyncio
    async def test_two_prose_responses_return_failure(self):
        llm = _LoopLLM(
            [_read(f"src/a{i}.py", f"c{i}") for i in range(6)],
            [
                StructuredOutputError(
                    "structured_output", model="m", tool="submit_review", excerpt="one"
                ),
                StructuredOutputError(
                    "structured_output", model="m", tool="submit_review", excerpt="two"
                ),
            ],
        )

        result = await agentic_review_loop(
            llm, [{"role": "user", "content": "review"}], _RecordingExecutor(), max_hops=6
        )

        assert result.raw_arguments == ""
        assert result.hit_hop_cap is True
        assert result.error is not None
        assert len(llm.forced_calls) == 2


class TestStructuredOutputFallback:
    def test_structured_output_error_is_not_retried(self):
        assert (
            _retriable(StructuredOutputError("structured_output", model="m", tool="t", excerpt="x"))
            is False
        )

    @pytest.mark.asyncio
    async def test_fallback_model_runs_after_structured_failure(self):
        provider = LLMProvider(
            MiraConfig().llm.model_copy(update={"model": "primary", "fallback_model": "fallback"})
        )
        calls: list[str] = []

        async def fake_call(model, messages, tools, temperature=None, force_tool=None):
            calls.append(model)
            if model == "primary":
                raise StructuredOutputError(
                    "structured_output", model=model, tool="submit_review", excerpt="prose"
                )
            return '{"comments": []}'

        provider._call_llm_with_tools = fake_call  # type: ignore[method-assign]

        raw = await provider.complete_with_tools(
            [{"role": "user", "content": "review"}],
            tools=[SUBMIT_REVIEW_TOOL],
            force_tool="submit_review",
        )

        assert raw == '{"comments": []}'
        assert calls == ["primary", "fallback"]


class _ReviewLLM:
    """Scripted LLM for engine-level chunk tests."""

    supports_temperature = True

    def __init__(self, failing_paths: set[str]) -> None:
        self.failing_paths = failing_paths
        self.hop_counts: dict[str, int] = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.complete_with_tools = AsyncMock(side_effect=self._complete_with_tools)

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def _text(self, messages: list[dict]) -> str:
        return "\n".join(str(m.get("content") or "") for m in messages)

    def _failing_path(self, messages: list[dict]) -> str | None:
        text = self._text(messages)
        return next((path for path in self.failing_paths if path in text), None)

    async def complete_agentic(self, messages, tools):
        path = self._failing_path(messages)
        if path is None:
            return _submit({"comments": [], "summary": "", "metadata": {"reviewed_files": 1}})
        count = self.hop_counts.get(path, 0) + 1
        self.hop_counts[path] = count
        if count <= 6:
            return (
                _read(path, f"read_{path}_{count}")
                if count % 2
                else _grep(path, f"grep_{path}_{count}")
            )
        return _submit({"comments": [], "summary": "", "metadata": {"reviewed_files": 1}})

    async def _complete_with_tools(self, messages, tools, temperature=None, force_tool=None):
        path = self._failing_path(messages)
        if path is not None:
            raise StructuredOutputError(
                "structured_output", model="m", tool="submit_review", excerpt="prose review"
            )
        return '{"comments": [], "summary": "", "metadata": {"reviewed_files": 1}}'

    async def review(self, messages, temperature=None):
        # A chunk that fails agentically also fails the plain-review fallback,
        # so these tests still exercise the "chunk really failed" path.
        if self._failing_path(messages) is not None:
            raise StructuredOutputError(
                "structured_output", model="m", tool="submit_review", excerpt="prose review"
            )
        return '{"comments": [], "summary": "", "metadata": {"reviewed_files": 0}}'

    async def complete(self, messages, json_mode=True, temperature=None, max_tokens=None):
        return "{}"

    async def walkthrough(self, messages):
        return json.dumps({"summary": "wt", "change_groups": []})


def _diff(*paths: str) -> str:
    chunks = []
    for path in paths:
        added = "+" + ("x" * 800) + "\n"
        chunks.append(
            f"diff --git a/{path} b/{path}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{path}\n"
            "@@ -0,0 +1,1 @@\n"
            f"{added}"
        )
    return "".join(chunks)


def _engine(
    llm: _ReviewLLM, *, security_llm=None, config: MiraConfig | None = None
) -> ReviewEngine:
    cfg = config or MiraConfig()
    cfg.review.agentic_tools = True
    # The fake provider scripts six tool hops before it would submit; the
    # failure paths under test only trigger when the cap forces a submission.
    cfg.review.agentic_max_hops = 6
    if config is None:
        cfg.review.walkthrough = False
        cfg.review.security_pass = False
        cfg.review.dependency_overlap = False
        cfg.review.self_critique = False
        cfg.review.include_summary = True
    return ReviewEngine(config=cfg, llm=llm, security_llm=security_llm)


def _force_one_chunk_per_file(monkeypatch) -> None:
    def fake_chunks(files, max_tokens, provider=None):
        return [ReviewChunk(files=[file], token_estimate=100) for file in files]

    monkeypatch.setattr(engine_mod, "chunk_files", fake_chunks)


@pytest.mark.asyncio
async def test_one_failed_chunk_marks_review_incomplete(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    _force_one_chunk_per_file(monkeypatch)
    engine = _engine(_ReviewLLM({"src/bad.py"}))
    engine._agentic_source_fetcher = _FakeFetcher()
    engine._agentic_repo_tree = ["src/good.py", "src/bad.py"]

    result = await engine.review_diff(_diff("src/good.py", "src/bad.py"))

    assert len(result.failed_chunks) == 1
    assert result.failed_chunks[0].paths == ["src/bad.py"]
    assert result.failed_paths == ["src/bad.py"]
    assert "src/good.py" in result.reviewed_paths
    assert "src/bad.py" not in result.reviewed_paths
    assert result.summary.startswith("Review incomplete: 1 of 2")
    assert "No issues found" not in result.summary
    assert any(e["stage"] == "chunk_failed" for e in result.audit)

    md = WalkthroughResult(summary=result.summary).to_markdown(
        failed_paths=result.failed_paths,
        failed_chunks=result.failed_chunks,
        total_chunks=result.total_chunks,
    )
    assert "Review incomplete: 1 of 2 chunks failed" in md
    assert "`src/bad.py`" in md


@pytest.mark.asyncio
async def test_all_chunks_failing_uses_failure_notice(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    _force_one_chunk_per_file(monkeypatch)
    engine = _engine(_ReviewLLM({"src/a.py", "src/b.py"}))
    engine._agentic_source_fetcher = _FakeFetcher()
    engine._agentic_repo_tree = ["src/a.py", "src/b.py"]

    result = await engine.review_diff(_diff("src/a.py", "src/b.py"))

    assert len(result.failed_chunks) == 2
    assert result.failed_paths == ["src/a.py", "src/b.py"]
    assert result.summary.startswith("Review incomplete: all 2")
    assert "No issues found" not in result.summary
    notice = engine._format_chunk_failure_notice(result.failed_chunks)
    md = WalkthroughResult(summary=result.summary).to_markdown(
        failed_paths=result.failed_paths,
        failed_chunks=result.failed_chunks,
        total_chunks=result.total_chunks,
        failure_notice=notice,
    )
    assert "❌ Review failed" in md
    assert "No findings were returned" in md


@pytest.mark.asyncio
async def test_failed_paths_are_retryable_through_review_rest(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    _force_one_chunk_per_file(monkeypatch)
    engine = _engine(_ReviewLLM({"src/bad.py"}))
    engine._agentic_source_fetcher = _FakeFetcher()
    engine._agentic_repo_tree = ["src/good.py", "src/bad.py"]

    progress_updates: list = []

    mock_db = MagicMock()
    mock_db.get_last_reviewed_sha.return_value = ""
    mock_db.get_repo.return_value = None
    mock_db.get_pr_review_progress.return_value = None
    mock_db.upsert_pr_review_progress.side_effect = lambda progress, platform="github": (
        progress_updates.append(progress)
    )
    monkeypatch.setattr("mira.dashboard.api._app_db", mock_db)

    provider = AsyncMock()
    provider.get_pr_info.return_value = PRInfo(
        title="Test",
        description="",
        base_branch="main",
        head_branch="feature",
        url="https://github.com/test/repo/pull/1",
        number=1,
        owner="test",
        repo="repo",
    )
    provider.get_pr_diff.return_value = _diff("src/good.py", "src/bad.py")
    provider.get_all_bot_threads.return_value = []
    provider.get_unresolved_bot_threads.return_value = []
    provider.get_file_history.return_value = {}
    provider.get_repo_tree.return_value = ["src/good.py", "src/bad.py"]
    provider.post_review = AsyncMock()
    provider.post_comment = AsyncMock()
    provider.update_comment = AsyncMock()
    provider.find_bot_comment = AsyncMock(return_value=None)

    engine.provider = provider
    engine.dry_run = True

    await engine.review_pr("https://github.com/test/repo/pull/1")

    assert progress_updates
    progress = progress_updates[0]
    assert "src/bad.py" in progress.skipped_paths
    assert "src/bad.py" not in progress.reviewed_paths
    assert "src/good.py" in progress.reviewed_paths


@pytest.mark.asyncio
async def test_security_failure_is_reported_separately(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))

    class _SecurityFailLLM(_ReviewLLM):
        async def _complete_with_tools(self, messages, tools, temperature=None, force_tool=None):
            raise StructuredOutputError(
                "structured_output", model="m", tool="submit_review", excerpt="prose"
            )

    config = MiraConfig()
    config.review.security_pass = True
    config.review.security_agentic = False
    config.review.walkthrough = False
    config.review.self_critique = False
    llm = _SecurityFailLLM(set())
    engine = _engine(llm, security_llm=llm, config=config)

    result = await engine.review_diff(_diff("src/auth.py"))

    assert len(result.security_failures) == 1
    assert result.failed_chunks == []
    assert any(e["stage"] == "security_failed" for e in result.audit)
    md = WalkthroughResult(summary=result.summary).to_markdown(
        security_failures=result.security_failures
    )
    assert "Security review incomplete" in md


@pytest.mark.asyncio
async def test_dependency_failure_is_logged_and_audited(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))

    class _DependencyFailLLM(_ReviewLLM):
        async def _complete_with_tools(self, messages, tools, temperature=None, force_tool=None):
            raise StructuredOutputError(
                "structured_output", model="m", tool="submit_review", excerpt="prose"
            )

    manifest_diff = """diff --git a/package.json b/package.json
index 1111111..2222222 100644
--- a/package.json
+++ b/package.json
@@ -1,3 +1,4 @@
 {
+  "left-pad": "^1.0.0",
   "react": "^18.0.0"
 }
"""
    config = MiraConfig()
    config.review.dependency_overlap = True
    config.review.security_pass = False
    config.review.walkthrough = False
    config.review.self_critique = False
    llm = _DependencyFailLLM(set())
    engine = _engine(llm, config=config)

    result = await engine.review_diff(manifest_diff)

    assert len(result.dependency_failures) == 1
    assert result.failed_chunks == []
    assert any(e["stage"] == "dependency_failed" for e in result.audit)


def test_response_excerpt_is_bounded_and_single_line():
    excerpt = _response_excerpt("secret\n" + "x" * 1000)
    assert "\n" not in excerpt
    assert len(excerpt) <= 301


# --- fallback to a single review call when the agentic pass output is unusable ---

_NO_PATH_RESPONSE = (
    '{"comments": [{"line": 3, "body": "missing path", "agent_prompt": "x"}], '
    '"summary": "s", "metadata": {"reviewed_files": 1}}'
)
_GOOD_RESPONSE = (
    '{"comments": [{"path": "src/a.py", "line": 1, "body": "real finding", '
    '"severity": "warning", "confidence": 0.9}], "summary": "s", '
    '"metadata": {"reviewed_files": 1}}'
)


class _ParseFailLLM(_ReviewLLM):
    """Agentic pass returns comments without a path; plain review is scripted."""

    def __init__(self, review_text) -> None:
        super().__init__(set())
        self._review_text = review_text
        self.review_calls = 0

    async def complete_agentic(self, messages, tools):
        return _submit(json.loads(_NO_PATH_RESPONSE))

    async def review(self, messages, temperature=None):
        self.review_calls += 1
        if isinstance(self._review_text, Exception):
            raise self._review_text
        return self._review_text


@pytest.mark.asyncio
async def test_unparseable_agentic_output_falls_back_to_single_review(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    llm = _ParseFailLLM(_GOOD_RESPONSE)
    engine = _engine(llm)
    engine._agentic_source_fetcher = _FakeFetcher()
    engine._agentic_repo_tree = ["src/a.py"]

    result = await engine.review_diff(_diff("src/a.py"))

    assert result.failed_chunks == []
    assert llm.review_calls == 1
    assert [c.path for c in result.comments] == ["src/a.py"]


@pytest.mark.asyncio
async def test_salvage_keeps_valid_comments_when_fallback_also_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    mixed = json.dumps(
        {
            "comments": [
                {"line": 3, "body": "no path"},
                {"path": "src/a.py", "line": 1, "body": "kept", "confidence": 0.9},
            ],
            "summary": "s",
            "metadata": {"reviewed_files": 1},
        }
    )

    class _Mixed(_ParseFailLLM):
        async def complete_agentic(self, messages, tools):
            return _submit(json.loads(mixed))

    llm = _Mixed(StructuredOutputError("structured_output", model="m", tool="t", excerpt="x"))
    engine = _engine(llm)
    engine._agentic_source_fetcher = _FakeFetcher()
    engine._agentic_repo_tree = ["src/a.py"]

    result = await engine.review_diff(_diff("src/a.py"))

    assert result.failed_chunks == []
    assert [c.path for c in result.comments] == ["src/a.py"]


@pytest.mark.asyncio
async def test_chunk_still_fails_when_fallback_and_salvage_both_fail(monkeypatch, tmp_path):
    monkeypatch.setenv("MIRA_INDEX_DIR", str(tmp_path))
    llm = _ParseFailLLM(
        StructuredOutputError("structured_output", model="m", tool="t", excerpt="x")
    )
    engine = _engine(llm)
    engine._agentic_source_fetcher = _FakeFetcher()
    engine._agentic_repo_tree = ["src/a.py"]

    result = await engine.review_diff(_diff("src/a.py"))

    assert len(result.failed_chunks) == 1
    assert llm.review_calls == 1
