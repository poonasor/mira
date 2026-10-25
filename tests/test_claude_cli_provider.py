from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import LLMConfig
from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm import create_llm
from mira.llm.agentic_tools import AGENTIC_TOOLS
from mira.llm.claude_cli import ClaudeCLIProvider
from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL

FIXTURES = Path(__file__).parent / "fixtures" / "claude_cli"
TOKEN = "sk-ant-oat01-test-token"


def envelope(**overrides: object) -> dict:
    data: dict = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "api_error_status": None,
        "result": '{"ok": true}',
        "usage": {
            "input_tokens": 10,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 20,
            "output_tokens": 7,
        },
    }
    data.update(overrides)
    return data


def fake_process(stdout: str, returncode: int, stderr: str = "") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.pid = 4321
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    return proc


def provider(**overrides: object) -> ClaudeCLIProvider:
    return ClaudeCLIProvider(
        LLMConfig(**{"provider": "claude-cli", "model": "sonnet", **overrides})
    )


@pytest.fixture
def token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)


@pytest.fixture
def spawn(monkeypatch: pytest.MonkeyPatch):
    def install(proc: MagicMock) -> AsyncMock:
        mock = AsyncMock(return_value=proc)
        monkeypatch.setattr("asyncio.create_subprocess_exec", mock)
        return mock

    return install


class TestCommand:
    def test_factory_selects_claude_cli_provider(self):
        assert isinstance(create_llm(LLMConfig(provider="claude-cli")), ClaudeCLIProvider)

    def test_command_runs_one_turn_with_no_tools_settings_hooks_or_mcp(self):
        assert provider(claude_command="/usr/local/bin/claude")._command() == [
            "/usr/local/bin/claude",
            "-p",
            "--output-format",
            "json",
            "--model=sonnet",
            "--tools",
            "",
            "--max-turns",
            "1",
            "--strict-mcp-config",
            "--setting-sources",
            "user",
            "--settings",
            '{"disableAllHooks":true}',
            "--disable-slash-commands",
            "--no-session-persistence",
        ]

    def test_model_is_bound_to_its_flag_so_it_cannot_inject_options(self):
        cmd = provider(model="--dangerously-skip-permissions")._command()

        assert "--model=--dangerously-skip-permissions" in cmd
        assert "--dangerously-skip-permissions" not in cmd

    def test_command_rejects_shell_metacharacters_or_arguments(self):
        with pytest.raises(ValueError, match="Invalid claude_command"):
            provider(claude_command="claude --dangerously-skip-permissions")._command()


class TestEnvironment:
    def test_child_env_carries_only_the_subscription_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        leaked = {
            "ANTHROPIC_API_KEY": "sk-ant-api-key",
            "ANTHROPIC_AUTH_TOKEN": "bearer",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "ZAI_API_KEY": "zai-key",
            "GITHUB_TOKEN": "gh-token",
            "MIRA_WEBHOOK_SECRET": "hook-secret",
        }
        for name, value in leaked.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)

        env = provider()._env(str(tmp_path / "home"), str(tmp_path / "config"))

        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
        assert env["HOME"] == str(tmp_path / "home")
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "config")
        assert env["PATH"] == "/usr/bin"
        assert env["DISABLE_AUTOUPDATER"] == "1"
        assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
        assert env["CLAUDE_CODE_MAX_RETRIES"] == "2"
        for name in leaked:
            assert name not in env

    def test_token_can_come_from_a_custom_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        monkeypatch.setenv("MIRA_CLAUDE_TOKEN", TOKEN)

        env = provider(claude_oauth_token_env="MIRA_CLAUDE_TOKEN")._env(
            str(tmp_path), str(tmp_path)
        )

        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
        assert "MIRA_CLAUDE_TOKEN" not in env

    @pytest.mark.asyncio
    async def test_missing_token_fails_before_starting_the_cli(
        self, monkeypatch: pytest.MonkeyPatch, spawn
    ):
        monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
        mock = spawn(fake_process(json.dumps(envelope()), 0))

        with pytest.raises(NonRetriableLLMError) as info:
            await provider()._run("hi")

        assert info.value.code == "claude_token_missing"
        mock.assert_not_awaited()


class TestRun:
    @pytest.mark.asyncio
    async def test_prompt_goes_over_stdin_from_an_isolated_directory(self, token, spawn):
        proc = fake_process(json.dumps(envelope()), 0)
        mock = spawn(proc)

        assert await provider()._run("review this diff") == '{"ok": true}'

        assert mock.await_args is not None
        kwargs = mock.await_args.kwargs
        assert kwargs["start_new_session"] is True
        assert Path(kwargs["cwd"]).name == "work"
        assert kwargs["cwd"] != kwargs["env"]["HOME"]
        proc.communicate.assert_awaited_once_with(b"review this diff")

    @pytest.mark.asyncio
    async def test_records_exact_usage_from_a_recorded_result_envelope(self, token, spawn):
        spawn(fake_process((FIXTURES / "success.json").read_text(), 0))
        llm = provider()

        result = await llm.complete([{"role": "user", "content": "return json"}])

        assert result == '{"ok": true, "n": 3}'
        assert llm.usage == {"prompt_tokens": 8483, "completion_tokens": 15, "total_tokens": 8498}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("fixture", "status"), [("auth_invalid.json", 401), ("not_logged_in.json", None)]
    )
    async def test_recorded_auth_failures_map_to_claude_auth_failed(
        self, fixture: str, status: int | None, token, spawn
    ):
        spawn(fake_process((FIXTURES / fixture).read_text(), 1))

        with pytest.raises(LLMError) as info:
            await provider()._run("hi")

        assert info.value.code == "claude_auth_failed"
        assert info.value.details["status"] == status

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "result", "code"),
        [
            (429, "API Error: 429 rate_limit_error", "claude_usage_limit"),
            (
                None,
                "Claude usage limit reached. Your limit will reset at 5pm",
                "claude_usage_limit",
            ),
            (529, "API Error: 529 Overloaded", "claude_api_error"),
        ],
    )
    async def test_error_envelopes_are_classified(
        self, status: int | None, result: str, code: str, token, spawn
    ):
        spawn(
            fake_process(
                json.dumps(envelope(is_error=True, api_error_status=status, result=result)), 1
            )
        )

        with pytest.raises(LLMError) as info:
            await provider()._run("hi")

        assert info.value.code == code
        assert info.value.details["status"] == status

    @pytest.mark.asyncio
    async def test_non_json_failure_reports_the_exit_code(self, token, spawn):
        spawn(fake_process("Segmentation fault", 139, stderr="boom"))

        with pytest.raises(LLMError) as info:
            await provider()._run("hi")

        assert info.value.code == "claude_exit_failed"
        assert info.value.details["exit_code"] == 139

    @pytest.mark.asyncio
    async def test_missing_binary_is_reported(self, token, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError)
        )

        with pytest.raises(LLMError) as info:
            await provider()._run("hi")

        assert info.value.code == "claude_command_not_found"

    @pytest.mark.asyncio
    async def test_timeout_kills_the_process_group(self, token, spawn):
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 99

        async def hang(data: bytes) -> tuple[bytes, bytes]:
            await asyncio.Event().wait()
            return b"", b""

        proc.communicate = hang
        spawn(proc)
        # model_copy skips validation, allowing a sub-second timeout for the test.
        llm = ClaudeCLIProvider(
            LLMConfig(provider="claude-cli").model_copy(update={"claude_timeout_seconds": 0.01})
        )
        llm._terminate_process_tree = AsyncMock()  # type: ignore[method-assign]

        with pytest.raises(LLMError) as info:
            await llm._run("hi")

        assert info.value.code == "claude_timeout"
        llm._terminate_process_tree.assert_awaited_once_with(proc)

    @pytest.mark.asyncio
    async def test_concurrent_calls_are_capped(self, token, monkeypatch: pytest.MonkeyPatch):
        active = peak = 0
        release = asyncio.Event()

        async def spawn_process(*args: object, **kwargs: object) -> MagicMock:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            proc = MagicMock()
            proc.returncode = 0

            async def communicate(data: bytes) -> tuple[bytes, bytes]:
                nonlocal active
                await release.wait()
                active -= 1
                return json.dumps(envelope()).encode(), b""

            proc.communicate = communicate
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", spawn_process)
        llm = provider(claude_max_concurrency=2)

        tasks = [asyncio.create_task(llm._run("hi")) for _ in range(4)]
        for _ in range(20):
            await asyncio.sleep(0)
        assert peak == 2

        release.set()
        assert await asyncio.gather(*tasks) == ['{"ok": true}'] * 4
        assert peak == 2


class TestProtocol:
    @pytest.mark.asyncio
    async def test_review_prompt_embeds_the_submit_review_schema(self):
        llm = provider()
        llm._run = AsyncMock(return_value='{"comments": [], "summary": "ok"}')  # type: ignore[method-assign]

        result = await llm.review([{"role": "system", "content": "be strict"}])

        assert result == '{"comments": [], "summary": "ok"}'
        assert llm._run.await_args is not None
        prompt = llm._run.await_args.args[0]
        assert "Claude Code CLI" in prompt
        assert "`submit_review`" in prompt

    def test_claude_cli_supports_agentic_tool_calling(self):
        assert provider().supports_tool_calling is True

    @pytest.mark.asyncio
    async def test_complete_agentic_returns_openai_style_tool_calls(self):
        llm = provider(model="claude-sonnet-5-5")
        llm._run = AsyncMock(  # type: ignore[method-assign]
            return_value='{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}'
        )

        msg = await llm.complete_agentic(
            [{"role": "user", "content": "hi"}], tools=[*AGENTIC_TOOLS, SUBMIT_REVIEW_TOOL]
        )

        [call] = msg["tool_calls"]
        assert call["type"] == "function"
        assert call["id"].startswith("call_")
        assert call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}
        prompt = llm._run.await_args.args[0]
        assert '"read_file"' in prompt and '"submit_review"' in prompt

    @pytest.mark.asyncio
    async def test_complete_agentic_replays_tool_results_into_the_next_prompt(self):
        llm = provider()
        llm._run = AsyncMock(return_value='{"tool_calls": []}')  # type: ignore[method-assign]
        convo = [
            {"role": "user", "content": "review"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "1: def f(): pass"},
        ]

        msg = await llm.complete_agentic(convo, tools=list(AGENTIC_TOOLS))

        assert msg == {"content": "", "tool_calls": []}
        prompt = llm._run.await_args.args[0]
        assert "tool result for call call_1" in prompt
        assert "1: def f(): pass" in prompt
        assert '"name":"read_file"' in prompt

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw",
        [
            '{"tool_calls": [{"name": "run_shell", "arguments": {}}]}',
            '{"tool_calls": [{"name": "read_file", "arguments": "not json"}]}',
            '{"tool_calls": [{"name": "read_file", "arguments": [1]}]}',
            '{"tool_calls": "read_file"}',
        ],
        ids=["unknown-tool", "string-args", "list-args", "not-a-list"],
    )
    async def test_invalid_tool_calls_raise_a_claude_error(self, raw: str):
        llm = provider()
        llm._run = AsyncMock(return_value=raw)  # type: ignore[method-assign]

        with pytest.raises(LLMError) as info:
            await llm.complete_agentic([{"role": "user", "content": "hi"}], tools=AGENTIC_TOOLS)

        assert info.value.code == "claude_invalid_tool_call"

    @pytest.mark.asyncio
    async def test_agentic_review_loop_runs_tools_then_submits_through_claude_cli(self):
        from mira.core.passes import agentic_review_loop

        review = {"comments": [], "summary": "looks fine"}
        llm = provider(model="claude-sonnet-5-5")
        llm._run = AsyncMock(  # type: ignore[method-assign]
            side_effect=[
                '{"tool_calls": [{"name": "read_file", "arguments": {"path": "a.py"}}]}',
                json.dumps({"tool_calls": [{"name": "submit_review", "arguments": review}]}),
            ]
        )
        executor = MagicMock()
        executor.execute = AsyncMock(return_value="1: x = 1")

        raw = await agentic_review_loop(
            llm,
            [{"role": "system", "content": "review"}, {"role": "user", "content": "d"}],
            executor,
        )

        assert json.loads(raw) == review
        executor.execute.assert_awaited_once_with("read_file", {"path": "a.py"})
        second_prompt = llm._run.await_args_list[1].args[0]
        assert "1: x = 1" in second_prompt

    def test_json_extraction_errors_use_claude_codes(self):
        with pytest.raises(LLMError) as info:
            provider()._extract_json_object("[1, 2]")

        assert info.value.code == "claude_non_object_json"


def _hop_loop_provider(replies: list[str]) -> tuple[ClaudeCLIProvider, list[dict]]:
    """A provider whose _run records (prompt, session, resume) per call."""
    llm = provider(model="claude-sonnet-5-5")
    calls: list[dict] = []
    queue = list(replies)

    async def fake_run(prompt: str, session=None, resume: bool = False) -> str:
        calls.append({"prompt": prompt, "session": session, "resume": resume})
        return queue.pop(0)

    llm._run = fake_run  # type: ignore[method-assign]
    return llm, calls


def _read(path: str) -> str:
    return json.dumps({"tool_calls": [{"name": "read_file", "arguments": {"path": path}}]})


class TestAgenticSessionResume:
    def test_session_flags_replace_no_session_persistence(self):
        llm = provider()
        assert llm._command()[-1] == "--no-session-persistence"
        first = llm._command(["--session-id", "abc"])
        later = llm._command(["--resume", "abc"])
        assert "--no-session-persistence" not in first + later
        assert first[-2:] == ["--session-id", "abc"]
        assert later[-2:] == ["--resume", "abc"]
        # Everything else (sandboxing flags) is identical to the stateless call.
        assert first[:-2] == llm._command()[:-1]

    @pytest.mark.asyncio
    async def test_later_hops_resume_and_send_only_new_tool_results(self):
        from mira.core.passes import agentic_review_loop

        review = {"comments": [], "summary": "ok"}
        llm, calls = _hop_loop_provider(
            [
                _read("a.py"),
                _read("b.py"),
                json.dumps({"tool_calls": [{"name": "submit_review", "arguments": review}]}),
            ]
        )
        executor = MagicMock()
        executor.max_hops = 6
        executor.execute = AsyncMock(side_effect=["A-CONTENT", "B-CONTENT"])
        system = "SYSTEM-PROMPT " * 200
        diff = "DIFF-LINE\n" * 500

        raw = await agentic_review_loop(
            llm,
            [{"role": "system", "content": system}, {"role": "user", "content": diff}],
            executor,
        )

        assert json.loads(raw) == review
        assert [c["resume"] for c in calls] == [False, True, True]
        session = calls[0]["session"]
        assert session is not None and all(c["session"] is session for c in calls)
        first, second, third = (c["prompt"] for c in calls)
        assert "SYSTEM-PROMPT" in first and "DIFF-LINE" in first and '"read_file"' in first
        for later in (second, third):
            assert "SYSTEM-PROMPT" not in later
            assert "DIFF-LINE" not in later
            assert '"description"' not in later  # tool schemas aren't re-sent
            assert '{"tool_calls"' in later  # but the output contract is
        assert "A-CONTENT" in second and "B-CONTENT" not in second
        assert "B-CONTENT" in third and "A-CONTENT" not in third
        assert 'read_file {"path": "b.py"}' in third
        # The loop closed the session: its private directory is gone.
        assert not session.root.exists()
        assert llm._threads == {}

    @pytest.mark.asyncio
    async def test_foreign_assistant_reply_starts_a_fresh_session(self):
        llm, calls = _hop_loop_provider([_read("a.py"), _read("b.py")])
        convo: list[dict] = [{"role": "user", "content": "DIFF"}]
        tools = list(AGENTIC_TOOLS)

        await llm.complete_agentic(convo, tools=tools)
        old_session = calls[0]["session"]
        # Another tier answered this hop, so the session never saw this reply.
        convo += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "other", "function": {}}]},
            {"role": "tool", "tool_call_id": "other", "content": "R"},
        ]
        await llm.complete_agentic(convo, tools=tools)

        assert calls[1]["resume"] is False
        assert calls[1]["session"] is not old_session
        assert "DIFF" in calls[1]["prompt"]
        assert not old_session.root.exists()
        await llm.close_agentic(convo)
        assert not calls[1]["session"].root.exists()

    @pytest.mark.asyncio
    async def test_failed_hop_drops_the_session_so_a_retry_resends_everything(self):
        llm = provider()
        seen: list[tuple] = []

        async def flaky(prompt: str, session=None, resume: bool = False) -> str:
            seen.append((prompt, session, resume))
            if len(seen) == 2:
                raise LLMError("claude_timeout", seconds=1)
            return _read("a.py")

        llm._run = flaky  # type: ignore[method-assign]
        convo: list[dict] = [{"role": "user", "content": "DIFF"}]
        msg = await llm.complete_agentic(convo, tools=list(AGENTIC_TOOLS))
        convo += [
            {"role": "assistant", "content": "", "tool_calls": msg["tool_calls"]},
            {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"], "content": "R"},
        ]
        with pytest.raises(LLMError):
            await llm.complete_agentic(convo, tools=list(AGENTIC_TOOLS))
        assert seen[1][2] is True
        assert not seen[0][1].root.exists()

        await llm.complete_agentic(convo, tools=list(AGENTIC_TOOLS))
        assert seen[2][2] is False and "DIFF" in seen[2][0]
        await llm.close_agentic(convo)

    @pytest.mark.asyncio
    async def test_run_uses_the_session_dirs_and_flags(self, token, spawn):
        llm = provider()
        session = llm._new_session()
        mock = spawn(fake_process(json.dumps(envelope(result='{"tool_calls": []}')), 0))

        await llm._run("p", session=session, resume=False)
        await llm._run("p", session=session, resume=True)

        first, later = (call.args for call in mock.await_args_list)
        assert first[-2:] == ("--session-id", session.session_id)
        assert later[-2:] == ("--resume", session.session_id)
        kwargs = mock.await_args_list[0].kwargs
        assert kwargs["cwd"] == str(session.root / "work")
        assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(session.root / "config")
        assert session.root.exists()  # kept between hops
        llm._close_session(session)
        assert not session.root.exists()

    @pytest.mark.asyncio
    async def test_tiered_provider_forwards_close_agentic(self):
        from mira.llm.tiered import TieredProvider

        llm, _ = _hop_loop_provider([_read("a.py")])
        tiered = TieredProvider([(llm.config, llm)], cooldown_seconds=60)
        convo: list[dict] = [{"role": "user", "content": "DIFF"}]
        await llm.complete_agentic(convo, tools=list(AGENTIC_TOOLS))
        assert llm._threads

        await tiered.close_agentic(convo)

        assert not llm._threads


class TestCliVersionCheck:
    def test_parse_cli_version(self):
        from mira.llm.claude_cli import parse_cli_version

        assert parse_cli_version("2.1.310 (Claude Code)\n") == (2, 1, 310)
        assert parse_cli_version("nope") is None

    @pytest.mark.asyncio
    async def test_no_claude_tier_skips_the_check(self, spawn):
        from mira.llm.claude_cli import log_claude_cli_version

        mock = spawn(fake_process("", 0))
        assert await log_claude_cli_version(LLMConfig(provider="openai", model="x")) is None
        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_logs_version_and_warns_when_too_old(self, spawn, caplog):
        from mira.llm.claude_cli import log_claude_cli_version

        config = LLMConfig(provider="claude-cli", model="claude-sonnet-5-5")
        spawn(fake_process("2.1.310 (Claude Code)\n", 0))
        with caplog.at_level("INFO", logger="mira.llm.claude_cli"):
            assert await log_claude_cli_version(config) == "2.1.310"
        assert "Claude Code CLI 2.1.310" in caplog.text
        assert "older than" not in caplog.text

        spawn(fake_process("2.1.270 (Claude Code)\n", 0))
        with caplog.at_level("WARNING", logger="mira.llm.claude_cli"):
            assert await log_claude_cli_version(config) == "2.1.270"
        assert "older than 2.1.284" in caplog.text

    @pytest.mark.asyncio
    async def test_missing_cli_never_raises(self, monkeypatch):
        from mira.llm.claude_cli import log_claude_cli_version

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec", AsyncMock(side_effect=FileNotFoundError("claude"))
        )
        config = LLMConfig(provider="claude-cli", model="sonnet")
        assert await log_claude_cli_version(config) is None
