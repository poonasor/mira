from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from mira.config import LLMConfig
from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm import create_llm
from mira.llm.claude_cli import ClaudeCLIProvider

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

    @pytest.mark.asyncio
    async def test_complete_agentic_defers_to_a_single_forced_review(self):
        llm = provider()
        llm._run = AsyncMock()  # type: ignore[method-assign]

        msg = await llm.complete_agentic([{"role": "user", "content": "hi"}], tools=[])

        assert msg == {"content": "", "tool_calls": []}
        llm._run.assert_not_awaited()

    def test_json_extraction_errors_use_claude_codes(self):
        with pytest.raises(LLMError) as info:
            provider()._extract_json_object("[1, 2]")

        assert info.value.code == "claude_non_object_json"
