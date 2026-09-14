"""Claude Code CLI-backed provider authenticated with a Claude subscription.

The OAuth token comes from ``claude setup-token`` and reaches the child process
only as ``CLAUDE_CODE_OAUTH_TOKEN``; no Anthropic API key is read or sent. The
prompt carries untrusted pull-request content, so the CLI runs with no tools,
settings, hooks, MCP servers, slash commands, or saved session, in an empty
working directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import tempfile
import weakref
from pathlib import Path

from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm.cli_base import CLIProviderBase

logger = logging.getLogger(__name__)

PROVIDER_NAMES = frozenset({"claude-cli", "claude_cli", "claude"})

# The CLI retries API errors itself (10 times by default); keep that short so a
# rate-limited subscription hands off to the next tier quickly.
_CLI_MAX_RETRIES = 2

_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _concurrency_gate(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    gate = _gates.get(loop)
    if gate is None:
        gate = _gates[loop] = asyncio.Semaphore(limit)
    return gate


class ClaudeCLIProvider(CLIProviderBase):
    """LLM provider that shells out to Anthropic's Claude Code CLI."""

    backend_label = "Claude Code CLI"
    error_prefix = "claude"
    records_own_usage = True

    def _token(self) -> str:
        token_env = self.config.claude_oauth_token_env
        token = os.environ.get(token_env, "")
        if not token:
            raise NonRetriableLLMError("claude_token_missing", token_env=token_env)
        return token

    def _env(self, runtime_home: str, config_dir: str) -> dict[str, str]:
        env = self._safe_env()
        env.update(
            {
                "HOME": runtime_home,
                "CLAUDE_CONFIG_DIR": config_dir,
                "CLAUDE_CODE_OAUTH_TOKEN": self._token(),
                "DISABLE_AUTOUPDATER": "1",
                "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "CLAUDE_CODE_MAX_RETRIES": str(_CLI_MAX_RETRIES),
                "API_TIMEOUT_MS": str(
                    int(self.config.claude_timeout_seconds * 1000) // (_CLI_MAX_RETRIES + 1)
                ),
            }
        )
        return env

    def _command(self) -> list[str]:
        command = self.config.claude_command or "claude"
        if any(char in command for char in (" ", "\t", "\n", ";", "|", "&")):
            raise ValueError(
                f"Invalid claude_command: {command!r}. "
                "Set it to a single executable path/name without arguments."
            )
        return [
            command,
            "-p",
            "--output-format",
            "json",
            # Bound form, so a model id starting with "-" can't become an option.
            f"--model={self.config.model}",
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

    async def _run(self, prompt: str) -> str:
        async with _concurrency_gate(self.config.claude_max_concurrency):
            with tempfile.TemporaryDirectory(prefix="mira-claude-") as tmpdir:
                runtime_home, config_dir, workdir = (
                    Path(tmpdir) / name for name in ("home", "config", "work")
                )
                for directory in (runtime_home, config_dir, workdir):
                    directory.mkdir(mode=0o700)
                env = self._env(str(runtime_home), str(config_dir))
                cmd = self._command()
                logger.debug("Running Claude CLI provider: %s", shlex.join(cmd))
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        cwd=str(workdir),
                        start_new_session=os.name == "posix",
                    )
                except FileNotFoundError as exc:
                    raise LLMError("claude_command_not_found", command=cmd[0]) from exc

                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(prompt.encode("utf-8")),
                        timeout=self.config.claude_timeout_seconds,
                    )
                except TimeoutError as exc:
                    await self._terminate_process_tree(proc)
                    raise LLMError(
                        "claude_timeout", seconds=self.config.claude_timeout_seconds
                    ) from exc
                except BaseException:
                    await self._terminate_process_tree(proc)
                    raise

        return self._parse_output(
            prompt,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            proc.returncode,
        )

    def _parse_output(self, prompt: str, stdout: str, stderr: str, returncode: int | None) -> str:
        # Failures are reported as a JSON result envelope on stdout with a
        # non-zero exit, so the envelope is read before the exit code.
        try:
            envelope = json.loads(stdout)
        except ValueError:
            envelope = None
        if not isinstance(envelope, dict):
            if returncode != 0:
                detail = (stderr or stdout).strip()[-2000:]
                raise LLMError("claude_exit_failed", exit_code=returncode, detail=detail)
            raise LLMError("claude_malformed_output", excerpt=stdout[:1000])

        if envelope.get("is_error") or returncode != 0:
            raise self._envelope_error(envelope, returncode)

        result = str(envelope.get("result") or "")
        self._record_usage(envelope.get("usage"), prompt, result)
        return result

    @staticmethod
    def _envelope_error(envelope: dict, returncode: int | None) -> LLMError:
        status = envelope.get("api_error_status")
        detail = str(envelope.get("result") or envelope.get("subtype") or "unknown error")[:500]
        lowered = detail.lower()
        if status in (401, 403) or "not logged in" in lowered or "authenticate" in lowered:
            return LLMError("claude_auth_failed", status=status, detail=detail)
        if status == 429 or any(
            term in lowered for term in ("usage limit", "rate limit", "limit reached")
        ):
            return LLMError("claude_usage_limit", status=status, detail=detail)
        return LLMError("claude_api_error", status=status, exit_code=returncode, detail=detail)

    def _record_usage(self, usage: object, prompt: str, result: str) -> None:
        if isinstance(usage, dict) and "input_tokens" in usage:
            self.total_prompt_tokens += (
                int(usage.get("input_tokens") or 0)
                + int(usage.get("cache_creation_input_tokens") or 0)
                + int(usage.get("cache_read_input_tokens") or 0)
            )
            self.total_completion_tokens += int(usage.get("output_tokens") or 0)
            return
        self.total_prompt_tokens += self.count_tokens(prompt)
        self.total_completion_tokens += self.count_tokens(result)
