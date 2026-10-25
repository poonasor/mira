"""Claude Code CLI-backed provider authenticated with a Claude subscription.

The OAuth token comes from ``claude setup-token`` and reaches the child process
only as ``CLAUDE_CODE_OAUTH_TOKEN``; no Anthropic API key is read or sent. The
prompt carries untrusted pull-request content, so the CLI runs with no tools,
settings, hooks, MCP servers or slash commands, in an empty working directory.
Agentic review hops still work: the model names the read_file/grep_repo calls
it wants as JSON and Mira runs them (see cli_base). One agentic loop is one CLI
session (``--session-id`` then ``--resume``) kept in a private temporary
directory that is deleted when the loop ends, so later hops send only the new
tool results and the API serves the earlier turns from its prompt cache.
Every other call runs with ``--no-session-persistence``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import tempfile
import uuid
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mira.config import LLMConfig
from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm.cli_base import CLIProviderBase

logger = logging.getLogger(__name__)

PROVIDER_NAMES = frozenset({"claude-cli", "claude_cli", "claude"})

# The CLI retries API errors itself (10 times by default); keep that short so a
# rate-limited subscription hands off to the next tier quickly.
_CLI_MAX_RETRIES = 2

# Oldest Claude Code release that knows claude-sonnet-5-5 (Mira's default
# claude-cli model). The image installs the newest release on every build;
# this only drives the startup warning.
MIN_CLI_VERSION = (2, 1, 284)
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")

_gates: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _concurrency_gate(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    gate = _gates.get(loop)
    if gate is None:
        gate = _gates[loop] = asyncio.Semaphore(limit)
    return gate


@dataclass
class _Session:
    """A resumable CLI session: its id and the private dirs it lives in."""

    session_id: str
    tmp: tempfile.TemporaryDirectory

    @property
    def root(self) -> Path:
        return Path(self.tmp.name)


class ClaudeCLIProvider(CLIProviderBase):
    """LLM provider that shells out to Anthropic's Claude Code CLI."""

    backend_label = "Claude Code CLI"
    error_prefix = "claude"
    records_own_usage = True
    # Agentic hops are emulated over text (see cli_base), so the CLI itself
    # still runs with `--tools ""` while Mira executes read_file/grep_repo.
    supports_tool_calling = True
    supports_session_resume = True

    def _new_session(self) -> _Session:
        session = _Session(
            session_id=str(uuid.uuid4()),
            tmp=tempfile.TemporaryDirectory(prefix="mira-claude-session-"),
        )
        for name in ("home", "config", "work"):
            (session.root / name).mkdir(mode=0o700)
        return session

    def _close_session(self, session: object) -> None:
        if isinstance(session, _Session):
            session.tmp.cleanup()

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

    def _command(self, session_args: list[str] | None = None) -> list[str]:
        command = self.config.claude_command or "claude"
        if any(char in command for char in (" ", "\t", "\n", ";", "|", "&")):
            raise ValueError(
                f"Invalid claude_command: {command!r}. "
                "Set it to a single executable path/name without arguments."
            )
        effort = self.config.reasoning_effort
        effort_args = (
            [f"--effort={effort}"] if effort in ("low", "medium", "high", "xhigh", "max") else []
        )
        return [
            command,
            "-p",
            "--output-format",
            "json",
            # Bound form, so a model id starting with "-" can't become an option.
            f"--model={self.config.model}",
            *effort_args,
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
            *(session_args if session_args is not None else ["--no-session-persistence"]),
        ]

    async def _run(self, prompt: str, session: _Session | None = None, resume: bool = False) -> str:
        async with _concurrency_gate(self.config.claude_max_concurrency):
            with (
                tempfile.TemporaryDirectory(prefix="mira-claude-")
                if session is None
                else _keep(session.tmp.name)
            ) as tmpdir:
                runtime_home, config_dir, workdir = (
                    Path(tmpdir) / name for name in ("home", "config", "work")
                )
                if session is None:
                    for directory in (runtime_home, config_dir, workdir):
                        directory.mkdir(mode=0o700)
                    cmd = self._command()
                else:
                    flag = "--resume" if resume else "--session-id"
                    cmd = self._command([flag, session.session_id])
                env = self._env(str(runtime_home), str(config_dir))
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


@contextmanager
def _keep(path: str) -> Iterator[str]:
    """Yield an existing session directory without deleting it afterwards."""
    yield path


def claude_cli_configs(config: LLMConfig) -> list[LLMConfig]:
    """The configured tiers (primary, then failover) that use the Claude CLI."""
    tiers = [config, *([config.failover] if config.failover else [])]
    return [tier for tier in tiers if tier.provider.lower() in PROVIDER_NAMES]


def parse_cli_version(text: str) -> tuple[int, int, int] | None:
    match = _VERSION_RE.search(text)
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


async def log_claude_cli_version(config: LLMConfig, timeout: float = 15.0) -> str | None:
    """Log the installed ``claude --version`` once; warn when it is too old.

    Never raises: a missing or broken CLI is reported here and again, as a
    normal provider failure, on the first call.
    """
    tiers = claude_cli_configs(config)
    if not tiers:
        return None
    command = tiers[0].claude_command or "claude"
    try:
        with tempfile.TemporaryDirectory(prefix="mira-claude-version-") as home:
            env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL")}
            env.update({"HOME": home, "CLAUDE_CONFIG_DIR": home, "DISABLE_AUTOUPDATER": "1"})
            proc = await asyncio.create_subprocess_exec(
                command,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=home,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise
    except Exception as exc:
        logger.warning("Claude Code CLI version check failed for %r: %s", command, exc)
        return None
    text = stdout.decode("utf-8", errors="replace").strip()
    version = parse_cli_version(text)
    if version is None:
        logger.warning("Claude Code CLI %r printed no version: %r", command, text[:200])
        return None
    shown = ".".join(map(str, version))
    if version < MIN_CLI_VERSION:
        logger.warning(
            "Claude Code CLI %s is older than %s, which claude-sonnet-5-5 needs; "
            "rebuild the image to install the latest CLI",
            shown,
            ".".join(map(str, MIN_CLI_VERSION)),
        )
    else:
        logger.info("Claude Code CLI %s (models: %s)", shown, ", ".join(t.model for t in tiers))
    return shown
