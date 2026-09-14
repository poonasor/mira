"""Codex CLI-backed provider using the local Codex OAuth session.

This provider intentionally keeps Mira's review contract unchanged: callers pass
Mira's existing chat messages/tool schemas and receive the same JSON strings the
OpenAI-compatible provider would have returned. The only difference is that the
model execution happens through ``codex exec`` and ``CODEX_HOME`` instead of an
HTTP API key.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tempfile
from datetime import timedelta
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mira.exceptions import LLMError
from mira.llm.cli_base import CLIProviderBase
from mira.llm.codex_auth import CodexAuthSync

logger = logging.getLogger(__name__)

# The prompt carries untrusted pull-request content, and the read-only sandbox
# blocks writes but not reads, so every Codex tool that can run commands, read
# files, reach the network, or start other agents is switched off. `--disable`
# rejects unknown feature names, so a Codex release that drops one of these fails
# the review loudly instead of silently re-enabling a tool.
_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "view_image",
    "multi_agent",
    "multi_agent_v2",
    "code_mode",
    "apps",
    "plugins",
    "browser_use",
    "computer_use",
    "image_generation",
    "hooks",
)


class CodexCLIProvider(CLIProviderBase):
    """LLM provider that shells out to OpenAI Codex CLI.

    Auth is provided by Codex itself, normally via ``$CODEX_HOME/auth.json`` from
    ``codex login``. No OpenAI API key is read or sent by this provider.
    """

    backend_label = "Codex CLI"
    error_prefix = "codex"

    def _env(self, runtime_home: str, runtime_codex_home: str) -> dict[str, str]:
        """Build a minimal child environment without Mira service credentials."""
        env = self._safe_env()
        env["HOME"] = runtime_home
        env["CODEX_HOME"] = runtime_codex_home
        return env

    def _source_auth_path(self) -> Path | None:
        """The operator's ``auth.json``, or None to let Codex use its defaults."""
        source_home = self.config.codex_home or os.environ.get("CODEX_HOME")
        if not source_home:
            return None
        source_auth = Path(source_home).expanduser() / "auth.json"
        if not source_auth.is_file():
            raise LLMError("codex_auth_file_missing", path=str(source_auth))
        return source_auth

    def _prepare_codex_home(self, invocation_root: str) -> str:
        """Create writable ephemeral Codex state containing only OAuth auth."""
        destination = Path(invocation_root) / "codex-home"
        destination.mkdir(parents=True, mode=0o700)
        source_auth = self._source_auth_path()
        if source_auth:
            destination_auth = destination / "auth.json"
            destination_auth.write_bytes(source_auth.read_bytes())
            destination_auth.chmod(0o600)
        return str(destination)

    def _command(self, output_path: str) -> list[str]:
        codex_command = self.config.codex_command or "codex"
        if any(char in codex_command for char in (" ", "\t", "\n", ";", "|", "&")):
            raise ValueError(
                f"Invalid codex_command: {codex_command!r}. "
                "Set it to a single executable path/name without arguments."
            )
        cmd = [
            codex_command,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            self.config.codex_sandbox,
            "--output-last-message",
            output_path,
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "-c",
            'shell_environment_policy.inherit="none"',
        ]
        for feature in _DISABLED_FEATURES:
            cmd.extend(["--disable", feature])
        cmd.extend(["-c", 'web_search="disabled"'])
        if self.config.model not in {"", "default", "codex-default"}:
            cmd.extend(["-m", self.config.model])
        cmd.append("-")
        return cmd

    async def _run(self, prompt: str) -> str:
        return await self._run_codex(prompt)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(LLMError),
        reraise=True,
    )
    async def _run_codex(self, prompt: str) -> str:
        with tempfile.TemporaryDirectory(prefix="mira-codex-") as tmpdir:
            runtime_codex_home = self._prepare_codex_home(tmpdir)
            source_auth = self._source_auth_path()
            if source_auth is None:
                return await self._exec_codex(prompt, tmpdir, runtime_codex_home)

            # Codex rotates OAuth refresh tokens into the temporary copy; save them back.
            auth_sync = CodexAuthSync(source_auth, Path(runtime_codex_home) / "auth.json")
            try:
                await auth_sync.start(timedelta(seconds=self.config.codex_timeout_seconds))
                watcher = asyncio.create_task(auth_sync.watch())
                try:
                    return await self._exec_codex(prompt, tmpdir, runtime_codex_home)
                finally:
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
            finally:
                # Also after failed, timed-out, or cancelled runs: the refresh already happened.
                auth_sync.close()

    async def _exec_codex(self, prompt: str, tmpdir: str, runtime_codex_home: str) -> str:
        output_path = str(Path(tmpdir) / "last-message.txt")
        runtime_home = str(Path(tmpdir) / "runtime")
        Path(runtime_home).mkdir(mode=0o700)
        cmd = self._command(output_path)
        logger.debug("Running Codex CLI provider: %s", shlex.join(cmd[:-1] + ["<stdin>"]))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env(runtime_home, runtime_codex_home),
                cwd=runtime_home,
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as exc:
            raise LLMError("codex_command_not_found", command=self.config.codex_command) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")),
                timeout=self.config.codex_timeout_seconds,
            )
        except TimeoutError as exc:
            await self._terminate_process_tree(proc)
            raise LLMError("codex_timeout", seconds=self.config.codex_timeout_seconds) from exc
        except BaseException:
            await self._terminate_process_tree(proc)
            raise

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        output_file = Path(output_path)
        last_message = output_file.read_text(encoding="utf-8") if output_file.exists() else ""

        if proc.returncode != 0:
            detail = (stderr_text or stdout_text or last_message).strip()
            raise LLMError(
                "codex_exit_failed",
                exit_code=proc.returncode,
                detail=detail[-2000:],
            )

        return (last_message or stdout_text).strip()
