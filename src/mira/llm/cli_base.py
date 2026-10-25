"""Shared plumbing for LLM providers that run a local CLI as a subprocess.

Callers pass Mira's existing chat messages and tool schemas and receive the
same JSON strings an HTTP provider would return. Subclasses only supply
``_run(prompt)``, which executes the CLI and returns its final answer text.

Agentic tool use is emulated over plain text: each hop renders the conversation
and the tool schemas into one prompt, the model answers with a JSON object
naming the tool calls it wants, and Mira executes them itself. The CLI never
gets tools of its own, so the sandboxing above is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import uuid
from contextlib import suppress

from mira.config import LLMConfig
from mira.exceptions import LLMError

SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
    }
)


class CLIProviderBase:
    """Base for providers that execute a model through a local CLI."""

    supports_json_mode: bool = True
    # Whether complete_agentic runs a real hop (text-emulated tool calls) or
    # defers straight to one forced review. TieredProvider routes agentic
    # calls only to tiers where this is True.
    supports_tool_calling: bool = False
    supports_temperature: bool = False
    supports_max_tokens: bool = False

    backend_label: str = "CLI"
    error_prefix: str = "cli"
    # True when _run() records exact token usage, so estimates aren't added on top.
    records_own_usage: bool = False

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4) if text else 0

    def _safe_env(self) -> dict[str, str]:
        """Minimal child environment without Mira service credentials."""
        return {key: value for key, value in os.environ.items() if key in SAFE_ENV_KEYS}

    async def _run(self, prompt: str) -> str:
        raise NotImplementedError

    async def _terminate_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Terminate the CLI and any descendants before returning."""
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if proc.returncode is None:
                    with suppress(ProcessLookupError):
                        proc.kill()
        elif proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.kill()
        if proc.returncode is None:
            await proc.wait()

    def _messages_prompt(self, messages: list[dict]) -> str:
        parts = [
            f"You are running as Mira's model backend through {self.backend_label}.",
            f"Follow the Mira review instructions exactly. Do not mention {self.backend_label}.",
            "Return only the requested final answer; no prose wrappers unless explicitly requested.",
            "",
            "## Mira messages",
        ]
        for i, message in enumerate(messages, 1):
            role = message.get("role", "user")
            content = message.get("content") or ""
            if role == "tool":
                call_id = message.get("tool_call_id") or ""
                parts.append(f"\n### Message {i}: tool result for call {call_id}\n{content}")
                continue
            parts.append(f"\n### Message {i}: {role}\n{content}")
            calls = [
                {
                    "id": call.get("id") or "",
                    "name": (call.get("function") or {}).get("name") or "",
                    "arguments": (call.get("function") or {}).get("arguments") or "{}",
                }
                for call in message.get("tool_calls") or []
                if isinstance(call, dict)
            ]
            if calls:
                parts.append("Tool calls made:\n" + json.dumps(calls, indent=2))
        return "\n".join(parts)

    def _agentic_prompt(self, messages: list[dict], tools: list[dict]) -> str:
        specs = [
            {
                "name": (tool.get("function") or {}).get("name"),
                "description": (tool.get("function") or {}).get("description", ""),
                "parameters": (tool.get("function") or {}).get("parameters") or {"type": "object"},
            }
            for tool in tools
        ]
        return (
            self._messages_prompt(messages)
            + "\n\n## Tools\n"
            + "You can call these tools. Mira runs them and sends back the results.\n"
            + json.dumps(specs, indent=2, sort_keys=True)
            + "\n\n## Required output\n"
            + "Return ONLY one JSON object of the form "
            + '{"tool_calls": [{"name": "<tool name>", "arguments": {...}}]}.\n'
            + "Each `arguments` object must conform to that tool's `parameters` schema.\n"
            + "To finish, call the terminal submit tool with your complete result.\n"
            + "Do not wrap the JSON in markdown fences. Do not include explanatory text."
        )

    def _parse_tool_calls(self, raw: str, tools: list[dict]) -> dict:
        """Turn the model's JSON answer into an OpenAI-style assistant message."""
        known = {(tool.get("function") or {}).get("name") for tool in tools}
        payload = json.loads(self._extract_json_object(raw))
        requested = payload.get("tool_calls")
        if requested is None:
            requested = []
        if not isinstance(requested, list):
            raise self._error("invalid_tool_call", reason="`tool_calls` is not a list")
        content = payload.get("content")
        calls: list[dict] = []
        for index, call in enumerate(requested):
            if not isinstance(call, dict):
                raise self._error("invalid_tool_call", reason=f"call {index} is not an object")
            name = call.get("name")
            if name not in known:
                raise self._error("invalid_tool_call", reason=f"unknown tool {name!r}")
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError as exc:
                    raise self._error(
                        "invalid_tool_call", reason=f"arguments for {name} are not JSON"
                    ) from exc
            if not isinstance(arguments, dict):
                raise self._error(
                    "invalid_tool_call", reason=f"arguments for {name} are not an object"
                )
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            )
        return {"content": content if isinstance(content, str) else "", "tool_calls": calls}

    def _tool_prompt(self, messages: list[dict], tools: list[dict]) -> str:
        tool = tools[0].get("function", {}) if tools else {}
        tool_name = tool.get("name") or "submit_result"
        schema = tool.get("parameters") or {"type": "object"}
        return (
            self._messages_prompt(messages)
            + "\n\n## Required output\n"
            + f"Return ONLY a JSON object containing the arguments for `{tool_name}`.\n"
            + "Do not wrap the JSON in markdown fences. Do not include explanatory text.\n"
            + "The JSON object must conform to this schema:\n"
            + json.dumps(schema, indent=2, sort_keys=True)
        )

    def _error(self, suffix: str, **kwargs: object) -> LLMError:
        return LLMError(f"{self.error_prefix}_{suffix}", **kwargs)

    def _extract_json_object(self, text: str) -> str:
        candidate = text.strip()
        if not candidate:
            raise self._error("empty_response")

        def is_object(value: str) -> bool:
            try:
                return isinstance(json.loads(value), dict)
            except Exception:
                return False

        if "```" in candidate:
            blocks = candidate.split("```")
            for block in blocks[1::2]:
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if is_object(block):
                    return block

        try:
            parsed_candidate = json.loads(candidate)
        except Exception:
            parsed_candidate = None
        else:
            if isinstance(parsed_candidate, dict):
                return candidate
            raise self._error("non_object_json", type=type(parsed_candidate).__name__)

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = candidate[start : end + 1]
            try:
                parsed = json.loads(obj)
            except Exception as exc:
                raise self._error(
                    "malformed_json", error=str(exc), excerpt=candidate[:1000]
                ) from exc
            if isinstance(parsed, dict):
                return obj

        raise self._error("no_json_object", excerpt=candidate[:1000])

    def _record_estimated_usage(self, prompt: str, result: str) -> None:
        if self.records_own_usage:
            return
        self.total_prompt_tokens += self.count_tokens(prompt)
        self.total_completion_tokens += self.count_tokens(result)

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = self._messages_prompt(messages)
        if json_mode:
            prompt += (
                "\n\n## Required output\n"
                "Return ONLY one valid JSON object. No markdown fences or explanatory text."
            )
        raw = await self._run(prompt)
        result = self._extract_json_object(raw) if json_mode else raw
        self._record_estimated_usage(prompt, result)
        return result

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        prompt = self._tool_prompt(messages, tools)
        raw = await self._run(prompt)
        result = self._extract_json_object(raw)
        self._record_estimated_usage(prompt, result)
        return result

    async def complete_agentic(
        self,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        if not self.supports_tool_calling:
            # Defer immediately so the caller performs exactly one forced review.
            return {"content": "", "tool_calls": []}
        prompt = self._agentic_prompt(messages, tools)
        raw = await self._run(prompt)
        message = self._parse_tool_calls(raw, tools)
        self._record_estimated_usage(prompt, raw)
        return message

    async def review(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL

        return await self.complete_with_tools(
            messages, tools=[SUBMIT_REVIEW_TOOL], temperature=temperature
        )

    async def walkthrough(self, messages: list[dict[str, str]]) -> str:
        from mira.llm.tool_schemas import SUBMIT_WALKTHROUGH_TOOL

        return await self.complete_with_tools(messages, tools=[SUBMIT_WALKTHROUGH_TOOL])
