"""Shared plumbing for LLM providers that run a local CLI as a subprocess.

Callers pass Mira's existing chat messages and tool schemas and receive the
same JSON strings an HTTP provider would return. Subclasses only supply
``_run(prompt)``, which executes the CLI and returns its final answer text.

Agentic tool use is emulated over plain text: the model answers with a JSON
object naming the tool calls it wants, and Mira executes them itself. The CLI
never gets tools of its own, so the sandboxing above is unchanged.

Providers that can resume a CLI session (``supports_session_resume``) send the
system prompt, diff and tool schemas once per agentic loop; later hops send
only the new tool results. Others re-render the whole conversation per hop.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import uuid
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field

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


# Open agentic sessions kept per provider before the oldest is closed; loops
# normally close their own via close_agentic.
_MAX_AGENTIC_THREADS = 32

_OUTPUT_CONTRACT = (
    "Return ONLY one JSON object of the form "
    '{"tool_calls": [{"name": "<tool name>", "arguments": {...}}]}. '
    "Each `arguments` object must match that tool's `parameters` schema. "
    "Put independent calls in one reply. To finish, call the terminal submit "
    "tool with your complete result. No markdown fences or explanatory text."
)


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


@dataclass
class _AgenticThread:
    """A caller's agentic conversation mapped onto one resumable CLI session."""

    convo: list
    session: object
    tool_names: tuple[str, ...]
    # How many of ``convo``'s messages the CLI session already holds.
    sent: int = 0
    started: bool = False
    # Calls from the session's last reply: id -> (name, arguments JSON).
    last_calls: dict[str, tuple[str, str]] = field(default_factory=dict)


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

    # Whether _run can continue one CLI session across agentic hops
    # (_new_session / _close_session / _run(..., session=, resume=)).
    supports_session_resume: bool = False

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._threads: OrderedDict[int, _AgenticThread] = OrderedDict()

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

    def _new_session(self) -> object:
        raise NotImplementedError

    def _close_session(self, session: object) -> None:
        return None

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
                parts.append("Tool calls made:\n" + _compact_json(calls))
        return "\n".join(parts)

    @staticmethod
    def _tool_specs(tools: list[dict]) -> str:
        specs = [
            {
                "name": (tool.get("function") or {}).get("name"),
                "description": (tool.get("function") or {}).get("description", ""),
                "parameters": (tool.get("function") or {}).get("parameters") or {"type": "object"},
            }
            for tool in tools
        ]
        return _compact_json(specs)

    def _agentic_prompt(self, messages: list[dict], tools: list[dict]) -> str:
        return (
            self._messages_prompt(messages)
            + "\n\n## Tools\n"
            + "You can call these tools. Mira runs them and sends back the results.\n"
            + self._tool_specs(tools)
            + "\n\n## Required output\n"
            + _OUTPUT_CONTRACT
        )

    def _followup_prompt(
        self, thread: _AgenticThread, new_messages: list[dict], tools: list[dict]
    ) -> str:
        """Only what the resumed session hasn't seen: the latest tool results."""
        parts = ["## Tool results"]
        for message in new_messages:
            content = message.get("content") or ""
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id") or ""
                name, arguments = thread.last_calls.get(call_id, ("tool", ""))
                parts.append(f"\n### {name} {arguments}\n{content}")
            else:
                parts.append(f"\n### Mira\n{content}")
        names = tuple(sorted((t.get("function") or {}).get("name") or "" for t in tools))
        if names != thread.tool_names:
            parts.append("\n## Tools now available\n" + self._tool_specs(tools))
        parts.append("\n## Required output\n" + _OUTPUT_CONTRACT)
        return "\n".join(parts)

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
        thread = self._threads.get(id(messages))
        new_messages = self._unsent_messages(thread, messages) if thread else None
        if thread is not None and new_messages is not None:
            prompt = self._followup_prompt(thread, new_messages, tools)
        else:
            self._drop_thread(messages)
            thread = self._open_thread(messages, tools)
            prompt = self._agentic_prompt(messages, tools)
        try:
            if thread is None:
                raw = await self._run(prompt)
            else:
                raw = await self._run(prompt, session=thread.session, resume=thread.started)  # type: ignore[call-arg]
            message = self._parse_tool_calls(raw, tools)
        except BaseException:
            # The session may hold a half-finished turn; the next hop (or a
            # retry) starts a fresh one with the full conversation.
            self._drop_thread(messages)
            raise
        if thread is not None:
            thread.started = True
            thread.sent = len(messages) + 1  # plus the reply the caller appends
            thread.tool_names = tuple(
                sorted((t.get("function") or {}).get("name") or "" for t in tools)
            )
            thread.last_calls = {
                call["id"]: (call["function"]["name"], call["function"]["arguments"])
                for call in message["tool_calls"]
            }
        self._record_estimated_usage(prompt, raw)
        return message

    def _open_thread(self, messages: list, tools: list[dict]) -> _AgenticThread | None:
        if not self.supports_session_resume:
            return None
        thread = _AgenticThread(
            convo=messages,
            session=self._new_session(),
            tool_names=tuple(sorted((t.get("function") or {}).get("name") or "" for t in tools)),
        )
        self._threads[id(messages)] = thread
        while len(self._threads) > _MAX_AGENTIC_THREADS:
            _, oldest = self._threads.popitem(last=False)
            self._close_session(oldest.session)
        return thread

    def _drop_thread(self, messages: list) -> None:
        thread = self._threads.pop(id(messages), None)
        if thread is not None:
            self._close_session(thread.session)

    @staticmethod
    def _unsent_messages(thread: _AgenticThread, messages: list) -> list[dict] | None:
        """Messages the session hasn't seen, or None if it can't continue.

        Continuing is only safe when the caller appended our own last reply
        unchanged and then only tool results (another tier answering a hop,
        or an edited history, starts a fresh session instead).
        """
        if thread.convo is not messages or not thread.started or len(messages) <= thread.sent:
            return None
        reply = messages[thread.sent - 1]
        ids = [call.get("id") for call in reply.get("tool_calls") or [] if isinstance(call, dict)]
        if reply.get("role") != "assistant" or set(ids) != set(thread.last_calls):
            return None
        new = messages[thread.sent :]
        if any(message.get("role") not in ("tool", "user") for message in new):
            return None
        return new

    async def close_agentic(self, messages: list) -> None:
        """End the agentic loop over ``messages`` and free its CLI session."""
        self._drop_thread(messages)

    async def review(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL

        return await self.complete_with_tools(
            messages, tools=[SUBMIT_REVIEW_TOOL], temperature=temperature
        )

    async def walkthrough(self, messages: list[dict[str, str]]) -> str:
        from mira.llm.tool_schemas import SUBMIT_WALKTHROUGH_TOOL

        return await self.complete_with_tools(messages, tools=[SUBMIT_WALKTHROUGH_TOOL])
