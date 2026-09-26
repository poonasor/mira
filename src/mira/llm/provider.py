"""OpenAI-compatible API provider with retry/fallback and tool calling support.

Per-provider quirks (attribution headers, model-prefix policy, reasoning
remapping, and tool-choice constraints) come from the profile registry in
``mira.llm.provider_profiles``, matched to the configured ``base_url``. Other
OpenAI-compatible endpoints work off the portable default, no entry needed.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import httpx

from mira.exceptions import LLMError
from mira.llm.base import (
    OpenAICompatibleProvider,
    _strip_model_prefix,
)
from mira.llm.response_parser import validate_tool_arguments
from mira.llm.utils import _ensure_json_hint

logger = logging.getLogger(__name__)


class LLMProvider(OpenAICompatibleProvider):
    """OpenAI-compatible API client for LLM completions (/chat/completions).

    Inherits protocol-agnostic infrastructure (retry setup, headers, reasoning,
    fallback model logic, public API) from :class:`OpenAICompatibleProvider`.
    """

    supports_json_mode: ClassVar[bool] = True
    supports_tool_calling: ClassVar[bool] = True

    def _chat_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    async def _call_llm(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Make a single LLM call with retries against the /chat/completions endpoint.

        ``json_mode`` adds ``response_format`` AND guarantees the JSON-only
        instruction lives in a user-role message: AxonHub moves system content
        into Responses ``instructions`` and may route a Chat request to a
        Responses-only upstream, which rejects ``json_object`` without an
        explicit user-turn instruction.
        """
        body: dict = {
            "model": _strip_model_prefix(model, self.config.base_url),
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
            # Copied container — caller-owned ``messages`` stay untouched.
            body["messages"] = _ensure_json_hint(messages)
        self._apply_reasoning(body)

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            resp = await client.post(
                self._chat_url(),
                headers=self._build_headers(),
                json=body,
            )
            self._handle_error(resp)
            data = resp.json()

        self._account_usage(data)
        return data["choices"][0]["message"].get("content") or ""

    async def _call_llm_with_tools(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Make an LLM call with tool/function calling and retries.

        The LLM returns structured data by 'calling' a tool. We extract the
        tool arguments as the JSON response.
        """
        api_model = _strip_model_prefix(model, self.config.base_url)
        if not tools:
            raise LLMError("no_tools")
        forced_choice: dict | str = {
            "type": "function",
            "function": {"name": tools[0]["function"]["name"]},
        }
        body: dict = {
            "model": api_model,
            "messages": messages,
            "tools": tools,
            # Force the one tool for structured args when supported. Profiles
            # can select "auto" up front; unknown rejections fall back below.
            "tool_choice": self._tool_choice(api_model, forced_choice),
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        self._apply_reasoning(body)

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            resp = await client.post(
                self._chat_url(),
                headers=self._build_headers(),
                json=body,
            )
            if (
                resp.status_code == 400
                and body["tool_choice"] != "auto"
                and "tool_choice" in resp.text.lower()
            ):
                # Forced choice unsupported — remember it and let the model pick.
                logger.info("Model %s rejected forced tool_choice; retrying with auto", api_model)
                self._no_forced_tool_choice.add(api_model)
                body["tool_choice"] = "auto"
                resp = await client.post(self._chat_url(), headers=self._build_headers(), json=body)
            if (
                resp.status_code == 400
                and self._reasoning_is_enabled(body)
                and any(term in resp.text.lower() for term in ("reasoning", "thinking"))
            ):
                # Reasoning effort unsupported on this model/endpoint — drop it
                # and review without thinking instead of failing the review.
                logger.info("Model %s rejected reasoning effort; retrying without it", api_model)
                self._no_reasoning.add(api_model)
                self._disable_reasoning(body)
                body["temperature"] = (
                    temperature if temperature is not None else self.config.temperature
                )
                resp = await client.post(self._chat_url(), headers=self._build_headers(), json=body)
            self._handle_error(resp)
            data = resp.json()

        self._account_usage(data)

        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        if tool_calls and len(tool_calls) > 0:
            arguments = tool_calls[0]["function"]["arguments"]
            # Some backends intermittently corrupt the arguments (mid-string
            # quote/brace permutations). Validate before returning so the
            # corruption is a request-scoped LLMError the tiered failover
            # layer can act on, instead of a parse error the caller can only
            # re-roll on the same model.
            return validate_tool_arguments(
                arguments, provider=self.config.provider, model=api_model
            )

        # Fallback: if the model returned content instead of a tool call,
        # return the content as-is (some models may not support tool calling)
        content = message.get("content") or ""
        if content:
            logger.warning("Model returned content instead of tool call, using content as fallback")
            return validate_tool_arguments(content, provider=self.config.provider, model=api_model)

        raise LLMError("no_tool_call")

    async def _call_llm_agentic(
        self,
        model: str,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        """Make a tool-using LLM call without forcing a specific tool.

        Returns the full assistant message (with ``tool_calls`` and ``content``)
        so the caller can dispatch the calls and continue the conversation.
        """
        if not tools:
            raise LLMError("no_tools")
        api_model = _strip_model_prefix(model, self.config.base_url)
        body: dict = {
            "model": api_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        self._apply_reasoning(body)

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            resp = await client.post(
                self._chat_url(),
                headers=self._build_headers(),
                json=body,
            )
            if (
                resp.status_code == 400
                and self._reasoning_is_enabled(body)
                and any(term in resp.text.lower() for term in ("reasoning", "thinking"))
            ):
                logger.info("Model %s rejected reasoning effort; retrying without it", api_model)
                self._no_reasoning.add(api_model)
                self._disable_reasoning(body)
                body["temperature"] = (
                    temperature if temperature is not None else self.config.temperature
                )
                resp = await client.post(self._chat_url(), headers=self._build_headers(), json=body)
            self._handle_error(resp)
            data = resp.json()

        self._account_usage(data)
        return data["choices"][0]["message"]
