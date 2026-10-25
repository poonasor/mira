"""Rough prompt-size estimate for one 6-hop agentic chunk over the Claude CLI.

Counts characters Mira writes to the CLI's stdin and divides by 4. This is
not an API token count: Claude Code adds its own system prompt to every call,
and on a resumed session the API replays earlier turns from its prompt cache.
"""

from __future__ import annotations

import json

import pytest

from mira.config import LLMConfig
from mira.core.passes import agentic_review_loop
from mira.llm.agentic_tools import AgenticToolExecutor
from mira.llm.claude_cli import ClaudeCLIProvider

SYSTEM = "You are Mira, a code reviewer. " * 300  # ~9k chars of rules
DIFF = "+    value = compute(thing)  # changed line\n" * 450  # ~20k-char chunk


class _Fetcher:
    async def fetch(self, path: str) -> str | None:
        return "\n".join(f"def f{n}(x):  return x + {n}" for n in range(400))  # ~11k chars


def _replies() -> list[str]:
    hops = [
        {
            "tool_calls": [
                {"name": "read_file", "arguments": {"path": f"src/m{2 * i}.py"}},
                {"name": "grep_repo", "arguments": {"pattern": f"f{i}\\("}},
            ]
        }
        for i in range(5)
    ]
    hops.append({"tool_calls": [{"name": "submit_review", "arguments": {"comments": []}}]})
    return [json.dumps(h) for h in hops]


async def _prompt_chars(resume: bool) -> list[int]:
    llm = ClaudeCLIProvider(LLMConfig(provider="claude-cli", model="claude-sonnet-5-5"))
    llm.supports_session_resume = resume  # type: ignore[misc]
    replies = _replies()
    sizes: list[int] = []

    async def fake_run(prompt: str, session=None, resume: bool = False) -> str:
        sizes.append(len(prompt))
        return replies.pop(0)

    llm._run = fake_run  # type: ignore[method-assign]
    tree = [f"src/m{n}.py" for n in range(10)]
    executor = AgenticToolExecutor(source_fetcher=_Fetcher(), repo_tree=tree)
    await agentic_review_loop(
        llm,
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": DIFF}],
        executor,
    )
    assert len(sizes) == 6
    return sizes


@pytest.mark.asyncio
async def test_session_resume_cuts_prompt_chars_for_a_six_hop_chunk():
    stateless = await _prompt_chars(resume=False)
    resumed = await _prompt_chars(resume=True)
    before, after = sum(stateless) // 4, sum(resumed) // 4
    print(f"\n6-hop chunk, est. prompt tokens (chars/4): stateless={before} resumed={after}")
    # First hop is the same full prompt either way; later hops shrink to just
    # the new tool results instead of re-sending system + diff + all results.
    assert resumed[0] == stateless[0]
    assert all(r < s / 3 for r, s in zip(resumed[1:], stateless[1:], strict=True))
    assert after < before * 0.35
