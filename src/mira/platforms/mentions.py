"""Bot @-mention matching, shared across platforms.

Mira answers to two handles: the configured ``bot_name`` (what it writes in its
own comments, e.g. ``@mira``) and its real account identity (GitHub App slug /
GitLab bot username, e.g. ``@project_123_bot_abc``). A teammate might type
either — the friendly name, or whatever the platform autocompletes — so command
detection accepts both.
"""

from __future__ import annotations

import re


def mention_names(bot_name: str, bot_identity: str | None) -> list[str]:
    """The handles to match: the configured name plus the real identity."""
    names = [bot_name]
    if bot_identity and bot_identity not in names:
        names.append(bot_identity)
    return names


def _mention_re(name: str) -> str:
    # The handle must end at a non-handle character, so "@mira" doesn't match
    # the front of "@mira-pm" and leave "-pm review" behind.
    return rf"@{re.escape(name)}(?![\w-])"


def _longest_first(names: list[str]) -> list[str]:
    return sorted(names, key=len, reverse=True)


def has_mention(text: str, names: list[str]) -> bool:
    return any(re.search(_mention_re(n), text, re.IGNORECASE) for n in names)


def strip_mentions(text: str, names: list[str]) -> str:
    out = text
    for n in _longest_first(names):
        out = re.sub(rf"{_mention_re(n)}\s*", "", out, flags=re.IGNORECASE)
    return out.strip()


def command_after_mention(text: str, names: list[str]) -> str:
    """The first word following ``@<name>`` for any matched name (lowercased),
    or "" — e.g. "@mira review" → "review"."""
    for n in _longest_first(names):
        m = re.search(rf"{_mention_re(n)}\s+(\w+)", text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
    return ""


def author_is_filtered(login: str, allowed: list[str], blocked: list[str]) -> bool:
    """True if `login` should be skipped for auto-review.

    Precedence (matches GitHub issue #180):
      - empty `login` → never filtered (no sender info → let the
        dispatcher's existing bot-self check do its job);
      - `blocked` wins: matches against the raw login OR the login
        with a trailing `[bot]` stripped (so `dependabot` in `blocked`
        matches `dependabot[bot]` from the webhook);
      - then if `allowed` is non-empty and `login` (raw or stripped)
        is not in it → filtered.
    """
    if not login:
        return False
    variants = {login, login.removesuffix("[bot]")}
    if any(v in blocked for v in variants):
        return True
    return bool(allowed and not any(v in allowed for v in variants))
