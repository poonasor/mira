"""Platform-neutral webhook handlers — shared by the GitHub and GitLab
webhook layers. Each takes a provider/auth and operates through the engine;
none is tied to a specific platform's payload shape."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from mira.config import load_config
from mira.core.engine import ReviewEngine
from mira.core.review_status import tracker as review_tracker
from mira.dashboard.models_config import llm_config_for
from mira.index.store import IndexStore
from mira.llm import create_llm
from mira.llm.prompts.review import build_conversation_prompt
from mira.llm.tool_schemas import SUBMIT_THREAD_REPLY_TOOL
from mira.llm.utils import strip_code_fences, strip_think_blocks
from mira.platforms.command_policy import is_command_allowed
from mira.platforms.mentions import strip_mentions

logger = logging.getLogger(__name__)

_REVIEW_KEYWORDS = {"review", "review this", "review this pr"}

_REJECT_KEYWORDS = {"reject", "dismiss", "resolve", "ignore"}

_REVIEW_REST_KEYWORDS = {"review-rest", "review rest", "rest", "continue"}

_HELP_KEYWORDS = {"help", "?", "commands"}

_THREAD_REPLY_ENV = Environment(
    loader=FileSystemLoader(
        str(Path(__file__).resolve().parents[1] / "llm" / "prompts" / "templates")
    ),
    trim_blocks=True,
    lstrip_blocks=True,
)

_THREAD_REPLY_TEMPLATE = _THREAD_REPLY_ENV.get_template("thread_reply.jinja2")

PAUSE_LABEL = "mira-paused"

_PAUSE_KEYWORDS = {"pause"}

_RESUME_KEYWORDS = {"resume"}


def _in_progress_ttl_seconds(config: Any) -> float:
    """How long an in-progress marker may block a duplicate trigger."""
    return float(config.review.in_progress_ttl_minutes) * 60


def classify_command(text: str, names: list[str], *, inline: bool = False) -> str:
    """Map a comment body to the command key used by command authorization."""
    normalized = strip_mentions(text, names).lower().strip()
    first_word = normalized.split()[0] if normalized.split() else ""
    if inline:
        return "reject" if first_word in _REJECT_KEYWORDS else "ask"
    if normalized in _REVIEW_KEYWORDS:
        return "review"
    if normalized in _REVIEW_REST_KEYWORDS:
        return "review_rest"
    if normalized in _HELP_KEYWORDS:
        return "help"
    if first_word in _PAUSE_KEYWORDS:
        return "pause"
    if first_word in _RESUME_KEYWORDS:
        return "resume"
    return "ask"


def _open_store(owner: str, repo: str, platform: str = "github") -> IndexStore:
    """Open an IndexStore for the given owner/repo."""
    return IndexStore.open(owner, repo, platform=platform)


def _help_message(bot_name: str) -> str:
    """Markdown help comment listing every command Mira understands."""
    return (
        f"### Mira commands\n\n"
        f"Mention `@{bot_name}` in a PR comment followed by one of these verbs:\n\n"
        f"| Command | What it does |\n"
        f"|---|---|\n"
        f"| `@{bot_name} review` | Re-run the full review on this PR. Useful after force-pushes or when you want a fresh pass. |\n"
        f"| `@{bot_name} review-rest` | Review files that were skipped on the first pass because the PR was too large. Aliases: `rest`, `continue`. |\n"
        f"| `@{bot_name} pause` | Pause Mira on this PR. No more reviews until you resume. Adds a `mira-paused` label. |\n"
        f"| `@{bot_name} resume` | Resume Mira on a paused PR and re-review the latest diff. |\n"
        f"| `@{bot_name} help` | Show this message. Aliases: `?`, `commands`. |\n"
        f"| `@{bot_name} <anything else>` | Ask a free-form question about the PR. Mira will reply inline using the PR diff as context. |\n\n"
        f"On an inline review comment Mira posted, reply with `@{bot_name} reject` "
        f"(aliases: `dismiss`, `resolve`, `ignore`) to mark the thread resolved and "
        f"teach Mira not to make similar suggestions in the future.\n\n"
        f"To skip a PR entirely, include `@{bot_name} ignore` in the PR body.\n\n"
        f"Full docs: https://docs.miracode.ai/commands"
    )


def review_completed_payload(repo_full: str, pr_url: str, number: int, result: Any) -> dict:
    """The ``review.completed`` webhook payload for a finished review."""
    from mira.models import build_review_stats

    stats = build_review_stats(result.comments)
    failed = list(getattr(result, "failed_chunks", None) or [])
    score = getattr(result, "confidence_score", None)
    return {
        "repo": repo_full,
        "pr_url": pr_url,
        "number": number,
        "comments": len(result.comments),
        "key_issues": len(result.key_issues),
        "severities": {sev.name.lower(): n for sev, n in stats.items()},
        # Chunks that couldn't be reviewed (model output never parsed), so a
        # 0-comment review isn't mistaken for a clean one.
        "failed_chunks": len(failed),
        "failed_chunk_list": failed,
        # Head commit the review ran against (the PR head fetched at review
        # start); "" when unknown.
        "head_sha": getattr(result, "head_sha", "") or "",
        # Final 1-5 confidence score for that head; None when not scored.
        "confidence_score": score if isinstance(score, int) else None,
        # True when part of the diff was not reviewed (chunks whose output never
        # parsed, or files the host returned no patch for), so the score and
        # findings don't cover the whole PR.
        "partial": bool(failed) or bool(getattr(result, "unreviewed_paths", None)),
        "unreviewed_paths": list(getattr(result, "unreviewed_paths", None) or []),
    }


async def emit_review_events(repo_full: str, pr_url: str, number: int, result: Any) -> None:
    """Send ``review.completed``, plus ``review.high_severity`` when any finding
    is WARNING or above, for a finished review.

    Every path that runs a review (new PR, push, ``review`` and ``review-rest``
    commands) goes through here so the outbound events can't drift apart.
    """
    from mira.models import Severity, build_review_stats
    from mira.outbound_webhooks import REVIEW_COMPLETED, REVIEW_HIGH_SEVERITY, dispatch_event

    event_data = review_completed_payload(repo_full, pr_url, number, result)
    await dispatch_event(REVIEW_COMPLETED, event_data)
    if any(sev >= Severity.WARNING for sev in build_review_stats(result.comments)):
        await dispatch_event(REVIEW_HIGH_SEVERITY, event_data)


# Upper bound on the review.failed send, so a hung webhook target can't hold
# up re-raising a cancellation during shutdown.
_FAILED_EVENT_TIMEOUT_SECONDS = 10.0

_FAILURE_REPORTED_ATTR = "_mira_review_failure_reported"


def review_failed_payload(
    repo_full: str, pr_url: str, number: int, head_sha: str, reason: str
) -> dict:
    """The ``review.failed`` webhook payload."""
    return {
        "repo": repo_full,
        "pr_url": pr_url,
        "number": number,
        "pr": number,
        "head_sha": head_sha or "",
        "reason": reason,
        # Older consumers (and the default message renderer) read ``error``.
        "error": reason,
    }


def failure_reported(exc: BaseException) -> bool:
    """Whether ``review.failed`` was already sent for this exception."""
    return bool(getattr(exc, _FAILURE_REPORTED_ATTR, False))


async def emit_review_failed(
    repo_full: str, pr_url: str, number: int, head_sha: str, reason: str
) -> None:
    """Send ``review.failed``. Best-effort: never raises."""
    from mira.outbound_webhooks import REVIEW_FAILED, dispatch_event

    payload = review_failed_payload(repo_full, pr_url, number, head_sha, reason)
    try:
        await asyncio.wait_for(
            dispatch_event(REVIEW_FAILED, payload), timeout=_FAILED_EVENT_TIMEOUT_SECONDS
        )
    except (Exception, asyncio.CancelledError) as exc:
        logger.warning("Could not send review.failed for %s: %r", pr_url, exc)


def _failure_reason(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled before the review finished (shutdown, restart or task cancellation)"
    detail = str(exc)
    name = type(exc).__name__
    return f"{name}: {detail}" if detail else name


@asynccontextmanager
async def report_review_failure(
    repo_full: str, pr_url: str, number: int, head_sha: str = "", trigger: str = "review"
) -> AsyncIterator[None]:
    """Log and report any way a background review can end without a result.

    Background review tasks run after the webhook already answered 200, so an
    exception or cancellation here is otherwise invisible. Every exit other
    than a clean one is logged at ERROR with its traceback and sent as
    ``review.failed``, then re-raised. ``except Exception`` alone would miss
    ``asyncio.CancelledError`` (and ``SystemExit``), which are BaseExceptions.
    """
    try:
        yield
    except BaseException as exc:
        if failure_reported(exc):
            raise
        reason = _failure_reason(exc)
        if isinstance(exc, Exception):
            logger.exception("Review (%s) failed for %s: %s", trigger, pr_url, reason)
        else:
            logger.error(
                "Review (%s) for %s ended by %s: %s",
                trigger,
                pr_url,
                type(exc).__name__,
                reason,
                exc_info=True,
            )
        await emit_review_failed(repo_full, pr_url, number, head_sha, reason)
        with suppress(Exception):
            setattr(exc, _FAILURE_REPORTED_ATTR, True)
        raise


def mark_first_review(platform: str, owner: str, repo: str, number: int, result: Any) -> None:
    """Stamp ``pull_requests.first_review_at`` once Mira has reviewed the PR.

    The human-review webhook skips bot reviews, so without this a PR only
    Mira reviewed stays "awaiting review" forever. Only GitHub fills the
    pull_requests table (it has no platform column), so other platforms are
    left alone. Keeps the earliest timestamp; never raises.
    """
    if platform != "github" or getattr(result, "skipped_reason", None):
        return
    try:
        from mira.dashboard.api import _app_db

        _app_db.set_pr_first_review(owner, repo, number, time.time())
    except Exception as exc:
        logger.warning(
            "Could not record first review time for %s/%s#%s: %s", owner, repo, number, exc
        )


async def run_pr_review(
    provider: Any,
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    is_private: bool,
    bot_name: str,
    platform: str = "github",
    pr_title: str = "",
    head_sha: str = "",
) -> None:
    """Platform-neutral review core: review a PR/MR and post the result.

    Shared by the GitHub and GitLab webhook handlers — everything here goes
    through the ``provider`` abstraction and the engine, so it's the same for
    every platform.
    """
    repo_full = f"{owner}/{repo}"
    async with report_review_failure(repo_full, pr_url, number, head_sha, "pull request"):
        await _run_pr_review(
            provider,
            owner,
            repo,
            number,
            pr_url,
            is_private,
            bot_name,
            platform=platform,
            pr_title=pr_title,
            head_sha=head_sha,
        )


async def _run_pr_review(
    provider: Any,
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    is_private: bool,
    bot_name: str,
    platform: str,
    pr_title: str,
    head_sha: str,
) -> None:
    repo_full = f"{owner}/{repo}"
    config = load_config()

    # Atomically claim the slot — avoids stacking redundant runs when two
    # concurrent webhooks arrive. The claim yields None if a live review
    # already covers this head; it is always finished when the block exits,
    # however the review ends (error, cancellation, hang past the TTL).
    with review_tracker.claim(
        repo_full,
        number,
        pr_title,
        pr_url,
        head_sha=head_sha,
        ttl_seconds=_in_progress_ttl_seconds(config),
    ) as run_id:
        if run_id is None:
            logger.info("Review already in progress for %s, skipping", pr_url)
            return

        llm = create_llm(llm_config_for("review", config.llm))
        indexing_llm = create_llm(llm_config_for("indexing", config.llm))
        security_llm = create_llm(llm_config_for("security", config.llm))
        engine = ReviewEngine(
            config=config,
            llm=llm,
            provider=provider,
            bot_name=bot_name,
            indexing_llm=indexing_llm,
            security_llm=security_llm,
        )

        from mira.dashboard.api import _app_db

        # Keep visibility current — the blast-radius filter relies on it to avoid
        # naming private repos in a public repo's review.
        try:
            _app_db.set_repo_visibility(owner, repo, is_private, platform=platform)
        except sqlite3.OperationalError as exc:
            logger.debug("set_repo_visibility failed (ignored): %s", exc)

        repo_record = _app_db.get_repo(owner, repo, platform=platform)
        is_indexed = bool(repo_record and repo_record.status == "ready")

        logger.info("Reviewing %s (indexed=%s)", pr_url, is_indexed)
        result = await engine.review_pr(pr_url)

    # The walkthrough comment already carries the "more accurate after indexing"
    # nudge for unindexed repos, so we don't post a separate note here — that
    # would repeat on every push.

    logger.info("Review complete for %s", pr_url)

    mark_first_review(platform, owner, repo, number, result)
    await emit_review_events(repo_full, pr_url, number, result)


async def run_pr_command(
    provider: Any,
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    question: str,
    actor: str,
    bot_name: str,
    platform: str = "github",
    pr_title: str = "",
    user_id: int | str | None = None,
) -> None:
    """Platform-neutral handler for an @-mention command on a PR/MR.

    Dispatches help / review / review-rest / free-form Q&A through the provider
    and engine. Shared by the GitHub and GitLab comment handlers.
    """
    repo_full = f"{owner}/{repo}"
    config = load_config()

    command = classify_command(question, [])
    if not await is_command_allowed(
        config,
        command,
        actor,
        provider=provider,
        owner=owner,
        repo=repo,
        user_id=user_id,
    ):
        return

    # Pause/resume/reject have dedicated handlers; never fall through to a
    # paid free-form answer if one is misrouted here.
    if command not in {"review", "review_rest", "help", "ask"}:
        logger.warning("Command %r cannot run through run_pr_command", command)
        return

    from mira.dashboard.models_config import llm_config_for

    llm = create_llm(llm_config_for("review", config.llm))
    indexing_llm = create_llm(llm_config_for("indexing", config.llm))
    security_llm = create_llm(llm_config_for("security", config.llm))

    is_review = command == "review"
    is_review_rest = command == "review_rest"
    is_help = command == "help"

    if is_help:
        pr_info_for_help = await provider.get_pr_info(pr_url)
        await provider.post_comment(pr_info_for_help, _help_message(bot_name))
        logger.info("Help requested on %s by @%s", pr_url, actor)
        return

    if is_review or is_review_rest:
        trigger = "review-rest" if is_review_rest else "review command"
        async with report_review_failure(repo_full, pr_url, number, trigger=trigger):
            result = await _run_command_review(
                provider,
                owner,
                repo,
                number,
                pr_url,
                actor,
                pr_title,
                platform,
                config,
                ReviewEngine(
                    config=config,
                    llm=llm,
                    provider=provider,
                    bot_name=bot_name,
                    indexing_llm=indexing_llm,
                    security_llm=security_llm,
                ),
                rest=is_review_rest,
            )
        if result is not None:
            mark_first_review(platform, owner, repo, number, result)
            await emit_review_events(repo_full, pr_url, number, result)
    else:
        pr_info = await provider.get_pr_info(pr_url)
        diff_text = await provider.get_pr_diff(pr_info)
        messages = build_conversation_prompt(
            question=question,
            diff_text=diff_text,
            pr_title=pr_info.title,
            pr_description=pr_info.description,
        )
        response = await llm.complete(messages, json_mode=False)
        await provider.post_comment(pr_info, f"> {actor} asked: {question}\n\n{response}")
        logger.info("Replied to comment on %s", pr_url)


async def _run_command_review(
    provider: Any,
    owner: str,
    repo: str,
    number: int,
    pr_url: str,
    actor: str,
    pr_title: str,
    platform: str,
    config: Any,
    engine: Any,
    *,
    rest: bool,
) -> Any:
    """Run a ``review`` / ``review-rest`` command; None when nothing ran."""
    repo_full = f"{owner}/{repo}"
    if rest:
        from mira.dashboard.api import _app_db

        progress = _app_db.get_pr_review_progress(owner, repo, number, platform=platform)
        if not progress or not progress.skipped_paths:
            pr_info_for_reply = await provider.get_pr_info(pr_url)
            await provider.post_comment(
                pr_info_for_reply,
                f"> @{actor}: nothing left to review — every file in this "
                "PR has already been covered. 🎉",
            )
            return None
        engine._review_only_paths = set(progress.skipped_paths)
    with review_tracker.claim(
        repo_full, number, pr_title, pr_url, ttl_seconds=_in_progress_ttl_seconds(config)
    ) as run_id:
        if run_id is None:
            logger.info("Review already in progress for %s, skipping", pr_url)
            return None
        if rest:
            logger.info(
                "review-rest on %s by @%s — %d file(s)",
                pr_url,
                actor,
                len(engine._review_only_paths),
            )
        else:
            logger.info("Re-review triggered for %s by @%s", pr_url, actor)
        return await engine.review_pr(pr_url)


async def run_thread_reply(
    provider: Any,
    pr_info: Any,
    human_reply: str,
    comment_id: int,
    *,
    original_suggestion: str = "",
    thread_id: str | None = None,
    comment_node_id: str | None = None,
    comment_path: str = "",
    comment_line: int = 0,
    actor: str = "",
    bot_name: str = "miracodeai",
    platform: str = "github",
    user_id: int | str | None = None,
) -> None:
    """Platform-neutral free-form thread reply with intent classification.

    The LLM classifies the human's message and we respond accordingly:
    ``disagreement`` → reply + resolve the thread + record a ``rejected``
    feedback signal (same learning signal as an explicit reject); ``question``
    → answer, leave open; ``agreement`` / ``other`` → acknowledge, leave open.
    """
    config = load_config()
    if not await is_command_allowed(
        config,
        "ask",
        actor,
        provider=provider,
        owner=pr_info.owner,
        repo=pr_info.repo,
        user_id=user_id,
    ):
        return
    llm = create_llm(llm_config_for("indexing", config.llm))
    prompt = _THREAD_REPLY_TEMPLATE.render(
        user_reply=human_reply or "(empty)",
        original_suggestion=original_suggestion,
    )
    # Tool calling forces a schema-valid result — more reliable than parsing
    # free-form JSON. The provider's tenacity decorator retries transient fails.
    try:
        raw = await llm.complete_with_tools(
            messages=[{"role": "user", "content": prompt}],
            tools=[SUBMIT_THREAD_REPLY_TOOL],
            temperature=0.0,
        )
        data = json.loads(strip_think_blocks(strip_code_fences(raw))) if raw else {}
    except Exception as exc:
        logger.warning("Free-form thread reply LLM call failed: %s", exc)
        return

    intent = str(data.get("intent", "other")).lower()
    reply_text = str(data.get("reply", "")).strip()
    if not reply_text:
        logger.warning("Free-form thread reply: empty reply (intent=%s). Skipping.", intent)
        return

    try:
        await provider.reply_to_review_comment(pr_info, comment_id, reply_text)
    except Exception as exc:
        logger.warning("Failed to post thread reply: %s", exc)
        return

    # A free-form disagreement is an implicit ``reject``: resolving the thread
    # and recording feedback must honour the reject policy, not just ``ask``.
    if intent == "disagreement" and await is_command_allowed(
        config,
        "reject",
        actor,
        provider=provider,
        owner=pr_info.owner,
        repo=pr_info.repo,
        user_id=user_id,
    ):
        try:
            tid = thread_id
            if tid is None and comment_node_id:
                tid = await provider.get_thread_id_for_comment(comment_node_id, pr_info)
            if tid:
                await provider.resolve_threads(pr_info, [tid])
        except Exception as exc:
            logger.warning("Failed to resolve disagreement thread: %s", exc)
        try:
            store = _open_store(pr_info.owner, pr_info.repo, platform)
            try:
                store.record_feedback(
                    pr_number=pr_info.number,
                    pr_url=pr_info.url,
                    comment_path=comment_path,
                    comment_line=comment_line,
                    comment_category="",
                    comment_severity="",
                    comment_title="",
                    signal="rejected",
                    actor=actor,
                )
            finally:
                store.close()
        except Exception as fb_err:
            logger.debug("Failed to record disagreement feedback: %s", fb_err)

    logger.info("Thread reply (%s) on %s: %s", intent, pr_info.url, reply_text[:80])


async def run_pr_merged_learning(
    provider: Any,
    pr_info: Any,
    bot_name: str,
    merged_by: str,
    platform: str = "github",
) -> None:
    """Platform-neutral merge-time learning: record accept/reject + human-review
    signals and synthesize rules. Shared by GitHub and GitLab."""
    from mira.providers.formatting import parse_bot_comment_metadata

    owner, repo, number, pr_url = pr_info.owner, pr_info.repo, pr_info.number, pr_info.url
    store = _open_store(owner, repo, platform)
    accepted = 0
    human_recorded = 0
    deterministic_rules = 0
    llm_rules = 0
    try:
        existing = store.list_feedback(limit=2000)
        if any(
            e.signal in ("accepted", "human_review") and e.pr_number == number for e in existing
        ):
            logger.info("PR %s already processed for merge-time learning", pr_url)
            return
        rejected_locations = {
            (e.comment_path, e.comment_line)
            for e in existing
            if e.signal == "rejected" and e.pr_number == number
        }

        try:
            bot_threads = await provider.get_all_bot_threads(pr_info)
        except Exception as exc:
            logger.warning("Failed to fetch bot threads for %s: %s", pr_url, exc)
            bot_threads = []

        bot_events: list[dict] = []
        for thread in bot_threads:
            if (thread.path, thread.line) in rejected_locations:
                continue
            meta = parse_bot_comment_metadata(thread.body)
            if not meta["category"]:
                continue
            bot_events.append(
                {
                    "pr_number": number,
                    "pr_url": pr_url,
                    "comment_path": thread.path,
                    "comment_line": thread.line,
                    "comment_category": meta["category"],
                    "comment_severity": meta["severity"],
                    "comment_title": meta["title"],
                    "signal": "accepted",
                    "actor": merged_by,
                    "pr_author": pr_info.author,
                }
            )

        try:
            human_comments = await provider.get_human_review_comments(pr_info, bot_name)
        except Exception as exc:
            logger.warning("Failed to fetch human review comments for %s: %s", pr_url, exc)
            human_comments = []

        human_events: list[dict] = []
        for hc in human_comments:
            body = (hc.body or "").strip()
            if not body:
                continue
            human_events.append(
                {
                    "pr_number": number,
                    "pr_url": pr_url,
                    "comment_path": hc.path,
                    "comment_line": hc.line,
                    "comment_category": "human_review",
                    "comment_severity": "",
                    "comment_title": body[:2000],
                    "signal": "human_review",
                    "actor": hc.author,
                    "pr_author": pr_info.author,
                }
            )

        if bot_events:
            store.record_bulk_feedback(bot_events)
            accepted = len(bot_events)
        if human_events:
            store.record_bulk_feedback(human_events)
            human_recorded = len(human_events)

        from mira.analysis.feedback import synthesize_from_human_reviews, synthesize_rules

        deterministic_rules = synthesize_rules(store)

        if human_recorded > 0:
            try:
                config = load_config()
                from mira.dashboard.models_config import llm_config_for

                indexing_llm = create_llm(llm_config_for("indexing", config.llm))
                llm_rules = await synthesize_from_human_reviews(store, indexing_llm)
            except Exception as exc:
                logger.warning("LLM rule synthesis failed for %s: %s", pr_url, exc)
    finally:
        store.close()

    logger.info(
        "PR merged %s: recorded %d accepted + %d human review events; "
        "upserted %d deterministic + %d LLM rules",
        pr_url,
        accepted,
        human_recorded,
        deterministic_rules,
        llm_rules,
    )
