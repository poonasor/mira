"""Tools the reviewer LLM can call during review (`read_file`, `grep_repo`).

On unindexed repos the tools cover what JIT pre-fetch can't reach
(Java/Go import resolution); on indexed repos they let the model trace
callers and dispatch points beyond the pre-fetched index context.

This module owns the tool *schemas* and a per-review *executor* that
dispatches calls, caches results, and hard-caps total output. The agentic
loop itself lives in ``passes.py:agentic_review_loop``.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field

from mira.index.context import SourceFetcher

logger = logging.getLogger(__name__)


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a file from the repository at the PR head, with line numbers. "
            "Use it to verify cross-file claims before filing a comment. Long "
            "files are truncated; pass start_line/end_line to read just the "
            "part you need."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path, e.g. `src/auth/middleware.py`.",
                },
                "start_line": {
                    "type": ["integer", "null"],
                    "description": "Optional first line to return (1-based).",
                },
                "end_line": {
                    "type": ["integer", "null"],
                    "description": "Optional last line to return (inclusive).",
                },
            },
            "required": ["path"],
        },
    },
}


GREP_REPO_TOOL = {
    "type": "function",
    "function": {
        "name": "grep_repo",
        "description": (
            "Search the repo for files whose path or content matches a pattern. "
            "Use this to find where a symbol is defined or called when you don't "
            "already know the file path. Returns up to 30 matching paths (path "
            "search) or up to 30 line-level hits (content search)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "What to look for. Treated as a regex against file content "
                        "unless `path_only` is true. Keep it specific — short or "
                        "common patterns return too many matches and get capped."
                    ),
                },
                "path_glob": {
                    "type": ["string", "null"],
                    "description": (
                        "Optional glob to restrict the search, e.g. `**/*.java` "
                        "or `src/auth/*`. Defaults to all files."
                    ),
                },
                "path_only": {
                    "type": ["boolean", "null"],
                    "description": (
                        "If true, only match paths (don't read file contents). "
                        "Faster and cheaper. Use when you're hunting for a file "
                        "by name."
                    ),
                },
            },
            "required": ["pattern"],
        },
    },
}


# Default caps to keep tool output from blowing up the prompt or the API
# budget; `review.agentic_*` settings override them per deployment.
DEFAULT_MAX_HOPS = 6
_MAX_FILE_BYTES = 12_000  # ~3k tokens of source; truncate larger files
_MAX_GREP_HITS = 30
_MAX_GREP_BYTES_PER_HIT = 240
_MAX_TOTAL_OUTPUT_BYTES = 50_000  # all tool calls in one review combined


@dataclass
class AgenticToolExecutor:
    """Per-review tool dispatcher with caching and an output-size budget.

    Construct one per chunk review. The `repo_tree` is captured once
    (already fetched for JIT) so `grep_repo` doesn't keep re-listing the
    repo. The `source_fetcher` already has its own per-path cache, so
    repeated `read_file` calls don't re-hit GitHub.
    """

    source_fetcher: SourceFetcher
    repo_tree: list[str]
    # Model calls per chunk, including the final submit_review hop.
    max_hops: int = DEFAULT_MAX_HOPS
    max_file_chars: int = _MAX_FILE_BYTES
    max_total_chars: int = _MAX_TOTAL_OUTPUT_BYTES
    max_grep_hits: int = _MAX_GREP_HITS
    bytes_used: int = 0
    _content_cache: dict[str, str | None] = field(default_factory=dict)
    # Calls already answered in full; a repeat gets a short pointer instead
    # of the same output again (the earlier result is still in the context).
    _answered: set[tuple] = field(default_factory=set)
    call_log: list[dict] = field(default_factory=list)

    @classmethod
    def from_review_config(
        cls, review: object, source_fetcher: SourceFetcher, repo_tree: list[str]
    ) -> AgenticToolExecutor:
        """Build an executor with the `review.agentic_*` limits from config."""
        return cls(
            source_fetcher=source_fetcher,
            repo_tree=repo_tree,
            max_hops=getattr(review, "agentic_max_hops", DEFAULT_MAX_HOPS),
            max_file_chars=getattr(review, "agentic_max_file_chars", _MAX_FILE_BYTES),
            max_total_chars=getattr(review, "agentic_max_total_chars", _MAX_TOTAL_OUTPUT_BYTES),
            max_grep_hits=getattr(review, "agentic_max_grep_hits", _MAX_GREP_HITS),
        )

    @property
    def budget_exhausted(self) -> bool:
        return self.bytes_used >= self.max_total_chars

    async def execute(self, name: str, args: dict) -> str:
        """Dispatch a tool call. Returns the tool result as a string.

        Always returns *something* — errors are reported back to the LLM
        rather than raised, so the model can recover (e.g. by trying a
        different path). The string is what gets fed back into the next
        LLM hop as a `tool` message.
        """
        arg = (args or {}).get("path") or (args or {}).get("pattern") or ""
        self.call_log.append({"tool": name, "arg": arg})

        if self.budget_exhausted:
            return "[tool budget exhausted — no more tool calls accepted; submit your review now]"

        args = args or {}
        try:
            if name == "read_file":
                path = args.get("path") or ""
                if not path:
                    return "[error: missing `path` argument]"
                start, end = _line_arg(args.get("start_line")), _line_arg(args.get("end_line"))
                key: tuple = (name, path, start, end)
                if key in self._answered:
                    return f"[`{path}` was already returned above with the same lines]"
                result = await self._read_file(path, start, end)
            elif name == "grep_repo":
                pattern = args.get("pattern") or ""
                if not pattern:
                    return "[error: missing `pattern` argument]"
                path_glob = args.get("path_glob") or None
                path_only = bool(args.get("path_only") or False)
                key = (name, pattern, path_glob, path_only)
                if key in self._answered:
                    return f"[grep for `{pattern}` was already answered above]"
                result = await self._grep_repo(pattern, path_glob, path_only)
            else:
                return f"[error: unknown tool `{name}`]"
        except Exception as exc:
            logger.debug("Tool %s failed: %s", name, exc)
            return f"[error executing `{name}`: {exc}]"

        if not result.startswith("["):
            self._answered.add(key)
        remaining = self.max_total_chars - self.bytes_used
        if len(result) > remaining:
            result = result[:remaining] + "\n... [truncated: tool output budget used up]"
        self.bytes_used += len(result)
        return result

    async def _read_file(self, path: str, start: int | None = None, end: int | None = None) -> str:
        # Don't waste tokens hunting for files we know don't exist.
        if self.repo_tree and path not in self.repo_tree:
            close = [p for p in self.repo_tree if path.lower() in p.lower()][:5]
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return f"[file not found at PR head: `{path}`.{hint}]"

        if path in self._content_cache:
            content = self._content_cache[path]
        else:
            content = await self.source_fetcher.fetch(path)
            self._content_cache[path] = content

        if content is None:
            return f"[failed to read `{path}`]"

        lines = content.split("\n")
        total = len(lines)
        first = max(1, start or 1)
        last = min(total, end or total)
        if first > total or last < first:
            return f"[`{path}` has {total} lines; requested {start}-{end}]"

        # Number lines so the LLM can cite them by number when filing comments.
        out: list[str] = []
        used = 0
        shown_last = first - 1
        suffix = ""
        for number in range(first, last + 1):
            row = f"{number}  {lines[number - 1]}"
            if used + len(row) > self.max_file_chars:
                if not out:
                    out.append(row[: self.max_file_chars])
                    shown_last = number
                    suffix = f"\n... [truncated: line {number} is longer than the read limit]"
                break
            out.append(row)
            used += len(row) + 1
            shown_last = number
        if shown_last < last:
            suffix = (
                f"\n... [truncated at line {shown_last} of {total}; "
                f"call read_file with start_line={shown_last + 1} for more]"
            )
        span = "" if (first, last) == (1, total) else f" lines {first}-{shown_last} of {total}"
        return f"`{path}`{span}:\n```\n" + "\n".join(out) + f"{suffix}\n```"

    async def _grep_repo(self, pattern: str, path_glob: str | None, path_only: bool) -> str:
        if not self.repo_tree:
            return "[grep unavailable: repo tree not loaded]"

        candidates: list[str]
        if path_glob:
            candidates = [p for p in self.repo_tree if fnmatch.fnmatch(p, path_glob)]
        else:
            candidates = list(self.repo_tree)

        if path_only:
            try:
                rx = re.compile(pattern)
            except re.error:
                # Treat as substring on regex error.
                path_hits = [p for p in candidates if pattern in p][: self.max_grep_hits]
            else:
                path_hits = [p for p in candidates if rx.search(p)][: self.max_grep_hits]
            if not path_hits:
                return f"[no path matches for `{pattern}`]"
            return "Path matches:\n" + "\n".join(f"- `{p}`" for p in path_hits)

        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"[invalid regex `{pattern}`: {exc}]"

        # Skip giant binaries / lockfiles / vendored dirs we'd never want to read.
        skip_exts = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".pdf",
            ".zip",
            ".gz",
            ".tar",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".ico",
            ".svg",
            ".map",
            ".lock",
        )
        skip_dirs = ("node_modules/", "vendor/", "dist/", "build/", ".git/")

        hits: list[str] = []
        files_scanned = 0
        # Each scanned file is a network round-trip via source_fetcher; keep tight.
        max_files_to_scan = 15

        for cand in candidates:
            if len(hits) >= self.max_grep_hits or files_scanned >= max_files_to_scan:
                break
            if any(cand.endswith(ext) for ext in skip_exts):
                continue
            if any(d in cand for d in skip_dirs):
                continue

            files_scanned += 1
            content = self._content_cache.get(cand)
            if content is None and cand not in self._content_cache:
                content = await self.source_fetcher.fetch(cand)
                self._content_cache[cand] = content
            if not content:
                continue

            for lineno, line in enumerate(content.split("\n"), start=1):
                if rx.search(line):
                    snippet = line.strip()
                    if len(snippet) > _MAX_GREP_BYTES_PER_HIT:
                        snippet = snippet[:_MAX_GREP_BYTES_PER_HIT] + "…"
                    hits.append(f"{cand}:{lineno}: {snippet}")
                    if len(hits) >= self.max_grep_hits:
                        break

        if not hits:
            return f"[no content matches for `{pattern}` (scanned {files_scanned} files)]"
        prefix = f"Content matches (scanned {files_scanned} files):"
        if len(hits) == self.max_grep_hits:
            prefix += f" showing first {self.max_grep_hits}"
        return prefix + "\n" + "\n".join(hits)


def _line_arg(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


AGENTIC_TOOLS = [READ_FILE_TOOL, GREP_REPO_TOOL]
