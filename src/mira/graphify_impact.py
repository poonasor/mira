"""Graphify impact annotations for PR reviews.

Reads the code graphs that the runner's bin/graphify_refresh.py keeps fresh
(mounted read-only at /graphs in the mira container: registry.json plus one
graph.json per repo). For a reviewed PR it answers one question: "if this
change lands, how much of the repo's dependency graph is downstream of it?"

Pure stdlib, read-only, and fail-soft by design: any missing graph, missing
field or parse error yields an empty section, never a failed review. Buckets
use the same cut-offs the runner's routing was calibrated on (see
runner-engine routing_live.BlastCutoffs): low <= 60 nodes and <= 6
communities, high >= 300 nodes or >= 40 communities, else medium. Override
per deployment with GRAPHIFY_BLAST_* env vars.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

GRAPHS_DIR = Path(os.environ.get("GRAPHIFY_GRAPHS", "/graphs"))
REGISTRY = GRAPHS_DIR / "registry.json"
DEPTH = int(os.environ.get("GRAPHIFY_BLAST_DEPTH", "2"))
LOW_MAX_NODES = int(os.environ.get("GRAPHIFY_BLAST_LOW_NODES", "60"))
LOW_MAX_COMMUNITIES = int(os.environ.get("GRAPHIFY_BLAST_LOW_COMMUNITIES", "6"))
HIGH_MIN_NODES = int(os.environ.get("GRAPHIFY_BLAST_HIGH_NODES", "300"))
HIGH_MIN_COMMUNITIES = int(os.environ.get("GRAPHIFY_BLAST_HIGH_COMMUNITIES", "40"))

_graph_cache: dict[str, dict | None] = {}


def _registry() -> dict:
    try:
        return json.loads(REGISTRY.read_text())
    except Exception:  # noqa: BLE001 - absent registry = no annotations
        return {}


def _graph_file(entry: dict) -> Path | None:
    """Locate the graph.json for a registry entry, host paths included.

    The registry stores absolute host paths (/Users/x/selfhost/graphify/graphs/…)
    while the container mounts the same tree at /graphs: remap by cutting at the
    '/graphify/graphs/' marker and re-rooting under GRAPHS_DIR.
    """
    p = str(entry.get("graph") or entry.get("graph_path") or "")
    cands = [p] if p else []
    marker = "/graphify/graphs/"
    if marker in p:
        cands.append(str(GRAPHS_DIR / p.split(marker, 1)[1].lstrip("/")))
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def find_repo_entry(owner: str, repo: str) -> tuple[str, dict] | None:
    """Match a GitHub owner/repo against the registry's local repo_dirs.

    Matched by directory tail or repo_slug tail (e.g. the local checkout
    /Users/x/Development/open-seo-runner answers to the GitHub repo
    "open-seo-runner"). Ambiguous or missing -> None.
    """
    reg = _registry()
    if not reg:
        return None
    r = repo.strip().lower()
    hits: dict[str, dict] = {}
    for rd, entry in reg.items():
        tail = rd.rstrip("/").split("/")[-1].lower()
        slug = str(entry.get("repo_slug") or "").lower()
        if r == tail or (slug and r == slug.split("/")[-1]):
            hits[rd] = entry
    if len(hits) != 1:
        return None
    rd, entry = next(iter(hits.items()))
    if _graph_file(entry) is None:
        return None
    return rd, entry


def has_repo(repo: str) -> bool:
    """Cheap gate: is there exactly one graph for this repo? No API calls."""
    return find_repo_entry("", repo) is not None


def _load_graph(path: Path) -> dict | None:
    key = str(path)
    if key not in _graph_cache:
        try:
            _graph_cache[key] = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            _graph_cache[key] = None
    return _graph_cache[key]


def impact_for_changes(repo: str, changed_paths: list[str]) -> dict | None:
    """Reverse-dependency blast for changed paths, mirroring the runner's
    graphify_blast_helper semantics (path seeds, depth-2 reverse traversal)."""
    hit = find_repo_entry("", repo)
    if not hit or not changed_paths:
        return None
    rd, entry = hit
    data = _load_graph(_graph_file(entry))
    if not data:
        return None
    nodes = data.get("nodes") or []
    links = data.get("links") or []
    by_file: dict[str, list[str]] = {}
    comm_of: dict[str, object] = {}
    file_of: dict[str, str] = {}
    for n in nodes:
        nid = str(n.get("id"))
        sf = str(n.get("source_file") or "").lower()
        if sf:
            by_file.setdefault(sf, []).append(nid)
            file_of[nid] = sf
        comm_of[nid] = n.get("community")
    rev: dict[str, list[str]] = {}
    for link in links:
        rev.setdefault(str(link.get("target")), []).append(str(link.get("source")))

    seeds: set[str] = set()
    for p in changed_paths or []:
        pl = p.lower().lstrip("./")
        matches = [f for f in by_file if f == pl or f.endswith("/" + pl)]
        limit = 5
        if not matches:  # a directory: every file under it, wider cap
            matches = [f for f in by_file if f.startswith(pl + "/") or ("/" + pl + "/") in f]
            limit = 30
        if matches and len(matches) <= limit:
            for f in matches:
                seeds.update(by_file[f])

    if not seeds:
        return {
            "repo_dir": rd,
            "seeds": 0,
            "affected_nodes": None,
            "affected_communities": None,
            "affected_files": None,
            "bucket": None,
            "depth": DEPTH,
        }
    frontier, seen = set(seeds), set(seeds)
    for _ in range(DEPTH):
        nxt: set[str] = set()
        for nid in frontier:
            for dep in rev.get(nid, ()):
                if dep not in seen:
                    nxt.add(dep)
        seen |= nxt
        frontier = nxt
    hit_nodes = seen
    comms = {comm_of[n] for n in hit_nodes if comm_of[n] is not None}
    files = {file_of[n] for n in hit_nodes if file_of.get(n)}
    n, c = len(hit_nodes), len(comms)
    if n >= HIGH_MIN_NODES or c >= HIGH_MIN_COMMUNITIES:
        bucket = "high"
    elif n <= LOW_MAX_NODES and c <= LOW_MAX_COMMUNITIES:
        bucket = "low"
    else:
        bucket = "medium"
    return {
        "repo_dir": rd,
        "seeds": len(seeds),
        "affected_nodes": n,
        "affected_communities": c,
        "affected_files": len(files),
        "bucket": bucket,
        "depth": DEPTH,
    }


def impact_section(owner: str, repo: str, changed_paths: list[str]) -> str:
    """Markdown block for the review body; "" when there is nothing to say."""
    try:
        m = impact_for_changes(repo, changed_paths)
    except Exception:  # noqa: BLE001 - an annotation must never fail a review
        return ""
    if not m or m.get("bucket") is None:
        return ""
    label = {"low": "🟢 low", "medium": "🟡 medium", "high": "🔴 high"}.get(
        str(m.get("bucket")), str(m.get("bucket"))
    )
    return (
        f"\n\n---\n\n**Graphify impact:** {label} — this PR is upstream of "
        f"{m['affected_nodes']} graph nodes in {m['affected_communities']} "
        f"communities across {m['affected_files']} files "
        f"(reverse dependents, depth {m['depth']}, seeds: {m['seeds']} changed "
        f"paths). Larger blast radius means the change deserves wider review "
        f"attention."
    )
