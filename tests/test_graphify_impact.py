"""Graphify impact annotations: registry matching, BFS blast, section text."""

import json
import sys
from pathlib import Path

import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mira import graphify_impact as gi  # noqa: E402


def _write_graph(tmp_path, nodes, links):
    data = {"directed": True, "multigraph": False, "graph": {}, "nodes": nodes, "links": links}
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    """One repo registered, tiny graph: a.py -> b.py -> c.py (call chain)."""
    g = _write_graph(
        tmp_path,
        [
            {"id": "a", "label": "a", "source_file": "src/a.py", "community": 1},
            {"id": "b", "label": "b", "source_file": "src/b.py", "community": 1},
            {"id": "c", "label": "c", "source_file": "src/c.py", "community": 2},
            {"id": "d", "label": "d", "source_file": "src/d.py", "community": 3},
        ],
        [
            {"source": "b", "target": "a", "relation": "calls"},  # b calls a
            {"source": "c", "target": "b", "relation": "calls"},
        ],
    )  # c calls b
    reg = {
        "/Users/x/Development/open-seo-runner": {
            "graph": str(g),
            "repo_slug": "poonasor/open-seo-runner",
        }
    }
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(reg))
    monkeypatch.setattr(gi, "REGISTRY", p)
    monkeypatch.setattr(gi, "_graph_cache", {})
    return reg


def test_find_repo_entry_by_tail_and_slug(env):
    assert gi.find_repo_entry("", "open-seo-runner")
    assert gi.find_repo_entry("poonasor", "open-seo-runner")
    assert gi.find_repo_entry("", "unknown-repo") is None


def test_impact_counts_reverse_dependents_at_depth(env):
    m = gi.impact_for_changes("open-seo-runner", ["src/a.py"])
    # a.py changed -> dependents b, c (depth 2); d untouched. 3 nodes /
    # 2 communities is inside the low cut-offs (60/6).
    assert (m["affected_nodes"], m["affected_communities"], m["bucket"]) == (3, 2, "low")
    m = gi.impact_for_changes("open-seo-runner", ["src/d.py"])
    assert m["affected_nodes"] == 1 and m["bucket"] == "low"


def test_directory_seed_and_no_match(env):
    m = gi.impact_for_changes("open-seo-runner", ["src"])
    assert m["seeds"] == 4 and m["affected_nodes"] == 4
    m = gi.impact_for_changes("open-seo-runner", ["nope/zzz.py"])
    assert m["bucket"] is None and m["affected_nodes"] is None


def test_section_empty_when_no_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(gi, "REGISTRY", tmp_path / "missing.json")
    monkeypatch.setattr(gi, "_graph_cache", {})
    assert gi.impact_section("poonasor", "whatever", ["src/a.py"]) == ""


def test_section_marks_high(monkeypatch):
    monkeypatch.setattr(
        gi,
        "impact_for_changes",
        lambda *a, **k: {
            "repo_dir": "/r",
            "seeds": 3,
            "affected_nodes": 900,
            "affected_communities": 90,
            "affected_files": 100,
            "bucket": "high",
            "depth": 2,
        },
    )
    s = gi.impact_section("o", "r", ["a.py"])
    assert "Graphify impact:" in s and "high" in s and "900" in s
