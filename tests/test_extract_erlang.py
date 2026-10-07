"""Erlang symbol extraction — modules, functions, and macros/records from headers."""

from __future__ import annotations

from mira.index.extract import extract_symbols, find_symbol_by_name

MODULE_STYLE = """\
-module(dht_crawler).
-export([crawl/1, start_link/0]).
-import(lists, [map/2]).

%% @doc Crawl the DHT network starting from Node.
crawl(Node) ->
    Peers = get_peers(Node),
    [store_torrent(P) || P <- Peers],
    ok.

get_peers(Node) ->
    case dht:lookup(Node) of
        {ok, Peers} -> Peers;
        _ -> []
    end.

store_torrent(_P) ->
    ok.
"""

HEADER_STYLE = """\
-ifndef(DHT_CRAWLER_HRL).
-define(DHT_CRAWLER_HRL, true).

-define(MAX_PEERS, 1024).

-record(torrent, {
    info_hash :: binary(),
    name :: string()
}).

-endif.
"""


def test_erlang_functions_are_extracted():
    symbols = extract_symbols(MODULE_STYLE, "erlang")
    by_name = {s.name: s for s in symbols}
    assert "crawl/1" in by_name
    assert "get_peers/1" in by_name
    assert "store_torrent/1" in by_name


def test_erlang_function_kind_is_function():
    symbols = extract_symbols(MODULE_STYLE, "erlang")
    assert all(s.kind == "function" for s in symbols)


def test_erlang_function_body_includes_clause():
    crawl = find_symbol_by_name(MODULE_STYLE, "erlang", "crawl/1")
    assert crawl is not None
    assert "store_torrent(P)" in crawl.source
    # The span must end before the next clause's definition line.
    assert "get_peers(Node) ->" not in crawl.source


def test_erlang_header_macros_and_records_are_extracted():
    symbols = extract_symbols(HEADER_STYLE, "erlang")
    by_name = {s.name: s for s in symbols}
    assert by_name.get("MAX_PEERS") is not None
    assert by_name.get("torrent") is not None
    assert by_name["torrent"].kind == "record"


def test_erlang_directives_are_not_symbols():
    symbols = extract_symbols(MODULE_STYLE, "erlang")
    names = {s.name for s in symbols}
    assert "dht_crawler" not in names  # -module
    assert "crawl/1, start_link/0" not in names  # -export
