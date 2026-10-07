from mira.core.file_types import (
    Language,
    extension_from_path,
    is_indexable_path,
    language_from_extension,
    language_from_path,
    normalize_extension,
)


def test_extension_normalization_is_case_insensitive():
    assert normalize_extension(" TSX ") == ".tsx"
    assert extension_from_path("src/Widget.TSX") == ".tsx"


def test_language_lookup_accepts_extensions_and_paths():
    assert language_from_extension(".cs") == Language.CSHARP.value
    assert language_from_path("src/main.KTS") == Language.KOTLIN.value
    assert language_from_path("README", default="text") == "text"


def test_indexability_uses_the_shared_normalized_extensions():
    assert is_indexable_path("src/main.PY")
    assert is_indexable_path("schema.graphql")
    assert not is_indexable_path("README.md")


def test_erlang_sources_are_indexable():
    assert is_indexable_path("src/dht_crawler.erl")
    assert is_indexable_path("include/dht_crawler.hrl")
    assert language_from_path("src/dht_crawler.erl") == Language.ERLANG.value
    assert language_from_path("include/dht_crawler.hrl") == Language.ERLANG.value
    assert not is_indexable_path("ebin/dht_crawler.beam")
