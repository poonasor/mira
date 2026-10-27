"""Tests for ForgejoProvider permission lookups."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mira.exceptions import ProviderError
from mira.providers.forgejo import ForgejoProvider


class _FakeResp:
    def __init__(self, status=200, json_data=None):
        self.status_code = status
        self._json = json_data
        self.text = ""

    def json(self):
        return self._json


class _FakeClient:
    def __init__(self, handler):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def request(self, method, url, **kw):
        return self._handler(method, url, **kw)


def _patch(handler):
    return patch("mira.providers.forgejo.httpx.AsyncClient", lambda *a, **k: _FakeClient(handler))


class TestGetRepoPermission:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("raw", "normalized"),
        [
            ("admin", "write"),
            ("owner", "write"),
            ("write", "write"),
            ("read", "read"),
            ("none", "none"),
        ],
    )
    async def test_normalizes_permission_levels(self, raw, normalized):
        def handler(method, url, **kw):
            assert "/collaborators/alice/permission" in url
            return _FakeResp(json_data={"permission": raw})

        with _patch(handler):
            permission = await ForgejoProvider("tok").get_repo_permission(
                "acme", "widgets", "alice"
            )
        assert permission == normalized

    @pytest.mark.asyncio
    async def test_missing_collaborator_is_none(self):
        def handler(method, url, **kw):
            return _FakeResp(status=404, json_data={})

        with _patch(handler):
            permission = await ForgejoProvider("tok").get_repo_permission(
                "acme", "widgets", "alice"
            )
        assert permission == "none"

    @pytest.mark.asyncio
    async def test_unknown_permission_is_none(self):
        def handler(method, url, **kw):
            return _FakeResp(json_data={"permission": "mystery"})

        with _patch(handler):
            permission = await ForgejoProvider("tok").get_repo_permission(
                "acme", "widgets", "alice"
            )
        assert permission is None

    @pytest.mark.asyncio
    async def test_forbidden_lookup_explains_admin_requirement(self):
        def handler(method, url, **kw):
            return _FakeResp(status=403)

        with _patch(handler), pytest.raises(ProviderError, match="repository admin/owner"):
            await ForgejoProvider("tok").get_repo_permission("acme", "widgets", "alice")
