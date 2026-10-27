"""Unit tests for the Cursor Origin provider helpers."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import mira.platforms.origin.auth as origin_auth
import mira.providers.origin as origin_provider
from mira.exceptions import ProviderError
from mira.providers.origin import OriginProvider, _patches_to_diff, parse_pr_url


def test_parse_pr_url_codebase() -> None:
    owner, repo, number = parse_pr_url("https://cursor.com/codebase/acme/rocket/pull/17")
    assert (owner, repo, number) == ("acme", "rocket", 17)


def test_parse_pr_url_git_host() -> None:
    owner, repo, number = parse_pr_url("https://origin.cursor.com/acme/rocket/pull/17")
    assert (owner, repo, number) == ("acme", "rocket", 17)


def test_parse_pr_url_rejects_garbage() -> None:
    with pytest.raises(ProviderError):
        parse_pr_url("https://github.com/acme/rocket/pull/17")


def test_patches_to_diff_modified() -> None:
    diff = _patches_to_diff(
        [
            {
                "filename": "src/a.ts",
                "status": "modified",
                "patch": "@@ -1,2 +1,3 @@\n a\n+b\n c\n",
            }
        ]
    )
    assert "diff --git a/src/a.ts b/src/a.ts" in diff
    assert "+++ b/src/a.ts" in diff
    assert "+b" in diff


# ── token factory / expiry handling ──────────────────────────────────────────
# Origin installation tokens expire ~5 minutes after mint (capped by the app
# JWT). The provider must therefore accept a token factory and re-resolve it
# per request, retrying once with a force-refresh when the API answers 401.


def _mock_transport_client(handler: Any) -> Any:
    """Wrap httpx.AsyncClient so every instance uses a MockTransport."""

    real_client = httpx.AsyncClient

    def client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return real_client(*args, **kwargs)

    return client_factory


@pytest.mark.asyncio
async def test_token_factory_resolved_per_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_auth: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={})

    async def factory(force: bool = False) -> str:
        return f"tok-{len(seen_auth)}"

    monkeypatch.setattr(origin_provider.httpx, "AsyncClient", _mock_transport_client(handler))
    provider = OriginProvider(factory)

    await provider._request("GET", "https://api.example.test/x")
    await provider._request("GET", "https://api.example.test/x")

    assert seen_auth == ["Bearer tok-0", "Bearer tok-1"]


@pytest.mark.asyncio
async def test_401_force_refreshes_token_and_retries_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("Authorization", "")
        attempts.append(auth)
        if auth != "Bearer fresh":
            return httpx.Response(
                401, json={"code": 16, "message": "The access token expired"}
            )
        return httpx.Response(200, json={"ok": True})

    async def factory(force: bool = False) -> str:
        return "fresh" if force else "stale"

    monkeypatch.setattr(origin_provider.httpx, "AsyncClient", _mock_transport_client(handler))
    provider = OriginProvider(factory)

    resp = await provider._request("GET", "https://api.example.test/pulls/1")

    assert resp.status_code == 200
    assert attempts == ["Bearer stale", "Bearer fresh"]


@pytest.mark.asyncio
async def test_static_token_does_not_retry_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            401, json={"code": 16, "message": "The access token expired"}
        )

    monkeypatch.setattr(origin_provider.httpx, "AsyncClient", _mock_transport_client(handler))
    provider = OriginProvider("static-token")

    with pytest.raises(ProviderError) as excinfo:
        await provider._request("GET", "https://api.example.test/x")

    assert getattr(excinfo.value, "status_code", None) == 401
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_get_installation_token_force_bypasses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    mints: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        mints.append(1)
        return httpx.Response(
            200,
            json={"token": f"oit_{len(mints)}", "expiresAt": "2099-01-01T00:00:00Z"},
        )

    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()

    monkeypatch.setattr(origin_auth.httpx, "AsyncClient", _mock_transport_client(handler))
    auth = origin_auth.OriginAppAuth("app_1", pem)

    first = await auth.get_installation_token("i_1")
    cached = await auth.get_installation_token("i_1")
    forced = await auth.get_installation_token("i_1", force=True)

    assert first == cached == "oit_1"
    assert forced == "oit_2"
    assert len(mints) == 2


# ── command authorization: collaborator permission lookup ───────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (200, {"permission": "admin"}, "write"),
        (200, {"permission": "write"}, "write"),
        (200, {"permission": "read"}, "read"),
        (200, {"permission": "custom"}, "read"),
        (404, {"code": 5}, "none"),
    ],
)
async def test_get_repo_permission_normalizes(
    monkeypatch: pytest.MonkeyPatch, status: int, body: dict, expected: str
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(status, json=body)

    monkeypatch.setattr(origin_provider.httpx, "AsyncClient", _mock_transport_client(handler))
    provider = OriginProvider("tok")

    perm = await provider.get_repo_permission("acme", "rocket", "alice", user_id="user_abc")

    assert perm == expected
    assert seen == ["/v1/origin/repos/acme/rocket/collaborators/user_abc/permission"]


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [None, 42, "app_123"])
async def test_get_repo_permission_without_user_id_is_undetermined(
    monkeypatch: pytest.MonkeyPatch, user_id: Any
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("no lookup without a user_… id")

    monkeypatch.setattr(origin_provider.httpx, "AsyncClient", _mock_transport_client(handler))
    provider = OriginProvider("tok")

    assert await provider.get_repo_permission("acme", "rocket", "alice", user_id=user_id) is None


@pytest.mark.asyncio
async def test_get_repo_permission_missing_scope_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without repository:members:read the lookup fails; command_policy fails closed.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"code": 7})

    monkeypatch.setattr(origin_provider.httpx, "AsyncClient", _mock_transport_client(handler))
    provider = OriginProvider("tok")

    with pytest.raises(ProviderError):
        await provider.get_repo_permission("acme", "rocket", "alice", user_id="user_abc")
