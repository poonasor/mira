"""Unit tests for Origin webhook envelope + signature helpers."""

from __future__ import annotations

import base64
import hashlib
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mira.platforms.origin import webhook as wh


def test_unwrap_envelope() -> None:
    event_type, installation_id, payload = wh.unwrap_envelope(
        {
            "deliveryId": "whd_1",
            "appId": "app_1",
            "installationId": "i_99",
            "event": {
                "id": "evt_1",
                "type": "pull_request.created",
                "eventTime": "2026-01-01T00:00:00Z",
                "payload": {"pullRequest": {"number": "1"}},
            },
        }
    )
    assert event_type == "pull_request.created"
    assert installation_id == "i_99"
    assert payload["pullRequest"]["number"] == "1"


@pytest.mark.asyncio
async def test_verify_origin_signature_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    private = Ed25519PrivateKey.generate()
    public = private.public_key()

    async def fake_jwks(force: bool = False):
        return [public]

    monkeypatch.setattr(wh, "_load_jwks", fake_jwks)

    body = b'{"event":{"type":"ping"}}'
    webhook_id = "whd_test"
    ts = str(int(time.time()))
    digest = hashlib.sha256(f"{webhook_id}.{ts}.".encode() + body).hexdigest().encode()
    signature = base64.b64encode(private.sign(digest)).decode()
    header = f"v1ed,{signature}"

    assert await wh.verify_origin_signature(body, webhook_id, ts, header) is True
    assert await wh.verify_origin_signature(body, webhook_id, ts, "v1ed,AAAA") is False
    assert await wh.verify_origin_signature(body, "other", ts, header) is False


# ── command authorization on Origin comments ────────────────────────────────


def _origin_comment_payload(body: str) -> dict:
    return {
        "comment": {
            "id": "cmt_1",
            "body": body,
            "author": {"user": {"id": "user_alice", "handle": "alice"}},
        },
        "pullRequest": {
            "number": 7,
            "repository": {"owner": {"slug": "acme"}, "name": "rocket"},
        },
    }


async def _run_origin_comment(body: str, permission: str | None):
    from unittest.mock import AsyncMock, MagicMock, patch

    from mira.config import MiraConfig
    from mira.platforms.origin.webhook import handle_origin_comment

    provider = MagicMock()
    provider.get_pr_info = AsyncMock(return_value=MagicMock(owner="acme", repo="rocket"))
    provider.get_repo_permission = AsyncMock(return_value=permission)
    provider.add_label = AsyncMock()
    provider.post_comment = AsyncMock()
    auth = MagicMock()
    auth.get_bot_identity = AsyncMock(return_value="")
    with (
        patch("mira.platforms.origin.webhook.create_provider", return_value=provider),
        patch("mira.platforms.origin.webhook.load_config", return_value=MiraConfig()),
        patch("mira.platforms.handlers.run_pr_command", new_callable=AsyncMock) as run_cmd,
    ):
        await handle_origin_comment(_origin_comment_payload(body), auth, "mira-bot", "inst_1")
    return provider, run_cmd


async def test_origin_pause_requires_write_permission() -> None:
    provider, _ = await _run_origin_comment("@mira-bot pause", "read")
    provider.get_repo_permission.assert_awaited_once()
    assert provider.get_repo_permission.await_args.kwargs["user_id"] == "user_alice"
    provider.add_label.assert_not_awaited()


async def test_origin_pause_allowed_for_writer() -> None:
    provider, _ = await _run_origin_comment("@mira-bot pause", "write")
    provider.add_label.assert_awaited_once()


async def test_origin_review_command_passes_user_id() -> None:
    _, run_cmd = await _run_origin_comment("@mira-bot review", "write")
    run_cmd.assert_awaited_once()
    assert run_cmd.await_args.kwargs["user_id"] == "user_alice"
