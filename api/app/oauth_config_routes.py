"""Authenticated, redacted Google OAuth client configuration endpoints."""

from __future__ import annotations

import hashlib
import os
import re

from fastapi import APIRouter, Depends, Request

from .common import APIError, admin, current_user, get_store, ok
from .gpu.google import GoogleService
from .gpu.repository import Repository
from .security import decrypt_secret, encrypt_secret

router = APIRouter(prefix="/api/gpu")
_CLIENT_ID = re.compile(r"^[0-9]{3,32}-[A-Za-z0-9_-]{3,128}\.apps\.googleusercontent\.com$")
_MAX_SECRET = 4096


def config_id(owner_id: str) -> str:
    return "google:oauth-client:" + hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:40]


def callback_uri() -> str:
    return GoogleService.callback_uri()


def _environment_config() -> tuple[str, str]:
    client_id = os.getenv("GPU_GOOGLE_CLIENT_ID", "").strip()
    secret = (os.getenv("GPU_GOOGLE_SECRET") or os.getenv("GPU_GOOGLE_CLIENT_SECRET") or "").strip()
    return client_id, secret


def _parse(body: dict) -> tuple[str, str]:
    if not isinstance(body, dict) or body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "ต้องยืนยันอย่างชัดเจนเพื่อบันทึกการตั้งค่า Google OAuth Client", 403)
    client_json = body.get("client_json")
    if client_json is not None:
        if not isinstance(client_json, dict) or "installed" in client_json or "web" not in client_json:
            raise APIError("GOOGLE_CLIENT_INVALID", "รองรับเฉพาะ Google OAuth Client ประเภท Web Application เท่านั้น", 422)
        # auth_uri/token_uri are intentionally ignored; endpoints are constants
        # in GoogleService and cannot be supplied by a client.
        web = client_json["web"]
        if not isinstance(web, dict):
            raise APIError("GOOGLE_CLIENT_INVALID", "client_json.web ต้องเป็น object", 422)
        client_id, client_secret = web.get("client_id"), web.get("client_secret")
        redirects = web.get("redirect_uris")
        if not isinstance(redirects, list) or callback_uri() not in redirects:
            raise APIError("GOOGLE_REDIRECT_MISMATCH", "redirect_uris ต้องมี Callback URL ของเซิร์ฟเวอร์", 422)
    else:
        client_id, client_secret = body.get("client_id"), body.get("client_secret")
    if not isinstance(client_id, str) or not _CLIENT_ID.fullmatch(client_id.strip()):
        raise APIError("GOOGLE_CLIENT_INVALID", "รูปแบบ client_id ไม่ถูกต้อง", 422)
    if not isinstance(client_secret, str) or not 8 <= len(client_secret) <= _MAX_SECRET:
        raise APIError("GOOGLE_CLIENT_INVALID", "ความยาว client_secret ไม่ถูกต้อง", 422)
    return client_id.strip(), client_secret


def _public(repo: Repository, owner_id: str) -> dict:
    row = repo.get("credentials", config_id(owner_id), owner_id)
    if (
        row
        and row.get("provider") == "google"
        and row.get("purpose") == "oauth_client_config"
        and not row.get("revoked")
    ):
        return {
            "configured": True,
            "client_id": row.get("client_id"),
            "redirect_uri": callback_uri(),
            "source": "encrypted_database",
        }
    env_id, env_secret = _environment_config()
    return {
        "configured": bool(env_id and env_secret),
        "client_id": env_id or None,
        "redirect_uri": callback_uri(),
        "source": "environment" if env_id or env_secret else "unconfigured",
    }


@router.get("/auth/google/config")
def get_google_config(request: Request, user=Depends(current_user), st=Depends(get_store)):
    return ok(_public(Repository(st), user["id"]), request)


@router.post("/auth/google/config")
def save_google_config(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    client_id, client_secret = _parse(body)
    repo = Repository(st)
    owner = user["id"]
    record_id = config_id(owner)
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("GOOGLE_CONFIG_BUSY", "กำลังบันทึกการตั้งค่า Google OAuth กรุณาลองใหม่ในภายหลัง", 503, True)
        old = repo.get("credentials", record_id, owner)
        connected = [
            row
            for purpose in ("colab", "drive")
            for row in repo.list("credentials", owner)
            if row.get("provider", "google") == "google"
            and row.get("purpose") == purpose
            and not row.get("revoked")
            and row.get("encrypted_refresh_token")
        ]
        if connected:
            try:
                if old:
                    current_id = old.get("client_id")
                    current_secret = decrypt_secret(old["encrypted_client_secret"])
                else:
                    current_id, current_secret = _environment_config()
                unchanged = current_id == client_id and current_secret == client_secret
            except (KeyError, TypeError, ValueError):
                unchanged = False
            if not unchanged:
                raise APIError(
                    "GOOGLE_CONFIG_CONNECTED", "กรุณายกเลิกการเชื่อมต่อ Colab หรือ Drive ที่มีอยู่ก่อนเปลี่ยน Google Client", 409
                )
        data = {
            "provider": "google",
            "purpose": "oauth_client_config",
            "client_id": client_id,
            "encrypted_client_secret": encrypt_secret(client_secret),
        }
        if old:
            repo.update("credentials", record_id, data, owner)
        else:
            repo.create("credentials", owner, data, record_id=record_id)
        GoogleService(repo, owner).invalidate_oauth_states(owner)
    return ok(_public(repo, owner), request)


__all__ = ["router", "config_id"]
