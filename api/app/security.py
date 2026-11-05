"""No provider or Google credentials leave this process unencrypted."""

import base64
import hashlib
import hmac
import json
import time

from cryptography.fernet import Fernet, InvalidToken

from .config import settings


def _fernet():
    if not settings.encryption_key:
        raise RuntimeError("ต้องกำหนดค่า ENCRYPTION_KEY ก่อน; โปรดรัน scripts/init_env.py")
    return Fernet(settings.encryption_key.encode())


def encrypt_secret(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("secret ต้องไม่เป็นค่าว่าง")
    return _fernet().encrypt(value.encode()).decode()


def decrypt_secret(value: str) -> str:
    try:
        return _fernet().decrypt(value.encode()).decode()
    except (InvalidToken, TypeError, ValueError) as exc:
        raise ValueError("ไม่สามารถถอดรหัสข้อมูลประจำตัวได้ (unable to decrypt credentials)") from exc


def redact(value):
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if any(
                x in k.lower() for x in ("secret", "token", "password", "cookie", "authorization", "api_key")
            )
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def hash_request(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def sign_session(payload):
    if not settings.session_secret:
        raise RuntimeError("ยังไม่ได้กำหนดค่า SESSION_SECRET")
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    sig = hmac.new(settings.session_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return raw + "." + sig


def read_session(value):
    try:
        raw, sig = value.rsplit(".", 1)
        expected = hmac.new(settings.session_secret.encode(), raw.encode(), hashlib.sha256).hexdigest()
        if not settings.session_secret or not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        if payload.get("exp", 0) <= time.time() or payload.get("iat", 0) > time.time() + 60:
            return None
        return payload
    except (ValueError, AttributeError, TypeError):
        return None
