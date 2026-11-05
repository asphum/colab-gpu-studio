"""Shared authenticated API primitives; no import of the application composition root."""

import secrets
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from .config import settings
from .security import read_session


class APIError(Exception):
    def __init__(self, code, message, status=422, retryable=False):
        self.code, self.message, self.status, self.retryable = code, message, status, retryable


def envelope(data=None, error=None, request_id=None):
    return {"data": data, "error": error, "request_id": request_id or str(uuid.uuid4())}


def ok(data, request=None, status=200):
    return JSONResponse(
        envelope(data, request_id=getattr(getattr(request, "state", None), "request_id", None)),
        status_code=status,
    )


def get_store(request: Request):
    return request.app.state.store


def current_user(request: Request):
    authorization = request.headers.get("Authorization", "")
    raw = authorization[7:] if authorization.startswith("Bearer ") else ""
    if raw and settings.studio_api_token and secrets.compare_digest(raw, settings.studio_api_token):
        return {"id": "owner", "role": "admin", "api_client": True}
    if not raw and request.method in {"GET", "HEAD"}:
        raw = request.cookies.get("studio_session", "")
    payload = read_session(raw)
    if not payload or not payload.get("uid"):
        raise APIError("UNAUTHENTICATED", "กรุณาเข้าสู่ระบบสตูดิโอ GPU ส่วนตัวก่อน", 401)
    return {"id": payload["uid"], "role": payload.get("role", "user")}


def own(st, kind, record_id, user):
    value = st.get(kind, record_id, user["id"])
    if not value:
        raise APIError("NOT_FOUND", "ไม่พบทรัพยากร หรือทรัพยากรนี้ไม่ได้เป็นของผู้ใช้ปัจจุบัน", 404)
    return value


def admin(user):
    if user.get("role") != "admin":
        raise APIError("FORBIDDEN", "การดำเนินการนี้ต้องได้รับการยืนยันจากผู้ดูแลระบบ", 403)
