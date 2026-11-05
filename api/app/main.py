"""Independent personal AI Studio API."""

from __future__ import annotations

import copy
import io
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .common import APIError, admin, current_user, envelope, get_store, ok, own
from .compute_routes import router as compute_router
from .config import settings
from .db import Store
from .gpu import loras as lora_registry
from .gpu.compute import GpuConfig, MockColabProvider
from .gpu.google import GoogleError
from .gpu.inputs import resolve_input_images
from .gpu.manifest import bind_parameters, digest, resolve_parameters, validate_manifest
from .gpu.rates import estimate as estimate_units
from .gpu.rates import length_factor, mode_param
from .gpu.registry import Registry
from .gpu.repository import Repository
from .gpu.scheduler import Scheduler
from .oauth_config_routes import router as oauth_config_router
from .review_routes import router as review_router
from .security import sign_session
from .storage import asset_path, delete_asset_file, save_asset


def repo_registry(st, user):
    repo = Repository(st)
    registry = Registry(repo)
    registry.bootstrap(user["id"])
    return repo, registry


def public_asset(asset):
    return {k: v for k, v in asset.items() if k not in {"storage_key", "owner_id"}}


def public_job(job):
    # Only explicit business fields are exposed. Runtime handles/connectionInfo never leave the server.
    keys = {
        "id",
        "project_id",
        "model_id",
        "workflow_key",
        "workflow_version",
        "state",
        "provider",
        "params",
        "resolved_seed",
        "error_code",
        "error_message",
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "estimated_units",
        "actual_units",
        "simulated_units",
        "unit_rate",
        "max_runtime_minutes",
        "output_asset_ids",
        "cancel_requested",
        "progress",
        "runtime_deleted",
        "execution_id",
        "stage",
        "prep",
        "for_job_id",
        "prep_job_id",
        "awaiting_prep",
    }
    result = {k: v for k, v in job.items() if k in keys}
    result["job_id"] = job["id"]
    result["mock"] = job.get("provider") == "mock"
    result["resolved_params"] = (job.get("snapshot") or {}).get("resolved_params", {})
    return result


def _ts(value):
    from datetime import datetime, timezone

    moment = datetime.fromisoformat(value)
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).timestamp()


def job_timings(job, events):
    """Seconds per stage: worker-measured stages plus allocate/upload/delete from scheduler events."""
    timings = dict(job.get("worker_timings") or {})
    if job.get("prep_seconds") is not None:
        timings["prep_cache"] = job["prep_seconds"]
    entered = {}
    for event in sorted(events, key=lambda e: e["created_at"]):
        entered.setdefault(event["state"], _ts(event["created_at"]))
    end = next((entered[s] for s in ("completed", "failed", "cancelled") if s in entered), None)
    spans = {
        "allocate": ("allocating", "loading"),
        "upload": ("uploading", "cleaning_up"),
        "delete": ("cleaning_up", None),
    }
    for name, (first, last) in spans.items():
        stop = end if last is None else entered.get(last)
        if first in entered and stop is not None:
            timings[name] = round(stop - entered[first], 1)
    return timings


def usage(st, owner):
    repo = Repository(st)
    cfg = GpuConfig.from_env()
    now = time.time()
    rows = repo.list("usage", owner)
    real = [r for r in rows if not r.get("mock") and not r.get("simulated") and r.get("provider") != "mock"]
    estimate = sum(float(r.get("units_estimate", r.get("units", 0)) or 0) for r in real)
    actual_rows = [r.get("units_actual") for r in real if r.get("units_actual") is not None]
    daily = sum(
        float(r.get("units_estimate", r.get("units", 0)) or 0)
        for r in real
        if float(r.get("created_epoch", 0)) > now - 86400
    )
    leases = [r for r in repo.list("leases", owner) if r.get("state") != "deleted"]
    return {
        "estimated_units": estimate if cfg.unit_rate is not None else None,
        "actual_units": sum(actual_rows) if actual_rows else None,
        "daily_estimated_units": daily if cfg.unit_rate is not None else None,
        "unit_rate": cfg.unit_rate,
        "rate_unit": "compute_units_per_minute",
        "rate_source": "administrator_estimate" if cfg.unit_rate is not None else "unknown",
        "daily_limit": cfg.daily_units,
        "monthly_limit": cfg.monthly_units,
        "max_job_units": cfg.max_job_units,
        "max_runtime_minutes": cfg.max_runtime_minutes,
        "active_runtimes": len(leases),
        "ledger": rows,
        "notice": "อัตราค่าบริการเป็นค่าประมาณการจากผู้ดูแลระบบ การหักค่าใช้จ่ายจริงยึดตามบัญชี Google เป็นหลัก โหมด Mock ไม่สร้าง GPU และไม่เกิดค่าใช้จ่ายจริง",
    }


@asynccontextmanager
async def lifespan(app):
    if not getattr(app.state, "store", None):
        app.state.store = Store(settings.database_url)
    Repository(app.state.store)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Colab GPU Studio", version=__version__, lifespan=lifespan)
router = APIRouter()
_login_limits = {}


@app.middleware("http")
async def security_context(request, call_next):
    request.state.request_id = str(uuid.uuid4())
    try:
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            if origin and origin.rstrip("/") != settings.site_url.rstrip("/"):
                from urllib.parse import urlsplit

                if urlsplit(origin).netloc != request.url.netloc:
                    raise APIError("CSRF_ORIGIN", "คำขอข้ามไซต์ (Cross-Origin) ถูกปฏิเสธ", 403)
        if int(request.headers.get("Content-Length", "0")) > settings.max_upload_bytes + 1048576:
            raise APIError("PAYLOAD_TOO_LARGE", "ขนาดคำขอเกินขีดจำกัดการอัปโหลด", 413)
        response = await call_next(request)
    except APIError as exc:
        response = error_response(request, exc)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "private, no-store"
    return response


def error_response(request, exc):
    error = {"code": exc.code, "message": exc.message, "retryable": exc.retryable}
    result = envelope(error=error, request_id=getattr(request.state, "request_id", None))
    result.update(code=exc.code, message=exc.message)
    return JSONResponse(result, status_code=exc.status)


@app.exception_handler(APIError)
async def api_error(request, exc):
    return error_response(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    # Do not return rejected input values (they may contain secrets).
    return error_response(
        request, APIError("VALIDATION_ERROR", "รูปแบบคำขอไม่ถูกต้อง โปรดตรวจสอบประเภทพารามิเตอร์และฟิลด์ที่จำเป็น", 422)
    )


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return error_response(request, APIError("VALIDATION_ERROR", str(exc)[:300], 422))


@app.exception_handler(KeyError)
async def key_error(request, exc):
    return error_response(request, APIError("NOT_FOUND", "ไม่พบทรัพยากร", 404))


@app.exception_handler(Exception)
async def internal_error(request, exc):
    return error_response(request, APIError("INTERNAL_ERROR", "การดำเนินการหลังบ้านล้มเหลว โปรดดูบันทึกเหตุการณ์ของงานที่คัดกรองข้อมูลลับแล้ว", 500))


@app.get("/api/health")
def health(st=Depends(get_store)):
    from sqlalchemy import text

    with st.engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return {"status": "ok", "version": __version__, "application": "colab-gpu-studio", "independent": True}


class Login(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


@app.post("/api/auth/login")
def login(body: Login, request: Request):
    key = request.client.host if request.client else "unknown"
    previous = [t for t in _login_limits.get(key, []) if t > time.monotonic() - 60]
    if len(previous) >= 10:
        raise APIError("RATE_LIMITED", "พยายามเข้าสู่ระบบถี่เกินไป กรุณาลองใหม่ในภายหลัง", 429)
    previous.append(time.monotonic())
    _login_limits[key] = previous
    if not settings.studio_password or not secrets.compare_digest(body.password, settings.studio_password):
        raise APIError("INVALID_CREDENTIALS", "รหัสผ่านไม่ถูกต้อง หรือยังไม่ได้กำหนดค่าเริ่มต้น", 401)
    token = sign_session(
        {"uid": "owner", "role": "admin", "iat": int(time.time()), "exp": int(time.time()) + 43200}
    )
    response = JSONResponse({"access_token": token, "token_type": "bearer", "expires_in": 43200})
    response.set_cookie(
        "studio_session",
        token,
        httponly=True,
        secure=settings.site_url.startswith("https://"),
        samesite="lax",
        max_age=43200,
    )
    return response


@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


@router.get("/projects")
def projects(request: Request, user=Depends(current_user), st=Depends(get_store)):
    return ok(st.list("projects", user["id"]), request)


@router.post("/projects")
def create_project(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    title = str(body.get("title", "")).strip()[:200]
    if not title:
        raise APIError("VALIDATION_ERROR", "ชื่อโปรเจกต์ต้องไม่เป็นค่าว่าง")
    return ok(
        st.create(
            "projects",
            user["id"],
            {"title": title},
        ),
        request,
        201,
    )


@router.patch("/projects/{project_id}")
def update_project(project_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    own(st, "projects", project_id, user)
    if "title" not in body:
        raise APIError("PROJECT_INVALID", "รองรับเฉพาะการแก้ไขชื่อโปรเจกต์ภายในเครื่องเท่านั้น")
    title = str(body["title"]).strip()[:200]
    if not title:
        raise APIError("PROJECT_INVALID", "ชื่อโปรเจกต์ต้องไม่เป็นค่าว่าง")
    return ok(st.update("projects", project_id, {"title": title}, user["id"]), request)


@router.get("/models")
def models(request: Request, user=Depends(current_user), st=Depends(get_store)):
    _, registry = repo_registry(st, user)
    return ok(
        [
            {
                **r,
                "display_name": r.get("display_name", r.get("name", r["model_id"])),
                "reason": r.get("disabled_reason"),
            }
            for r in registry.models(user["id"])
        ],
        request,
    )


@router.post("/models/register")
def register_model(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    repo, _ = repo_registry(st, user)
    model_id = str(body.get("model_id", "")).strip()
    if (
        not model_id
        or len(model_id) > 100
        or any(x.get("model_id") == model_id for x in repo.list("models", user["id"]))
    ):
        raise APIError("MODEL_INVALID", "รหัสโมเดลไม่ถูกต้องหรือมีอยู่แล้ว")
    allowed = {
        "model_id",
        "display_name",
        "revision",
        "source",
        "license",
        "modality",
        "resource_profile",
        "variants",
    }
    payload = {k: v for k, v in body.items() if k in allowed}
    payload.update(
        enabled=False, is_mock=False, disabled_reason="รอผู้ดูแลระบบตรวจสอบแหล่งที่มา สัญญาอนุญาต และทดสอบความพร้อมบน GPU จริง"
    )
    return ok(repo.create("models", user["id"], payload), request, 201)


@router.get("/workflows")
def workflows(
    request: Request, model_id: str | None = None, user=Depends(current_user), st=Depends(get_store)
):
    _, registry = repo_registry(st, user)
    return ok(registry.workflows(user["id"], model_id), request)


@router.get("/workflows/{workflow_id}/schema")
def workflow_schema(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    manifest = row["manifest"]
    cap = min(manifest.get("max_runtime_minutes", 30), GpuConfig.from_env().max_runtime_minutes)
    spec = manifest.get("compute_spec")
    picked = mode_param(manifest)
    if picked:
        estimates = {m: estimate_units(repo, user["id"], row["workflow_key"], manifest, spec, cap, mode=m)[0] for m in picked[1]["variants"]}
        return ok({**manifest, "estimated_units": estimates[picked[1]["default"]], "estimates": estimates}, request)
    units, _ = estimate_units(repo, user["id"], row["workflow_key"], manifest, spec, cap)
    return ok({**manifest, "estimated_units": units}, request)


def _job_estimate(repo, owner, workflow, policy, cfg, params) -> dict:
    """Shown estimate is realistic; budget_units keeps reserving the max-runtime worst case."""
    if policy["provider"] == "mock":
        return {"estimated_units": 0.0, "unit_rate": cfg.unit_rate}
    cap = min(policy["max_runtime_minutes"], cfg.max_runtime_minutes)
    manifest = workflow["manifest"]
    picked = mode_param(manifest)
    mode = params.get(picked[0]) if picked else None
    units, rate = estimate_units(repo, owner, workflow["workflow_key"], manifest, manifest.get("compute_spec"), cap, mode=mode)
    units = round(min(units * length_factor(manifest, params), cap * rate), 3)
    return {"estimated_units": units, "unit_rate": rate, "budget_units": cap * cfg.unit_rate}


def cloud_sync_workflow(st, owner, row):
    if os.getenv("GPU_STORAGE", "local") != "drive":
        return row
    from .gpu.drive import DriveStorage

    storage = DriveStorage({"owner_id": owner}, Repository(st))
    folder = f"workflows/{row['workflow_key']}/{row['workflow_version']}"
    records = {}
    for name, field in [
        ("manifest.json", "manifest"),
        ("workflow.api.json", "api_graph"),
        ("workflow.editable.json", "editable_graph"),
        ("object_info.json", "object_info"),
    ]:
        content = json.dumps(row.get(field, {}), ensure_ascii=False, sort_keys=True).encode()
        record = storage.put(folder + "/" + name, content, "application/json")
        if not storage.verify(record, content):
            raise APIError("DRIVE_VERIFY_FAILED", "การตรวจสอบเวิร์กโฟลว์หลังบันทึกล้มเหลว")
        records[field] = record
    return Repository(st).update("workflows", row["id"], {"drive_files": records}, owner)


@router.post("/workflows/import")
def import_workflow(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    if len(json.dumps(body)) > 5 * 1024 * 1024:
        raise APIError("WORKFLOW_TOO_LARGE", "ไฟล์ JSON ของเวิร์กโฟลว์มีขนาดใหญ่เกินไป", 413)
    repo, registry = repo_registry(st, user)
    manifest = body.get("manifest") or {}
    body = {
        **body,
        "workflow_key": body.get("workflow_key", manifest.get("workflow_key")),
        "model_id": body.get("model_id", manifest.get("model_id")),
    }
    row = registry.import_workflow(user["id"], body)
    row = cloud_sync_workflow(st, user["id"], row)
    return ok(row, request, 201)


@router.post("/workflows/{workflow_id}/validate")
def validate_workflow(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    _, registry = repo_registry(st, user)
    return ok(registry.validate(user["id"], workflow_id), request)


@router.post("/workflows/{workflow_id}/smoke-test")
def smoke_workflow(workflow_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    repo, registry = repo_registry(st, user)
    row = own(repo, "workflows", workflow_id, user)
    model = next((m for m in registry.models(user["id"]) if m["model_id"] == row["model_id"]), None)
    if not model or not model.get("is_mock"):
        raise APIError(
            "REAL_SMOKE_REQUIRES_REVIEW",
            "เวิร์กโฟลว์จริงต้องผ่านการตรวจสอบแหล่งที่มาและรัน GPU Smoke Test ภายใต้การจำกัดงบประมาณก่อน ไม่สามารถส่งหลักฐานทดสอบปลอมได้",
        )
    validate_manifest(row["manifest"], row["api_graph"], row["object_info"])
    params = resolve_parameters(row["manifest"], {})
    graph = bind_parameters(row["manifest"], row["api_graph"], params)
    from .gpu.compute import MockGPUExecutor

    test_job = {
        "id": str(uuid.uuid4()),
        "provider": "mock",
        "snapshot": {"resolved_params": params, "bound_graph": graph},
        "execution_id": str(uuid.uuid4()),
    }
    executor = MockGPUExecutor()
    executor.start({"runtime_name": "mock-static-smoke"}, test_job, test_job["execution_id"])
    result = executor.poll({"runtime_name": "mock-static-smoke"}, test_job)
    if result.get("state") != "completed" or not result.get("outputs"):
        raise APIError("SMOKE_FAILED", "การทดสอบจำลอง (Simulation Smoke Test) ไม่ผ่าน")
    return ok(
        repo.update(
            "workflows", workflow_id, {"smoke_test_passed": True, "smoke_test_kind": "mock"}, user["id"]
        ),
        request,
    )


@router.post("/workflows/{workflow_id}/publish")
def publish_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    admin(user)
    _, registry = repo_registry(st, user)
    return ok(registry.publish(user["id"], workflow_id, confirmed=body.get("confirmed") is True), request)


@router.post("/workflows/rollback")
def rollback_workflow(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    _, registry = repo_registry(st, user)
    return ok(registry.rollback(user["id"], body.get("workflow_key"), body.get("version")), request)


@router.get("/presets")
def presets(
    request: Request,
    workflow_version_id: str | None = None,
    user=Depends(current_user),
    st=Depends(get_store),
):
    rows = Repository(st).list("presets", user["id"])
    return ok(
        [r for r in rows if not workflow_version_id or r.get("workflow_version_id") == workflow_version_id],
        request,
    )


@router.post("/presets")
def save_preset(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    workflow = own(repo, "workflows", body.get("workflow_version_id"), user)
    overrides = body.get("overrides", {})
    resolve_parameters(workflow["manifest"], overrides)
    row = repo.create(
        "presets",
        user["id"],
        {
            "name": str(body.get("name", "พรีเซ็ตไม่มีชื่อ"))[:100],
            "workflow_version_id": workflow["id"],
            "overrides": overrides,
        },
    )
    return ok(row, request, 201)


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = "mock"
    shutdown_after_job: bool = True
    max_runtime_minutes: int = Field(default=30, ge=1, le=1440)
    idle_grace_minutes: int = 0


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str
    model_id: str
    workflow_key: str
    workflow_version: int | str
    params: dict = Field(default_factory=dict)
    asset_ids: list[str] = Field(default_factory=list, max_length=10)
    loras: list[dict] = Field(default_factory=list, max_length=lora_registry.MAX_LORAS)
    preset_id: str | None = None
    execution_policy: Policy = Field(default_factory=Policy)


def admit_job(st, user, body, key):
    repo, registry = repo_registry(st, user)
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "ต้องระบุ Idempotency-Key และหากลองใหม่ต้องใช้ค่าเดิม")
    existing = next((j for j in repo.list("jobs", user["id"]) if j.get("idempotency_key") == key), None)
    if existing:
        if existing.get("body_hash") != digest(body):
            raise APIError("IDEMPOTENCY_CONFLICT", "Idempotency Key เดียวกันไม่สามารถใช้กับพารามิเตอร์ที่ต่างกันได้", 409)
        return existing
    project = own(st, "projects", body["project_id"], user)
    cfg = GpuConfig.from_env()
    policy = body["execution_policy"]
    if policy["provider"] not in {"mock", "colab"}:
        raise APIError("PROVIDER_INVALID", "อนุญาตเฉพาะ Provider mock หรือ colab ที่กำหนดค่าไว้เท่านั้น")
    if policy["provider"] == "colab" and not cfg.real_enabled:
        raise APIError("REAL_GPU_DISABLED", "GPU จริงถูกปิดไว้เป็นค่าเริ่มต้น: กรุณาตั้งค่า OAuth, งบประมาณ และผ่านการตรวจสอบโมเดลก่อน")
    if not policy["shutdown_after_job"] or policy["idle_grace_minutes"] != 0:
        raise APIError("POLICY_RESTRICTED", "เวอร์ชันปัจจุบันบังคับปล่อยคืน GPU ทันทีหลังจบงาน ไม่อนุญาตให้ขยายเวลา idle")
    workflow = registry.get_workflow(user["id"], body["workflow_key"], body["workflow_version"])
    if workflow.get("state") != "published" or workflow["model_id"] != body["model_id"]:
        raise APIError("WORKFLOW_UNPUBLISHED", "เวิร์กโฟลว์ยังไม่ได้เผยแพร่หรือไม่ตรงกับโมเดล")
    model = next((m for m in registry.models(user["id"]) if m["model_id"] == body["model_id"]), None)
    if not model or not model.get("enabled"):
        raise APIError("MODEL_UNAVAILABLE", "โมเดลยังไม่ผ่านการตรวจสอบสัญญาอนุญาต น้ำหนัก และฮาร์ดแวร์")
    if bool(model.get("is_mock")) != (policy["provider"] == "mock"):
        raise APIError("EXECUTOR_MISMATCH", "เทมเพลตจำลองใช้ได้เฉพาะ Mock โมเดลจริงไม่สามารถแกล้งจำลองสำเร็จได้")
    validate_manifest(workflow["manifest"], workflow["api_graph"], workflow["object_info"])
    preset = own(repo, "presets", body["preset_id"], user) if body.get("preset_id") else None
    if preset and preset["workflow_version_id"] != workflow["id"]:
        raise APIError("PRESET_INCOMPATIBLE", "พรีเซ็ตเป็นของเวิร์กโฟลว์คนละเวอร์ชัน กรุณาย้ายพรีเซ็ตก่อน")
    params = resolve_parameters(
        workflow["manifest"], body["params"], preset.get("overrides") if preset else None
    )
    bind_params, input_files = resolve_input_images(
        lambda aid: own(st, "assets", aid, user), project["id"], workflow["manifest"], params
    )
    bound = bind_parameters(workflow["manifest"], workflow["api_graph"], bind_params)
    for asset_id in body["asset_ids"]:
        asset = own(st, "assets", asset_id, user)
        if asset["project_id"] != project["id"]:
            raise APIError("ASSET_PROJECT_MISMATCH", "ไฟล์อ้างอิงไม่ได้เป็นของโปรเจกต์ปัจจุบัน")
    snapshot = {
        "manifest": workflow["manifest"],
        "api_graph": workflow["api_graph"],
        "object_info": workflow["object_info"],
        "resolved_params": params,
        "bound_graph": bound,
        "input_files": input_files,
        "workflow_sha256": workflow["sha256"],
        "model_revision": model.get("revision"),
        "dependency_lock": workflow["manifest"].get("dependencies", {}),
    }
    if policy["provider"] == "colab":
        if not workflow.get("review_approved") or workflow.get("smoke_test_kind") != "real":
            raise APIError("REVIEW_REQUIRED", "เวิร์กโฟลว์จริงยังไม่ผ่านการตรวจสอบหลังบ้านและการทดสอบความพร้อมจริง")
        snapshot["manifest"] = {**snapshot["manifest"], "state": "published", "reviewed": True}
        if cfg.unit_rate is None:
            raise APIError("RATE_UNKNOWN", "ยังไม่ได้ตั้งค่าอัตราค่าบริการประเมินที่เชื่อถือได้ ปฏิเสธการเปิดเครื่อง")
        from .gpu.google import GoogleService

        status = GoogleService(repo, user["id"]).status()
        if not status["colab"]["connected"]:
            raise APIError("GOOGLE_NOT_CONNECTED", "กรุณายืนยันสิทธิ์บัญชีส่วนตัวของ Google Colab ก่อน")
    # After the published stamp, so the curated LoRA weight files stay in the manifest the runtime restores.
    if body.get("loras"):
        _apply_loras(repo, user, workflow, body["loras"], snapshot, policy["provider"])
    # A worker cannot observe a queued record without its immutable snapshot.
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "คิวกำลังซิงค์ข้อมูล กรุณาลองใหม่ในภายหลังด้วย Idempotency Key เดิม", 503, True)
        job, created = repo.create_job(user["id"], body, key)
        if created:
            job = repo.update(
                "jobs",
                job["id"],
                {
                    "snapshot": snapshot,
                    "params": params,
                    "resolved_seed": params.get("seed"),
                    "max_runtime_minutes": min(policy["max_runtime_minutes"], cfg.max_runtime_minutes),
                    **_job_estimate(repo, user["id"], workflow, policy, cfg, params),
                },
                user["id"],
            )
            repo.event(job["id"], user["id"], "queued", "บันทึกงานลงฐานข้อมูลแล้ว การปิดมือถือจะไม่ทำให้คิวหลุด")
    return job


def _apply_loras(repo, user, workflow, picks, snapshot, provider):
    """Curated picks become extra weight files; own picks are Drive files the runtime fetches."""
    if workflow["workflow_key"] not in lora_registry.H3_WORKFLOWS:
        raise APIError("LORA_UNSUPPORTED", "เวิร์กโฟลว์นี้ไม่รองรับ Custom LoRA")
    curated = {c["id"]: c for c in lora_registry.CURATED if workflow["workflow_key"] in c["workflows"]}
    own_files = None
    chain, weights, user_files = [], [], []
    for pick in picks:
        ref, strength = str(pick.get("id") or ""), pick.get("strength", 0.8)
        if not isinstance(strength, int | float) or not -3.0 <= strength <= 3.0:
            raise APIError("VALIDATION_ERROR", "ความแรง LoRA ต้องอยู่ระหว่าง -3.0 ถึง 3.0")
        if ref in curated:
            weight = lora_registry.curated_weight(curated[ref])
            weights.append(weight)
            chain.append((weight["target"].removeprefix("loras/"), float(strength)))
            continue
        if own_files is None:
            own_files = {} if provider == "mock" else {f["id"]: f for f in _drive(repo, user).list_loras()}
        found = own_files.get(ref)
        if not found or not lora_registry.is_lora_file(found["name"], found["size"]):
            raise APIError("LORA_NOT_FOUND", "ไม่พบไฟล์ LoRA ในโฟลเดอร์ loras/h3 บน Drive หรือไฟล์ไม่ใช่ .safetensors ขนาดไม่เกิน 1.5GB")
        name = f"user/{found['id']}_{found['name']}"
        user_files.append({"id": found["id"], "name": found["name"], "size": found["size"], "md5Checksum": found["md5"], "target": name})
        chain.append((name, float(strength)))
    snapshot["bound_graph"] = lora_registry.insert_loras(snapshot["bound_graph"], chain)
    manifest = copy.deepcopy(snapshot["manifest"])
    manifest["weight_files"] = [*(manifest.get("weight_files") or []), *weights]
    allow = manifest.get("node_allowlist")
    if isinstance(allow, list) and "LoraLoaderModelOnly" not in allow:
        allow.append("LoraLoaderModelOnly")
    snapshot["manifest"] = manifest
    snapshot["lora_files"] = user_files
    snapshot["loras"] = [{"id": p.get("id"), "strength": p.get("strength", 0.8)} for p in picks]


def _drive(repo, user):
    from .gpu.drive import DriveStorage

    return DriveStorage({"owner_id": user["id"]}, repo)


@router.get("/loras")
def list_loras(request: Request, workflow_key: str | None = None, user=Depends(current_user), st=Depends(get_store)):
    # LoRAs always live in Drive (like the model cache), whatever GPU_STORAGE says about results.
    files, error = [], None
    try:
        files = _drive(Repository(st), user).list_loras()
    except Exception:  # noqa: BLE001 - Drive down must not hide the curated list
        error = "ยังไม่สามารถอ่านโฟลเดอร์ LoRA บน Drive ได้ในขณะนี้"
    return ok({"curated": lora_registry.curated_for(workflow_key), "files": files, "folder": "AI-Studio/โฟลเดอร์ของฉัน/loras/h3",
               "max_bytes": lora_registry.MAX_BYTES, "error": error}, request)


@router.post("/loras/upload")
def lora_upload(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    name, size = str(body.get("name") or "").strip(), body.get("size")
    if "/" in name or not lora_registry.is_lora_file(name, size if isinstance(size, int) else None):
        raise APIError("VALIDATION_ERROR", "รับเฉพาะไฟล์ .safetensors ขนาดไม่เกิน 1.5GB เท่านั้น")
    origin = request.headers.get("origin") or f"{request.url.scheme}://{request.url.netloc}"
    return ok({"upload_url": _drive(Repository(st), user).lora_upload_session(name, size, origin)}, request)


@router.post("/jobs", status_code=202)
def create_job(
    body: JobInput,
    request: Request,
    idempotency_key: str | None = Header(default=None),
    user=Depends(current_user),
    st=Depends(get_store),
):
    return ok(public_job(admit_job(st, user, body.model_dump(), idempotency_key)), request, 202)


@router.get("/jobs")
def jobs(request: Request, project_id: str | None = None, user=Depends(current_user), st=Depends(get_store)):
    if project_id:
        own(st, "projects", project_id, user)
    return ok([public_job(j) for j in Repository(st).list("jobs", user["id"], project_id)], request)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    job = own(repo, "jobs", job_id, user)
    result = public_job(job)
    result["events"] = [e for e in repo.list("events", user["id"]) if e.get("job_id") == job_id][::-1]
    result["timings"] = job_timings(job, result["events"])
    return ok(result, request)


@router.get("/jobs/{job_id}/events")
def job_events(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    repo = Repository(st)
    own(repo, "jobs", job_id, user)
    return ok([e for e in repo.list("events", user["id"]) if e.get("job_id") == job_id][::-1], request)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    own(Repository(st), "jobs", job_id, user)
    return ok(public_job(Scheduler(st).cancel(user["id"], job_id)), request)


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    original = own(Repository(st), "jobs", job_id, user)
    if original["state"] not in {"failed", "cancelled", "completed"}:
        raise APIError("JOB_ACTIVE", "งานเดิมยังไม่เสร็จสิ้น ไม่สามารถส่งซ้ำได้")
    key = request.headers.get("Idempotency-Key")
    if not key:
        raise APIError("IDEMPOTENCY_REQUIRED", "การสร้างใหม่อีกครั้งต้องระบุ Idempotency Key ใหม่")
    return ok(public_job(admit_job(st, user, original["request_body"], key)), request, 202)


@router.get("/jobs/{job_id}/outputs")
def outputs(job_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    job = own(Repository(st), "jobs", job_id, user)
    rows = [own(st, "assets", aid, user) for aid in job.get("output_asset_ids", [])]
    return ok([public_asset(a) for a in rows], request)


@router.get("/gallery")
def gallery(
    request: Request, project_id: str | None = None, user=Depends(current_user), st=Depends(get_store)
):
    if project_id:
        own(st, "projects", project_id, user)
    return ok(
        [
            public_asset(a)
            for a in st.list("assets", user["id"], project_id)
            if a.get("source_job_id") or a.get("gpu_job_id")
        ],
        request,
    )


@app.get("/api/assets")
def reference_assets(request: Request, project_id: str, user=Depends(current_user), st=Depends(get_store)):
    own(st, "projects", project_id, user)
    rows = [
        public_asset(a)
        for a in st.list("assets", user["id"], project_id)
        if not a.get("mock")
        and not (a.get("metadata") or {}).get("mock")
        and str(a.get("mime", "")).startswith(("image/", "video/", "audio/"))
    ]
    return ok(rows, request)


@app.post("/api/assets")
async def upload_asset(
    request: Request,
    project_id: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(current_user),
    st=Depends(get_store),
):
    own(st, "projects", project_id, user)
    content = await file.read(settings.max_upload_bytes + 1)
    return ok(
        public_asset(save_asset(st, user["id"], project_id, content, file.filename or "upload.bin")),
        request,
        201,
    )


@app.get("/api/assets/{asset_id}/download")
@app.get("/api/assets/{asset_id}/preview")
def download_asset(asset_id: str, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    return FileResponse(asset_path(asset), media_type=asset["mime"], filename=asset["filename"])


@app.get("/api/assets/{asset_id}/thumbnail")
def thumbnail(asset_id: str, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    if not asset["mime"].startswith("image/"):
        raise APIError("THUMBNAIL_UNAVAILABLE", "ไฟล์ที่ไม่ใช่รูปภาพไม่มีภาพตัวอย่างขนาดย่อ", 404)
    with Image.open(asset_path(asset)) as image:
        image.thumbnail((480, 480))
        target = io.BytesIO()
        image.convert("RGB").save(target, "WEBP", quality=80)
    return Response(target.getvalue(), media_type="image/webp")


@app.patch("/api/assets/{asset_id}")
def favorite_asset(
    asset_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):
    own(st, "assets", asset_id, user)
    if not isinstance(body.get("favorite"), bool):
        raise APIError("VALIDATION_ERROR", "favorite ต้องเป็นค่าบูลีน (boolean)")
    return ok(
        public_asset(st.update("assets", asset_id, {"favorite": body["favorite"]}, user["id"])), request
    )


@app.delete("/api/assets/{asset_id}")
def delete_asset(asset_id: str, request: Request, user=Depends(current_user), st=Depends(get_store)):
    asset = own(st, "assets", asset_id, user)
    delete_asset_file(asset)
    st.delete("assets", asset_id, user["id"])
    return ok({"deleted": True}, request)


@router.get("/usage")
def get_usage(request: Request, user=Depends(current_user), st=Depends(get_store)):
    return ok(usage(st, user["id"]), request)


@router.get("/settings")
def get_settings(request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    cfg = GpuConfig.from_env()
    return ok(
        {
            "provider_default": "mock",
            "real_enabled": cfg.real_enabled,
            "google": GoogleService(Repository(st), user["id"]).status(),
            "budget": usage(st, user["id"]),
            "storage": os.getenv("GPU_STORAGE", "local"),
            "independent": True,
        },
        request,
    )


@router.get("/compute/specs")
def compute_specs(
    request: Request, provider: str = "mock", user=Depends(current_user), st=Depends(get_store)
):
    if provider == "mock":
        return ok(MockColabProvider().list_specs(), request)
    if provider != "colab":
        raise APIError("PROVIDER_INVALID", "ไม่รองรับ Provider นี้")
    from .gpu.colab import GoogleColabProvider

    return ok(GoogleColabProvider(Repository(st), user["id"]).list_specs(), request)


@router.get("/compute/subscription")
def subscription(request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.colab import GoogleColabProvider

    return ok(GoogleColabProvider(Repository(st), user["id"]).subscription(), request)


@router.post("/admin/compute/stop")
def emergency_stop(request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    return ok(Scheduler(st).stop(user["id"]), request)


@router.post("/auth/google/start")
def google_start(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    result = GoogleService(Repository(st), user["id"]).start(body.get("purpose"))
    return ok({"authorization_url": result["authorization_url"]}, request)


@router.get("/auth/google/callback")
def google_callback(code: str, state: str, st=Depends(get_store)):
    from .gpu.google import GoogleService

    GoogleService(Repository(st), None).callback(code, state)
    return RedirectResponse("/?google=connected", status_code=303)


@router.post("/auth/google/disconnect")
def google_disconnect(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    from .gpu.google import GoogleService

    return ok(GoogleService(Repository(st), user["id"]).disconnect(body.get("purpose")), request)


# White-listed Agent API: it is not a shell and has no budget or compute-create tool.
AGENT_TOOLS = {
    "workflow.list",
    "workflow.import_draft",
    "workflow.validate",
    "model.list",
    "model.inspect",
    "model.register_draft",
    "job.create",
    "job.get",
    "job.cancel",
    "job.list_logs_redacted",
    "storage.list_project_files",
    "usage.inspect",
}


def agent_user(request):
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
    expected = os.getenv("GPU_AGENT_TOKEN", "")
    owner = os.getenv("GPU_AGENT_OWNER_ID", "")
    if expected and owner and supplied and secrets.compare_digest(supplied, expected):
        return {"id": owner, "role": "agent"}
    return current_user(request)


@router.get("/agent/tools")
def agent_tools(request: Request):
    agent_user(request)
    return ok(sorted(AGENT_TOOLS), request)


@router.post("/agent/invoke")
def agent_invoke(body: dict, request: Request, st=Depends(get_store)):
    user = agent_user(request)
    tool, args = body.get("tool"), body.get("arguments", {})
    if tool not in AGENT_TOOLS or not isinstance(args, dict):
        raise APIError("TOOL_FORBIDDEN", "Agent ไม่มีสิทธิ์เครื่องมือนี้ การจำกัดงบประมาณ การเผยแพร่ และการเปิดเครื่องโดยตรงไม่สามารถบายพาสได้", 403)
    repo, registry = repo_registry(st, user)
    if tool == "usage.inspect":
        result = usage(st, user["id"])
    elif tool in {"workflow.list", "model.list"}:
        result = registry.workflows(user["id"]) if tool == "workflow.list" else registry.models(user["id"])
    elif tool == "model.inspect":
        result = next((m for m in registry.models(user["id"]) if m["model_id"] == args.get("model_id")), None)
    elif tool == "model.register_draft":
        if not args.get("model_id"):
            raise APIError("MODEL_INVALID", "ขาดรหัสโมเดล (model ID)")
        result = repo.create(
            "models",
            user["id"],
            {
                k: v
                for k, v in args.items()
                if k in {"model_id", "display_name", "source", "revision", "license", "resource_profile"}
            }
            | {"enabled": False, "is_mock": False, "disabled_reason": "รอผู้ดูแลระบบตรวจสอบ"},
        )
    elif tool == "workflow.import_draft":
        result = registry.import_workflow(user["id"], args)
    elif tool == "workflow.validate":
        result = registry.validate(user["id"], args.get("id"))
    elif tool == "job.create":
        key = args.pop("idempotency_key", None)
        result = public_job(admit_job(st, user, JobInput(**args).model_dump(), key))
    elif tool in {"job.get", "job.cancel", "job.list_logs_redacted"}:
        job = own(repo, "jobs", args.get("id"), user)
        if tool == "job.cancel":
            job = Scheduler(st).cancel(user["id"], job["id"])
        result = (
            [e for e in repo.list("events", user["id"]) if e.get("job_id") == job["id"]]
            if tool.endswith("redacted")
            else public_job(job)
        )
    elif tool == "storage.list_project_files":
        own(st, "projects", args.get("project_id"), user)
        result = [public_asset(a) for a in st.list("assets", user["id"], args["project_id"])]
    else:
        raise APIError("TOOL_FORBIDDEN", "เครื่องมือไม่พร้อมใช้งาน", 403)
    return ok(result, request)


app.include_router(compute_router)
app.include_router(review_router)
app.include_router(oauth_config_router)


@app.exception_handler(GoogleError)
async def google_error(request, exc):
    return error_response(request, APIError("GOOGLE_REQUEST_FAILED", str(exc)[:250], 422))


app.include_router(router, prefix="/api/gpu")
# Canonical task-book endpoints are available too; provider switches keep /api/jobs unchanged.
app.include_router(router, prefix="/api", include_in_schema=False)

# Same-origin production PWA. The API always registers before the SPA fallback.

_static_root = Path(__file__).resolve().parents[2] / "web" / "dist"
if not _static_root.exists():
    _static_root = Path(__file__).resolve().parents[3] / "web" / "dist"
if _static_root.exists():
    if (_static_root / "assets").exists():
        app.mount("/assets", StaticFiles(directory=_static_root / "assets"), name="web-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise APIError("NOT_FOUND", "ไม่พบ Endpoint นี้", 404)
        target = (_static_root / path).resolve()
        if path and target.is_relative_to(_static_root.resolve()) and target.is_file():
            return FileResponse(target)
        return FileResponse(_static_root / "index.html")
