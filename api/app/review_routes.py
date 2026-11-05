"""Human review gates and the budgeted real-model smoke-test workflow.

This router is deliberately the only admission path for an unenabled real
model.  It records review evidence, creates only an immutable queued job, and
lets the Scheduler (never this module) perform provider side effects.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import time
from collections.abc import Mapping
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Request

from .common import APIError, admin, current_user, get_store, ok, own
from .gpu.colab import GoogleColabProvider
from .gpu.compute import GpuConfig
from .gpu.google import GoogleService
from .gpu.inputs import resolve_input_images
from .gpu.manifest import bind_parameters, digest, resolve_parameters, validate_manifest
from .gpu.prep import create_prep_job, prep_idempotency_key
from .gpu.rates import estimate as estimate_units
from .gpu.rates import mode_param
from .gpu.repository import Repository
from .gpu.runtime_worker import _reviewed_custom_nodes, _validate_weights
from .gpu.scheduler import Scheduler
from .storage import read_asset

router = APIRouter(prefix="/api/gpu")

_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_SOURCE_HOSTS = {"huggingface.co", "www.huggingface.co", "github.com", "www.github.com"}
_DANGEROUS = {
    "python",
    "exec",
    "eval",
    "shell",
    "subprocess",
    "command",
    "bash",
    "powershell",
    "httprequest",
    "download",
    "websocket",
}


def _confirmed(body: dict) -> None:
    if not isinstance(body, dict) or body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "การดำเนินการนี้ต้องระบุ confirmed=true อย่างชัดเจน", 403)


def _admin(user: dict) -> None:
    # Keep this explicit at every write boundary: agent tokens are never an
    # acceptable substitute for a human administrator review.
    admin(user)


def _fixed_commit(value, field: str) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise APIError("REVIEW_INVALID", f"{field}ต้องเป็น commit hash 40 หลักที่แน่นอน")
    return value.lower()


def _source(value) -> str:
    if not isinstance(value, str) or len(value) > 2048:
        raise APIError("MODEL_REVIEW_INVALID", "source ของโมเดลต้องเป็นแหล่งที่มา HTTPS ที่เชื่อถือได้")
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or host not in _SOURCE_HOSTS
        or parsed.username
        or parsed.password
        or not parsed.path.strip("/")
    ):
        raise APIError("MODEL_REVIEW_INVALID", "source ของโมเดลอนุญาตเฉพาะ HTTPS จาก huggingface.co หรือ github.com เท่านั้น")
    return value


def _resource_profile(value) -> dict:
    if not isinstance(value, Mapping) or not value:
        raise APIError("MODEL_REVIEW_INVALID", "resource_profile ต้องระบุตัวเลขทรัพยากร")
    numeric = 0

    def visit(item):
        nonlocal numeric
        if isinstance(item, bool):
            raise APIError("MODEL_REVIEW_INVALID", "ตัวเลขใน resource_profile ไม่สมเหตุสมผล")
        if isinstance(item, (int, float)):
            if not math.isfinite(float(item)) or float(item) <= 0 or float(item) > 65536:
                raise APIError("MODEL_REVIEW_INVALID", "ตัวเลขใน resource_profile ไม่สมเหตุสมผล")
            numeric += 1
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if not isinstance(item, str):
            raise APIError("MODEL_REVIEW_INVALID", "รูปแบบ resource_profile ไม่ถูกต้อง")

    visit(value)
    if not numeric:
        raise APIError("MODEL_REVIEW_INVALID", "resource_profile ต้องระบุตัวเลขทรัพยากร")
    return copy.deepcopy(dict(value))


def _model(repo: Repository, owner: str, model_id: str) -> dict:
    row = repo.get("models", model_id, owner)
    if row:
        return row
    row = next((item for item in repo.list("models", owner) if item.get("model_id") == model_id), None)
    if not row:
        raise APIError("MODEL_NOT_FOUND", "ไม่พบโมเดลที่ผูกกับเวิร์กโฟลว์นี้", 404)
    return row


def _reviewed_model(model: dict) -> None:
    if not (model.get("review_approved") is True and model.get("license_accepted") is True):
        raise APIError("MODEL_NOT_REVIEWED", "โมเดลยังไม่ผ่านการตรวจสอบแหล่งที่มาและสัญญาอนุญาตจากผู้ดูแลระบบ")


def _review_manifest(row: dict) -> tuple[dict, list[str], str, list[dict]]:
    try:
        manifest = validate_manifest(row.get("manifest"), row.get("api_graph"), row.get("object_info"))
    except (TypeError, ValueError) as exc:
        raise APIError("WORKFLOW_INVALID", str(exc)[:300]) from exc

    dependencies = manifest.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise APIError("REVIEW_INVALID", "manifest.dependencies ขาดหายไป")
    commit = _fixed_commit(dependencies.get("comfyui_commit"), "dependencies.comfyui_commit")
    try:
        _reviewed_custom_nodes(dependencies.get("custom_nodes"))
    except Exception as exc:
        raise APIError("REVIEW_INVALID", "custom_nodes อนุญาตเฉพาะคลังที่ผ่านการตรวจสอบและล็อก commit แล้วเท่านั้น") from exc
    weights = dependencies.get("weight_files")
    try:
        checked = _validate_weights({"weight_files": weights})
    except Exception as exc:
        raise APIError("REVIEW_INVALID", str(exc)[:300]) from exc
    for item in checked:
        target = item.get("target")
        if not isinstance(target, str) or not target or "\\" in target:
            raise APIError("REVIEW_INVALID", "เส้นทาง weight target ไม่ปลอดภัย")

    classes = set()
    for node in row.get("api_graph", {}).values():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str):
            raise APIError("WORKFLOW_INVALID", "ประเภทโหนดใน api_graph ไม่ถูกต้อง")
        class_type = node["class_type"]
        compact = class_type.lower().replace("_", "")
        if any(token in compact for token in _DANGEROUS):
            raise APIError("REVIEW_INVALID", f"ห้ามใช้ประเภทโหนดที่เป็นอันตราย: {class_type}")
        classes.add(class_type)
    if not classes:
        raise APIError("WORKFLOW_INVALID", "api_graph ต้องไม่เป็นค่าว่าง")
    return manifest, sorted(classes), commit, checked


def _reviewed_snapshot(row: dict) -> dict:
    manifest, allowlist, commit, weights = _review_manifest(row)
    snapshot_manifest = copy.deepcopy(manifest)
    # These top-level fields are the adapter contract.  Keep both commit names
    # during the transition so runtime_worker and older Comfy adapters agree.
    snapshot_manifest.update(
        {
            "state": "published",
            "reviewed": True,
            "node_allowlist": allowlist,
            "comfyui_commit": commit,
            "comfy_commit": commit,
            "custom_nodes": list(manifest["dependencies"].get("custom_nodes") or []),
            "weight_files": weights,
        }
    )
    return {
        "manifest": snapshot_manifest,
        "api_graph": copy.deepcopy(row["api_graph"]),
        "object_info": copy.deepcopy(row["object_info"]),
        "node_allowlist": allowlist,
        "comfyui_commit": commit,
        "weight_files": weights,
    }


def _workflow_digest(row: dict) -> str:
    """Recompute the digest from the stored artifact, never from client input."""
    try:
        return digest(
            {"manifest": row["manifest"], "api_graph": row["api_graph"], "object_info": row["object_info"]}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise APIError("WORKFLOW_INVALID", "artifact ของเวิร์กโฟลว์ไม่สมบูรณ์ ไม่สามารถคำนวณ digest ได้") from exc


def _job_request(workflow_id: str, row: dict, body: Mapping) -> dict:
    """Build the exact persisted idempotency body without external validation."""
    return {
        "workflow_id": workflow_id,
        "project_id": body.get("project_id"),
        "model_id": row.get("model_id"),
        "workflow_key": row.get("workflow_key"),
        "workflow_version": row.get("workflow_version"),
        "params": copy.deepcopy(body.get("params", {})),
        "asset_ids": copy.deepcopy(body.get("asset_ids", [])),
        "spec": copy.deepcopy(body.get("spec")),
        "execution_policy": {
            "provider": "colab",
            "shutdown_after_job": True,
            "max_runtime_minutes": 5,
            "idle_grace_minutes": 0,
        },
    }


def _idempotency_key(request: Request) -> str:
    key = request.headers.get("Idempotency-Key")
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "การทดสอบความพร้อมจริงต้องระบุ Idempotency-Key ที่แน่นอน")
    return key


def _spec_key(value):
    if not isinstance(value, Mapping):
        return None
    key = value.get("key")
    return key if isinstance(key, Mapping) else None


def _eligible_spec(specs, requested_key):
    if not isinstance(specs, list):
        raise APIError("GPU_UNAVAILABLE", "การตอบกลับสเปกของ Colab ไม่ถูกต้อง")
    for spec in specs:
        if not isinstance(spec, Mapping) or spec.get("eligible") is not True:
            continue
        key = spec.get("key")
        if not isinstance(key, Mapping) or key.get("variant") not in {"GPU", "VARIANT_GPU"}:
            continue
        if dict(key) == dict(requested_key):
            return copy.deepcopy(dict(spec))
    raise APIError("GPU_UNAVAILABLE", "สเปก GPU ที่ร้องขอไม่พร้อมใช้งานจริงในบัญชีปัจจุบัน")


def _budget_check(st, owner: str, cfg: GpuConfig, units: float) -> None:
    if not math.isfinite(units) or units <= 0:
        raise APIError("RATE_UNKNOWN", "อัตราค่าบริการหรือฟิลด์งบประมาณของ GPU จริงไม่ถูกต้อง")
    if units > cfg.max_job_units:
        raise APIError("GPU_BUDGET_EXCEEDED", "งบประมาณการทดสอบจริงเกินขีดจำกัดต่อหนึ่งงาน")
    daily, monthly = Scheduler(st, config=cfg)._usage_totals(owner)
    if daily + units > cfg.daily_units or monthly + units > cfg.monthly_units:
        raise APIError("GPU_BUDGET_EXCEEDED", "งบประมาณการทดสอบจริงเกินขีดจำกัดรายวัน/รายเดือนของบัญชี")


@router.post("/models/{model_id}/review")
def review_model(
    model_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    _admin(user)
    _confirmed(body)
    if body.get("license_accepted") is not True:
        raise APIError("LICENSE_CONFIRMATION_REQUIRED", "ต้องยอมรับสัญญาอนุญาตของโมเดลอย่างชัดเจน", 403)
    repo = Repository(st)
    model = _model(repo, user["id"], model_id)
    _source(model.get("source"))
    _fixed_commit(model.get("revision"), "revision")
    license_name = model.get("license")
    if not isinstance(license_name, str) or not license_name.strip():
        raise APIError("MODEL_REVIEW_INVALID", "สัญญาอนุญาต (license) ของโมเดลต้องไม่เป็นค่าว่าง")
    _resource_profile(model.get("resource_profile"))
    reviewed = repo.update(
        "models",
        model["id"],
        {
            "review_approved": True,
            "license_accepted": True,
            "license_review_approved": True,
            "reviewed_by": user["id"],
            "reviewed_at": time.time(),
            # Review is not enablement.  Only accept-smoke may flip this bit.
            "enabled": False,
            "is_mock": False,
            "disabled_reason": "ผ่านการตรวจสอบโดยผู้ดูแลแล้ว รอการตรวจรับการทดสอบบน GPU จริงที่ได้รับอนุญาต",
        },
        user["id"],
    )
    return ok(reviewed, request)


@router.post("/workflows/{workflow_id}/review")
def review_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    _admin(user)
    _confirmed(body)
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("state") not in {"draft", "validated"}:
        raise APIError("WORKFLOW_REVIEW_INVALID", "สามารถตรวจสอบได้เฉพาะเวิร์กโฟลว์ที่อยู่ในสถานะ draft หรือ validated เท่านั้น")
    model = _model(repo, user["id"], row.get("model_id"))
    _reviewed_model(model)
    manifest, allowlist, commit, weights = _review_manifest(row)
    reviewed_manifest = copy.deepcopy(manifest)
    reviewed_manifest.update(
        {
            "node_allowlist": allowlist,
            "comfyui_commit": commit,
            "comfy_commit": commit,
            "custom_nodes": list(manifest["dependencies"].get("custom_nodes") or []),
            "weight_files": weights,
        }
    )
    reviewed = repo.update(
        "workflows",
        workflow_id,
        {
            "manifest": reviewed_manifest,
            "sha256": digest(
                {
                    "manifest": reviewed_manifest,
                    "api_graph": row["api_graph"],
                    "object_info": row["object_info"],
                }
            ),
            "review_approved": True,
            "reviewed_by": user["id"],
            "reviewed_at": time.time(),
        },
        user["id"],
    )
    return ok(reviewed, request)


@router.post("/workflows/{workflow_id}/real-smoke")
def real_smoke(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    _admin(user)
    _confirmed(body)
    if (
        not isinstance(body, dict)
        or not isinstance(body.get("project_id"), str)
        or not isinstance(body.get("params", {}), dict)
    ):
        raise APIError("VALIDATION_ERROR", "รูปแบบ project_id และ params ไม่ถูกต้อง")
    asset_ids = body.get("asset_ids", [])
    if (
        not isinstance(asset_ids, list)
        or len(asset_ids) > 10
        or any(not isinstance(value, str) for value in asset_ids)
    ):
        raise APIError("VALIDATION_ERROR", "รูปแบบ asset_ids ไม่ถูกต้อง")
    requested_key = _spec_key(body.get("spec"))
    if requested_key is None:
        raise APIError("GPU_SPEC_REQUIRED", "ต้องระบุ spec.key จากบัญชีปัจจุบัน")
    key = _idempotency_key(request)
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("state") not in {"draft", "validated"}:
        raise APIError("WORKFLOW_REVIEW_INVALID", "การทดสอบความพร้อมจริงใช้ได้เฉพาะกับเวิร์กโฟลว์ draft หรือ validated ที่ยังไม่เผยแพร่เท่านั้น")
    model = _model(repo, user["id"], row.get("model_id"))
    _reviewed_model(model)
    snapshot_parts = _reviewed_snapshot(row) if row.get("review_approved") is True else None
    if snapshot_parts is None:
        raise APIError("WORKFLOW_NOT_REVIEWED", "เวิร์กโฟลว์ยังไม่ผ่านการตรวจสอบจากผู้ดูแลระบบ")
    try:
        params = resolve_parameters(row["manifest"], body.get("params", {}))
    except (TypeError, ValueError) as exc:
        raise APIError("WORKFLOW_INVALID", str(exc)[:300]) from exc
    project = own(st, "projects", body["project_id"], user)
    bind_params, input_files = resolve_input_images(
        lambda aid: own(st, "assets", aid, user), project["id"], row["manifest"], params
    )
    try:
        bound = bind_parameters(row["manifest"], row["api_graph"], bind_params)
    except (TypeError, ValueError) as exc:
        raise APIError("WORKFLOW_INVALID", str(exc)[:300]) from exc
    for asset_id in asset_ids:
        asset = own(st, "assets", asset_id, user)
        if asset.get("project_id") != project["id"]:
            raise APIError("ASSET_PROJECT_MISMATCH", "ไฟล์อ้างอิงไม่ได้เป็นของโปรเจกต์ปัจจุบัน")

    raw = {
        "project_id": project["id"],
        "model_id": row["model_id"],
        "workflow_key": row["workflow_key"],
        "workflow_version": row["workflow_version"],
        "params": copy.deepcopy(body.get("params", {})),
        "asset_ids": copy.deepcopy(asset_ids),
        "loras": copy.deepcopy(body.get("loras") or []),
        "spec": copy.deepcopy(body["spec"]),
        "execution_policy": {
            "provider": "colab",
            "shutdown_after_job": True,
            "max_runtime_minutes": 5,
            "idle_grace_minutes": 0,
        },
    }
    previous = next(
        (item for item in repo.list("jobs", user["id"]) if item.get("idempotency_key") == key), None
    )
    if previous:
        if previous.get("body_hash") != digest(raw):
            raise APIError("IDEMPOTENCY_CONFLICT", "Idempotency Key เดียวกันไม่สามารถใช้กับพารามิเตอร์ทดสอบที่แตกต่างกันได้", 409)
        return ok(
            {
                "id": previous["id"],
                "job_id": previous["id"],
                "state": previous["state"],
                "estimated_units": previous.get("estimated_units"),
            },
            request,
            202,
        )

    cfg = GpuConfig.from_env()
    if not cfg.real_enabled:
        raise APIError("REAL_GPU_DISABLED", "GPU_REAL_ENABLED ยังไม่ได้เปิดใช้งาน")
    if cfg.unit_rate is None or cfg.unit_rate <= 0:
        raise APIError("RATE_UNKNOWN", "ต้องกำหนดอัตราค่าบริการประเมินต่อนาทีของ GPU จริงก่อน")
    google = GoogleService(repo, user["id"])
    if not google.status().get("colab", {}).get("connected"):
        raise APIError("GOOGLE_NOT_CONNECTED", "กรุณายืนยันสิทธิ์ Colab OAuth ส่วนตัวก่อน")
    chosen_spec = _eligible_spec(GoogleColabProvider(repo, user["id"]).list_specs(), requested_key)
    # โมเดลใหญ่ต้องดาวน์โหลดน้ำหนักก่อน 5 นาทีอาจไม่พอ ให้กำหนดผ่าน manifest.max_runtime_minutes โดยยังอยู่ภายใต้ขีดจำกัดส่วนกลาง
    declared = row["manifest"].get("max_runtime_minutes", 5)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise APIError("WORKFLOW_INVALID", "manifest.max_runtime_minutes ต้องเป็นจำนวนเต็มบวก")
    runtime = min(declared, cfg.max_runtime_minutes)
    estimated = runtime * cfg.unit_rate  # budget worst case
    picked = mode_param(row["manifest"])
    shown, rate = estimate_units(repo, user["id"], row["workflow_key"], row["manifest"], chosen_spec.get("key"), runtime,
                                 mode=params.get(picked[0]) if picked else None)
    # Check once before the lock for a fast response and again inside the lock
    # so concurrent admissions cannot reserve the same budget twice.
    _budget_check(st, user["id"], cfg, estimated)
    snapshot = {
        **snapshot_parts,
        "bound_graph": bound,
        "input_files": input_files,
        "resolved_params": params,
        "workflow_sha256": row.get("sha256"),
        "model_revision": model.get("revision"),
        "dependency_lock": {
            "comfyui_commit": snapshot_parts["comfyui_commit"],
            "weight_files": snapshot_parts["weight_files"],
        },
        "workflow_smoke": True,
        "system_smoke": False,
        "target_workflow_id": workflow_id,
        "target_model_id": row["model_id"],
    }
    loras = body.get("loras") or []
    if loras:
        # Smoke runs may only use the curated (pinned, SFW) list, never owner-uploaded files.
        from .gpu import loras as lora_registry
        from .main import _apply_loras

        curated_ids = {c["id"] for c in lora_registry.CURATED if row["workflow_key"] in c["workflows"]}
        if not isinstance(loras, list) or len(loras) > 3 or any(
            not isinstance(p, dict) or p.get("id") not in curated_ids for p in loras
        ):
            raise APIError("VALIDATION_ERROR", "การทดสอบความพร้อมอนุญาตให้ใช้เฉพาะ LoRA ที่คัดสรรไว้เท่านั้น")
        _apply_loras(repo, user, row, loras, snapshot, "colab")
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "ระบบจัดคิวกำลังซิงค์ข้อมูล กรุณาลองใหม่ในภายหลังด้วย Idempotency Key เดิม", 503, True)
        _budget_check(st, user["id"], cfg, estimated)
        job, created = repo.create_job(user["id"], raw, key)
        if created:
            job = repo.update(
                "jobs",
                job["id"],
                {
                    "project_id": project["id"],
                    "provider": "colab",
                    "model_id": row["model_id"],
                    "workflow_key": row["workflow_key"],
                    "workflow_version": row["workflow_version"],
                    "target_workflow_id": workflow_id,
                    "target_model_id": row["model_id"],
                    "snapshot": snapshot,
                    "params": copy.deepcopy(body.get("params", {})),
                    "resolved_seed": params.get("seed"),
                    "requested_spec": chosen_spec,
                    "max_runtime_minutes": runtime,
                    "estimated_units": shown,
                    "unit_rate": rate,
                    "budget_units": estimated,
                    "workflow_smoke": True,
                    "system_smoke": False,
                },
                user["id"],
            )
            repo.event(
                job["id"],
                user["id"],
                "queued",
                "ผ่านการตรวจสอบโดยผู้ดูแลแล้วและเข้าคิวทดสอบบนโมเดลจริง โดยจำกัดเวลาสูงสุดตามที่ระบุใน manifest และจะปล่อยคืน GPU ทันทีเมื่อเสร็จสิ้น",
            )
    return ok(
        {
            "id": job["id"],
            "job_id": job["id"],
            "state": job["state"],
            "estimated_units": job.get("estimated_units"),
        },
        request,
        202,
    )


@router.post("/workflows/{workflow_id}/prepare")
def prepare_workflow(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    """Queue a CPU-only job that builds the env archive and fills the weight cache in Drive."""
    _admin(user)
    _confirmed(body)
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("review_approved") is not True:
        raise APIError("WORKFLOW_NOT_REVIEWED", "เวิร์กโฟลว์ยังไม่ผ่านการตรวจสอบจากผู้ดูแลระบบ")
    cfg = GpuConfig.from_env()
    if not cfg.real_enabled:
        raise APIError("REAL_GPU_DISABLED", "GPU_REAL_ENABLED ยังไม่ได้เปิดใช้งาน")
    if not GoogleService(repo, user["id"]).status().get("colab", {}).get("connected"):
        raise APIError("GOOGLE_NOT_CONNECTED", "กรุณายืนยันสิทธิ์ Colab OAuth ส่วนตัวก่อน")
    project_id = body.get("project_id")
    if not isinstance(project_id, str):
        raise APIError("VALIDATION_ERROR", "จำเป็นต้องระบุ project_id")
    own(st, "projects", project_id, user)
    meta = {
        "workflow_sha256": row.get("sha256"),
        "model_id": row["model_id"],
        "workflow_key": row["workflow_key"],
        "workflow_version": row["workflow_version"],
    }
    key = prep_idempotency_key(row.get("sha256") or workflow_id, _idempotency_key(request))
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "ระบบจัดคิวกำลังซิงค์ข้อมูล กรุณาลองใหม่ในภายหลังด้วย Idempotency Key เดิม", 503, True)
        job = create_prep_job(repo, user["id"], project_id, _reviewed_snapshot(row), meta, key)
    return ok({"id": job["id"], "job_id": job["id"], "state": job["state"]}, request, 202)


@router.post("/loras/prepare")
def prepare_loras(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):  # noqa: B008
    """CPU-only: copy the curated H3 LoRAs from Hugging Face into the Drive model cache."""
    from .gpu import loras as lora_registry

    _admin(user)
    _confirmed(body)
    repo = Repository(st)
    rows = [w for w in repo.list("workflows", user["id"]) if w.get("workflow_key") in lora_registry.H3_WORKFLOWS
            and w.get("state") == "published" and w.get("review_approved") is True]
    if not rows:
        raise APIError("WORKFLOW_NOT_REVIEWED", "ไม่มีเวิร์กโฟลว์ H3 ที่เผยแพร่แล้ว")
    project_id = body.get("project_id")
    if not isinstance(project_id, str):
        raise APIError("VALIDATION_ERROR", "จำเป็นต้องระบุ project_id")
    own(st, "projects", project_id, user)
    row = rows[0]
    snapshot = _reviewed_snapshot(row)
    curated = [lora_registry.curated_weight(c) for c in lora_registry.CURATED]
    snapshot["manifest"] = {**snapshot["manifest"], "weight_files": curated}
    snapshot["weight_files"] = curated
    meta = {"workflow_sha256": row.get("sha256"), "model_id": row["model_id"], "workflow_key": row["workflow_key"],
            "workflow_version": row["workflow_version"]}
    key = prep_idempotency_key("curated-loras", _idempotency_key(request))
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "ระบบจัดคิวกำลังซิงค์ข้อมูล กรุณาลองใหม่ในภายหลังด้วย Idempotency Key เดิม", 503, True)
        job = create_prep_job(repo, user["id"], project_id, snapshot, meta, key)
    return ok({"id": job["id"], "job_id": job["id"], "state": job["state"]}, request, 202)


def _job_gpu_diagnostics(job: dict, leases: list[dict]) -> dict | None:
    # Only server-persisted scheduler/provider evidence is accepted.  Never use
    # a diagnostic value from the request body or the accept endpoint body.
    candidates = [job.get(name) for name in ("gpu_diagnostics", "gpu_diagnostic", "runtime_diagnostics")]
    snapshot = job.get("snapshot") or {}
    candidates.extend(
        snapshot.get(name) for name in ("gpu_diagnostics", "gpu_diagnostic", "runtime_diagnostics")
    )
    for lease in leases:
        candidates.extend(
            lease.get(name)
            for name in ("gpu_diagnostics", "gpu_diagnostic", "runtime_diagnostics", "selected_spec")
        )
    for value in candidates:
        if isinstance(value, Mapping) and value:
            return copy.deepcopy(dict(value))
    return None


@router.post("/workflows/{workflow_id}/accept-smoke")
def accept_smoke(
    workflow_id: str, body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)
):  # noqa: B008
    _admin(user)
    _confirmed(body)
    if (
        not isinstance(body, dict)
        or set(body) - {"job_id", "confirmed"}
        or not isinstance(body.get("job_id"), str)
    ):
        raise APIError("INVALID_SMOKE_EVIDENCE", "ส่งได้เฉพาะ job_id ที่เซิร์ฟเวอร์สร้างขึ้นเท่านั้น ไม่สามารถส่งหลักฐานทดสอบจากฝั่งไคลเอนต์ได้")
    repo = Repository(st)
    row = own(repo, "workflows", workflow_id, user)
    if row.get("state") not in {"draft", "validated"}:
        raise APIError("WORKFLOW_ALREADY_PUBLISHED", "เวอร์ชันที่เผยแพร่แล้วไม่สามารถเขียนทับผลการตรวจรับการทดสอบได้")
    model = _model(repo, user["id"], row.get("model_id"))
    _reviewed_model(model)
    if row.get("review_approved") is not True:
        raise APIError("WORKFLOW_NOT_REVIEWED", "เวิร์กโฟลว์ยังไม่ผ่านการตรวจสอบจากผู้ดูแลระบบ")
    job = own(repo, "jobs", body["job_id"], user)
    snapshot = job.get("snapshot") or {}
    if (
        job.get("state") != "completed"
        or job.get("provider") != "colab"
        or snapshot.get("workflow_smoke") is not True
    ):
        raise APIError("SMOKE_NOT_COMPLETE", "สามารถตรวจรับได้เฉพาะงานทดสอบเวิร์กโฟลว์จริงบนเซิร์ฟเวอร์ที่เสร็จสมบูรณ์แล้วเท่านั้น")
    if job.get("target_workflow_id") != workflow_id or job.get("target_model_id") != row.get("model_id"):
        raise APIError("SMOKE_TARGET_MISMATCH", "เป้าหมายของงานทดสอบไม่ตรงกับเวิร์กโฟลว์/โมเดลปัจจุบัน")
    output_ids = job.get("output_asset_ids")
    if not isinstance(output_ids, list) or not output_ids:
        raise APIError("SMOKE_OUTPUT_INVALID", "การทดสอบความพร้อมจริงไม่มีผลลัพธ์ที่สามารถตรวจรับได้")
    for asset_id in output_ids:
        asset = own(st, "assets", asset_id, user)
        try:
            content = read_asset(asset)
        except (OSError, ValueError) as exc:
            raise APIError("SMOKE_OUTPUT_INVALID", "ไม่สามารถอ่านไฟล์ผลลัพธ์ของการทดสอบได้") from exc
        if asset.get("size") != len(content) or asset.get("sha256") != hashlib.sha256(content).hexdigest():
            raise APIError("SMOKE_OUTPUT_INVALID", "การตรวจสอบ SHA ของผลลัพธ์การทดสอบล้มเหลว")
    leases = [lease for lease in repo.list("leases", user["id"]) if lease.get("owner_job_id") == job["id"]]
    if not leases or any(lease.get("state") != "deleted" for lease in leases):
        raise APIError("GPU_NOT_RELEASED", "สัญญาเช่า GPU ยังไม่ได้รับการยืนยันการปล่อยคืนจากเซิร์ฟเวอร์")
    diagnostics = _job_gpu_diagnostics(job, leases)
    requested = job.get("requested_spec")
    if not isinstance(requested, Mapping) or diagnostics is None:
        raise APIError("GPU_EVIDENCE_MISSING", "ขาดหลักฐานการวินิจฉัย GPU จริงที่บันทึกโดยเซิร์ฟเวอร์")
    tested = {"requested_spec": copy.deepcopy(dict(requested)), "actual_gpu": diagnostics}
    updated_workflow = repo.update(
        "workflows",
        workflow_id,
        {
            "smoke_test_passed": True,
            "smoke_test_kind": "real",
            "smoke_job_id": job["id"],
        },
        user["id"],
    )
    repo.update(
        "models",
        model["id"],
        {
            "enabled": True,
            "disabled_reason": None,
            "tested_compute_specs": tested,
            "enabled_at": time.time(),
        },
        user["id"],
    )
    return ok(updated_workflow, request)


__all__ = ["accept_smoke", "real_smoke", "review_model", "review_workflow", "router"]
