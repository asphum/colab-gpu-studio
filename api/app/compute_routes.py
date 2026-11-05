"""A bounded CUDA-only feasibility job, governed by the same scheduler and budget."""

from fastapi import APIRouter, Depends, Request

from .common import APIError, admin, current_user, get_store, ok
from .gpu.colab import GoogleColabProvider
from .gpu.compute import GpuConfig
from .gpu.google import GoogleService
from .gpu.manifest import digest
from .gpu.repository import Repository

router = APIRouter(prefix="/api/gpu")


@router.post("/admin/compute/smoke")
def cuda_smoke(body: dict, request: Request, user=Depends(current_user), st=Depends(get_store)):
    admin(user)
    cfg = GpuConfig.from_env()
    if body.get("confirmed") is not True:
        raise APIError("CONFIRMATION_REQUIRED", "การทดสอบนี้จะใช้คะแนนจริง ต้องได้รับการยืนยันอย่างชัดเจน", 403)
    if not cfg.real_enabled:
        raise APIError("REAL_GPU_DISABLED", "สวิตช์ GPU จริงยังไม่ได้เปิดใช้งาน")
    if cfg.unit_rate is None or cfg.unit_rate <= 0:
        raise APIError("RATE_UNKNOWN", "ต้องระบุอัตราค่าบริการประเมินต่อนาทีและงบประมาณก่อน")
    key = request.headers.get("Idempotency-Key")
    if not key or len(key) > 200:
        raise APIError("IDEMPOTENCY_REQUIRED", "ต้องระบุ Idempotency Key ที่แน่นอน หากลองใหม่ให้ใช้ค่าเดิม")
    repo = Repository(st)
    previous = next((j for j in repo.list("jobs", user["id"]) if j.get("idempotency_key") == key), None)
    raw = {
        "kind": "system-cuda-smoke",
        "requested_spec": body.get("spec"),
        "max_runtime_minutes": min(5, cfg.max_runtime_minutes),
    }
    if previous:
        if previous.get("body_hash") != digest(raw):
            raise APIError("IDEMPOTENCY_CONFLICT", "Idempotency Key นี้ถูกใช้กับพารามิเตอร์อื่นไปแล้ว", 409)
        return ok({"id": previous["id"], "job_id": previous["id"], "state": previous["state"]}, request, 202)
    google = GoogleService(repo, user["id"])
    if not google.status()["colab"]["connected"]:
        raise APIError("GOOGLE_NOT_CONNECTED", "กรุณายืนยันสิทธิ์ Colab OAuth ส่วนตัวก่อน")
    specs = GoogleColabProvider(repo, user["id"]).list_specs()
    eligible = [
        s
        for s in specs
        if s.get("eligible") is True and (s.get("key") or {}).get("variant") in {"GPU", "VARIANT_GPU"}
    ]
    requested = body.get("spec")
    if requested:
        eligible = [s for s in eligible if s.get("key") == requested.get("key", requested)]
    if not eligible:
        raise APIError("GPU_UNAVAILABLE", "บัญชีปัจจุบันไม่มีสเปก GPU ที่พร้อมใช้งานตรงตามที่ร้องขอ")
    project_id = "system-smoke-" + user["id"][:40]
    with st.dispatch_lock() as acquired:
        if not acquired:
            raise APIError("SCHEDULER_BUSY", "ระบบจัดคิวกำลังซิงค์ข้อมูล กรุณาลองใหม่ในภายหลังด้วย Idempotency Key เดิม", 503, True)
        if not st.get("projects", project_id, user["id"]):
            st.create(
                "projects", user["id"], {"title": "การตรวจสอบความพร้อมของ GPU", "system": True}, record_id=project_id
            )
        job, created = repo.create_job(user["id"], raw, key)
        if created:
            snapshot = {
                "system_smoke": True,
                "manifest": {
                    "executor": "cuda-smoke",
                    "parameters": {},
                    "state": "published",
                    "reviewed": True,
                },
                "resolved_params": {},
                "api_graph": {},
                "bound_graph": {},
                "model_revision": None,
                "workflow_sha256": digest({"system": "cuda-smoke", "version": 1}),
                "dependency_lock": {},
            }
            job = repo.update(
                "jobs",
                job["id"],
                {
                    "project_id": project_id,
                    "provider": "colab",
                    "model_id": "system-cuda-smoke",
                    "workflow_key": "system/cuda-smoke",
                    "workflow_version": 1,
                    "snapshot": snapshot,
                    "requested_spec": eligible[0],
                    "max_runtime_minutes": raw["max_runtime_minutes"],
                    "estimated_units": raw["max_runtime_minutes"] * cfg.unit_rate,
                    "unit_rate": cfg.unit_rate,
                },
                user["id"],
            )
            repo.event(
                job["id"],
                user["id"],
                "queued",
                f"ยืนยันการทดสอบ CUDA Smoke จริงแล้ว: ทดสอบเฉพาะ GPU สูงสุด {raw['max_runtime_minutes']} นาที และจะปล่อยคืน runtime ทันทีเมื่อเสร็จสิ้น",
            )
    return ok(
        {
            "id": job["id"],
            "job_id": job["id"],
            "state": job["state"],
            "estimated_units": job["estimated_units"],
        },
        request,
        202,
    )
