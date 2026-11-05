"""CPU cache-prep jobs: fill the Drive env/weight cache on a cheap CPU runtime so GPU jobs start warm."""

from __future__ import annotations

import copy
import os

from .compute import GpuConfig
from .manifest import digest

# Standard CPU runtime (~0.07 CU/h); the prep job never needs a GPU.
PREP_SPEC_KEY = {"variant": "VARIANT_CPU", "accelerator": "NONE", "shape": "SHAPE_STANDARD"}
PREP_MAX_MINUTES = 40
_DROP = {"bound_graph", "resolved_params", "workflow_smoke", "system_smoke", "target_workflow_id", "target_model_id"}


def prep_unit_rate() -> float:
    """CU per minute for the CPU runtime; override with GPU_CPU_UNIT_RATE (default is a rounded-up 0.12 CU/h)."""
    return float(os.getenv("GPU_CPU_UNIT_RATE", "0.002"))


def is_prep(job: dict) -> bool:
    return (job.get("snapshot") or {}).get("prep") is True


def cache_key(snapshot: dict) -> str:
    """What the Drive cache holds: the ComfyUI commit and the weight files. A graph-only change keeps the key."""
    manifest = snapshot.get("manifest") or {}
    weights = manifest.get("weight_files") or snapshot.get("weight_files") or []
    # Picked LoRAs ride along per job; they must not split the base cache.
    files = sorted((w.get("repo_id"), w.get("filename"), w.get("revision"), w.get("sha256")) for w in weights if not w.get("lora"))
    # Prep snapshots carry the commit at the top level; user-job snapshots only inside the reviewed manifest.
    commit = snapshot.get("comfyui_commit") or manifest.get("comfyui_commit") or (manifest.get("dependencies") or {}).get("comfyui_commit")
    nodes = manifest.get("custom_nodes") or (manifest.get("dependencies") or {}).get("custom_nodes") or []
    pins = sorted(f"{n.get('repo')}@{n.get('commit')}" for n in nodes)
    # Only added when present, so caches of workflows without custom nodes keep their keys.
    return digest({"comfy": commit, "weights": files, **({"custom_nodes": pins} if pins else {})})


def prep_jobs_for(repo, owner_id: str, snapshot: dict) -> list[dict]:
    key = cache_key(snapshot)
    return [j for j in repo.list("jobs", owner_id) if is_prep(j) and cache_key(j["snapshot"]) == key]


def create_prep_job(
    repo, owner_id: str, project_id: str, snapshot_parts: dict, meta: dict, key: str, for_job_id: str | None = None
) -> dict:
    """Persist one CPU prep job. ``snapshot_parts`` is the reviewed snapshot (manifest/api_graph/object_info...)."""
    cfg = GpuConfig.from_env()
    runtime = min(PREP_MAX_MINUTES, cfg.max_runtime_minutes)
    snapshot = {
        **{k: copy.deepcopy(v) for k, v in snapshot_parts.items() if k not in _DROP},
        "bound_graph": {},
        "resolved_params": {},
        "workflow_sha256": meta["workflow_sha256"],
        "prep": True,
    }
    raw = {
        "project_id": project_id,
        "model_id": meta["model_id"],
        "workflow_key": meta["workflow_key"],
        "workflow_version": meta["workflow_version"],
        "params": {},
        "asset_ids": [],
        "prep_of": meta["workflow_sha256"],
        "execution_policy": {
            "provider": "colab",
            "shutdown_after_job": True,
            "max_runtime_minutes": runtime,
            "idle_grace_minutes": 0,
        },
    }
    job, created = repo.create_job(owner_id, raw, key)
    if created:
        job = repo.update(
            "jobs",
            job["id"],
            {
                "project_id": project_id,
                "provider": "colab",
                "model_id": meta["model_id"],
                "workflow_key": meta["workflow_key"],
                "workflow_version": meta["workflow_version"],
                "snapshot": snapshot,
                "params": {},
                "requested_spec": {"key": dict(PREP_SPEC_KEY)},
                "max_runtime_minutes": runtime,
                **_prep_estimate(repo, owner_id, meta, snapshot, runtime),
                "prep": True,
                **({"for_job_id": for_job_id} if for_job_id else {}),
            },
            owner_id,
        )
        repo.event(job["id"], owner_id, "queued", "เตรียมแคชสำหรับงานโดยอัตโนมัติ (CPU runtime ไม่สร้างภาพ ไม่กิน GPU)" if for_job_id else "สร้างงานเตรียมแคชแล้ว (CPU runtime ไม่กิน GPU)")
    return job


def prep_idempotency_key(workflow_sha256: str, suffix: str = "") -> str:
    return "prep-" + digest({"sha": workflow_sha256, "n": suffix})[:48]


def _prep_estimate(repo, owner_id: str, meta: dict, snapshot: dict, runtime: int) -> dict:
    from .rates import estimate

    units, rate = estimate(repo, owner_id, meta["workflow_key"], snapshot.get("manifest") or {}, PREP_SPEC_KEY, 20, prep=True)
    return {"estimated_units": units, "unit_rate": rate, "budget_units": runtime * prep_unit_rate()}
