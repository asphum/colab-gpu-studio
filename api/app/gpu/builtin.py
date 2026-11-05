"""Offline, owner-scoped built-in workflow drafts; never carry execution approval."""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path

from .manifest import validate_manifest

_ROOT = Path(__file__).with_name("templates") / "builtin"
_MODEL_FIELDS = {
    "model_id", "display_name", "revision", "source", "license", "modality", "resource_profile", "variants",
}


@lru_cache(maxsize=1)
def _bundles() -> tuple[dict, ...]:
    catalog = json.loads((_ROOT / "catalog.json").read_text(encoding="utf-8"))
    result = []
    identities = set()
    for entry in catalog["workflows"]:
        filename = entry["file"]
        if not isinstance(filename, str) or not re.fullmatch(r"[a-z0-9.-]+\.json", filename):
            raise ValueError("ชื่อไฟล์เทมเพลตในตัวไม่ปลอดภัย (unsafe builtin template filename)")
        bundle = json.loads((_ROOT / filename).read_text(encoding="utf-8"))
        manifest = validate_manifest(bundle["manifest"], bundle["api_graph"], bundle["object_info"])
        identity = (bundle["workflow_key"], manifest["version"])
        if identity != (entry["workflow_key"], entry["version"]) or identity in identities:
            raise ValueError("ข้อมูลระบุตัวตนของเทมเพลตในตัวไม่สอดคล้องกันหรือซ้ำซ้อน")
        if bundle["model_id"] != manifest["model_id"] or bundle["model_id"] != bundle["model"]["model_id"]:
            raise ValueError("ข้อมูลระบุตัวตนของโมเดลในเทมเพลตไม่สอดคล้องกัน")
        identities.add(identity)
        bundle["manifest"] = manifest
        result.append(bundle)
    return tuple(result)


def seed_builtin_templates(registry, owner_id: str) -> None:
    """Called under Registry.bootstrap's existing cross-process dispatch lock.

    Add only absent records. Existing approvals, drafts, published versions and
    active pointers are not touched. Import never creates jobs or runtimes.
    """
    bundles = _bundles()
    known_models = {row.get("model_id") for row in registry.models(owner_id)}
    known_workflows = {
        (row.get("workflow_key"), row.get("workflow_version")) for row in registry.workflows(owner_id)
    }
    for bundle in bundles:
        model_id = bundle["model_id"]
        if model_id not in known_models:
            model = {key: copy.deepcopy(value) for key, value in bundle["model"].items() if key in _MODEL_FIELDS}
            model.update({
                "enabled": False,
                "is_mock": False,
                "disabled_reason": "เทมเพลตในตัวยังไม่ผ่านการตรวจสอบสัญญาอนุญาตและการทดสอบจริงในการติดตั้งปัจจุบัน จึงไม่เปิดใช้งานอัตโนมัติ",
            })
            registry.repo.create("models", owner_id, model)
            known_models.add(model_id)
        identity = (bundle["workflow_key"], bundle["manifest"]["version"])
        if identity not in known_workflows:
            registry.import_workflow(owner_id, copy.deepcopy(bundle))
            known_workflows.add(identity)
