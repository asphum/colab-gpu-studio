"""Reference images: a parameter of type "image" carries an asset id; the file ships inside the job snapshot."""

from __future__ import annotations

import base64
import io

from PIL import Image

from ..common import APIError

# All files travel base64-twice inside the one execute_code string; a 0.9 MB PNG made Colab answer
# "service unavailable", a 100 KB JPEG worked. So every image is re-encoded as JPEG and the total is capped.
MAX_TOTAL_BYTES = 400 * 1024
DEFAULT_MAX_EDGE = 1088


def _jpeg(data: bytes, max_edge: int) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=85, optimize=True)
    return out.getvalue()


def resolve_input_images(own_asset, project_id: str, manifest: dict, params: dict) -> tuple[dict, list[dict]]:
    """Swap asset ids for the file names ComfyUI will see; return (params, files to ship)."""
    from ..storage import asset_path

    resolved, files = dict(params), []
    for name, spec in (manifest.get("parameters") or {}).items():
        if spec.get("type") != "image" or params.get(name) in (None, ""):
            continue
        asset = own_asset(params[name])
        if asset.get("project_id") != project_id:
            raise APIError("ASSET_PROJECT_MISMATCH", "ไฟล์อ้างอิงไม่ได้เป็นของโปรเจกต์ปัจจุบัน")
        if asset.get("mime") not in ("image/png", "image/jpeg", "image/webp"):
            raise APIError("VALIDATION_ERROR", f"พารามิเตอร์ {name} รับเฉพาะรูปภาพ PNG/JPEG/WebP เท่านั้น")
        filename = f"input_{name}.jpg"
        resolved[name] = filename
        files.append({"filename": filename, "content": _jpeg(asset_path(asset).read_bytes(), spec.get("max_edge", DEFAULT_MAX_EDGE))})
    if sum(len(f["content"]) for f in files) > MAX_TOTAL_BYTES:
        raise APIError("VALIDATION_ERROR", "ภาพอ้างอิงหลังบีบอัดแล้วยังมีขนาดเกิน 400KB (ขีดจำกัดที่ส่งพร้อมโค้ดเริ่มต้น) โปรดลดจำนวนภาพหรือเลือกภาพที่มีขนาดเล็กลง")
    for f in files:
        f["content_b64"] = base64.b64encode(f.pop("content")).decode("ascii")
    return resolved, files
