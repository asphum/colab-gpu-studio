"""Private durable asset files; no executable uploads or user paths."""

import hashlib
import io
import mimetypes
import re
import uuid
from pathlib import Path

from PIL import Image

from .config import settings

ALLOWED = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "video/mp4",
    "video/webm",
    "audio/wav",
    "audio/mpeg",
}


def asset_path(asset):
    root = (settings.data_dir / "assets").resolve()
    key = asset.get("storage_key", "")
    path = (root / key).resolve()
    if not key or not path.is_relative_to(root) or path == root:
        raise ValueError("เส้นทางไฟล์ไม่ถูกต้อง (invalid asset path)")
    return path


def save_asset(store, owner_id, project_id, content, filename, source_job_id=None, metadata=None):
    if not content or len(content) > settings.max_upload_bytes:
        raise ValueError("ไฟล์ว่างเปล่าหรือมีขนาดเกินขีดจำกัด (empty or exceeds limit)")
    name = re.sub(r"[^\w.()\-\u4e00-\u9fff ]+", "_", Path(filename).name).strip(" .")[:140] or "file.bin"
    mime = mimetypes.guess_type(name)[0]
    if mime not in ALLOWED:
        if mime != "application/json" or not (metadata or {}).get("system_smoke"):
            raise ValueError("ประเภทไฟล์ไม่รองรับ (unsupported asset type)")
        __import__("json").loads(content)  # Diagnostic JSON only from the trusted CUDA executor.
    width = height = None
    if mime.startswith("image/"):
        with Image.open(io.BytesIO(content)) as im:
            im.verify()
        with Image.open(io.BytesIO(content)) as im:
            width, height = im.size
            if Image.MIME.get(im.format) != mime:
                raise ValueError("นามสกุลรูปภาพไม่ตรงกับเนื้อหาไฟล์ (mismatched image extension)")
    elif mime == "video/mp4" and content[4:8] != b"ftyp":
        raise ValueError("เนื้อหาไฟล์ MP4 ไม่ถูกต้อง (invalid MP4 content)")
    elif mime == "video/webm" and not content.startswith(b"\x1a\x45\xdf\xa3"):
        raise ValueError("เนื้อหาไฟล์ WebM ไม่ถูกต้อง (invalid WebM content)")
    elif mime == "audio/wav" and not (content[:4] == b"RIFF" and content[8:12] == b"WAVE"):
        raise ValueError("เนื้อหาไฟล์ WAV ไม่ถูกต้อง (invalid WAV content)")
    elif mime == "audio/mpeg" and not (
        content.startswith(b"ID3") or content[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    ):
        raise ValueError("เนื้อหาไฟล์ MP3 ไม่ถูกต้อง (invalid MP3 content)")
    key = f"{owner_id}/{project_id}/{uuid.uuid4().hex}-{name}"
    row = {
        "filename": name,
        "mime": mime,
        "storage_key": key,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "width": width,
        "height": height,
        "favorite": False,
        "source_job_id": source_job_id,
    }
    for k, v in (metadata or {}).items():
        if k not in row and k not in {"id", "owner_id", "project_id"}:
            row[k] = v
    target = asset_path(row)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    tmp.write_bytes(content)
    tmp.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
        raise OSError("การตรวจสอบไฟล์ลงดิสก์ล้มเหลว (asset disk write verification failed)")
    return store.create("assets", owner_id, row, project_id=project_id)


def read_asset(asset):
    return asset_path(asset).read_bytes()


def delete_asset_file(asset):
    asset_path(asset).unlink(missing_ok=True)
