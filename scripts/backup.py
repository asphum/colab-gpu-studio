"""Consistent SQLite snapshot and private media/config backup without printing secrets."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone


def main():
    from app.config import settings

    p = argparse.ArgumentParser()
    p.add_argument("--output", default="backups")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = out / ("gpu-studio-" + stamp + ".tgz")
    if not settings.database_url.startswith("sqlite:///"):
        raise SystemExit("สำหรับ PostgreSQL โปรดใช้ pg_dump -Fc ก่อน อย่าบีบอัดไดเรกทอรีฐานข้อมูลที่กำลังทำงานอยู่โดยตรง")
    db = Path(settings.database_url.removeprefix("sqlite:///"))
    with tempfile.TemporaryDirectory(prefix="gpu-backup-") as d:
        snapshot = Path(d) / "studio.db"
        with sqlite3.connect(str(db)) as source, sqlite3.connect(str(snapshot)) as target:
            source.backup(target)
            assert target.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as archive:
            archive.add(snapshot, arcname="studio.db")
            for child in settings.data_dir.iterdir():
                if child.is_dir() and child.name in {"assets", "gpu", "gpu-results", "metadata"}:
                    archive.add(child, arcname=child.name)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(
            json.dumps(
                {
                    "path": str(dest.resolve()),
                    "sha256": digest,
                    "notice": "มีข้อมูลประจำตัวทางธุรกิจที่เข้ารหัส; คีย์หลักจำเป็นต้องสำรองแยกต่างหากในที่ปลอดภัย ไม่ได้รวมไฟล์ .env ในแพ็กเกจ",
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
