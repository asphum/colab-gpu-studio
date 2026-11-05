"""Initialize private configuration once. Existing files are never overwritten."""

import argparse
import os
import secrets
from pathlib import Path

from cryptography.fernet import Fernet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=".env")
    p.add_argument("--site-url", default="http://localhost:8788")
    a = p.parse_args()
    path = Path(a.output)
    if path.exists():
        raise SystemExit("ไฟล์คอนฟิกมีอยู่แล้ว; เพื่อปกป้องข้อมูลประจำตัวจะไม่เขียนทับ")
    values = {
        "ENVIRONMENT": "development",
        "SITE_URL": a.site_url.rstrip("/"),
        "DATABASE_URL": "sqlite:///./data/studio.db",
        "DATA_DIR": "./data",
        "STUDIO_PASSWORD": secrets.token_urlsafe(24),
        "ENCRYPTION_KEY": Fernet.generate_key().decode(),
        "SESSION_SECRET": secrets.token_urlsafe(48),
        "STUDIO_API_TOKEN": secrets.token_urlsafe(40),
        "GPU_AGENT_TOKEN": secrets.token_urlsafe(40),
        "GPU_AGENT_OWNER_ID": "owner",
        "GPU_REAL_ENABLED": "0",
        "GPU_STORAGE": "local",
        "GPU_MAX_RUNTIME_MINUTES": "30",
        "GPU_DAILY_UNITS": "10",
        "GPU_MONTHLY_UNITS": "100",
        "GPU_MAX_JOB_UNITS": "5",
        "GPU_UNIT_RATE": "",
        "GPU_GOOGLE_CLIENT_ID": "",
        "GPU_GOOGLE_CLIENT_SECRET": "",
        "GPU_GOOGLE_REDIRECT_URI": "",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(k + "=" + v for k, v in values.items()) + "\n")
    print("สร้างไฟล์คอนฟิกส่วนตัวเรียบร้อยแล้ว: " + str(path) + " รหัสผ่านอยู่ในไฟล์และจะไม่แสดงในล็อก GPU จริงถูกปิดไว้เป็นค่าเริ่มต้น")


if __name__ == "__main__":
    main()
