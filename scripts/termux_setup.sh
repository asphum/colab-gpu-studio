#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

echo "========================================================"
echo "  Colab GPU Studio — สคริปต์ติดตั้งและเริ่มใช้งานบน Android Termux"
echo "========================================================"

SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Android shared storage (/storage/emulated/0 or /sdcard) blocks symlinks and exec permissions.
# Automatically relocate/sync into Termux $HOME/colab-gpu-studio when run from shared storage.
case "$SRC_DIR" in
  /storage/*|/sdcard/*)
    TARGET_DIR="$HOME/colab-gpu-studio"
    echo "[คำแนะนำ] ตรวจพบว่าไดเรกทอรีปัจจุบันอยู่บน Android Shared Storage ($SRC_DIR) ซึ่งไม่รองรับ Symlink และสิทธิ์การ Execute"
    echo "       กำลังซิงค์ไปยังไดเรกทอรีภายในของ Termux โดยอัตโนมัติ: $TARGET_DIR ..."
    mkdir -p "$TARGET_DIR"
    cp -rf "$SRC_DIR"/* "$TARGET_DIR"/
    if [ -f "$SRC_DIR/.env.example" ]; then
      cp -f "$SRC_DIR/.env.example" "$TARGET_DIR/.env.example"
    fi
    if [ -f "$SRC_DIR/.env" ] && [ ! -f "$TARGET_DIR/.env" ]; then
      cp -f "$SRC_DIR/.env" "$TARGET_DIR/.env"
    fi
    ROOT_DIR="$TARGET_DIR"
    ;;
  *)
    ROOT_DIR="$SRC_DIR"
    ;;
esac

cd "$ROOT_DIR"

if ! command -v pkg >/dev/null 2>&1; then
  echo "คำแนะนำ: ไม่พบคำสั่ง pkg ของ Termux จะพยายามรันด้วย Python ของระบบปัจจุบัน"
else
  echo "[1/4] กำลังติดตั้งแพ็กเกจพื้นฐานและส่วนขยายที่คอมไพล์แล้ว (python, python-cryptography, python-pillow)..."
  pkg update -y
  pkg install -y python python-pip python-cryptography python-pillow git openssl libffi
fi

if [ ! -x ".venv/bin/python" ] || ! grep -q "include-system-site-packages = true" ".venv/pyvenv.cfg" 2>/dev/null; then
  rm -rf .venv
  echo "[2/4] กำลังสร้าง Python Virtual Environment (นำส่วนขยายที่คอมไพล์แล้วของระบบมาใช้)..."
  python3 -m venv --system-site-packages .venv
fi

echo "[3/4] กำลังติดตั้ง Dependencies ของ Python (ใช้คลัง Precompiled Wheels สำหรับ Termux โดยไม่ต้องคอมไพล์เอง)..."
./.venv/bin/pip install --upgrade pip

# Clean requirements for Termux:
# 1. psycopg is postgres driver, not needed for SQLite
# 2. cryptography and pillow are already provided by Termux pkg
# 3. argon2-cffi was legacy unused dependency
grep -Ev '^(psycopg|cryptography==|pillow==|argon2)' requirements.lock > .requirements-termux.txt

# Use TUR PyPI index so pydantic-core and other native extensions download precompiled aarch64 wheels directly
./.venv/bin/pip install \
  --extra-index-url https://termux-user-repository.github.io/pypi/ \
  --prefer-binary \
  -r .requirements-termux.txt

rm -f .requirements-termux.txt

echo "[4/4] กำลังเริ่มต้น Colab GPU Studio..."
if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock || true
  echo "เปิดใช้งาน Termux WakeLock เพื่อป้องกันการหลับเบื้องหลังแล้ว"
fi

export PYTHONPATH="$ROOT_DIR/api${PYTHONPATH:+:$PYTHONPATH}"
exec ./.venv/bin/python -m app.launcher --host 127.0.0.1 --port 8788
