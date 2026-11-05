#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
  printf '%s\n' 'กรุณารันคำสั่ง uv run python scripts/init_env.py --site-url https://โดเมนของคุณ ก่อน'
  exit 1
fi
docker compose config -q
docker compose build
docker compose up -d --wait --wait-timeout 180
docker compose ps
