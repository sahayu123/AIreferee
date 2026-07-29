#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  echo "Model runtime is missing. Run: npm run setup:models"
  exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1

"$APP_DIR/.venv/bin/python" -m uvicorn backend.server:app \
  --host 127.0.0.1 --port 8200 --app-dir "$APP_DIR" &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$APP_DIR"
npm run dev
