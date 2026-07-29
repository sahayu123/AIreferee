#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_REPO="$(cd "$APP_DIR/.." && pwd)"

git -C "$SOURCE_REPO" pull --ff-only
