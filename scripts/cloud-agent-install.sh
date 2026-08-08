#!/usr/bin/env bash
# Cloud agent / local bootstrap for DeepGuard AI
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Installing backend Python dependencies"
python3 -m venv "$ROOT/.venv" 2>/dev/null || true
# Prefer system/user pip if venv creation is restricted
if [[ -x "$ROOT/.venv/bin/pip" ]]; then
  PIP="$ROOT/.venv/bin/pip"
  PY="$ROOT/.venv/bin/python"
else
  PIP="pip3"
  PY="python3"
fi

$PIP install --upgrade pip
$PIP install -r "$ROOT/backend/requirements.txt"

echo "==> Installing frontend dependencies"
cd "$ROOT/frontend"
npm install

echo "==> DeepGuard install complete"
