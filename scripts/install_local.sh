#!/usr/bin/env bash
# Mac / Linux local install helper for DeepGuard AI
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
echo "==> Installing PyTorch (large download — long timeout)…"
pip install --default-timeout=1000 --retries 10 torch==2.5.1 torchvision==0.20.1
echo "==> Installing remaining backend deps…"
pip install --default-timeout=1000 --retries 10 -r backend/requirements.txt

echo "==> Building frontend…"
cd frontend
npm install
npm run build
cd "$ROOT"

if [[ ! -f backend/weights/efficientnet.pth ]]; then
  echo "==> Bootstrapping model weights…"
  python scripts/bootstrap_weights.py --epochs 3 --train-samples 400 --val-samples 80
fi

echo ""
echo "Done. Start the app with:"
echo "  source .venv/bin/activate"
echo "  cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "Then open http://localhost:8000"
