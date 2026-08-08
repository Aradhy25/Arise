#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Verifying Node.js toolchain"
node --version
npm --version

echo "==> Arise install complete (no package dependencies yet)"
