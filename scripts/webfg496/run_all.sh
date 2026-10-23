#!/usr/bin/env bash
set -euo pipefail

# This script owns no global lock and never changes CUDA/environment settings.
# All scheduling, GPU inspection, resume decisions, and WebFG-only locking are
# delegated to the independent Python runner.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "${1:-}" in
  "") python -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --execute ;;
  --resume) python -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --resume ;;
  --status) python -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --status ;;
  --summary) python -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --summary ;;
  --background) python -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --background ;;
  --preflight) python -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --preflight ;;
  *)
    echo "Usage: bash scripts/webfg496/run_all.sh [--status|--resume|--summary|--background|--preflight]" >&2
    exit 2
    ;;
esac
