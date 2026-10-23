#!/usr/bin/env bash
set -euo pipefail

# This script owns no global lock and never changes CUDA/environment settings.
# All scheduling, GPU inspection, resume decisions, and WebFG-only locking are
# delegated to the independent Python runner.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  echo "Usage: bash scripts/webfg496/run_all.sh [--resume|--status|--summary|--background|--preflight] [--data-root PATH] [--output-root PATH]" >&2
}

MODE="--execute"
if [[ $# -gt 0 && "${1}" != --data-root && "${1}" != --output-root ]]; then
  MODE="$1"
  shift
fi

PATH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root|--output-root)
      if [[ $# -lt 2 || -z "${2}" ]]; then
        echo "Missing path value for $1" >&2
        usage
        exit 2
      fi
      PATH_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python}"
case "$MODE" in
  --execute) "$PYTHON_BIN" -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --execute "${PATH_ARGS[@]}" ;;
  --resume) "$PYTHON_BIN" -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --resume "${PATH_ARGS[@]}" ;;
  --status) "$PYTHON_BIN" -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --status "${PATH_ARGS[@]}" ;;
  --summary) "$PYTHON_BIN" -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --summary "${PATH_ARGS[@]}" ;;
  --background) "$PYTHON_BIN" -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --background "${PATH_ARGS[@]}" ;;
  --preflight) "$PYTHON_BIN" -m experiments.webfg496.runner --config configs/webfg496/formal.yaml --preflight "${PATH_ARGS[@]}" ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage
    exit 2
    ;;
esac
