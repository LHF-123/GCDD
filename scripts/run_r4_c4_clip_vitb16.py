"""Formal R4-C4 CLIP ViT-B/16 cyclic-asym40 experiment entry point.

The runner has an explicit GPU gate: ``--run-all`` and ``--resume`` require a
fresh successful real CUDA preflight on an eligible 24GB-class device.  CPU
preparation, validation-manifest auditing, dry-run command generation, and
status inspection never start a model-training process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.r4_c4_clip_vitb16 import (
    FORMAL_RUNS,
    analyze,
    canonical_sha256,
    current_code_fingerprints,
    gpu_preflight,
    load_plan,
    preflight,
    resolve_config_paths,
    run_tasks,
    task_command,
    task_status,
    tasks_for,
    validate_frozen_config,
    write_prepared_assets,
)
from gcdd.clip_vit import download_clip_vitb16_weights


DEFAULT_CONFIG = ROOT / "configs" / "r4_c4_clip_vitb16_cyclic40.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--fetch-clip-weights",
        action="store_true",
        help="Download and SHA-256-verify the frozen OpenAI ViT-B/16 file; prints only its local path.",
    )
    modes.add_argument("--preflight", action="store_true", help="CPU/data/weight/GPU-capacity audit; does not load CLIP on CUDA.")
    modes.add_argument("--prepare-only", action="store_true", help="Write isolated manifest adapters, protocol audit, and frozen 36-run commands only.")
    modes.add_argument("--dry-run", action="store_true", help="Print the 36 formal commands without preparing or training.")
    modes.add_argument("--gpu-preflight", action="store_true", help="Run actual 24GB CUDA smoke paths for all four methods; never counts as a formal run.")
    modes.add_argument("--status", action="store_true", help="Report formal-run completion state without writing a result summary.")
    modes.add_argument("--run-all", action="store_true", help="Run all pending formal tasks after the frozen GPU preflight passes.")
    modes.add_argument("--resume", action="store_true", help="Run only pending tasks; completed verified tasks remain untouched.")
    modes.add_argument("--analyze", action="store_true", help="Summarize only real verified formal outputs.")
    parser.add_argument(
        "--weights-cache-dir",
        type=Path,
        default=None,
        help="Cache directory for --fetch-clip-weights (default: $R4_C4_CLIP_VITB16_CACHE_DIR or ~/.cache/clip).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = resolve_config_paths(load_plan(path), path)
    validate_frozen_config(config)
    return config


def require_gpu_preflight(config: dict) -> None:
    report = Path(config["output"]["root"]) / "GPU_24GB_MEMORY_PREFLIGHT.json"
    if not report.is_file():
        raise RuntimeError("Formal launch is blocked: run --gpu-preflight successfully on the target ≥24GB GPU first.")
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"Formal launch is blocked by GPU preflight status={payload.get('status')!r}.")
    if payload.get("config_sha256") != canonical_sha256(config):
        raise RuntimeError("Formal launch is blocked: GPU preflight was run under a different frozen configuration.")
    if payload.get("source_code_sha256") != current_code_fingerprints():
        raise RuntimeError("Formal launch is blocked: source files changed after the GPU preflight; rerun --gpu-preflight.")


def acquire_lock(root: Path) -> Path:
    lock = root / ".r4_c4_clip_vitb16.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"A cross-backbone matrix is already active: {lock}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid()}, handle)
    return lock


def main() -> int:
    args = parse_args()
    if args.fetch_clip_weights:
        cache_dir = args.weights_cache_dir or Path(
            os.environ.get("R4_C4_CLIP_VITB16_CACHE_DIR", str(Path.home() / ".cache" / "clip"))
        )
        print(download_clip_vitb16_weights(cache_dir.expanduser()))
        return 0
    config = load_config(args.config)
    if args.preflight:
        print(json.dumps(preflight(config), ensure_ascii=False, indent=2))
        return 0
    if args.dry_run:
        # The command list is deterministic from the frozen protocol and does
        # not depend on prepared output state.
        prepared, tasks = write_prepared_assets(config)
        del prepared
        for position, task in enumerate(tasks, start=1):
            print(f"[{position:02d}/{FORMAL_RUNS}] " + " ".join(task_command(task)))
        return 0
    if args.prepare_only:
        _, tasks = write_prepared_assets(config)
        print(f"PREPARED {len(tasks)}/{FORMAL_RUNS} formal CLIP tasks under {config['output']['root']}")
        return 0
    if args.gpu_preflight:
        prepared, _ = write_prepared_assets(config)
        payload = gpu_preflight(config, prepared)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("status") == "PASS" else 2
    if args.status:
        # Status intentionally does not materialize sources; it only reports
        # planned task locations if preparation has already occurred.
        root = Path(config["output"]["root"])
        if not (root / "prepared_inputs").is_dir():
            print("NOT_PREPARED 0/36; run --prepare-only before requesting formal task status.")
            return 0
        _, tasks = write_prepared_assets(config)
        for task in tasks:
            state, issues = task_status(task)
            print(f"{state:18} {task.dataset_key:8} {task.method_key:20} seed={task.seed} {';'.join(issues)}")
        return 0
    prepared, tasks = write_prepared_assets(config)
    if args.analyze:
        print(json.dumps(analyze(tasks, prepared), ensure_ascii=False, sort_keys=True))
        return 0
    readiness = preflight(config)
    if readiness["status"] != "READY_FOR_GPU_PREFLIGHT":
        print(json.dumps(readiness, ensure_ascii=False, indent=2))
        return 2
    require_gpu_preflight(config)
    lock = acquire_lock(Path(config["output"]["root"]))
    try:
        return run_tasks(tasks, resume=bool(args.resume))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
