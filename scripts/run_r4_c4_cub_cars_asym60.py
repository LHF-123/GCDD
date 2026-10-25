"""Formal R4-C4 CUB/Cars cyclic-asym60 batch entry point.

The command deliberately delegates every model run to the established
checkpoint-validation trainer.  It never launches a run unless the frozen
historical-compatible protocol, assets, and minimum GPU capacity pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.r4_c4_asym60 import (
    FROZEN,
    METHODS,
    Task,
    analyze,
    gpu_status,
    load_plan,
    read_csv,
    resolve_config_paths,
    run_tasks,
    task_command,
    task_status,
    validate_frozen_config,
    write_prepared_assets,
)


DEFAULT_CONFIG = ROOT / "configs" / "r4_c4_cub_cars_asym60.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--prepare-only", action="store_true", help="Create only manifests, input adapters, and audit artifacts; no model/GPU call.")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--run-all", action="store_true")
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--analyze", action="store_true")
    return parser.parse_args()


def expected_tasks(config: dict) -> list[Task]:
    root = Path(config["output"]["root"])
    instrumentation = Path(config["output"]["instrumentation_root"])
    tasks: list[Task] = []
    for dataset_key in ("cub", "cars"):
        data = config["datasets"][dataset_key]
        name = f"{data['display_name']}-asym60-s42"
        for method_key, runner_method, selection_logging in METHODS:
            for seed in FROZEN["training_seeds"]:
                tasks.append(Task(dataset_key, name, method_key, runner_method, selection_logging, seed, root,
                                  root / "prepared_inputs" / dataset_key,
                                  root / "noise_manifests" / dataset_key / f"{dataset_key}_cyclic_asym60_s42_index.csv",
                                  root / "fixed_validation" / dataset_key, instrumentation,
                                  tuple((str(item[0]), str(item[1])) for item in data.get("path_maps", []))))
    return tasks


def preflight(config: dict) -> dict:
    findings: list[dict[str, str]] = []
    data_locations: list[dict[str, str]] = []
    for dataset_key, data in config["datasets"].items():
        for field in ("base_input_dir", "source_noise_index", "source_resolved_config"):
            path = Path(data[field])
            findings.append({"dataset": dataset_key, "check": field, "status": "PASS" if path.exists() else "FAIL", "path": str(path)})
        source_config = yaml.safe_load(Path(data["source_resolved_config"]).read_text(encoding="utf-8")) or {}
        path_maps = tuple((str(item[0]), str(item[1])) for item in data.get("path_maps", []))

        def mapped(value: str) -> Path:
            for old_root, new_root in path_maps:
                normalized = value.replace("\\", "/")
                old_normalized = old_root.replace("\\", "/").rstrip("/")
                if normalized == old_normalized:
                    return Path(new_root)
                if normalized.startswith(old_normalized + "/"):
                    return Path(new_root + value[len(old_root):])
            return Path(value)

        raw_root = mapped(str(source_config.get("dataset", {}).get("root", "")))
        base_paths = (Path(data["base_input_dir"]) / "paths.txt").read_text(encoding="utf-8").splitlines()
        sample = mapped(base_paths[0]) if base_paths else raw_root
        data_locations.extend([
            {"dataset": dataset_key, "check": "dataset_root_or_path_map", "status": "PASS" if raw_root.is_dir() else "REMOTE_CONFIGURATION_REQUIRED", "path": str(raw_root)},
            {"dataset": dataset_key, "check": "first_training_image_or_path_map", "status": "PASS" if sample.is_file() else "REMOTE_CONFIGURATION_REQUIRED", "path": str(sample)},
        ])
    historical = Path(config["historical"]["repository_root"])
    for relative in ("scripts/run_lora_checkpoint_validation.py", "gcdd/lora_dynamic.py", "gcdd/lora_training.py"):
        path = historical / relative
        findings.append({"dataset": "historical", "check": relative, "status": "PASS" if path.is_file() else "FAIL", "path": str(path)})
    capacity = gpu_status(int(config["runtime"]["minimum_gpu_memory_gib"]))
    required_ok = all(item["status"] == "PASS" for item in findings)
    local_data_ok = all(item["status"] == "PASS" for item in data_locations)
    status = (
        "BLOCKED_PROTOCOL" if not required_ok else "READY_WAIT_GPU" if capacity["status"] != "READY"
        else "BLOCKED_DATASET_LOCATION" if not local_data_ok else "READY"
    )
    return {"status": status, "planned_runs": len(expected_tasks(config)), "checks": findings, "data_location": data_locations, "gpu": capacity}


def load_prepared(config: dict) -> dict[str, dict]:
    """Load completed preparation inputs without changing any asset."""
    root = Path(config["output"]["root"])
    prepared: dict[str, dict] = {}
    for key in ("cub", "cars"):
        manifest_path = root / "noise_manifests" / key / f"{key}_cyclic_asym60_s42_index.csv"
        input_dir = root / "prepared_inputs" / key
        validation_dir = root / "fixed_validation" / key
        required = (manifest_path, input_dir / "paths.txt", input_dir / "labels.npy", validation_dir / "validation_manifest.csv")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError("Run --prepare-only before --analyze; missing: " + "; ".join(missing))
        paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
        observed = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
        manifest = read_csv(manifest_path)
        clean_by_path: dict[str, str] = {}
        for row in manifest:
            if row.get("split", "").lower() != "train":
                continue
            relative = row["path"].replace("\\", "/")
            clean_by_path[relative] = row["clean_label"]
            if row.get("abs_path"):
                clean_by_path[row["abs_path"].replace("\\", "/")] = row["clean_label"]
        clean_values = []
        for path in paths:
            normalized = path.replace("\\", "/")
            candidates = [normalized]
            for marker in ("/images/", "/train/"):
                if marker in normalized:
                    candidates.append(marker[1:] + normalized.split(marker, 1)[1])
            clean_values.append(next((clean_by_path[key] for key in candidates if key in clean_by_path), ""))
        clean = np.asarray(clean_values, dtype=str)
        if np.any(clean == ""):
            raise ValueError(f"Prepared {key} clean labels no longer align with the manifest")
        pool = np.zeros(len(paths), dtype=bool)
        for row in read_csv(validation_dir / "validation_manifest.csv"):
            if row.get("partition") == "training_pool":
                pool[int(row["index"])] = True
        if not pool.any():
            raise ValueError(f"Prepared {key} validation manifest has no training-pool rows")
        prepared[key] = {"key": key, "manifest_rows": manifest, "clean_labels": clean, "observed_labels": observed,
                         "split": SimpleNamespace(train_mask=pool)}
    return prepared


def acquire_lock(root: Path) -> Path:
    path = root / ".r4_c4_batch.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"R4-C4 batch lock already exists: {path}; do not start a concurrent matrix.") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid()}, handle)
    return path


def main() -> int:
    args = parse_args()
    config = resolve_config_paths(load_plan(args.config), args.config)
    validate_frozen_config(config)
    if args.preflight:
        print(json.dumps(preflight(config), indent=2, ensure_ascii=False))
        return 0
    if args.dry_run:
        for ordinal, task in enumerate(expected_tasks(config), start=1):
            print(f"[{ordinal:02d}/24] " + " ".join(task_command(task)))
        return 0
    if args.status:
        for task in expected_tasks(config):
            status, issues = task_status(task)
            print(f"{status:18} {task.dataset_key:4} {task.method_key:20} seed={task.seed} {str(task.run_dir)} {';'.join(issues)}")
        return 0
    if args.prepare_only:
        _, tasks = write_prepared_assets(config)
        print(f"PREPARED {len(tasks)} planned runs under {config['output']['root']}")
        return 0
    if args.analyze:
        counts = analyze(config, load_prepared(config), expected_tasks(config))
        print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
        return 0

    readiness = preflight(config)
    # Preparation is CPU-only and establishes the frozen noise/validation facts
    # even if an eligible training GPU is not currently available.
    prepared, tasks = write_prepared_assets(config)
    if readiness["status"] != "READY":
        print(json.dumps(readiness, indent=2, ensure_ascii=False))
        return 2
    lock = acquire_lock(Path(config["output"]["root"]))
    try:
        exit_code = run_tasks(tasks, resume=bool(args.resume))
        analyze(config, prepared, tasks)
        return exit_code
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
