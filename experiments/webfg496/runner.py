"""Isolated WebFG-496 matrix scheduler for the PGDF major revision.

The runner consumes the original WebFG list files directly, calls existing
public training functions, and confines every generated artifact to
``outputs_major_revision/webfg496``.  It intentionally has no dependency on
synthetic-noise indexes or historic experiment outputs.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np
import yaml

from .dataset import inventory_webfg, load_webfg_splits
from .corruption import prepare_corrupt_image_resolution


ROOT = Path(__file__).resolve().parents[2]
METHOD_ORDER = ("pgdf_dynamic_proto", "all_noisy_ce", "jal_ce", "coteaching")
RUN_COMPLETE_STATES = {"COMPLETE", "COMPLETE_WITH_PROFILING_GAP"}
FORMAL_COMPLETE_STATE = "COMPLETE"
PROFILING_GAP = "BLOCKED_SHARED_CODE_CHANGE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebFG-496 isolated 36-run scheduler.")
    parser.add_argument("--config", default="configs/webfg496/formal.yaml")
    parser.add_argument("--data-root", help="Override the configured read-only WebFG root for this invocation.")
    parser.add_argument("--output-root", help="Override the configured WebFG-only output root.")
    parser.add_argument("--execute", action="store_true", help="Run pending matrix entries after preflight and GPU isolation checks.")
    parser.add_argument("--resume", action="store_true", help="Create a new isolated attempt only for unfinished matrix entries.")
    parser.add_argument("--preflight", action="store_true", help="Build the read-only data inventory and interface/isolation checks only.")
    parser.add_argument("--status", action="store_true", help="Read and print existing WebFG status only.")
    parser.add_argument("--summary", action="store_true", help="Read and print existing WebFG summaries only.")
    parser.add_argument("--background", action="store_true", help="Launch this matrix in a persistent WebFG-only background process.")
    parser.add_argument("--verify-images", action="store_true", help="Force exhaustive PIL verification of all listed images during preflight.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume:
        args.execute = True
    selected = sum(bool(value) for value in (args.execute, args.preflight, args.status, args.summary, args.background))
    if selected == 0:
        args.execute = True
    elif selected != 1:
        raise SystemExit("Choose exactly one of --execute/--resume, --preflight, --status, --summary, or --background.")
    config = load_config(Path(args.config))
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.output_root:
        config["output_root"] = args.output_root
    output_root = resolve_output_root(config)
    if args.status:
        print_status(output_root)
        return 0
    if args.summary:
        print_summary(output_root)
        return 0
    if args.background:
        return launch_background(args, config, output_root)
    if args.preflight:
        preflight(config, output_root, verify_images=True if args.verify_images else True)
        print(f"PREFLIGHT_PASS output_root={output_root}")
        return 0
    return execute_matrix(config, output_root, resume=bool(args.resume), verify_images=True if args.verify_images else True)


def load_config(path: Path) -> dict[str, Any]:
    full_path = path if path.is_absolute() else ROOT / path
    if not full_path.is_file():
        raise FileNotFoundError(f"WebFG configuration does not exist: {full_path}")
    with full_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config)
    config["_config_path"] = str(full_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {"data", "protocol", "model", "training", "methods", "hardware", "output_root"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"WebFG configuration is missing required sections: {missing}")
    if tuple(config["methods"].keys()) != METHOD_ORDER:
        raise ValueError(f"Method order must be exactly {METHOD_ORDER}; got {tuple(config['methods'].keys())}.")
    expected = config["data"].get("expected_classes", {})
    if set(expected) != {"web-bird", "web-car", "web-aircraft"}:
        raise ValueError("data.expected_classes must define web-bird, web-car, and web-aircraft.")
    if list(config["training"].get("seeds", [])) != [1, 42, 88]:
        raise ValueError("The formal WebFG matrix is frozen to seeds [1, 42, 88].")
    if int(config["training"].get("epochs", 0)) != 30:
        raise ValueError("The formal WebFG protocol is frozen to 30 epochs.")
    pgdf = config["methods"]["pgdf_dynamic_proto"]
    if (float(pgdf["selection_ratio"]), float(pgdf["prototype_keep_ratio"]), int(pgdf["warmup_epochs"]), int(pgdf["selection_interval"])) != (0.8, 0.4, 5, 5):
        raise ValueError("PGDF formal settings must remain r=0.8, p=0.4, warmup=5, interval=5.")
    if config["protocol"].get("epoch_selection_rule") != "fixed_last_epoch":
        raise ValueError("WebFG evaluation must use the pre-declared fixed last epoch.")


def resolve_output_root(config: dict[str, Any]) -> Path:
    path = Path(str(config["output_root"]))
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_yaml(path: Path, payload: Any) -> None:
    atomic_text(path, yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(config: dict[str, Any], output_root: Path, *, verify_images: bool) -> None:
    """Perform read-only data and shared-code checks, then write WebFG-only evidence."""
    output_root.mkdir(parents=True, exist_ok=True)
    public_interface = verify_public_interfaces()
    isolation = verify_shared_code_isolation()
    if isolation["status"] != "PASS":
        raise RuntimeError(f"Shared-code isolation failed: {isolation['violations']}")
    resolution_paths = prepare_corrupt_image_resolution(Path(str(config["data"]["root"])), output_root)
    train_manifests = {
        dataset: output_root / "manifests" / f"{dataset}_effective_train.csv"
        for dataset in config["data"]["expected_classes"]
    }
    rows, class_mapping = inventory_webfg(
        Path(str(config["data"]["root"])),
        {str(key): int(value) for key, value in config["data"]["expected_classes"].items()},
        verify_images=verify_images,
        train_manifests=train_manifests,
    )
    mapping_failures = [
        f"{dataset}:{entry['label']}"
        for dataset, info in class_mapping.items()
        for entry in info["classes"]
        if not entry["mapping_consistent"]
    ]
    failed_rows = [row for row in rows if row["dataset_status"] != "PASS"]
    if failed_rows or mapping_failures:
        failed_coordinates = [f"{row['dataset']}/{row['split']}" for row in failed_rows]
        raise RuntimeError(
            "WebFG data inventory failed: "
            f"rows={failed_coordinates}, "
            f"mapping_failures={mapping_failures[:10]}"
        )
    inventory_fields = list(rows[0].keys()) if rows else []
    write_csv(output_root / "dataset_inventory.csv", rows, inventory_fields)
    write_json(output_root / "class_mapping.json", class_mapping)
    write_dataset_protocol(output_root / "DATASET_PROTOCOL.md", config, rows)
    provenance = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "repository_root": str(ROOT),
        "config_path": config["_config_path"],
        "config_sha256": file_sha256(Path(config["_config_path"])),
        "data_root_checked": str(config["data"]["root"]),
        "corrupt_image_resolution_artifacts": {key: str(value) for key, value in resolution_paths.items()},
        "effective_train_manifests": {key: str(value) for key, value in train_manifests.items()},
        "public_interface_check": public_interface,
        "shared_code_isolation": isolation,
        "matrix": build_matrix(config),
        "matrix_size": len(build_matrix(config)),
        "protocol": config["protocol"],
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    write_json(output_root / "protocol_and_provenance.json", provenance)
    aggregate_outputs(output_root, config)


def verify_public_interfaces() -> dict[str, Any]:
    """Confirm the independent adapter still targets existing public APIs."""
    from gcdd.lora_dynamic import train_dynamic_loss_lora
    from gcdd.lora_noisy_baselines import train_coteaching_lora
    from gcdd.lora_training import train_dinov2_lora

    required = {
        "train_dinov2_lora": (train_dinov2_lora, {"train_paths", "train_labels", "eval_paths", "eval_labels", "train_mask", "cfg", "method", "seed"}),
        "train_dynamic_loss_lora": (train_dynamic_loss_lora, {"candidate_mask", "retention_ratio", "warmup_epochs", "update_interval", "profiler"}),
        "train_coteaching_lora": (train_coteaching_lora, {"remember_mode", "remember_rate", "final_remember_rate", "warmup_epochs", "grad_accum_steps"}),
    }
    signatures: dict[str, list[str]] = {}
    for name, (function, expected) in required.items():
        observed = set(inspect.signature(function).parameters)
        missing = sorted(expected - observed)
        if missing:
            raise RuntimeError(f"Public training API {name} lacks required parameters: {missing}")
        signatures[name] = sorted(observed)
    return {"status": "PASS", "signatures": signatures}


def verify_shared_code_isolation() -> dict[str, Any]:
    """Fail closed only for tracked edits to protected shared training surfaces."""
    command = ["git", "diff", "--name-only"]
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=True)
    changed = [line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()]
    protected_prefixes = ("gcdd/", "losses/", "revision_experiments/", "configs/", "scripts/")
    allowed_prefixes = ("configs/webfg496/", "scripts/webfg496/")
    violations = [
        path for path in changed
        if path.startswith(protected_prefixes) and not path.startswith(allowed_prefixes)
    ]
    fingerprint_paths = [
        "gcdd/lora_training.py",
        "gcdd/lora_dynamic.py",
        "gcdd/lora_noisy_baselines.py",
        "scripts/run_lora_checkpoint_validation.py",
        "scripts/run_lora_noisy_baseline.py",
    ]
    fingerprints = {relative: file_sha256(ROOT / relative) for relative in fingerprint_paths}
    return {
        "status": "PASS" if completed.returncode == 0 and not violations else "FAIL",
        "tracked_changed_paths": changed,
        "violations": violations,
        "protected_source_sha256": fingerprints,
    }


def write_dataset_protocol(path: Path, config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    counts = {
        f"{row['dataset']}/{row['split']}": {
            "source": row["list_entries"],
            "effective": row.get("effective_list_entries", row["list_entries"]),
            "excluded": row.get("excluded_from_effective_train", 0),
        }
        for row in rows
    }
    pgdf = config["methods"]["pgdf_dynamic_proto"]
    lines = [
        "# WebFG-496 Dataset and Evaluation Protocol",
        "",
        "## Frozen protocol",
        "",
        f"- Official reference: {config['protocol']['official_reference']}",
        f"- Checked data root: `{config['data']['root']}` (the server root is configured, not inferred from this Windows host).",
        "- Training inputs: only `train-list.txt` observed web labels and listed `train/` images; no synthetic 40% noise is injected.",
        "- Confirmed unreadable training files are resolved only through the WebFG-specific manifest: a fully decoded raw archive member is mapped from `recovered_originals/`; otherwise the exact documented record is excluded. Source data remain unchanged.",
        "- Evaluation inputs: official package `val-list.txt` and listed `val/` images, never used as a tuning/selection validation set.",
        "- No clean/noisy identity, clean validation split, or oracle filtering signal is loaded.",
        "- Checkpoint rule for every method: fixed final epoch 30. The final Top-1 is the epoch-30 official `val` Top-1.",
        "- Co-teaching primary value: arithmetic mean of branch-A and branch-B Top-1 at epoch 30; no branch is selected by final evaluation performance.",
        f"- PGDF: warm-up {pgdf['warmup_epochs']} epochs; selection updates after epochs 5, 10, 15, 20, and 25; r={pgdf['selection_ratio']}, p={pgdf['prototype_keep_ratio']}.",
        "",
        "## Local inventory counts",
        "",
        "| dataset/split | source list entries | effective entries | fixed exclusions |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {key} | {value['source']} | {value['effective']} | {value['excluded']} |"
        for key, value in sorted(counts.items())
    )
    lines.extend(
        [
            "",
            "## Server preflight",
            "",
            "Before any AutoDL training, `run_all.sh` re-runs the same inventory against `/root/autodl-tmp`. Any path, count, mapping, duplicate, unlisted-file, or readability mismatch fails closed; it is never silently skipped.",
        ]
    )
    atomic_text(path, "\n".join(lines) + "\n")


def build_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"dataset": dataset, "method": method, "seed": int(seed)}
        for dataset in config["data"]["expected_classes"]
        for method in METHOD_ORDER
        for seed in config["training"]["seeds"]
    ]


def execute_matrix(config: dict[str, Any], output_root: Path, *, resume: bool, verify_images: bool) -> int:
    preflight(config, output_root, verify_images=verify_images)
    gpu = inspect_gpu(config)
    write_json(output_root / "gpu_preflight.json", gpu)
    if gpu["status"] != "READY":
        aggregate_outputs(output_root, config, run_state=gpu["status"])
        print(f"{gpu['status']}: {gpu['reason']}")
        return 0
    with webfg_lock(output_root):
        for task in build_matrix(config):
            coordinate = coordinate_dir(output_root, task)
            completed = completed_attempt(coordinate)
            if completed is not None:
                continue
            existing_attempts = sorted(coordinate.glob("attempt-*")) if coordinate.exists() else []
            if existing_attempts and not resume:
                raise RuntimeError(
                    f"Unfinished WebFG run exists at {coordinate}; use --resume to create a new isolated attempt without overwriting it."
                )
            attempt = next_attempt_dir(coordinate)
            run_single_task(config, output_root, task, attempt, gpu)
            aggregate_outputs(output_root, config)
    aggregate_outputs(output_root, config)
    print_status(output_root)
    return 0


def inspect_gpu(config: dict[str, Any]) -> dict[str, Any]:
    """Require an exclusively idle requested GPU; never kills or alters other jobs."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "READY_WAIT_GPU", "reason": "nvidia-smi is unavailable; this host cannot prove an isolated CUDA device.", "checked_at_utc": utc_now()}
    gpu_result = subprocess.run(
        [executable, "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        capture_output=True,
    )
    if gpu_result.returncode != 0:
        return {"status": "READY_WAIT_GPU", "reason": f"nvidia-smi GPU query failed: {gpu_result.stderr.strip()}", "checked_at_utc": utc_now()}
    gpu_rows = list(csv.reader(line for line in gpu_result.stdout.splitlines() if line.strip()))
    target_index = int(config["hardware"]["gpu_index"])
    target = next((row for row in gpu_rows if row and int(row[0].strip()) == target_index), None)
    if target is None:
        return {"status": "READY_WAIT_GPU", "reason": f"Requested GPU index {target_index} does not exist.", "checked_at_utc": utc_now(), "gpus": gpu_rows}
    process_result = subprocess.run(
        [executable, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=False,
        text=True,
        capture_output=True,
    )
    process_rows = list(csv.reader(line for line in process_result.stdout.splitlines() if line.strip())) if process_result.returncode == 0 else []
    busy = [row for row in process_rows if row and row[0].strip() == target[1].strip()]
    if bool(config["hardware"].get("require_exclusive_gpu", True)) and busy:
        return {
            "status": "READY_WAIT_GPU",
            "reason": f"GPU {target_index} has active compute process(es); WebFG will not share or interrupt them.",
            "checked_at_utc": utc_now(),
            "target_gpu": target,
            "active_processes": busy,
        }
    return {"status": "READY", "reason": "Requested GPU is available and exclusively idle at launch check.", "checked_at_utc": utc_now(), "target_gpu": target, "active_processes": busy}


@contextlib.contextmanager
def webfg_lock(output_root: Path) -> Iterator[None]:
    """An exclusive lock scoped only to this WebFG matrix output root."""
    lock_path = output_root / ".matrix.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        prior = read_json(lock_path) or {}
        prior_pid = int(prior.get("pid", -1)) if str(prior.get("pid", "")).lstrip("-").isdigit() else -1
        if pid_alive(prior_pid):
            raise RuntimeError(f"WebFG matrix is already running under PID {prior_pid}; refusing duplicate launch.")
        lock_path.unlink()
    try:
        handle = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError("WebFG matrix lock appeared during launch; refusing duplicate launch.") from exc
    with os.fdopen(handle, "w", encoding="utf-8") as file_handle:
        json.dump({"pid": os.getpid(), "started_at_utc": utc_now()}, file_handle)
    try:
        yield
    finally:
        if lock_path.exists():
            lock_path.unlink()
        remove_own_background_pid(output_root)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def coordinate_dir(output_root: Path, task: dict[str, Any]) -> Path:
    return output_root / "runs" / str(task["dataset"]) / str(task["method"]) / f"seed{int(task['seed'])}"


def next_attempt_dir(coordinate: Path) -> Path:
    coordinate.mkdir(parents=True, exist_ok=True)
    numbers = [int(path.name.split("-")[-1]) for path in coordinate.glob("attempt-*") if path.name.split("-")[-1].isdigit()]
    attempt = coordinate / f"attempt-{(max(numbers) + 1 if numbers else 1):03d}"
    attempt.mkdir()
    return attempt


def completed_attempt(coordinate: Path) -> Path | None:
    if not coordinate.is_dir():
        return None
    for attempt in sorted(coordinate.glob("attempt-*"), reverse=True):
        status = read_json(attempt / "run_status.json") or {}
        if status.get("state") in RUN_COMPLETE_STATES and (attempt / "result.json").is_file() and (attempt / "profiling.json").is_file():
            return attempt
    return None


def run_single_task(config: dict[str, Any], output_root: Path, task: dict[str, Any], attempt: Path, gpu: dict[str, Any]) -> None:
    """Execute one independent coordinate while preserving failures for resume."""
    status_path = attempt / "run_status.json"
    started = utc_now()
    write_json(status_path, {"state": "RUNNING", "task": task, "attempt": attempt.name, "started_at_utc": started, "gpu": gpu})
    write_yaml(attempt / "resolved_config.yaml", resolved_run_config(config, task))
    log_path = output_root / "logs" / f"{task['dataset']}__{task['method']}__seed{task['seed']}__{attempt.name}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_handle, contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            result, profile = execute_public_training(config, output_root, task, attempt)
        write_json(attempt / "result.json", result)
        write_json(attempt / "profiling.json", profile)
        completion_state = FORMAL_COMPLETE_STATE if profile["profiling_completeness"] == "COMPLETE" else "COMPLETE_WITH_PROFILING_GAP"
        write_json(
            status_path,
            {
                "state": completion_state,
                "task": task,
                "attempt": attempt.name,
                "started_at_utc": started,
                "finished_at_utc": utc_now(),
                "log": str(log_path),
                "final_checkpoint": str(attempt / "checkpoints" / "last.pt"),
                "profiling_completeness": profile["profiling_completeness"],
            },
        )
    except Exception as exc:  # noqa: BLE001 - failures are recorded and later tasks continue.
        with log_path.open("a", encoding="utf-8") as log_handle:
            traceback.print_exc(file=log_handle)
        write_json(
            status_path,
            {
                "state": "FAILED",
                "task": task,
                "attempt": attempt.name,
                "started_at_utc": started,
                "finished_at_utc": utc_now(),
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "log": str(log_path),
            },
        )


def resolved_run_config(config: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(config))
    payload.pop("_config_path", None)
    payload["resolved_task"] = task
    payload["resolved_at_utc"] = utc_now()
    return payload


class Measurement:
    """Per-process wall-clock and CUDA allocator peak capture without patching training."""

    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self.torch: Any | None = None
        self.cuda = False
        self.started: float | None = None

    def start(self) -> None:
        import torch

        self.torch = torch
        self.cuda = bool(torch.cuda.is_available())
        if self.cuda:
            torch.cuda.set_device(self.gpu_index)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(self.gpu_index)
        self.started = time.perf_counter()

    def finish(self) -> dict[str, Any]:
        if self.started is None or self.torch is None:
            raise RuntimeError("Measurement was not started.")
        if self.cuda:
            self.torch.cuda.synchronize()
            properties = self.torch.cuda.get_device_properties(self.gpu_index)
            return {
                "training_wall_seconds": time.perf_counter() - self.started,
                "peak_allocated_memory_gib": self.torch.cuda.max_memory_allocated(self.gpu_index) / 1024**3,
                "peak_reserved_memory_gib": self.torch.cuda.max_memory_reserved(self.gpu_index) / 1024**3,
                "hardware": {"gpu_name": properties.name, "gpu_total_memory_gib": properties.total_memory / 1024**3},
                "runtime": {"torch": self.torch.__version__, "cuda": self.torch.version.cuda},
                "measurement_status": "cuda_synchronized_allocator_peaks",
            }
        return {
            "training_wall_seconds": time.perf_counter() - self.started,
            "peak_allocated_memory_gib": None,
            "peak_reserved_memory_gib": None,
            "hardware": {"gpu_name": None, "gpu_total_memory_gib": None},
            "runtime": {"torch": self.torch.__version__, "cuda": self.torch.version.cuda},
            "measurement_status": "cpu_or_cuda_unavailable",
        }


def execute_public_training(
    config: dict[str, Any],
    output_root: Path,
    task: dict[str, Any],
    attempt: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Call only the existing public method functions; no shared-code patching."""
    import numpy as local_np

    dataset = str(task["dataset"])
    method = str(task["method"])
    seed = int(task["seed"])
    manifest = output_root / "manifests" / f"{dataset}_effective_train.csv"
    train, evaluation = load_webfg_splits(
        Path(str(config["data"]["root"])),
        dataset,
        train_manifest=manifest,
    )
    train_labels = local_np.asarray(train.labels, dtype=str)
    eval_labels = local_np.asarray(evaluation.labels, dtype=str)
    candidate_mask = local_np.ones(len(train_labels), dtype=bool)
    base_cfg = public_training_config(config, dataset, seed)
    (attempt / "checkpoints").mkdir(exist_ok=True)
    measurement = Measurement(int(config["hardware"]["gpu_index"]))
    measurement.start()
    if method == "pgdf_dynamic_proto":
        from gcdd.lora_dynamic import train_dynamic_loss_lora
        from revision_experiments.profiling import ProfilingMode, RunProfiler

        method_cfg = config["methods"][method]
        profiler = RunProfiler(
            f"webfg496_{dataset}_{method}_seed{seed}",
            ProfilingMode(str(config["hardware"].get("profiling_mode", "detailed"))),
            device_type="cuda",
        )
        result = train_dynamic_loss_lora(
            train.paths, train_labels, evaluation.paths, eval_labels, candidate_mask, base_cfg,
            method="PGDF DynamicProto", seed=seed, retention_ratio=float(method_cfg["selection_ratio"]),
            warmup_epochs=int(method_cfg["warmup_epochs"]), update_interval=int(method_cfg["selection_interval"]),
            proto_keep_ratio=float(method_cfg["prototype_keep_ratio"]), selection_strategy="loss_and_proto",
            prototype_mode="dynamic_lora", final_checkpoint_path=attempt / "checkpoints" / "last.pt",
            checkpoint_protocol="webfg496_fixed_last_epoch_v1", profiler=profiler,
        )
        measured = measurement.finish()
        write_training_tables(attempt, result.logs, result.update_rows, result.per_class_rows)
        profile = dynamic_profile(profile_base(measured, config, task, result.logs), profiler.to_dict(), result.update_rows)
        final = result.logs[-1]
        result_payload = {
            "task": task, "method_display": "PGDF DynamicProto", "checkpoint_rule": "fixed_last_epoch_30",
            "final_epoch": int(final["epoch"]), "final_top1": float(final["top1"]), "final_top5": float(final["top5"]),
            "sample_exposure": int(sum(int(row["train_samples"]) for row in result.logs)),
            "selection_updates": result.update_rows, "summary": result.summary,
        }
        return result_payload, profile
    if method in {"all_noisy_ce", "jal_ce"}:
        from gcdd.lora_training import train_dinov2_lora

        if method == "jal_ce":
            set_jal_config(base_cfg, config["methods"][method])
            method_display = "JAL-CE"
        else:
            method_display = "All-noisy CE"
        result = train_dinov2_lora(
            train.paths, train_labels, evaluation.paths, eval_labels, candidate_mask, base_cfg,
            method=method_display, seed=seed, final_checkpoint_path=attempt / "checkpoints" / "last.pt",
            full_noisy_candidate_mask=candidate_mask, checkpoint_protocol="webfg496_fixed_last_epoch_v1",
        )
        measured = measurement.finish()
        write_training_tables(attempt, result.logs, [], [])
        final = result.logs[-1]
        profile = static_profile(profile_base(measured, config, task, result.logs), method, grad_accum_steps=1)
        result_payload = {
            "task": task, "method_display": method_display, "checkpoint_rule": "fixed_last_epoch_30",
            "final_epoch": int(final["epoch"]), "final_top1": float(final["top1"]), "final_top5": float(final["top5"]),
            "sample_exposure": int(sum(int(row["train_samples"]) for row in result.logs)), "summary": result.summary,
        }
        return result_payload, profile
    if method == "coteaching":
        from gcdd.lora_noisy_baselines import train_coteaching_lora

        method_cfg = config["methods"][method]
        result = train_coteaching_lora(
            train.paths, train_labels, evaluation.paths, eval_labels, candidate_mask, base_cfg,
            method="Co-teaching", seed=seed, remember_mode=str(method_cfg["remember_mode"]),
            remember_rate=float(method_cfg["remember_rate"]), final_remember_rate=float(method_cfg["final_remember_rate"]),
            warmup_epochs=int(method_cfg["warmup_epochs"]), grad_accum_steps=int(method_cfg["grad_accum_steps"]),
            final_checkpoint_path=attempt / "checkpoints" / "last.pt", checkpoint_protocol="webfg496_fixed_last_epoch_v1",
        )
        measured = measurement.finish()
        write_training_tables(attempt, result.logs, [], [])
        final = result.logs[-1]
        profile = static_profile(profile_base(measured, config, task, result.logs), method, grad_accum_steps=int(method_cfg["grad_accum_steps"]))
        result_payload = {
            "task": task, "method_display": "Co-teaching", "checkpoint_rule": "fixed_last_epoch_30_branch_mean",
            "final_epoch": int(final["epoch"]), "final_top1": float(final["mean_ab_top1"]), "final_top5": float(final["mean_ab_top5"]),
            "final_branch_a_top1": float(final["top1_a"]), "final_branch_b_top1": float(final["top1_b"]),
            "primary_report_rule": str(method_cfg["primary_report_rule"]),
            "sample_exposure": int(sum(int(row["train_samples"]) for row in result.logs)), "summary": result.summary,
        }
        return result_payload, profile
    raise ValueError(f"Unsupported WebFG method: {method}")


def public_training_config(config: dict[str, Any], dataset: str, seed: int) -> dict[str, Any]:
    model = config["model"]
    train = config["training"]
    return {
        "dataset": {"name": dataset},
        "feature": {"backend": str(model["backbone"]), "device": str(config["hardware"]["device"]), "input_size": int(model["feature_input_size"]), "local_repo": ""},
        "lora": {"rank": int(model["lora_rank"]), "alpha": float(model["lora_alpha"]), "dropout": float(model["lora_dropout"]), "target_modules": str(model["lora_target_modules"])},
        "lora_train": {
            "epochs": int(train["epochs"]), "batch_size": int(train["batch_size"]), "eval_batch_size": int(train["eval_batch_size"]),
            "num_workers": int(train["num_workers"]), "pin_memory": bool(train["pin_memory"]), "lora_lr": float(train["lora_lr"]),
            "head_lr": float(train["head_lr"]), "weight_decay": float(train["weight_decay"]), "scheduler": str(train["scheduler"]),
            "warmup_ratio": float(train["warmup_ratio"]), "amp": bool(train["amp"]), "seed": seed,
        },
        "loss_type": "ce",
    }


def set_jal_config(cfg: dict[str, Any], method_cfg: dict[str, Any]) -> None:
    cfg.update({"loss_type": "jal_ce", "jal_alpha": float(method_cfg["jal_alpha"]), "jal_beta": float(method_cfg["jal_beta"]), "jal_a": float(method_cfg["jal_a"]), "jal_eps": float(method_cfg["jal_eps"])})
    cfg["jal_baseline"] = {"selection": "full_noisy", "use_filter": False, "dynamic_selection": False, "prototype_gate": False, "use_pgdf": False, "use_centroid_filter": False, "use_gcdd_budget": False}


def write_training_tables(attempt: Path, logs: list[dict[str, Any]], updates: list[dict[str, Any]], per_class: list[dict[str, Any]]) -> None:
    log_fields = sorted({key for row in logs for key in row})
    update_fields = sorted({key for row in updates for key in row}) or ["epoch"]
    class_fields = sorted({key for row in per_class for key in row}) or ["epoch"]
    write_csv(attempt / "train_log.csv", logs, log_fields)
    write_csv(attempt / "selection_updates.csv", updates, update_fields)
    write_csv(attempt / "selection_per_class.csv", per_class, class_fields)


def profile_base(measured: dict[str, Any], config: dict[str, Any], task: dict[str, Any], logs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": task,
        "recorded_at_utc": utc_now(),
        "timing_scope": "public_training_function_call_with_cuda_synchronization_at_boundaries",
        "training_wall_seconds": float(measured["training_wall_seconds"]),
        "peak_allocated_memory_gib": measured["peak_allocated_memory_gib"],
        "peak_reserved_memory_gib": measured["peak_reserved_memory_gib"],
        "hardware": measured["hardware"], "runtime": measured["runtime"], "measurement_status": measured["measurement_status"],
        "effective_batch": {"batch_size": int(config["training"]["batch_size"]), "gradient_accumulation_steps": 1},
        "sample_exposure": int(sum(int(row["train_samples"]) for row in logs)),
        "epoch_count": len(logs), "checkpoint_rule": "fixed_last_epoch_30",
    }


def static_profile(base: dict[str, Any], method: str, *, grad_accum_steps: int) -> dict[str, Any]:
    samples_per_epoch = int(base["sample_exposure"]) // max(1, int(base["epoch_count"]))
    attempted = int(base["epoch_count"]) * math.ceil(
        math.ceil(samples_per_epoch / int(base["effective_batch"]["batch_size"])) / max(1, int(grad_accum_steps))
    )
    # The public static APIs do not expose GradScaler's overflow result.  We
    # retain exact attempted counts from log-visible batches but deliberately
    # leave successful-update counts unavailable instead of fabricating them.
    base["effective_batch"]["gradient_accumulation_steps"] = int(grad_accum_steps)
    base.update({
        "optimizer_steps_attempted": attempted,
        "optimizer_steps_actual": None,
        "optimizer_steps_actual_status": f"{PROFILING_GAP}: public {method} trainer has no observer/callback for GradScaler successful-update events.",
        "scheduler_steps_actual": None,
        "scheduler_steps_actual_status": f"{PROFILING_GAP}: public {method} trainer does not expose per-call scheduler instrumentation.",
        "selection": {"selection_update_count": 0, "selection_total_seconds": 0.0, "selection_fraction_of_training": 0.0},
        "profiling_completeness": "PARTIAL_SHARED_TRAINER_OBSERVER_GAP",
        "profiling_limit": "No shared training file was modified and no runtime monkey patch was applied. Formal completion is therefore blocked until explicit approval for observer hooks in existing public static trainers.",
    })
    return base


def dynamic_profile(base: dict[str, Any], profiler_payload: dict[str, Any], updates: list[dict[str, Any]]) -> dict[str, Any]:
    totals = profiler_payload.get("stage_totals", {})
    duration = lambda name: float(totals.get(name, {}).get("duration_seconds", 0.0))
    selection_total = duration("selection_update_total")
    events = profiler_payload.get("detailed_events", [])
    selection_updates: list[dict[str, Any]] = []
    for update in updates:
        phase = f"epoch_{int(update['epoch'])}"
        stage_durations: dict[str, float] = {}
        for stage in ("selection_loss_forward", "prototype_construction", "fusion_selection", "selection_update_total"):
            stage_durations[stage] = sum(float(event.get("duration_seconds", 0.0)) for event in events if event.get("name") == stage and event.get("phase") == phase)
        selection_updates.append({
            "epoch": int(update["epoch"]), "active_subset_size": int(update["num_selected"]), "fallback_count": int(update.get("fallback_count", 0)),
            "full_pool_loss_forward_seconds": stage_durations["selection_loss_forward"],
            "cls_feature_and_prototype_seconds": stage_durations["prototype_construction"],
            "prototype_filter_seconds": stage_durations["fusion_selection"],
            "selection_total_seconds": stage_durations["selection_update_total"],
        })
    base.update({
        "optimizer_steps_attempted": profiler_payload.get("optimizer_steps_attempted"),
        "optimizer_steps_actual": profiler_payload.get("optimizer_steps_actual"),
        "scheduler_steps_actual": profiler_payload.get("scheduler_steps_actual"),
        "training_batches_seen": profiler_payload.get("training_batches_seen"),
        "selection": {
            "selection_update_count": len(selection_updates), "selection_total_seconds": selection_total,
            "selection_fraction_of_training": selection_total / base["training_wall_seconds"] if base["training_wall_seconds"] else None,
            "full_pool_loss_forward_seconds": duration("selection_loss_forward"),
            "cls_feature_and_prototype_seconds": duration("prototype_construction"),
            "prototype_filter_seconds": duration("fusion_selection"),
            "updates": selection_updates,
        },
        "revision_profiler": profiler_payload,
        "profiling_completeness": "COMPLETE",
    })
    return base


def aggregate_outputs(output_root: Path, config: dict[str, Any], run_state: str | None = None) -> None:
    inventory: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    for task in build_matrix(config):
        coordinate = coordinate_dir(output_root, task)
        attempts = sorted(coordinate.glob("attempt-*")) if coordinate.is_dir() else []
        latest_status = "PENDING"
        for attempt in attempts:
            status = read_json(attempt / "run_status.json") or {"state": "CORRUPT_STATUS"}
            row = {**task, "attempt": attempt.name, "state": status.get("state", "UNKNOWN"), "started_at_utc": status.get("started_at_utc", ""), "finished_at_utc": status.get("finished_at_utc", ""), "run_dir": str(attempt), "log": status.get("log", ""), "profiling_completeness": status.get("profiling_completeness", "")}
            inventory.append(row)
            latest_status = str(status.get("state", "UNKNOWN"))
        chosen = completed_attempt(coordinate)
        if chosen is not None:
            result = read_json(chosen / "result.json") or {}
            profile = read_json(chosen / "profiling.json") or {}
            completed.append({"task": task, "attempt": chosen, "result": result, "profile": profile})
        elif not attempts:
            inventory.append({**task, "attempt": "", "state": "PENDING", "started_at_utc": "", "finished_at_utc": "", "run_dir": str(coordinate), "log": "", "profiling_completeness": ""})
        elif latest_status == "RUNNING":
            pass
    inventory_fields = ["dataset", "method", "seed", "attempt", "state", "profiling_completeness", "started_at_utc", "finished_at_utc", "run_dir", "log"]
    write_csv(output_root / "training_run_inventory.csv", inventory, inventory_fields)
    accuracy_rows = [
        {"dataset": item["task"]["dataset"], "method": item["task"]["method"], "seed": item["task"]["seed"], "attempt": item["attempt"].name,
         "final_epoch": item["result"].get("final_epoch"), "final_top1": item["result"].get("final_top1"), "final_top5": item["result"].get("final_top5"),
         "checkpoint_rule": item["result"].get("checkpoint_rule"), "primary_report_rule": item["result"].get("primary_report_rule", ""), "run_dir": str(item["attempt"])}
        for item in completed
    ]
    write_csv(output_root / "seed_level_accuracy.csv", accuracy_rows, ["dataset", "method", "seed", "attempt", "final_epoch", "final_top1", "final_top5", "checkpoint_rule", "primary_report_rule", "run_dir"])
    profile_rows = [profile_row(item) for item in completed]
    write_csv(output_root / "seed_level_profiling.csv", profile_rows, profiling_fields())
    write_csv(output_root / "accuracy_summary.csv", accuracy_summary(accuracy_rows), ["dataset", "method", "completed_seeds", "mean_final_top1", "std_final_top1", "min_final_top1", "max_final_top1"])
    write_csv(output_root / "compute_cost_summary.csv", compute_summary(profile_rows), ["dataset", "method", "completed_seeds", "mean_training_wall_seconds", "mean_peak_allocated_memory_gib", "mean_peak_reserved_memory_gib", "mean_sample_exposure", "mean_selection_total_seconds", "mean_selection_fraction_of_training", "profiling_completeness"])
    write_final_report(output_root, inventory, accuracy_rows, profile_rows, run_state)


def profile_row(item: dict[str, Any]) -> dict[str, Any]:
    profile = item["profile"]
    selection = profile.get("selection", {})
    return {
        "dataset": item["task"]["dataset"], "method": item["task"]["method"], "seed": item["task"]["seed"], "attempt": item["attempt"].name,
        "training_wall_seconds": profile.get("training_wall_seconds"), "peak_allocated_memory_gib": profile.get("peak_allocated_memory_gib"), "peak_reserved_memory_gib": profile.get("peak_reserved_memory_gib"),
        "gpu_name": profile.get("hardware", {}).get("gpu_name"), "gpu_total_memory_gib": profile.get("hardware", {}).get("gpu_total_memory_gib"),
        "optimizer_steps_attempted": profile.get("optimizer_steps_attempted"), "optimizer_steps_actual": profile.get("optimizer_steps_actual"), "scheduler_steps_actual": profile.get("scheduler_steps_actual"),
        "sample_exposure": profile.get("sample_exposure"), "batch_size": profile.get("effective_batch", {}).get("batch_size"), "gradient_accumulation_steps": profile.get("effective_batch", {}).get("gradient_accumulation_steps"),
        "selection_update_count": selection.get("selection_update_count"), "selection_total_seconds": selection.get("selection_total_seconds"), "selection_fraction_of_training": selection.get("selection_fraction_of_training"),
        "full_pool_loss_forward_seconds": selection.get("full_pool_loss_forward_seconds", ""), "cls_feature_and_prototype_seconds": selection.get("cls_feature_and_prototype_seconds", ""), "prototype_filter_seconds": selection.get("prototype_filter_seconds", ""),
        "profiling_completeness": profile.get("profiling_completeness"), "profiling_limit": profile.get("profiling_limit", ""), "run_dir": str(item["attempt"]),
    }


def profiling_fields() -> list[str]:
    return ["dataset", "method", "seed", "attempt", "training_wall_seconds", "peak_allocated_memory_gib", "peak_reserved_memory_gib", "gpu_name", "gpu_total_memory_gib", "optimizer_steps_attempted", "optimizer_steps_actual", "scheduler_steps_actual", "sample_exposure", "batch_size", "gradient_accumulation_steps", "selection_update_count", "selection_total_seconds", "selection_fraction_of_training", "full_pool_loss_forward_seconds", "cls_feature_and_prototype_seconds", "prototype_filter_seconds", "profiling_completeness", "profiling_limit", "run_dir"]


def accuracy_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["method"])), []).append(float(row["final_top1"]))
    return [
        {"dataset": key[0], "method": key[1], "completed_seeds": len(values), "mean_final_top1": float(np.mean(values)), "std_final_top1": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "min_final_top1": float(np.min(values)), "max_final_top1": float(np.max(values))}
        for key, values in sorted(grouped.items())
    ]


def compute_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    def average(items: list[dict[str, Any]], field: str) -> float | str:
        values = [float(item[field]) for item in items if item.get(field) not in (None, "")]
        return float(np.mean(values)) if values else ""
    return [
        {"dataset": key[0], "method": key[1], "completed_seeds": len(items), "mean_training_wall_seconds": average(items, "training_wall_seconds"), "mean_peak_allocated_memory_gib": average(items, "peak_allocated_memory_gib"), "mean_peak_reserved_memory_gib": average(items, "peak_reserved_memory_gib"), "mean_sample_exposure": average(items, "sample_exposure"), "mean_selection_total_seconds": average(items, "selection_total_seconds"), "mean_selection_fraction_of_training": average(items, "selection_fraction_of_training"), "profiling_completeness": ";".join(sorted({str(item.get("profiling_completeness", "")) for item in items}))}
        for key, items in sorted(grouped.items())
    ]


def write_final_report(output_root: Path, inventory: list[dict[str, Any]], accuracy: list[dict[str, Any]], profiling: list[dict[str, Any]], run_state: str | None) -> None:
    complete = sum(1 for row in inventory if row["state"] == "COMPLETE")
    gap = sum(1 for row in inventory if row["state"] == "COMPLETE_WITH_PROFILING_GAP")
    failed = sum(1 for row in inventory if row["state"] == "FAILED")
    pending = sum(1 for row in inventory if row["state"] == "PENDING")
    overall = f"COMPLETE: 36/36" if complete == 36 else (run_state or f"PARTIAL: formal={complete}/36, training_complete_with_profiling_gap={gap}, failed={failed}, pending={pending}")
    resolution_rows = read_csv_rows(output_root / "corrupt_image_resolution.csv")
    excluded_rows = [row for row in resolution_rows if row.get("final_action") == "exclude_confirmed_unrecoverable"]
    recovered_rows = [row for row in resolution_rows if row.get("final_action") == "use_recovered_original"]
    data_rows = read_csv_rows(output_root / "dataset_inventory.csv")
    train_rows = [row for row in data_rows if row.get("split") == "train"]
    lines = [
        "# WebFG-496 Final Report", "", f"- Overall status: **{overall}**",
        "- Formal-complete requires all 36 runs and complete actual optimizer/scheduler profiling.",
        "- Every result here is isolated under this WebFG output root; historical experiment output is not read or written.",
        "", "## Corrupt-image resolution", "",
        f"- Requested corrupt files: {len(resolution_rows)}; recovered raw originals: {len(recovered_rows)}; fixed exclusions: {len(excluded_rows)}.",
        "- Original data were not changed. Archive bytes and local bytes were SHA-256 compared and fully decoded; no truncated-image tolerance was enabled.",
        "- Shared manifest is the sole training-pool source for PGDF, All-noisy CE, JAL-CE, and Co-teaching; the public trainers receive paths directly and do not rescan directories.",
        "", "| dataset | effective training images | fixed exclusions |", "| --- | ---: | ---: |",
    ]
    for row in sorted(train_rows, key=lambda item: str(item.get("dataset", ""))):
        lines.append(f"| {row['dataset']} | {row.get('effective_list_entries', '')} | {row.get('excluded_from_effective_train', '')} |")
    lines.extend(["", "## Matrix status", "", f"- COMPLETE: {complete}/36", f"- COMPLETE_WITH_PROFILING_GAP: {gap}/36", f"- FAILED attempts: {failed}", f"- PENDING coordinates: {pending}", "", "## Accuracy", "", "| dataset | method | seed | final Top-1 | final epoch |", "| --- | --- | ---: | ---: | ---: |"])
    for row in accuracy:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['seed']} | {row['final_top1']} | {row['final_epoch']} |")
    lines.extend(["", "## Profiling", "", "| dataset | method | seed | wall seconds | allocated GiB | selection seconds | profiling |", "| --- | --- | ---: | ---: | ---: | --- |"])
    for row in profiling:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['seed']} | {row['training_wall_seconds']} | {row['peak_allocated_memory_gib']} | {row['selection_total_seconds']} | {row['profiling_completeness']} |")
    lines.extend(["", "## Known formal blocker", "", f"- `{PROFILING_GAP}` applies to the existing All-noisy CE, JAL-CE, and Co-teaching public trainer interfaces: they expose no record-only optimizer/scheduler callback or actual GradScaler-update outcome. This isolated adapter neither patches them at runtime nor modifies shared source. PGDF has existing detailed observer support."])
    atomic_text(output_root / "WEBFG496_FINAL_REPORT.md", "\n".join(lines) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def print_status(output_root: Path) -> None:
    path = output_root / "training_run_inventory.csv"
    if not path.is_file():
        print(f"NO_WEBFG_STATUS: {path} does not exist")
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {state: sum(1 for row in rows if row.get("state") == state) for state in sorted({row.get("state", "") for row in rows})}
    print(json.dumps({"output_root": str(output_root), "status_counts": counts, "inventory": str(path)}, ensure_ascii=False, indent=2))


def print_summary(output_root: Path) -> None:
    for name in ("accuracy_summary.csv", "compute_cost_summary.csv", "WEBFG496_FINAL_REPORT.md"):
        path = output_root / name
        if path.is_file():
            print(path.read_text(encoding="utf-8"))
        else:
            print(f"NO_WEBFG_SUMMARY: {path}")


def launch_background(args: argparse.Namespace, config: dict[str, Any], output_root: Path) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    pid_path = logs / "background_launcher.pid"
    if pid_path.exists():
        prior = read_json(pid_path) or {}
        if pid_alive(int(prior.get("pid", -1)) if str(prior.get("pid", "")).lstrip("-").isdigit() else -1):
            raise RuntimeError(f"WebFG background launcher is already active: PID {prior.get('pid')}")
        pid_path.unlink()
    command = [sys.executable, "-m", "experiments.webfg496.runner", "--config", str(config["_config_path"]), "--execute", "--resume"]
    if args.data_root:
        command.extend(["--data-root", args.data_root])
    if args.output_root:
        command.extend(["--output-root", args.output_root])
    log_path = logs / f"background_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(pid_path, {"pid": process.pid, "started_at_utc": utc_now(), "log": str(log_path), "command": command})
    print(json.dumps({"status": "BACKGROUND_STARTED", "pid": process.pid, "log": str(log_path)}, ensure_ascii=False))
    return 0


def remove_own_background_pid(output_root: Path) -> None:
    path = output_root / "logs" / "background_launcher.pid"
    payload = read_json(path)
    if payload is not None and int(payload.get("pid", -1)) == os.getpid():
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
