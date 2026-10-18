"""Prepare or sequentially run the formal IDN/GCCN generalization matrix.

The launcher owns orchestration only.  Noise construction and model training
remain in ``tools/prepare_noise_generalization.py`` and
``scripts/run_lora_checkpoint_validation.py`` respectively.  A task is skipped
only after its saved result, validation protocol, checkpoint, epoch log, and
noise-realization metadata pass a strict completion audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.checkpoint_validation import (
    PROTOCOL_NAME,
    build_or_load_fixed_validation_split,
    load_clean_labels_from_noise_index,
)
from gcdd.noise_generalization import (
    GCCN_STRATEGY,
    IDN_STRATEGY,
    validate_noise_generalization_training_input,
)


SCHEMA_VERSION = 1
DATASET_ORDER = ("cub", "cars", "aircraft")
DATASET_NAMES = {
    "cub": "CUB-200-2011",
    "cars": "Stanford Cars",
    "aircraft": "FGVC-Aircraft",
}
PROTOCOLS = ("idn", "gccn")
SINGLE_BRANCH_METHODS = ("pgdf_dynamic_proto", "all_noisy", "dynamic_r08", "jal_ce", "fine")
DUAL_BRANCH_METHODS = ("coteaching", "jocor")
METHODS = (*SINGLE_BRANCH_METHODS, *DUAL_BRANCH_METHODS)
TRAINING_SEEDS = (1, 42, 88)
NOISE_RATE = 0.4
NOISE_SEED = 42
VALIDATION_SEED = 20250726
EPOCHS = 30
WARMUP_EPOCHS = 5
UPDATE_INTERVAL = 5
DYNAMIC_RATIO = 0.8
FIXED_P = 0.4
IDN_PROBABILITY_MODULATION = 0.5
BASE_V1_REQUIRED = (
    "paths.txt",
    "eval_paths.txt",
    "eval_labels.npy",
    "features_cls.npy",
    "features_gap.npy",
    "features_top.npy",
    "resolved_config.yaml",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    base_input_dir: Path
    source_noise_index: Path
    validation_dir: Path
    path_maps: tuple[str, ...]
    training_config: Path | None


@dataclass(frozen=True)
class InstrumentationSpec:
    profiling_mode: str
    selection_logging: bool
    output_root: Path | None


@dataclass(frozen=True)
class ExperimentConfig:
    config_path: Path
    python_executable: str
    noise_assets_root: Path
    training_outputs_root: Path
    instrumentation: InstrumentationSpec
    datasets: tuple[DatasetSpec, ...]


@dataclass(frozen=True)
class Task:
    dataset: DatasetSpec
    protocol: str
    method: str
    seed: int
    artifact_dir: Path
    run_root: Path
    run_dir: Path

    @property
    def task_id(self) -> str:
        return f"{self.dataset.key}__{self.protocol}__{self.method}__seed{self.seed}"

    @property
    def single_branch(self) -> bool:
        return self.method in SINGLE_BRANCH_METHODS

    @property
    def expected_noise_strategy(self) -> str:
        return IDN_STRATEGY if self.protocol == "idn" else GCCN_STRATEGY

    @property
    def expected_test_protocol(self) -> str:
        return "validation_selected_only" if self.single_branch else "dual_validation_selected_mean_with_final_last5_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Server-specific YAML based on configs/noise_generalization_all.example.yaml.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print all 6 preparation plus 126 training commands without launching subprocesses.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create or strictly validate the 6 IDN/GCCN noise assets, then stop before any training subprocess.",
    )
    parser.add_argument("--summarize-only", action="store_true", help="Read strictly complete runs only and refresh status/summary files without generating data or training.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True, help="Default: skip only strictly verified completed tasks.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Fail if any task is already complete instead of skipping it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_modes = sum(bool(value) for value in (args.dry_run, args.prepare_only, args.summarize_only))
    if selected_modes > 1:
        raise ValueError("Use at most one of --dry-run, --prepare-only, and --summarize-only.")
    config = load_config(args.config)
    run_orchestration(
        config,
        dry_run=bool(args.dry_run),
        prepare_only=bool(args.prepare_only),
        summarize_only=bool(args.summarize_only),
        resume=bool(args.resume),
    )


def load_config(path: Path) -> ExperimentConfig:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Noise-generalization configuration must be a YAML mapping.")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION}.")

    config_dir = path.parent
    runtime = require_mapping(payload, "runtime")
    output = require_mapping(payload, "output")
    instrumentation_data = require_mapping(payload, "instrumentation")
    formal = require_mapping(payload, "formal")
    _validate_formal_contract(formal)

    python_executable = require_string(runtime, "python_executable")
    noise_assets_root = resolve_path(require_string(output, "noise_assets_root"), config_dir)
    training_outputs_root = resolve_path(require_string(output, "training_outputs_root"), config_dir)
    profiling_mode = require_string(instrumentation_data, "profiling_mode")
    if profiling_mode not in {"off", "basic", "detailed"}:
        raise ValueError("instrumentation.profiling_mode must be one of off, basic, detailed.")
    selection_logging = require_bool(instrumentation_data, "selection_logging")
    raw_instrumentation_root = instrumentation_data.get("output_root")
    instrumentation_root = None if raw_instrumentation_root in (None, "") else resolve_path(str(raw_instrumentation_root), config_dir)
    if (profiling_mode != "off" or selection_logging) and instrumentation_root is None:
        raise ValueError("instrumentation.output_root is required when D1 or D2 is enabled.")

    datasets_data = require_mapping(payload, "datasets")
    if set(datasets_data) != set(DATASET_ORDER):
        raise ValueError(f"datasets must contain exactly {list(DATASET_ORDER)}.")
    datasets: list[DatasetSpec] = []
    for key in DATASET_ORDER:
        item = require_mapping(datasets_data, key)
        display_name = require_string(item, "display_name")
        if display_name != DATASET_NAMES[key]:
            raise ValueError(f"datasets.{key}.display_name must be {DATASET_NAMES[key]!r}.")
        path_maps_data = item.get("path_maps", [])
        if not isinstance(path_maps_data, list) or not all(isinstance(value, str) and "=" in value for value in path_maps_data):
            raise ValueError(f"datasets.{key}.path_maps must be a list of OLD=NEW strings.")
        raw_training_config = item.get("training_config")
        training_config = None if raw_training_config in (None, "") else resolve_path(str(raw_training_config), config_dir)
        datasets.append(
            DatasetSpec(
                key=key,
                display_name=display_name,
                base_input_dir=resolve_path(require_string(item, "base_input_dir"), config_dir),
                source_noise_index=resolve_path(require_string(item, "source_noise_index"), config_dir),
                validation_dir=resolve_path(require_string(item, "validation_dir"), config_dir),
                path_maps=tuple(path_maps_data),
                training_config=training_config,
            )
        )
    return ExperimentConfig(
        config_path=path,
        python_executable=python_executable,
        noise_assets_root=noise_assets_root,
        training_outputs_root=training_outputs_root,
        instrumentation=InstrumentationSpec(profiling_mode, selection_logging, instrumentation_root),
        datasets=tuple(datasets),
    )


def run_orchestration(
    config: ExperimentConfig,
    *,
    dry_run: bool,
    prepare_only: bool,
    summarize_only: bool,
    resume: bool,
) -> None:
    plan = build_plan(config)
    if len(plan) != 126:
        raise RuntimeError(f"Formal plan must contain exactly 126 tasks, found {len(plan)}.")
    if summarize_only:
        write_global_metadata(config, plan)
        write_status_and_summary(config, plan)
        print("SUMMARY PASS: no data preparation or training process was started.", flush=True)
        return

    validate_runtime(config)
    validate_all_dataset_inputs(config.datasets)
    for dataset in config.datasets:
        ensure_fixed_validation_manifest(dataset, dry_run=dry_run)
    if prepare_only:
        for dataset in config.datasets:
            for protocol in PROTOCOLS:
                prepare_or_validate_artifact(config, dataset, protocol, artifact_directory(config, dataset, protocol))
        print("PREPARATION COMPLETE: 6 IDN/GCCN assets are ready; no training subprocess was started.", flush=True)
        return

    write_global_metadata(config, plan)
    write_status_and_summary(config, plan)
    print_plan(config, plan, dry_run=dry_run)
    if dry_run:
        print("DRY RUN PASS: no noise asset or training subprocess was started.", flush=True)
        return

    for dataset in config.datasets:
        for protocol in PROTOCOLS:
            artifact_dir = artifact_directory(config, dataset, protocol)
            prepare_or_validate_artifact(config, dataset, protocol, artifact_dir)
            for task in (item for item in plan if item.dataset.key == dataset.key and item.protocol == protocol):
                try:
                    run_or_skip_task(config, task, resume=resume)
                except Exception:
                    write_status_and_summary(config, plan)
                    raise
                write_status_and_summary(config, plan)
    write_status_and_summary(config, plan)
    completed = sum(is_task_complete(task)[0] for task in plan)
    print(f"FORMAL MATRIX COMPLETE: {completed}/{len(plan)} strict task completions.", flush=True)


def build_plan(config: ExperimentConfig) -> list[Task]:
    tasks: list[Task] = []
    for dataset in config.datasets:
        for protocol in PROTOCOLS:
            artifact_dir = artifact_directory(config, dataset, protocol)
            run_root = training_directory(config, dataset, protocol)
            for method in METHODS:
                for seed in TRAINING_SEEDS:
                    tasks.append(
                        Task(
                            dataset=dataset,
                            protocol=protocol,
                            method=method,
                            seed=seed,
                            artifact_dir=artifact_dir,
                            run_root=run_root,
                            run_dir=run_root / method / f"seed{seed}",
                        )
                    )
    return tasks


def artifact_directory(config: ExperimentConfig, dataset: DatasetSpec, protocol: str) -> Path:
    return config.noise_assets_root / dataset.key / protocol_name(protocol)


def training_directory(config: ExperimentConfig, dataset: DatasetSpec, protocol: str) -> Path:
    return config.training_outputs_root / dataset.key / protocol_name(protocol)


def protocol_name(protocol: str) -> str:
    return f"{protocol}_r0p4_s{NOISE_SEED}"


def validate_runtime(config: ExperimentConfig) -> None:
    executable = Path(config.python_executable).expanduser()
    if executable.is_absolute() and not executable.is_file():
        raise FileNotFoundError(f"Configured Python executable does not exist: {executable}")
    if not executable.is_absolute() and shutil.which(config.python_executable) is None:
        raise FileNotFoundError(f"Configured Python executable is not on PATH: {config.python_executable}")
    roots = [config.noise_assets_root.resolve(), config.training_outputs_root.resolve()]
    if config.instrumentation.output_root is not None:
        roots.append(config.instrumentation.output_root.resolve())
    if len(set(roots)) != len(roots):
        raise ValueError("Noise assets, training outputs, and instrumentation outputs must use distinct roots.")


def validate_all_dataset_inputs(datasets: Iterable[DatasetSpec]) -> None:
    for dataset in datasets:
        if not dataset.base_input_dir.is_dir():
            raise FileNotFoundError(f"{dataset.key}: base_input_dir is missing: {dataset.base_input_dir}")
        missing = [str(dataset.base_input_dir / name) for name in BASE_V1_REQUIRED if not (dataset.base_input_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset.key}: base V1 input is incomplete: {missing}")
        if not dataset.source_noise_index.is_file():
            raise FileNotFoundError(f"{dataset.key}: source_noise_index is missing: {dataset.source_noise_index}")
        if dataset.training_config is not None and not dataset.training_config.is_file():
            raise FileNotFoundError(f"{dataset.key}: training_config is missing: {dataset.training_config}")


def ensure_fixed_validation_manifest(dataset: DatasetSpec, *, dry_run: bool) -> str:
    """Create or validate the protocol-owned fixed clean validation split.

    This is deliberately independent of every training result, including the
    dynamic-prototype fusion references.  It uses only source-train paths and
    clean labels from the source noise index, which is the established
    ``fixed_clean_validation_v1`` construction rule.
    """

    train_paths = (dataset.base_input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    clean_labels = load_clean_labels_from_noise_index(dataset.source_noise_index, train_paths)
    validation_csv, validation_json = validation_manifest_paths(dataset.validation_dir)
    if validation_csv.is_file() != validation_json.is_file():
        raise FileNotFoundError(
            f"{dataset.key}: fixed validation manifest is only partially present under {dataset.validation_dir}."
        )
    if validation_csv.is_file():
        build_or_load_fixed_validation_split(
            dataset.validation_dir,
            train_paths,
            clean_labels,
            validation_ratio=0.10,
            validation_seed=VALIDATION_SEED,
        )
        print(f"REUSE VALIDATED FIXED VALIDATION dataset={dataset.key} path={dataset.validation_dir}", flush=True)
        return "REUSED_VALIDATED"

    for label in sorted(set(clean_labels.tolist())):
        if int(np.sum(clean_labels == label)) < 2:
            raise ValueError(
                f"{dataset.key}: cannot construct fixed validation split; clean class {label!r} has fewer than two samples."
            )
    if dataset.validation_dir.exists() and any(dataset.validation_dir.iterdir()):
        raise FileExistsError(
            f"{dataset.key}: independent fixed validation directory is non-empty but has no manifest: "
            f"{dataset.validation_dir}"
        )
    if dry_run:
        print(f"PLAN CREATE FIXED VALIDATION dataset={dataset.key} path={dataset.validation_dir}", flush=True)
        return "WILL_CREATE"

    dataset.validation_dir.mkdir(parents=True, exist_ok=True)
    build_or_load_fixed_validation_split(
        dataset.validation_dir,
        train_paths,
        clean_labels,
        validation_ratio=0.10,
        validation_seed=VALIDATION_SEED,
    )
    print(f"CREATED FIXED VALIDATION dataset={dataset.key} path={dataset.validation_dir}", flush=True)
    return "CREATED"


def validation_manifest_paths(root: Path) -> tuple[Path, Path]:
    return root / "validation_manifest.csv", root / "validation_manifest.json"


def prepare_or_validate_artifact(config: ExperimentConfig, dataset: DatasetSpec, protocol: str, artifact_dir: Path) -> None:
    if artifact_dir.exists():
        metadata = validate_noise_artifact(dataset, protocol, artifact_dir)
        write_json(
            artifact_dir / "orchestration_asset_state.json",
            {
                "status": "REUSED_VALIDATED",
                "timestamp_utc": utc_now(),
                "artifact_validation": metadata,
            },
        )
        print(f"REUSE VALIDATED ASSET dataset={dataset.key} protocol={protocol} path={artifact_dir}", flush=True)
        return

    command = build_prepare_command(config, dataset, protocol, artifact_dir)
    log_path = config.noise_assets_root / "logs" / f"prepare__{dataset.key}__{protocol}.log"
    print(f"PREPARE dataset={dataset.key} protocol={protocol}", flush=True)
    return_code = run_logged_command(command, log_path)
    if return_code != 0:
        raise RuntimeError(f"Noise preparation failed for {dataset.key}/{protocol} with exit code {return_code}. See {log_path}")
    metadata = validate_noise_artifact(dataset, protocol, artifact_dir)
    write_json(
        artifact_dir / "orchestration_asset_state.json",
        {
            "status": "GENERATED_VALIDATED",
            "timestamp_utc": utc_now(),
            "prepare_command": command,
            "artifact_validation": metadata,
        },
    )


def build_prepare_command(config: ExperimentConfig, dataset: DatasetSpec, protocol: str, artifact_dir: Path) -> list[str]:
    command = [
        config.python_executable,
        str(ROOT / "tools" / "prepare_noise_generalization.py"),
        "--protocol", protocol,
        "--dataset", dataset.display_name,
        "--base-input-dir", str(dataset.base_input_dir),
        "--source-noise-index", str(dataset.source_noise_index),
        "--validation-dir", str(dataset.validation_dir),
        "--output-dir", str(artifact_dir),
        "--noise-rate", str(NOISE_RATE),
        "--noise-seed", str(NOISE_SEED),
        "--validation-seed", str(VALIDATION_SEED),
    ]
    if protocol == "idn":
        command.extend(["--idn-probability-modulation", str(IDN_PROBABILITY_MODULATION)])
    return command


def validate_noise_artifact(dataset: DatasetSpec, protocol: str, artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "noise_manifest.csv"
    statistics_path = artifact_dir / "noise_statistics.json"
    adapter_path = artifact_dir / "prepared_input" / "adapter_metadata.json"
    prepared_input = artifact_dir / "prepared_input"
    protocol_dir = artifact_dir / "protocol"
    required = [manifest_path, statistics_path, adapter_path, protocol_dir / "validation_manifest.csv", protocol_dir / "validation_manifest.json"]
    required.extend(prepared_input / name for name in ("paths.txt", "labels.npy", "eval_paths.txt", "eval_labels.npy", "resolved_config.yaml"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Existing IDN/GCCN asset is incomplete and will not be overwritten: {missing}")
    statistics_payload = read_json(statistics_path)
    adapter_payload = read_json(adapter_path)
    expected_strategy = IDN_STRATEGY if protocol == "idn" else GCCN_STRATEGY
    expected = {
        "dataset": dataset.display_name,
        "noise_protocol": protocol,
        "noise_strategy": expected_strategy,
        "requested_noise_rate": NOISE_RATE,
        "noise_seed": NOISE_SEED,
        "validation_seed": VALIDATION_SEED,
    }
    mismatches = [key for key, value in expected.items() if not equivalent(statistics_payload.get(key), value)]
    manifest_sha256 = sha256_file(manifest_path)
    if statistics_payload.get("manifest_sha256") != manifest_sha256:
        mismatches.append("manifest_sha256")
    sha_file = artifact_dir / "noise_manifest.csv.sha256"
    sha_tokens = sha_file.read_text(encoding="ascii").split() if sha_file.is_file() else []
    if not sha_tokens or sha_tokens[0] != manifest_sha256:
        mismatches.append("noise_manifest.csv.sha256")
    if adapter_payload.get("manifest_sha256") != manifest_sha256:
        mismatches.append("adapter_manifest_sha256")
    if statistics_payload.get("source_noise_index_sha256") != sha256_file(dataset.source_noise_index):
        mismatches.append("source_noise_index_sha256")
    if (prepared_input / "paths.txt").read_bytes() != (dataset.base_input_dir / "paths.txt").read_bytes():
        mismatches.append("prepared_paths.txt")
    for name in ("validation_manifest.csv", "validation_manifest.json"):
        if (protocol_dir / name).read_bytes() != (dataset.validation_dir / name).read_bytes():
            mismatches.append(f"protocol/{name}")
    labels = np.load(prepared_input / "labels.npy", allow_pickle=True).astype(str)
    train_paths = (prepared_input / "paths.txt").read_text(encoding="utf-8").splitlines()
    alignment = validate_noise_generalization_training_input(
        manifest_path=manifest_path,
        train_paths=train_paths,
        train_labels=labels,
        labels_path=prepared_input / "labels.npy",
    )
    if alignment["noise_strategy"] != expected_strategy:
        mismatches.append("prepared_observed_label_strategy")
    if mismatches:
        raise ValueError(f"Existing IDN/GCCN asset does not match the requested realization: {sorted(mismatches)}")
    return {
        "manifest_sha256": manifest_sha256,
        "labels_sha256": alignment["labels_sha256"],
        "noise_strategy": expected_strategy,
        "realized_noise_rate": statistics_payload.get("realized_noise_rate"),
        "source_train_sample_count": statistics_payload.get("source_train_sample_count"),
    }


def run_or_skip_task(config: ExperimentConfig, task: Task, *, resume: bool) -> None:
    materialize_training_validation_manifest(task)
    complete, reason, _ = is_task_complete(task)
    if complete:
        if not resume:
            raise RuntimeError(f"Completed task encountered with --no-resume: {task.task_id}")
        write_task_state(task, "SKIPPED_COMPLETE", reason, command=build_training_command(config, task))
        print(f"SKIP COMPLETE {task.task_id}", flush=True)
        return
    if task.run_dir.exists() and any(task.run_dir.iterdir()):
        write_task_state(task, "BLOCKED_PARTIAL", reason, command=build_training_command(config, task))
        raise RuntimeError(
            f"Refusing to overwrite incomplete task {task.task_id}: {reason}. "
            f"Move the run directory aside manually before retrying: {task.run_dir}"
        )

    command = build_training_command(config, task)
    write_task_state(task, "RUNNING", "existing validation-selected trainer launched", command=command)
    print(f"START {task.task_id}", flush=True)
    return_code = run_logged_command(command, task.run_root / "orchestration" / "logs" / f"{task.task_id}.log")
    complete, reason, _ = is_task_complete(task)
    if return_code != 0 or not complete:
        detail = f"trainer_return_code={return_code}; completion_audit={reason}"
        write_task_state(task, "FAILED", detail, command=command, return_code=return_code)
        raise RuntimeError(f"Task failed: {task.task_id}; {detail}")
    write_task_state(task, "COMPLETED", reason, command=command, return_code=return_code)
    print(f"END COMPLETE {task.task_id}", flush=True)


def build_training_command(config: ExperimentConfig, task: Task) -> list[str]:
    command = [
        config.python_executable,
        str(ROOT / "scripts" / "run_lora_checkpoint_validation.py"),
        "--input-dir", str(task.artifact_dir / "prepared_input"),
        "--noise-index", str(task.artifact_dir / "noise_manifest.csv"),
        "--output-dir", str(task.run_root),
        "--methods", task.method,
        "--seeds", str(task.seed),
        "--validation-ratio", "0.10",
        "--validation-seed", str(VALIDATION_SEED),
        "--epochs", str(EPOCHS),
        "--warmup-epochs", str(WARMUP_EPOCHS),
        "--update-interval", str(UPDATE_INTERVAL),
        "--dynamic-ratio", str(DYNAMIC_RATIO),
        "--fixed-p", str(FIXED_P),
        "--no-posthoc-oracle-test",
    ]
    if task.single_branch:
        command.append("--official-test-selected-only")
    if task.dataset.training_config is not None:
        command.extend(["--config", str(task.dataset.training_config)])
    for path_map in task.dataset.path_maps:
        command.extend(["--path-map", path_map])
    instrumentation = config.instrumentation
    if instrumentation.profiling_mode != "off":
        command.extend(["--revision-profiling-mode", instrumentation.profiling_mode])
    if instrumentation.selection_logging:
        command.append("--revision-selection-logging")
    if instrumentation.profiling_mode != "off" or instrumentation.selection_logging:
        assert instrumentation.output_root is not None
        command.extend(["--revision-output-root", str(instrumentation.output_root)])
    return command


def materialize_training_validation_manifest(task: Task) -> None:
    """Make each protocol training root use the canonical independent split."""

    source_csv, source_json = validation_manifest_paths(task.dataset.validation_dir)
    target_csv, target_json = validation_manifest_paths(task.run_root)
    if not source_csv.is_file() or not source_json.is_file():
        raise FileNotFoundError(
            f"Canonical fixed validation manifest is unavailable for {task.dataset.key}: {task.dataset.validation_dir}"
        )
    if target_csv.is_file() != target_json.is_file():
        raise FileNotFoundError(f"Training root has only one validation manifest file: {task.run_root}")
    if target_csv.is_file():
        for source, target in ((source_csv, target_csv), (source_json, target_json)):
            if source.read_bytes() != target.read_bytes():
                raise ValueError(
                    f"Training root validation manifest differs from the canonical independent split: {target}"
                )
        return
    if task.run_root.exists() and any(task.run_root.iterdir()):
        raise FileExistsError(
            f"Training root is non-empty but lacks the canonical fixed validation manifest: {task.run_root}"
        )
    task.run_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, target_csv)
    shutil.copy2(source_json, target_json)


def is_task_complete(task: Task) -> tuple[bool, str, dict[str, Any] | None]:
    source_csv, source_json = validation_manifest_paths(task.dataset.validation_dir)
    target_csv, target_json = validation_manifest_paths(task.run_root)
    if not source_csv.is_file() or not source_json.is_file():
        return False, f"canonical fixed validation manifest is missing: {task.dataset.validation_dir}", None
    if not target_csv.is_file() or not target_json.is_file():
        return False, f"training root validation manifest is missing: {task.run_root}", None
    for source, target in ((source_csv, target_csv), (source_json, target_json)):
        if source.read_bytes() != target.read_bytes():
            return False, f"training root validation manifest differs from canonical split: {target}", None
    required = [
        task.run_dir / "result.json",
        task.run_dir / "train_log.csv",
        task.run_dir / "selection_policy.json",
        task.run_dir / "checkpoints" / "best_val.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, f"missing required task artifacts: {missing}", None
    try:
        result = read_json(task.run_dir / "result.json")
        policy = read_json(task.run_dir / "selection_policy.json")
        log_rows = read_csv(task.run_dir / "train_log.csv")
    except (OSError, json.JSONDecodeError, csv.Error, ValueError) as exc:
        return False, f"cannot parse task artifacts: {exc}", None
    expected = {
        "method_key": task.method,
        "seed": task.seed,
        "noise_strategy": task.expected_noise_strategy,
        "noise_seed": NOISE_SEED,
        "validation_seed": VALIDATION_SEED,
        "checkpoint_protocol": PROTOCOL_NAME,
    }
    mismatches = [key for key, value in expected.items() if not equivalent(result.get(key), value)]
    if not task.single_branch and result.get("official_test_evaluation") not in (None, ""):
        mismatches.append("dual_official_test_evaluation_unexpected")
    if task.single_branch and result.get("official_test_evaluation") != "validation_selected_only":
        mismatches.append("official_test_evaluation")
    if not equivalent(result.get("noise_ratio"), NOISE_RATE):
        mismatches.append("noise_ratio")
    if not numeric(result.get("validation_selected_test_top1")):
        mismatches.append("validation_selected_test_top1")
    if policy.get("official_test_used_for_checkpoint_selection") is not False:
        mismatches.append("selection_policy.official_test_used_for_checkpoint_selection")
    if policy.get("validation_samples_are_training_candidates") is not False:
        mismatches.append("selection_policy.validation_samples_are_training_candidates")
    try:
        epochs = [int(row["epoch"]) for row in log_rows]
    except (KeyError, TypeError, ValueError):
        mismatches.append("train_log.epoch")
        epochs = []
    if epochs != list(range(1, EPOCHS + 1)):
        mismatches.append("train_log.complete_30_epochs")
    if mismatches:
        return False, f"completion metadata mismatch: {sorted(mismatches)}", result
    return True, "strict completion audit passed", result


def write_status_and_summary(config: ExperimentConfig, plan: list[Task]) -> None:
    root = orchestration_root(config)
    root.mkdir(parents=True, exist_ok=True)
    status_rows: list[dict[str, Any]] = []
    complete_results: dict[tuple[str, str, str], list[tuple[Task, dict[str, Any]]]] = {}
    for task in plan:
        complete, detail, result = is_task_complete(task)
        state = read_task_state(task)
        status = "COMPLETED" if complete else str(state.get("status", "PENDING")) if state else "PENDING"
        status_rows.append({**task_row(task), "status": status, "detail": detail})
        if complete and result is not None:
            complete_results.setdefault((task.dataset.key, task.protocol, task.method), []).append((task, result))
    write_csv(root / "run_status.csv", status_rows, list(status_rows[0]) if status_rows else status_fields())

    summary_rows: list[dict[str, Any]] = []
    for dataset in config.datasets:
        for protocol in PROTOCOLS:
            for method in METHODS:
                members = complete_results.get((dataset.key, protocol, method), [])
                by_seed = {task.seed: result for task, result in members}
                complete = set(by_seed) == set(TRAINING_SEEDS)
                values = [float(by_seed[seed]["validation_selected_test_top1"]) for seed in TRAINING_SEEDS] if complete else []
                summary_rows.append(
                    {
                        "dataset": dataset.display_name,
                        "dataset_key": dataset.key,
                        "noise_protocol": protocol,
                        "method_key": method,
                        "checkpoint_evaluation_protocol": (
                            "validation_selected_only" if method in SINGLE_BRANCH_METHODS else "dual_branch_mean_validation_selected_plus_final_last5"
                        ),
                        "required_seed_count": len(TRAINING_SEEDS),
                        "completed_seed_count": len(by_seed),
                        "status": "COMPLETE_3_SEEDS" if complete else "INCOMPLETE",
                        "mean_validation_selected_official_test_top1": statistics.mean(values) if values else "",
                        "std_validation_selected_official_test_top1": statistics.stdev(values) if len(values) > 1 else "",
                        "seed1_top1": value_or_empty(by_seed.get(1), "validation_selected_test_top1"),
                        "seed42_top1": value_or_empty(by_seed.get(42), "validation_selected_test_top1"),
                        "seed88_top1": value_or_empty(by_seed.get(88), "validation_selected_test_top1"),
                    }
                )
    write_csv(root / "validation_selected_official_test_summary.csv", summary_rows, list(summary_rows[0]))
    write_json(
        root / "orchestration_summary.json",
        {
            "generated_at_utc": utc_now(),
            "planned_task_count": len(plan),
            "strict_completed_task_count": sum(row["status"] == "COMPLETED" for row in status_rows),
            "complete_method_dataset_protocol_cells": sum(row["status"] == "COMPLETE_3_SEEDS" for row in summary_rows),
            "summary_csv": str(root / "validation_selected_official_test_summary.csv"),
        },
    )


def write_global_metadata(config: ExperimentConfig, plan: list[Task]) -> None:
    root = orchestration_root(config)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "resolved_orchestration_config.json", config_to_dict(config))
    write_json(root / "git_provenance.json", git_provenance())
    rows = [task_row(task) for task in plan]
    write_csv(root / "run_plan.csv", rows, list(rows[0]))
    write_json(root / "run_plan.json", {"planned_task_count": len(rows), "tasks": rows})


def print_plan(config: ExperimentConfig, plan: list[Task], *, dry_run: bool) -> None:
    print(f"PLAN formal_tasks={len(plan)} (3 datasets × 2 protocols × 7 methods × 3 seeds)", flush=True)
    for dataset in config.datasets:
        for protocol in PROTOCOLS:
            artifact_dir = artifact_directory(config, dataset, protocol)
            command = build_prepare_command(config, dataset, protocol, artifact_dir)
            action = "REUSE_OR_VALIDATE" if artifact_dir.exists() else "PREPARE"
            print(f"{action} {shlex.join(command)}", flush=True)
    for task in plan:
        complete, _, _ = is_task_complete(task)
        action = "SKIP_COMPLETE" if complete else "RUN"
        print(f"{action} {task.task_id} {shlex.join(build_training_command(config, task))}", flush=True)


def run_logged_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] COMMAND {command_text}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
        handle.write(f"[{utc_now()}] RETURN_CODE {return_code}\n")
    return return_code


def write_task_state(task: Task, status: str, detail: str, *, command: list[str], return_code: int | None = None) -> None:
    state_path = task.run_root / "orchestration" / "run_states" / f"{task.task_id}.json"
    write_json(
        state_path,
        {
            **task_row(task),
            "status": status,
            "detail": detail,
            "timestamp_utc": utc_now(),
            "command": command,
            "return_code": return_code,
        },
    )


def read_task_state(task: Task) -> dict[str, Any]:
    path = task.run_root / "orchestration" / "run_states" / f"{task.task_id}.json"
    return read_json(path) if path.is_file() else {}


def orchestration_root(config: ExperimentConfig) -> Path:
    return config.training_outputs_root / "orchestration"


def task_row(task: Task) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "dataset_key": task.dataset.key,
        "dataset": task.dataset.display_name,
        "noise_protocol": task.protocol,
        "noise_rate": NOISE_RATE,
        "noise_seed": NOISE_SEED,
        "method_key": task.method,
        "training_seed": task.seed,
        "validation_seed": VALIDATION_SEED,
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "update_interval": UPDATE_INTERVAL,
        "dynamic_ratio": DYNAMIC_RATIO,
        "fixed_p": FIXED_P,
        "artifact_dir": str(task.artifact_dir),
        "run_dir": str(task.run_dir),
        "expected_test_protocol": task.expected_test_protocol,
    }


def status_fields() -> list[str]:
    return [*task_row_placeholder(), "status", "detail"]


def task_row_placeholder() -> list[str]:
    return [
        "task_id", "dataset_key", "dataset", "noise_protocol", "noise_rate", "noise_seed", "method_key", "training_seed",
        "validation_seed", "epochs", "warmup_epochs", "update_interval", "dynamic_ratio", "fixed_p", "artifact_dir", "run_dir",
        "expected_test_protocol",
    ]


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_config": str(config.config_path),
        "python_executable": config.python_executable,
        "noise_assets_root": str(config.noise_assets_root),
        "training_outputs_root": str(config.training_outputs_root),
        "instrumentation": {
            "profiling_mode": config.instrumentation.profiling_mode,
            "selection_logging": config.instrumentation.selection_logging,
            "output_root": str(config.instrumentation.output_root) if config.instrumentation.output_root else "",
        },
        "formal": {
            "protocols": list(PROTOCOLS), "methods": list(METHODS), "training_seeds": list(TRAINING_SEEDS),
            "noise_rate": NOISE_RATE, "noise_seed": NOISE_SEED, "validation_seed": VALIDATION_SEED,
            "epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS, "update_interval": UPDATE_INTERVAL,
            "dynamic_ratio": DYNAMIC_RATIO, "fixed_p": FIXED_P,
        },
        "datasets": [
            {
                "key": item.key, "display_name": item.display_name, "base_input_dir": str(item.base_input_dir),
                "source_noise_index": str(item.source_noise_index), "validation_dir": str(item.validation_dir),
                "path_maps": list(item.path_maps), "training_config": str(item.training_config) if item.training_config else "",
            }
            for item in config.datasets
        ],
    }


def git_provenance() -> dict[str, Any]:
    def capture(command: list[str]) -> str:
        completed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        return completed.stdout.strip() if completed.returncode == 0 else f"UNAVAILABLE: {completed.stderr.strip()}"

    return {"git_head": capture(["git", "rev-parse", "HEAD"]), "git_status_short": capture(["git", "status", "--short"]), "captured_at_utc": utc_now()}


def _validate_formal_contract(formal: dict[str, Any]) -> None:
    expected = {
        "noise_rate": NOISE_RATE,
        "noise_seed": NOISE_SEED,
        "validation_seed": VALIDATION_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "update_interval": UPDATE_INTERVAL,
        "dynamic_ratio": DYNAMIC_RATIO,
        "fixed_p": FIXED_P,
    }
    mismatches = [key for key, value in expected.items() if not equivalent(formal.get(key), value)]
    if formal.get("protocols") != list(PROTOCOLS):
        mismatches.append("protocols")
    if mismatches:
        raise ValueError(f"Formal configuration is fixed for this experiment and mismatches: {sorted(mismatches)}")


def require_mapping(container: dict[str, Any], key: str) -> dict[str, Any]:
    value = container.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration key {key!r} must be a mapping.")
    return value


def require_string(container: dict[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration key {key!r} must be a non-empty string.")
    return value.strip()


def require_bool(container: dict[str, Any], key: str) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Configuration key {key!r} must be true or false.")
    return value


def resolve_path(raw: str, config_dir: Path) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def numeric(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def equivalent(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        return numeric(left) and math.isclose(float(left), right, rel_tol=0.0, abs_tol=1.0e-12)
    if isinstance(right, int):
        try:
            return int(left) == right
        except (TypeError, ValueError):
            return False
    return left == right


def value_or_empty(payload: dict[str, Any] | None, field: str) -> Any:
    return "" if payload is None else payload.get(field, "")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
