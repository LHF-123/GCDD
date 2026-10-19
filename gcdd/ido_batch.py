"""Safe, shared orchestration for the 9+18 IDO controlled-adaptation matrix.

The module never constructs noise labels and never reads clean labels for an
IDO training decision.  It consumes immutable V1/IDN/GCCN contracts, validates
their observable-label alignment, and invokes the existing single-run entry
only when both the YAML protocol and the command line explicitly authorize it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .lora_training import (
    DINO_VITB14_EMBED_DIM,
    build_dinov2_vitb14_lora_structure_contract,
    canonical_structure_sha256,
)
from .ido import ido_adaptation_name, ido_bmm_scope, resolve_ido_bmm_mode


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
DATASET_ORDER = ("cub", "cars", "aircraft")
PROTOCOL_ORDER = ("cyclic", "idn", "gccn")
SEEDS = (1, 42, 88)
REQUIRED_V1 = ("paths.txt", "labels.npy", "eval_paths.txt", "eval_labels.npy", "resolved_config.yaml")
TASK_CONTRACT_FILENAME = "ido_task_contract.json"
TASK_CONTRACT_SCHEMA_VERSION = 1
CHECKPOINT_FILENAMES = ("stage1_base.pt", "best_val.pt", "last.pt")
# The batch contract hashes the local modules that can affect the invoked IDO
# subprocess or the integrity checks that decide whether a prior run may be
# skipped/aggregated.  Keep this explicit rather than relying on a partial
# provenance helper intended for the older PGDF paths.
IDO_RUNTIME_CODE_PATHS = (
    "gcdd/__init__.py",
    "gcdd/baselines.py",
    "gcdd/budget_matching.py",
    "gcdd/checkpoint_validation.py",
    "gcdd/config.py",
    "gcdd/data.py",
    "gcdd/features.py",
    "gcdd/fusion_ablation.py",
    "gcdd/graph.py",
    "gcdd/ido.py",
    "gcdd/ido_batch.py",
    "gcdd/io_utils.py",
    "gcdd/lora_dynamic.py",
    "gcdd/lora_noisy_baselines.py",
    "gcdd/lora_training.py",
    "gcdd/noise_generalization.py",
    "gcdd/noise_realization.py",
    "gcdd/progress.py",
    "gcdd/scoring.py",
    "gcdd/selection_utils.py",
    "revision_experiments/__init__.py",
    "revision_experiments/_paths.py",
    "revision_experiments/offline_diagnostics.py",
    "revision_experiments/profiling.py",
    "revision_experiments/selection_logging.py",
    "revision_experiments/training_integration.py",
    "scripts/run_lora_checkpoint_validation.py",
)


@dataclass(frozen=True)
class ProtocolInput:
    input_dir: Path
    noise_index: Path
    validation_dir: Path
    reference_126_training_root: Path | None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    expected_classes: int
    path_maps: tuple[str, ...]
    protocols: dict[str, ProtocolInput]


@dataclass(frozen=True)
class IDOBatchConfig:
    path: Path
    python_executable: str
    output_root: Path
    instrumentation_root: Path
    protocol_frozen: bool
    epochs: int
    stage1_epochs: int | None
    validation_seed: int
    bmm_mode: str
    datasets: tuple[DatasetSpec, ...]


@dataclass(frozen=True)
class IDOTask:
    dataset: DatasetSpec
    protocol: str
    seed: int
    attempt: int
    run_root: Path
    run_dir: Path

    @property
    def task_id(self) -> str:
        return f"{self.dataset.key}__{self.protocol}__ido__seed{self.seed}"

    @property
    def attempt_id(self) -> str:
        return f"attempt-{self.attempt:03d}"

    @property
    def input(self) -> ProtocolInput:
        return self.dataset.protocols[self.protocol]

    @property
    def contract_path(self) -> Path:
        # A task invokes the single-run entry point with a seed-specific
        # ``run_dir``.  Keeping the immutable contract beside that result
        # prevents seed 1/42/88 from trying to reuse one another's task
        # identity while still sharing the dataset-level validation manifest
        # in ``run_root``.
        return self.run_dir / TASK_CONTRACT_FILENAME


def parse_cli(*, protocols: Sequence[str], description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="IDO batch YAML; relative paths are resolved from this file.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default-safe: preflight and print tasks, never start training.")
    mode.add_argument("--execute", action="store_true", help="Start sequential training only after protocol_frozen=true preflight.")
    mode.add_argument("--summarize-only", action="store_true", help="Read attempt directories and refresh only status/mean±sample-SD summaries.")
    parser.add_argument(
        "--new-attempt",
        action="store_true",
        help="For --execute only: create a new attempt-NNN after an incomplete attempt; never overwrite it.",
    )
    parser.add_argument("--protocols", default=",".join(protocols), help=argparse.SUPPRESS)
    args = parser.parse_args()
    requested = tuple(item.strip() for item in args.protocols.split(",") if item.strip())
    if not requested or any(item not in protocols for item in requested):
        raise ValueError(f"This launcher accepts only protocols {tuple(protocols)}.")
    args.protocols = requested
    if args.new_attempt and not args.execute:
        raise ValueError("--new-attempt requires --execute.")
    return args


def main_for_protocols(*, protocols: Sequence[str], description: str) -> None:
    args = parse_cli(protocols=protocols, description=description)
    config = load_config(args.config)
    selected = tuple(args.protocols)
    if args.summarize_only:
        summarize_all(config, selected)
        print("SUMMARY PASS: no training subprocess was started.", flush=True)
        return
    execute = bool(args.execute)
    dry_run = not execute
    if execute and not config.protocol_frozen:
        raise PermissionError("Refusing formal IDO training: protocol.protocol_frozen must be true in the reviewed YAML.")
    attempt = choose_attempt(config, selected, new_attempt=bool(args.new_attempt))
    tasks = build_plan(config, selected, attempt=attempt)
    expected = len(DATASET_ORDER) * len(selected) * len(SEEDS)
    if len(tasks) != expected or len({task.task_id for task in tasks}) != expected:
        raise RuntimeError(f"Invalid IDO task plan: expected {expected} unique tasks, found {len(tasks)}.")
    findings = preflight(config, tasks)
    print_plan(config, tasks, findings, dry_run=dry_run)
    if dry_run:
        print(f"DRY RUN COMPLETE: planned={len(tasks)}, blocked={sum(not item['ok'] for item in findings)}; no training subprocess was started.", flush=True)
        return
    blocking = [item for item in findings if not item["ok"]]
    if blocking:
        raise RuntimeError(f"IDO preflight blocked {len(blocking)} task(s); see printed findings. No training was started.")
    for task in tasks:
        complete, detail = is_complete(task, config)
        if complete:
            print(f"SKIP COMPLETE {task.task_id} ({task.attempt_id})", flush=True)
            continue
        if args.new_attempt:
            reusable = find_reusable_completed_task(config, task)
            if reusable is not None:
                print(
                    f"SKIP REUSE COMPLETE {task.task_id} ({task.attempt_id}) "
                    f"from {reusable.attempt_id}; only incomplete seeds are retried.",
                    flush=True,
                )
                continue
        if task.run_dir.exists() and any(task.run_dir.iterdir()):
            raise RuntimeError(
                f"Incomplete task is fail-closed: {task.task_id} ({task.attempt_id}): {detail}. "
                "Use --new-attempt to create a separate attempt directory; do not overwrite this one."
            )
        materialize_validation_manifest(task)
        materialize_task_contract(config, task)
        command = build_training_command(config, task)
        print(f"START {task.task_id} ({task.attempt_id})", flush=True)
        return_code = subprocess.call(command, cwd=ROOT)
        complete, detail = is_complete(task, config)
        if return_code != 0 or not complete:
            raise RuntimeError(f"IDO task failed: {task.task_id}; return_code={return_code}; completion={detail}")
        print(f"END COMPLETE {task.task_id} ({task.attempt_id})", flush=True)
    summarize_all(config, selected)


def load_config(path: Path) -> IDOBatchConfig:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"IDO batch config requires schema_version: {SCHEMA_VERSION}.")
    base = path.parent
    runtime = require_mapping(payload, "runtime")
    output = require_mapping(payload, "output")
    protocol = require_mapping(payload, "protocol")
    formal = require_mapping(payload, "formal")
    if int(formal.get("epochs", 0)) <= 0:
        raise ValueError("formal.epochs must be positive.")
    stage1_raw = formal.get("stage1_epochs")
    stage1_epochs = None if stage1_raw in (None, "") else int(stage1_raw)
    if stage1_epochs is not None and not 0 < stage1_epochs < int(formal["epochs"]):
        raise ValueError("formal.stage1_epochs must lie strictly within total epochs.")
    if stage1_epochs is None and int(formal["epochs"]) % 3:
        raise ValueError("formal.stage1_epochs is required when formal.epochs is not divisible by 3.")
    bmm_mode = resolve_ido_bmm_mode(formal.get("bmm_mode", "classwise"))
    datasets_data = require_mapping(payload, "datasets")
    if set(datasets_data) != set(DATASET_ORDER):
        raise ValueError(f"datasets must contain exactly {list(DATASET_ORDER)}.")
    datasets: list[DatasetSpec] = []
    for key in DATASET_ORDER:
        item = require_mapping(datasets_data, key)
        maps = item.get("path_maps", [])
        if not isinstance(maps, list) or not all(isinstance(value, str) and "=" in value for value in maps):
            raise ValueError(f"datasets.{key}.path_maps must be OLD=NEW entries.")
        protocol_data = require_mapping(item, "protocols")
        if set(protocol_data) != set(PROTOCOL_ORDER):
            raise ValueError(f"datasets.{key}.protocols must contain {list(PROTOCOL_ORDER)}.")
        inputs: dict[str, ProtocolInput] = {}
        for protocol_name in PROTOCOL_ORDER:
            raw = require_mapping(protocol_data, protocol_name)
            reference = raw.get("reference_126_training_root")
            inputs[protocol_name] = ProtocolInput(
                input_dir=resolve_path(require_string(raw, "input_dir"), base),
                noise_index=resolve_path(require_string(raw, "noise_index"), base),
                validation_dir=resolve_path(require_string(raw, "validation_dir"), base),
                reference_126_training_root=None if reference in (None, "") else resolve_path(str(reference), base),
            )
        datasets.append(
            DatasetSpec(
                key=key,
                display_name=require_string(item, "display_name"),
                expected_classes=int(item["expected_classes"]),
                path_maps=tuple(maps),
                protocols=inputs,
            )
        )
    frozen_value = protocol.get("protocol_frozen", False)
    if not isinstance(frozen_value, bool):
        raise ValueError("protocol.protocol_frozen must be a YAML boolean, not a string or numeric value.")
    return IDOBatchConfig(
        path=path,
        python_executable=require_string(runtime, "python_executable"),
        output_root=resolve_path(require_string(output, "root"), base),
        instrumentation_root=resolve_path(require_string(output, "instrumentation_root"), base),
        protocol_frozen=frozen_value,
        epochs=int(formal["epochs"]),
        stage1_epochs=stage1_epochs,
        validation_seed=int(formal["validation_seed"]),
        bmm_mode=bmm_mode,
        datasets=tuple(datasets),
    )


def build_plan(config: IDOBatchConfig, protocols: Sequence[str], *, attempt: int) -> list[IDOTask]:
    return [
        IDOTask(
            dataset=dataset,
            protocol=protocol,
            seed=seed,
            attempt=attempt,
            run_root=config.output_root / "attempts" / f"attempt-{attempt:03d}" / protocol / dataset.key,
            run_dir=config.output_root / "attempts" / f"attempt-{attempt:03d}" / protocol / dataset.key / "ido" / f"seed{seed}",
        )
        for dataset in config.datasets
        for protocol in protocols
        for seed in SEEDS
    ]


def choose_attempt(config: IDOBatchConfig, protocols: Sequence[str], *, new_attempt: bool) -> int:
    attempts_root = config.output_root / "attempts"
    existing: list[int] = []
    if attempts_root.is_dir():
        for child in attempts_root.iterdir():
            if child.is_dir() and child.name.startswith("attempt-") and child.name[8:].isdigit():
                existing.append(int(child.name[8:]))
    if new_attempt:
        return (max(existing) if existing else 0) + 1
    return max(existing) if existing else 1


def preflight(config: IDOBatchConfig, tasks: Iterable[IDOTask]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    checked: set[tuple[str, str]] = set()
    for task in tasks:
        key = (task.dataset.key, task.protocol)
        if key in checked:
            continue
        checked.add(key)
        try:
            findings.append({"dataset": task.dataset.key, "protocol": task.protocol, "ok": True, "detail": inspect_input_contract(task)})
        except Exception as exc:
            findings.append({"dataset": task.dataset.key, "protocol": task.protocol, "ok": False, "detail": str(exc)})
    roots = [config.output_root.resolve(), config.instrumentation_root.resolve()]
    if roots[0] == roots[1]:
        findings.append({"dataset": "all", "protocol": "all", "ok": False, "detail": "output.root and output.instrumentation_root must differ"})
    for task in tasks:
        protected = [task.input.input_dir, task.input.validation_dir]
        if task.input.reference_126_training_root is not None:
            protected.append(task.input.reference_126_training_root)
        for output_root, name in ((config.output_root, "output.root"), (config.instrumentation_root, "output.instrumentation_root")):
            for source_root in protected:
                if paths_overlap(output_root, source_root):
                    findings.append(
                        {
                            "dataset": task.dataset.key,
                            "protocol": task.protocol,
                            "ok": False,
                            "detail": f"{name} must not overlap immutable input/reference root: {source_root}",
                        }
                    )
    return findings


def inspect_input_contract(task: IDOTask) -> dict[str, Any]:
    source = task.input
    missing = [str(source.input_dir / name) for name in REQUIRED_V1 if not (source.input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing V1 contract files: {missing}")
    if not source.noise_index.is_file():
        raise FileNotFoundError(f"noise index missing: {source.noise_index}")
    for name in ("validation_manifest.csv", "validation_manifest.json"):
        if not (source.validation_dir / name).is_file():
            raise FileNotFoundError(f"canonical validation manifest missing: {source.validation_dir / name}")
    paths = (source.input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    labels = np.load(source.input_dir / "labels.npy", allow_pickle=True).astype(str)
    if len(paths) != len(labels):
        raise ValueError("paths.txt and labels.npy lengths differ")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != task.dataset.expected_classes:
        raise ValueError(f"observed class count {len(classes)} != expected {task.dataset.expected_classes}")
    if int(counts.min()) < 1:
        raise ValueError("observed labels contain an empty class")
    noise_rows = read_csv(source.noise_index)
    training_rows = [row for row in noise_rows if str(row.get("split", "train")).lower() == "train"]
    if len(training_rows) != len(paths):
        raise ValueError(f"noise-index training rows {len(training_rows)} != V1 paths {len(paths)}")
    alignment = validate_observed_label_alignment(
        task=task,
        paths=paths,
        labels=labels,
        training_rows=training_rows,
    )
    metadata = uniform_noise_metadata(training_rows)
    expected_strategy = {
        "cyclic": "adjacent_cyclic",
        "idn": "instance_dependent_dinov2_cls_v1",
        "gccn": "general_class_conditional_multi_offset_v1",
    }[task.protocol]
    if str(metadata["noise_strategy"]) != expected_strategy:
        raise ValueError(f"noise strategy {metadata['noise_strategy']!r} != {expected_strategy!r}")
    reference_validation_alignment = "not_applicable_to_cyclic"
    if task.protocol in {"idn", "gccn"}:
        reference = source.reference_126_training_root
        if reference is None:
            raise ValueError("IDN/GCCN requires reference_126_training_root for realization parity preflight")
        resolved = reference / "resolved_config.yaml"
        if not resolved.is_file():
            raise FileNotFoundError(f"126-task reference config missing: {resolved}")
        reference_payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        reference_hash = str(reference_payload.get("checkpoint_validation", {}).get("noise_index_sha256", ""))
        if reference_hash != sha256_file(source.noise_index):
            raise ValueError("IDO IDN/GCCN noise-index SHA256 differs from the existing 126-task realization")
        # Do not infer validation parity from a shared-looking directory name.
        # The original 126-task output is required to retain the actual fixed
        # validation artifacts, and both immutable files must be byte-identical
        # to the ones supplied to the IDO task.
        reference_validation_alignment = validate_reference_126_validation_manifests(
            reference_root=reference,
            ido_validation_dir=source.validation_dir,
        )
    manifest_rows = read_csv(source.validation_dir / "validation_manifest.csv")
    validate_manifest(manifest_rows, len(paths), expected_paths=paths)
    training_indices = np.asarray(
        [int(row["index"]) for row in manifest_rows if str(row.get("partition", "")) == "training_pool"],
        dtype=np.int64,
    )
    training_labels = labels[training_indices]
    pool_classes, pool_counts = np.unique(training_labels, return_counts=True)
    if len(pool_classes) != task.dataset.expected_classes:
        missing_classes = sorted(set(classes.tolist()) - set(pool_classes.tolist()))
        raise ValueError(
            "actual IDO training pool has empty observed-label class(es): "
            f"expected={task.dataset.expected_classes}, present={len(pool_classes)}, missing={missing_classes[:10]}"
        )
    validate_path_maps(paths, task.dataset.path_maps)
    return {
        "input_samples": len(paths),
        "source_observed_class_count": int(len(classes)),
        "source_observed_class_min": int(counts.min()),
        "source_observed_class_p5": float(np.percentile(counts, 5)),
        "source_observed_class_median": float(np.median(counts)),
        "source_observed_class_p95": float(np.percentile(counts, 95)),
        "source_observed_class_max": int(counts.max()),
        "source_empty_class_count": 0,
        "training_pool_samples": int(len(training_indices)),
        "training_pool_observed_class_count": int(len(pool_classes)),
        "training_pool_observed_class_min": int(pool_counts.min()),
        "training_pool_observed_class_p5": float(np.percentile(pool_counts, 5)),
        "training_pool_observed_class_median": float(np.median(pool_counts)),
        "training_pool_observed_class_p95": float(np.percentile(pool_counts, 95)),
        "training_pool_observed_class_max": int(pool_counts.max()),
        "training_pool_empty_class_count": int(task.dataset.expected_classes - len(pool_classes)),
        "noise_index_sha256": sha256_file(source.noise_index),
        "validation_manifest_sha256": sha256_file(source.validation_dir / "validation_manifest.csv"),
        "validation_manifest_json_sha256": sha256_file(source.validation_dir / "validation_manifest.json"),
        "noise_strategy": metadata["noise_strategy"],
        "noise_alignment": alignment["status"],
        "reference_126_validation_alignment": reference_validation_alignment,
        "path_maps_checked": len(task.dataset.path_maps),
    }


def validate_reference_126_validation_manifests(*, reference_root: Path, ido_validation_dir: Path) -> str:
    """Prove IDN/GCCN validation parity using the historical run's artifacts.

    This is intentionally a read-only byte comparison.  It does not parse or
    use clean labels for any IDO training decision; it only prevents a batch
    task from claiming comparability to the 126-run matrix without evidence
    that both used exactly the same frozen validation partition.
    """
    for name in ("validation_manifest.csv", "validation_manifest.json"):
        reference_path = reference_root / name
        ido_path = ido_validation_dir / name
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"126-task reference validation manifest missing: {reference_path}; "
                "cannot establish IDN/GCCN validation parity."
            )
        if sha256_file(reference_path) != sha256_file(ido_path):
            raise ValueError(
                f"IDO validation {name} SHA256 differs from the existing 126-task reference: "
                f"{ido_path} != {reference_path}"
            )
    return "exact_csv_and_json_sha256_match_to_reference_126"


def validate_observed_label_alignment(
    *,
    task: IDOTask,
    paths: list[str],
    labels: np.ndarray,
    training_rows: list[dict[str, str]],
) -> dict[str, str]:
    """Fail closed unless the immutable index describes the V1 observed labels.

    This is a read-only preflight check.  It never reads ``clean_label`` and
    cannot influence IDO's BMM or loss: it establishes only that the prepared
    V1 ``labels.npy`` is the observed/noisy-label realization named by the
    requested noise index.  IDN/GCCN manifests have an explicit
    ``source_train_index`` contract; cyclic has no V1-order index, so its
    immutable path-to-row mapping plus ``web_label`` agreement is the source
    of truth.
    """
    if task.protocol in {"idn", "gccn"}:
        from .noise_generalization import validate_noise_generalization_training_input

        return validate_noise_generalization_training_input(
            manifest_path=task.input.noise_index,
            train_paths=paths,
            train_labels=labels,
            labels_path=task.input.input_dir / "labels.npy",
        )

    if task.protocol != "cyclic":
        raise ValueError(f"Unsupported IDO noise protocol for observed-label alignment: {task.protocol!r}")
    from .noise_realization import align_noise_rows

    required = {"path", "web_label", "noise_strategy"}
    missing = sorted(required - set(training_rows[0] if training_rows else {}))
    if missing:
        raise ValueError(f"Cyclic noise index is missing required observable-alignment columns: {missing}")
    aligned_rows = align_noise_rows(paths, training_rows, source_name=str(task.input.noise_index))
    for v1_index, row in enumerate(aligned_rows):
        observed = str(row["web_label"])
        if str(labels[v1_index]) != observed:
            raise ValueError(
                "Prepared V1 labels.npy disagrees with the cyclic noise-index observed label at "
                f"V1 index {v1_index} ({paths[v1_index]!r}): "
                f"labels.npy={labels[v1_index]!r}, noise_index={observed!r}."
            )
        # Some cyclic sources are full-dataset indices, where ``index`` is not
        # the V1 source-train position.  Only enforce a declared V1-order key.
        declared_source_index = str(row.get("source_train_index", "")).strip()
        if declared_source_index:
            try:
                source_index = int(declared_source_index)
            except ValueError as exc:
                raise ValueError(f"Invalid cyclic source_train_index at V1 index {v1_index}.") from exc
            if source_index != v1_index:
                raise ValueError(
                    "Cyclic noise-index source_train_index does not match the aligned V1 order: "
                    f"V1 index {v1_index} declares {source_index}."
                )
    return {
        "status": "verified_path_aligned_observed_labels",
        "alignment": "path_keys_and_web_label",
    }


def uniform_noise_metadata(rows: list[dict[str, str]]) -> dict[str, str]:
    required = ("noise_strategy", "noise_ratio", "noise_seed")
    result: dict[str, str] = {}
    for name in required:
        values = {str(row.get(name, "")) for row in rows}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"noise index has inconsistent or missing {name}")
        result[name] = next(iter(values))
    return result


def validate_manifest(rows: list[dict[str, str]], expected_count: int, *, expected_paths: Sequence[str] | None = None) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"validation manifest rows {len(rows)} != V1 sample count {expected_count}")
    seen: set[int] = set()
    training: set[int] = set()
    validation: set[int] = set()
    for row in rows:
        index = int(row["index"])
        partition = row.get("partition", "")
        if index in seen or index < 0 or index >= expected_count:
            raise ValueError("validation manifest indices are not a unique full-pool partition")
        if expected_paths is not None and row.get("path") != expected_paths[index]:
            raise ValueError(f"validation manifest path does not match V1 paths.txt at index {index}")
        seen.add(index)
        if partition == "training_pool":
            training.add(index)
        elif partition == "validation":
            validation.add(index)
        else:
            raise ValueError(f"unknown validation manifest partition {partition!r}")
    if len(seen) != expected_count or not training or not validation or training & validation:
        raise ValueError("training/validation partition is incomplete or not disjoint")


def validate_path_maps(paths: list[str], path_maps: Sequence[str]) -> None:
    mappings = [item.split("=", 1) for item in path_maps]
    for old, new in mappings:
        if not old or not new:
            raise ValueError("path_maps entries must have non-empty OLD=NEW sides")
    # Verify paths after applying the declared map without consulting labels.
    for stored in paths:
        if Path(stored).is_file():
            continue
        mapped = stored
        for old, new in mappings:
            if mapped.startswith(old):
                mapped = new + mapped[len(old) :]
                break
        if mappings and mapped == stored:
            raise ValueError(f"path-map does not cover sampled stored path: {stored}")
        if not Path(mapped).is_file():
            raise FileNotFoundError(f"sampled image path is inaccessible after path mapping: {mapped}")


def build_training_command(config: IDOBatchConfig, task: IDOTask) -> list[str]:
    command = [
        config.python_executable,
        str(ROOT / "scripts" / "run_lora_checkpoint_validation.py"),
        "--input-dir", str(task.input.input_dir),
        "--noise-index", str(task.input.noise_index),
        "--output-dir", str(task.run_root),
        "--methods", "ido",
        "--seeds", str(task.seed),
        "--epochs", str(config.epochs),
        "--validation-ratio", "0.10",
        "--validation-seed", str(config.validation_seed),
        "--official-test-selected-only",
        "--no-posthoc-oracle-test",
        "--device", "cuda",
        "--revision-profiling-mode", "basic",
        "--revision-output-root", str(config.instrumentation_root / task.attempt_id),
        "--ido-task-contract", str(task.contract_path),
        "--ido-bmm-mode", config.bmm_mode,
    ]
    if config.stage1_epochs is not None:
        command.extend(["--ido-stage1-epochs", str(config.stage1_epochs)])
    for mapping in task.dataset.path_maps:
        command.extend(["--path-map", mapping])
    return command


def materialize_validation_manifest(task: IDOTask) -> None:
    source = task.input.validation_dir
    target = task.run_root
    source_paths = (source / "validation_manifest.csv", source / "validation_manifest.json")
    target_paths = (target / "validation_manifest.csv", target / "validation_manifest.json")
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"run root is not a directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for source_path, target_path in zip(source_paths, target_paths):
        if target_path.exists() and target_path.read_bytes() != source_path.read_bytes():
            raise ValueError(f"attempt validation manifest conflicts with canonical source: {target_path}")
        if not target_path.exists():
            target_path.write_bytes(source_path.read_bytes())


def materialize_task_contract(config: IDOBatchConfig, task: IDOTask) -> dict[str, Any]:
    """Persist an immutable, task-specific execution contract before subprocess launch.

    A completed run must echo this digest into result and checkpoint metadata.  A
    changed YAML, input asset, validation split, or relevant source file cannot
    silently be reused from another attempt.
    """
    contract = build_task_contract(config, task)
    path = task.contract_path
    if path.is_file():
        existing = read_json(path)
        if canonical_json_sha256(existing) != canonical_json_sha256(contract):
            raise ValueError(f"attempt task contract conflicts with current frozen inputs/config: {path}")
        return existing
    if path.exists():
        raise FileExistsError(f"task contract path is not a file: {path}")
    write_json(path, contract)
    return contract


def build_task_contract(config: IDOBatchConfig, task: IDOTask) -> dict[str, Any]:
    """Return the frozen inputs/code/config identity for one IDO task.

    The experiment signature intentionally excludes seed and attempt so that a
    retried seed can be combined only when every immutable experiment input is
    identical.  The task signature includes both, preventing a checkpoint from
    one seed/attempt being accepted for another.
    """
    stage1_epochs = config.stage1_epochs if config.stage1_epochs is not None else config.epochs // 3
    stage2_epochs = config.epochs - stage1_epochs
    source = task.input
    assets = {
        "paths_txt_sha256": sha256_file(source.input_dir / "paths.txt"),
        "observed_labels_sha256": sha256_file(source.input_dir / "labels.npy"),
        "v1_resolved_config_sha256": sha256_file(source.input_dir / "resolved_config.yaml"),
        "noise_index_sha256": sha256_file(source.noise_index),
        "validation_manifest_csv_sha256": sha256_file(source.validation_dir / "validation_manifest.csv"),
        "validation_manifest_json_sha256": sha256_file(source.validation_dir / "validation_manifest.json"),
    }
    code_paths = {relative: ROOT / relative for relative in IDO_RUNTIME_CODE_PATHS}
    missing_code = [name for name, path in code_paths.items() if not path.is_file()]
    if missing_code:
        raise FileNotFoundError(f"cannot freeze IDO task contract; source missing: {missing_code}")
    experiment = {
        "schema_version": TASK_CONTRACT_SCHEMA_VERSION,
        "method_key": "ido",
        "dataset_key": task.dataset.key,
        "noise_protocol": task.protocol,
        "expected_classes": task.dataset.expected_classes,
        "path_maps": list(task.dataset.path_maps),
        "epochs": int(config.epochs),
        "stage1_epochs": int(stage1_epochs),
        "stage2_epochs": int(stage2_epochs),
        "validation_seed": int(config.validation_seed),
        "official_test_selected_only": True,
        "posthoc_oracle_test": False,
        "checkpoint_protocol": "fixed_clean_validation_v1",
        "bmm_mode": config.bmm_mode,
        "bmm_scope": ido_bmm_scope(config.bmm_mode),
        "batch_config_sha256": sha256_file(config.path),
        "assets": assets,
        "code_sha256": {name: sha256_file(path) for name, path in code_paths.items()},
    }
    experiment_signature = canonical_json_sha256(experiment)
    task_identity = {
        "task_id": task.task_id,
        "attempt_id": task.attempt_id,
        "seed": int(task.seed),
        "experiment_signature": experiment_signature,
    }
    return {
        "schema_version": TASK_CONTRACT_SCHEMA_VERSION,
        "experiment": experiment,
        "experiment_signature": experiment_signature,
        "task": task_identity,
        "task_signature": canonical_json_sha256(task_identity),
    }


def is_complete(task: IDOTask, config: IDOBatchConfig) -> tuple[bool, str]:
    required = (
        task.run_dir / "result.json",
        task.run_dir / "train_log.csv",
        task.run_dir / "selection_policy.json",
        task.run_dir / "stage1_wrong_event.csv",
        task.run_dir / "bmm_class_fits.csv",
        *(task.run_dir / "checkpoints" / name for name in CHECKPOINT_FILENAMES),
        task.run_root / "validation_manifest.csv",
        task.run_root / "validation_manifest.json",
        task.contract_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, f"missing required artifacts: {missing}"
    try:
        result = read_json(task.run_dir / "result.json")
        policy = read_json(task.run_dir / "selection_policy.json")
        logs = read_csv(task.run_dir / "train_log.csv")
        stage1_rows = read_csv(task.run_dir / "stage1_wrong_event.csv")
        bmm_rows = read_csv(task.run_dir / "bmm_class_fits.csv")
        canonical_paths = (task.input.input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
        canonical_labels = np.load(task.input.input_dir / "labels.npy", allow_pickle=True).astype(str)
        manifest_rows = read_csv(task.run_root / "validation_manifest.csv")
        validate_manifest(manifest_rows, len(canonical_paths), expected_paths=canonical_paths)
        contract = read_json(task.contract_path)
        expected_contract = build_task_contract(config, task)
    except Exception as exc:
        return False, f"unreadable completion artifact: {exc}"
    contract_sha256 = canonical_json_sha256(contract)
    if canonical_json_sha256(contract) != canonical_json_sha256(expected_contract):
        return False, "task contract does not match the current frozen config/input/code identity"
    if result.get("method_key") != "ido" or int(result.get("seed", -1)) != task.seed:
        return False, "result metadata does not match stable task ID"
    if result.get("method") != ido_adaptation_name(config.bmm_mode):
        return False, "result method identity is not the controlled IDO adaptation"
    if policy.get("method_key") != "ido":
        return False, "selection policy method identity is not IDO"
    if result.get("ido_task_id") != task.task_id or result.get("ido_task_contract_sha256") != contract_sha256:
        return False, "result does not attest to this task contract"
    if result.get("ido_experiment_signature") != contract.get("experiment_signature"):
        return False, "result experiment signature does not match this task contract"
    if policy.get("ido_task_contract_sha256") != contract_sha256 or policy.get("ido_experiment_signature") != contract.get("experiment_signature"):
        return False, "selection policy does not attest to this task contract"
    if result.get("bmm_degenerate_policy") != "fail_closed_record_and_abort":
        return False, "result does not use fail-closed BMM policy"
    if result.get("bmm_mode") != config.bmm_mode or result.get("bmm_scope") != ido_bmm_scope(config.bmm_mode):
        return False, "result BMM mode/scope does not match the frozen batch config"
    if policy.get("ido_bmm_mode") != config.bmm_mode or policy.get("ido_bmm_scope") != ido_bmm_scope(config.bmm_mode):
        return False, "selection policy BMM mode/scope does not match the frozen batch config"
    if result.get("d1_recording_status") != "complete":
        return False, "IDO D1 output is incomplete"
    expected_stage1 = config.stage1_epochs if config.stage1_epochs is not None else config.epochs // 3
    expected_stage2 = config.epochs - expected_stage1
    if result.get("stage1_epochs") != expected_stage1 or result.get("stage2_epochs") != expected_stage2:
        return False, "result stage budget does not match the frozen batch config"
    if result.get("official_test_evaluation") != "validation_selected_only":
        return False, "result does not use the required validation-selected-only official-test protocol"
    if result.get("checkpoint_protocol") != contract["experiment"]["checkpoint_protocol"]:
        return False, "result checkpoint protocol does not match the frozen task contract"
    for key in ("validation_selected_test_top1", "validation_selected_test_top5", "best_top1", "best_top5"):
        if not finite_fraction(result.get(key)):
            return False, f"result has missing/non-finite/out-of-range {key}"
    best_epoch = result.get("best_epoch")
    if not isinstance(best_epoch, int) or not expected_stage1 < best_epoch <= config.epochs:
        return False, "result best_epoch is outside the Stage-2 global epoch range"
    if not isinstance(result.get("train_samples"), int) or int(result["train_samples"]) <= 0:
        return False, "result train_samples is missing or invalid"
    manifest_train_samples = sum(row.get("partition") == "training_pool" for row in manifest_rows)
    if int(result["train_samples"]) != manifest_train_samples:
        return False, "result train_samples does not match the immutable training-pool manifest"
    canonical_manifests = ("validation_manifest.csv", "validation_manifest.json")
    for name in canonical_manifests:
        if (task.run_root / name).read_bytes() != (task.input.validation_dir / name).read_bytes():
            return False, f"attempt {name} differs from its canonical validation manifest"
    expected_epochs = list(range(1, config.epochs + 1))
    try:
        epochs = [int(row["epoch"]) for row in logs]
    except (KeyError, ValueError):
        return False, "train log has no parseable epoch sequence"
    if epochs != expected_epochs:
        return False, f"train log is incomplete: {epochs[:3]}... expected 1..{config.epochs}"
    stage1_error = validate_stage1_wrong_event_artifact(
        task=task,
        rows=stage1_rows,
        canonical_paths=canonical_paths,
        canonical_labels=canonical_labels,
        manifest_rows=manifest_rows,
        expected_stage1_epochs=expected_stage1,
    )
    if stage1_error:
        return False, stage1_error
    bmm_error = validate_bmm_fit_artifact(
        rows=bmm_rows,
        stage1_rows=stage1_rows,
        manifest_rows=manifest_rows,
        canonical_labels=canonical_labels,
        expected_classes=task.dataset.expected_classes,
        expected_stage1_epochs=expected_stage1,
        expected_stage2_epochs=expected_stage2,
        bmm_mode=config.bmm_mode,
    )
    if bmm_error:
        return False, bmm_error
    try:
        checkpoint_errors = validate_checkpoints(task, result, contract, expected_stage1, expected_stage2)
    except Exception as exc:
        return False, f"checkpoint validation failed to run: {exc}"
    if checkpoint_errors:
        return False, checkpoint_errors
    return True, "strict IDO completion audit passed"


def validate_stage1_wrong_event_artifact(
    *,
    task: IDOTask,
    rows: list[dict[str, str]],
    canonical_paths: Sequence[str],
    canonical_labels: np.ndarray,
    manifest_rows: Sequence[dict[str, str]],
    expected_stage1_epochs: int,
) -> str:
    """Verify the Stage-1 wrong-event CSV against the immutable pool/checkpoint.

    The artifact is not a free-form diagnostic: it records one observed-label
    row for every training-pool sample in stable source-index order.  Comparing
    it with the frozen Stage-1 checkpoint prevents a truncated or substituted
    CSV from making a failed run look complete.  No clean label is read here.
    """
    if canonical_labels.shape != (len(canonical_paths),):
        return "Stage-1 audit cannot align V1 labels with canonical paths"
    training_indices = sorted(
        int(row["index"]) for row in manifest_rows if row.get("partition") == "training_pool"
    )
    if len(rows) != len(training_indices):
        return (
            "stage1_wrong_event.csv row count does not equal the immutable training-pool sample count: "
            f"{len(rows)} != {len(training_indices)}"
        )
    csv_wrong_event: dict[int, int] = {}
    for position, (row, expected_index) in enumerate(zip(rows, training_indices)):
        required = ("index", "path", "observed_label", "wrong_event")
        missing = [key for key in required if key not in row or row[key] == ""]
        if missing:
            return f"stage1_wrong_event.csv row {position} is missing required fields: {missing}"
        try:
            index = int(row["index"])
            wrong_event = int(row["wrong_event"])
        except (TypeError, ValueError):
            return f"stage1_wrong_event.csv row {position} has a non-integer index or wrong_event"
        if index != expected_index:
            return (
                "stage1_wrong_event.csv must be a unique stable-index training-pool sequence; "
                f"row {position} has index={index}, expected={expected_index}"
            )
        if row["path"] != canonical_paths[index]:
            return f"stage1_wrong_event.csv path does not match V1 paths.txt at index {index}"
        if row["observed_label"] != str(canonical_labels[index]):
            return f"stage1_wrong_event.csv observed_label does not match V1 labels.npy at index {index}"
        if wrong_event < 0 or wrong_event > expected_stage1_epochs:
            return (
                "stage1_wrong_event.csv wrong_event is outside the one-prediction-per-Stage-1-epoch range "
                f"at index {index}: {wrong_event}"
            )
        csv_wrong_event[index] = wrong_event

    try:
        import torch

        path = task.run_dir / "checkpoints" / "stage1_base.pt"
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch releases before weights_only accepted this argument.
            payload = torch.load(path, map_location="cpu")
        checkpoint_wrong_event = np.asarray(payload.get("wrong_event"), dtype=np.float64)
    except Exception as exc:
        return f"stage1_wrong_event.csv cannot be cross-checked against stage1_base.pt: {type(exc).__name__}: {exc}"
    if checkpoint_wrong_event.shape != (len(canonical_paths),):
        return "stage1_base checkpoint wrong_event does not align with the full V1 source pool"
    if not np.all(np.isfinite(checkpoint_wrong_event)) or not np.all(checkpoint_wrong_event == np.floor(checkpoint_wrong_event)):
        return "stage1_base checkpoint wrong_event is not a finite integer vector"
    if np.any(checkpoint_wrong_event < 0) or np.any(checkpoint_wrong_event > expected_stage1_epochs):
        return "stage1_base checkpoint wrong_event is outside the Stage-1 range"
    training_set = set(training_indices)
    for index, value in enumerate(checkpoint_wrong_event.astype(np.int64).tolist()):
        if index in training_set:
            if csv_wrong_event[index] != value:
                return f"stage1_wrong_event.csv disagrees with stage1_base checkpoint at training-pool index {index}"
        elif value != 0:
            return f"stage1_base checkpoint wrong_event is nonzero outside the immutable training pool at index {index}"
    return ""


def validate_bmm_fit_artifact(
    *,
    rows: list[dict[str, str]],
    stage1_rows: Sequence[dict[str, str]],
    manifest_rows: Sequence[dict[str, str]],
    canonical_labels: np.ndarray,
    expected_classes: int,
    expected_stage1_epochs: int,
    expected_stage2_epochs: int,
    bmm_mode: str = "classwise",
) -> str:
    """Require every BMM fit actually consumed by Stage 2.

    The class-wise variant requires one row per observed class and fit round.
    The explicit global variant requires exactly one all-training-pool row per
    fit round; it must never masquerade as 200 independent class fits.  Both
    paths intentionally omit the terminal refit because no later optimizer
    step consumes it.
    """
    bmm_mode = resolve_ido_bmm_mode(bmm_mode)
    expected_scope = ido_bmm_scope(bmm_mode)
    classes = sorted(set(np.asarray(canonical_labels, dtype=str).tolist()))
    if len(classes) != expected_classes:
        return "BMM audit observed-label vocabulary does not match the configured class count"
    training_indices = sorted(
        int(row["index"]) for row in manifest_rows if row.get("partition") == "training_pool"
    )
    training_labels = [str(canonical_labels[index]) for index in training_indices]
    support = {label: training_labels.count(label) for label in classes}
    if any(count <= 0 for count in support.values()):
        return "BMM audit found an empty observed-label class in the training pool"
    stage1_by_index: dict[int, int] = {}
    for row in stage1_rows:
        try:
            stage1_by_index[int(row["index"])] = int(row["wrong_event"])
        except (KeyError, TypeError, ValueError):
            return "BMM audit cannot parse the already-validated Stage-1 wrong-event artifact"

    expected_rows: list[tuple[str, int, int | str, str, int]] = []
    for stage2_epoch in range(0, expected_stage2_epochs):
        fit_round = "stage2_initial" if stage2_epoch == 0 else "stage2_epoch_update"
        if bmm_mode == "global":
            expected_rows.append((fit_round, stage2_epoch, "", "", len(training_indices)))
        else:
            for label_id, label in enumerate(classes):
                expected_rows.append((fit_round, stage2_epoch, label_id, label, support[label]))
    if len(rows) != len(expected_rows):
        display_mode = "class-wise" if bmm_mode == "classwise" else "global"
        return (
            f"bmm_class_fits.csv does not contain the required {display_mode} initial plus nonterminal Stage-2 fits: "
            f"{len(rows)} != {len(expected_rows)}"
        )

    numeric_fields = (
        "wrong_event_min",
        "wrong_event_max",
        "alpha_1",
        "beta_1",
        "weight_1",
        "alpha_2",
        "beta_2",
        "weight_2",
        "component_1_normalized_mean",
        "component_2_normalized_mean",
        "component_1_soft_assignment",
        "component_1_kish_ess",
        "component_1_weight_times_n",
        "component_1_max_responsibility",
        "component_2_soft_assignment",
        "component_2_kish_ess",
        "component_2_weight_times_n",
        "component_2_max_responsibility",
        "posterior_1_min",
        "posterior_1_max",
        "posterior_2_min",
        "posterior_2_max",
        "cdf_1_min",
        "cdf_1_max",
        "cdf_2_min",
        "cdf_2_max",
        "posterior_mass_sum",
        "posterior_mass_deficit",
        "posterior_mass_min",
        "posterior_mass_max",
    )
    for position, (row, expected) in enumerate(zip(rows, expected_rows)):
        fit_round, stage2_epoch, label_id, label, expected_sample_count = expected
        required = (
            "bmm_mode", "bmm_scope", "fit_round", "stage2_epoch", "observed_label_id", "observed_label", "sample_count",
            "unique_wrong_event_count", "bmm_status", "bmm_reason", "bmm_fit_iterations", *numeric_fields,
            "component_1_responsibility_ge_half_count", "component_2_responsibility_ge_half_count",
            "bmm_prediction_clipped_count",
        )
        if "bmm_reason" not in row:
            return f"bmm_class_fits.csv row {position} is missing the bmm_reason schema column"
        # A successful BMM fit has an intentionally empty bmm_reason value,
        # while all other fields must have a parseable value.
        blank_allowed = {"bmm_reason"}
        if bmm_mode == "global":
            blank_allowed.update({"observed_label_id", "observed_label"})
        missing = [key for key in required if key not in blank_allowed and (key not in row or row[key] == "")]
        if missing:
            return f"bmm_class_fits.csv row {position} is missing required fields: {missing}"
        try:
            actual_epoch = int(row["stage2_epoch"])
            actual_label_id = row["observed_label_id"] if bmm_mode == "global" else int(row["observed_label_id"])
            sample_count = int(row["sample_count"])
            unique_count = int(row["unique_wrong_event_count"])
            iterations = int(row["bmm_fit_iterations"])
            values = {field: float(row[field]) for field in numeric_fields}
            ge_half = (int(row["component_1_responsibility_ge_half_count"]), int(row["component_2_responsibility_ge_half_count"]))
            clipped = int(row["bmm_prediction_clipped_count"])
        except (TypeError, ValueError):
            return f"bmm_class_fits.csv row {position} has a non-numeric required value"
        if row["fit_round"] != fit_round or actual_epoch != stage2_epoch:
            return f"bmm_class_fits.csv row {position} has an unexpected fit round/Stage-2 epoch"
        if row["bmm_mode"] != bmm_mode or row["bmm_scope"] != expected_scope:
            return f"bmm_class_fits.csv row {position} BMM mode/scope differs from the frozen protocol"
        if actual_label_id != label_id or row["observed_label"] != label:
            return f"bmm_class_fits.csv row {position} has an unexpected observed-label class identity"
        if row["bmm_status"] != "fitted" or str(row.get("bmm_reason", "")):
            return f"bmm_class_fits.csv row {position} is not a successful, reason-free BMM fit"
        if sample_count != expected_sample_count or unique_count < 2 or unique_count > sample_count:
            return f"bmm_class_fits.csv row {position} has invalid BMM support or diversity metadata"
        if iterations != (20 if stage2_epoch == 0 else 2):
            return f"bmm_class_fits.csv row {position} has an unexpected BMM iteration budget"
        if not all(math.isfinite(value) for value in values.values()):
            return f"bmm_class_fits.csv row {position} contains a non-finite BMM diagnostic"
        if values["wrong_event_min"] < 0.0 or values["wrong_event_max"] < values["wrong_event_min"]:
            return f"bmm_class_fits.csv row {position} has an invalid wrong-event range"
        maximum_possible = expected_stage1_epochs + 2 * stage2_epoch
        if values["wrong_event_max"] > maximum_possible:
            return f"bmm_class_fits.csv row {position} exceeds the Stage-1/Stage-2 wrong-event bound"
        if any(values[field] <= 0.0 for field in ("alpha_1", "beta_1", "alpha_2", "beta_2")):
            return f"bmm_class_fits.csv row {position} has non-positive Beta parameters"
        if not (0.0 < values["weight_1"] < 1.0 and 0.0 < values["weight_2"] < 1.0):
            return f"bmm_class_fits.csv row {position} has collapsed mixture weights"
        if not math.isclose(values["weight_1"] + values["weight_2"], 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            return f"bmm_class_fits.csv row {position} mixture weights do not sum to one"
        means = (values["component_1_normalized_mean"], values["component_2_normalized_mean"])
        expected_means = (
            values["alpha_1"] / (values["alpha_1"] + values["beta_1"]),
            values["alpha_2"] / (values["alpha_2"] + values["beta_2"]),
        )
        if not (0.0 <= means[0] <= means[1] <= 1.0):
            return f"bmm_class_fits.csv row {position} violates ordered component-mean semantics"
        if not all(math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-6) for value, expected in zip(means, expected_means)):
            return f"bmm_class_fits.csv row {position} component means do not match its Beta parameters"
        if any(value < 0.0 or value > sample_count for value in (*ge_half, clipped)):
            return f"bmm_class_fits.csv row {position} has invalid BMM responsibility/clipping counts"
        if not (0.0 <= values["posterior_mass_min"] <= values["posterior_mass_max"] <= 1.0):
            return f"bmm_class_fits.csv row {position} has invalid posterior-mass range"
        if values["posterior_mass_deficit"] < 0.0 or values["posterior_mass_sum"] > sample_count + 1.0e-6:
            return f"bmm_class_fits.csv row {position} violates the official epsilon posterior-mass semantics"
        if any(
            not (0.0 <= values[field] <= 1.0)
            for field in ("posterior_1_min", "posterior_1_max", "posterior_2_min", "posterior_2_max", "cdf_1_min", "cdf_1_max", "cdf_2_min", "cdf_2_max")
        ):
            return f"bmm_class_fits.csv row {position} has out-of-range posterior/CDF diagnostics"
        if stage2_epoch == 0:
            initial_values = (
                [stage1_by_index[index] for index in training_indices]
                if bmm_mode == "global"
                else [stage1_by_index[index] for index in training_indices if str(canonical_labels[index]) == label]
            )
            if (
                unique_count != len(set(initial_values))
                or not math.isclose(values["wrong_event_min"], float(min(initial_values)), rel_tol=0.0, abs_tol=1.0e-8)
                or not math.isclose(values["wrong_event_max"], float(max(initial_values)), rel_tol=0.0, abs_tol=1.0e-8)
            ):
                return f"bmm_class_fits.csv initial row {position} does not match Stage-1 wrong-event statistics"
    return ""


def summarize_all(config: IDOBatchConfig, protocols: Sequence[str]) -> None:
    root = config.output_root / "orchestration"
    root.mkdir(parents=True, exist_ok=True)
    task_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in config.datasets:
        for protocol in protocols:
            groups: dict[str, dict[int, tuple[dict[str, Any], IDOTask]]] = {}
            duplicate_seeds: dict[str, set[int]] = {}
            for attempt in discover_attempts(config):
                matching = [task for task in build_plan(config, (protocol,), attempt=attempt) if task.dataset.key == dataset.key]
                for task in matching:
                    complete, detail = is_complete(task, config)
                    signature = ""
                    if complete:
                        result = read_json(task.run_dir / "result.json")
                        signature = str(result["ido_experiment_signature"])
                        group = groups.setdefault(signature, {})
                        if task.seed in group:
                            duplicate_seeds.setdefault(signature, set()).add(task.seed)
                        else:
                            group[task.seed] = (result, task)
                    task_rows.append(
                        {
                            "task_id": task.task_id,
                            "attempt": task.attempt_id,
                            "experiment_signature": signature,
                            "status": "COMPLETED" if complete else "INCOMPLETE",
                            "detail": detail,
                            "run_dir": str(task.run_dir),
                        }
                    )
            if not groups:
                groups = {"": {}}
            for signature, results in groups.items():
                values = [float(results[seed][0]["validation_selected_test_top1"]) for seed in SEEDS if seed in results]
                duplicate = sorted(duplicate_seeds.get(signature, set()))
                complete_group = len(values) == len(SEEDS) and not duplicate
                def result_value(seed: int, key: str) -> Any:
                    item = results.get(seed)
                    return "" if item is None else item[0].get(key, "")

                def attempt_value(seed: int) -> str:
                    item = results.get(seed)
                    return "" if item is None else item[1].attempt_id

                summary_rows.append(
                    {
                        "dataset": dataset.display_name,
                        "dataset_key": dataset.key,
                        "noise_protocol": protocol,
                        "method_key": "ido",
                        "experiment_signature": signature,
                        "required_seed_count": len(SEEDS),
                        "completed_seed_count": len(values),
                        "status": "CONFLICTING_DUPLICATE_SEED" if duplicate else "COMPLETE_3_SEEDS" if complete_group else "INCOMPLETE",
                        "duplicate_seeds": ",".join(str(seed) for seed in duplicate),
                        "mean_validation_selected_official_test_top1": round(statistics.mean(values), 12) if complete_group else "",
                        "sample_sd_validation_selected_official_test_top1": round(statistics.stdev(values), 12) if complete_group else "",
                        "seed1_top1": result_value(1, "validation_selected_test_top1"),
                        "seed42_top1": result_value(42, "validation_selected_test_top1"),
                        "seed88_top1": result_value(88, "validation_selected_test_top1"),
                        "seed1_attempt": attempt_value(1),
                        "seed42_attempt": attempt_value(42),
                        "seed88_attempt": attempt_value(88),
                    }
                )
    write_csv(root / "ido_run_status.csv", task_rows)
    write_csv(root / "ido_3seed_summary.csv", summary_rows)
    write_json(root / "ido_orchestration_summary.json", {"planned_protocols": list(protocols), "summary": summary_rows})


def discover_attempts(config: IDOBatchConfig) -> list[int]:
    root = config.output_root / "attempts"
    if not root.is_dir():
        return [1]
    found = sorted(int(path.name[8:]) for path in root.iterdir() if path.is_dir() and path.name.startswith("attempt-") and path.name[8:].isdigit())
    return found or [1]


def find_reusable_completed_task(config: IDOBatchConfig, task: IDOTask) -> IDOTask | None:
    """Return the one prior completed result compatible with a new attempt.

    ``--new-attempt`` exists to isolate incomplete artifacts, not to retrain a
    seed that already has a complete result.  A prior result is reusable only
    when the current immutable task contract still validates it, including the
    code/config/input experiment signature.  More than one valid prior result
    for the same seed is ambiguous and intentionally fails closed rather than
    selecting a result by score or filesystem order.
    """
    if task.attempt <= 1:
        return None
    expected_signature = str(build_task_contract(config, task)["experiment_signature"])
    matches: list[IDOTask] = []
    for attempt in discover_attempts(config):
        if attempt >= task.attempt:
            continue
        prior = next(
            candidate
            for candidate in build_plan(config, (task.protocol,), attempt=attempt)
            if candidate.dataset.key == task.dataset.key and candidate.seed == task.seed
        )
        complete, _ = is_complete(prior, config)
        if not complete:
            continue
        result = read_json(prior.run_dir / "result.json")
        if str(result.get("ido_experiment_signature", "")) == expected_signature:
            matches.append(prior)
    if len(matches) > 1:
        locations = ", ".join(f"{item.attempt_id}:{item.run_dir}" for item in matches)
        raise RuntimeError(
            f"Ambiguous reusable completed IDO task for {task.task_id}; multiple attempts match the "
            f"current frozen experiment signature: {locations}. Resolve the duplicate explicitly; do not retrain or merge it."
        )
    return matches[0] if matches else None


def print_plan(config: IDOBatchConfig, tasks: list[IDOTask], findings: list[dict[str, Any]], *, dry_run: bool) -> None:
    print(f"IDO PLAN tasks={len(tasks)} execute={not dry_run} protocol_frozen={config.protocol_frozen}", flush=True)
    for item in findings:
        print(f"PREFLIGHT {'PASS' if item['ok'] else 'BLOCKED'} {item['dataset']}/{item['protocol']}: {item['detail']}", flush=True)
    for task in tasks:
        action = "PLAN" if dry_run else "RUN"
        print(f"{action} {task.task_id} {task.attempt_id} {' '.join(build_training_command(config, task))}", flush=True)


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def resolve_path(raw: str, base: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def paths_overlap(first: Path, second: Path) -> bool:
    """Return whether two resolved paths are equal or either contains the other."""
    left = first.expanduser().resolve()
    right = second.expanduser().resolve()
    return left == right or left in right.parents or right in left.parents


def finite_fraction(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def validate_checkpoints(
    task: IDOTask,
    result: dict[str, Any],
    contract: dict[str, Any],
    expected_stage1: int,
    expected_stage2: int,
) -> str:
    """Strictly verify that saved checkpoints are readable and match the result.

    This is an orchestration audit, not an epoch-level resume implementation.
    It rejects malformed or mismatched artifacts before a future --execute run is
    allowed to skip them.
    """
    import torch

    contract_sha256 = canonical_json_sha256(contract)
    contract_bmm_mode = resolve_ido_bmm_mode(contract["experiment"].get("bmm_mode", "classwise"))
    expected_common = {
        "method": ido_adaptation_name(contract_bmm_mode),
        "seed": task.seed,
        "ido_adaptation": ido_adaptation_name(contract_bmm_mode),
        "stage1_epochs": expected_stage1,
        "stage2_epochs": expected_stage2,
        "bmm_mode": contract_bmm_mode,
        "bmm_scope": ido_bmm_scope(contract_bmm_mode),
        "task_contract_sha256": contract_sha256,
        "task_id": task.task_id,
        "experiment_signature": contract["experiment_signature"],
    }
    expected_classes = sorted(set(np.load(task.input.input_dir / "labels.npy", allow_pickle=True).astype(str).tolist()))
    if len(expected_classes) != task.dataset.expected_classes:
        return "prepared observed-label vocabulary does not match the configured class count"
    effective_config_path = task.run_root / "resolved_config.yaml"
    if not effective_config_path.is_file():
        return "run directory lacks the effective resolved_config.yaml needed for checkpoint-structure audit"
    try:
        # This is the configuration written by the invoked training entry after
        # applying its defaults and CLI overrides, not merely the prepared V1
        # source config.  The batch command does not override rank/targets, but
        # reading the effective file keeps the audit coupled to what trained.
        effective_config = yaml.safe_load(effective_config_path.read_text(encoding="utf-8")) or {}
        lora_config = effective_config.get("lora", {})
        feature_config = effective_config.get("feature", {})
        if not isinstance(lora_config, dict) or not isinstance(feature_config, dict):
            return "prepared V1 resolved_config.yaml has an invalid feature/lora mapping"
        expected_lora_rank = int(lora_config.get("rank", 8))
        target_modules_raw = lora_config.get("target_modules", "qkv")
        if not isinstance(target_modules_raw, str):
            return "prepared V1 LoRA target_modules must be a comma-separated string"
        expected_target_modules = tuple(item.strip() for item in target_modules_raw.split(",") if item.strip())
        backend = str(feature_config.get("backend", "dinov2_vitb14"))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return f"unable to obtain effective DINOv2-LoRA checkpoint audit settings: {exc}"
    if expected_lora_rank <= 0:
        return "prepared V1 LoRA rank is not a positive integer"
    if not expected_target_modules:
        return "prepared V1 LoRA target_modules is empty"
    if backend != "dinov2_vitb14":
        return f"controlled IDO checkpoint audit requires feature.backend=dinov2_vitb14, got {backend!r}"
    try:
        expected_structure = build_dinov2_vitb14_lora_structure_contract(
            target_modules=expected_target_modules,
            rank=expected_lora_rank,
            num_classes=task.dataset.expected_classes,
        )
    except ValueError as exc:
        return f"effective LoRA configuration cannot form a controlled DINOv2-B/14 structure contract: {exc}"
    expected_structure_sha256 = canonical_structure_sha256(expected_structure)
    payloads: dict[str, dict[str, Any]] = {}
    state_layout: tuple[tuple[str, tuple[int, ...], str], ...] | None = None
    for name in CHECKPOINT_FILENAMES:
        path = task.run_dir / "checkpoints" / name
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:  # PyTorch releases before weights_only accepted this argument.
                payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            return f"checkpoint is unreadable ({name}): {type(exc).__name__}: {exc}"
        if not isinstance(payload, dict):
            return f"checkpoint payload is not a mapping: {name}"
        state = payload.get("state_dict")
        if not isinstance(state, dict) or not state:
            return f"checkpoint has no non-empty state_dict: {name}"
        if not isinstance(payload.get("classes"), list) or not payload["classes"]:
            return f"checkpoint has no non-empty classes metadata: {name}"
        if [str(value) for value in payload["classes"]] != expected_classes:
            return f"checkpoint classes metadata does not equal the prepared observed-label vocabulary: {name}"
        checkpoint_structure = payload.get("lora_structure_contract")
        if not isinstance(checkpoint_structure, dict):
            return f"checkpoint lacks the frozen LoRA structure contract: {name}"
        if canonical_structure_sha256(checkpoint_structure) != expected_structure_sha256:
            return f"checkpoint LoRA structure contract does not match the effective controlled DINOv2-B/14 configuration: {name}"
        if payload.get("lora_structure_contract_sha256") != expected_structure_sha256:
            return f"checkpoint LoRA structure contract hash does not match the frozen contract: {name}"
        runtime_structure = payload.get("lora_runtime_structure")
        if not isinstance(runtime_structure, dict) or not isinstance(payload.get("lora_runtime_structure_sha256"), str):
            return f"checkpoint lacks live-model LoRA structure provenance: {name}"
        if canonical_structure_sha256(runtime_structure) != payload["lora_runtime_structure_sha256"]:
            return f"checkpoint live-model LoRA structure provenance hash is invalid: {name}"
        structural_error, layout = validate_ido_trainable_state_dict(
            torch=torch,
            state=state,
            expected_classes=task.dataset.expected_classes,
            expected_lora_rank=expected_lora_rank,
            expected_embed_dim=DINO_VITB14_EMBED_DIM,
            expected_target_modules=expected_target_modules,
            expected_structure=expected_structure,
        )
        if structural_error:
            return f"checkpoint state_dict fails the DINOv2+LoRA structural audit ({name}): {structural_error}"
        runtime_error = validate_lora_runtime_structure_provenance(
            runtime_structure=runtime_structure,
            expected_structure=expected_structure,
            state_layout=layout,
        )
        if runtime_error:
            return f"checkpoint live-model LoRA structure provenance is inconsistent ({name}): {runtime_error}"
        if state_layout is None:
            state_layout = layout
        elif layout != state_layout:
            return f"checkpoint state_dict key/shape layout differs from stage1_base.pt: {name}"
        for key, expected in expected_common.items():
            if payload.get(key) != expected:
                return f"checkpoint metadata mismatch for {name}: {key}"
        if payload.get("checkpoint_protocol") != result.get("checkpoint_protocol"):
            return f"checkpoint protocol mismatch for {name}"
        payloads[name] = payload
    stage1 = payloads["stage1_base.pt"]
    if "wrong_event" not in stage1 or int(stage1.get("base_stage_epoch", 0)) not in range(1, expected_stage1 + 1):
        return "stage1_base checkpoint lacks wrong-event state or a valid label-wave epoch"
    best = payloads["best_val.pt"]
    if best.get("best_epoch") != result.get("best_epoch"):
        return "best_val checkpoint epoch does not match result"
    if not finite_fraction(best.get("best_top1")) or not math.isclose(float(best["best_top1"]), float(result["best_top1"]), rel_tol=0.0, abs_tol=1.0e-8):
        return "best_val checkpoint score does not match result"
    last = payloads["last.pt"]
    if last.get("final_epoch") != expected_stage1 + expected_stage2:
        return "last checkpoint final_epoch does not match frozen stage budget"
    return ""


def validate_ido_trainable_state_dict(
    *,
    torch: Any,
    state: dict[str, Any],
    expected_classes: int,
    expected_lora_rank: int = 8,
    expected_embed_dim: int = DINO_VITB14_EMBED_DIM,
    expected_target_modules: Sequence[str] = ("qkv",),
    expected_structure: dict[str, Any] | None = None,
) -> tuple[str, tuple[tuple[str, tuple[int, ...], str], ...]]:
    """Perform a CPU-only structural audit of a saved IDO LoRA/head state.

    IDO checkpoints intentionally retain only trainable ``backbone.*.lora_*``
    parameters and ``head.*`` (see ``snapshot_trainable_state``).  The allowed
    keys are compared against the independent, frozen DINOv2-B/14 catalog
    resolved from the effective run configuration.  The audit therefore does
    not infer a valid Transformer topology from the checkpoint under review.
    A real DINOv2 instantiation/load remains required in the GPU pilot.
    """
    if not isinstance(expected_classes, int) or expected_classes <= 1:
        return "invalid expected class count", ()
    if not isinstance(expected_lora_rank, int) or expected_lora_rank <= 0:
        return "invalid expected LoRA rank", ()
    if not isinstance(expected_embed_dim, int) or expected_embed_dim <= 0:
        return "invalid expected DINOv2 embedding dimension", ()
    if not expected_target_modules or any(not isinstance(module, str) or not module for module in expected_target_modules):
        return "invalid expected LoRA target module list", ()
    try:
        structure = expected_structure or build_dinov2_vitb14_lora_structure_contract(
            target_modules=expected_target_modules,
            rank=expected_lora_rank,
            num_classes=expected_classes,
        )
    except ValueError as exc:
        return f"invalid frozen DINOv2-B/14 LoRA structure contract: {exc}", ()
    if structure.get("architecture") != "dinov2_vitb14":
        return "frozen LoRA structure contract is not for DINOv2-B/14", ()
    if structure.get("embed_dim") != expected_embed_dim:
        return "frozen LoRA structure contract has an unexpected embedding dimension", ()
    if structure.get("rank") != expected_lora_rank:
        return "frozen LoRA structure contract rank does not match the effective configuration", ()
    if tuple(structure.get("target_modules", [])) != tuple(expected_target_modules):
        return "frozen LoRA structure contract target modules do not match the effective configuration", ()

    head = structure.get("head")
    modules = structure.get("lora_modules")
    if not isinstance(head, dict) or not isinstance(modules, list) or not modules:
        return "frozen LoRA structure contract has no complete head/module specification", ()
    try:
        expected_shapes: dict[str, tuple[int, ...]] = {
            str(head["weight_key"]): tuple(int(value) for value in head["weight_shape"]),
            str(head["bias_key"]): tuple(int(value) for value in head["bias_shape"]),
        }
        expected_module_names: set[str] = set()
        for module in modules:
            if not isinstance(module, dict):
                return "frozen LoRA structure contract contains a malformed module record", ()
            backbone_module = str(module["backbone_module"])
            a_key = str(module["lora_a_key"])
            b_key = str(module["lora_b_key"])
            if backbone_module in expected_module_names or a_key in expected_shapes or b_key in expected_shapes:
                return "frozen LoRA structure contract contains duplicate module/key entries", ()
            expected_module_names.add(backbone_module)
            expected_shapes[a_key] = tuple(int(value) for value in module["lora_a_shape"])
            expected_shapes[b_key] = tuple(int(value) for value in module["lora_b_shape"])
    except (KeyError, TypeError, ValueError):
        return "frozen LoRA structure contract contains invalid key/shape entries", ()

    keys = list(state)
    if any(not isinstance(key, str) or not key for key in keys):
        return "state_dict has an invalid parameter key", ()
    actual_key_set = set(keys)
    expected_key_set = set(expected_shapes)
    if actual_key_set != expected_key_set:
        missing = sorted(expected_key_set - actual_key_set)
        unexpected = sorted(actual_key_set - expected_key_set)
        return (
            "state_dict key set does not equal the frozen DINOv2-B/14 LoRA contract; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        ), ()
    for key, expected_shape in expected_shapes.items():
        value = state[key]
        if not torch.is_tensor(value) or tuple(int(size) for size in value.shape) != expected_shape:
            return f"{key!r} has shape incompatible with the frozen DINOv2-B/14 LoRA contract", ()
    layout: list[tuple[str, tuple[int, ...], str]] = []
    for key in sorted(keys):
        value = state[key]
        if not torch.is_tensor(value) or value.numel() <= 0:
            return f"{key!r} is not a non-empty tensor", ()
        if not (value.is_floating_point() or value.is_complex()):
            return f"{key!r} is not a floating-point trainable tensor", ()
        if not bool(torch.isfinite(value).all().item()):
            return f"{key!r} contains NaN or Inf", ()
        layout.append((key, tuple(int(size) for size in value.shape), str(value.dtype)))
    return "", tuple(layout)


def validate_lora_runtime_structure_provenance(
    *,
    runtime_structure: dict[str, Any],
    expected_structure: dict[str, Any],
    state_layout: tuple[tuple[str, tuple[int, ...], str], ...],
) -> str:
    """Check that checkpoint provenance reports the live injected structure.

    This verifies a training-time diagnostic record against the independently
    frozen structure contract and the saved state layout.  It does not replace
    the forthcoming GPU-pilot ``load_state_dict`` check.
    """
    if runtime_structure.get("record_source") != "live_model_after_inject_lora":
        return "record_source is not live_model_after_inject_lora"
    expected_modules = [str(item["backbone_module"]) for item in expected_structure["lora_modules"]]
    if runtime_structure.get("injected_backbone_modules") != expected_modules:
        return "reported injected backbone modules do not equal the frozen contract"
    if runtime_structure.get("target_modules") != expected_structure["target_modules"]:
        return "reported target modules do not equal the frozen contract"
    if runtime_structure.get("rank") != expected_structure["rank"]:
        return "reported LoRA rank does not equal the frozen contract"
    reported = runtime_structure.get("trainable_parameters")
    if not isinstance(reported, list):
        return "trainable_parameters is not a list"
    expected_layout = [
        {"key": key, "shape": list(shape), "dtype": dtype}
        for key, shape, dtype in state_layout
    ]
    if reported != expected_layout:
        return "reported trainable parameter layout does not equal saved state_dict layout"
    return ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["task_id", "attempt", "status", "detail", "run_dir"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
