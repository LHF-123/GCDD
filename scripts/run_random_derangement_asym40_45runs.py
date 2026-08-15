"""Run the controlled 45-run fixed random-derangement asym40 experiment.

The launcher prepares label-only adapters over the existing formal V1 inputs,
audits all three datasets before training, invokes the existing validation-
selected trainer, enforces PGDF -> budget-matched Dynamic dependencies, and
creates descriptive result summaries after all 45 runs are complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.budget_matching import file_sha256 as budget_file_sha256
from gcdd.budget_matching import load_pgdf_class_budget_schedule
from gcdd.checkpoint_validation import PROTOCOL_NAME
from gcdd.lora_dynamic import selection_update_epochs
from gcdd.noise_realization import align_noise_rows, validate_cyclic_noise_rows
from gcdd.random_derangement import (
    MAPPING_TYPE,
    audit_noise_control,
    compare_validation_manifests,
    file_sha256,
    prepare_input_adapter,
    prepare_mapping,
    prepare_noise_manifest,
    read_csv_with_fields,
)


EXPERIMENT_TAG = "random_derangement_asym40_map20260815_noise42"
DEFAULT_EXPERIMENT_ROOT = Path("outputs") / EXPERIMENT_TAG
MAPPING_ROOT = Path("outputs/noise_mappings/random_derangement_seed20260815")
MANIFEST_ROOT = Path("outputs/noise_manifests/random_derangement_asym40_seed42_map20260815")
PROTOCOL_DIR_NAME = "checkpoint_validation_s20250726"
NOISE_RATE = 0.4
NOISE_SEED = 42
MAPPING_SEED = 20260815
VALIDATION_SEED = 20250726
TRAINING_SEEDS = (1, 42, 88)
FORMAL_EPOCHS = 30
WARMUP_EPOCHS = 5
UPDATE_INTERVAL = 5
DYNAMIC_RATIO = 0.8
FIXED_P = 0.4


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    output_name: str
    artifact_stem: str
    base_input_dir: Path
    original_noise_index: Path
    formal_validation_dir: Path
    peer_validation_dirs: tuple[Path, ...]
    expected_source_samples: int
    expected_classes: int
    expected_pool: tuple[int, int, int]


@dataclass(frozen=True)
class PreparedDataset:
    spec: DatasetSpec
    mapping_file: Path
    mapping_sha256: str
    noise_manifest: Path
    noise_manifest_sha256: str
    input_dir: Path
    protocol_dir: Path
    validation_manifest: Path
    validation_manifest_sha256: str
    audit: dict[str, Any]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    runner_key: str
    phase: int


DATASETS = (
    DatasetSpec(
        key="cub",
        display_name="CUB-200-2011",
        output_name="CUB_200_2011",
        artifact_stem="cub200",
        base_input_dir=Path("outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42"),
        original_noise_index=Path("outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"),
        formal_validation_dir=Path(
            "outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/"
            "checkpoint_validation_s20250726"
        ),
        peer_validation_dirs=(
            Path(
                "outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/"
                "checkpoint_validation_missing9_s20250726"
            ),
            Path(
                "outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/"
                "checkpoint_validation_dynamic_budget_matched_s20250726"
            ),
        ),
        expected_source_samples=5994,
        expected_classes=200,
        expected_pool=(5400, 3251, 2149),
    ),
    DatasetSpec(
        key="cars",
        display_name="Stanford Cars",
        output_name="Stanford_Cars",
        artifact_stem="stanford_cars",
        base_input_dir=Path("outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42"),
        original_noise_index=Path("outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"),
        formal_validation_dir=Path(
            "outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/"
            "checkpoint_validation_s20250726"
        ),
        peer_validation_dirs=(
            Path(
                "outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/"
                "checkpoint_validation_missing9_s20250726"
            ),
            Path(
                "outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/"
                "checkpoint_validation_dynamic_budget_matched_s20250726"
            ),
        ),
        expected_source_samples=8144,
        expected_classes=196,
        expected_pool=(7410, 4453, 2957),
    ),
    DatasetSpec(
        key="aircraft",
        display_name="FGVC-Aircraft",
        output_name="FGVC_Aircraft",
        artifact_stem="fgvc_aircraft",
        base_input_dir=Path("outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42"),
        original_noise_index=Path("outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"),
        formal_validation_dir=Path(
            "outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/"
            "checkpoint_validation_s20250726"
        ),
        peer_validation_dirs=(
            Path(
                "outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/"
                "checkpoint_validation_missing9_s20250726"
            ),
            Path(
                "outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/"
                "checkpoint_validation_dynamic_budget_matched_s20250726"
            ),
        ),
        expected_source_samples=6667,
        expected_classes=100,
        expected_pool=(6067, 3656, 2411),
    ),
)

METHODS = (
    MethodSpec("all_noisy", "LoRA all-noisy CE", "all_noisy", 1),
    MethodSpec("dynamic_small_loss", "Dynamic small-loss", "dynamic_r08", 1),
    MethodSpec("jal_ce", "JAL-CE", "jal_ce", 1),
    MethodSpec("pgdf", "PGDF", "pgdf_fixed", 1),
    MethodSpec("budget_matched_dynamic", "Budget-matched Dynamic", "dynamic_budget_matched", 3),
)

METHOD_ALIASES = {
    "all_noisy": "all_noisy",
    "all-noisy": "all_noisy",
    "dynamic": "dynamic_small_loss",
    "dynamic_r08": "dynamic_small_loss",
    "dynamic_small_loss": "dynamic_small_loss",
    "jal": "jal_ce",
    "jal_ce": "jal_ce",
    "pgdf": "pgdf",
    "pgdf_fixed": "pgdf",
    "bmd": "budget_matched_dynamic",
    "dynamic_budget_matched": "budget_matched_dynamic",
    "budget_matched_dynamic": "budget_matched_dynamic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Prepare/audit artifacts and print the selected plan without training.")
    parser.add_argument("--preflight-only", action="store_true", help="Prepare/audit mappings, manifests, validation parity, and run plan only.")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip strictly complete runs (enabled by default).")
    parser.add_argument("--force", action="store_true", help="Archive and rerun selected complete runs; never overwrite them in place.")
    parser.add_argument("--dataset", action="append", choices=[spec.key for spec in DATASETS], help="Limit training/debug output; preflight still audits all datasets.")
    parser.add_argument("--method", action="append", help="Limit methods (all_noisy, dynamic, jal_ce, pgdf, or bmd).")
    parser.add_argument("--training-seed", action="append", type=int, choices=list(TRAINING_SEEDS), help="Limit training seeds.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Forward a stored-path mapping to the existing trainer.")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT, help="Isolated output root for this experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    experiment_root = normalize_repo_path(args.experiment_root)
    validate_cli(args)

    print("[Phase 0] Preparing fixed mappings/manifests and auditing all three datasets.", flush=True)
    prepared = prepare_all_datasets(experiment_root)
    mapping_hashes = {key: item.mapping_sha256 for key, item in prepared.items()}
    master_plan = build_run_plan(experiment_root, mapping_hashes)
    write_run_plan(experiment_root / "run_plan.csv", master_plan)
    selected_plan = filter_plan(master_plan, args)
    write_run_plan(experiment_root / "selected_run_plan.csv", selected_plan)
    print_plan_summary(master_plan, selected_plan)

    if args.dry_run:
        print_dry_run(selected_plan)
        print("DRY RUN PASS: no training process was started.", flush=True)
        return
    if args.preflight_only:
        print("PREFLIGHT PASS: all mappings, manifests, validation identities, and pool counts are frozen.", flush=True)
        return

    phase1 = [row for row in selected_plan if int(row["phase"]) == 1]
    if phase1:
        print(f"[Phase 1] Running/resuming {len(phase1)} non-BMD runs.", flush=True)
        run_grouped_tasks(phase1, prepared, experiment_root, args)

    phase3 = [row for row in selected_plan if int(row["phase"]) == 3]
    if phase3:
        dependencies = dependency_rows(master_plan, phase3)
        print(f"[Phase 2] Auditing {len(dependencies)} exact PGDF budget dependencies.", flush=True)
        audit_pgdf_dependencies(dependencies, prepared, experiment_root)
        print(f"[Phase 3] Running/resuming {len(phase3)} Budget-matched Dynamic runs.", flush=True)
        run_grouped_tasks(phase3, prepared, experiment_root, args)

    status_rows = collect_status(master_plan, prepared)
    write_csv(experiment_root / "run_status.csv", status_rows, status_fields())
    complete_count = sum(row["status"] == "COMPLETED" for row in status_rows)
    if complete_count == len(master_plan):
        write_final_outputs(master_plan, prepared, experiment_root)
        print("All 45 formal runs are complete; validation-selected official-test summaries were written.", flush=True)
        print_mechanism_audit_command(experiment_root)
    else:
        print(
            f"Selected work finished; master experiment is {complete_count}/{len(master_plan)} complete. "
            "Final result summaries are emitted only after all 45 strict completions.",
            flush=True,
        )


def validate_cli(args: argparse.Namespace) -> None:
    if args.dry_run and args.preflight_only:
        raise ValueError("Use either --dry-run or --preflight-only, not both.")
    if args.method:
        unknown = sorted({item for item in args.method if item not in METHOD_ALIASES})
        if unknown:
            raise ValueError(f"Unknown --method values: {unknown}; supported aliases: {sorted(METHOD_ALIASES)}")
    for item in args.path_map:
        if "=" not in item or not all(item.split("=", 1)):
            raise ValueError(f"Invalid --path-map {item!r}; expected OLD=NEW.")


def prepare_all_datasets(experiment_root: Path) -> dict[str, PreparedDataset]:
    prepared: dict[str, PreparedDataset] = {}
    audits: list[dict[str, Any]] = []
    for spec in DATASETS:
        if not spec.base_input_dir.is_dir():
            raise FileNotFoundError(f"Formal V1 input directory is missing: {spec.base_input_dir}")
        train_paths = (spec.base_input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
        if len(train_paths) != spec.expected_source_samples:
            raise ValueError(
                f"{spec.display_name}: source size {len(train_paths)} != frozen {spec.expected_source_samples}."
            )
        original_rows = align_noise_rows(
            train_paths,
            read_csv_with_fields(spec.original_noise_index)[0],
            source_name=str(spec.original_noise_index),
        )
        cyclic_summary = validate_cyclic_noise_rows(
            original_rows,
            noise_seed=NOISE_SEED,
            noise_rate=NOISE_RATE,
        )
        if len(cyclic_summary["class_order"]) != spec.expected_classes:
            raise ValueError(f"{spec.display_name}: canonical class-universe size changed.")

        mapping_file = MAPPING_ROOT / f"{spec.artifact_stem}_random_derangement.json"
        mapping, mapping_sha256 = prepare_mapping(
            mapping_file,
            dataset=spec.display_name,
            class_order=cyclic_summary["class_order"],
            mapping_seed=MAPPING_SEED,
        )
        manifest = MANIFEST_ROOT / f"{spec.artifact_stem}_random_derangement_asym40.csv"
        manifest_rows, manifest_metadata = prepare_noise_manifest(
            manifest,
            dataset=spec.display_name,
            original_noise_index=spec.original_noise_index,
            train_paths=train_paths,
            mapping=mapping,
            mapping_file=mapping_file,
            mapping_sha256=mapping_sha256,
            noise_seed=NOISE_SEED,
            mapping_seed=MAPPING_SEED,
            validation_seed=VALIDATION_SEED,
            noise_rate=NOISE_RATE,
        )
        protocol_dir = experiment_root / spec.output_name / PROTOCOL_DIR_NAME
        audit = audit_noise_control(
            dataset=spec.display_name,
            train_paths=train_paths,
            original_noise_index=spec.original_noise_index,
            alternative_manifest=manifest,
            validation_source_dir=spec.formal_validation_dir,
            validation_destination_dir=protocol_dir,
            mapping=mapping,
            mapping_file=mapping_file,
            mapping_sha256=mapping_sha256,
            noise_seed=NOISE_SEED,
            mapping_seed=MAPPING_SEED,
            validation_seed=VALIDATION_SEED,
            noise_rate=NOISE_RATE,
            expected_pool=spec.expected_pool,
            peer_validation_dirs=spec.peer_validation_dirs,
        )
        input_dir = experiment_root / "prepared_inputs" / spec.output_name
        aligned_alternative = align_noise_rows(
            train_paths,
            manifest_rows,
            source_name=str(manifest),
        )
        prepare_input_adapter(
            base_input_dir=spec.base_input_dir,
            adapter_dir=input_dir,
            alternative_manifest=manifest,
            aligned_alternative_rows=aligned_alternative,
            dataset=spec.display_name,
            experiment_tag=EXPERIMENT_TAG,
            mapping_file=mapping_file,
            mapping_sha256=mapping_sha256,
            mapping_seed=MAPPING_SEED,
            noise_seed=NOISE_SEED,
            validation_seed=VALIDATION_SEED,
        )
        item = PreparedDataset(
            spec=spec,
            mapping_file=mapping_file,
            mapping_sha256=mapping_sha256,
            noise_manifest=manifest,
            noise_manifest_sha256=str(manifest_metadata["manifest_sha256"]),
            input_dir=input_dir,
            protocol_dir=protocol_dir,
            validation_manifest=protocol_dir / "validation_manifest.csv",
            validation_manifest_sha256=str(audit["validation_manifest_sha256"]),
            audit=audit,
        )
        prepared[spec.key] = item
        audits.append(audit)
        print(
            f"  PASS {spec.display_name}: mapping={mapping_sha256}, flip_mismatch=0, "
            f"validation_mismatch=0, pool={spec.expected_pool[0]} "
            f"clean/noisy={spec.expected_pool[1]}/{spec.expected_pool[2]}",
            flush=True,
        )

    experiment_root.mkdir(parents=True, exist_ok=True)
    write_json(
        experiment_root / "preflight_audit.json",
        {
            "experiment_tag": EXPERIMENT_TAG,
            "status": "PASS",
            "noise_seed": NOISE_SEED,
            "mapping_seed": MAPPING_SEED,
            "validation_seed": VALIDATION_SEED,
            "training_seeds": list(TRAINING_SEEDS),
            "datasets": audits,
        },
    )
    write_csv(
        experiment_root / "preflight_audit.csv",
        audits,
        [
            "dataset", "status", "source_train_size", "flipped_count", "flip_mask_mismatch_count",
            "validation_count", "validation_manifest_sha256", "validation_manifest_mismatch_count",
            "training_pool", "training_pool_clean", "training_pool_noisy", "training_pool_clean_ratio",
            "num_classes", "num_source_classes", "num_target_classes", "mapping_no_self_loop",
            "mapping_bijective", "mapping_seed", "mapping_sha256",
        ],
    )
    return prepared


def build_run_plan(experiment_root: Path, mapping_hashes: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        for dataset in DATASETS:
            for seed in TRAINING_SEEDS:
                run_id = make_run_id(dataset.key, method.key, seed)
                dependency = make_run_id(dataset.key, "pgdf", seed) if method.key == "budget_matched_dynamic" else ""
                output_dir = (
                    experiment_root / dataset.output_name / PROTOCOL_DIR_NAME / method.runner_key / f"seed{seed}"
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "phase": method.phase,
                        "dataset": dataset.key,
                        "dataset_name": dataset.display_name,
                        "method": method.key,
                        "method_name": method.label,
                        "runner_method": method.runner_key,
                        "training_seed": seed,
                        "noise_seed": NOISE_SEED,
                        "mapping_seed": MAPPING_SEED,
                        "mapping_sha256": mapping_hashes[dataset.key],
                        "output_dir": str(output_dir),
                        "dependency_run_id": dependency,
                    }
                )
    validate_run_plan(rows)
    return rows


def validate_run_plan(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 45:
        raise ValueError(f"Master run plan must contain exactly 45 rows, found {len(rows)}.")
    run_ids = [str(row["run_id"]) for row in rows]
    if len(set(run_ids)) != 45:
        raise ValueError("Master run plan contains duplicate run IDs.")
    counts = {method.key: sum(row["method"] == method.key for row in rows) for method in METHODS}
    if any(value != 9 for value in counts.values()):
        raise ValueError(f"Each method must have exactly 9 runs: {counts}")
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset.key]
        hashes = {str(row["mapping_sha256"]) for row in dataset_rows}
        seeds = {int(row["training_seed"]) for row in dataset_rows}
        if len(hashes) != 1 or seeds != set(TRAINING_SEEDS):
            raise ValueError(
                f"{dataset.key}: mapping must be seed-invariant and cover exactly {TRAINING_SEEDS}."
            )
    index = {str(row["run_id"]): row for row in rows}
    for row in rows:
        if row["method"] != "budget_matched_dynamic":
            if row["dependency_run_id"]:
                raise ValueError(f"Unexpected dependency for {row['run_id']}.")
            continue
        dependency = index.get(str(row["dependency_run_id"]))
        if dependency is None or dependency["method"] != "pgdf":
            raise ValueError(f"BMD dependency is not a PGDF run: {row['run_id']}")
        exact = ("dataset", "training_seed", "mapping_sha256")
        if any(row[field] != dependency[field] for field in exact):
            raise ValueError(f"BMD dependency does not match dataset/seed/mapping: {row['run_id']}")


def filter_plan(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_keys = set(args.dataset or [spec.key for spec in DATASETS])
    method_keys = {
        METHOD_ALIASES[item] for item in args.method
    } if args.method else {method.key for method in METHODS}
    seeds = set(args.training_seed or TRAINING_SEEDS)
    return [
        row for row in rows
        if row["dataset"] in dataset_keys and row["method"] in method_keys and int(row["training_seed"]) in seeds
    ]


def dependency_rows(master_plan: list[dict[str, Any]], bmd_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["run_id"]): row for row in master_plan}
    dependencies = [by_id[str(row["dependency_run_id"])] for row in bmd_rows]
    unique = {str(row["run_id"]): row for row in dependencies}
    return list(unique.values())


def run_grouped_tasks(
    rows: list[dict[str, Any]],
    prepared: dict[str, PreparedDataset],
    experiment_root: Path,
    args: argparse.Namespace,
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["dataset"]), str(row["method"])), []).append(row)
    for key in sorted(groups, key=lambda item: (dataset_position(item[0]), method_position(item[1]))):
        group = sorted(groups[key], key=lambda row: int(row["training_seed"]))
        pending: list[dict[str, Any]] = []
        for row in group:
            complete, reason = is_run_complete(row, prepared[str(row["dataset"])])
            if not complete and raw_run_complete(row):
                # A prior process may have finished training just before the
                # launcher wrote experiment metadata. Recover that run without
                # repeating 30 epochs.
                enrich_run_metadata(row, prepared[str(row["dataset"])])
                complete, reason = is_run_complete(row, prepared[str(row["dataset"])])
            if complete and not args.force:
                print(f"SKIP COMPLETED {row['run_id']}", flush=True)
                write_state(experiment_root, row, "COMPLETED", "strict completion audit passed")
                continue
            run_dir = Path(str(row["output_dir"]))
            if run_dir.exists() and any(run_dir.iterdir()):
                archive_run_dir(run_dir, experiment_root, str(row["run_id"]), reason)
            pending.append(row)
        if not pending:
            continue
        run_existing_trainer(pending, prepared[str(pending[0]["dataset"])], experiment_root, args)


def run_existing_trainer(
    rows: list[dict[str, Any]],
    prepared: PreparedDataset,
    experiment_root: Path,
    args: argparse.Namespace,
) -> None:
    method_keys = {str(row["runner_method"]) for row in rows}
    if len(method_keys) != 1:
        raise ValueError("One existing-trainer invocation may contain only one method.")
    method_key = method_keys.pop()
    seeds = [int(row["training_seed"]) for row in rows]
    for row in rows:
        print(
            f"START dataset={row['dataset_name']} method={row['method_name']} seed={row['training_seed']}",
            flush=True,
        )
        write_state(experiment_root, row, "RUNNING", "existing trainer launched")

    command = [
        sys.executable,
        str(ROOT / "scripts/run_lora_checkpoint_validation.py"),
        "--input-dir",
        str(prepared.input_dir),
        "--noise-index",
        str(prepared.noise_manifest),
        "--output-dir",
        str(prepared.protocol_dir),
        "--methods",
        method_key,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--validation-ratio",
        "0.10",
        "--validation-seed",
        str(VALIDATION_SEED),
        "--dynamic-ratio",
        str(DYNAMIC_RATIO),
        "--fixed-p",
        str(FIXED_P),
        "--warmup-epochs",
        str(WARMUP_EPOCHS),
        "--update-interval",
        str(UPDATE_INTERVAL),
        "--official-test-selected-only",
        "--no-posthoc-oracle-test",
    ]
    if method_key == "dynamic_budget_matched":
        command.extend(["--pgdf-budget-root", str(prepared.protocol_dir / "pgdf_fixed")])
    for path_map in args.path_map:
        command.extend(["--path-map", path_map])
    print("COMMAND " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)

    finished: list[str] = []
    failed: list[str] = []
    for row in rows:
        if raw_run_complete(row):
            enrich_run_metadata(row, prepared)
            strict, reason = is_run_complete(row, prepared)
            if strict:
                write_state(experiment_root, row, "COMPLETED", "strict completion audit passed")
                print(f"END COMPLETED {row['run_id']}", flush=True)
                finished.append(str(row["run_id"]))
                continue
            failed.append(f"{row['run_id']}: {reason}")
        else:
            failed.append(f"{row['run_id']}: raw trainer artifacts incomplete")
        write_state(experiment_root, row, "FAILED", failed[-1])
        print(f"END FAILED {failed[-1]}", flush=True)
    if completed.returncode != 0 or failed:
        raise RuntimeError(
            f"Existing trainer failed with exit code {completed.returncode}; completed={finished}; failed={failed}. "
            "No later dependent task was started. Re-run this launcher to archive/retry partial runs."
        )


def raw_run_complete(row: dict[str, Any]) -> bool:
    run_dir = Path(str(row["output_dir"]))
    result_path = run_dir / "result.json"
    checkpoint = run_dir / "checkpoints/best_val.pt"
    train_log = run_dir / "train_log.csv"
    if not all(path.is_file() for path in (result_path, checkpoint, train_log)):
        return False
    try:
        result = read_json(result_path)
        epochs = [int(item["epoch"]) for item in read_csv(train_log)]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        result.get("method_key") == row["runner_method"]
        and int(result.get("seed", -1)) == int(row["training_seed"])
        and result.get("checkpoint_protocol") == PROTOCOL_NAME
        and result.get("official_test_evaluation") == "validation_selected_only"
        and numeric(result.get("validation_selected_test_top1")) is not None
        and numeric(result.get("validation_selected_test_top5")) is not None
        and len(epochs) == FORMAL_EPOCHS
        and max(epochs, default=-1) == FORMAL_EPOCHS
    )


def enrich_run_metadata(row: dict[str, Any], prepared: PreparedDataset) -> None:
    run_dir = Path(str(row["output_dir"]))
    result_path = run_dir / "result.json"
    result = read_json(result_path)
    metadata = expected_run_metadata(row, prepared)
    result.update(metadata)
    write_json(result_path, result)
    write_json(run_dir / "run_metadata.json", metadata)


def expected_run_metadata(row: dict[str, Any], prepared: PreparedDataset) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "experiment_tag": EXPERIMENT_TAG,
        "run_id": row["run_id"],
        "dataset": prepared.spec.display_name,
        "method_key": row["runner_method"],
        "method_name": row["method_name"],
        "noise_rate": NOISE_RATE,
        "noise_seed": NOISE_SEED,
        "noise_manifest": str(prepared.noise_manifest),
        "noise_manifest_sha256": prepared.noise_manifest_sha256,
        "mapping_type": MAPPING_TYPE,
        "mapping_seed": MAPPING_SEED,
        "mapping_file": str(prepared.mapping_file),
        "mapping_sha256": prepared.mapping_sha256,
        "validation_seed": VALIDATION_SEED,
        "validation_manifest": str(prepared.validation_manifest),
        "validation_manifest_sha256": prepared.validation_manifest_sha256,
        "training_seed": int(row["training_seed"]),
        "protocol": PROTOCOL_NAME,
        "checkpoint_selection": "held-out clean validation Top-1",
        "official_test_policy": "validation_selected_only",
        "flipped_sample_identity_policy": "exact formal cyclic-asym40 noise-seed-42 mask reuse",
        "training_epochs": FORMAL_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS if row["method"] in {
            "dynamic_small_loss", "pgdf", "budget_matched_dynamic"
        } else "",
        "selection_interval": UPDATE_INTERVAL if row["method"] in {
            "dynamic_small_loss", "pgdf", "budget_matched_dynamic"
        } else "",
        "r": DYNAMIC_RATIO if row["method"] in {
            "dynamic_small_loss", "pgdf", "budget_matched_dynamic"
        } else "",
        "p": FIXED_P if row["method"] in {"pgdf", "budget_matched_dynamic"} else "",
        "jal_alpha": 1.0 if row["method"] == "jal_ce" else "",
        "jal_beta": 1.0 if row["method"] == "jal_ce" else "",
        "jal_a": 30.0 if row["method"] == "jal_ce" else "",
        "jal_eps": 1.0e-8 if row["method"] == "jal_ce" else "",
    }
    if row["method"] == "budget_matched_dynamic":
        budget = prepared.protocol_dir / "pgdf_fixed" / f"seed{row['training_seed']}" / "selection_per_class.csv"
        metadata.update(
            {
                "pgdf_budget_source_run_id": row["dependency_run_id"],
                "pgdf_budget_source": str(budget),
                "pgdf_budget_source_sha256": file_sha256(budget) if budget.is_file() else "",
                "pgdf_budget_match_scope": "same dataset + same training seed + same mapping SHA-256 + class/update",
            }
        )
    return metadata


def is_run_complete(row: dict[str, Any], prepared: PreparedDataset) -> tuple[bool, str]:
    if not raw_run_complete(row):
        return False, "raw artifacts or validation-selected official-test result incomplete"
    run_dir = Path(str(row["output_dir"]))
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        return False, "run_metadata.json missing"
    result = read_json(run_dir / "result.json")
    metadata = read_json(metadata_path)
    expected = expected_run_metadata(row, prepared)
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value or result.get(key) != value]
    if mismatches:
        return False, f"mapping/protocol metadata mismatches: {mismatches}"
    if row["method"] == "pgdf":
        selection = run_dir / "selection_per_class.csv"
        if not selection.is_file():
            return False, "PGDF selection_per_class.csv missing"
        try:
            labels = np.load(prepared.input_dir / "labels.npy", allow_pickle=True).astype(str)
            manifest = read_csv(prepared.validation_manifest)
            candidate_mask = np.asarray(
                [entry["partition"] == "training_pool" for entry in manifest],
                dtype=bool,
            )
            schedule = load_pgdf_class_budget_schedule(
                selection,
                expected_seed=int(row["training_seed"]),
                expected_update_epochs=selection_update_epochs(
                    FORMAL_EPOCHS,
                    WARMUP_EPOCHS,
                    UPDATE_INTERVAL,
                ),
                labels=labels,
                candidate_mask=candidate_mask,
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            return False, f"PGDF class/update budget artifact is invalid: {exc}"
        if not math.isclose(schedule.source_retention_ratio, DYNAMIC_RATIO) or not math.isclose(
            schedule.source_proto_keep_ratio,
            FIXED_P,
        ):
            return False, "PGDF class/update budget artifact does not use frozen r=0.8, p=0.4"
    if row["method"] == "budget_matched_dynamic":
        budget = Path(str(expected["pgdf_budget_source"]))
        if not budget.is_file() or file_sha256(budget) != expected["pgdf_budget_source_sha256"]:
            return False, "BMD budget source is missing or changed"
        source_meta = prepared.protocol_dir / "pgdf_fixed" / f"seed{row['training_seed']}" / "run_metadata.json"
        if not source_meta.is_file():
            return False, "BMD PGDF dependency metadata missing"
        source = read_json(source_meta)
        if (
            source.get("run_id") != row["dependency_run_id"]
            or source.get("mapping_sha256") != prepared.mapping_sha256
            or int(source.get("training_seed", -1)) != int(row["training_seed"])
        ):
            return False, "BMD dependency is not the exact dataset/seed/mapping PGDF run"
        budget_metadata = run_dir / "budget_source.json"
        if not budget_metadata.is_file():
            return False, "BMD budget_source.json missing"
        payload = read_json(budget_metadata)
        if payload.get("source_sha256") != expected["pgdf_budget_source_sha256"]:
            return False, "BMD trainer recorded a different PGDF budget hash"
    return True, "strict completion audit passed"


def audit_pgdf_dependencies(
    rows: list[dict[str, Any]],
    prepared: dict[str, PreparedDataset],
    experiment_root: Path,
) -> None:
    audit_rows: list[dict[str, Any]] = []
    expected_updates = selection_update_epochs(FORMAL_EPOCHS, WARMUP_EPOCHS, UPDATE_INTERVAL)
    for row in rows:
        item = prepared[str(row["dataset"])]
        complete, reason = is_run_complete(row, item)
        if not complete:
            raise RuntimeError(f"PGDF dependency {row['run_id']} is incomplete: {reason}")
        labels = np.load(item.input_dir / "labels.npy", allow_pickle=True).astype(str)
        manifest = read_csv(item.validation_manifest)
        candidate_mask = np.asarray([entry["partition"] == "training_pool" for entry in manifest], dtype=bool)
        budget_path = Path(str(row["output_dir"])) / "selection_per_class.csv"
        schedule = load_pgdf_class_budget_schedule(
            budget_path,
            expected_seed=int(row["training_seed"]),
            expected_update_epochs=expected_updates,
            labels=labels,
            candidate_mask=candidate_mask,
        )
        if not math.isclose(schedule.source_retention_ratio, DYNAMIC_RATIO) or not math.isclose(
            schedule.source_proto_keep_ratio, FIXED_P
        ):
            raise ValueError(f"PGDF dependency {row['run_id']} does not use frozen r/p.")
        metadata = read_json(Path(str(row["output_dir"])) / "run_metadata.json")
        if metadata.get("mapping_sha256") != item.mapping_sha256:
            raise ValueError(f"PGDF dependency {row['run_id']} has a different mapping hash.")
        audit_rows.append(
            {
                "pgdf_run_id": row["run_id"],
                "dataset": row["dataset"],
                "training_seed": row["training_seed"],
                "mapping_sha256": item.mapping_sha256,
                "budget_file": str(budget_path),
                "budget_sha256": budget_file_sha256(budget_path),
                "update_epochs": ",".join(str(epoch) for epoch in sorted(schedule.budgets)),
                "class_update_rows": len(schedule.rows),
                "status": "PASS",
            }
        )
    write_csv(
        experiment_root / "pgdf_budget_audit.csv",
        audit_rows,
        [
            "pgdf_run_id", "dataset", "training_seed", "mapping_sha256", "budget_file",
            "budget_sha256", "update_epochs", "class_update_rows", "status",
        ],
    )


def archive_run_dir(run_dir: Path, experiment_root: Path, run_id: str, reason: str) -> None:
    resolved_run = run_dir.resolve()
    resolved_root = experiment_root.resolve()
    try:
        resolved_run.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to archive a run outside experiment root: {run_dir}") from exc
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = experiment_root / "attempt_archive" / run_id / timestamp
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(destination))
    write_json(
        destination / "archive_reason.json",
        {"run_id": run_id, "reason": reason, "archived_at_utc": timestamp, "recoverable": True},
    )
    print(f"ARCHIVE {run_id}: {run_dir} -> {destination} ({reason})", flush=True)


def collect_status(
    rows: list[dict[str, Any]],
    prepared: dict[str, PreparedDataset],
) -> list[dict[str, Any]]:
    status: list[dict[str, Any]] = []
    for row in rows:
        complete, reason = is_run_complete(row, prepared[str(row["dataset"])])
        status.append({**row, "status": "COMPLETED" if complete else "PENDING", "detail": reason})
    return status


def write_final_outputs(
    plan: list[dict[str, Any]],
    prepared: dict[str, PreparedDataset],
    experiment_root: Path,
) -> None:
    results: list[dict[str, Any]] = []
    for row in plan:
        complete, reason = is_run_complete(row, prepared[str(row["dataset"])])
        if not complete:
            raise RuntimeError(f"Cannot summarize incomplete run {row['run_id']}: {reason}")
        payload = read_json(Path(str(row["output_dir"])) / "result.json")
        results.append(
            {
                "dataset": row["dataset_name"],
                "method": row["method_name"],
                "training_seed": row["training_seed"],
                "validation_selected_epoch": int(payload["best_val_epoch"]),
                "validation_top1": float(payload["best_val_top1"]),
                "official_test_top1": float(payload["validation_selected_test_top1"]),
                "official_test_top5": float(payload["validation_selected_test_top5"]),
                "status": "COMPLETED",
                "mapping_sha256": row["mapping_sha256"],
            }
        )
    write_csv(
        experiment_root / "random_derangement_asym40_results.csv",
        results,
        [
            "dataset", "method", "training_seed", "validation_selected_epoch", "validation_top1",
            "official_test_top1", "official_test_top5", "status", "mapping_sha256",
        ],
    )
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            values = [
                row for row in results
                if row["dataset"] == dataset.display_name and row["method"] == method.label
            ]
            if len(values) != 3:
                raise RuntimeError(f"Expected three result seeds for {dataset.key}/{method.key}.")
            top1 = [float(row["official_test_top1"]) for row in values]
            top5 = [float(row["official_test_top5"]) for row in values]
            summaries.append(
                {
                    "method": method.label,
                    "dataset": dataset.display_name,
                    "top1_mean": statistics.mean(top1),
                    "top1_sample_std": statistics.stdev(top1),
                    "top5_mean": statistics.mean(top5),
                    "top5_sample_std": statistics.stdev(top5),
                }
            )
    write_csv(
        experiment_root / "random_derangement_asym40_summary.csv",
        summaries,
        ["method", "dataset", "top1_mean", "top1_sample_std", "top5_mean", "top5_sample_std"],
    )
    write_paired_effects(results, experiment_root)
    write_mapping_sensitivity(results, prepared, experiment_root)
    write_audit_compatible_dataset_results(plan, experiment_root)


def write_paired_effects(results: list[dict[str, Any]], experiment_root: Path) -> None:
    comparisons = (
        ("PGDF vs Budget-matched Dynamic", "Budget-matched Dynamic"),
        ("PGDF vs Dynamic small-loss", "Dynamic small-loss"),
        ("PGDF vs JAL-CE", "JAL-CE"),
    )
    keyed = {
        (str(row["dataset"]), str(row["method"]), int(row["training_seed"])): float(row["official_test_top1"])
        for row in results
    }
    rows: list[dict[str, Any]] = []
    for comparison, baseline in comparisons:
        for dataset in DATASETS:
            for seed in TRAINING_SEEDS:
                pgdf = keyed[(dataset.display_name, "PGDF", seed)]
                baseline_value = keyed[(dataset.display_name, baseline, seed)]
                rows.append(
                    {
                        "comparison": comparison,
                        "dataset": dataset.display_name,
                        "training_seed": seed,
                        "pgdf_top1": pgdf,
                        "baseline_top1": baseline_value,
                        "delta_pp": 100.0 * (pgdf - baseline_value),
                    }
                )
    write_csv(
        experiment_root / "paired_effects_random_derangement.csv",
        rows,
        ["comparison", "dataset", "training_seed", "pgdf_top1", "baseline_top1", "delta_pp"],
    )
    summaries: list[dict[str, Any]] = []
    for comparison, _ in comparisons:
        comparison_rows = [row for row in rows if row["comparison"] == comparison]
        for dataset in DATASETS:
            values = [
                float(row["delta_pp"]) for row in comparison_rows if row["dataset"] == dataset.display_name
            ]
            summaries.append(paired_summary_row(comparison, dataset.display_name, "dataset", values))
        summaries.append(
            paired_summary_row(
                comparison,
                "ALL_3_DATASETS",
                "pooled_9_pairs",
                [float(row["delta_pp"]) for row in comparison_rows],
            )
        )
    write_csv(
        experiment_root / "paired_effects_random_derangement_summary.csv",
        summaries,
        [
            "comparison", "summary_scope", "dataset", "pairs", "mean_paired_delta_pp",
            "sample_std_paired_delta_pp", "median_paired_delta_pp", "min_paired_delta_pp",
            "max_paired_delta_pp", "pgdf_higher_count", "baseline_higher_count", "tie_count",
        ],
    )


def paired_summary_row(comparison: str, dataset: str, scope: str, values: list[float]) -> dict[str, Any]:
    tolerance = 1.0e-12
    return {
        "comparison": comparison,
        "summary_scope": scope,
        "dataset": dataset,
        "pairs": len(values),
        "mean_paired_delta_pp": statistics.mean(values),
        "sample_std_paired_delta_pp": statistics.stdev(values),
        "median_paired_delta_pp": statistics.median(values),
        "min_paired_delta_pp": min(values),
        "max_paired_delta_pp": max(values),
        "pgdf_higher_count": sum(value > tolerance for value in values),
        "baseline_higher_count": sum(value < -tolerance for value in values),
        "tie_count": sum(abs(value) <= tolerance for value in values),
    }


def write_mapping_sensitivity(
    results: list[dict[str, Any]],
    prepared: dict[str, PreparedDataset],
    experiment_root: Path,
) -> None:
    new = {
        (str(row["dataset"]), str(row["method"]), int(row["training_seed"])): float(row["official_test_top1"])
        for row in results
    }
    old_locations = {
        "LoRA all-noisy CE": ("checkpoint_validation_missing9_s20250726", "all_noisy"),
        "Dynamic small-loss": ("checkpoint_validation_s20250726", "dynamic"),
        "JAL-CE": ("checkpoint_validation_s20250726", "jal_ce"),
        "PGDF": ("checkpoint_validation_s20250726", "pgdf_fixed"),
    }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in DATASETS:
        item = prepared[spec.key]
        for method, (protocol_dir, old_method) in old_locations.items():
            validation = spec.base_input_dir / protocol_dir / "validation_manifest.csv"
            validation_json = validation.with_suffix(".json")
            try:
                comparison = compare_validation_manifests(
                    item.validation_manifest,
                    item.validation_manifest.with_suffix(".json"),
                    validation,
                    validation_json,
                )
            except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
                missing.append(f"{spec.key}/{method}: validation manifest is unavailable: {exc}")
                continue
            if not comparison["semantic_equal"]:
                missing.append(f"{spec.key}/{method}: validation manifest is not reliably identical")
                continue
            for seed in TRAINING_SEEDS:
                source = spec.base_input_dir / protocol_dir / old_method / f"seed{seed}" / "result.json"
                if not source.is_file():
                    missing.append(f"{spec.key}/{method}/seed{seed}: {source} missing")
                    continue
                payload = read_json(source)
                value = numeric(payload.get("validation_selected_test_top1"))
                if (
                    payload.get("checkpoint_protocol") != PROTOCOL_NAME
                    or payload.get("method_key") != old_method
                    or int(payload.get("seed", -1)) != seed
                    or value is None
                ):
                    missing.append(f"{spec.key}/{method}/seed{seed}: old formal result is not reliable")
                    continue
                random_value = new[(spec.display_name, method, seed)]
                rows.append(
                    {
                        "dataset": spec.display_name,
                        "method": method,
                        "training_seed": seed,
                        "random_derangement_top1": random_value,
                        "original_cyclic_top1": value,
                        "delta_mapping_pp": 100.0 * (random_value - value),
                        "random_mapping_sha256": item.mapping_sha256,
                        "original_result": str(source),
                    }
                )
    if missing:
        write_json(experiment_root / "mapping_sensitivity_unavailable.json", {"status": "NOT_GENERATED", "reasons": missing})
        print("Mapping-sensitivity comparison was not generated because old formal result locations were unreliable.", flush=True)
        return
    write_csv(
        experiment_root / "mapping_sensitivity_paired.csv",
        rows,
        [
            "dataset", "method", "training_seed", "random_derangement_top1", "original_cyclic_top1",
            "delta_mapping_pp", "random_mapping_sha256", "original_result",
        ],
    )


def write_audit_compatible_dataset_results(
    plan: list[dict[str, Any]],
    experiment_root: Path,
) -> None:
    for spec in DATASETS:
        dataset_rows = [row for row in plan if row["dataset"] == spec.key]
        payloads = [read_json(Path(str(row["output_dir"])) / "result.json") for row in dataset_rows]
        fields = ordered_union(payloads, preferred=("method_key", "method", "seed"))
        write_csv(
            experiment_root / spec.output_name / PROTOCOL_DIR_NAME / "checkpoint_validation_results.csv",
            payloads,
            fields,
        )


def print_mechanism_audit_command(experiment_root: Path) -> None:
    audit_output = experiment_root / "mechanism_audit"
    command = [
        sys.executable,
        str(ROOT / "scripts/audit_asym40_revision.py"),
        "--experiment-root",
        str(experiment_root),
        "--mapping-type",
        MAPPING_TYPE,
        "--output-dir",
        str(audit_output),
    ]
    print("Offline mechanism audit (does not affect training):", flush=True)
    print(subprocess.list2cmdline(command), flush=True)


def print_plan_summary(master: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    counts = {method.key: sum(row["method"] == method.key for row in master) for method in METHODS}
    print(f"Run plan PASS: master={len(master)} unique runs, selected={len(selected)}, per_method={counts}", flush=True)


def print_dry_run(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows, start=1):
        dependency = f" dependency={row['dependency_run_id']}" if row["dependency_run_id"] else ""
        print(
            f"PLAN {index:02d}/{len(rows):02d} phase={row['phase']} run_id={row['run_id']} "
            f"dataset={row['dataset']} method={row['method']} seed={row['training_seed']}{dependency}",
            flush=True,
        )


def write_run_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(
        path,
        rows,
        [
            "run_id", "phase", "dataset", "dataset_name", "method", "method_name", "runner_method",
            "training_seed", "noise_seed", "mapping_seed", "mapping_sha256", "output_dir", "dependency_run_id",
        ],
    )


def write_state(
    experiment_root: Path,
    row: dict[str, Any],
    status: str,
    detail: str,
) -> None:
    path = experiment_root / "run_states" / f"{row['run_id']}.json"
    write_json(
        path,
        {
            "run_id": row["run_id"],
            "dataset": row["dataset"],
            "method": row["method"],
            "training_seed": row["training_seed"],
            "status": status,
            "detail": detail,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def make_run_id(dataset: str, method: str, seed: int) -> str:
    return f"{dataset}__{method}__seed{seed}"


def dataset_position(key: str) -> int:
    return next(index for index, spec in enumerate(DATASETS) if spec.key == key)


def method_position(key: str) -> int:
    return next(index for index, spec in enumerate(METHODS) if spec.key == key)


def normalize_repo_path(path: Path) -> Path:
    if path.is_absolute():
        try:
            return path.resolve().relative_to(ROOT.resolve())
        except ValueError:
            return path.resolve()
    return path


def numeric(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def ordered_union(rows: list[dict[str, Any]], preferred: tuple[str, ...]) -> list[str]:
    fields = list(preferred)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def status_fields() -> list[str]:
    return [
        "run_id", "phase", "dataset", "dataset_name", "method", "method_name", "runner_method",
        "training_seed", "noise_seed", "mapping_seed", "mapping_sha256", "output_dir",
        "dependency_run_id", "status", "detail",
    ]


if __name__ == "__main__":
    main()
