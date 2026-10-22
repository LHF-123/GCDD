"""Run R4-C2 fixed-mask replay and oracle clean replacement intervention.

The only source of selection membership is the nine instrumented Dynamic PGDF
runs.  R replays each saved mask.  O swaps deterministically chosen oracle
clean A members for B members within the same observed-label class and update.
No current-model loss/prototype score, validation result, or test result enters
either mask.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

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
from gcdd.clean_replacement import (
    REPLACEMENT_RATE,
    REPLACEMENT_RULE_ID,
    ReplacementTrajectory,
    build_oracle_clean_replacement,
    file_sha256,
)
from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json, write_yaml
from gcdd.lora_dynamic import selection_update_epochs, train_dynamic_loss_lora
from gcdd.lora_training import resolve_image_path


VALIDATION_SEED = 20250726
VALIDATION_RATIO = 0.10
TRAINING_SEEDS = (1, 42, 88)
UPDATE_EPOCHS = (5, 10, 15, 20, 25)
METHOD_GROUPS = ("R_fixed_mask_replay", "O_oracle_clean_replacement")

DATASETS: dict[str, dict[str, str]] = {
    "cub": {
        "display": "CUB-200-2011",
        "dataset_id": "CUB-200-2011-asym40-s42",
        "input_rel": "outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42",
        "noise_rel": "outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv",
        "instrument_rel": "outputs_major_revision/pgdf_dynamic_ref_cub40_instrumentation",
        "source_config_rel": "checkpoint_validation_pgdf_dynamic_proto_r08_p04_s20250726/resolved_config.yaml",
        "source_manifest_rel": "checkpoint_validation_s20250726/validation_manifest.json",
        "experiment_prefix": "CUB-200-2011-asym40-s42_adjacent_cyclic_noise42_pgdf_dynamic_proto_seed",
    },
    "cars": {
        "display": "Stanford Cars",
        "dataset_id": "Stanford-Cars-asym40-s42",
        "input_rel": "outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42",
        "noise_rel": "outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv",
        "instrument_rel": "outputs_major_revision/pgdf_dynamic_ref_cars40_instrumentation",
        "source_config_rel": "checkpoint_validation_pgdf_dynamic_proto_r08_p04_s20250726/resolved_config.yaml",
        "source_manifest_rel": "checkpoint_validation_s20250726/validation_manifest.json",
        "experiment_prefix": "Stanford-Cars-asym40-s42_adjacent_cyclic_noise42_pgdf_dynamic_proto_seed",
    },
    "aircraft": {
        "display": "FGVC-Aircraft",
        "dataset_id": "FGVC-Aircraft-asym40-s42",
        "input_rel": "outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42",
        "noise_rel": "outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv",
        "instrument_rel": "outputs_major_revision/pgdf_dynamic_ref_aircraft40_instrumentation",
        "source_config_rel": "checkpoint_validation_pgdf_dynamic_proto_r08_p04_s20250726/resolved_config.yaml",
        "source_manifest_rel": "checkpoint_validation_s20250726/validation_manifest.json",
        "experiment_prefix": "FGVC-Aircraft-asym40-s42_adjacent_cyclic_noise42_pgdf_dynamic_proto_seed",
    },
}

MEMBERSHIP_FIELDS = [
    "dataset", "training_seed", "selection_epoch", "stable_sample_id", "train_index", "sample_path",
    "observed_label", "clean_label_oracle", "is_clean_oracle", "loss_candidate", "prototype_candidate",
    "source_group", "r_active", "o_active", "intervention_role", "replacement_rule",
]
CLASSWISE_FIELDS = [
    "dataset", "training_seed", "selection_epoch", "observed_label", "source_total_count",
    "source_final_selected_count", "r_active_count", "o_active_count", "r_clean_active_count",
    "o_clean_active_count", "r_noisy_active_count", "o_noisy_active_count", "a_clean_l_and_p_count",
    "b_clean_l_not_p_count", "replacement_count", "replacement_rate_target", "replacement_rate_realized_of_a",
    "r_o_active_count_match", "r_o_clean_count_match", "r_o_noisy_count_match", "unmodified_members_match", "status",
]
INVENTORY_FIELDS = [
    "dataset", "dataset_key", "training_seed", "group", "status", "reason", "run_dir", "result_path",
    "source_instrumented_run", "source_selection_samples_sha256", "validation_manifest_sha256",
    "resolved_config_sha256", "image_preflight_status", "optimizer_steps_actual", "initial_trainable_checksum",
]
SEED_RESULT_FIELDS = [
    "dataset", "dataset_key", "training_seed", "r_status", "o_status", "r_official_test_top1_percent",
    "o_official_test_top1_percent", "o_minus_r_pp", "r_best_val_epoch", "o_best_val_epoch",
    "r_optimizer_steps_actual", "o_optimizer_steps_actual", "initialization_checksum_match", "protocol_match", "pair_status",
]
PAIRED_SUMMARY_FIELDS = [
    "dataset", "dataset_key", "completed_pairs", "r_mean_official_test_top1_percent",
    "r_sample_sd_official_test_top1_percent", "o_mean_official_test_top1_percent",
    "o_sample_sd_official_test_top1_percent", "o_minus_r_mean_pp", "o_minus_r_sample_sd_pp",
    "wins", "ties", "losses", "summary_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R4-C2 frozen-mask replay / oracle clean replacement experiment.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs_major_revision/r4_c2_clean_replacement")
    parser.add_argument("--shared-root", type=Path, default=default_shared_root())
    parser.add_argument("--datasets", default="cub,cars,aircraft")
    parser.add_argument("--seeds", default="1,42,88")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--execute", action="store_true", help="Start the 18 formal runs only after all data/GPU preflights pass.")
    parser.add_argument("--resume", action="store_true", help="Allow an existing R4-C2 output directory and skip completed runs.")
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    return parser.parse_args()


def default_shared_root() -> Path:
    candidate = ROOT.parent.parent
    return candidate if (candidate / "outputs").is_dir() else ROOT


def main() -> None:
    args = parse_args()
    dataset_keys = parse_dataset_keys(args.datasets)
    seeds = parse_seeds(args.seeds)
    output_dir = Path(args.output_dir).resolve()
    shared_root = Path(args.shared_root).resolve()
    prepare_output_directory(output_dir, resume=bool(args.resume))
    path_maps = default_path_maps(shared_root) + parse_path_maps(args.path_map)

    preparations: dict[tuple[str, int], dict[str, Any]] = {}
    membership_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    provenance_assets: list[dict[str, Any]] = []
    image_preflight_rows: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        dataset = DATASETS[dataset_key]
        prepared = prepare_dataset_contract(
            dataset_key=dataset_key,
            dataset=dataset,
            shared_root=shared_root,
            output_dir=output_dir,
        )
        image_check = preflight_images(dataset_key, dataset["display"], prepared["data"], path_maps)
        prepared["image_check"] = image_check
        image_preflight_rows.extend(image_check["rows"])
        provenance_assets.extend(prepared["assets"])
        for seed in seeds:
            trajectory = build_trajectory(prepared, dataset_key, dataset, seed)
            preparations[(dataset_key, seed)] = {**prepared, "trajectory": trajectory}
            membership_rows.extend(trajectory.membership_rows)
            classwise_rows.extend(trajectory.classwise_rows)
            provenance_assets.extend(source_selection_assets(prepared, dataset, seed, trajectory))

    validate_all_classwise_rows(classwise_rows, dataset_keys, seeds)
    write_csv(output_dir / "replacement_membership.csv", membership_rows, MEMBERSHIP_FIELDS)
    write_csv(output_dir / "classwise_budget_validation.csv", classwise_rows, CLASSWISE_FIELDS)
    write_csv(output_dir / "image_access_preflight.csv", image_preflight_rows, image_preflight_fields())

    gpu_ready, gpu_detail = gpu_preflight()
    all_images_ready = all(bool(preparations[(key, seed)]["image_check"]["ready"]) for key in dataset_keys for seed in seeds)
    inventory = build_inventory(
        dataset_keys=dataset_keys,
        seeds=seeds,
        preparations=preparations,
        output_dir=output_dir,
        requested_execute=bool(args.execute),
        gpu_ready=gpu_ready,
        gpu_detail=gpu_detail,
        images_ready=all_images_ready,
    )
    write_csv(output_dir / "training_run_inventory.csv", inventory, INVENTORY_FIELDS)

    if args.execute and gpu_ready and all_images_ready:
        for dataset_key in dataset_keys:
            for seed in seeds:
                prepared = preparations[(dataset_key, seed)]
                for group in METHOD_GROUPS:
                    run_one_formal_training(
                        output_dir=output_dir,
                        prepared=prepared,
                        dataset_key=dataset_key,
                        group=group,
                        seed=seed,
                        path_maps=path_maps,
                        device=args.device,
                        resume=bool(args.resume),
                    )
        inventory = refresh_inventory_from_runs(inventory)
        write_csv(output_dir / "training_run_inventory.csv", inventory, INVENTORY_FIELDS)

    seed_results, paired_summary = collect_results(output_dir, dataset_keys, seeds)
    write_csv(output_dir / "seed_level_official_test.csv", seed_results, SEED_RESULT_FIELDS)
    write_csv(output_dir / "paired_accuracy_summary.csv", paired_summary, PAIRED_SUMMARY_FIELDS)
    completion_state = completion_status(inventory, seed_results, dataset_keys, seeds)
    provenance = build_provenance(
        output_dir=output_dir,
        shared_root=shared_root,
        dataset_keys=dataset_keys,
        seeds=seeds,
        assets=provenance_assets,
        path_maps=path_maps,
        gpu_ready=gpu_ready,
        gpu_detail=gpu_detail,
        images_ready=all_images_ready,
        requested_execute=bool(args.execute),
        completion_state=completion_state,
    )
    write_json(output_dir / "analysis_provenance.json", provenance)
    write_report(
        output_dir=output_dir,
        completion_state=completion_state,
        inventory=inventory,
        classwise_rows=classwise_rows,
        seed_results=seed_results,
        paired_summary=paired_summary,
        gpu_detail=gpu_detail,
        all_images_ready=all_images_ready,
    )
    print(f"R4-C2 {completion_state}: artifacts written to {output_dir}", flush=True)


def parse_dataset_keys(value: str) -> tuple[str, ...]:
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not items or len(set(items)) != len(items) or any(item not in DATASETS for item in items):
        raise ValueError(f"--datasets must be a non-repeating subset of {sorted(DATASETS)}.")
    return items


def parse_seeds(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated integers.") from exc
    if not items or len(set(items)) != len(items) or any(item not in TRAINING_SEEDS for item in items):
        raise ValueError(f"--seeds must be a non-repeating subset of {TRAINING_SEEDS}.")
    return items


def parse_path_maps(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("Each --path-map must use OLD=NEW.")
        old, new = value.split("=", 1)
        if not old.strip() or not new.strip():
            raise ValueError("Each --path-map must contain non-empty OLD and NEW values.")
        parsed.append((old.strip(), new.strip()))
    return parsed


def default_path_maps(shared_root: Path) -> list[tuple[str, str]]:
    cub = shared_root / "dataset/CUB_200_2011/CUB_200_2011"
    return [("/root/autodl-tmp/CUB_200_2011", str(cub))] if (cub / "images").is_dir() else []


def prepare_output_directory(output_dir: Path, *, resume: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite existing R4-C2 output directory: {output_dir}. Use --resume after review.")
    ensure_dir(output_dir)


def prepare_dataset_contract(
    *,
    dataset_key: str,
    dataset: dict[str, str],
    shared_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    input_dir = shared_root / dataset["input_rel"]
    noise_index = shared_root / dataset["noise_rel"]
    source_config = input_dir / dataset["source_config_rel"]
    source_manifest = input_dir / dataset["source_manifest_rel"]
    for path in (input_dir, noise_index, source_config, source_manifest):
        if not path.exists():
            raise FileNotFoundError(f"Required R4-C2 input is missing: {path}")
    data = load_v1_data(input_dir)
    with source_config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    validate_protocol_config(cfg, dataset["dataset_id"])
    clean_labels = load_clean_labels_from_noise_index(noise_index, data["train_paths"])
    contract_dir = output_dir / "dataset_contracts" / dataset_key
    ensure_dir(contract_dir)
    split = build_or_load_fixed_validation_split(
        contract_dir,
        data["train_paths"],
        clean_labels,
        validation_ratio=VALIDATION_RATIO,
        validation_seed=VALIDATION_SEED,
    )
    assert_manifest_matches_source(contract_dir / "validation_manifest.json", source_manifest)
    write_yaml(contract_dir / "resolved_config.yaml", cfg)
    assets = [
        asset_record(f"{dataset_key}_input_paths", input_dir / "paths.txt"),
        asset_record(f"{dataset_key}_observed_labels", input_dir / "labels.npy"),
        asset_record(f"{dataset_key}_official_test_paths", input_dir / "eval_paths.txt"),
        asset_record(f"{dataset_key}_official_test_labels", input_dir / "eval_labels.npy"),
        asset_record(f"{dataset_key}_noise_index", noise_index),
        asset_record(f"{dataset_key}_source_resolved_config", source_config),
        asset_record(f"{dataset_key}_source_validation_manifest", source_manifest),
        asset_record(f"{dataset_key}_r4c2_validation_manifest", contract_dir / "validation_manifest.json"),
    ]
    return {
        "dataset_key": dataset_key,
        "dataset": dataset,
        "input_dir": input_dir,
        "noise_index": noise_index,
        "source_config": source_config,
        "source_manifest": source_manifest,
        "data": data,
        "cfg": cfg,
        "clean_labels": clean_labels,
        "split": split,
        "assets": assets,
        "contract_dir": contract_dir,
    }


def load_v1_data(input_dir: Path) -> dict[str, Any]:
    required = ("paths.txt", "labels.npy", "eval_paths.txt", "eval_labels.npy")
    missing = [str(input_dir / name) for name in required if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"V1 input is incomplete: {missing}")
    train_paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    test_paths = (input_dir / "eval_paths.txt").read_text(encoding="utf-8").splitlines()
    train_labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    test_labels = np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str)
    if len(train_paths) != len(train_labels) or len(test_paths) != len(test_labels):
        raise ValueError(f"V1 paths/labels do not align in {input_dir}.")
    return {"train_paths": train_paths, "train_labels": train_labels, "test_paths": test_paths, "test_labels": test_labels}


def validate_protocol_config(cfg: dict[str, Any], expected_dataset_id: str) -> None:
    if cfg.get("dataset", {}).get("name") != expected_dataset_id:
        raise ValueError("Resolved config dataset does not match the R4-C2 source mapping.")
    train = cfg.get("lora_train", {})
    required = {"epochs": 30, "batch_size": 80, "eval_batch_size": 160, "scheduler": "cosine", "warmup_ratio": 0.1}
    for field, expected in required.items():
        if train.get(field) != expected:
            raise ValueError(f"R4-C2 source config mismatch: lora_train.{field}={train.get(field)!r}, expected {expected!r}.")
    check = cfg.get("checkpoint_validation", {})
    expected_check = {"validation_seed": VALIDATION_SEED, "dynamic_ratio": 0.8, "fixed_p": 0.4, "warmup_epochs": 5, "update_interval": 5, "official_test_selected_only": True}
    for field, expected in expected_check.items():
        if check.get(field) != expected:
            raise ValueError(f"R4-C2 source config mismatch: checkpoint_validation.{field}.")


def assert_manifest_matches_source(generated_path: Path, source_path: Path) -> None:
    with generated_path.open("r", encoding="utf-8") as handle:
        generated = json.load(handle)
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    if generated != source:
        raise ValueError("New R4-C2 validation manifest differs from the historical fixed clean validation manifest.")


def preflight_images(
    dataset_key: str,
    dataset_display: str,
    data: dict[str, Any],
    path_maps: list[tuple[str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing = 0
    for split_name, paths in (("train", data["train_paths"]), ("official_test", data["test_paths"])):
        for index, raw_path in enumerate(paths):
            try:
                resolved = resolve_image_path(raw_path, path_maps)
                if not resolved.is_file():
                    raise FileNotFoundError(str(resolved))
            except FileNotFoundError:
                missing += 1
                if len([row for row in rows if row["status"] == "MISSING"]) < 25:
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "dataset": dataset_display,
                            "split": split_name,
                            "index": index,
                            "path": raw_path,
                            "status": "MISSING",
                            "resolved_path": "",
                        }
                    )
    if missing == 0:
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": dataset_display,
                "split": "all",
                "index": "",
                "path": "",
                "status": "PASS",
                "resolved_path": "all paths accessible",
            }
        )
    return {"ready": missing == 0, "missing_count": missing, "rows": rows}


def image_preflight_fields() -> list[str]:
    return ["dataset_key", "dataset", "split", "index", "path", "status", "resolved_path"]


def build_trajectory(prepared: dict[str, Any], dataset_key: str, dataset: dict[str, str], seed: int) -> ReplacementTrajectory:
    instrument_root = ROOT / dataset["instrument_rel"]
    experiment = f"{dataset['experiment_prefix']}{seed}"
    run_dir = instrument_root / experiment
    records = run_dir / "selection_records"
    expected_epochs = selection_update_epochs(
        int(prepared["cfg"]["lora_train"]["epochs"]),
        int(prepared["cfg"]["checkpoint_validation"]["warmup_epochs"]),
        int(prepared["cfg"]["checkpoint_validation"]["update_interval"]),
    )
    if tuple(expected_epochs) != UPDATE_EPOCHS:
        raise ValueError(f"R4-C2 expected selection epochs {UPDATE_EPOCHS}, found {expected_epochs}.")
    return build_oracle_clean_replacement(
        dataset=dataset["display"],
        training_seed=seed,
        train_paths=prepared["data"]["train_paths"],
        observed_labels=prepared["data"]["train_labels"],
        clean_labels=prepared["clean_labels"],
        training_pool_mask=prepared["split"].train_mask,
        selection_samples_path=records / "selection_samples.csv",
        selection_per_class_path=records / "selection_per_class.csv",
        selection_updates_path=records / "selection_updates.csv",
        expected_update_epochs=expected_epochs,
        replacement_rate=REPLACEMENT_RATE,
    )


def source_selection_assets(prepared: dict[str, Any], dataset: dict[str, str], seed: int, trajectory: ReplacementTrajectory) -> list[dict[str, Any]]:
    run_dir = ROOT / dataset["instrument_rel"] / f"{dataset['experiment_prefix']}{seed}"
    return [
        asset_record(f"{prepared['dataset_key']}_seed{seed}_instrumented_metadata", run_dir / "run_metadata.json"),
        asset_record(f"{prepared['dataset_key']}_seed{seed}_selection_samples", run_dir / "selection_records/selection_samples.csv"),
        asset_record(f"{prepared['dataset_key']}_seed{seed}_selection_per_class", run_dir / "selection_records/selection_per_class.csv"),
        asset_record(f"{prepared['dataset_key']}_seed{seed}_selection_updates", run_dir / "selection_records/selection_updates.csv"),
        {"role": f"{prepared['dataset_key']}_seed{seed}_trajectory", "sha256": trajectory.summary["selection_samples_sha256"], "detail": trajectory.summary},
    ]


def validate_all_classwise_rows(rows: list[dict[str, Any]], dataset_keys: tuple[str, ...], seeds: tuple[int, ...]) -> None:
    expected = sum(len(DATASETS[key]["display"]) >= 0 for key in dataset_keys) * len(seeds) * len(UPDATE_EPOCHS)
    observed = {(row["dataset"], int(row["training_seed"]), int(row["selection_epoch"]), row["observed_label"]) for row in rows}
    if not rows or len(observed) != len(rows):
        raise ValueError("Class-wise validation contains no rows or duplicate dataset/seed/update/class records.")
    if any(row["status"] != "PASS" for row in rows):
        raise ValueError("Class-wise validation contains a non-PASS row.")
    del expected  # Class cardinalities differ; uniqueness and source coverage are validated per trajectory.


def gpu_preflight() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return False, f"PyTorch unavailable: {exc}"
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false"
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError as exc:  # pragma: no cover - environment diagnostic
        output = f"nvidia-smi unavailable: {exc}"
    return True, output or "CUDA available; nvidia-smi details unavailable"


def build_inventory(
    *,
    dataset_keys: tuple[str, ...],
    seeds: tuple[int, ...],
    preparations: dict[tuple[str, int], dict[str, Any]],
    output_dir: Path,
    requested_execute: bool,
    gpu_ready: bool,
    gpu_detail: str,
    images_ready: bool,
) -> list[dict[str, Any]]:
    del gpu_detail
    rows: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        for seed in seeds:
            prepared = preparations[(dataset_key, seed)]
            trajectory: ReplacementTrajectory = prepared["trajectory"]
            run_source = ROOT / prepared["dataset"]["instrument_rel"] / f"{prepared['dataset']['experiment_prefix']}{seed}"
            if not prepared["image_check"]["ready"]:
                status, reason = "READY_WAIT_DATA", f"{prepared['image_check']['missing_count']} image path(s) inaccessible for this dataset"
            elif not images_ready:
                status, reason = "READY_WAIT_DATA", "this dataset's paths pass; the fixed 18-run matrix waits for another required dataset's images"
            elif not gpu_ready:
                status, reason = "READY_WAIT_GPU", "GPU preflight did not pass"
            elif requested_execute:
                status, reason = "PENDING_EXECUTION", "preflight passed; execution loop will update this row"
            else:
                status, reason = "READY", "CPU contracts and masks validated; --execute not requested"
            for group in METHOD_GROUPS:
                run_dir = output_dir / "runs" / dataset_key / f"seed{seed}" / group
                rows.append(
                    {
                        "dataset": prepared["dataset"]["display"],
                        "dataset_key": dataset_key,
                        "training_seed": seed,
                        "group": group,
                        "status": status,
                        "reason": reason,
                        "run_dir": str(run_dir),
                        "result_path": str(run_dir / "result.json"),
                        "source_instrumented_run": str(run_source),
                        "source_selection_samples_sha256": trajectory.summary["selection_samples_sha256"],
                        "validation_manifest_sha256": file_sha256(prepared["contract_dir"] / "validation_manifest.json"),
                        "resolved_config_sha256": file_sha256(prepared["contract_dir"] / "resolved_config.yaml"),
                        "image_preflight_status": "PASS" if prepared["image_check"]["ready"] else "MISSING",
                        "optimizer_steps_actual": "",
                        "initial_trainable_checksum": "",
                    }
                )
    return rows


def run_one_formal_training(
    *,
    output_dir: Path,
    prepared: dict[str, Any],
    dataset_key: str,
    group: str,
    seed: int,
    path_maps: list[tuple[str, str]],
    device: str,
    resume: bool,
) -> None:
    run_dir = output_dir / "runs" / dataset_key / f"seed{seed}" / group
    result_path = run_dir / "result.json"
    if result_path.is_file():
        if resume:
            return
        raise FileExistsError(f"Refusing to overwrite completed R4-C2 run: {run_dir}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix a new R4-C2 run with partial output: {run_dir}")
    ensure_dir(run_dir)
    trajectory: ReplacementTrajectory = prepared["trajectory"]
    cfg = copy.deepcopy(prepared["cfg"])
    cfg["feature"]["device"] = device
    cfg["lora_train"]["seed"] = int(seed)
    masks = trajectory.replay_masks if group == "R_fixed_mask_replay" else trajectory.oracle_masks
    result = train_dynamic_loss_lora(
        prepared["data"]["train_paths"],
        prepared["data"]["train_labels"],
        [prepared["data"]["train_paths"][int(index)] for index in np.flatnonzero(prepared["split"].validation_mask)],
        prepared["clean_labels"][prepared["split"].validation_mask],
        prepared["split"].train_mask,
        cfg,
        method=group,
        seed=seed,
        retention_ratio=float(cfg["checkpoint_validation"]["dynamic_ratio"]),
        warmup_epochs=int(cfg["checkpoint_validation"]["warmup_epochs"]),
        update_interval=int(cfg["checkpoint_validation"]["update_interval"]),
        path_maps=path_maps,
        proto_keep_ratio=float(cfg["checkpoint_validation"]["fixed_p"]),
        checkpoint_path=run_dir / "checkpoints/best_val.pt",
        test_paths=prepared["data"]["test_paths"],
        test_labels=prepared["data"]["test_labels"],
        final_checkpoint_path=run_dir / "checkpoints/last.pt",
        last5_checkpoint_dir=run_dir / "checkpoints/last5",
        checkpoint_protocol=PROTOCOL_NAME,
        posthoc_oracle_test=False,
        replay_selection_masks=masks,
        scheduler_retention_ratio=float(cfg["checkpoint_validation"]["fixed_p"]),
        official_test_selected_only=True,
        selection_strategy="fixed_replay",
        prototype_mode="fixed",
    )
    write_csv(run_dir / "train_log.csv", result.logs, list(result.logs[0]) if result.logs else [])
    write_csv(run_dir / "selection_rows.csv", result.selection_rows, list(result.selection_rows[0]) if result.selection_rows else [])
    write_csv(run_dir / "selection_updates.csv", result.update_rows, list(result.update_rows[0]) if result.update_rows else [])
    write_csv(run_dir / "selection_per_class.csv", result.per_class_rows, list(result.per_class_rows[0]) if result.per_class_rows else [])
    payload = {
        "dataset": prepared["dataset"]["display"],
        "dataset_key": dataset_key,
        "group": group,
        "training_seed": seed,
        "selection_protocol": "fixed_historical_mask_replay",
        "oracle_usage": "O replacement membership only; training targets remain observed noisy labels",
        "source_trajectory": trajectory.summary,
        "summary": result.summary,
    }
    write_json(result_path, payload)


def refresh_inventory_from_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refreshed: list[dict[str, Any]] = []
    for row in rows:
        result_path = Path(str(row["result_path"]))
        updated = dict(row)
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            summary = payload["summary"]
            updated.update(
                {
                    "status": "COMPLETE",
                    "reason": "30-epoch formal run completed",
                    "optimizer_steps_actual": summary.get("optimizer_steps_actual", ""),
                    "initial_trainable_checksum": summary.get("initial_trainable_checksum", ""),
                }
            )
        refreshed.append(updated)
    return refreshed


def collect_results(output_dir: Path, dataset_keys: tuple[str, ...], seeds: tuple[int, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        for seed in seeds:
            payloads = {
                group: read_result(output_dir / "runs" / dataset_key / f"seed{seed}" / group / "result.json")
                for group in METHOD_GROUPS
            }
            r_payload, o_payload = payloads[METHOD_GROUPS[0]], payloads[METHOD_GROUPS[1]]
            r_summary = r_payload.get("summary", {}) if r_payload else {}
            o_summary = o_payload.get("summary", {}) if o_payload else {}
            r_value = r_summary.get("validation_selected_test_top1", "")
            o_value = o_summary.get("validation_selected_test_top1", "")
            complete_pair = r_value != "" and o_value != ""
            r_steps, o_steps = r_summary.get("optimizer_steps_actual", ""), o_summary.get("optimizer_steps_actual", "")
            r_hash, o_hash = r_summary.get("initial_trainable_checksum", ""), o_summary.get("initial_trainable_checksum", "")
            protocol_match = bool(complete_pair and r_steps == o_steps and r_summary.get("scheduler_estimated_total_steps") == o_summary.get("scheduler_estimated_total_steps"))
            init_match = bool(complete_pair and r_hash and r_hash == o_hash)
            rows.append(
                {
                    "dataset": DATASETS[dataset_key]["display"],
                    "dataset_key": dataset_key,
                    "training_seed": seed,
                    "r_status": "COMPLETE" if r_payload else "NOT_RUN",
                    "o_status": "COMPLETE" if o_payload else "NOT_RUN",
                    "r_official_test_top1_percent": "" if r_value == "" else 100.0 * float(r_value),
                    "o_official_test_top1_percent": "" if o_value == "" else 100.0 * float(o_value),
                    "o_minus_r_pp": "" if not complete_pair else 100.0 * (float(o_value) - float(r_value)),
                    "r_best_val_epoch": r_summary.get("best_val_epoch", ""),
                    "o_best_val_epoch": o_summary.get("best_val_epoch", ""),
                    "r_optimizer_steps_actual": r_steps,
                    "o_optimizer_steps_actual": o_steps,
                    "initialization_checksum_match": init_match if complete_pair else "",
                    "protocol_match": protocol_match if complete_pair else "",
                    "pair_status": "COMPLETE_PASS" if complete_pair and init_match and protocol_match else "NOT_COMPLETE" if not complete_pair else "FAILED_VALIDATION",
                }
            )
    summary: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        subset = [row for row in rows if row["dataset_key"] == dataset_key and row["pair_status"] == "COMPLETE_PASS"]
        r_values = [float(row["r_official_test_top1_percent"]) for row in subset]
        o_values = [float(row["o_official_test_top1_percent"]) for row in subset]
        deltas = [float(row["o_minus_r_pp"]) for row in subset]
        wins = sum(delta > 0.0 for delta in deltas)
        ties = sum(math.isclose(delta, 0.0, abs_tol=1.0e-12) for delta in deltas)
        losses = sum(delta < 0.0 for delta in deltas)
        summary.append(
            {
                "dataset": DATASETS[dataset_key]["display"],
                "dataset_key": dataset_key,
                "completed_pairs": len(subset),
                "r_mean_official_test_top1_percent": mean(r_values) if r_values else "",
                "r_sample_sd_official_test_top1_percent": sample_sd(r_values),
                "o_mean_official_test_top1_percent": mean(o_values) if o_values else "",
                "o_sample_sd_official_test_top1_percent": sample_sd(o_values),
                "o_minus_r_mean_pp": mean(deltas) if deltas else "",
                "o_minus_r_sample_sd_pp": sample_sd(deltas),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "summary_status": "COMPLETE" if len(subset) == 3 else "PARTIAL" if subset else "NOT_RUN",
            }
        )
    return rows, summary


def read_result(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def sample_sd(values: list[float]) -> float | str:
    return stdev(values) if len(values) >= 2 else 0.0 if len(values) == 1 else ""


def completion_status(inventory: list[dict[str, Any]], seed_results: list[dict[str, Any]], dataset_keys: tuple[str, ...], seeds: tuple[int, ...]) -> str:
    expected_runs = len(dataset_keys) * len(seeds) * len(METHOD_GROUPS)
    expected_pairs = len(dataset_keys) * len(seeds)
    complete_runs = sum(row["status"] == "COMPLETE" for row in inventory)
    complete_pairs = sum(row["pair_status"] == "COMPLETE_PASS" for row in seed_results)
    if complete_runs == expected_runs and complete_pairs == expected_pairs:
        return "COMPLETE"
    if complete_runs:
        return "PARTIAL"
    return "BLOCKED" if any(row["status"] == "READY_WAIT_DATA" for row in inventory) else "READY_WAIT_GPU" if any(row["status"] == "READY_WAIT_GPU" for row in inventory) else "READY"


def asset_record(role: str, path: Path) -> dict[str, Any]:
    return {"role": role, "path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def build_provenance(
    *,
    output_dir: Path,
    shared_root: Path,
    dataset_keys: tuple[str, ...],
    seeds: tuple[int, ...],
    assets: list[dict[str, Any]],
    path_maps: list[tuple[str, str]],
    gpu_ready: bool,
    gpu_detail: str,
    images_ready: bool,
    requested_execute: bool,
    completion_state: str,
) -> dict[str, Any]:
    return {
        "schema_version": "r4_c2_clean_replacement_v1",
        "output_dir": str(output_dir),
        "worktree_root": str(ROOT),
        "shared_read_only_source_root": str(shared_root),
        "experiment_scope": "fixed historical instrumented-PGDF mask replay; not a rerun of dynamic PGDF",
        "groups": {
            "R": "historical final_selected mask replay after epochs 5,10,15,20,25",
            "O": "same mask with deterministic within-observed-class clean A->B replacement at rate 0.20",
        },
        "oracle_boundary": "clean labels are used only for O membership construction; all training targets remain observed noisy labels",
        "replacement_rule": REPLACEMENT_RULE_ID,
        "replacement_rate": REPLACEMENT_RATE,
        "datasets": list(dataset_keys),
        "training_seeds": list(seeds),
        "expected_update_epochs": list(UPDATE_EPOCHS),
        "path_maps": [{"old": old, "new": new} for old, new in path_maps],
        "gpu_preflight": {"ready": gpu_ready, "detail": gpu_detail},
        "image_preflight_all_pass": images_ready,
        "execution_requested": requested_execute,
        "completion_state": completion_state,
        "code_assets": [asset_record("r4_c2_runner", Path(__file__)), asset_record("replacement_builder", ROOT / "gcdd/clean_replacement.py"), asset_record("replay_trainer", ROOT / "gcdd/lora_dynamic.py")],
        "input_assets": assets,
        "not_claimed": [
            "deployable oracle method",
            "full dynamic PGDF causal improvement",
            "selection-time checkpoint reconstruction",
            "test-driven replacement or ratio choice",
        ],
    }


def write_report(
    *,
    output_dir: Path,
    completion_state: str,
    inventory: list[dict[str, Any]],
    classwise_rows: list[dict[str, Any]],
    seed_results: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    gpu_detail: str,
    all_images_ready: bool,
) -> None:
    total_runs = len(inventory)
    complete_runs = sum(row["status"] == "COMPLETE" for row in inventory)
    total_replacements = sum(int(row["replacement_count"]) for row in classwise_rows)
    budget_pass = sum(row["status"] == "PASS" for row in classwise_rows)
    lines = [
        "# R4-C2 困难干净样本学习价值干预实验",
        "",
        f"- 状态：`{completion_state}`；正式训练完成 `{complete_runs}/{total_runs}`。",
        "- 设计：R 组固定回放同一 instrumented PGDF 历史 final mask；O 组仅在相同 observed-label class 与更新时点内，用 B=clean∩L∩not-P 替换 A=clean∩L∩P 的确定性 20% 子集。",
        "- 训练目标始终是原始 observed noisy label；oracle clean 仅用于预先固定 O 成员，未进入 checkpoint 或 official-test 选择。",
        "- 历史 mask 生效时点：epoch 5/10/15/20/25 后，分别用于下一训练阶段；epoch 1–5 均为完整带噪训练池。",
        f"- class-wise 预算/组成核验：`{budget_pass}/{len(classwise_rows)}` PASS；累计替换 `{total_replacements}` 个 class-update 成员。",
        f"- GPU 预检：{gpu_detail}。图像可访问性：{'全部通过' if all_images_ready else '未通过，未启动正式训练'}。",
        "",
        "## Official-test 配对结果",
        "",
        "| Dataset | Seed | R Top-1 (%) | O Top-1 (%) | O−R (pp) | Pair status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in seed_results:
        lines.append(
            f"| {row['dataset']} | {row['training_seed']} | {fmt(row['r_official_test_top1_percent'])} | "
            f"{fmt(row['o_official_test_top1_percent'])} | {fmt(row['o_minus_r_pp'])} | {row['pair_status']} |"
        )
    lines.extend(["", "## 三 seed 描述性汇总", "", "| Dataset | n | R mean±SD (%) | O mean±SD (%) | O−R mean±SD (pp) | W/T/L |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for row in paired_summary:
        lines.append(
            f"| {row['dataset']} | {row['completed_pairs']} | {fmt_pm(row['r_mean_official_test_top1_percent'], row['r_sample_sd_official_test_top1_percent'])} | "
            f"{fmt_pm(row['o_mean_official_test_top1_percent'], row['o_sample_sd_official_test_top1_percent'])} | "
            f"{fmt_pm(row['o_minus_r_mean_pp'], row['o_minus_r_sample_sd_pp'])} | {row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本实验是固定历史筛选轨迹下的训练干预，而非完整动态 PGDF 重训。它只能描述将一部分被 prototype gate 排除的 clean L∖P 成员重新纳入，在该受控轨迹与预算下是否伴随 official-test 准确率变化。每个数据集仅有三个配对 seed，结果只作描述性报告，不作显著性夸大。",
            "",
            "## 运行状态与失败项",
            "",
        ]
    )
    for row in inventory:
        if row["status"] != "COMPLETE":
            lines.append(f"- `{row['dataset_key']}/seed{row['training_seed']}/{row['group']}`：`{row['status']}` — {row['reason']}。")
    (output_dir / "R4_C2_CLEAN_REPLACEMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    return "" if value == "" else f"{float(value):.4f}"


def fmt_pm(mean_value: Any, sd_value: Any) -> str:
    return "" if mean_value == "" else f"{float(mean_value):.4f}±{float(sd_value):.4f}"


if __name__ == "__main__":
    main()
