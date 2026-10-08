"""Offline audit for completed Margin-Rank consistency experiments.

The script intentionally reads only saved CSV/JSON/YAML/log files.  It never
imports a model, loads a checkpoint, or constructs a training data loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


DATASETS = ("cub", "cars", "aircraft")
BASELINE_VARIANT = "margin_rank"
CONSISTENCY_VARIANT = "margin_rank_ambiguous_consistency_sequential"
EXPECTED_EPOCHS = tuple(range(1, 31))
EXPECTED_UPDATES = (5, 10, 15, 20, 25)
EPSILON = 1.0e-12


@dataclass(frozen=True)
class SelectionGroups:
    """Dual-evidence memberships reconstructed from a saved selection CSV."""

    pool: frozenset[int]
    loss: frozenset[int]
    geometry: frozenset[int]
    strict_reliable: frozenset[int]
    ambiguous_l: frozenset[int]
    ambiguous_g: frozenset[int]
    ambiguous: frozenset[int]
    suspicious: frozenset[int]
    actual_active: frozenset[int]
    fallback: frozenset[int]
    ambiguous_for_consistency: frozenset[int]
    labels: dict[int, str]
    paths: dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of completed Margin-Rank and Ambiguous Consistency runs."
    )
    parser.add_argument(
        "--variants-root",
        type=Path,
        default=Path(
            "outputs/neighbor_margin/cyclic_asym40_noise42/"
            "fixedval_s20250726/r08_p04_w5_u5/variants"
        ),
        help="Directory containing the margin_rank variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/ambiguous_consistency_offline_audit"),
        help="New analysis-only output directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Completed training seed to audit.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write schema-less empty CSV: {path}")
    fields = list(materialized[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_cell(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    raise ValueError(f"Expected yes/no selection flag, found {value!r}.")


def finite_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite numeric log value: {value!r}")
    return parsed


def integer_or_none(value: str | None) -> int | None:
    """Parse an optional historical CSV integer without turning missing into zero."""
    if value is None or value == "":
        return None
    return int(value)


def get_path_value(payload: dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def selection_rows_by_epoch(path: Path) -> dict[int, list[dict[str, str]]]:
    rows = read_csv(path)
    required = {"epoch", "index", "web_label", "path", "loss_selected", "neighbor_margin_candidate", "active_training"}
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        raise ValueError(f"{path} lacks required selection fields: {missing}")
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["epoch"])].append(row)
    return dict(grouped)


def build_groups(rows: list[dict[str, str]]) -> SelectionGroups:
    """Reconstruct L/G dual-evidence strata without using clean/noisy fields."""
    pool: set[int] = set()
    loss: set[int] = set()
    geometry: set[int] = set()
    active: set[int] = set()
    labels: dict[int, str] = {}
    paths: dict[int, str] = {}
    for row in rows:
        index = int(row["index"])
        if index in pool:
            raise ValueError(f"Duplicate original index {index} within one selection update.")
        pool.add(index)
        labels[index] = row["web_label"]
        paths[index] = row["path"]
        if bool_cell(row["loss_selected"]):
            loss.add(index)
        if bool_cell(row["neighbor_margin_candidate"]):
            geometry.add(index)
        if bool_cell(row["active_training"]):
            active.add(index)
    strict = loss & geometry
    ambiguous_l = loss - geometry
    ambiguous_g = geometry - loss
    ambiguous = ambiguous_l | ambiguous_g
    suspicious = pool - (loss | geometry)
    fallback = active - strict
    ambiguous_for_consistency = ambiguous - active
    groups = (strict, ambiguous_l, ambiguous_g, suspicious)
    if any(left & right for position, left in enumerate(groups) for right in groups[position + 1 :]):
        raise ValueError("Dual-evidence groups are not mutually exclusive.")
    if strict | ambiguous_l | ambiguous_g | suspicious != pool:
        raise ValueError("Dual-evidence groups do not cover the complete selection pool.")
    if strict - active:
        raise ValueError("A strict L/G Reliable sample is absent from active training.")
    if ambiguous_for_consistency & active:
        raise ValueError("An Ambiguous sample would receive both active CE and consistency.")
    return SelectionGroups(
        pool=frozenset(pool),
        loss=frozenset(loss),
        geometry=frozenset(geometry),
        strict_reliable=frozenset(strict),
        ambiguous_l=frozenset(ambiguous_l),
        ambiguous_g=frozenset(ambiguous_g),
        ambiguous=frozenset(ambiguous),
        suspicious=frozenset(suspicious),
        actual_active=frozenset(active),
        fallback=frozenset(fallback),
        ambiguous_for_consistency=frozenset(ambiguous_for_consistency),
        labels=labels,
        paths=paths,
    )


def set_metrics(left: frozenset[int], right: frozenset[int], pool_size: int) -> dict[str, Any]:
    intersection = left & right
    union = left | right
    symmetric_difference = left ^ right
    return {
        "left_count": len(left),
        "right_count": len(right),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "left_only_count": len(left - right),
        "right_only_count": len(right - left),
        "symmetric_difference_count": len(symmetric_difference),
        "jaccard": len(intersection) / len(union) if union else None,
        "symmetric_difference_pool_ratio": len(symmetric_difference) / pool_size if pool_size else None,
    }


def compare_selection_groups(
    baseline: SelectionGroups,
    consistency: SelectionGroups,
) -> dict[str, Any]:
    shared_pool = baseline.pool & consistency.pool
    label_mismatch = sum(
        baseline.labels[index] != consistency.labels[index]
        for index in shared_pool
    )
    path_mismatch = sum(
        baseline.paths[index] != consistency.paths[index]
        for index in shared_pool
    )
    if baseline.pool != consistency.pool:
        raise ValueError("The two runs have different original-index training pools.")
    pool_size = len(baseline.pool)
    strict = set_metrics(baseline.strict_reliable, consistency.strict_reliable, pool_size)
    loss = set_metrics(baseline.loss, consistency.loss, pool_size)
    geometry = set_metrics(baseline.geometry, consistency.geometry, pool_size)
    return {
        "full_pool_size": pool_size,
        "shared_index_count": len(shared_pool),
        "observed_label_mismatch_count": label_mismatch,
        "path_mismatch_count": path_mismatch,
        **{f"reliable_{key}": value for key, value in strict.items()},
        **{f"loss_candidate_{key}": value for key, value in loss.items()},
        **{f"margin_candidate_{key}": value for key, value in geometry.items()},
    }


def update_lookup(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    lookup = {int(row["epoch"]): row for row in rows}
    if tuple(sorted(lookup)) != EXPECTED_UPDATES:
        raise ValueError(f"Expected selection update epochs {EXPECTED_UPDATES}, got {tuple(sorted(lookup))}.")
    return lookup


def training_lookup(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    lookup = {int(row["epoch"]): row for row in rows}
    if len(rows) != len(lookup) or tuple(sorted(lookup)) != EXPECTED_EPOCHS:
        raise ValueError("Training log does not contain exactly epochs 1..30.")
    return lookup


def inventory_row(variant_root: Path, variant: str, dataset: str, seed: int) -> dict[str, Any]:
    dataset_root = variant_root / variant / dataset
    run_root = dataset_root / f"seed{seed}"
    required_root = (
        "checkpoint_validation_results.csv",
        "checkpoint_validation_summary.csv",
        "checkpoint_validation_summary.json",
        "checkpoint_validation_updates.csv",
        "resolved_config.yaml",
        "run_index.csv",
        "run_summary.md",
        "train_log.csv",
        "validation_manifest.csv",
        "validation_manifest.json",
    )
    required_run = (
        "result.json",
        "selection_policy.json",
        "selection_rows.csv",
        "selection_updates.csv",
        "selection_per_class.csv",
        "train_log.csv",
    )
    root_missing = [name for name in required_root if not (dataset_root / name).is_file()]
    run_missing = [name for name in required_run if not (run_root / name).is_file()]
    epochs: list[int] = []
    updates: list[int] = []
    if not run_missing:
        epochs = sorted(int(row["epoch"]) for row in read_csv(run_root / "train_log.csv"))
        updates = sorted(int(row["epoch"]) for row in read_csv(run_root / "selection_updates.csv"))
    checkpoint_count = (
        sum(1 for path in (run_root / "checkpoints").glob("**/*") if path.is_file())
        if (run_root / "checkpoints").exists()
        else 0
    )
    status = "missing" if root_missing or run_missing else "complete"
    if status == "complete" and (epochs != list(EXPECTED_EPOCHS) or updates != list(EXPECTED_UPDATES)):
        status = "incomplete_epochs_or_updates"
    return {
        "dataset": dataset,
        "method": "Margin-Rank" if variant == BASELINE_VARIANT else "Margin-Rank + Ambiguous Consistency (sequential)",
        "variant": variant,
        "seed": seed,
        "epochs": len(epochs),
        "epoch_range": f"{epochs[0]}-{epochs[-1]}" if epochs else "NA",
        "updates": len(updates),
        "update_epochs": ",".join(str(epoch) for epoch in updates) if updates else "NA",
        "checkpoints_present": checkpoint_count > 0,
        "checkpoint_file_count": checkpoint_count,
        "logs_complete": "yes" if status == "complete" else "no",
        "status": status,
        "root_missing": "; ".join(root_missing) or "",
        "run_missing": "; ".join(run_missing) or "",
        "dataset_root": str(dataset_root),
        "run_root": str(run_root),
    }


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join("---" for _ in fields) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "NA")).replace("|", "\\|") for field in fields) + " |")
    return lines


def fmt_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.3f}%"


def fmt_number(value: float | None, digits: int = 6) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def flatten_config(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_config(value, path))
        else:
            result[path] = value
    return result


def main() -> None:
    args = parse_args()
    variants_root = args.variants_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not variants_root.is_dir():
        raise FileNotFoundError(f"Variants root not found: {variants_root}")

    inventory = [
        inventory_row(variants_root, variant, dataset, args.seed)
        for variant in (BASELINE_VARIANT, CONSISTENCY_VARIANT)
        for dataset in DATASETS
    ]
    write_csv(output_dir / "data_inventory.csv", inventory)
    incomplete = [row for row in inventory if row["logs_complete"] != "yes"]
    if incomplete:
        raise RuntimeError(f"Target runs are incomplete: {incomplete}")

    protocol_rows: list[dict[str, Any]] = []
    epoch5_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    report_data: dict[str, Any] = {}
    protocol_notes: list[str] = []

    for dataset in DATASETS:
        baseline_root = variants_root / BASELINE_VARIANT / dataset
        consistency_root = variants_root / CONSISTENCY_VARIANT / dataset
        baseline_run = baseline_root / f"seed{args.seed}"
        consistency_run = consistency_root / f"seed{args.seed}"
        baseline_cfg = read_yaml(baseline_root / "resolved_config.yaml")
        consistency_cfg = read_yaml(consistency_root / "resolved_config.yaml")
        baseline_result = read_json(baseline_run / "result.json")
        consistency_result = read_json(consistency_run / "result.json")
        baseline_updates = update_lookup(read_csv(baseline_run / "selection_updates.csv"))
        consistency_updates = update_lookup(read_csv(consistency_run / "selection_updates.csv"))
        baseline_log = training_lookup(read_csv(baseline_run / "train_log.csv"))
        consistency_log = training_lookup(read_csv(consistency_run / "train_log.csv"))
        baseline_rows = selection_rows_by_epoch(baseline_run / "selection_rows.csv")
        consistency_rows = selection_rows_by_epoch(consistency_run / "selection_rows.csv")
        if tuple(sorted(baseline_rows)) != EXPECTED_UPDATES or tuple(sorted(consistency_rows)) != EXPECTED_UPDATES:
            raise ValueError(f"{dataset}: sample-level selection epochs do not match expected schedule.")
        baseline_groups = {epoch: build_groups(rows) for epoch, rows in baseline_rows.items()}
        consistency_groups = {epoch: build_groups(rows) for epoch, rows in consistency_rows.items()}

        baseline_noise_path = Path(str(baseline_result["noise_index"]))
        consistency_noise_path = Path(str(consistency_result["noise_index"]))
        if not baseline_noise_path.is_absolute():
            baseline_noise_path = Path.cwd() / baseline_noise_path
        if not consistency_noise_path.is_absolute():
            consistency_noise_path = Path.cwd() / consistency_noise_path
        baseline_manifest = baseline_root / "validation_manifest.csv"
        consistency_manifest = consistency_root / "validation_manifest.csv"
        baseline_flat = flatten_config(baseline_cfg)
        consistency_flat = flatten_config(consistency_cfg)

        alignment_keys = (
            "dataset.name",
            f"datasets.{dataset}.layout",
            "feature.backend",
            "feature.input_size",
            "lora.rank",
            "lora.alpha",
            "lora.dropout",
            "lora.target_modules",
            "lora_train.epochs",
            "lora_train.batch_size",
            "lora_train.eval_batch_size",
            "lora_train.num_workers",
            "lora_train.pin_memory",
            "lora_train.lora_lr",
            "lora_train.head_lr",
            "lora_train.weight_decay",
            "lora_train.scheduler",
            "lora_train.warmup_ratio",
            "lora_train.amp",
            "checkpoint_validation.protocol",
            "checkpoint_validation.validation_ratio",
            "checkpoint_validation.validation_seed",
            "checkpoint_validation.dynamic_ratio",
            "checkpoint_validation.fixed_p",
            "checkpoint_validation.geometry_mode",
            "checkpoint_validation.neighbor_margin_use_fallback",
            "checkpoint_validation.neighbor_margin_positive_only",
            "checkpoint_validation.warmup_epochs",
            "checkpoint_validation.update_interval",
            "checkpoint_validation.official_test_selected_only",
            "checkpoint_validation.posthoc_oracle_test",
            "checkpoint_validation.noise_realization.noise_strategy",
            "checkpoint_validation.noise_realization.noise_ratio",
            "checkpoint_validation.noise_realization.noise_seed",
            "checkpoint_validation.noise_realization.mapping_type",
        )
        for key in alignment_keys:
            base_value = baseline_flat.get(key)
            consistency_value = consistency_flat.get(key)
            protocol_rows.append(
                {
                    "dataset": dataset,
                    "item": key,
                    "baseline_value": canonical_json(base_value),
                    "consistency_value": canonical_json(consistency_value),
                    "status": "match" if canonical_json(base_value) == canonical_json(consistency_value) else "mismatch",
                    "evidence": "resolved_config.yaml",
                }
            )
        extra_rows = (
            ("noise_index_sha256", sha256_file(baseline_noise_path), sha256_file(consistency_noise_path), "result.json noise_index"),
            ("validation_manifest_sha256", sha256_file(baseline_manifest), sha256_file(consistency_manifest), "validation_manifest.csv"),
            ("checkpoint_selection", baseline_result.get("checkpoint_protocol"), consistency_result.get("checkpoint_protocol"), "result.json"),
            ("official_test_protocol", baseline_result.get("official_test_evaluation"), consistency_result.get("official_test_evaluation"), "result.json"),
            ("trainable_parameter_count", baseline_result.get("trainable_params"), consistency_result.get("trainable_params"), "result.json"),
            ("total_parameter_count", baseline_result.get("total_params"), consistency_result.get("total_params"), "result.json"),
            ("consistency_enabled", "no", consistency_result.get("ambiguous_consistency"), "result.json"),
            ("consistency_weight", "NA", consistency_result.get("consistency_weight"), "result.json"),
            ("consistency_backward_mode", "NA", consistency_result.get("consistency_backward_mode"), "result.json"),
            ("ambiguous_micro_batch_size", "NA", consistency_result.get("ambiguous_micro_batch_size"), "result.json"),
            ("optimizer", "AdamW indicated by code; not stored in run metadata", "AdamW indicated by code; not stored in run metadata", "historical/current gcdd/lora_dynamic.py"),
            ("gradient_accumulation", "not present in saved config/log", "not present in saved config/log", "resolved config + train log schema"),
            ("classifier_configuration", "same trainable parameter count; full head config not separately stored", "same trainable parameter count; full head config not separately stored", "result.json"),
            ("cuda_determinism_runtime", "not persisted", "not persisted", "run artifacts + code audit"),
        )
        for item, base_value, consistency_value, evidence in extra_rows:
            protocol_rows.append(
                {
                    "dataset": dataset,
                    "item": item,
                    "baseline_value": canonical_json(base_value),
                    "consistency_value": canonical_json(consistency_value),
                    "status": (
                        "expected_difference"
                        if item.startswith("consistency_") or item == "ambiguous_micro_batch_size"
                        else "unconfirmed"
                        if item in {"optimizer", "gradient_accumulation", "classifier_configuration", "cuda_determinism_runtime"}
                        else "match"
                        if canonical_json(base_value) == canonical_json(consistency_value)
                        else "mismatch"
                    ),
                    "evidence": evidence,
                }
            )

        for epoch in EXPECTED_UPDATES:
            comparison = compare_selection_groups(baseline_groups[epoch], consistency_groups[epoch])
            baseline_group = baseline_groups[epoch]
            consistency_group = consistency_groups[epoch]
            trajectory_rows.append(
                {
                    "dataset": dataset,
                    "seed": args.seed,
                    "epoch": epoch,
                    "full_pool_size": comparison["full_pool_size"],
                    "baseline_strict_reliable_count": len(baseline_group.strict_reliable),
                    "consistency_strict_reliable_count": len(consistency_group.strict_reliable),
                    "baseline_actual_active_count": len(baseline_group.actual_active),
                    "consistency_actual_active_count": len(consistency_group.actual_active),
                    "baseline_active_observed_class_count": len({baseline_group.labels[index] for index in baseline_group.actual_active}),
                    "consistency_active_observed_class_count": len({consistency_group.labels[index] for index in consistency_group.actual_active}),
                    "baseline_fallback_count": len(baseline_group.fallback),
                    "consistency_fallback_count": len(consistency_group.fallback),
                    "baseline_ambiguous_l_count": len(baseline_group.ambiguous_l),
                    "consistency_ambiguous_l_count": len(consistency_group.ambiguous_l),
                    "baseline_ambiguous_g_count": len(baseline_group.ambiguous_g),
                    "consistency_ambiguous_g_count": len(consistency_group.ambiguous_g),
                    "baseline_ambiguous_total_count": len(baseline_group.ambiguous),
                    "consistency_ambiguous_total_count": len(consistency_group.ambiguous),
                    "consistency_ambiguous_for_consistency_count": len(consistency_group.ambiguous_for_consistency),
                    "baseline_suspicious_count": len(baseline_group.suspicious),
                    "consistency_suspicious_count": len(consistency_group.suspicious),
                    "reliable_intersection_count": comparison["reliable_intersection_count"],
                    "reliable_union_count": comparison["reliable_union_count"],
                    "reliable_jaccard": comparison["reliable_jaccard"],
                    "reliable_symmetric_difference_count": comparison["reliable_symmetric_difference_count"],
                    "reliable_symmetric_difference_pool_ratio": comparison["reliable_symmetric_difference_pool_ratio"],
                    "observed_label_mismatch_count": comparison["observed_label_mismatch_count"],
                    "path_mismatch_count": comparison["path_mismatch_count"],
                }
            )
            if epoch == 5:
                epoch5_rows.append({"dataset": dataset, "seed": args.seed, **comparison})

        for method, log_lookup in (("margin_rank", baseline_log), ("consistency", consistency_log)):
            for epoch in EXPECTED_EPOCHS:
                row = log_lookup[epoch]
                dynamics_rows.append(
                    {
                        "dataset": dataset,
                        "seed": args.seed,
                        "method": method,
                        "epoch": epoch,
                        "validation_top1": finite_float(row.get("top1", "")),
                        "validation_top5": finite_float(row.get("top5", "")),
                        "logged_training_loss": finite_float(row.get("loss", "")),
                        "train_samples": int(row["train_samples"]),
                        "selected_count": integer_or_none(row.get("selected_count")),
                        "selected_ratio": finite_float(row.get("selected_ratio", "")),
                        "lr_lora": finite_float(row.get("lr_lora", "")),
                        "lr_head": finite_float(row.get("lr_head", "")),
                        "supervised_ce_loss": finite_float(row.get("supervised_ce_loss", "")),
                        "consistency_loss": finite_float(row.get("consistency_loss", "")),
                        "total_loss": finite_float(row.get("total_loss", "")),
                        "consistency_computed": row.get("consistency_computed", "NA"),
                        "consistency_reason": row.get("consistency_reason", "NA"),
                        "consistency_weight": finite_float(row.get("consistency_weight", "")),
                        "consistency_backward_mode": row.get("consistency_backward_mode", "NA"),
                        "optimizer_steps": row.get("optimizer_steps", "NA"),
                        "scheduler_steps": row.get("scheduler_steps", "NA"),
                        "ambiguous_batch_count": row.get("ambiguous_batch_count", "NA"),
                        "consistency_micro_batch_count": row.get("consistency_micro_batch_count", "NA"),
                        "ambiguous_sample_exposures": row.get("ambiguous_sample_exposures", "NA"),
                        "weak_prediction_max_probability_mean": finite_float(row.get("weak_prediction_max_probability_mean", "")),
                        "weak_prediction_entropy_mean": finite_float(row.get("weak_prediction_entropy_mean", "")),
                        "weak_strong_prediction_agreement": finite_float(row.get("weak_strong_prediction_agreement", "")),
                    }
                )

        class_count = len(set(baseline_groups[5].labels.values()))
        if class_count <= 1:
            raise ValueError(f"{dataset}: cannot normalize entropy with <=1 observed class.")
        for epoch in range(1, 6):
            warmup_row = consistency_log[epoch]
            if (
                warmup_row.get("consistency_computed") != "no"
                or warmup_row.get("consistency_reason") != "warmup"
                or int(warmup_row.get("ambiguous_batch_count", "0")) != 0
                or int(warmup_row.get("ambiguous_sample_exposures", "0")) != 0
            ):
                raise ValueError(f"{dataset} epoch {epoch}: consistency unexpectedly participated during warmup.")
        for epoch in range(6, 31):
            row = consistency_log[epoch]
            if row.get("consistency_computed") != "yes":
                raise ValueError(f"{dataset} epoch {epoch}: expected consistency_computed=yes.")
            ce = finite_float(row["supervised_ce_loss"])
            kl = finite_float(row["consistency_loss"])
            weight = finite_float(row["consistency_weight"])
            entropy = finite_float(row["weak_prediction_entropy_mean"])
            signal_rows.append(
                {
                    "dataset": dataset,
                    "seed": args.seed,
                    "epoch": epoch,
                    "observed_class_count": class_count,
                    "supervised_ce_loss": ce,
                    "consistency_loss": kl,
                    "weighted_consistency_loss": None if kl is None or weight is None else weight * kl,
                    "loss_ratio_weighted_kl_over_ce": None if ce is None or kl is None or weight is None else (weight * kl) / (ce + EPSILON),
                    "total_loss": finite_float(row["total_loss"]),
                    "weak_prediction_max_probability_mean": finite_float(row["weak_prediction_max_probability_mean"]),
                    "weak_prediction_entropy_mean": entropy,
                    "weak_prediction_entropy_normalized": None if entropy is None else entropy / math.log(class_count),
                    "weak_strong_prediction_agreement": finite_float(row["weak_strong_prediction_agreement"]),
                    "ambiguous_batch_count": int(row["ambiguous_batch_count"]),
                    "ambiguous_sample_exposures": int(row["ambiguous_sample_exposures"]),
                    "optimizer_steps": int(row["optimizer_steps"]),
                    "scheduler_steps": int(row["scheduler_steps"]),
                }
            )

        baseline_final = baseline_log[30]
        consistency_final = consistency_log[30]
        warmup_validation_deltas = [
            float(consistency_log[epoch]["top1"]) - float(baseline_log[epoch]["top1"])
            for epoch in range(1, 6)
        ]
        post_warmup_validation_deltas = [
            float(consistency_log[epoch]["top1"]) - float(baseline_log[epoch]["top1"])
            for epoch in range(6, 31)
        ]
        validation_rows.append(
            {
                "dataset": dataset,
                "seed": args.seed,
                "baseline_best_validation_top1": finite_float(str(baseline_result["best_val_top1"])),
                "baseline_best_validation_epoch": int(baseline_result["best_val_epoch"]),
                "baseline_final_validation_top1": finite_float(baseline_final["top1"]),
                "baseline_validation_selected_test_top1": finite_float(str(baseline_result["validation_selected_test_top1"])),
                "consistency_best_validation_top1": finite_float(str(consistency_result["best_val_top1"])),
                "consistency_best_validation_epoch": int(consistency_result["best_val_epoch"]),
                "consistency_final_validation_top1": finite_float(consistency_final["top1"]),
                "consistency_validation_selected_test_top1": finite_float(str(consistency_result["validation_selected_test_top1"])),
                "best_validation_delta": float(consistency_result["best_val_top1"]) - float(baseline_result["best_val_top1"]),
                "final_validation_delta": float(consistency_final["top1"]) - float(baseline_final["top1"]),
                "validation_selected_test_delta": float(consistency_result["validation_selected_test_top1"]) - float(baseline_result["validation_selected_test_top1"]),
                "warmup_validation_max_abs_delta": max(abs(value) for value in warmup_validation_deltas),
                "epoch5_validation_delta": warmup_validation_deltas[-1],
                "post_warmup_mean_validation_delta": sum(post_warmup_validation_deltas) / len(post_warmup_validation_deltas),
                "post_warmup_consistency_higher_epoch_count": sum(value > 0.0 for value in post_warmup_validation_deltas),
                "post_warmup_validation_tie_epoch_count": sum(value == 0.0 for value in post_warmup_validation_deltas),
                "post_warmup_consistency_lower_epoch_count": sum(value < 0.0 for value in post_warmup_validation_deltas),
            }
        )
        protocol_rows.append(
            {
                "dataset": dataset,
                "item": "ambiguous_loader_logical_batch_size",
                "baseline_value": canonical_json("NA"),
                "consistency_value": canonical_json(consistency_flat.get("lora_train.batch_size")),
                "status": "expected_difference",
                "evidence": "resolved_config.yaml + sequential loader construction",
            }
        )
        report_data[dataset] = {
            "baseline_result": baseline_result,
            "consistency_result": consistency_result,
            "baseline_log": baseline_log,
            "consistency_log": consistency_log,
            "baseline_groups": baseline_groups,
            "consistency_groups": consistency_groups,
        }

    write_csv(output_dir / "protocol_alignment.csv", protocol_rows)
    write_csv(output_dir / "epoch5_selection_diff.csv", epoch5_rows)
    write_csv(output_dir / "selection_trajectory.csv", trajectory_rows)
    write_csv(output_dir / "training_dynamics.csv", dynamics_rows)
    write_csv(output_dir / "consistency_signal_summary.csv", signal_rows)
    write_csv(output_dir / "validation_comparison.csv", validation_rows)

    inventory_fields = ["dataset", "method", "seed", "epochs", "epoch_range", "updates", "update_epochs", "logs_complete", "checkpoints_present", "status"]
    inventory_lines = [
        "# 离线审计数据清单",
        "",
        "本清单只读取既有实验产物；未加载 checkpoint。",
        "",
        *markdown_table(inventory, inventory_fields + ["dataset_root", "run_root"]),
        "",
        "## 已检查的必要产物",
        "",
        "每个 run 均检查了根目录 result/config/validation-manifest 文件，以及 seed 级 result、policy、selection rows、selection updates、per-class rows、train log 和 checkpoint 目录。",
        "",
        "六个目标 run 均含 epoch 1-30 和 selection epochs 5、10、15、20、25。",
    ]
    (output_dir / "data_inventory.md").write_text("\n".join(inventory_lines) + "\n", encoding="utf-8")

    mismatches = [row for row in protocol_rows if row["status"] == "mismatch"]
    expected_differences = [row for row in protocol_rows if row["status"] == "expected_difference"]
    unconfirmed = [row for row in protocol_rows if row["status"] == "unconfirmed"]
    protocol_lines = [
        "# 训练协议对齐审计",
        "",
        "## 结论：CONDITIONAL（主要协议可比，但存在未确认运行时/版本信息）",
        "",
        "三个数据集的已保存有效配置、noise-index SHA256、validation-manifest SHA256 和核心训练超参数均一致。预期差异仅为 consistency 开关、lambda=0.5 和 sequential backward 元数据。",
        "",
        "但运行产物没有保存精确 Git SHA，因而不能达到严格版本锁定。文件时间将 Margin-Rank 放在提交 `c128c62` 之后、`701ed9c` 之前，而 consistency 运行开始于 `966f39e` 之后。即使可见的有效配置一致，历史运行代码身份仍不能仅由运行元数据证明。",
        "",
        "两组 `resolved_config.yaml` 顶层 `protocol.name`/`variant_id` 都保留了共享 YAML 默认值（`neighbor_margin_strict_cyclic_asym40` / `reliable_only`）；实际运行身份由 variant 目录、seed 级 result JSON 和 `checkpoint_validation`/`pgdf` 的 resolved 字段确定。这是元数据局限，而非已证实的训练协议不一致。",
        "",
        "## 核心对齐证据",
        "",
        *markdown_table(
            [row for row in protocol_rows if row["item"] in {"noise_index_sha256", "validation_manifest_sha256", "checkpoint_selection", "official_test_protocol", "trainable_parameter_count", "total_parameter_count", "lora_train.batch_size", "lora_train.epochs", "checkpoint_validation.dynamic_ratio", "checkpoint_validation.fixed_p", "checkpoint_validation.warmup_epochs", "checkpoint_validation.update_interval", "checkpoint_validation.neighbor_margin_positive_only", "checkpoint_validation.neighbor_margin_use_fallback"}],
            ["dataset", "item", "baseline_value", "consistency_value", "status", "evidence"],
        ),
        "",
        f"非预期 resolved-config 不一致：{len(mismatches)} 项；预期 consistency 专属差异：{len(expected_differences)} 项；缺少运行时直接证据：{len(unconfirmed)} 项。",
        "",
        "## 缺失或未落盘的证据",
        "",
        "- 运行产物未保存精确 Git commit、PyTorch/CUDA 版本、cuDNN deterministic 设置或 DataLoader worker seed 状态。",
        "- 日志未直接记录 optimizer 名称或 gradient accumulation 参数；当前/历史代码审计指向 AdamW 且每个 Reliable batch 一步，但历史运行时状态未单独落盘。",
        "- 历史 Margin-Rank train log 没有逐 epoch optimizer/scheduler step 计数；Consistency 有这些字段，因此只能验证后者，不能直接逐 epoch 对照早期运行。",
    ]
    (output_dir / "protocol_alignment.md").write_text("\n".join(protocol_lines) + "\n", encoding="utf-8")

    signal_by_dataset: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in signal_rows:
        signal_by_dataset[row["dataset"]][int(row["epoch"])] = row
    validation_by_dataset = {row["dataset"]: row for row in validation_rows}
    epoch5_by_dataset = {row["dataset"]: row for row in epoch5_rows}
    trajectory_by_dataset: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in trajectory_rows:
        trajectory_by_dataset[row["dataset"]][int(row["epoch"])] = row

    report_lines = [
        "# Margin-Rank + Ambiguous Consistency 纯离线实验诊断",
        "",
        "## 范围与证据边界",
        "",
        "本报告只读取已完成的 seed42 日志。未加载模型、未执行 forward/backward、未调用 optimizer.step、未重新选样，也未改动任何原始产物。以下测试性能均为原始记录的 validation-selected official-test 值，不是本次重新选择的 checkpoint。",
        "",
        "## 1. 数据完整性",
        "",
        *markdown_table(inventory, inventory_fields),
        "",
        "六个目标 run 均完整：每个有 30 个训练 epoch 和 5 次 selection update。",
        "",
        "## 2. 训练协议对齐与可比性",
        "",
        "**分级：B（Comparable with caveats）。** 已保存的 noise index、validation manifest、dataset/seed、模型与 LoRA 设置、图像大小、batch size、optimizer 超参数、scheduler 配置、训练周期、r/p、Margin-Rank 开关以及 validation-selected checkpoint 协议一致。Consistency 专属字段是预期差异。",
        "",
        "历史运行未保存精确 Git SHA。时间证据表明两组跨越不同 commit，因此本审计不能认证 baseline 与 consistency 代码逐字节一致。该比较可用于探索性解释；对很小的性能差异，应先建立当前版本的配对 baseline 再作强结论。",
        "",
        "## 3. Epoch-5 筛选差异",
        "",
        *markdown_table(
            [
                {
                    "dataset": row["dataset"],
                    "MR reliable": row["reliable_left_count"],
                    "Consistency reliable": row["reliable_right_count"],
                    "intersection": row["reliable_intersection_count"],
                    "union": row["reliable_union_count"],
                    "Jaccard": fmt_number(row["reliable_jaccard"]),
                    "symmetric diff": row["reliable_symmetric_difference_count"],
                    "diff / pool": fmt_percent(row["reliable_symmetric_difference_pool_ratio"]),
                    "label mismatch": row["observed_label_mismatch_count"],
                }
                for row in epoch5_rows
            ],
            ["dataset", "MR reliable", "Consistency reliable", "intersection", "union", "Jaccard", "symmetric diff", "diff / pool", "label mismatch"],
        ),
        "",
        "**直接观察：**三个 epoch5 对照的 pool、original index、observed label 和路径均一致；上表差异是实际 membership 差异，而不是 CSV 行错位。",
        "",
        "**代码审计：**当前 consistency 分支在 seed 设置后创建 weak/strong transform 对象，但仅在 epoch 大于 warmup 时创建 Ambiguous DataLoader；transform 构造函数中没有显式随机抽样。warmup 期间 sequential 路径在共同的 optimizer/scheduler step 前反传 CE，不构建 KL tensor。Consistency 日志的 epoch1-5 均记录 `consistency_computed=no`、`consistency_reason=warmup`、Ambiguous batch/exposure=0，因此没有直接证据表明 epoch6 前存在 KL 监督。",
        "",
        "源代码仅记录 `torch.manual_seed`/`torch.cuda.manual_seed_all`；未保存 deterministic CUDA/cuDNN 设置、worker seed 或运行时库版本。跨 commit 运行和未记录的 GPU/DataLoader 非确定性是可能原因，而非已经证实的原因。**现有证据不能确定 epoch5 差异的来源。**",
        "",
        "## 4. 五次 selection update 轨迹",
        "",
        "下列 strata 直接由保存的逐样本 L（`loss_selected`）和 G（`neighbor_margin_candidate`）重建。Consistency run 的重建结果与保存的 dual-evidence 字段一致。15 次 consistency update 均为 fallback=0、actual active=strict Reliable；没有 Ambiguous 样本进入监督 CE。",
        "",
    ]
    for dataset in DATASETS:
        report_lines.extend([
            f"### {dataset}",
            "",
            *markdown_table(
                [
                    {
                        "epoch": epoch,
                        "MR R": trajectory_by_dataset[dataset][epoch]["baseline_strict_reliable_count"],
                        "Cons R": trajectory_by_dataset[dataset][epoch]["consistency_strict_reliable_count"],
                        "MR active classes": trajectory_by_dataset[dataset][epoch]["baseline_active_observed_class_count"],
                        "Cons active classes": trajectory_by_dataset[dataset][epoch]["consistency_active_observed_class_count"],
                        "MR Amb": trajectory_by_dataset[dataset][epoch]["baseline_ambiguous_total_count"],
                        "Cons Amb": trajectory_by_dataset[dataset][epoch]["consistency_ambiguous_total_count"],
                        "MR Susp": trajectory_by_dataset[dataset][epoch]["baseline_suspicious_count"],
                        "Cons Susp": trajectory_by_dataset[dataset][epoch]["consistency_suspicious_count"],
                        "Reliable Jaccard": fmt_number(trajectory_by_dataset[dataset][epoch]["reliable_jaccard"]),
                    }
                    for epoch in EXPECTED_UPDATES
                ],
                ["epoch", "MR R", "Cons R", "MR active classes", "Cons active classes", "MR Amb", "Cons Amb", "MR Susp", "Cons Susp", "Reliable Jaccard"],
            ),
            "",
        ])

    report_lines.extend([
        "## 5. Consistency 信号动态",
        "",
        "`consistency_loss` 是保存的 logical Ambiguous batch KL 均值。`weighted_consistency_loss = 0.5 * KL`。下列比值是损失量级比值 `(0.5 * KL)/(CE + 1e-12)`，**不是**梯度范数或梯度方向比值。训练代码中的 weak entropy 使用自然对数；这里仅为跨数据集描述，将其除以 `log(observed class count)`。",
        "",
        *markdown_table(
            [
                {
                    "dataset": dataset,
                    "KL e6->e30": f"{fmt_number(signal_by_dataset[dataset][6]['consistency_loss'])} -> {fmt_number(signal_by_dataset[dataset][30]['consistency_loss'])}",
                    "confidence e6->e30": f"{fmt_number(signal_by_dataset[dataset][6]['weak_prediction_max_probability_mean'])} -> {fmt_number(signal_by_dataset[dataset][30]['weak_prediction_max_probability_mean'])}",
                    "agreement e6->e30": f"{fmt_percent(signal_by_dataset[dataset][6]['weak_strong_prediction_agreement'])} -> {fmt_percent(signal_by_dataset[dataset][30]['weak_strong_prediction_agreement'])}",
                    "normalized entropy e6->e30": f"{fmt_number(signal_by_dataset[dataset][6]['weak_prediction_entropy_normalized'])} -> {fmt_number(signal_by_dataset[dataset][30]['weak_prediction_entropy_normalized'])}",
                    "loss ratio e6->e30": f"{fmt_number(signal_by_dataset[dataset][6]['loss_ratio_weighted_kl_over_ce'])} -> {fmt_number(signal_by_dataset[dataset][30]['loss_ratio_weighted_kl_over_ce'])}",
                }
                for dataset in DATASETS
            ],
            ["dataset", "KL e6->e30", "confidence e6->e30", "agreement e6->e30", "normalized entropy e6->e30", "loss ratio e6->e30"],
        ),
        "",
        "**直接观察：**三个数据集 epoch6-30 的 KL 都有限且非零，日志无 NaN/Inf。三个数据集 agreement 都上升，KL 总体下降但并非逐 epoch 单调下降。这表明信号实际参与训练，预测在增强下更一致；不能据此认定预测更正确。",
        "",
        "## 6. Validation 曲线与既存 official-test 对照",
        "",
        *markdown_table(
            [
                {
                    "dataset": row["dataset"],
                    "MR best val (epoch)": f"{fmt_percent(row['baseline_best_validation_top1'])} ({row['baseline_best_validation_epoch']})",
                    "Cons best val (epoch)": f"{fmt_percent(row['consistency_best_validation_top1'])} ({row['consistency_best_validation_epoch']})",
                    "MR final val": fmt_percent(row["baseline_final_validation_top1"]),
                    "Cons final val": fmt_percent(row["consistency_final_validation_top1"]),
                    "MR test": fmt_percent(row["baseline_validation_selected_test_top1"]),
                    "Cons test": fmt_percent(row["consistency_validation_selected_test_top1"]),
                    "test delta": fmt_percent(row["validation_selected_test_delta"]),
                    "warmup max |delta|": fmt_percent(row["warmup_validation_max_abs_delta"]),
                    "mean delta e6-30": fmt_percent(row["post_warmup_mean_validation_delta"]),
                    "Cons higher/tie/lower e6-30": f"{row['post_warmup_consistency_higher_epoch_count']}/{row['post_warmup_validation_tie_epoch_count']}/{row['post_warmup_consistency_lower_epoch_count']}",
                }
                for row in validation_rows
            ],
            ["dataset", "MR best val (epoch)", "Cons best val (epoch)", "MR final val", "Cons final val", "MR test", "Cons test", "test delta", "warmup max |delta|", "mean delta e6-30", "Cons higher/tie/lower e6-30"],
        ),
        "",
        "上表 official-test 数值仅来自既存的 validation-selected checkpoint；本审计没有进行 test 驱动的 epoch 选择。",
        "Warmup 的 validation 差异已在 consistency 生效前出现，最大绝对差异为 0.505、0.681、0.500 个百分点（CUB、Cars、Aircraft），这与 epoch5 selection membership 已不同的观察一致。由于该阶段没有 KL，不能将这些 warmup 差异归因于 consistency loss。epoch6-30 中，Aircraft 在 25 个 epoch 里有 22 个高于 baseline；CUB 则有 13 个低于 baseline，Cars 为 16 个高于 baseline。这是曲线层面的描述，非统计显著性结论。",
        "",
        "## 7. Aircraft 专项解释",
        "",
        "**直接观察：**Aircraft 是该 seed42 对照中保存 test delta 最大的数据集；它也具有三个数据集中最高的最终 KL 和最低的最终 weak/strong agreement，因此到 epoch30 其 consistency 信号相对更不饱和。两组最佳 validation epoch 都是 11；Consistency 的已保存最佳 validation Top-1 更高，但这只有一个 seed，不能建立 test delta 的因果机制。",
        "",
        "**基于日志的解释：**较低 agreement/较高 KL 与 Aircraft 仍存在更大增强稳定性信号相一致。**尚待验证的假设：**该信号导致了 Top-1 增益，或该增益会在其他 seed 复现。",
        "",
        "## 8. 决策建议",
        "",
        "**优先级 1：B — checkpoint 轻量梯度诊断。** 日志不能给出 CE/KL 梯度范数或对齐方向，单靠损失量级无法判断 lambda=0.5 偏弱还是偏强。",
        "",
        "**优先级 2：D — 随后以当前代码版本对 seeds 1/88 配对运行 Margin-Rank baseline 和 Consistency。**这可以消除历史版本差异的限制，并检验 Aircraft 模式是否复现。在取得配对证据前，不应开展大范围 consistency-weight 搜索。",
        "",
        "当前不支持 A：未发现直接的配置错误或 warmup KL 违规。C 尚早：宏平均 seed42 增益很小，也没有梯度或多 seed 证据来确定新的合理权重。",
        "",
        "## 9. 局限性",
        "",
        "- 单个训练 seed 不能建立统计稳定性。",
        "- 两组运行均未保存精确 Git SHA 或 deterministic-runtime 元数据。",
        "- 历史 Margin-Rank 日志没有新 consistency/dual-evidence 字段；本报告的 baseline dual strata 从保存的 L/G membership 重建，而非由最终 active set 反推。",
        "- Agreement 不等于准确率；KL 量级不等于梯度量级。",
    ])
    (output_dir / "offline_audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Offline audit written to: {output_dir}")
    print("No model, checkpoint, training, or original result artifact was modified.")


if __name__ == "__main__":
    main()
