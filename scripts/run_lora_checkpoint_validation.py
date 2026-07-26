"""Run validation-selected LoRA baselines on synthetic asymmetric-noise datasets.

This entry point intentionally separates model selection from the official test set.
It is for CUB/Cars/Aircraft synthetic-noise experiments, whose noise-index CSVs
contain the original clean labels needed for a fixed held-out validation split.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.checkpoint_validation import (
    PROTOCOL_NAME,
    build_or_load_fixed_validation_split,
    build_validation_safe_pgdf_reference,
    load_clean_labels_from_noise_index,
)
from gcdd.config import deep_update
from gcdd.io_utils import ensure_dir, write_csv, write_json, write_yaml
from gcdd.lora_dynamic import train_dynamic_loss_lora
from gcdd.lora_training import train_dinov2_lora
from gcdd.progress import log_stage


METHODS = ("dynamic", "jal_ce", "pgdf_auto", "pgdf_fixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dynamic, JAL-CE, PGDF-auto, and fixed-p PGDF with fixed clean validation checkpoint selection."
    )
    parser.add_argument("--input-dir", required=True, help="V1 directory containing paths.txt, labels.npy, features_*.npy, and official test files.")
    parser.add_argument("--noise-index", required=True, help="Synthetic-noise index CSV containing clean_label for the original train split.")
    parser.add_argument("--config", help="Optional YAML merged over <input-dir>/resolved_config.yaml for all methods.")
    parser.add_argument("--output-dir", help="Defaults to <input-dir>/checkpoint_validation_s<validation-seed>.")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated methods: dynamic,jal_ce,pgdf_auto,pgdf_fixed.")
    parser.add_argument("--seeds", default="1,42,88", help="Comma-separated model/dataloader seeds.")
    parser.add_argument("--validation-ratio", type=float, default=0.10, help="Fixed clean validation fraction per clean class.")
    parser.add_argument("--validation-seed", type=int, default=20250726, help="Dataset-level seed used once to create the shared validation manifest.")
    parser.add_argument("--dynamic-ratio", type=float, default=0.8, help="Class-wise dynamic small-loss keep ratio r.")
    parser.add_argument("--fixed-p", type=float, default=0.4, help="Pre-declared global prototype keep ratio for PGDF fixed-p.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Dynamic/PGDF full-pool warm-up epochs.")
    parser.add_argument("--update-interval", type=int, default=5, help="Dynamic/PGDF selection update interval.")
    parser.add_argument("--auto-high-jaccard", type=float, default=0.75)
    parser.add_argument("--auto-mid-jaccard", type=float, default=0.60)
    parser.add_argument("--auto-low-jaccard", type=float, default=0.50)
    parser.add_argument("--auto-p-high", type=float, default=0.8)
    parser.add_argument("--auto-p-mid", type=float, default=0.6)
    parser.add_argument("--auto-p-low", type=float, default=0.5)
    parser.add_argument("--auto-p-very-low", type=float, default=0.4)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map stored image root to local image root.")
    parser.add_argument("--posthoc-oracle-test", dest="posthoc_oracle_test", action="store_true", default=True, help="After fitting, evaluate every epoch state on test only for a clearly labelled oracle supplement.")
    parser.add_argument("--no-posthoc-oracle-test", dest="posthoc_oracle_test", action="store_false", help="Skip the optional oracle test curve; main validation-selected and final metrics remain available.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override LoRA training batch size.")
    parser.add_argument("--eval-batch-size", type=int, help="Override validation/test batch size.")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--lora-lr", type=float, help="Override LoRA learning rate.")
    parser.add_argument("--head-lr", type=float, help="Override classifier-head learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Override AdamW weight decay.")
    parser.add_argument("--rank", type=int, help="Override LoRA rank.")
    parser.add_argument("--alpha", type=float, help="Override LoRA alpha.")
    parser.add_argument("--dropout", type=float, help="Override LoRA dropout.")
    parser.add_argument("--target-modules", help="Override comma-separated LoRA target module patterns.")
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], help="Override scheduler.")
    parser.add_argument("--warmup-ratio", type=float, help="Override LR warm-up ratio.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override device.")
    parser.add_argument("--local-repo", help="Override local DINOv2 torch-hub repository.")
    parser.add_argument("--input-size", type=int, help="Override input size.")
    parser.add_argument("--jal-alpha", type=float, help="Override JAL-CE alpha. Defaults to YAML, then 1.0.")
    parser.add_argument("--jal-beta", type=float, help="Override JAL-CE beta. Defaults to YAML, then 1.0.")
    parser.add_argument("--jal-a", type=float, help="Override JAL-CE AMSE scale. Defaults to YAML, then 30.0.")
    parser.add_argument("--jal-eps", type=float, help="Override JAL-CE epsilon. Defaults to YAML, then 1e-8.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    noise_index = Path(args.noise_index)
    if not noise_index.exists():
        raise FileNotFoundError(f"Noise index does not exist: {noise_index}")
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / f"checkpoint_validation_s{args.validation_seed}"
    ensure_dir(output_dir)

    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)
    path_maps = parse_path_maps(args.path_map)
    cfg = load_config(input_dir, Path(args.config) if args.config else None)
    apply_lora_defaults(cfg)
    apply_overrides(cfg, args)
    validate_args(args)
    jal_params = resolve_jal_params(cfg, args)
    cfg["checkpoint_validation"] = {
        "protocol": PROTOCOL_NAME,
        "validation_ratio": float(args.validation_ratio),
        "validation_seed": int(args.validation_seed),
        "dynamic_ratio": float(args.dynamic_ratio),
        "fixed_p": float(args.fixed_p),
        "warmup_epochs": int(args.warmup_epochs),
        "update_interval": int(args.update_interval),
        "posthoc_oracle_test": bool(args.posthoc_oracle_test),
        "jal": {"alpha": jal_params["jal_alpha"], "beta": jal_params["jal_beta"], "a": jal_params["jal_a"], "eps": jal_params["jal_eps"]},
    }
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/5] Loading V1 train/test arrays and creating the shared clean validation manifest.")
    data = load_data(input_dir)
    clean_labels = load_clean_labels_from_noise_index(noise_index, data["train_paths"])
    split = build_or_load_fixed_validation_split(
        output_dir,
        data["train_paths"],
        clean_labels,
        validation_ratio=float(args.validation_ratio),
        validation_seed=int(args.validation_seed),
    )
    validation_idx = np.where(split.validation_mask)[0]
    validation_paths = [data["train_paths"][int(index)] for index in validation_idx]
    validation_labels = clean_labels[validation_idx]
    log_stage(
        f"[checkpoint-validation] training_pool={int(split.train_mask.sum())}, validation={len(validation_idx)}, "
        f"official_test={len(data['test_paths'])}; test is not evaluated during fitting."
    )

    pgdf_reference: dict[str, Any] | None = None
    if any(method.startswith("pgdf_") for method in methods):
        log_stage("[2/5] Recomputing graph budget, centroid reference, and prototype scores on the training pool only.")
        features = load_pgdf_features(input_dir, len(data["train_paths"]))
        pgdf_reference = build_validation_safe_pgdf_reference(features, data["train_labels"], split.train_mask, cfg)
        write_pgdf_reference(output_dir, data, split.train_mask, pgdf_reference)
    else:
        log_stage("[2/5] PGDF was not requested; skipping validation-safe graph reference construction.")

    auto_rule = build_auto_rule(args)
    all_logs: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    run_index: list[dict[str, Any]] = []
    log_stage("[3/5] Running methods and seeds sequentially. A failed run raises immediately; later runs are not silently accepted.")
    for method_key in methods:
        for seed in seeds:
            run_dir = output_dir / method_key / f"seed{seed}"
            ensure_dir(run_dir)
            checkpoints = run_dir / "checkpoints"
            run_cfg = copy.deepcopy(cfg)
            run_cfg["lora_train"]["seed"] = int(seed)
            if method_key == "jal_ce":
                configure_jal(run_cfg, args)
                result = train_dinov2_lora(
                    data["train_paths"],
                    data["train_labels"],
                    validation_paths,
                    validation_labels,
                    split.train_mask,
                    run_cfg,
                    method="JAL-CE-DINOv2+LoRA (full noisy training pool)",
                    seed=seed,
                    path_maps=path_maps,
                    checkpoint_path=checkpoints / "best_val.pt",
                    test_paths=data["test_paths"],
                    test_labels=data["test_labels"],
                    final_checkpoint_path=checkpoints / "last.pt",
                    last5_checkpoint_dir=checkpoints / "last5",
                    full_noisy_candidate_mask=split.train_mask,
                    checkpoint_protocol=PROTOCOL_NAME,
                    posthoc_oracle_test=bool(args.posthoc_oracle_test),
                )
                all_logs.extend(result.logs)
                result_row = {"method_key": method_key, **result.summary}
            else:
                run_cfg["loss_type"] = "ce"
                if method_key == "dynamic":
                    method_name = f"DINOv2 LoRA Dynamic small-loss r={args.dynamic_ratio:g}"
                    proto_scores = None
                    centroid_mask = None
                    proto_keep_ratio = None
                    auto_proto_keep = None
                elif method_key == "pgdf_auto":
                    if pgdf_reference is None:
                        raise RuntimeError("PGDF reference was not constructed.")
                    method_name = f"DINOv2 LoRA PGDF-auto r={args.dynamic_ratio:g}"
                    proto_scores = pgdf_reference["proto_scores"]
                    centroid_mask = pgdf_reference["centroid_reference_mask"]
                    proto_keep_ratio = None
                    auto_proto_keep = auto_rule
                else:
                    if pgdf_reference is None:
                        raise RuntimeError("PGDF reference was not constructed.")
                    method_name = f"DINOv2 LoRA PGDF fixed-p r={args.dynamic_ratio:g} p={args.fixed_p:g}"
                    proto_scores = pgdf_reference["proto_scores"]
                    centroid_mask = pgdf_reference["centroid_reference_mask"]
                    proto_keep_ratio = float(args.fixed_p)
                    auto_proto_keep = None
                result = train_dynamic_loss_lora(
                    data["train_paths"],
                    data["train_labels"],
                    validation_paths,
                    validation_labels,
                    split.train_mask,
                    run_cfg,
                    method=method_name,
                    seed=seed,
                    retention_ratio=float(args.dynamic_ratio),
                    warmup_epochs=int(args.warmup_epochs),
                    update_interval=int(args.update_interval),
                    path_maps=path_maps,
                    centroid_mask=centroid_mask,
                    proto_scores=proto_scores,
                    proto_keep_ratio=proto_keep_ratio,
                    auto_proto_keep=auto_proto_keep,
                    checkpoint_path=checkpoints / "best_val.pt",
                    test_paths=data["test_paths"],
                    test_labels=data["test_labels"],
                    final_checkpoint_path=checkpoints / "last.pt",
                    last5_checkpoint_dir=checkpoints / "last5",
                    checkpoint_protocol=PROTOCOL_NAME,
                    posthoc_oracle_test=bool(args.posthoc_oracle_test),
                )
                all_logs.extend(result.logs)
                all_updates.extend(result.update_rows)
                write_csv(run_dir / "selection_rows.csv", result.selection_rows, selection_fields())
                write_csv(run_dir / "selection_updates.csv", result.update_rows, update_fields())
                write_csv(run_dir / "selection_per_class.csv", result.per_class_rows, per_class_fields())
                result_row = {"method_key": method_key, **result.summary}
            all_results.append(result_row)
            write_json(run_dir / "result.json", result_row)
            run_index.append(
                {
                    "method_key": method_key,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "best_val_checkpoint": str(checkpoints / "best_val.pt"),
                    "last_checkpoint": str(checkpoints / "last.pt"),
                    "status": "complete",
                }
            )

    log_stage("[4/5] Writing validation-selected result tables.")
    write_csv(output_dir / "train_log.csv", all_logs, train_log_fields())
    write_csv(output_dir / "checkpoint_validation_results.csv", all_results, result_fields())
    write_csv(output_dir / "checkpoint_validation_updates.csv", all_updates, update_fields())
    write_csv(output_dir / "run_index.csv", run_index, ["method_key", "seed", "run_dir", "best_val_checkpoint", "last_checkpoint", "status"])
    summary = summarize_methods(all_results)
    write_csv(output_dir / "checkpoint_validation_summary.csv", summary, summary_fields())
    write_json(
        output_dir / "checkpoint_validation_summary.json",
        {
            "protocol": PROTOCOL_NAME,
            "input_dir": str(input_dir),
            "noise_index": str(noise_index),
            "methods": methods,
            "seeds": seeds,
            "validation_manifest": split.metadata,
            "posthoc_oracle_test": bool(args.posthoc_oracle_test),
            "results": all_results,
            "summary": summary,
        },
    )
    write_run_summary(output_dir / "run_summary.md", args, split.metadata, summary)
    log_stage(f"[5/5] Completed validation-selected checkpoint experiment: {output_dir}")


def load_data(input_dir: Path) -> dict[str, Any]:
    required = [input_dir / name for name in ("paths.txt", "labels.npy", "eval_paths.txt", "eval_labels.npy")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing V1 inputs: {missing}")
    train_paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    test_paths = (input_dir / "eval_paths.txt").read_text(encoding="utf-8").splitlines()
    train_labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    test_labels = np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str)
    if len(train_paths) != len(train_labels) or len(test_paths) != len(test_labels):
        raise ValueError("Path and label array lengths do not match.")
    return {"train_paths": train_paths, "train_labels": train_labels, "test_paths": test_paths, "test_labels": test_labels}


def load_pgdf_features(input_dir: Path, expected_rows: int) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for name in ("cls", "gap", "top"):
        path = input_dir / f"features_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"PGDF requires {path}.")
        values = np.load(path, mmap_mode="r")
        if values.shape[0] != expected_rows:
            raise ValueError(f"{path} has {values.shape[0]} rows, expected {expected_rows}.")
        features[name] = values
    return features


def write_pgdf_reference(output_dir: Path, data: dict[str, Any], training_pool_mask: np.ndarray, reference: dict[str, Any]) -> None:
    rows = []
    for index in np.where(training_pool_mask)[0]:
        rows.append(
            {
                "index": int(index),
                "path": data["train_paths"][int(index)],
                "noisy_label": str(data["train_labels"][int(index)]),
                "prototype_score": float(reference["proto_scores"][int(index)]),
                "gcdd_clean": "yes" if reference["gcdd_clean_mask"][int(index)] else "no",
                "centroid_reference": "yes" if reference["centroid_reference_mask"][int(index)] else "no",
            }
        )
    write_csv(output_dir / "pgdf_training_pool_reference.csv", rows, ["index", "path", "noisy_label", "prototype_score", "gcdd_clean", "centroid_reference"])
    per_class = [
        {"noisy_label": label, "gcdd_clean_budget_q_c": int(count)}
        for label, count in sorted(reference["per_class_keep_counts"].items())
    ]
    write_csv(output_dir / "pgdf_training_pool_budget.csv", per_class, ["noisy_label", "gcdd_clean_budget_q_c"])


def load_config(input_dir: Path, override_path: Path | None) -> dict[str, Any]:
    path = input_dir / "resolved_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"V1 resolved config is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if override_path is not None:
        with override_path.open("r", encoding="utf-8") as handle:
            deep_update(cfg, yaml.safe_load(handle) or {})
    return cfg


def apply_lora_defaults(cfg: dict[str, Any]) -> None:
    cfg.setdefault("feature", {})
    cfg["feature"].setdefault("backend", "dinov2_vitb14")
    cfg["feature"].setdefault("device", "auto")
    cfg["feature"].setdefault("input_size", 448)
    cfg["feature"].setdefault("local_repo", "")
    cfg.setdefault("lora", {})
    cfg["lora"].setdefault("rank", 8)
    cfg["lora"].setdefault("alpha", 16.0)
    cfg["lora"].setdefault("dropout", 0.05)
    cfg["lora"].setdefault("target_modules", "qkv")
    cfg.setdefault("lora_train", {})
    defaults = {
        "epochs": 30,
        "batch_size": 80,
        "eval_batch_size": 160,
        "num_workers": 4,
        "pin_memory": True,
        "lora_lr": 1.0e-4,
        "head_lr": 1.0e-3,
        "weight_decay": 0.05,
        "scheduler": "cosine",
        "warmup_ratio": 0.1,
        "amp": True,
    }
    for key, value in defaults.items():
        cfg["lora_train"].setdefault(key, value)


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    feature_cfg = cfg["feature"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["lora_train"]
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "lora_lr": args.lora_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
    }
    for key, value in overrides.items():
        if value is not None:
            train_cfg[key] = value
    for key, value in {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout, "target_modules": args.target_modules}.items():
        if value is not None:
            lora_cfg[key] = value
    for key, value in {"device": args.device, "local_repo": args.local_repo, "input_size": args.input_size}.items():
        if value is not None:
            feature_cfg[key] = value


def configure_jal(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    params = resolve_jal_params(cfg, args)
    cfg["loss_type"] = "jal_ce"
    cfg.update(params)


def resolve_jal_params(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    return {
        "jal_alpha": float(args.jal_alpha if args.jal_alpha is not None else cfg.get("jal_alpha", 1.0)),
        "jal_beta": float(args.jal_beta if args.jal_beta is not None else cfg.get("jal_beta", 1.0)),
        "jal_a": float(args.jal_a if args.jal_a is not None else cfg.get("jal_a", 30.0)),
        "jal_eps": float(args.jal_eps if args.jal_eps is not None else cfg.get("jal_eps", 1.0e-8)),
    }


def build_auto_rule(args: argparse.Namespace) -> dict[str, float]:
    rule = {
        "high_jaccard": float(args.auto_high_jaccard),
        "mid_jaccard": float(args.auto_mid_jaccard),
        "low_jaccard": float(args.auto_low_jaccard),
        "p_high": float(args.auto_p_high),
        "p_mid": float(args.auto_p_mid),
        "p_low": float(args.auto_p_low),
        "p_very_low": float(args.auto_p_very_low),
    }
    if not rule["high_jaccard"] >= rule["mid_jaccard"] >= rule["low_jaccard"]:
        raise ValueError("Auto-p Jaccard thresholds must satisfy high >= mid >= low.")
    if not all(0.0 < value <= 1.0 for value in rule.values()):
        raise ValueError("Auto-p thresholds and keep ratios must be in (0, 1].")
    return rule


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.dynamic_ratio <= 1.0 or not 0.0 < args.fixed_p <= 1.0:
        raise ValueError("--dynamic-ratio and --fixed-p must be in (0, 1].")
    if args.warmup_epochs < 0 or args.update_interval <= 0:
        raise ValueError("--warmup-epochs must be non-negative and --update-interval must be positive.")


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one method is required.")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed methods: {METHODS}")
    if len(set(methods)) != len(methods):
        raise ValueError("--methods contains duplicates.")
    return methods


def parse_seeds(raw: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated integers.") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain one or more unique values.")
    return seeds


def parse_path_maps(items: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --path-map '{item}'; expected OLD=NEW.")
        old, new = item.split("=", 1)
        if not old or not new:
            raise ValueError(f"Invalid --path-map '{item}'; both OLD and NEW are required.")
        parsed.append((old, new))
    return parsed


def summarize_methods(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method_key in sorted({str(row["method_key"]) for row in rows}):
        method_rows = [row for row in rows if row["method_key"] == method_key]
        item: dict[str, Any] = {"method_key": method_key, "method": method_rows[0]["method"], "num_seeds": len(method_rows)}
        for field in (
            "best_val_top1",
            "validation_selected_test_top1",
            "final_test_top1",
            "last5_test_mean",
            "oracle_best_test_top1",
            "oracle_best_to_final_drop",
        ):
            values = np.asarray([float(row[field]) for row in method_rows if row.get(field, "") != ""], dtype=np.float64)
            item[f"mean_{field}"] = float(values.mean()) if len(values) else ""
            item[f"std_{field}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else ""
        summary.append(item)
    return summary


def write_run_summary(path: Path, args: argparse.Namespace, manifest: dict[str, Any], summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Validation-Selected Checkpoint Run",
        "",
        f"- Protocol: `{PROTOCOL_NAME}`.",
        f"- Shared fixed clean validation: {manifest['validation_samples']}/{manifest['source_train_samples']} samples, ratio={manifest['validation_ratio']}, seed={manifest['validation_seed']}.",
        "- During fitting, checkpoint selection uses validation Top-1 only. The official test set is evaluated after fitting.",
        f"- Post-hoc oracle test curve: {'enabled (supplementary only)' if args.posthoc_oracle_test else 'disabled'}.",
        f"- Fixed PGDF p: {args.fixed_p}; this is a pre-declared global value, not selected from this run's test results.",
        "",
        "| method | seeds | val-selected test Top-1 | final test Top-1 | last-5 test mean | oracle best test Top-1 | oracle best-to-final drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['num_seeds']} | {format_pct(row['mean_validation_selected_test_top1'])} | "
            f"{format_pct(row['mean_final_test_top1'])} | {format_pct(row['mean_last5_test_mean'])} | "
            f"{format_pct(row['mean_oracle_best_test_top1'])} | {format_pct(row['mean_oracle_best_to_final_drop'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_pct(value: Any) -> str:
    return "" if value == "" else f"{100.0 * float(value):.2f}"


def train_log_fields() -> list[str]:
    return [
        "method", "seed", "epoch", "lr_lora", "lr_head", "loss", "train_loss", "top1", "top5", "val_top1", "val_top5",
        "best_top1", "best_epoch", "train_samples", "candidate_samples", "selected_ratio", "eval_samples", "trainable_params", "total_params",
        "loss_type", "jal_alpha", "jal_beta", "jal_a", "jal_eps", "selection_mode",
    ]


def result_fields() -> list[str]:
    return [
        "method_key", "method", "seed", "checkpoint_protocol", "train_samples", "eval_samples", "validation_samples", "test_samples",
        "best_val_epoch", "best_val_top1", "best_val_top5", "validation_selected_test_top1", "validation_selected_test_top5",
        "final_test_top1", "final_test_top5", "last5_test_mean", "last5_test_std", "oracle_best_test_epoch",
        "oracle_best_test_top1", "oracle_best_test_top5", "oracle_best_to_final_drop", "retention_ratio", "proto_keep_ratio",
        "auto_proto_keep", "auto_proto_jaccard", "warmup_epochs", "update_interval", "candidate_samples", "final_selected_samples",
        "selection_updates", "loss_type", "jal_alpha", "jal_beta", "jal_a", "jal_eps", "selection_mode",
    ]


def summary_fields() -> list[str]:
    return [
        "method_key", "method", "num_seeds", "mean_best_val_top1", "std_best_val_top1",
        "mean_validation_selected_test_top1", "std_validation_selected_test_top1", "mean_final_test_top1", "std_final_test_top1",
        "mean_last5_test_mean", "std_last5_test_mean", "mean_oracle_best_test_top1", "std_oracle_best_test_top1",
        "mean_oracle_best_to_final_drop", "std_oracle_best_to_final_drop",
    ]


def selection_fields() -> list[str]:
    return ["method", "seed", "retention_ratio", "proto_keep_ratio", "epoch", "index", "path", "web_label", "loss", "confidence", "proto_score", "loss_selected", "proto_pass", "state"]


def update_fields() -> list[str]:
    return [
        "method", "seed", "retention_ratio", "proto_keep_ratio", "auto_proto_jaccard", "epoch", "num_candidates", "num_loss_selected",
        "num_proto_pass", "num_selected", "proto_reject_count", "selected_ratio", "mean_loss_selected", "mean_loss_unselected",
        "mean_loss_proto_rejected", "mean_proto_selected", "mean_proto_unselected", "overlap_with_previous_selection", "overlap_with_centroid",
    ]


def per_class_fields() -> list[str]:
    return [
        "method", "seed", "retention_ratio", "proto_keep_ratio", "epoch", "web_label", "total_count", "loss_selected_count",
        "proto_pass_count", "selected_count", "proto_reject_count", "selected_ratio", "mean_loss_selected", "mean_loss_unselected",
        "mean_loss_proto_rejected", "mean_proto_selected", "mean_proto_unselected",
    ]


if __name__ == "__main__":
    main()
