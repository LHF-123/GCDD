"""Run validation-selected LoRA baselines on synthetic asymmetric-noise datasets.

This entry point intentionally separates model selection from the official test set.
It is for CUB/Cars/Aircraft synthetic-noise experiments, whose noise-index CSVs
contain the original clean labels needed for a fixed held-out validation split.
"""

from __future__ import annotations

import argparse
import copy
import json
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
    build_validation_safe_static_selections,
    load_clean_labels_from_noise_index,
)
from gcdd.budget_matching import ClassBudgetSchedule, load_pgdf_class_budget_schedule
from gcdd.config import deep_update
from gcdd.io_utils import ensure_dir, write_csv, write_json, write_yaml
from gcdd.lora_dynamic import selection_update_epochs, train_dynamic_loss_lora
from gcdd.lora_noisy_baselines import train_coteaching_lora, train_jocor_lora
from gcdd.lora_training import train_dinov2_lora
from gcdd.progress import log_stage


METHODS = (
    "all_noisy",
    "full_gcdd",
    "centroid",
    "both_only",
    "gcdd_proto",
    "fine",
    "dynamic_r08",
    "dynamic_r09",
    "jal_ce",
    "coteaching",
    "jocor",
    "pgdf_auto",
    "pgdf_fixed",
)
ABLATION_METHODS = ("proto_only", "dynamic_budget_matched")
STATIC_METHODS = ("full_gcdd", "centroid", "proto_only", "both_only", "gcdd_proto", "fine")
DYNAMIC_METHODS = ("dynamic", "dynamic_r08", "dynamic_r09", "dynamic_budget_matched", "pgdf_auto", "pgdf_fixed")
LEGACY_METHOD_ALIASES = {"dynamic": "dynamic_r08"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all 13 comparison methods with fixed clean validation checkpoint selection."
    )
    parser.add_argument("--input-dir", required=True, help="V1 directory containing paths.txt, labels.npy, features_*.npy, and official test files.")
    parser.add_argument("--noise-index", required=True, help="Synthetic-noise index CSV containing clean_label for the original train split.")
    parser.add_argument("--config", help="Optional YAML merged over <input-dir>/resolved_config.yaml for all methods.")
    parser.add_argument("--output-dir", help="Defaults to <input-dir>/checkpoint_validation_all_methods_s<validation-seed>.")
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Use 'all' for the 13 primary methods, or an explicit subset. "
            "proto_only and dynamic_budget_matched are explicit ablations; dynamic aliases dynamic_r08."
        ),
    )
    parser.add_argument("--seeds", default="1,42,88", help="Comma-separated model/dataloader seeds.")
    parser.add_argument("--validation-ratio", type=float, default=0.10, help="Fixed clean validation fraction per clean class.")
    parser.add_argument("--validation-seed", type=int, default=20250726, help="Dataset-level seed used once to create the shared validation manifest.")
    parser.add_argument("--dynamic-ratio", type=float, default=0.8, help="Class-wise dynamic small-loss keep ratio r.")
    parser.add_argument("--fixed-p", type=float, default=0.4, help="Pre-declared global prototype keep ratio for PGDF fixed-p.")
    parser.add_argument(
        "--pgdf-budget-root",
        help=(
            "Reference directory containing seed*/selection_per_class.csv from PGDF fixed-p. "
            "Required only by dynamic_budget_matched."
        ),
    )
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
    parser.add_argument("--posthoc-oracle-test", dest="posthoc_oracle_test", action="store_true", default=False, help="After fitting, evaluate every epoch state on test only for a clearly labelled oracle supplement.")
    parser.add_argument("--no-posthoc-oracle-test", dest="posthoc_oracle_test", action="store_false", help="Skip the optional oracle test curve; main validation-selected and final metrics remain available.")
    parser.add_argument(
        "--official-test-selected-only",
        action="store_true",
        help=(
            "Evaluate official test only once on best_val.pt. Required for the explicit "
            "proto_only and dynamic_budget_matched ablations."
        ),
    )
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
    parser.add_argument("--dual-batch-size", type=int, default=16, help="Per-device batch size for two-branch Co-teaching/JoCoR runs.")
    parser.add_argument("--dual-grad-accum-steps", type=int, default=2, help="Gradient accumulation steps for two-branch runs.")
    parser.add_argument("--jocor-lambda", type=float, default=0.1, help="JoCoR symmetric-KL weight.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    noise_index = Path(args.noise_index)
    if not noise_index.exists():
        raise FileNotFoundError(f"Noise index does not exist: {noise_index}")
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / f"checkpoint_validation_all_methods_s{args.validation_seed}"
    legacy_output_dir = input_dir / f"checkpoint_validation_s{args.validation_seed}"
    if output_dir.resolve() == legacy_output_dir.resolve() and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into legacy raw-output directory {output_dir}. "
            "Use a new --output-dir such as checkpoint_validation_all_methods_s<validation-seed>."
        )
    ensure_dir(output_dir)

    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)
    validate_official_test_request(methods, bool(args.official_test_selected_only))
    pgdf_budget_root = Path(args.pgdf_budget_root) if args.pgdf_budget_root else None
    validate_budget_method_request(methods, pgdf_budget_root, seeds)
    preflight_run_dirs(output_dir, methods, seeds)
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
        "official_test_selected_only": bool(args.official_test_selected_only),
        "pgdf_budget_root": str(pgdf_budget_root) if pgdf_budget_root is not None else "",
        "jal": {"alpha": jal_params["jal_alpha"], "beta": jal_params["jal_beta"], "a": jal_params["jal_a"], "eps": jal_params["jal_eps"]},
        "two_branch": {
            "remember_mode": "fixed",
            "remember_rate": 0.8,
            "batch_size": int(args.dual_batch_size),
            "grad_accum_steps": int(args.dual_grad_accum_steps),
            "jocor_lambda": float(args.jocor_lambda),
            "selection_metric": "arithmetic mean of branch A/B validation Top-1; not ensemble prediction",
        },
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
    if pgdf_budget_root is not None:
        validate_budget_source_manifest(pgdf_budget_root, split.metadata)
    validation_idx = np.where(split.validation_mask)[0]
    validation_paths = [data["train_paths"][int(index)] for index in validation_idx]
    validation_labels = clean_labels[validation_idx]
    log_stage(
        f"[checkpoint-validation] training_pool={int(split.train_mask.sum())}, validation={len(validation_idx)}, "
        f"official_test={len(data['test_paths'])}; test is not evaluated during fitting."
    )

    canonical_methods = [canonical_method_key(method) for method in methods]
    pgdf_reference: dict[str, Any] | None = None
    static_selections: dict[str, dict[str, Any]] = {}
    graph_methods_requested = any(method in STATIC_METHODS or method.startswith("pgdf_") for method in canonical_methods)
    if graph_methods_requested:
        log_stage("[2/5] Recomputing graph/static references on the training pool only.")
        features = load_pgdf_features(input_dir, len(data["train_paths"]))
        pgdf_reference = build_validation_safe_pgdf_reference(features, data["train_labels"], split.train_mask, cfg)
        write_pgdf_reference(output_dir, data, split.train_mask, pgdf_reference)
        if any(method in STATIC_METHODS for method in canonical_methods):
            static_selections = build_validation_safe_static_selections(
                features,
                data["train_labels"],
                split.train_mask,
                cfg,
                pgdf_reference=pgdf_reference,
                proto_keep_ratio=float(args.fixed_p),
                fine_keep_ratio=0.6,
                fine_center=False,
            )
    else:
        log_stage("[2/5] No graph/static method requested; skipping validation-safe reference construction.")

    auto_rule = build_auto_rule(args)
    all_logs: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    all_updates: list[dict[str, Any]] = []
    run_index: list[dict[str, Any]] = []
    log_stage("[3/5] Running methods and seeds sequentially. A failed run raises immediately; later runs are not silently accepted.")
    for method_key in methods:
        canonical_key = canonical_method_key(method_key)
        for seed in seeds:
            run_dir = output_dir / method_key / f"seed{seed}"
            require_fresh_run_dir(run_dir)
            ensure_dir(run_dir)
            checkpoints = run_dir / "checkpoints"
            run_cfg = copy.deepcopy(cfg)
            run_cfg["lora_train"]["seed"] = int(seed)
            selection_record = ""
            if canonical_key in {"all_noisy", "jal_ce", *STATIC_METHODS}:
                if canonical_key == "jal_ce":
                    configure_jal(run_cfg, args)
                    method_name = "JAL-CE-DINOv2+LoRA (full noisy training pool)"
                    train_mask = split.train_mask
                    selection_mode = "full_noisy_training_pool"
                    static_score = np.full(len(train_mask), np.nan, dtype=np.float32)
                elif canonical_key in STATIC_METHODS:
                    run_cfg["loss_type"] = "ce"
                    static_info = static_selections[canonical_key]
                    train_mask = np.asarray(static_info["mask"], dtype=bool)
                    selection_mode = str(static_info["selection_mode"])
                    static_score = np.asarray(static_info["score"], dtype=np.float32)
                    method_name = static_method_name(canonical_key, proto_keep_ratio=float(args.fixed_p))
                else:
                    run_cfg["loss_type"] = "ce"
                    method_name = "DINOv2+LoRA all noisy CE (full noisy training pool)"
                    train_mask = split.train_mask
                    selection_mode = "full_noisy_training_pool"
                    static_score = np.full(len(train_mask), np.nan, dtype=np.float32)
                result = train_dinov2_lora(
                    data["train_paths"],
                    data["train_labels"],
                    validation_paths,
                    validation_labels,
                    train_mask,
                    run_cfg,
                    method=method_name,
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
                    official_test_selected_only=bool(args.official_test_selected_only),
                )
                selection_record = str(run_dir / "static_selection.csv")
                write_static_selection(
                    Path(selection_record), data, split.train_mask, train_mask, static_score, selection_mode
                )
                result_row = finalize_result_row(
                    method_key,
                    result.summary,
                    selection_mode,
                    selected_count=int(train_mask.sum()),
                    candidate_count=int(split.train_mask.sum()),
                )
                if canonical_key == "proto_only":
                    result_row["proto_keep_ratio"] = float(args.fixed_p)
            elif canonical_key in DYNAMIC_METHODS:
                run_cfg["loss_type"] = "ce"
                budget_schedule: ClassBudgetSchedule | None = None
                scheduler_retention_ratio: float | None = None
                if canonical_key in {"dynamic_r08", "dynamic_r09"}:
                    retention_ratio = resolve_retention_ratio(canonical_key, float(args.dynamic_ratio))
                    method_name = f"DINOv2 LoRA Dynamic small-loss r={retention_ratio:g}"
                    proto_scores = None
                    centroid_mask = None
                    proto_keep_ratio = None
                    auto_proto_keep = None
                    selection_mode = f"dynamic_training_pool_small_loss_r{retention_ratio:g}"
                elif canonical_key == "dynamic_budget_matched":
                    if pgdf_budget_root is None:
                        raise RuntimeError("PGDF budget root was not configured.")
                    budget_path = pgdf_budget_root / f"seed{seed}" / "selection_per_class.csv"
                    expected_updates = selection_update_epochs(
                        int(run_cfg["lora_train"]["epochs"]),
                        int(args.warmup_epochs),
                        int(args.update_interval),
                    )
                    budget_schedule = load_pgdf_class_budget_schedule(
                        budget_path,
                        expected_seed=seed,
                        expected_update_epochs=expected_updates,
                        labels=data["train_labels"],
                        candidate_mask=split.train_mask,
                    )
                    validate_budget_source_hyperparameters(
                        budget_schedule,
                        dynamic_ratio=float(args.dynamic_ratio),
                        fixed_p=float(args.fixed_p),
                    )
                    retention_ratio = float(budget_schedule.source_retention_ratio)
                    scheduler_retention_ratio = min(
                        budget_schedule.source_retention_ratio,
                        budget_schedule.source_proto_keep_ratio,
                    )
                    method_name = (
                        "DINOv2 LoRA Dynamic small-loss matched to PGDF class budget "
                        f"r={budget_schedule.source_retention_ratio:g} "
                        f"p={budget_schedule.source_proto_keep_ratio:g}"
                    )
                    proto_scores = None
                    centroid_mask = None
                    proto_keep_ratio = None
                    auto_proto_keep = None
                    selection_mode = "dynamic_small_loss_pgdf_per_class_budget_matched"
                    write_budget_source(run_dir, budget_schedule)
                elif canonical_key == "pgdf_auto":
                    if pgdf_reference is None:
                        raise RuntimeError("PGDF reference was not constructed.")
                    retention_ratio = resolve_retention_ratio(canonical_key, float(args.dynamic_ratio))
                    method_name = f"DINOv2 LoRA PGDF-auto r={args.dynamic_ratio:g}"
                    proto_scores = pgdf_reference["proto_scores"]
                    centroid_mask = pgdf_reference["centroid_reference_mask"]
                    proto_keep_ratio = None
                    auto_proto_keep = auto_rule
                    selection_mode = "pgdf_auto_training_pool_dynamic_loss_and_prototype"
                else:
                    if pgdf_reference is None:
                        raise RuntimeError("PGDF reference was not constructed.")
                    retention_ratio = resolve_retention_ratio(canonical_key, float(args.dynamic_ratio))
                    method_name = f"DINOv2 LoRA PGDF fixed-p r={args.dynamic_ratio:g} p={args.fixed_p:g}"
                    proto_scores = pgdf_reference["proto_scores"]
                    centroid_mask = pgdf_reference["centroid_reference_mask"]
                    proto_keep_ratio = float(args.fixed_p)
                    auto_proto_keep = None
                    selection_mode = "pgdf_fixed_training_pool_dynamic_loss_and_prototype"
                result = train_dynamic_loss_lora(
                    data["train_paths"],
                    data["train_labels"],
                    validation_paths,
                    validation_labels,
                    split.train_mask,
                    run_cfg,
                    method=method_name,
                    seed=seed,
                    retention_ratio=retention_ratio,
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
                    class_budget_schedule=budget_schedule.budgets if budget_schedule is not None else None,
                    scheduler_retention_ratio=scheduler_retention_ratio,
                    official_test_selected_only=bool(args.official_test_selected_only),
                )
                if budget_schedule is not None:
                    verify_observed_budget_match(result.per_class_rows, budget_schedule)
                all_updates.extend(result.update_rows)
                write_csv(run_dir / "selection_rows.csv", result.selection_rows, selection_fields())
                write_csv(run_dir / "selection_updates.csv", result.update_rows, update_fields())
                write_csv(run_dir / "selection_per_class.csv", result.per_class_rows, per_class_fields())
                selection_record = str(run_dir / "selection_rows.csv")
                result_row = finalize_result_row(
                    method_key,
                    result.summary,
                    selection_mode,
                    selected_count=int(result.summary["final_selected_samples"]),
                    candidate_count=int(result.summary["candidate_samples"]),
                )
                if budget_schedule is not None:
                    result_row.update(
                        {
                            "budget_source_path": budget_schedule.source_path,
                            "budget_source_sha256": budget_schedule.source_sha256,
                            "budget_source_retention_ratio": budget_schedule.source_retention_ratio,
                            "budget_source_proto_keep_ratio": budget_schedule.source_proto_keep_ratio,
                            "budget_match_verified": "yes",
                        }
                    )
            elif canonical_key in {"coteaching", "jocor"}:
                run_cfg["loss_type"] = "ce"
                run_cfg["lora_train"]["batch_size"] = int(args.dual_batch_size)
                method_name = (
                    "Co-teaching-DINOv2+LoRA fixed r=0.8 (two-branch validation mean)"
                    if canonical_key == "coteaching"
                    else f"JoCoR-DINOv2+LoRA r=0.8 lambda={args.jocor_lambda:g} (two-branch validation mean)"
                )
                dual_kwargs: dict[str, Any] = {
                    "train_paths": data["train_paths"],
                    "train_labels": data["train_labels"],
                    "eval_paths": validation_paths,
                    "eval_labels": validation_labels,
                    "train_mask": split.train_mask,
                    "cfg": run_cfg,
                    "method": method_name,
                    "seed": seed,
                    "remember_mode": "fixed",
                    "remember_rate": 0.8,
                    "final_remember_rate": 0.8,
                    "warmup_epochs": int(args.warmup_epochs),
                    "grad_accum_steps": int(args.dual_grad_accum_steps),
                    "path_maps": path_maps,
                    # Synthetic clean/noisy GT is intentionally not supplied to
                    # either training decision path in the unified protocol.
                    "gt_clean_mask": None,
                    "checkpoint_path": checkpoints / "best_val.pt",
                    "test_paths": data["test_paths"],
                    "test_labels": data["test_labels"],
                    "final_checkpoint_path": checkpoints / "last.pt",
                    "last5_checkpoint_dir": checkpoints / "last5",
                    "checkpoint_protocol": PROTOCOL_NAME,
                    "posthoc_oracle_test": bool(args.posthoc_oracle_test),
                }
                if canonical_key == "jocor":
                    dual_kwargs["lambda_cor"] = float(args.jocor_lambda)
                    result = train_jocor_lora(**dual_kwargs)
                else:
                    result = train_coteaching_lora(**dual_kwargs)
                selection_record = str(run_dir / "selection_history.csv")
                write_csv(Path(selection_record), result.logs, dual_selection_fields())
                result_row = finalize_result_row(
                    method_key,
                    result.summary,
                    "two_branch_mean_validation_top1",
                    selected_count=float(result.summary["final_selected_count"]),
                    candidate_count=int(split.train_mask.sum()),
                )
            else:
                raise RuntimeError(f"Unhandled method: {method_key}")

            all_logs.extend(result.logs)
            write_csv(run_dir / "train_log.csv", result.logs, train_log_fields())
            write_json(
                run_dir / "selection_policy.json",
                {
                    "method_key": method_key,
                    "selection_mode": result_row["selection_mode"],
                    "selected_count": result_row["selected_count"],
                    "selection_ratio": result_row["selection_ratio"],
                    "validation_samples_are_training_candidates": False,
                    "official_test_used_for_checkpoint_selection": False,
                    "budget_source_path": result_row.get("budget_source_path", ""),
                    "budget_source_sha256": result_row.get("budget_source_sha256", ""),
                    "budget_match_verified": result_row.get("budget_match_verified", ""),
                },
            )
            all_results.append(result_row)
            write_json(run_dir / "result.json", result_row)
            run_index.append(
                {
                    "method_key": method_key,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "best_val_checkpoint": str(checkpoints / "best_val.pt"),
                    "last_checkpoint": str(checkpoints / "last.pt"),
                    "selection_record": selection_record,
                    "status": "complete",
                }
            )

    log_stage("[4/5] Writing validation-selected result tables.")
    write_csv(output_dir / "train_log.csv", all_logs, train_log_fields())
    write_csv(output_dir / "checkpoint_validation_results.csv", all_results, result_fields())
    write_csv(output_dir / "checkpoint_validation_updates.csv", all_updates, update_fields())
    write_csv(output_dir / "run_index.csv", run_index, ["method_key", "seed", "run_dir", "best_val_checkpoint", "last_checkpoint", "selection_record", "status"])
    summary = summarize_methods(all_results)
    write_csv(output_dir / "checkpoint_validation_summary.csv", summary, summary_fields())
    write_json(
        output_dir / "checkpoint_validation_summary.json",
        {
            "protocol": PROTOCOL_NAME,
            "input_dir": str(input_dir),
            "noise_index": str(noise_index),
            "pgdf_budget_root": str(pgdf_budget_root) if pgdf_budget_root is not None else "",
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


def write_static_selection(
    path: Path,
    data: dict[str, Any],
    training_pool_mask: np.ndarray,
    selected_mask: np.ndarray,
    scores: np.ndarray,
    selection_mode: str,
) -> None:
    """Write an original-index-aligned static selection audit."""
    rows = []
    for index, sample_path in enumerate(data["train_paths"]):
        in_pool = bool(training_pool_mask[index])
        score = float(scores[index]) if in_pool and np.isfinite(scores[index]) else ""
        rows.append(
            {
                "index": int(index),
                "path": sample_path,
                "noisy_label": str(data["train_labels"][index]),
                "partition": "training_pool" if in_pool else "validation",
                "eligible": "yes" if in_pool else "no",
                "selected": "yes" if selected_mask[index] else "no",
                "state": "clean" if selected_mask[index] else "ignored",
                "score": score,
                "selection_mode": selection_mode,
            }
        )
    write_csv(
        path,
        rows,
        ["index", "path", "noisy_label", "partition", "eligible", "selected", "state", "score", "selection_mode"],
    )


def static_method_name(method_key: str, *, proto_keep_ratio: float = 0.4) -> str:
    names = {
        "full_gcdd": "DINOv2 LoRA + Full GCDD-clean (training-pool-only)",
        "centroid": "DINOv2 LoRA + Centroid filtering (training-pool-only GCDD budget)",
        "proto_only": f"DINOv2 LoRA + Prototype gate only p={proto_keep_ratio:g} (training-pool-only)",
        "both_only": "DINOv2 LoRA + both only (training-pool-only intersection)",
        "gcdd_proto": "DINOv2 LoRA + GCDD+Proto (training-pool-only)",
        "fine": "FINE-DINOv2 feature nocenter p=0.6 (training-pool-only)",
    }
    return names[method_key]


def finalize_result_row(
    method_key: str,
    summary: dict[str, Any],
    selection_mode: str,
    *,
    selected_count: int | float,
    candidate_count: int,
) -> dict[str, Any]:
    row = {"method_key": method_key, **summary}
    row["selection_mode"] = selection_mode
    row["selected_count"] = selected_count
    row["selection_ratio"] = float(selected_count) / max(1, candidate_count)
    row.setdefault("candidate_samples", int(candidate_count))
    return row


def validate_budget_method_request(
    methods: list[str],
    pgdf_budget_root: Path | None,
    seeds: list[int],
) -> None:
    requested = "dynamic_budget_matched" in {canonical_method_key(method) for method in methods}
    if requested and pgdf_budget_root is None:
        raise ValueError("dynamic_budget_matched requires --pgdf-budget-root.")
    if not requested:
        return
    if not pgdf_budget_root.is_dir():
        raise FileNotFoundError(f"PGDF budget root does not exist: {pgdf_budget_root}")
    missing = [
        str(pgdf_budget_root / f"seed{seed}" / "selection_per_class.csv")
        for seed in seeds
        if not (pgdf_budget_root / f"seed{seed}" / "selection_per_class.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing PGDF per-class budget files: {missing}")


def validate_budget_source_manifest(pgdf_budget_root: Path, current_manifest: dict[str, Any]) -> None:
    """Require the budget source to use the exact fixed validation partition."""
    source_path = pgdf_budget_root.parent / "validation_manifest.json"
    if not source_path.exists():
        raise FileNotFoundError(f"PGDF budget source validation manifest is missing: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    exact_fields = (
        "protocol",
        "validation_seed",
        "train_paths_sha256",
        "source_train_samples",
        "training_pool_samples",
        "validation_samples",
        "stratification_label",
        "validation_label",
    )
    mismatches = [
        field for field in exact_fields if source.get(field) != current_manifest.get(field)
    ]
    if not np.isclose(
        float(source.get("validation_ratio", -1.0)),
        float(current_manifest.get("validation_ratio", -2.0)),
    ):
        mismatches.append("validation_ratio")
    if mismatches:
        raise ValueError(
            "PGDF budget source does not use the current fixed validation manifest; "
            f"mismatched fields: {sorted(set(mismatches))}."
        )


def validate_budget_source_hyperparameters(
    schedule: ClassBudgetSchedule,
    *,
    dynamic_ratio: float,
    fixed_p: float,
) -> None:
    if not np.isclose(schedule.source_retention_ratio, dynamic_ratio):
        raise ValueError(
            f"PGDF budget r={schedule.source_retention_ratio} does not match "
            f"--dynamic-ratio={dynamic_ratio}."
        )
    if not np.isclose(schedule.source_proto_keep_ratio, fixed_p):
        raise ValueError(
            f"PGDF budget p={schedule.source_proto_keep_ratio} does not match --fixed-p={fixed_p}."
        )


def write_budget_source(run_dir: Path, schedule: ClassBudgetSchedule) -> None:
    """Freeze only PGDF class counts, never its selected sample identities or scores."""
    write_csv(
        run_dir / "budget_source.csv",
        schedule.rows,
        ["seed", "epoch", "noisy_label", "total_count", "selected_count", "selected_ratio"],
    )
    write_json(
        run_dir / "budget_source.json",
        {
            "source_path": schedule.source_path,
            "source_sha256": schedule.source_sha256,
            "source_method": "pgdf_fixed",
            "source_retention_ratio": schedule.source_retention_ratio,
            "source_proto_keep_ratio": schedule.source_proto_keep_ratio,
            "seed": schedule.seed,
            "update_epochs": sorted(schedule.budgets),
            "selection_rule": "current-model class-wise small-loss with exact PGDF selected_count",
            "uses_pgdf_sample_identity": False,
            "uses_prototype_or_graph_scores": False,
        },
    )


def verify_observed_budget_match(
    per_class_rows: list[dict[str, Any]],
    schedule: ClassBudgetSchedule,
) -> None:
    observed = {
        (int(row["epoch"]), str(row["web_label"])): int(row["selected_count"])
        for row in per_class_rows
    }
    expected = {
        (epoch, noisy_label): int(selected_count)
        for epoch, class_counts in schedule.budgets.items()
        for noisy_label, selected_count in class_counts.items()
    }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        wrong = sorted(
            key for key in set(expected) & set(observed) if expected[key] != observed[key]
        )
        raise RuntimeError(
            "Observed Dynamic selections do not exactly match the PGDF class budgets: "
            f"missing={missing[:5]}, extra={extra[:5]}, wrong_counts={wrong[:5]}."
        )


def require_fresh_run_dir(run_dir: Path) -> None:
    """Protect completed or partial raw runs from accidental overwrite."""
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty and will not be overwritten: {run_dir}. "
            "Choose a new --output-dir or remove/move the partial run explicitly."
        )


def preflight_run_dirs(output_dir: Path, methods: list[str], seeds: list[int]) -> None:
    """Check the complete request before any config/reference file is rewritten."""
    for method_key in methods:
        for seed in seeds:
            require_fresh_run_dir(output_dir / method_key / f"seed{seed}")


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
    if args.dual_batch_size <= 0 or args.dual_grad_accum_steps <= 0:
        raise ValueError("--dual-batch-size and --dual-grad-accum-steps must be positive.")
    if args.jocor_lambda < 0.0:
        raise ValueError("--jocor-lambda must be non-negative.")
    if args.official_test_selected_only and args.posthoc_oracle_test:
        raise ValueError("--official-test-selected-only cannot be combined with --posthoc-oracle-test.")


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one method is required.")
    if "all" in methods:
        if methods != ["all"]:
            raise ValueError("--methods all cannot be combined with explicit method keys.")
        return list(METHODS)
    allowed = set(METHODS) | set(ABLATION_METHODS) | set(LEGACY_METHOD_ALIASES)
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}. Allowed primary methods: {METHODS}; "
            f"explicit ablations: {ABLATION_METHODS}; legacy alias: dynamic"
        )
    canonical = [canonical_method_key(method) for method in methods]
    if len(set(canonical)) != len(canonical):
        raise ValueError("--methods contains duplicate or aliased-duplicate methods.")
    return methods


def validate_official_test_request(methods: list[str], selected_only: bool) -> None:
    """Keep strict ablations on their validation-selected one-shot test protocol."""
    canonical = [canonical_method_key(method) for method in methods]
    strict_methods = {"proto_only", "dynamic_budget_matched"}
    requested_strict = sorted(strict_methods & set(canonical))
    if requested_strict and not selected_only:
        raise ValueError(f"{', '.join(requested_strict)} requires --official-test-selected-only.")
    if selected_only and (len(canonical) != 1 or canonical[0] not in strict_methods):
        raise ValueError(
            "--official-test-selected-only is restricted to a single strict ablation: "
            "proto_only or dynamic_budget_matched."
        )


def canonical_method_key(method_key: str) -> str:
    return LEGACY_METHOD_ALIASES.get(method_key, method_key)


def resolve_retention_ratio(method_key: str, pgdf_dynamic_ratio: float) -> float:
    canonical = canonical_method_key(method_key)
    if canonical == "dynamic_r08":
        return 0.8
    if canonical == "dynamic_r09":
        return 0.9
    if canonical in {"pgdf_auto", "pgdf_fixed"}:
        return float(pgdf_dynamic_ratio)
    raise ValueError(f"Method {method_key} has no dynamic retention ratio.")


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
    official_test_mode = (
        "validation-selected checkpoint only"
        if args.official_test_selected_only
        else "validation-selected, final, and last-5 checkpoints"
    )
    lines = [
        "# Validation-Selected Checkpoint Run",
        "",
        f"- Protocol: `{PROTOCOL_NAME}`.",
        f"- Shared fixed clean validation: {manifest['validation_samples']}/{manifest['source_train_samples']} samples, ratio={manifest['validation_ratio']}, seed={manifest['validation_seed']}.",
        "- During fitting, checkpoint selection uses validation Top-1 only. The official test set is evaluated after fitting.",
        f"- Official test evaluation: {official_test_mode}.",
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
        "loss_type", "jal_alpha", "jal_beta", "jal_a", "jal_eps", "selection_mode", "remember_rate", "selected_count",
        "selected_clean", "selected_purity", "clean_recall", "top1_a", "top5_a", "top1_b", "top5_b", "mean_ab_top1", "mean_ab_top5",
    ]


def result_fields() -> list[str]:
    return [
        "method_key", "method", "seed", "checkpoint_protocol", "official_test_evaluation", "train_samples", "eval_samples", "validation_samples", "test_samples",
        "best_val_epoch", "best_val_top1", "best_val_top5", "validation_selected_test_top1", "validation_selected_test_top5",
        "validation_selected_test_model_a_top1", "validation_selected_test_model_a_top5", "validation_selected_test_model_b_top1", "validation_selected_test_model_b_top5",
        "final_test_top1", "final_test_top5", "final_test_model_a_top1", "final_test_model_a_top5", "final_test_model_b_top1", "final_test_model_b_top5",
        "last5_test_mean", "last5_test_std", "last5_test_top5_mean", "last5_test_top5_std", "oracle_best_test_epoch",
        "oracle_best_test_top1", "oracle_best_test_top5", "oracle_best_to_final_drop", "retention_ratio", "proto_keep_ratio",
        "auto_proto_keep", "auto_proto_jaccard", "warmup_epochs", "update_interval", "candidate_samples", "final_selected_samples",
        "selection_updates", "budget_matched", "scheduler_retention_ratio", "budget_source_path", "budget_source_sha256",
        "budget_source_retention_ratio", "budget_source_proto_keep_ratio", "budget_match_verified",
        "remember_mode", "remember_rate", "final_remember_rate", "lambda_cor", "selected_count", "selection_ratio",
        "loss_type", "jal_alpha", "jal_beta", "jal_a", "jal_eps", "selection_mode",
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


def dual_selection_fields() -> list[str]:
    return [
        "method", "seed", "epoch", "remember_rate", "selected_count", "selected_ratio", "top1_a", "top5_a",
        "top1_b", "top5_b", "mean_ab_top1", "mean_ab_top5", "selection_mode",
    ]


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
