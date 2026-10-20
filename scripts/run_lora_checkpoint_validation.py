"""Run validation-selected LoRA baselines on synthetic asymmetric-noise datasets.

This entry point intentionally separates model selection from the official test set.
It is for CUB/Cars/Aircraft synthetic-noise experiments, whose noise-index CSVs
contain the original clean labels needed for a fixed held-out validation split.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from gcdd.budget_matching import ClassBudgetSchedule, file_sha256, load_pgdf_class_budget_schedule
from gcdd.config import deep_update
from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json, write_yaml
from gcdd.ido import (
    BMMDegenerateError,
    IDO_DEGENERATE_POLICY,
    IDO_WAVE_SNAPSHOT_POLICY,
    IDO_WELoss_SOURCE,
    ido_adaptation_name,
    ido_bmm_scope,
    resolve_ido_bmm_mode,
    resolve_stage2_microbatch_size,
    resolve_stage_budget,
    train_ido_lora,
)
from gcdd.lora_dynamic import (
    BUDGET_MATCHED_PROTO_STRATEGY,
    BUDGET_MATCHED_RANDOM_STRATEGY,
    RANDOM_BUDGET_SELECTION_SCHEME,
    selection_update_epochs,
    train_dynamic_loss_lora,
)
from gcdd.lora_noisy_baselines import train_coteaching_lora, train_jocor_lora
from gcdd.lora_training import train_dinov2_lora
from gcdd.noise_generalization import (
    GCCN_STRATEGY,
    IDN_STRATEGY,
    validate_noise_generalization_training_input,
)
from gcdd.progress import log_stage
from revision_experiments._paths import protect_output_root
from revision_experiments.profiling import ProfilingMode, RunProfiler
from revision_experiments.selection_logging import SelectionLogger
from revision_experiments.training_integration import (
    SelectionRunMetadata,
    build_experiment_id,
    build_code_provenance,
    protect_monitored_training_output_dir,
    write_instrumentation_outputs,
)


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
# IDO is intentionally opt-in rather than part of ``all``.  It is a controlled
# DINOv2-LoRA adaptation of an external method, not one of the frozen original
# 13-method comparison runs or a PGDF ablation.
CONTROLLED_ADAPTATION_METHODS = ("ido",)
EXACT_BUDGET_CONTROL_METHODS = (
    "pgdf_exact_budget_proto",
    "pgdf_exact_budget_random",
)
ABLATION_METHODS = (
    "proto_only",
    "dynamic_budget_matched",
    "dynamic_budget_matched_dynamic_proto",
    *EXACT_BUDGET_CONTROL_METHODS,
    "dynamic_proto_only",
    "pgdf_dynamic_proto",
    "fixed_proto_warmup_matched",
    "pgdf_fusion_weighted",
    "pgdf_fusion_union",
    "pgdf_fusion_soft",
    "pgdf_dynamic_fusion_weighted",
    "pgdf_dynamic_fusion_union",
    "pgdf_dynamic_fusion_soft",
)
FIXED_FUSION_ABLATION_METHODS = (
    "pgdf_fusion_weighted",
    "pgdf_fusion_union",
    "pgdf_fusion_soft",
)
DYNAMIC_FUSION_ABLATION_METHODS = (
    "pgdf_dynamic_fusion_weighted",
    "pgdf_dynamic_fusion_union",
    "pgdf_dynamic_fusion_soft",
)
FUSION_ABLATION_METHODS = (
    *FIXED_FUSION_ABLATION_METHODS,
    *DYNAMIC_FUSION_ABLATION_METHODS,
)
FIXED_FUSION_SPECIFICATION = {
    "pgdf_fusion_weighted": ("fusion_weighted", "weighted_ranking", "equal-weight percentile ranking"),
    "pgdf_fusion_union": ("fusion_union", "union_relaxed", "union gate with best-rank ordering"),
    "pgdf_fusion_soft": ("fusion_soft", "soft_weighting", "continuous equal-weight reliability"),
}
DYNAMIC_FUSION_SPECIFICATION = {
    "pgdf_dynamic_fusion_weighted": ("fusion_weighted", "weighted_ranking", "equal-weight percentile ranking"),
    "pgdf_dynamic_fusion_union": ("fusion_union", "union_relaxed", "union gate with best-rank ordering"),
    "pgdf_dynamic_fusion_soft": ("fusion_soft", "soft_weighting", "continuous equal-weight reliability"),
}
STATIC_METHODS = ("full_gcdd", "centroid", "proto_only", "both_only", "gcdd_proto", "fine")
DYNAMIC_METHODS = (
    "dynamic",
    "dynamic_r08",
    "dynamic_r09",
    "dynamic_budget_matched",
    "dynamic_budget_matched_dynamic_proto",
    *EXACT_BUDGET_CONTROL_METHODS,
    "pgdf_auto",
    "pgdf_fixed",
    "dynamic_proto_only",
    "pgdf_dynamic_proto",
    "fixed_proto_warmup_matched",
    *FUSION_ABLATION_METHODS,
)
LEGACY_METHOD_ALIASES = {"dynamic": "dynamic_r08"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-validation noisy-label baselines and explicit controlled adaptations."
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
            "proto_only, dynamic_budget_matched, dynamic_proto_only, pgdf_dynamic_proto, and pgdf_fusion_* are explicit ablations; "
            "pgdf_exact_budget_{proto,random} are exact class-wise budget controls; "
            "ido is an opt-in controlled adaptation; dynamic aliases dynamic_r08."
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
            "Reference directory containing seed*/selection_per_class.csv from the declared PGDF source. "
            "Required by budget-matched methods and pgdf_fusion_*; fixed fusion requires pgdf_fixed and "
            "pgdf_dynamic_fusion_* and pgdf_exact_budget_* require pgdf_dynamic_proto. "
            "Budget-matched methods read counts only, never PGDF sample identities."
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
            "Evaluate official test only once on best_val.pt for single-branch methods. "
            "Required for proto_only, dynamic_budget_matched, dynamic_proto_only, pgdf_dynamic_proto, "
            "pgdf_exact_budget_*, and pgdf_fusion_* methods."
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
    parser.add_argument(
        "--ido-stage1-epochs",
        type=int,
        help=(
            "Optional explicitly pre-registered Stage-1 budget for IDO. By default, the total epoch budget must be "
            "divisible by 3 and the official 5:10 ratio is scaled exactly (30 -> 10+20)."
        ),
    )
    parser.add_argument(
        "--ido-bmm-mode",
        choices=["classwise", "global"],
        help=(
            "Explicit IDO BMM fit scope. If omitted, uses config.ido_controlled_adaptation.bmm_mode "
            "or the backward-compatible classwise default. Global is the separately named whole-training-pool variant."
        ),
    )
    parser.add_argument(
        "--ido-stage2-microbatch-size",
        type=int,
        help=(
            "Optional physical Stage-2 microbatch size for gradient accumulation. "
            "The IDO DataLoader/effective batch, optimizer-step count, and scheduler-step count remain unchanged."
        ),
    )
    parser.add_argument(
        "--ido-task-contract",
        type=Path,
        help=(
            "Optional immutable JSON contract written by gcdd.ido_batch. It is accepted only for one IDO seed and "
            "must be stored in the resulting ido/seed<N> directory; it is echoed into result/checkpoint metadata "
            "for strict batch completion auditing."
        ),
    )
    parser.add_argument(
        "--ido-pilot-no-official-test",
        action="store_true",
        help=(
            "IDO-only GPU-pilot safeguard: do not pass official-test data into the IDO trainer. "
            "It cannot be combined with a batch task contract or validation-selected test evaluation."
        ),
    )
    parser.add_argument(
        "--initialize-output-contract-only",
        action="store_true",
        help=(
            "Write the resolved configuration and fixed validation manifest, then exit before graph construction, "
            "model creation, or training. Intended only to initialize a future training root's immutable input contract."
        ),
    )
    parser.add_argument(
        "--revision-profiling-mode",
        choices=[mode.value for mode in ProfilingMode],
        default=ProfilingMode.OFF.value,
        help="Opt-in D1 monitoring for dynamic PGDF-family runs and explicit IDO controlled adaptation; default preserves historical behavior.",
    )
    parser.add_argument(
        "--revision-selection-logging",
        action="store_true",
        help="Opt-in D2 selection facts for dynamic PGDF-family runs; never records oracle labels.",
    )
    parser.add_argument(
        "--revision-output-root",
        help="Required when either D1/D2 option is enabled; must be outside protected historical outputs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    noise_index = Path(args.noise_index)
    if not noise_index.exists():
        raise FileNotFoundError(f"Noise index does not exist: {noise_index}")
    instrumentation_requested = (
        args.revision_profiling_mode != ProfilingMode.OFF.value or bool(args.revision_selection_logging)
    )
    if instrumentation_requested and not args.revision_output_root:
        raise ValueError("--revision-output-root is required when revision profiling or selection logging is enabled")
    revision_output_root = (
        protect_output_root(Path(args.revision_output_root)) if instrumentation_requested else None
    )
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / f"checkpoint_validation_all_methods_s{args.validation_seed}"
    output_dir = protect_monitored_training_output_dir(
        output_dir,
        instrumentation_requested=instrumentation_requested,
    )
    legacy_output_dir = input_dir / f"checkpoint_validation_s{args.validation_seed}"
    if output_dir.resolve() == legacy_output_dir.resolve() and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into legacy raw-output directory {output_dir}. "
            "Use a new --output-dir such as checkpoint_validation_all_methods_s<validation-seed>."
        )
    ensure_dir(output_dir)

    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)
    canonical_methods = [canonical_method_key(method) for method in methods]
    validate_ido_pilot_no_official_test_request(args, canonical_methods, seeds)
    dynamic_fusion_requested = bool(set(canonical_methods) & set(DYNAMIC_FUSION_ABLATION_METHODS))
    dynamic_budget_control_requested = bool(
        set(canonical_methods)
        & {"dynamic_budget_matched_dynamic_proto", *EXACT_BUDGET_CONTROL_METHODS}
    )
    dynamic_prototype_score_requested = bool(
        set(canonical_methods)
        & {*DYNAMIC_FUSION_ABLATION_METHODS, "pgdf_exact_budget_proto"}
    )
    fixed_fusion_requested = bool(set(canonical_methods) & set(FIXED_FUSION_ABLATION_METHODS))
    dynamic_budget_source_provenance = ""
    validate_official_test_request(methods, bool(args.official_test_selected_only))
    pgdf_budget_root = Path(args.pgdf_budget_root) if args.pgdf_budget_root else None
    validate_budget_method_request(methods, pgdf_budget_root, seeds)
    preflight_run_dirs(output_dir, methods, seeds)
    path_maps = parse_path_maps(args.path_map)
    cfg = load_config(input_dir, Path(args.config) if args.config else None)
    apply_lora_defaults(cfg)
    apply_overrides(cfg, args)
    validate_args(args)
    ido_stage_budget: tuple[int, int] | None = None
    ido_bmm_mode = "classwise"
    ido_stage2_microbatch_size: int | None = None
    ido_execution_signature = ""
    if "ido" in canonical_methods:
        ido_stage_budget = resolve_stage_budget(int(cfg["lora_train"]["epochs"]), args.ido_stage1_epochs)
        existing_ido = cfg.get("ido_controlled_adaptation", {})
        if not isinstance(existing_ido, dict):
            raise ValueError("ido_controlled_adaptation must be a mapping when supplied in config.")
        configured_mode = existing_ido.get("bmm_mode")
        ido_bmm_mode = resolve_ido_bmm_mode(args.ido_bmm_mode if args.ido_bmm_mode is not None else configured_mode)
        configured_microbatch = existing_ido.get("stage2_microbatch_size")
        ido_stage2_microbatch_size = resolve_stage2_microbatch_size(
            args.ido_stage2_microbatch_size
            if args.ido_stage2_microbatch_size is not None
            else configured_microbatch
        )
        stage2_effective_batch_size = int(cfg["lora_train"]["batch_size"])
        if stage2_effective_batch_size <= 0:
            raise ValueError("lora_train.batch_size must be positive for IDO Stage 2.")
        accumulation_enabled = bool(
            ido_stage2_microbatch_size is not None
            and ido_stage2_microbatch_size < stage2_effective_batch_size
        )
        ido_execution_signature = canonical_json_sha256(
            {
                "schema": "ido_execution_protocol_v1",
                "identity": ido_adaptation_name(ido_bmm_mode),
                "stage1_epochs": int(ido_stage_budget[0]),
                "stage2_epochs": int(ido_stage_budget[1]),
                "bmm_mode": ido_bmm_mode,
                "bmm_scope": ido_bmm_scope(ido_bmm_mode),
                "stage2_effective_batch_size": stage2_effective_batch_size,
                "stage2_microbatch_size": ido_stage2_microbatch_size,
                "stage2_gradient_accumulation_enabled": accumulation_enabled,
            }
        )
        cfg["ido_controlled_adaptation"] = {
            "identity": ido_adaptation_name(ido_bmm_mode),
            "official_reference": "https://github.com/iTheresaApocalypse/IDO",
            "scope": "DINOv2-B/14+LoRA controlled adaptation; global-BMM is not a class-wise-equivalent or original-architecture-equivalent reproduction",
            "stage_budget_rule": "scale official Stage-1:Stage-2 5:10 ratio to the total controlled epoch budget",
            "stage_budget_source": "explicit_cli_pre_registered" if args.ido_stage1_epochs is not None else "default_official_ratio_scaling",
            "stage1_epochs": int(ido_stage_budget[0]),
            "stage2_epochs": int(ido_stage_budget[1]),
            "bmm_mode": ido_bmm_mode,
            "bmm_scope": ido_bmm_scope(ido_bmm_mode),
            "bmm_mode_source": "explicit_cli" if args.ido_bmm_mode is not None else ("config" if configured_mode is not None else "backward_compatible_default"),
            "stage2_effective_batch_size": stage2_effective_batch_size,
            "stage2_microbatch_size": ido_stage2_microbatch_size,
            "stage2_microbatch_size_source": (
                "explicit_cli" if args.ido_stage2_microbatch_size is not None
                else ("config" if configured_microbatch is not None else "disabled_default")
            ),
            "stage2_gradient_accumulation_enabled": accumulation_enabled,
            "stage2_nominal_accumulation_steps": (
                1 if ido_stage2_microbatch_size is None
                else (stage2_effective_batch_size + ido_stage2_microbatch_size - 1) // ido_stage2_microbatch_size
            ),
            "execution_signature": ido_execution_signature,
            "bmm_degenerate_policy": IDO_DEGENERATE_POLICY,
            "bmm_initialization": {
                "alpha": [1.0, 2.0],
                "beta": [2.0, 1.0],
                "mixture_weights": [0.5, 0.5],
            },
            "bmm_epsilon": 1.0e-6,
            "bmm_component_boundary": 1.0e-5,
            "bmm_responsibility_denominator": "component_density_sum_plus_epsilon",
            "bmm_posterior_complementarity": "not_forced; each posterior uses component_density_sum_plus_epsilon",
            "label_wave_snapshot_policy": IDO_WAVE_SNAPSHOT_POLICY,
            "weloss_source": IDO_WELoss_SOURCE,
            "bmm_initial_fit_iterations": 20,
            "bmm_epoch_update_iterations": 2,
            "stage2_augmentation": "weak RandomResizedCrop+Flip; strong RandomResizedCrop+Flip+RandAugment(2,10)",
        }
    ido_task_contract = load_ido_task_contract(
        args.ido_task_contract,
        input_dir=input_dir,
        noise_index=noise_index,
        output_dir=output_dir,
        methods=canonical_methods,
        seeds=seeds,
        validation_seed=int(args.validation_seed),
        stage_budget=ido_stage_budget,
        path_maps=path_maps,
        official_test_selected_only=bool(args.official_test_selected_only),
        posthoc_oracle_test=bool(args.posthoc_oracle_test),
        bmm_mode=ido_bmm_mode,
        stage2_microbatch_size=ido_stage2_microbatch_size,
    )
    if ido_task_contract is not None:
        cfg["ido_batch_task_contract"] = {
            "task_id": ido_task_contract["task"]["task_id"],
            "attempt_id": ido_task_contract["task"]["attempt_id"],
            "task_contract_sha256": canonical_json_sha256(ido_task_contract),
            "experiment_signature": ido_task_contract["experiment_signature"],
        }
    jal_params = resolve_jal_params(cfg, args)
    noise_metadata = load_noise_metadata(noise_index)
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
        "fusion_ablation": {
            "prototype_source": (
                "method-specific mixed frozen and current-LoRA prototype similarity"
                if dynamic_fusion_requested and fixed_fusion_requested
                else "current LoRA-adapted CLS rebuilt at every selection update"
                if dynamic_prototype_score_requested
                else "not used by the requested small-loss-only/random actual-budget controls"
                if dynamic_budget_control_requested
                else "existing frozen PGDF prototype similarity"
            ),
            "loss_source": "current-model full-training-pool CE loss",
            "ranking_percentile": "within observed-label class; deterministic original-index tie-break",
            "weighted_ranking": {"loss_weight": 0.5, "prototype_weight": 0.5},
            "union_relaxed": {"candidate_set": "loss_union_prototype", "ranking": "max_signal_reliability"},
            "soft_weighting": {
                "loss_weight": 0.5,
                "prototype_weight": 0.5,
                "positive_weight_floor": 1.0e-6,
                "class_normalization": "sum_to_reference_pgdf_budget",
            },
            "dynamic_pgdf_budget_controls": {
                "reference": "actual selected_count per observed-label class and selection update only",
                "source_method": "pgdf_dynamic_proto",
                "controls": ["dynamic_budget_matched_dynamic_proto", *EXACT_BUDGET_CONTROL_METHODS],
                "random_selection": RANDOM_BUDGET_SELECTION_SCHEME,
            },
        },
        "noise_realization": noise_metadata,
        "noise_index_sha256": file_sha256(noise_index),
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
    log_stage("[1/5] Loading V1 train/test arrays and creating the shared clean validation manifest.")
    data = load_data(input_dir)
    if noise_metadata["noise_strategy"] in {IDN_STRATEGY, GCCN_STRATEGY}:
        cfg["checkpoint_validation"]["noise_generalization_input_validation"] = (
            validate_noise_generalization_training_input(
                manifest_path=noise_index,
                train_paths=data["train_paths"],
                train_labels=data["train_labels"],
                labels_path=input_dir / "labels.npy",
            )
        )
    write_yaml(output_dir / "resolved_config.yaml", cfg)
    clean_labels = load_clean_labels_from_noise_index(noise_index, data["train_paths"])
    split = build_or_load_fixed_validation_split(
        output_dir,
        data["train_paths"],
        clean_labels,
        validation_ratio=float(args.validation_ratio),
        validation_seed=int(args.validation_seed),
    )
    if args.initialize_output_contract_only:
        print(
            "OUTPUT CONTRACT INITIALIZED: resolved configuration and fixed validation manifest were written; "
            "no graph, model, GPU, or training process was started.",
            flush=True,
        )
        return
    if pgdf_budget_root is not None:
        validate_budget_source_manifest(pgdf_budget_root, split.metadata)
        if dynamic_fusion_requested or dynamic_budget_control_requested:
            dynamic_budget_source_provenance = validate_dynamic_pgdf_budget_source_provenance(
                pgdf_budget_root,
                dataset_name=str(cfg.get("dataset", {}).get("name", "")),
                noise_metadata=noise_metadata,
                noise_index_sha256=file_sha256(noise_index),
                noise_index_name=noise_index.name,
            )
    validation_idx = np.where(split.validation_mask)[0]
    validation_paths = [data["train_paths"][int(index)] for index in validation_idx]
    validation_labels = clean_labels[validation_idx]
    log_stage(
        f"[checkpoint-validation] training_pool={int(split.train_mask.sum())}, validation={len(validation_idx)}, "
        f"official_test={'not passed to IDO pilot' if args.ido_pilot_no_official_test else len(data['test_paths'])}; "
        "test is not evaluated during fitting."
    )

    pgdf_reference: dict[str, Any] | None = None
    static_selections: dict[str, dict[str, Any]] = {}
    # Dynamic-prototype ablations intentionally do not touch the frozen CLS
    # cache or graph reference; their geometry is rebuilt inside the LoRA
    # training loop from the current model state.
    graph_methods_requested = any(
        method in STATIC_METHODS
        or method in {
            "pgdf_auto",
            "pgdf_fixed",
            "fixed_proto_warmup_matched",
            *FIXED_FUSION_ABLATION_METHODS,
        }
        for method in canonical_methods
    )
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
            source_active_budget_audit: dict[str, Any] | None = None
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
            elif canonical_key == "ido":
                if ido_stage_budget is None:
                    raise RuntimeError("IDO stage budget was not resolved.")
                run_cfg["loss_type"] = "ido_official_weloss"
                run_cfg["ido_controlled_adaptation"] = copy.deepcopy(cfg["ido_controlled_adaptation"])
                method_name = ido_adaptation_name(ido_bmm_mode)
                profiler: RunProfiler | None = None
                if args.revision_profiling_mode != ProfilingMode.OFF.value:
                    profiler = RunProfiler(
                        build_experiment_id(
                            dataset=str(run_cfg.get("dataset", {}).get("name", "")),
                            noise_protocol=str(noise_metadata["noise_strategy"]),
                            noise_seed=int(noise_metadata["noise_seed"]),
                            method=canonical_key,
                            training_seed=int(seed),
                        ),
                        ProfilingMode(args.revision_profiling_mode),
                        device_type="cpu",
                    )
                try:
                    result = train_ido_lora(
                        data["train_paths"],
                        data["train_labels"],
                        validation_paths,
                        validation_labels,
                        split.train_mask,
                        run_cfg,
                        method=method_name,
                        seed=seed,
                        stage1_epochs=int(ido_stage_budget[0]),
                        bmm_mode=ido_bmm_mode,
                        stage2_microbatch_size=ido_stage2_microbatch_size,
                        path_maps=path_maps,
                        checkpoint_path=checkpoints / "best_val.pt",
                        test_paths=None if args.ido_pilot_no_official_test else data["test_paths"],
                        test_labels=None if args.ido_pilot_no_official_test else data["test_labels"],
                        final_checkpoint_path=checkpoints / "last.pt",
                        last5_checkpoint_dir=checkpoints / "last5",
                        checkpoint_protocol=PROTOCOL_NAME,
                        posthoc_oracle_test=bool(args.posthoc_oracle_test),
                        official_test_selected_only=bool(args.official_test_selected_only),
                        profiler=profiler,
                        task_contract_sha256="" if ido_task_contract is None else canonical_json_sha256(ido_task_contract),
                        task_id="" if ido_task_contract is None else str(ido_task_contract["task"]["task_id"]),
                        experiment_signature="" if ido_task_contract is None else str(ido_task_contract["experiment_signature"]),
                        execution_signature=ido_execution_signature,
                    )
                except BMMDegenerateError as exc:
                    write_csv(run_dir / "bmm_class_fits.csv", [exc.record], ido_bmm_fit_fields())
                    write_json(
                        run_dir / "bmm_degeneracy.json",
                        {
                            "status": "ABORTED_FAIL_CLOSED",
                            "policy": IDO_DEGENERATE_POLICY,
                            "record": exc.record,
                            "message": str(exc),
                        },
                    )
                    raise
                write_csv(run_dir / "bmm_class_fits.csv", result.bmm_fit_rows, ido_bmm_fit_fields())
                write_ido_wrong_event(
                    run_dir / "stage1_wrong_event.csv",
                    data,
                    split.train_mask,
                    result.stage1_wrong_event,
                )
                selection_record = str(run_dir / "bmm_class_fits.csv")
                if profiler is not None:
                    assert revision_output_root is not None
                    code_provenance = build_code_provenance(ROOT)
                    write_instrumentation_outputs(
                        output_root=revision_output_root,
                        profiler=profiler,
                        selection_logger=None,
                        run_metadata={
                            "schema_version": "ido_d1_training_integration_v1",
                            "dataset": str(run_cfg.get("dataset", {}).get("name", "")),
                            "noise_protocol": str(noise_metadata["noise_strategy"]),
                            "noise_seed": int(noise_metadata["noise_seed"]),
                            "training_seed": int(seed),
                            "method": canonical_key,
                            "profiling_mode": args.revision_profiling_mode,
                            "ido_stage1_epochs": int(ido_stage_budget[0]),
                            "ido_stage2_epochs": int(ido_stage_budget[1]),
                            "ido_bmm_mode": ido_bmm_mode,
                            "ido_bmm_scope": ido_bmm_scope(ido_bmm_mode),
                            "ido_stage2_effective_batch_size": int(run_cfg["lora_train"]["batch_size"]),
                            "ido_stage2_microbatch_size": ido_stage2_microbatch_size,
                            "ido_stage2_gradient_accumulation_enabled": bool(
                                run_cfg["ido_controlled_adaptation"]["stage2_gradient_accumulation_enabled"]
                            ),
                            "ido_execution_signature": ido_execution_signature,
                            "weloss_source": IDO_WELoss_SOURCE,
                            "bmm_degenerate_policy": IDO_DEGENERATE_POLICY,
                            "code_commit": code_provenance["git_head"],
                            "code_provenance": code_provenance,
                        },
                    )
                result_row = finalize_result_row(
                    method_key,
                    result.summary,
                    f"ido_{ido_bmm_mode}_bmm_all_training_pool",
                    selected_count=int(split.train_mask.sum()),
                    candidate_count=int(split.train_mask.sum()),
                )
                if args.ido_pilot_no_official_test:
                    # Preserve a machine-readable indication that this output
                    # cannot satisfy the formal batch completion contract.
                    result_row["official_test_evaluation"] = "not_run_ido_pilot"
                if ido_task_contract is not None:
                    result_row.update(
                        {
                            "ido_task_id": str(ido_task_contract["task"]["task_id"]),
                            "ido_task_contract_sha256": canonical_json_sha256(ido_task_contract),
                            "ido_experiment_signature": str(ido_task_contract["experiment_signature"]),
                        }
                    )
            elif canonical_key in DYNAMIC_METHODS:
                run_cfg["loss_type"] = "ce"
                budget_schedule: ClassBudgetSchedule | None = None
                scheduler_retention_ratio: float | None = None
                selection_strategy = "loss_only"
                prototype_mode = "fixed"
                fusion_mode: str | None = None
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
                    write_budget_source(run_dir, budget_schedule, source_method="pgdf_fixed")
                elif canonical_key in {
                    "dynamic_budget_matched_dynamic_proto",
                    *EXACT_BUDGET_CONTROL_METHODS,
                }:
                    if pgdf_budget_root is None:
                        raise RuntimeError("Dynamic PGDF budget root was not configured.")
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
                        expected_selection_strategy="loss_and_proto",
                        expected_prototype_mode="dynamic_lora",
                    )
                    validate_budget_source_hyperparameters(
                        budget_schedule,
                        dynamic_ratio=float(args.dynamic_ratio),
                        fixed_p=float(args.fixed_p),
                    )
                    source_active_budget_audit = inspect_pgdf_active_budget_source(
                        pgdf_budget_root,
                        budget_schedule,
                    )
                    retention_ratio = float(budget_schedule.source_retention_ratio)
                    proto_keep_ratio = float(budget_schedule.source_proto_keep_ratio)
                    scheduler_retention_ratio = min(retention_ratio, proto_keep_ratio)
                    centroid_mask = None
                    auto_proto_keep = None
                    if canonical_key in {
                        "dynamic_budget_matched_dynamic_proto",
                    }:
                        method_name = (
                            "DINOv2 LoRA PGDF-DynamicProto actual-budget Small-loss Only "
                            f"r={retention_ratio:g} p={proto_keep_ratio:g}"
                        )
                        proto_scores = None
                        proto_keep_ratio = None
                        selection_strategy = "loss_only"
                        prototype_mode = "fixed"
                        selection_mode = "pgdf_dynamic_proto_actual_classwise_budget_small_loss_only"
                        source_selection_rule = (
                            "current-model observed-class CE-loss ascending with exact PGDF selected_count"
                        )
                        uses_current_model_loss = True
                        uses_current_model_prototype = False
                        random_selection_scheme = ""
                    elif canonical_key == "pgdf_exact_budget_proto":
                        method_name = (
                            "DINOv2 LoRA PGDF-DynamicProto actual-budget Prototype Only "
                            f"r={retention_ratio:g} p={proto_keep_ratio:g}"
                        )
                        proto_scores = None
                        selection_strategy = BUDGET_MATCHED_PROTO_STRATEGY
                        prototype_mode = "dynamic_lora"
                        selection_mode = "pgdf_dynamic_proto_actual_classwise_budget_prototype_only"
                        source_selection_rule = (
                            "current-model observed-class prototype-similarity descending with exact PGDF selected_count"
                        )
                        uses_current_model_loss = False
                        uses_current_model_prototype = True
                        random_selection_scheme = ""
                    else:
                        method_name = (
                            "DINOv2 LoRA PGDF-DynamicProto actual-budget Random "
                            f"r={retention_ratio:g} p={proto_keep_ratio:g}"
                        )
                        proto_scores = None
                        proto_keep_ratio = None
                        selection_strategy = BUDGET_MATCHED_RANDOM_STRATEGY
                        prototype_mode = "fixed"
                        selection_mode = "pgdf_dynamic_proto_actual_classwise_budget_random_sha256_v1"
                        source_selection_rule = (
                            "deterministic current-pool observed-class SHA-256 order with exact PGDF selected_count"
                        )
                        uses_current_model_loss = False
                        uses_current_model_prototype = False
                        random_selection_scheme = RANDOM_BUDGET_SELECTION_SCHEME
                    write_budget_source(
                        run_dir,
                        budget_schedule,
                        source_method="pgdf_dynamic_proto",
                        source_provenance_status=dynamic_budget_source_provenance,
                        selection_rule=source_selection_rule,
                        uses_current_model_loss=uses_current_model_loss,
                        uses_current_model_prototype=uses_current_model_prototype,
                        random_selection_scheme=random_selection_scheme,
                        noise_index_sha256=file_sha256(noise_index),
                        source_active_budget_audit=source_active_budget_audit,
                    )
                elif canonical_key in FIXED_FUSION_ABLATION_METHODS:
                    if pgdf_budget_root is None or pgdf_reference is None:
                        raise RuntimeError("Fusion ablations require validated pgdf_fixed budgets and frozen prototype scores.")
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
                    selection_strategy, fusion_mode, fusion_description = FIXED_FUSION_SPECIFICATION[canonical_key]
                    retention_ratio = float(budget_schedule.source_retention_ratio)
                    proto_keep_ratio = float(budget_schedule.source_proto_keep_ratio)
                    scheduler_retention_ratio = min(retention_ratio, proto_keep_ratio)
                    method_name = (
                        f"DINOv2 LoRA PGDF fusion ablation ({fusion_description}) "
                        f"matched to frozen PGDF class budgets r={retention_ratio:g} p={proto_keep_ratio:g}"
                    )
                    proto_scores = pgdf_reference["proto_scores"]
                    centroid_mask = None
                    auto_proto_keep = None
                    prototype_mode = "fixed"
                    selection_mode = f"pgdf_fixed_budget_matched_{fusion_mode}"
                    write_budget_source(run_dir, budget_schedule, source_method="pgdf_fixed")
                elif canonical_key in DYNAMIC_FUSION_ABLATION_METHODS:
                    if pgdf_budget_root is None:
                        raise RuntimeError("Dynamic-prototype fusion ablations require validated pgdf_dynamic_proto budgets.")
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
                        expected_selection_strategy="loss_and_proto",
                        expected_prototype_mode="dynamic_lora",
                    )
                    validate_budget_source_hyperparameters(
                        budget_schedule,
                        dynamic_ratio=float(args.dynamic_ratio),
                        fixed_p=float(args.fixed_p),
                    )
                    selection_strategy, fusion_mode, fusion_description = DYNAMIC_FUSION_SPECIFICATION[canonical_key]
                    retention_ratio = float(budget_schedule.source_retention_ratio)
                    proto_keep_ratio = float(budget_schedule.source_proto_keep_ratio)
                    scheduler_retention_ratio = min(retention_ratio, proto_keep_ratio)
                    method_name = (
                        f"DINOv2 LoRA PGDF-DynamicProto fusion ablation ({fusion_description}) "
                        f"matched to dynamic-prototype PGDF class budgets r={retention_ratio:g} p={proto_keep_ratio:g}"
                    )
                    proto_scores = None
                    centroid_mask = None
                    auto_proto_keep = None
                    prototype_mode = "dynamic_lora"
                    selection_mode = f"pgdf_dynamic_proto_budget_matched_{fusion_mode}"
                    write_budget_source(
                        run_dir,
                        budget_schedule,
                        source_method="pgdf_dynamic_proto",
                        source_provenance_status=dynamic_budget_source_provenance,
                    )
                elif canonical_key == "dynamic_proto_only":
                    # This is deliberately not the legacy static proto_only
                    # baseline.  It follows the requested full-pool warm-up
                    # and periodic update schedule, with no CE ranking.
                    retention_ratio = 1.0
                    method_name = f"DINOv2 LoRA Dynamic Prototype Gate Only p={args.fixed_p:g}"
                    proto_scores = None
                    centroid_mask = None
                    proto_keep_ratio = float(args.fixed_p)
                    auto_proto_keep = None
                    selection_strategy = "proto_only"
                    prototype_mode = "dynamic_lora"
                    selection_mode = "dynamic_lora_prototype_only_training_pool"
                    scheduler_retention_ratio = float(args.fixed_p)
                elif canonical_key == "pgdf_dynamic_proto":
                    retention_ratio = float(args.dynamic_ratio)
                    method_name = (
                        f"DINOv2 LoRA PGDF-DynamicProto r={args.dynamic_ratio:g} p={args.fixed_p:g}"
                    )
                    proto_scores = None
                    centroid_mask = None
                    proto_keep_ratio = float(args.fixed_p)
                    auto_proto_keep = None
                    selection_strategy = "loss_and_proto"
                    prototype_mode = "dynamic_lora"
                    selection_mode = "pgdf_dynamic_lora_prototype_training_pool_dynamic_loss_and_prototype"
                elif canonical_key == "fixed_proto_warmup_matched":
                    if pgdf_reference is None:
                        raise RuntimeError("Fixed prototype reference was not constructed.")
                    retention_ratio = 1.0
                    method_name = f"DINOv2 LoRA Warmup-Matched Fixed Prototype Gate Only p={args.fixed_p:g}"
                    proto_scores = pgdf_reference["proto_scores"]
                    centroid_mask = None
                    proto_keep_ratio = float(args.fixed_p)
                    auto_proto_keep = None
                    selection_strategy = "proto_only"
                    prototype_mode = "fixed"
                    selection_mode = "fixed_frozen_prototype_warmup_matched_training_pool"
                    scheduler_retention_ratio = float(args.fixed_p)
                elif canonical_key == "pgdf_auto":
                    if pgdf_reference is None:
                        raise RuntimeError("PGDF reference was not constructed.")
                    retention_ratio = resolve_retention_ratio(canonical_key, float(args.dynamic_ratio))
                    method_name = f"DINOv2 LoRA PGDF-auto r={args.dynamic_ratio:g}"
                    proto_scores = pgdf_reference["proto_scores"]
                    centroid_mask = pgdf_reference["centroid_reference_mask"]
                    proto_keep_ratio = None
                    auto_proto_keep = auto_rule
                    selection_strategy = "loss_and_proto"
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
                    selection_strategy = "loss_and_proto"
                    selection_mode = "pgdf_fixed_training_pool_dynamic_loss_and_prototype"
                profiler: RunProfiler | None = None
                selection_logger: SelectionLogger | None = None
                selection_log_metadata: SelectionRunMetadata | None = None
                if instrumentation_requested:
                    instrumentation_experiment_id = build_experiment_id(
                        dataset=str(run_cfg.get("dataset", {}).get("name", "")),
                        noise_protocol=str(noise_metadata["noise_strategy"]),
                        noise_seed=int(noise_metadata["noise_seed"]),
                        method=canonical_key,
                        training_seed=int(seed),
                    )
                    if args.revision_profiling_mode != ProfilingMode.OFF.value:
                        profiler = RunProfiler(
                            instrumentation_experiment_id,
                            ProfilingMode(args.revision_profiling_mode),
                            # The training function updates this after resolve_device("auto").
                            device_type="cpu",
                        )
                    if args.revision_selection_logging:
                        selection_logger = SelectionLogger(instrumentation_experiment_id)
                        selection_log_metadata = SelectionRunMetadata(
                            dataset=str(run_cfg.get("dataset", {}).get("name", "")),
                            noise_protocol=str(noise_metadata["noise_strategy"]),
                            noise_seed=int(noise_metadata["noise_seed"]),
                            training_seed=int(seed),
                        )
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
                    selection_strategy=selection_strategy,
                    prototype_mode=prototype_mode,
                    fusion_mode=fusion_mode,
                    profiler=profiler,
                    selection_logger=selection_logger,
                    selection_log_metadata=selection_log_metadata,
                )
                if revision_output_root is not None:
                    code_provenance = build_code_provenance(ROOT)
                    write_instrumentation_outputs(
                        output_root=revision_output_root,
                        profiler=profiler,
                        selection_logger=selection_logger,
                        run_metadata={
                            "schema_version": "revision_training_integration_v1",
                            "dataset": str(run_cfg.get("dataset", {}).get("name", "")),
                            "noise_protocol": str(noise_metadata["noise_strategy"]),
                            "noise_seed": int(noise_metadata["noise_seed"]),
                            "training_seed": int(seed),
                            "method": canonical_key,
                            "profiling_mode": args.revision_profiling_mode,
                            "selection_logging_enabled": bool(args.revision_selection_logging),
                            "selection_strategy": selection_strategy,
                            "prototype_mode": prototype_mode,
                            "fusion_mode": fusion_mode or "",
                            "retention_ratio": float(retention_ratio),
                            "proto_keep_ratio": proto_keep_ratio,
                            "warmup_epochs": int(args.warmup_epochs),
                            "update_interval": int(args.update_interval),
                            "code_commit": code_provenance["git_head"],
                            "code_provenance": code_provenance,
                        },
                    )
                if budget_schedule is not None:
                    if fusion_mode == "soft_weighting":
                        verify_soft_weight_budget_match(result.fusion_per_class_rows, budget_schedule)
                    else:
                        verify_observed_budget_match(result.per_class_rows, budget_schedule)
                    if canonical_key in {"dynamic_budget_matched_dynamic_proto", *EXACT_BUDGET_CONTROL_METHODS}:
                        write_json(
                            run_dir / "budget_match_validation.json",
                            {
                                "status": "PASS",
                                "method_key": canonical_key,
                                "source_method": "pgdf_dynamic_proto",
                                "source_budget_path": budget_schedule.source_path,
                                "source_budget_sha256": budget_schedule.source_sha256,
                                "selection_updates": sorted(budget_schedule.budgets),
                                "expected_classwise_budget": budget_schedule.budgets,
                                "observed_selection_counts_verified": True,
                                "membership_rule": {
                                    "uses_pgdf_sample_identity": False,
                                    "uses_clean_noisy_oracle": False,
                                    "uses_official_test_for_selection": False,
                                },
                                "source_active_budget_audit": source_active_budget_audit or {},
                            },
                        )
                all_updates.extend(result.update_rows)
                write_csv(run_dir / "selection_rows.csv", result.selection_rows, selection_fields())
                write_csv(run_dir / "selection_updates.csv", result.update_rows, update_fields())
                write_csv(run_dir / "selection_per_class.csv", result.per_class_rows, per_class_fields())
                if result.fusion_update_rows:
                    write_csv(run_dir / "fusion_selection_updates.csv", result.fusion_update_rows, fusion_update_fields())
                    write_csv(run_dir / "fusion_selection_rows.csv", result.fusion_sample_rows, fusion_sample_fields())
                    write_csv(
                        run_dir / "fusion_selection_per_class.csv",
                        result.fusion_per_class_rows,
                        fusion_per_class_fields(),
                    )
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
                            "budget_match_verified": (
                                "effective_weight_sum_matches_reference_budget"
                                if fusion_mode == "soft_weighting"
                                else "selected_count_matches_reference_budget"
                                if fusion_mode is not None
                                else "yes"
                            ),
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

            result_row.update(
                {
                    "noise_index": str(noise_index),
                    "noise_strategy": noise_metadata["noise_strategy"],
                    "noise_ratio": noise_metadata["noise_ratio"],
                    "noise_seed": noise_metadata["noise_seed"],
                    "mapping_type": noise_metadata["mapping_type"],
                    "mapping_seed": noise_metadata["mapping_seed"],
                    "validation_seed": int(args.validation_seed),
                }
            )
            all_logs.extend(result.logs)
            write_csv(run_dir / "train_log.csv", result.logs, train_log_fields())
            budget_source_path = str(result_row.get("budget_source_path", ""))
            source_root = (
                Path(budget_source_path).parents[2]
                if budget_source_path
                else None
            )
            source_validation_manifest = (
                source_root / "validation_manifest.json" if source_root is not None else None
            )
            write_json(
                run_dir / "selection_policy.json",
                {
                    "method_key": method_key,
                    "dataset": str(run_cfg.get("dataset", {}).get("name", "")),
                    "training_seed": int(seed),
                    "noise_seed": noise_metadata["noise_seed"],
                    "validation_seed": int(args.validation_seed),
                    "selection_mode": result_row["selection_mode"],
                    "selected_count": result_row["selected_count"],
                    "selection_ratio": result_row["selection_ratio"],
                    "validation_samples_are_training_candidates": False,
                    "official_test_used_for_checkpoint_selection": False,
                    "budget_source_path": result_row.get("budget_source_path", ""),
                    "budget_source_sha256": result_row.get("budget_source_sha256", ""),
                    "budget_match_verified": result_row.get("budget_match_verified", ""),
                    "source_pgdf_method": (
                        "pgdf_dynamic_proto"
                        if canonical_key in {"dynamic_budget_matched_dynamic_proto", *EXACT_BUDGET_CONTROL_METHODS}
                        else ""
                    ),
                    "source_validation_manifest_sha256": (
                        file_sha256(source_validation_manifest)
                        if source_validation_manifest is not None and source_validation_manifest.is_file()
                        else ""
                    ),
                    "validation_manifest_sha256": file_sha256(output_dir / "validation_manifest.json"),
                    "noise_manifest_sha256": file_sha256(noise_index),
                    "resolved_training_config_sha256": file_sha256(output_dir / "resolved_config.yaml"),
                    "code_provenance": build_code_provenance(ROOT),
                    "control_interpretation": (
                        "internal ablation using PGDF historical class-count budgets; not an autonomous external baseline"
                        if canonical_key in {"dynamic_budget_matched_dynamic_proto", *EXACT_BUDGET_CONTROL_METHODS}
                        else ""
                    ),
                    "selection_strategy": result.summary.get("selection_strategy", ""),
                    "prototype_mode": result.summary.get("prototype_mode", ""),
                    "random_selection_scheme": (
                        RANDOM_BUDGET_SELECTION_SCHEME
                        if result.summary.get("selection_strategy") == BUDGET_MATCHED_RANDOM_STRATEGY
                        else ""
                    ),
                    "ido_stage1_epochs": result.summary.get("stage1_epochs", ""),
                    "ido_stage2_epochs": result.summary.get("stage2_epochs", ""),
                    "ido_base_stage1_epoch": result.summary.get("base_stage1_epoch", ""),
                    "ido_bmm_scope": result.summary.get("bmm_scope", ""),
                    "ido_bmm_mode": result.summary.get("bmm_mode", ""),
                    "ido_stage2_effective_batch_size": result.summary.get("stage2_effective_batch_size", ""),
                    "ido_stage2_microbatch_size": result.summary.get("stage2_microbatch_size", ""),
                    "ido_stage2_gradient_accumulation_enabled": result.summary.get("stage2_gradient_accumulation_enabled", ""),
                    "ido_stage2_microbatch_count_total": result.summary.get("stage2_microbatch_count_total", ""),
                    "ido_execution_signature": result.summary.get("ido_execution_signature", ""),
                    "ido_bmm_degenerate_policy": result.summary.get("bmm_degenerate_policy", ""),
                    "ido_final_bmm_fallback_classes": result.summary.get("final_bmm_fallback_classes", ""),
                    "ido_task_id": result_row.get("ido_task_id", ""),
                    "ido_task_contract_sha256": result_row.get("ido_task_contract_sha256", ""),
                    "ido_experiment_signature": result_row.get("ido_experiment_signature", ""),
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
            "noise_realization": noise_metadata,
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


def write_ido_wrong_event(
    path: Path,
    data: dict[str, Any],
    training_pool_mask: np.ndarray,
    wrong_event: np.ndarray,
) -> None:
    """Persist non-oracle Stage-1 wrong-event facts aligned to the original pool."""
    wrong_event = np.asarray(wrong_event, dtype=np.int64)
    if wrong_event.shape != training_pool_mask.shape:
        raise ValueError("IDO wrong-event array must align with the training-pool mask.")
    rows = [
        {
            "index": int(index),
            "path": data["train_paths"][index],
            "observed_label": str(data["train_labels"][index]),
            "wrong_event": int(wrong_event[index]),
        }
        for index in np.where(training_pool_mask)[0]
    ]
    write_csv(path, rows, ["index", "path", "observed_label", "wrong_event"])


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_ido_task_contract(
    path: Path | None,
    *,
    input_dir: Path,
    noise_index: Path,
    output_dir: Path,
    methods: list[str],
    seeds: list[int],
    validation_seed: int,
    stage_budget: tuple[int, int] | None,
    path_maps: list[tuple[str, str]],
    official_test_selected_only: bool,
    posthoc_oracle_test: bool,
    bmm_mode: str = "classwise",
    stage2_microbatch_size: int | None = None,
) -> dict[str, Any] | None:
    """Load and verify the batch-owned immutable IDO task contract.

    The contract is optional for manual single-run use.  When supplied, it is a
    strict guard: a batch subprocess cannot silently run another method, seed,
    stage budget, validation split, observed-label input, or noise realization.
    """
    if path is None:
        return None
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"--ido-task-contract is not a file: {path}")
    if methods != ["ido"] or len(seeds) != 1 or stage_budget is None:
        raise ValueError("--ido-task-contract requires exactly --methods ido and one seed.")
    expected_run_dir = output_dir.resolve() / "ido" / f"seed{seeds[0]}"
    if path.parent != expected_run_dir:
        raise ValueError("--ido-task-contract must be stored under the resulting ido/seed<N> output directory.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("IDO task contract must be a JSON object.")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported IDO task-contract schema version.")
    experiment = payload.get("experiment")
    task = payload.get("task")
    if not isinstance(experiment, dict) or not isinstance(task, dict):
        raise ValueError("IDO task contract needs experiment and task mappings.")
    if payload.get("experiment_signature") != canonical_json_sha256(experiment):
        raise ValueError("IDO task contract experiment signature is invalid.")
    if payload.get("task_signature") != canonical_json_sha256(task):
        raise ValueError("IDO task contract task signature is invalid.")
    if experiment.get("method_key") != "ido" or int(task.get("seed", -1)) != seeds[0]:
        raise ValueError("IDO task contract method or seed differs from CLI.")
    if int(experiment.get("validation_seed", -1)) != validation_seed:
        raise ValueError("IDO task contract validation seed differs from CLI.")
    if (int(experiment.get("stage1_epochs", -1)), int(experiment.get("stage2_epochs", -1))) != stage_budget:
        raise ValueError("IDO task contract stage budget differs from resolved CLI/config budget.")
    if experiment.get("bmm_mode", "classwise") != resolve_ido_bmm_mode(bmm_mode):
        raise ValueError("IDO task contract BMM mode differs from the resolved CLI/config mode.")
    if experiment.get("stage2_microbatch_size") != resolve_stage2_microbatch_size(stage2_microbatch_size):
        raise ValueError("IDO task contract Stage-2 microbatch size differs from the resolved CLI/config mode.")
    if experiment.get("official_test_selected_only") is not True or official_test_selected_only is not True or posthoc_oracle_test:
        raise ValueError("IDO task contract requires validation-selected-only test evaluation with no post-hoc oracle test.")
    if experiment.get("path_maps") != [f"{old}={new}" for old, new in path_maps]:
        raise ValueError("IDO task contract path maps differ from CLI.")
    assets = experiment.get("assets")
    if not isinstance(assets, dict):
        raise ValueError("IDO task contract lacks asset hashes.")
    expected_assets = {
        "paths_txt_sha256": file_sha256(input_dir / "paths.txt"),
        "observed_labels_sha256": file_sha256(input_dir / "labels.npy"),
        "v1_resolved_config_sha256": file_sha256(input_dir / "resolved_config.yaml"),
        "noise_index_sha256": file_sha256(noise_index),
        "validation_manifest_csv_sha256": file_sha256(output_dir / "validation_manifest.csv"),
        "validation_manifest_json_sha256": file_sha256(output_dir / "validation_manifest.json"),
    }
    if assets != expected_assets:
        raise ValueError("IDO task contract asset hashes differ from current V1/noise/validation inputs.")
    return payload


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


def load_noise_metadata(noise_index: Path) -> dict[str, Any]:
    """Read immutable noise provenance once and record it in every new result row."""
    rows = [row for row in read_csv(noise_index) if row.get("split", "train").lower() == "train"]
    if not rows:
        raise ValueError(f"Noise index has no training rows: {noise_index}")

    def uniform(column: str, fallback: str = "") -> str:
        values = {str(row.get(column, fallback)) for row in rows}
        if len(values) != 1:
            raise ValueError(f"Noise index has inconsistent {column}: {sorted(values)}")
        return next(iter(values))

    return {
        "noise_strategy": uniform("noise_strategy"),
        "noise_ratio": float(uniform("noise_ratio")),
        "noise_seed": int(uniform("noise_seed")),
        "mapping_type": uniform("mapping_type", "cyclic"),
        "mapping_seed": uniform("mapping_seed", ""),
    }


def validate_budget_method_request(
    methods: list[str],
    pgdf_budget_root: Path | None,
    seeds: list[int],
) -> None:
    budget_matched_methods = {
        "dynamic_budget_matched",
        "dynamic_budget_matched_dynamic_proto",
        *EXACT_BUDGET_CONTROL_METHODS,
        *FUSION_ABLATION_METHODS,
    }
    requested = bool(budget_matched_methods & {canonical_method_key(method) for method in methods})
    if requested and pgdf_budget_root is None:
        raise ValueError("Budget-matched dynamic/fusion methods require --pgdf-budget-root.")
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


def validate_dynamic_pgdf_budget_source_provenance(
    pgdf_budget_root: Path,
    *,
    dataset_name: str,
    noise_metadata: dict[str, Any],
    noise_index_sha256: str,
    noise_index_name: str,
) -> str:
    """Require dynamic-fusion budgets to originate from the exact dynamic PGDF setup.

    The manifest comparison above already proves the validation split and train
    path pool. New reference runs additionally pin dataset and noise-index
    content by SHA256; legacy submission-era sources receive an explicitly
    downgraded filename-plus-manifest provenance status instead.
    """
    source_path = pgdf_budget_root.parent / "resolved_config.yaml"
    if not source_path.exists():
        raise FileNotFoundError(f"Dynamic PGDF budget source config is missing: {source_path}")
    with source_path.open("r", encoding="utf-8") as handle:
        source_cfg = yaml.safe_load(handle) or {}
    source_checkpoint_cfg = source_cfg.get("checkpoint_validation", {})
    source_noise = source_checkpoint_cfg.get("noise_realization")
    source_hash = source_checkpoint_cfg.get("noise_index_sha256")
    if str(source_cfg.get("dataset", {}).get("name", "")) != dataset_name:
        raise ValueError("Dynamic PGDF budget source dataset does not match the current dataset.")
    if not isinstance(source_noise, dict) or not isinstance(source_hash, str) or not source_hash:
        # Submission-era runs predate content-hash logging.  They remain
        # reusable only when their source V1 noise-index filename agrees with
        # the requested one and the exact validation manifest has already
        # matched above.  The downgraded evidence status is written into every
        # new fusion run's budget_source.json rather than represented as an
        # exact content-hash verification.
        source_index_name = Path(str(source_cfg.get("dataset", {}).get("index_file", ""))).name
        if not source_index_name or source_index_name != noise_index_name:
            raise ValueError(
                "Legacy dynamic PGDF budget source lacks content-hash provenance and its noise-index filename "
                "does not match the current request. Run pgdf_dynamic_proto with the current revision entry point."
            )
        return "legacy_manifest_and_noise_index_filename"
    fields = ("noise_strategy", "noise_ratio", "noise_seed", "mapping_type", "mapping_seed")
    mismatches = [field for field in fields if source_noise.get(field) != noise_metadata.get(field)]
    if source_hash != noise_index_sha256:
        mismatches.append("noise_index_sha256")
    if mismatches:
        raise ValueError(
            "Dynamic PGDF budget source does not match the current noise realization; "
            f"mismatched fields: {sorted(mismatches)}."
        )
    return "exact_noise_index_sha256"


def inspect_pgdf_active_budget_source(
    pgdf_budget_root: Path,
    schedule: ClassBudgetSchedule,
) -> dict[str, Any]:
    """Fail closed unless PGDF's recorded active subset agrees with class budgets.

    ``selection_per_class.csv`` supplies only the allowed class-wise count
    schedule.  This companion audit reads the source run's aggregate update
    counts and fallback counts, never sample identities, losses, prototype
    scores, oracle labels, or test metrics.  The PGDF fallback adds exactly
    one member only to an otherwise empty observed-label strict intersection,
    so ``active - fallback`` is the source strict-intersection count.
    """
    source_path = pgdf_budget_root / f"seed{schedule.seed}" / "selection_updates.csv"
    if not source_path.is_file():
        raise FileNotFoundError(
            "PGDF DynamicProto source is missing selection_updates.csv required to audit "
            f"active selected counts and fallback counts: {source_path}"
        )
    rows = read_csv(source_path)
    required = {"epoch", "num_selected", "fallback_count"}
    actual_fields = set(rows[0]) if rows else set()
    missing_fields = sorted(required - actual_fields)
    if missing_fields:
        raise ValueError(
            f"PGDF DynamicProto source update audit is missing columns {missing_fields}: {source_path}"
        )
    expected_epochs = set(schedule.budgets)
    observed: dict[int, dict[str, int]] = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            epoch = int(row["epoch"])
            active_count = int(row["num_selected"])
            fallback_count = int(row["fallback_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid PGDF DynamicProto update audit value at {source_path}:{row_number}."
            ) from exc
        if epoch not in expected_epochs:
            raise ValueError(
                f"Unexpected PGDF DynamicProto update epoch {epoch} in {source_path}; "
                f"expected {sorted(expected_epochs)}."
            )
        if epoch in observed:
            raise ValueError(f"Duplicate PGDF DynamicProto update epoch {epoch} in {source_path}.")
        expected_active_count = sum(int(value) for value in schedule.budgets[epoch].values())
        if active_count != expected_active_count:
            raise ValueError(
                "PGDF DynamicProto active selected count disagrees with its class-wise budget schedule "
                f"at epoch {epoch}: update={active_count}, class_sum={expected_active_count}."
            )
        if fallback_count < 0 or fallback_count > active_count:
            raise ValueError(
                f"Invalid PGDF DynamicProto fallback_count={fallback_count} at epoch {epoch}."
            )
        observed[epoch] = {
            "epoch": epoch,
            "active_selected_count": active_count,
            "fallback_count": fallback_count,
            "derived_strict_intersection_count": active_count - fallback_count,
        }
    if set(observed) != expected_epochs:
        missing_epochs = sorted(expected_epochs - set(observed))
        extra_epochs = sorted(set(observed) - expected_epochs)
        raise ValueError(
            "PGDF DynamicProto update audit is incomplete: "
            f"missing_epochs={missing_epochs}, extra_epochs={extra_epochs}."
        )
    updates = [observed[epoch] for epoch in sorted(observed)]
    return {
        "source_selection_updates_path": str(source_path),
        "source_selection_updates_sha256": file_sha256(source_path),
        "active_count_field": "num_selected",
        "fallback_count_field": "fallback_count",
        "derived_strict_intersection_count": "num_selected - fallback_count",
        "fallback_count_total": sum(row["fallback_count"] for row in updates),
        "active_selected_count_total": sum(row["active_selected_count"] for row in updates),
        "updates": updates,
    }


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


def write_budget_source(
    run_dir: Path,
    schedule: ClassBudgetSchedule,
    *,
    source_method: str,
    source_provenance_status: str = "",
    selection_rule: str = "current-model class-wise small-loss with exact PGDF selected_count",
    uses_current_model_loss: bool = True,
    uses_current_model_prototype: bool = False,
    random_selection_scheme: str = "",
    noise_index_sha256: str = "",
    source_active_budget_audit: dict[str, Any] | None = None,
) -> None:
    """Freeze only PGDF class counts, never its selected sample identities or scores."""
    source_root = Path(schedule.source_path).parents[2]
    source_validation_manifest = source_root / "validation_manifest.json"
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
            "source_method": source_method,
            "source_provenance_status": source_provenance_status,
            "source_validation_manifest_path": str(source_validation_manifest),
            "source_validation_manifest_sha256": (
                file_sha256(source_validation_manifest) if source_validation_manifest.is_file() else ""
            ),
            "source_resolved_config_sha256": (
                file_sha256(source_root / "resolved_config.yaml")
                if (source_root / "resolved_config.yaml").is_file()
                else ""
            ),
            "noise_index_sha256": noise_index_sha256,
            "source_retention_ratio": schedule.source_retention_ratio,
            "source_proto_keep_ratio": schedule.source_proto_keep_ratio,
            "seed": schedule.seed,
            "update_epochs": sorted(schedule.budgets),
            "selection_rule": selection_rule,
            "uses_pgdf_sample_identity": False,
            "uses_current_model_loss": uses_current_model_loss,
            "uses_current_model_prototype": uses_current_model_prototype,
            "uses_prototype_or_graph_scores": uses_current_model_prototype,
            "random_selection_scheme": random_selection_scheme,
            "source_active_budget_audit": source_active_budget_audit or {},
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


def verify_soft_weight_budget_match(
    fusion_per_class_rows: list[dict[str, Any]],
    schedule: ClassBudgetSchedule,
) -> None:
    """Require soft class weights, rather than sample identities, to match PGDF budgets."""
    observed: dict[tuple[int, str], tuple[int, float, int]] = {}
    for row in fusion_per_class_rows:
        key = (int(row["epoch"]), str(row["web_label"]))
        if key in observed:
            raise RuntimeError(f"Duplicate soft-weight fusion class record: {key}.")
        normalized_sum = row.get("normalized_weight_sum", "")
        nonzero_count = row.get("nonzero_weight_count", "")
        if normalized_sum == "" or nonzero_count == "":
            raise RuntimeError(f"Soft-weight fusion row is missing weight statistics: {key}.")
        observed[key] = (int(row["reference_budget"]), float(normalized_sum), int(nonzero_count))
    expected = {
        (epoch, noisy_label): int(selected_count)
        for epoch, class_counts in schedule.budgets.items()
        for noisy_label, selected_count in class_counts.items()
    }
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise RuntimeError(f"Soft-weight fusion budget rows differ from PGDF schedule: missing={missing[:5]}, extra={extra[:5]}.")
    for key, expected_budget in expected.items():
        reference_budget, normalized_sum, nonzero_count = observed[key]
        if reference_budget != expected_budget or not np.isclose(normalized_sum, expected_budget, rtol=0.0, atol=1.0e-8):
            raise RuntimeError(
                f"Soft-weight effective budget mismatch for {key}: "
                f"reference={reference_budget}, normalized_sum={normalized_sum}, expected={expected_budget}."
            )
        if nonzero_count <= 0:
            raise RuntimeError(f"Soft-weight fusion has no positive-weight samples for {key}.")


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
    if args.ido_stage1_epochs is not None and args.ido_stage1_epochs <= 0:
        raise ValueError("--ido-stage1-epochs must be positive when provided.")
    if args.ido_stage2_microbatch_size is not None and args.ido_stage2_microbatch_size <= 0:
        raise ValueError("--ido-stage2-microbatch-size must be positive when provided.")
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
    allowed = set(METHODS) | set(ABLATION_METHODS) | set(CONTROLLED_ADAPTATION_METHODS) | set(LEGACY_METHOD_ALIASES)
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}. Allowed primary methods: {METHODS}; "
            f"explicit ablations: {ABLATION_METHODS}; controlled adaptations: {CONTROLLED_ADAPTATION_METHODS}; "
            "legacy alias: dynamic"
        )
    canonical = [canonical_method_key(method) for method in methods]
    if len(set(canonical)) != len(canonical):
        raise ValueError("--methods contains duplicate or aliased-duplicate methods.")
    return methods


def validate_official_test_request(methods: list[str], selected_only: bool) -> None:
    """Keep strict ablations on their validation-selected one-shot test protocol."""
    canonical = [canonical_method_key(method) for method in methods]
    strict_methods = {
        "proto_only",
        "dynamic_budget_matched",
        "dynamic_budget_matched_dynamic_proto",
        *EXACT_BUDGET_CONTROL_METHODS,
        "dynamic_proto_only",
        "pgdf_dynamic_proto",
        "fixed_proto_warmup_matched",
        *FUSION_ABLATION_METHODS,
    }
    selected_only_supported = (set(METHODS) | set(ABLATION_METHODS) | set(CONTROLLED_ADAPTATION_METHODS)) - {"coteaching", "jocor"}
    requested_strict = sorted(strict_methods & set(canonical))
    if requested_strict and not selected_only:
        raise ValueError(f"{', '.join(requested_strict)} requires --official-test-selected-only.")
    unsupported = sorted(set(canonical) - selected_only_supported)
    if selected_only and unsupported:
        raise ValueError(
            "--official-test-selected-only is not implemented for methods that train two branches: "
            f"{unsupported}."
        )


def validate_ido_pilot_no_official_test_request(
    args: argparse.Namespace,
    canonical_methods: list[str],
    seeds: list[int],
) -> None:
    """Keep the no-test pilot path disjoint from any formal batch result."""
    if not args.ido_pilot_no_official_test:
        return
    if canonical_methods != ["ido"] or len(seeds) != 1:
        raise ValueError("--ido-pilot-no-official-test requires exactly '--methods ido' and one seed.")
    if args.official_test_selected_only or args.posthoc_oracle_test:
        raise ValueError("--ido-pilot-no-official-test cannot request any official-test evaluation.")
    if args.ido_task_contract is not None:
        raise ValueError("--ido-pilot-no-official-test cannot be used with a formal IDO batch task contract.")


def canonical_method_key(method_key: str) -> str:
    return LEGACY_METHOD_ALIASES.get(method_key, method_key)


def resolve_retention_ratio(method_key: str, pgdf_dynamic_ratio: float) -> float:
    canonical = canonical_method_key(method_key)
    if canonical == "dynamic_r08":
        return 0.8
    if canonical == "dynamic_r09":
        return 0.9
    if canonical in {"pgdf_auto", "pgdf_fixed", "pgdf_dynamic_proto", *FUSION_ABLATION_METHODS}:
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
        "not run (IDO pilot safeguard)"
        if args.ido_pilot_no_official_test
        else
        "validation-selected checkpoint only"
        if args.official_test_selected_only
        else "validation-selected, final, and last-5 checkpoints"
    )
    lines = [
        "# Validation-Selected Checkpoint Run",
        "",
        f"- Protocol: `{PROTOCOL_NAME}`.",
        f"- Shared fixed clean validation: {manifest['validation_samples']}/{manifest['source_train_samples']} samples, ratio={manifest['validation_ratio']}, seed={manifest['validation_seed']}.",
        "- During fitting, checkpoint selection uses validation Top-1 only. The official test set is evaluated after fitting unless this is the explicit no-test IDO pilot.",
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
        "stage", "stage_epoch", "wrong_event_total", "stage1_prediction_changes", "stage1_base_snapshot_selected",
        "ce_term", "mse_term", "entropy_term", "bmm_clean_posterior_mean", "bmm_fallback_predictions",
        "bmm_clipped_predictions", "bmm_fallback_classes", "bmm_mode", "bmm_scope",
        "bmm_posterior_mass_deficit_sum", "bmm_posterior_mass_deficit_max",
        "stage2_effective_batch_size", "stage2_microbatch_size", "stage2_gradient_accumulation_enabled",
        "stage2_microbatch_count_total", "stage2_microbatch_count_min", "stage2_microbatch_count_max",
    ]


def result_fields() -> list[str]:
    return [
        "method_key", "method", "seed", "noise_index", "noise_strategy", "noise_ratio", "noise_seed", "mapping_type", "mapping_seed", "validation_seed", "checkpoint_protocol", "official_test_evaluation", "train_samples", "eval_samples", "validation_samples", "test_samples",
        "best_val_epoch", "best_val_top1", "best_val_top5", "validation_selected_test_top1", "validation_selected_test_top5",
        "validation_selected_test_model_a_top1", "validation_selected_test_model_a_top5", "validation_selected_test_model_b_top1", "validation_selected_test_model_b_top5",
        "final_test_top1", "final_test_top5", "final_test_model_a_top1", "final_test_model_a_top5", "final_test_model_b_top1", "final_test_model_b_top5",
        "last5_test_mean", "last5_test_std", "last5_test_top5_mean", "last5_test_top5_std", "oracle_best_test_epoch",
        "oracle_best_test_top1", "oracle_best_test_top5", "oracle_best_to_final_drop", "retention_ratio", "proto_keep_ratio",
        "auto_proto_keep", "auto_proto_jaccard", "warmup_epochs", "update_interval", "candidate_samples", "final_selected_samples",
        "selection_updates", "selection_strategy", "prototype_mode", "fusion_mode", "optimizer_steps_attempted", "optimizer_steps_actual", "scheduler_steps_actual", "scheduler_estimated_total_steps", "scheduler_warmup_steps", "soft_weighting_effective_budget", "backbone_frozen", "lora_updated_before_selection", "budget_matched", "scheduler_retention_ratio", "budget_source_path", "budget_source_sha256",
        "budget_source_retention_ratio", "budget_source_proto_keep_ratio", "budget_match_verified",
        "remember_mode", "remember_rate", "final_remember_rate", "lambda_cor", "selected_count", "selection_ratio",
        "loss_type", "jal_alpha", "jal_beta", "jal_a", "jal_eps", "selection_mode",
        "stage1_epochs", "stage2_epochs", "base_stage1_epoch", "bmm_mode", "bmm_scope", "bmm_degenerate_policy",
        "label_wave_snapshot_policy", "weloss_source", "final_bmm_fallback_classes", "stage1_base_checkpoint",
        "stage2_effective_batch_size", "stage2_microbatch_size", "stage2_gradient_accumulation_enabled",
        "stage2_microbatch_count_total", "stage2_microbatch_count_min", "stage2_microbatch_count_max", "ido_execution_signature",
        "d1_recording_status", "d1_stage1_wall_seconds", "d1_stage2_wall_seconds", "d1_optimizer_loop_wall_seconds",
        "d1_bmm_fit_seconds", "d1_checkpoint_seconds", "d1_official_test_seconds", "d1_end_to_end_wall_seconds",
        "d1_total_wall_seconds", "d1_optimizer_steps_attempted", "d1_optimizer_steps_actual",
        "d1_peak_cuda_allocated_bytes", "d1_peak_cuda_reserved_bytes", "d1_validation_seconds",
        "d1_stage1_processed_samples", "d1_stage2_processed_samples", "d1_stage2_extra_view_forwards",
        "d1_stage2_prediction_prepass_view_forwards",
        "ido_task_id", "ido_task_contract_sha256", "ido_experiment_signature",
    ]


def ido_bmm_fit_fields() -> list[str]:
    return [
        "bmm_mode", "bmm_scope", "fit_round", "stage2_epoch", "observed_label_id", "observed_label", "sample_count",
        "unique_wrong_event_count", "wrong_event_min", "wrong_event_max", "bmm_status", "bmm_reason",
        "bmm_fit_iterations", "alpha_1", "beta_1", "weight_1", "alpha_2", "beta_2", "weight_2",
        "component_1_normalized_mean", "component_2_normalized_mean",
        "component_1_soft_assignment", "component_1_kish_ess", "component_1_weight_times_n",
        "component_1_max_responsibility", "component_1_responsibility_ge_half_count",
        "component_2_soft_assignment", "component_2_kish_ess", "component_2_weight_times_n",
        "component_2_max_responsibility", "component_2_responsibility_ge_half_count",
        "posterior_1_min", "posterior_1_max", "posterior_2_min", "posterior_2_max",
        "cdf_1_min", "cdf_1_max", "cdf_2_min", "cdf_2_max",
        "posterior_mass_sum", "posterior_mass_deficit", "posterior_mass_min", "posterior_mass_max",
        "bmm_prediction_clipped_count",
    ]


def summary_fields() -> list[str]:
    return [
        "method_key", "method", "num_seeds", "mean_best_val_top1", "std_best_val_top1",
        "mean_validation_selected_test_top1", "std_validation_selected_test_top1", "mean_final_test_top1", "std_final_test_top1",
        "mean_last5_test_mean", "std_last5_test_mean", "mean_oracle_best_test_top1", "std_oracle_best_test_top1",
        "mean_oracle_best_to_final_drop", "std_oracle_best_to_final_drop",
    ]


def selection_fields() -> list[str]:
    return [
        "method", "dataset", "seed", "retention_ratio", "proto_keep_ratio", "selection_strategy", "prototype_mode", "epoch",
        "index", "path", "web_label", "loss", "confidence", "proto_score", "loss_selected", "proto_pass", "state",
    ]


def dual_selection_fields() -> list[str]:
    return [
        "method", "seed", "epoch", "remember_rate", "selected_count", "selected_ratio", "top1_a", "top5_a",
        "top1_b", "top5_b", "mean_ab_top1", "mean_ab_top5", "selection_mode",
    ]


def update_fields() -> list[str]:
    return [
        "method", "dataset", "seed", "retention_ratio", "proto_keep_ratio", "auto_proto_jaccard", "selection_strategy", "prototype_mode", "epoch", "num_candidates", "full_training_pool_size", "num_loss_selected",
        "num_proto_pass", "num_selected", "selected_identity_sha256", "fallback_count", "optimizer_steps_attempted_since_previous_update", "optimizer_steps_actual_since_previous_update", "scheduler_steps_since_previous_update", "proto_reject_count", "selected_ratio", "mean_loss_selected", "mean_loss_unselected",
        "mean_loss_proto_rejected", "mean_proto_selected", "mean_proto_unselected", "overlap_with_previous_selection", "overlap_with_centroid",
        "prototype_feature_source", "prototype_feature_dim", "prototype_checksum", "prototype_gate_membership_hash",
        "prototype_mean", "prototype_std", "selection_model_state_checksum", "prototype_model_state_checksum",
        "same_model_state_for_loss_and_prototype", "lora_updated_since_initial",
    ]


def per_class_fields() -> list[str]:
    return [
        "method", "dataset", "seed", "retention_ratio", "proto_keep_ratio", "selection_strategy", "prototype_mode", "epoch", "web_label", "total_count", "loss_selected_count",
        "proto_pass_count", "selected_count", "proto_reject_count", "selected_ratio", "mean_loss_selected", "mean_loss_unselected",
        "mean_loss_proto_rejected", "mean_proto_selected", "mean_proto_unselected",
    ]


def fusion_update_fields() -> list[str]:
    return [
        "method", "dataset", "seed", "epoch", "fusion_mode", "reference_budget_total", "final_selected_total",
        "strict_intersection_total", "union_candidate_total", "budget_fill_total", "final_strict_intersection_total",
        "final_strict_union_total", "effective_weight_budget", "nonzero_weight_total",
        "optimizer_steps_attempted_since_previous_update", "optimizer_steps_actual_since_previous_update",
        "scheduler_steps_since_previous_update",
    ]


def fusion_per_class_fields() -> list[str]:
    return [
        "method", "dataset", "seed", "epoch", "fusion_mode", "web_label", "total_count", "loss_candidate_count",
        "prototype_candidate_count", "strict_intersection_count", "union_candidate_count", "reference_budget",
        "final_selected_count", "budget_fill_count", "final_strict_intersection_count", "final_strict_union_count",
        "final_strict_jaccard", "raw_weight_sum", "normalized_weight_sum", "nonzero_weight_count",
    ]


def fusion_sample_fields() -> list[str]:
    return [
        "method", "dataset", "seed", "epoch", "fusion_mode", "index", "web_label", "loss_reliability",
        "prototype_reliability", "equal_weight_reliability", "union_reliability", "loss_candidate",
        "prototype_candidate", "strict_intersection_candidate", "union_candidate", "final_selected", "sample_weight",
    ]


if __name__ == "__main__":
    main()
