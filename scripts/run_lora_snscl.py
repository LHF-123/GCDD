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

from gcdd.config import deep_update
from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json, write_yaml
from gcdd.lora_snscl import SNSCLRunResult, train_snscl_lora
from gcdd.progress import log_stage
from gcdd.selection_utils import build_gt_clean_mask_from_noise_rows


DEFAULT_SNSCL_CONFIG: dict[str, Any] = {
    "feature": {"backend": "dinov2_vitb14", "device": "auto", "input_size": 448, "local_repo": ""},
    "lora": {"rank": 8, "alpha": 16.0, "dropout": 0.05, "target_modules": "qkv"},
    "lora_train": {
        "epochs": 30,
        "batch_size": 16,
        "eval_batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "lora_lr": 1.0e-4,
        "head_lr": 1.0e-3,
        "weight_decay": 0.05,
        "scheduler": "cosine",
        "warmup_ratio": 0.1,
        "amp": True,
    },
    "snscl": {
        "warmup_epochs": 5,
        "proj_dim": 512,
        "stochastic_hidden_dim": 2048,
        "lambda_kl": 0.001,
        "lambda_ntcl": 1.0,
        "temperature": 0.07,
        "queue_size": 32,
        "queue_start_epoch": 6,
        "reliability_threshold": 0.5,
        "label_ma_alpha": 0.9,
        "projection_lr": 1.0e-4,
        "stochastic_lr": 1.0e-4,
        "max_grad_norm": 1.0,
        "amp_overflow_patience": 5,
        "reliability_save_interval": 5,
        "fail_on_health_check": True,
        "health_check_epoch": 7,
        "gmm_failure_patience": 2,
        "min_queue_fill_ratio": 1.0e-6,
        "min_valid_ntcl_anchors": 1,
        "min_gamma_std": 1.0e-6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone SNSCL-DINOv2+LoRA (adapted).")
    parser.add_argument("--input-dir", required=True, help="V1 output directory containing paths, labels, and resolved_config.yaml.")
    parser.add_argument("--config", required=True, help="SNSCL method YAML.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/snscl_lora.")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    parser.add_argument("--noise-index", help="Optional synthetic-noise index used only for mechanism metrics.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--no-save-checkpoints", action="store_true")

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lora-lr", type=float)
    parser.add_argument("--head-lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"])
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--target-modules")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local-repo")
    parser.add_argument("--input-size", type=int)

    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--proj-dim", type=int)
    parser.add_argument("--stochastic-hidden-dim", type=int)
    parser.add_argument("--lambda-kl", type=float)
    parser.add_argument("--lambda-ntcl", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--queue-size", type=int)
    parser.add_argument("--queue-start-epoch", type=int)
    parser.add_argument("--reliability-threshold", type=float)
    parser.add_argument("--label-ma-alpha", type=float)
    parser.add_argument("--projection-lr", type=float)
    parser.add_argument("--stochastic-lr", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--amp-overflow-patience", type=int)
    parser.add_argument("--reliability-save-interval", type=int)
    parser.add_argument("--health-check-epoch", type=int)
    parser.add_argument("--gmm-failure-patience", type=int)
    parser.add_argument("--min-queue-fill-ratio", type=float)
    parser.add_argument("--min-valid-ntcl-anchors", type=int)
    parser.add_argument("--min-gamma-std", type=float)
    parser.add_argument("--warn-only-health-checks", action="store_true", help="Log health failures without stopping training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "snscl_lora"
    ensure_dir(output_dir)
    ensure_dir(output_dir / "reliability")
    if not args.no_save_checkpoints:
        ensure_dir(output_dir / "checkpoints")

    cfg = resolve_snscl_config(input_dir / "resolved_config.yaml", Path(args.config), args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)
    seeds = parse_int_list(args.seeds)
    path_maps = parse_path_maps(args.path_map)

    log_stage("[1/4] Loading V1 data and optional analysis-only noise index.")
    data = load_data(input_dir)
    gt_clean_mask = None
    if args.noise_index:
        gt_clean_mask = build_gt_clean_mask_from_noise_rows(read_csv(Path(args.noise_index)), data["train_paths"])
    train_mask = np.ones(len(data["train_labels"]), dtype=bool)

    log_stage("[2/4] Running SNSCL-DINOv2+LoRA (adapted).")
    results: list[SNSCLRunResult] = []
    for seed in seeds:
        checkpoint_path = None if args.no_save_checkpoints else output_dir / "checkpoints" / f"snscl_seed{seed}_best.pt"
        latest_checkpoint_path = None if args.no_save_checkpoints else output_dir / "checkpoints" / f"snscl_seed{seed}_latest.pt"
        run_cfg = copy.deepcopy(cfg)
        run_cfg["lora_train"]["seed"] = int(seed)
        epoch_callback = build_epoch_callback(output_dir)
        write_run_status(output_dir, seed, status="running", last_completed_epoch=0)
        try:
            result = train_snscl_lora(
                train_paths=data["train_paths"],
                train_labels=data["train_labels"],
                eval_paths=data["eval_paths"],
                eval_labels=data["eval_labels"],
                train_mask=train_mask,
                cfg=run_cfg,
                method="SNSCL-DINOv2+LoRA (adapted)",
                seed=seed,
                path_maps=path_maps,
                gt_clean_mask=gt_clean_mask,
                checkpoint_path=checkpoint_path,
                latest_checkpoint_path=latest_checkpoint_path,
                epoch_callback=epoch_callback,
            )
        except Exception as exc:
            write_run_status(
                output_dir,
                seed,
                status="failed",
                exception_type=type(exc).__name__,
                exception=str(exc),
            )
            raise
        results.append(result)
        write_run_status(
            output_dir,
            seed,
            status="complete",
            last_completed_epoch=int(result.logs[-1]["epoch"]),
            health_status=str(result.logs[-1]["health_status"]),
            health_reasons=str(result.logs[-1]["health_reasons"]),
        )

    log_stage("[3/4] Writing SNSCL result and mechanism tables.")
    write_outputs(output_dir, seeds, results)
    log_stage("[4/4] Writing run summary.")
    write_run_summary(output_dir / "run_summary.md", input_dir, Path(args.config), args, cfg, results)
    write_json(
        output_dir / "snscl_summary.json",
        {
            "method": "SNSCL-DINOv2+LoRA (adapted)",
            "input_dir": str(input_dir),
            "method_config": str(args.config),
            "noise_index": str(args.noise_index or ""),
            "noise_index_role": "analysis_only",
            "seeds": seeds,
            "results": [result.summary for result in results],
            "summary": build_method_summary([result.summary for result in results]),
        },
    )
    log_stage(f"SNSCL results written to {output_dir}")


def resolve_snscl_config(v1_config_path: Path, method_config_path: Path, args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Resolve defaults < V1 config < SNSCL method config < CLI."""
    cfg = copy.deepcopy(DEFAULT_SNSCL_CONFIG)
    deep_update(cfg, read_yaml(v1_config_path))
    deep_update(cfg, read_yaml(method_config_path))
    if args is not None:
        apply_cli_overrides(cfg, args)
    return cfg


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file is missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    mappings = {
        "feature": ["device", "local_repo", "input_size"],
        "lora": ["rank", "alpha", "dropout", "target_modules"],
        "lora_train": [
            "epochs",
            "batch_size",
            "eval_batch_size",
            "num_workers",
            "lora_lr",
            "head_lr",
            "weight_decay",
            "scheduler",
            "warmup_ratio",
        ],
        "snscl": [
            "warmup_epochs",
            "proj_dim",
            "stochastic_hidden_dim",
            "lambda_kl",
            "lambda_ntcl",
            "temperature",
            "queue_size",
            "queue_start_epoch",
            "reliability_threshold",
            "label_ma_alpha",
            "projection_lr",
            "stochastic_lr",
            "max_grad_norm",
            "amp_overflow_patience",
            "reliability_save_interval",
            "health_check_epoch",
            "gmm_failure_patience",
            "min_queue_fill_ratio",
            "min_valid_ntcl_anchors",
            "min_gamma_std",
        ],
    }
    for section, attributes in mappings.items():
        target = cfg.setdefault(section, {})
        for attribute in attributes:
            value = getattr(args, attribute, None)
            if value is not None:
                target[attribute] = value
    if getattr(args, "warn_only_health_checks", False):
        cfg.setdefault("snscl", {})["fail_on_health_check"] = False


def load_data(input_dir: Path) -> dict[str, Any]:
    required = [input_dir / "paths.txt", input_dir / "labels.npy", input_dir / "eval_paths.txt", input_dir / "eval_labels.npy"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file is missing: {path}")
    train_paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    eval_paths = (input_dir / "eval_paths.txt").read_text(encoding="utf-8").splitlines()
    train_labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    eval_labels = np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str)
    if len(train_paths) != len(train_labels) or len(eval_paths) != len(eval_labels):
        raise ValueError("Path and label lengths do not match.")
    return {"train_paths": train_paths, "train_labels": train_labels, "eval_paths": eval_paths, "eval_labels": eval_labels}


def write_outputs(output_dir: Path, seeds: list[int], results: list[SNSCLRunResult]) -> None:
    logs = [row for result in results for row in result.logs]
    result_rows = [result.summary for result in results]
    reliability_summary = [row for result in results for row in result.reliability_summary_rows]
    queue_rows = [row for result in results for row in result.queue_rows]
    modules = []
    for seed, result in zip(seeds, results):
        modules.extend(
            {
                "method": "SNSCL-DINOv2+LoRA (adapted)",
                "seed": seed,
                "module": module,
                "trainable_params": result.trainable_params,
                "total_params": result.total_params,
            }
            for module in result.trainable_modules
        )
        by_epoch: dict[int, list[dict[str, Any]]] = {}
        for row in result.reliability_rows:
            by_epoch.setdefault(int(row["epoch"]), []).append(row)
        for epoch_id, rows in by_epoch.items():
            write_csv(output_dir / "reliability" / f"snscl_seed{seed}_epoch_{epoch_id:03d}.csv", rows, reliability_fields())

    write_csv(output_dir / "snscl_train_log.csv", logs, train_log_fields())
    write_csv(output_dir / "snscl_results.csv", result_rows, result_fields())
    write_csv(output_dir / "snscl_summary.csv", build_method_summary(result_rows), summary_fields())
    write_csv(output_dir / "snscl_queue_stats.csv", queue_rows, queue_fields())
    write_csv(output_dir / "snscl_reliability_summary.csv", reliability_summary, reliability_summary_fields())
    write_csv(output_dir / "snscl_modules.csv", modules, ["method", "seed", "module", "trainable_params", "total_params"])


def build_epoch_callback(output_dir: Path):
    """Write inspectable per-seed progress files after every completed epoch."""

    def callback(payload: dict[str, Any]) -> None:
        seed = int(payload["seed"])
        write_csv(output_dir / f"snscl_seed{seed}_train_log.csv", payload["logs"], train_log_fields())
        write_csv(output_dir / f"snscl_seed{seed}_queue_stats.csv", payload["queue_rows"], queue_fields())
        write_csv(
            output_dir / f"snscl_seed{seed}_reliability_summary.csv",
            payload["reliability_summary_rows"],
            reliability_summary_fields(),
        )
        by_epoch: dict[int, list[dict[str, Any]]] = {}
        for row in payload["reliability_rows"]:
            by_epoch.setdefault(int(row["epoch"]), []).append(row)
        for epoch_id, rows in by_epoch.items():
            write_csv(output_dir / "reliability" / f"snscl_seed{seed}_epoch_{epoch_id:03d}.csv", rows, reliability_fields())
        write_run_status(
            output_dir,
            seed,
            status="running" if payload["health_status"] == "ok" else "failed",
            method=payload["method"],
            last_completed_epoch=int(payload["epoch"]),
            health_status=payload["health_status"],
            health_reasons=payload["health_reasons"],
        )

    return callback


def write_run_status(output_dir: Path, seed: int, *, status: str, **updates: Any) -> None:
    path = output_dir / f"run_status_seed{seed}.json"
    current: dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            current = json.load(file)
    if status in {"running", "complete"}:
        current.pop("exception_type", None)
        current.pop("exception", None)
    if status == "running" and int(updates.get("last_completed_epoch", -1)) == 0:
        current.pop("health_status", None)
        current.pop("health_reasons", None)
    current.update({"seed": int(seed), "status": status, **updates})
    write_json(path, current)


def build_method_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    best = np.array([float(row["best_top1"]) for row in rows], dtype=np.float64)
    final = np.array([float(row["final_top1"]) for row in rows], dtype=np.float64)
    return [
        {
            "method": "SNSCL-DINOv2+LoRA (adapted)",
            "num_seeds": len(rows),
            "mean_best_top1": float(best.mean()),
            "std_best_top1": float(best.std(ddof=1)) if len(best) > 1 else 0.0,
            "max_best_top1": float(best.max()),
            "mean_final_top1": float(final.mean()),
            "train_samples": int(rows[0]["train_samples"]),
        }
    ]


def write_run_summary(
    path: Path,
    input_dir: Path,
    method_config: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    results: list[SNSCLRunResult],
) -> None:
    lines = [
        "# SNSCL-DINOv2+LoRA (adapted) Summary",
        "",
        f"- Source V1 output: `{input_dir}`",
        f"- Method config: `{method_config}`",
        f"- Seeds: `{args.seeds}`",
        f"- Noise index role: `analysis_only`",
        f"- Epochs / warmup: {cfg['lora_train']['epochs']} / {cfg['snscl']['warmup_epochs']}",
        f"- Queue start epoch / size: {cfg['snscl']['queue_start_epoch']} / {cfg['snscl']['queue_size']}",
        "",
        "| seed | best epoch | best Top-1 | final Top-1 | final gamma | final omega | queue fill |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        row = result.summary
        lines.append(
            f"| {int(row['seed'])} | {int(row['best_epoch'])} | {float(row['best_top1']):.6f} | "
            f"{float(row['final_top1']):.6f} | {float(row['final_mean_gamma']):.6f} | "
            f"{float(row['final_mean_omega']):.6f} | {float(row['final_queue_fill_ratio']):.6f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--seeds must contain at least one seed.")
    return values


def parse_path_maps(items: list[str]) -> list[tuple[str, str]]:
    maps = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must use OLD=NEW format, got: {item}")
        maps.append(tuple(item.split("=", 1)))
    return maps


def train_log_fields() -> list[str]:
    return [
        "method", "seed", "epoch", "lr_lora", "lr_head", "lr_projection", "lr_stochastic", "loss_total", "loss_cls", "loss_ntcl", "loss_kl",
        "mean_gamma", "gamma_std", "mean_omega", "queue_fill_ratio", "num_valid_ntcl_anchors", "valid_anchor_ratio",
        "mean_positive_count", "mean_negative_count", "mean_grad_norm", "amp_skipped_steps", "max_consecutive_amp_skips",
        "amp_scale_final", "amp_scale_min", "amp_scale_max", "amp_overflow_groups", "max_mu_abs", "min_logvar", "max_logvar",
        "max_model_param_abs", "max_projection_param_abs", "max_stochastic_param_abs", "corrected_label_changes",
        "corrected_label_change_ratio", "val_top1", "val_top5", "best_top1", "train_samples",
        "eval_samples", "trainable_params", "total_params", "gmm_success", "consecutive_gmm_failures", "health_status",
        "health_reasons",
    ]


def result_fields() -> list[str]:
    return [
        "method", "seed", "train_samples", "eval_samples", "best_epoch", "best_top1", "best_top5", "final_top1",
        "final_top5", "last5_mean", "last5_std", "final_mean_gamma", "final_mean_omega", "final_queue_fill_ratio",
        "trainable_params", "total_params",
    ]


def summary_fields() -> list[str]:
    return ["method", "num_seeds", "mean_best_top1", "std_best_top1", "max_best_top1", "mean_final_top1", "train_samples"]


def queue_fields() -> list[str]:
    return ["method", "seed", "epoch", "queue_fill_ratio", "filled_entries", "total_capacity", "min_class_count", "max_class_count", "mean_class_count"]


def reliability_summary_fields() -> list[str]:
    return [
        "method", "seed", "epoch", "gmm_success", "gmm_reason", "gmm_mean_0", "gmm_mean_1", "mean_gamma",
        "mean_omega", "threshold_selected", "gamma_clean_auc", "clean_gamma_mean", "noisy_gamma_mean",
        "threshold_purity", "threshold_clean_recall",
    ]


def reliability_fields() -> list[str]:
    return [
        "method", "seed", "epoch", "index", "path", "web_label", "noisy_label_id", "corrected_label_id",
        "noisy_ce_loss", "gamma", "omega", "is_clean",
    ]


if __name__ == "__main__":
    main()
