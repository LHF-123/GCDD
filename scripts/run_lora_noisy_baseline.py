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

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json, write_yaml
from gcdd.lora_noisy_baselines import train_coteaching_lora, train_jocor_lora
from gcdd.progress import log_stage
from gcdd.selection_utils import build_gt_clean_mask_from_noise_rows


def parse_args(default_method: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv2-LoRA noisy-label baselines: Co-teaching or JoCoR.")
    parser.add_argument("--method", choices=["coteaching", "jocor"], default=default_method, required=default_method is None)
    parser.add_argument("--input-dir", required=True, help="V1 output directory.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/<method>_lora.")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    parser.add_argument("--remember-mode", choices=["fixed", "schedule"], default="fixed")
    parser.add_argument("--remember-rate", type=float, default=0.8, help="Fixed remember rate.")
    parser.add_argument("--final-remember-rate", type=float, default=0.6, help="Schedule final remember rate.")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lambda-cor", type=float, default=0.1, help="JoCoR symmetric-KL weight.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--noise-index", help="Optional synthetic noise index for epoch-level purity/recall.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map stored image root to local root.")
    parser.add_argument("--no-save-checkpoints", action="store_true", help="Do not save best checkpoints.")

    parser.add_argument("--epochs", type=int, help="Override LoRA epochs.")
    parser.add_argument("--batch-size", type=int, help="Override LoRA train batch size.")
    parser.add_argument("--eval-batch-size", type=int, help="Override LoRA eval batch size.")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--lora-lr", type=float, help="Override LoRA parameter learning rate.")
    parser.add_argument("--head-lr", type=float, help="Override classifier head learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Override AdamW weight decay.")
    parser.add_argument("--rank", type=int, help="Override LoRA rank.")
    parser.add_argument("--alpha", type=float, help="Override LoRA alpha.")
    parser.add_argument("--dropout", type=float, help="Override LoRA dropout.")
    parser.add_argument("--target-modules", help="Comma-separated target module patterns. Default qkv.")
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], help="Override LoRA scheduler.")
    parser.add_argument("--warmup-ratio", type=float, help="Override LR warmup ratio.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override device.")
    parser.add_argument("--local-repo", help="Override DINOv2 local torch hub repo path.")
    parser.add_argument("--input-size", type=int, help="Override image input size.")
    return parser.parse_args()


def main(default_method: str | None = None) -> None:
    args = parse_args(default_method)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / f"{args.method}_lora"
    ensure_dir(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    if not args.no_save_checkpoints:
        ensure_dir(checkpoint_dir)

    seeds = parse_int_list(args.seeds)
    path_maps = parse_path_maps(args.path_map)
    cfg = load_config(input_dir)
    apply_lora_defaults(cfg)
    apply_overrides(cfg, args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/4] Loading paths, labels, and optional GT-noise mask.")
    data = load_data(input_dir)
    gt_clean_mask = build_gt_clean_mask_from_noise_rows(read_csv(Path(args.noise_index)), data["train_paths"]) if args.noise_index else None
    train_mask = np.ones(len(data["train_labels"]), dtype=bool)

    log_stage(f"[2/4] Running {args.method} DINOv2-LoRA.")
    train_logs: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint_path = None
        if not args.no_save_checkpoints:
            checkpoint_path = checkpoint_dir / f"{args.method}_seed{seed}_best.pt"
        run_cfg = copy.deepcopy(cfg)
        run_cfg["lora_train"]["seed"] = int(seed)
        method_name = build_method_name(args)
        train_fn = train_coteaching_lora if args.method == "coteaching" else train_jocor_lora
        kwargs = {
            "train_paths": data["train_paths"],
            "train_labels": data["train_labels"],
            "eval_paths": data["eval_paths"],
            "eval_labels": data["eval_labels"],
            "train_mask": train_mask,
            "cfg": run_cfg,
            "method": method_name,
            "seed": seed,
            "remember_mode": args.remember_mode,
            "remember_rate": args.remember_rate,
            "final_remember_rate": args.final_remember_rate,
            "warmup_epochs": args.warmup_epochs,
            "grad_accum_steps": args.grad_accum_steps,
            "path_maps": path_maps,
            "gt_clean_mask": gt_clean_mask,
            "checkpoint_path": checkpoint_path,
        }
        if args.method == "jocor":
            kwargs["lambda_cor"] = args.lambda_cor
        result = train_fn(**kwargs)
        train_logs.extend(result.logs)
        result_rows.append(result.summary)
        for item in result.trainable_modules:
            module_rows.append(
                {
                    "method": method_name,
                    "seed": seed,
                    "model": item["model"],
                    "module": item["module"],
                    "trainable_params": result.trainable_params,
                    "total_params": result.total_params,
                }
            )

    log_stage("[3/4] Writing result tables.")
    prefix = args.method
    write_csv(output_dir / f"{prefix}_train_log.csv", train_logs, train_log_fields())
    write_csv(output_dir / f"{prefix}_results.csv", result_rows, result_fields())
    write_csv(output_dir / f"{prefix}_summary.csv", build_method_summary(result_rows), summary_fields())
    write_csv(output_dir / f"{prefix}_modules.csv", module_rows, ["method", "seed", "model", "module", "trainable_params", "total_params"])

    log_stage("[4/4] Writing run summary.")
    write_summary(output_dir / "run_summary.md", args, result_rows)
    write_json(
        output_dir / f"{prefix}_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "method": args.method,
            "seeds": seeds,
            "remember_mode": args.remember_mode,
            "remember_rate": args.remember_rate,
            "final_remember_rate": args.final_remember_rate,
            "lambda_cor": args.lambda_cor if args.method == "jocor" else "",
            "grad_accum_steps": args.grad_accum_steps,
            "results": result_rows,
        },
    )
    log_stage(f"{args.method} results written to {output_dir}")


def build_method_name(args: argparse.Namespace) -> str:
    if args.method == "coteaching":
        suffix = f"{args.remember_mode} r={args.remember_rate:g}" if args.remember_mode == "fixed" else f"schedule final={args.final_remember_rate:g}"
        return f"Co-teaching-DINOv2+LoRA {suffix}"
    return f"JoCoR-DINOv2+LoRA r={args.remember_rate:g} lambda={args.lambda_cor:g}"


def load_data(input_dir: Path) -> dict[str, Any]:
    required = [input_dir / "paths.txt", input_dir / "labels.npy", input_dir / "eval_paths.txt", input_dir / "eval_labels.npy"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file is missing: {path}")
    train_paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    eval_paths = (input_dir / "eval_paths.txt").read_text(encoding="utf-8").splitlines()
    train_labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    eval_labels = np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str)
    if len(train_paths) != len(train_labels):
        raise ValueError("paths.txt and labels.npy lengths do not match.")
    if len(eval_paths) != len(eval_labels):
        raise ValueError("eval_paths.txt and eval_labels.npy lengths do not match.")
    return {"train_paths": train_paths, "train_labels": train_labels, "eval_paths": eval_paths, "eval_labels": eval_labels}


def load_config(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "resolved_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"V1 resolved config is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_lora_defaults(cfg: dict[str, Any]) -> None:
    cfg.setdefault("feature", {})
    cfg["feature"].setdefault("backend", "dinov2_vitb14")
    cfg["feature"].setdefault("device", "auto")
    cfg["feature"].setdefault("input_size", 448)
    cfg["feature"].setdefault("local_repo", "")
    cfg.setdefault("lora", {"rank": 8, "alpha": 16.0, "dropout": 0.05, "target_modules": "qkv"})
    cfg.setdefault(
        "lora_train",
        {
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
            "seed": 42,
            "amp": True,
        },
    )


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    feature_cfg = cfg.setdefault("feature", {})
    lora_cfg = cfg.setdefault("lora", {})
    train_cfg = cfg.setdefault("lora_train", {})
    for attr, target, key in [
        ("device", feature_cfg, "device"),
        ("local_repo", feature_cfg, "local_repo"),
        ("input_size", feature_cfg, "input_size"),
        ("epochs", train_cfg, "epochs"),
        ("batch_size", train_cfg, "batch_size"),
        ("eval_batch_size", train_cfg, "eval_batch_size"),
        ("num_workers", train_cfg, "num_workers"),
        ("lora_lr", train_cfg, "lora_lr"),
        ("head_lr", train_cfg, "head_lr"),
        ("weight_decay", train_cfg, "weight_decay"),
        ("scheduler", train_cfg, "scheduler"),
        ("warmup_ratio", train_cfg, "warmup_ratio"),
        ("rank", lora_cfg, "rank"),
        ("alpha", lora_cfg, "alpha"),
        ("dropout", lora_cfg, "dropout"),
        ("target_modules", lora_cfg, "target_modules"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            target[key] = value


def build_method_summary(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    summary = []
    for method, rows in by_method.items():
        best = np.array([float(row["best_mean_ab_top1"]) for row in rows], dtype=np.float64)
        final = np.array([float(row["final_mean_ab_top1"]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "num_seeds": len(rows),
                "mean_best_mean_ab_top1": float(best.mean()),
                "std_best_mean_ab_top1": float(best.std(ddof=1)) if len(best) > 1 else 0.0,
                "max_best_mean_ab_top1": float(best.max()),
                "mean_final_mean_ab_top1": float(final.mean()),
                "train_samples": int(rows[0]["train_samples"]),
            }
        )
    return summary


def write_summary(path: Path, args: argparse.Namespace, result_rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {args.method} DINOv2-LoRA Summary",
        "",
        f"- Remember mode: {args.remember_mode}",
        f"- Remember rate: {args.remember_rate}",
        f"- Final remember rate: {args.final_remember_rate}",
        f"- Warmup epochs: {args.warmup_epochs}",
        f"- Lambda cor: {args.lambda_cor if args.method == 'jocor' else ''}",
        f"- Grad accumulation steps: {args.grad_accum_steps}",
        "",
        "| method | seed | best_mean_ab_top1 | model_a | model_b | final_mean_ab_top1 | selected_ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['method']} | {int(row['seed'])} | {float(row['best_mean_ab_top1']):.6f} | "
            f"{float(row['best_model_a_top1']):.6f} | {float(row['best_model_b_top1']):.6f} | "
            f"{float(row['final_mean_ab_top1']):.6f} | {float(row['final_selected_ratio']):.6f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def parse_path_maps(items: list[str]) -> list[tuple[str, str]]:
    maps = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must use OLD=NEW format, got: {item}")
        old, new = item.split("=", 1)
        maps.append((old, new))
    return maps


def train_log_fields() -> list[str]:
    return [
        "method",
        "seed",
        "epoch",
        "lr_lora",
        "lr_head",
        "loss",
        "remember_rate",
        "selected_count",
        "selected_ratio",
        "selected_clean",
        "selected_purity",
        "clean_recall",
        "top1_a",
        "top5_a",
        "top1_b",
        "top5_b",
        "mean_ab_top1",
        "mean_ab_top5",
        "train_samples",
        "eval_samples",
        "trainable_params",
        "total_params",
    ]


def result_fields() -> list[str]:
    return [
        "method",
        "seed",
        "remember_mode",
        "remember_rate",
        "final_remember_rate",
        "lambda_cor",
        "warmup_epochs",
        "train_samples",
        "eval_samples",
        "best_epoch",
        "best_mean_ab_top1",
        "best_model_a_top1",
        "best_model_b_top1",
        "final_mean_ab_top1",
        "final_model_a_top1",
        "final_model_b_top1",
        "last5_mean",
        "last5_std",
        "final_selected_ratio",
        "final_selected_purity",
        "final_clean_recall",
        "trainable_params",
        "total_params",
    ]


def summary_fields() -> list[str]:
    return ["method", "num_seeds", "mean_best_mean_ab_top1", "std_best_mean_ab_top1", "max_best_mean_ab_top1", "mean_final_mean_ab_top1", "train_samples"]


if __name__ == "__main__":
    main()
