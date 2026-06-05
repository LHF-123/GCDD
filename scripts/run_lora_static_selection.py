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
from gcdd.lora_training import train_dinov2_lora
from gcdd.progress import log_stage
from gcdd.selection_utils import build_mask_from_selection_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv2-LoRA from a path-matched static selection CSV.")
    parser.add_argument("--input-dir", required=True, help="V1 output directory.")
    parser.add_argument("--selection-file", required=True, help="Selection CSV with path,state fields.")
    parser.add_argument("--method-name", default="FINE-DINOv2 feature", help="Method name written to result tables.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/static_selection.")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map stored image root to local root.")
    parser.add_argument("--no-save-checkpoints", action="store_true", help="Do not save best LoRA/head checkpoints.")

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


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    selection_file = Path(args.selection_file)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "static_selection"
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

    log_stage("[1/4] Loading paths, labels, eval split, and static selection.")
    data = load_data(input_dir)
    selection = build_mask_from_selection_rows(read_csv(selection_file), data["train_paths"], require_full_coverage=True)
    log_stage(
        f"[static-selection] matched={selection.matched_count}, selected={selection.selected_count}, "
        f"ratio={selection.selected_count / max(1, len(selection.mask)):.4f}"
    )

    log_stage("[2/4] Training DINOv2-LoRA on static selection.")
    train_logs: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint_path = None
        if not args.no_save_checkpoints:
            checkpoint_path = checkpoint_dir / f"static_seed{seed}_best.pt"
        run_cfg = copy.deepcopy(cfg)
        run_cfg["lora_train"]["seed"] = int(seed)
        result = train_dinov2_lora(
            data["train_paths"],
            data["train_labels"],
            data["eval_paths"],
            data["eval_labels"],
            selection.mask,
            run_cfg,
            method=args.method_name,
            seed=seed,
            path_maps=path_maps,
            checkpoint_path=checkpoint_path,
        )
        train_logs.extend(result.logs)
        result_rows.append(result.summary)
        for name in result.trainable_modules:
            module_rows.append(
                {
                    "method": args.method_name,
                    "seed": seed,
                    "module": name,
                    "trainable_params": result.trainable_params,
                    "total_params": result.total_params,
                }
            )

    log_stage("[3/4] Writing result tables.")
    write_csv(output_dir / "static_selection_train_log.csv", train_logs, train_log_fields())
    write_csv(output_dir / "static_selection_results.csv", result_rows, result_fields())
    write_csv(output_dir / "static_selection_summary.csv", build_method_summary(result_rows), summary_fields())
    write_csv(output_dir / "static_selection_modules.csv", module_rows, ["method", "seed", "module", "trainable_params", "total_params"])

    log_stage("[4/4] Writing run summary.")
    write_summary(output_dir / "run_summary.md", input_dir, selection_file, args, result_rows, selection.selected_count, len(selection.mask))
    write_json(
        output_dir / "static_selection_summary.json",
        {
            "input_dir": str(input_dir),
            "selection_file": str(selection_file),
            "output_dir": str(output_dir),
            "method_name": args.method_name,
            "selected_count": selection.selected_count,
            "train_count": len(selection.mask),
            "seeds": seeds,
            "results": result_rows,
        },
    )
    log_stage(f"Static-selection LoRA results written to {output_dir}")


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
            "batch_size": 32,
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
        best = np.array([float(row["best_top1"]) for row in rows], dtype=np.float64)
        final = np.array([float(row["final_top1"]) for row in rows], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "num_seeds": len(rows),
                "mean_best_top1": float(best.mean()),
                "std_best_top1": float(best.std(ddof=1)) if len(best) > 1 else 0.0,
                "max_best_top1": float(best.max()),
                "mean_final_top1": float(final.mean()),
                "train_samples": int(rows[0]["train_samples"]),
            }
        )
    return summary


def write_summary(path: Path, input_dir: Path, selection_file: Path, args: argparse.Namespace, result_rows: list[dict[str, Any]], selected_count: int, train_count: int) -> None:
    lines = [
        "# Static Selection LoRA Summary",
        "",
        f"- Input dir: {input_dir}",
        f"- Selection file: {selection_file}",
        f"- Method name: {args.method_name}",
        f"- Selected samples: {selected_count}/{train_count}",
        f"- Seeds: {args.seeds}",
        "",
        "| method | seed | best_top1 | final_top1 | train_samples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result_rows:
        lines.append(
            f"| {row['method']} | {int(row['seed'])} | {float(row['best_top1']):.6f} | "
            f"{float(row['final_top1']):.6f} | {int(row['train_samples'])} |"
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
    return ["method", "seed", "epoch", "lr_lora", "lr_head", "loss", "top1", "top5", "train_samples", "eval_samples", "trainable_params", "total_params"]


def result_fields() -> list[str]:
    return ["method", "seed", "train_samples", "eval_samples", "best_epoch", "best_top1", "best_top5", "final_top1", "final_top5", "last5_mean", "last5_std", "trainable_params", "total_params"]


def summary_fields() -> list[str]:
    return ["method", "num_seeds", "mean_best_top1", "std_best_top1", "max_best_top1", "mean_final_top1", "train_samples"]


if __name__ == "__main__":
    main()
