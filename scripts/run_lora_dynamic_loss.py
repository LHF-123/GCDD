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
from gcdd.lora_dynamic import train_dynamic_loss_lora
from gcdd.progress import log_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv2-LoRA with dynamic class-wise small-loss filtering.")
    parser.add_argument("--input-dir", default="outputs/Web-Car/v1_web_car_0.9_448", help="V1 output directory.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/dynamic_loss_42.")
    parser.add_argument("--retention-ratios", default="0.9", help="Comma-separated class-wise retention ratios, e.g. 0.8,0.9.")
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Epochs trained on all candidates before first selection update.")
    parser.add_argument("--update-interval", type=int, default=5, help="Selection update interval in epochs after warm-up.")
    parser.add_argument("--centroid-split", default="centroid_filtering_split.csv", help="Optional centroid split file relative to input-dir for overlap.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map stored image root to local root.")
    parser.add_argument("--no-save-checkpoints", action="store_true", help="Do not save best LoRA/head checkpoint files.")

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
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "dynamic_loss_42"
    ensure_dir(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    if not args.no_save_checkpoints:
        ensure_dir(checkpoint_dir)

    ratios = parse_float_list(args.retention_ratios, "--retention-ratios")
    seeds = parse_int_list(args.seeds, "--seeds")
    path_maps = parse_path_maps(args.path_map)

    cfg = load_config(input_dir)
    apply_lora_defaults(cfg)
    apply_overrides(cfg, args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/4] Loading V1 paths, labels, and optional centroid split.")
    data = load_data(input_dir, args.centroid_split)

    log_stage("[2/4] Running dynamic small-loss LoRA baselines.")
    train_logs: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for ratio in ratios:
        for seed in seeds:
            ratio_text = ratio_to_text(ratio)
            method = f"DINOv2 LoRA dynamic loss r={ratio:g}"
            checkpoint_path = None
            if not args.no_save_checkpoints:
                checkpoint_path = checkpoint_dir / f"dynamic_loss_r{ratio_text}_seed{seed}_best.pt"
            run_cfg = copy.deepcopy(cfg)
            run_cfg["lora_train"]["seed"] = int(seed)
            result = train_dynamic_loss_lora(
                data["train_paths"],
                data["train_labels"],
                data["eval_paths"],
                data["eval_labels"],
                data["candidate_mask"],
                run_cfg,
                method=method,
                seed=seed,
                retention_ratio=ratio,
                warmup_epochs=int(args.warmup_epochs),
                update_interval=int(args.update_interval),
                path_maps=path_maps,
                centroid_mask=data["centroid_mask"],
                checkpoint_path=checkpoint_path,
            )
            train_logs.extend(result.logs)
            result_rows.append(result.summary)
            update_rows.extend(result.update_rows)
            per_class_rows.extend(result.per_class_rows)
            write_selection_files(output_dir, ratio_text, seed, result.selection_rows)
            for name in result.trainable_modules:
                module_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "retention_ratio": ratio,
                        "module": name,
                        "trainable_params": result.trainable_params,
                        "total_params": result.total_params,
                    }
                )

    log_stage("[3/4] Writing dynamic-loss result tables.")
    write_csv(output_dir / "dynamic_loss_train_log.csv", train_logs, train_log_fields())
    write_csv(output_dir / "dynamic_loss_results.csv", result_rows, result_fields())
    write_csv(output_dir / "dynamic_loss_update_summary.csv", update_rows, update_fields())
    write_csv(output_dir / "dynamic_loss_per_class_summary.csv", per_class_rows, per_class_fields())
    write_csv(output_dir / "dynamic_loss_modules.csv", module_rows, ["method", "seed", "retention_ratio", "module", "trainable_params", "total_params"])
    method_summary = build_method_summary(result_rows)
    write_csv(output_dir / "dynamic_loss_summary.csv", method_summary, ["method", "retention_ratio", "num_seeds", "mean_best_top1", "std_best_top1", "max_best_top1", "mean_final_top1", "final_selected_samples"])

    log_stage("[4/4] Writing dynamic-loss run summary.")
    write_summary(output_dir / "run_summary.md", input_dir, args, cfg, method_summary, result_rows, update_rows)
    write_json(
        output_dir / "dynamic_loss_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "retention_ratios": ratios,
            "seeds": seeds,
            "warmup_epochs": int(args.warmup_epochs),
            "update_interval": int(args.update_interval),
            "results": result_rows,
            "summary": method_summary,
            "updates": update_rows,
        },
    )
    log_stage(f"Dynamic loss results written to {output_dir}")


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
    cfg["feature"].setdefault("input_size", 224)
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
            "seed": 1,
            "amp": True,
        },
    )


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    feature_cfg = cfg.setdefault("feature", {})
    lora_cfg = cfg.setdefault("lora", {})
    train_cfg = cfg.setdefault("lora_train", {})
    if args.device is not None:
        feature_cfg["device"] = args.device
    if args.local_repo is not None:
        feature_cfg["local_repo"] = args.local_repo
    if args.input_size is not None:
        feature_cfg["input_size"] = args.input_size
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        train_cfg["eval_batch_size"] = args.eval_batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.lora_lr is not None:
        train_cfg["lora_lr"] = args.lora_lr
    if args.head_lr is not None:
        train_cfg["head_lr"] = args.head_lr
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.scheduler is not None:
        train_cfg["scheduler"] = args.scheduler
    if args.warmup_ratio is not None:
        train_cfg["warmup_ratio"] = args.warmup_ratio
    if args.rank is not None:
        lora_cfg["rank"] = args.rank
    if args.alpha is not None:
        lora_cfg["alpha"] = args.alpha
    if args.dropout is not None:
        lora_cfg["dropout"] = args.dropout
    if args.target_modules is not None:
        lora_cfg["target_modules"] = args.target_modules


def load_data(input_dir: Path, centroid_split: str) -> dict[str, Any]:
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

    centroid_path = input_dir / centroid_split if centroid_split else None
    centroid_mask = read_clean_mask(centroid_path, len(train_labels)) if centroid_path and centroid_path.exists() else None
    return {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "eval_paths": eval_paths,
        "eval_labels": eval_labels,
        "candidate_mask": np.ones(len(train_labels), dtype=bool),
        "centroid_mask": centroid_mask,
    }


def read_clean_mask(path: Path, n: int) -> np.ndarray:
    rows = read_csv(path)
    if len(rows) != n:
        raise ValueError(f"{path} length {len(rows)} does not match expected length {n}.")
    mask = np.zeros(n, dtype=bool)
    for row in rows:
        idx = int(row["index"])
        if idx < 0 or idx >= n:
            raise ValueError(f"Index {idx} in {path} is outside [0, {n}).")
        mask[idx] = row["state"] == "clean"
    return mask


def write_selection_files(output_dir: Path, ratio_text: str, seed: int, rows: list[dict[str, Any]]) -> None:
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)
    for epoch, epoch_rows in sorted(by_epoch.items()):
        path = output_dir / f"dynamic_loss_selection_r{ratio_text}_seed{seed}_epoch_{epoch:03d}.csv"
        write_csv(path, epoch_rows, selection_fields())


def build_method_summary(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in result_rows:
        by_key.setdefault((str(row["method"]), float(row["retention_ratio"])), []).append(row)
    out = []
    for (method, ratio), rows in by_key.items():
        best = np.array([float(row["best_top1"]) for row in rows], dtype=np.float64)
        final = np.array([float(row["final_top1"]) for row in rows], dtype=np.float64)
        out.append(
            {
                "method": method,
                "retention_ratio": float(ratio),
                "num_seeds": int(len(rows)),
                "mean_best_top1": float(best.mean()),
                "std_best_top1": float(best.std(ddof=1)) if len(best) > 1 else 0.0,
                "max_best_top1": float(best.max()),
                "mean_final_top1": float(final.mean()),
                "final_selected_samples": int(rows[0]["final_selected_samples"]),
            }
        )
    return out


def write_summary(
    path: Path,
    input_dir: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]],
) -> None:
    resolved_epochs = int(cfg["lora_train"]["epochs"])
    lines = [
        "# Dynamic Loss Filtering LoRA Summary",
        "",
        "This baseline trains DINOv2-LoRA while periodically selecting class-wise small-loss samples.",
        "It is a minimal dynamic small-loss baseline, not a full DivideMix reproduction.",
        "",
        f"- Source V1 output: {input_dir}",
        f"- Retention ratios: {args.retention_ratios}",
        f"- Seeds: {args.seeds}",
        f"- Warm-up epochs: {args.warmup_epochs}",
        f"- Update interval: {args.update_interval}",
        f"- Total epochs: {resolved_epochs}",
        "",
        "## Method Summary",
        "| method | ratio | seeds | mean_best_top1 | std_best_top1 | mean_final_top1 | final_selected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {float(row['retention_ratio']):.3f} | {int(row['num_seeds'])} | "
            f"{float(row['mean_best_top1']):.6f} | {float(row['std_best_top1']):.6f} | "
            f"{float(row['mean_final_top1']):.6f} | {int(row['final_selected_samples'])} |"
        )

    if update_rows:
        lines.extend(["", "## Selection Updates", "| method | seed | ratio | epoch | selected | prev_jaccard | centroid_jaccard |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in update_rows:
            centroid = row["overlap_with_centroid"]
            centroid_text = f"{float(centroid):.6f}" if centroid != "" else ""
            lines.append(
                f"| {row['method']} | {int(row['seed'])} | {float(row['retention_ratio']):.3f} | "
                f"{int(row['epoch'])} | {int(row['num_selected'])} | "
                f"{float(row['overlap_with_previous_selection']):.6f} | {centroid_text} |"
            )
    lines.extend(
        [
            "",
            "## Output Files",
            "- `dynamic_loss_results.csv`: per-ratio, per-seed final metrics.",
            "- `dynamic_loss_train_log.csv`: epoch-level training log.",
            "- `dynamic_loss_update_summary.csv`: selection stability and overlap metrics per update.",
            "- `dynamic_loss_per_class_summary.csv`: class-wise selected counts per update.",
            "- `dynamic_loss_selection_r*_seed*_epoch_*.csv`: per-sample loss, confidence, and selected state at each update.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def parse_int_list(raw: str, name: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def parse_path_maps(items: list[str]) -> list[tuple[str, str]]:
    maps = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must use OLD=NEW format, got: {item}")
        old, new = item.split("=", 1)
        maps.append((old, new))
    return maps


def ratio_to_text(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def train_log_fields() -> list[str]:
    return [
        "method",
        "seed",
        "epoch",
        "lr_lora",
        "lr_head",
        "loss",
        "top1",
        "top5",
        "train_samples",
        "candidate_samples",
        "selected_ratio",
        "eval_samples",
        "trainable_params",
        "total_params",
    ]


def result_fields() -> list[str]:
    return [
        "method",
        "seed",
        "retention_ratio",
        "warmup_epochs",
        "update_interval",
        "candidate_samples",
        "final_selected_samples",
        "selection_updates",
        "train_samples",
        "eval_samples",
        "best_epoch",
        "best_top1",
        "best_top5",
        "final_top1",
        "final_top5",
        "last5_mean",
        "last5_std",
        "trainable_params",
        "total_params",
    ]


def update_fields() -> list[str]:
    return [
        "method",
        "seed",
        "retention_ratio",
        "epoch",
        "num_candidates",
        "num_selected",
        "selected_ratio",
        "mean_loss_selected",
        "mean_loss_unselected",
        "overlap_with_previous_selection",
        "overlap_with_centroid",
    ]


def per_class_fields() -> list[str]:
    return [
        "method",
        "seed",
        "retention_ratio",
        "epoch",
        "web_label",
        "total_count",
        "selected_count",
        "selected_ratio",
        "mean_loss_selected",
        "mean_loss_unselected",
    ]


def selection_fields() -> list[str]:
    return ["method", "seed", "retention_ratio", "epoch", "index", "path", "web_label", "loss", "confidence", "state"]


if __name__ == "__main__":
    main()
