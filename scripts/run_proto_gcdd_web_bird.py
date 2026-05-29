from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json, write_yaml
from gcdd.progress import log_stage
from gcdd.training import summarize_epoch_logs, train_linear_eval
from tools.build_proto_gcdd_scores import PROTO_SCORE_FIELDS, build_proto_gcdd_rows
from tools.split_clean_otsu import build_split_info, write_split


METHODS = [
    {"method": "Proto only (Otsu)", "score_col": "S_proto", "split_file": "split_proto_only.csv"},
    {"method": "GCDD + Proto", "score_col": "S_gcdd_proto", "split_file": "split_gcdd_proto.csv"},
    {"method": "GCDD + Proto no-I", "score_col": "S_gcdd_proto_noI", "split_file": "split_gcdd_proto_noI.csv"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prototype-aware GCDD clean-score variants on existing V1 Web-Bird outputs.")
    parser.add_argument("--input-dir", default="outputs/Web-Bird/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/proto_gcdd.")
    parser.add_argument("--epsilon", type=float, default=1.0e-6, help="Score product epsilon.")
    parser.add_argument("--otsu-bins", type=int, default=256, help="Otsu histogram bins.")
    parser.add_argument("--clean-ratio-clip", default="0.3,0.9", help="low,high clip values.")
    parser.add_argument("--epochs", type=int, help="Override linear classifier epochs.")
    parser.add_argument("--train-batch-size", type=int, help="Override linear classifier batch size.")
    parser.add_argument("--lr", type=float, help="Override linear classifier learning rate.")
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], help="Override learning-rate scheduler.")
    parser.add_argument("--seed", type=int, help="Override train seed.")
    parser.add_argument("--feature", choices=["cls", "gap", "top"], help="Feature file used for linear training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "proto_gcdd"
    ensure_dir(output_dir)
    split_dir = output_dir / "splits"
    ensure_dir(split_dir)

    cfg = load_resolved_config(input_dir)
    apply_train_overrides(cfg, args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/5] Building prototype-aware GCDD score table.")
    score_rows = build_proto_gcdd_rows(input_dir, epsilon=args.epsilon)
    score_path = output_dir / "proto_gcdd_scores.csv"
    write_csv(score_path, score_rows, PROTO_SCORE_FIELDS)

    log_stage("[2/5] Building Adaptive-Otsu splits from score variants.")
    clean_ratio_clip = parse_clip(args.clean_ratio_clip)
    split_rows = []
    split_masks: dict[str, np.ndarray] = {}
    audit_methods = [{"method": "Full GCDD rebuilt", "score_col": "S_gcdd", "split_file": "split_full_gcdd_rebuilt.csv"}, *METHODS]
    for item in audit_methods:
        split_info = build_split_info(score_rows, item["score_col"], args.otsu_bins, clean_ratio_clip)
        split_path = split_dir / item["split_file"]
        write_split(split_path, score_rows, split_info["state"])
        mask = split_info["state"] == "clean"
        split_masks[item["score_col"]] = mask
        split_rows.append(
            {
                "method": item["method"],
                "score_col": item["score_col"],
                "split_file": str(split_path),
                "train_samples": int(mask.sum()),
                "clean_ratio": float(mask.mean()),
                "clip_hit_rate_low": class_clip_rate(split_info["clip_low"], np.array([row["web_label"] for row in score_rows], dtype=object)),
                "clip_hit_rate_high": class_clip_rate(split_info["clip_high"], np.array([row["web_label"] for row in score_rows], dtype=object)),
            }
        )
    write_csv(output_dir / "proto_gcdd_split_summary.csv", split_rows, ["method", "score_col", "split_file", "train_samples", "clean_ratio", "clip_hit_rate_low", "clip_hit_rate_high"])

    log_stage("[3/5] Training proto-aware clean-only CE variants.")
    train_logs, result_rows = train_variants(input_dir, cfg, split_masks)

    log_stage("[4/5] Writing logs and comparison tables.")
    write_csv(
        output_dir / "train_log.csv",
        train_logs,
        ["method", "epoch", "lr", "loss", "top1", "top5", "train_samples", "eval_samples"],
    )
    write_csv(
        output_dir / "proto_gcdd_results.csv",
        result_rows,
        ["method", "score_col", "train_samples", "eval_samples", "best_epoch", "best_top1", "best_top5", "final_top1", "final_top5", "last10_mean", "last10_std"],
    )
    write_combined_compare(input_dir, output_dir, result_rows)

    log_stage("[5/5] Writing proto-aware GCDD summary.")
    write_summary(output_dir / "run_summary.md", input_dir, args, split_rows, result_rows)
    write_json(
        output_dir / "proto_gcdd_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "epsilon": args.epsilon,
            "otsu_bins": args.otsu_bins,
            "clean_ratio_clip": list(clean_ratio_clip),
            "splits": split_rows,
            "results": result_rows,
        },
    )
    log_stage(f"Prototype-aware GCDD results written to {output_dir}")


def load_resolved_config(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "resolved_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"V1 resolved config is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_train_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    train_cfg = cfg.setdefault("train", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.train_batch_size is not None:
        train_cfg["batch_size"] = args.train_batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.scheduler is not None:
        train_cfg["scheduler"] = args.scheduler
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.feature is not None:
        train_cfg["feature"] = args.feature


def train_variants(input_dir: Path, cfg: dict[str, Any], split_masks: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_name = cfg.get("train", {}).get("feature", "cls")
    train_features = load_feature(input_dir, "", feature_name)
    eval_features = load_feature(input_dir, "eval_", feature_name)
    train_labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    eval_labels = np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str)
    train_logs: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for item in METHODS:
        method = item["method"]
        score_col = item["score_col"]
        logs, _ = train_linear_eval(train_features, train_labels, eval_features, eval_labels, split_masks[score_col], cfg, method)
        train_logs.extend(logs)
        summary = summarize_epoch_logs(method, logs)
        summary["score_col"] = score_col
        result_rows.append(summary)
    return train_logs, result_rows


def load_feature(input_dir: Path, prefix: str, feature_name: str) -> np.ndarray:
    path = input_dir / f"{prefix}features_{feature_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Feature file is missing: {path}")
    return np.load(path)


def write_combined_compare(input_dir: Path, output_dir: Path, result_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    baseline_path = input_dir / "baseline_compare_web_bird.csv"
    if baseline_path.exists():
        for row in read_csv(baseline_path):
            if row["method"] in ["DINOv2 Linear all", "Centroid filtering", "Full GCDD-clean"]:
                out = dict(row)
                out["source"] = "existing_v1"
                out["score_col"] = ""
                rows.append(out)
    for row in result_rows:
        out = dict(row)
        out["source"] = "proto_gcdd"
        rows.append(out)
    write_csv(
        output_dir / "combined_compare_web_bird.csv",
        rows,
        ["source", "method", "score_col", "train_samples", "eval_samples", "best_epoch", "best_top1", "best_top5", "final_top1", "final_top5", "last10_mean", "last10_std"],
    )


def write_summary(path: Path, input_dir: Path, args: argparse.Namespace, split_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> None:
    best = max(result_rows, key=lambda row: float(row["best_top1"])) if result_rows else None
    lines = [
        "# Prototype-Aware GCDD Summary",
        "",
        "This experiment treats the class prototype score as a virtual super-node connection and changes only the clean score.",
        "Training remains clean-only CE on DINOv2 frozen features.",
        "",
        f"- Source V1 output: {input_dir}",
        f"- epsilon: {args.epsilon}",
        f"- otsu_bins: {args.otsu_bins}",
        f"- clean_ratio_clip: {args.clean_ratio_clip}",
        "",
        "## Split Summary",
        "| method | score_col | train_samples | clean_ratio | clip_low | clip_high |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in split_rows:
        lines.append(
            f"| {row['method']} | {row['score_col']} | {row['train_samples']} | {row['clean_ratio']:.4f} | "
            f"{row['clip_hit_rate_low']:.4f} | {row['clip_hit_rate_high']:.4f} |"
        )
    lines.extend(["", "## Training Results", "| method | score_col | best_top1 | final_top1 | last10_mean | best_epoch | train_samples |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in result_rows:
        lines.append(
            f"| {row['method']} | {row['score_col']} | {float(row['best_top1']):.4f} | "
            f"{float(row['final_top1']):.4f} | {float(row['last10_mean']):.4f} | {int(row['best_epoch'])} | {int(row['train_samples'])} |"
        )
    if best is not None:
        lines.extend(["", "## Immediate Read", f"- Best proto-aware variant: {best['method']} with best_top1={float(best['best_top1']):.4f}."])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_clip(raw: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError("--clean-ratio-clip must contain two comma-separated values.")
    if not (0.0 <= parts[0] <= parts[1] <= 1.0):
        raise ValueError("--clean-ratio-clip must satisfy 0 <= low <= high <= 1.")
    return parts[0], parts[1]


def class_clip_rate(flags: np.ndarray, labels: np.ndarray) -> float:
    hit = 0
    unique_labels = sorted(set(labels.tolist()))
    for label in unique_labels:
        idx = labels == label
        hit += int(np.any(flags[idx]))
    return float(hit / len(unique_labels)) if unique_labels else 0.0


if __name__ == "__main__":
    main()
