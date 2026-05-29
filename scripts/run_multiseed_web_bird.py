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
from gcdd.progress import log_stage
from gcdd.training import summarize_epoch_logs, train_linear_eval


METHODS = [
    {
        "method": "Centroid filtering",
        "split": "centroid_filtering_split.csv",
    },
    {
        "method": "Full GCDD-clean",
        "split": "full_gcdd_clean_split.csv",
    },
    {
        "method": "GCDD + Proto",
        "split": "proto_gcdd/splits/split_gcdd_proto.csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed verification for Web-Bird clean-selection methods.")
    parser.add_argument("--input-dir", default="outputs/Web-Bird/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/multiseed.")
    parser.add_argument("--seeds", default="1,2,3", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, help="Override linear classifier epochs.")
    parser.add_argument("--train-batch-size", type=int, help="Override linear classifier batch size.")
    parser.add_argument("--lr", type=float, help="Override linear classifier learning rate.")
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], help="Override learning-rate scheduler.")
    parser.add_argument("--feature", choices=["cls", "gap", "top"], help="Feature file used for linear training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "multiseed"
    ensure_dir(output_dir)

    seeds = parse_seeds(args.seeds)
    cfg = load_resolved_config(input_dir)
    apply_train_overrides(cfg, args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/4] Loading features, labels, and split masks.")
    data = load_data(input_dir, cfg)

    log_stage("[2/4] Running multi-seed training.")
    train_logs, result_rows = run_multiseed(data, cfg, seeds)

    log_stage("[3/4] Writing multi-seed result tables.")
    write_csv(
        output_dir / "train_log.csv",
        train_logs,
        ["method", "seed", "epoch", "lr", "loss", "top1", "top5", "train_samples", "eval_samples"],
    )
    write_csv(
        output_dir / "multiseed_results.csv",
        result_rows,
        [
            "method",
            "seed",
            "train_samples",
            "eval_samples",
            "best_epoch",
            "best_top1",
            "best_top5",
            "final_top1",
            "final_top5",
            "last10_mean",
            "last10_std",
        ],
    )
    summary_rows = build_summary_rows(result_rows, seeds)
    summary_fields = ["method", *[f"seed{seed}" for seed in seeds], "mean", "std"]
    write_csv(output_dir / "multiseed_summary.csv", summary_rows, summary_fields)

    log_stage("[4/4] Writing multi-seed summary.")
    write_summary(output_dir / "run_summary.md", input_dir, seeds, summary_rows, result_rows)
    write_json(
        output_dir / "multiseed_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "seeds": seeds,
            "results": result_rows,
            "summary": summary_rows,
        },
    )
    log_stage(f"Multi-seed verification written to {output_dir}")


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


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
    if args.feature is not None:
        train_cfg["feature"] = args.feature


def load_data(input_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    feature_name = cfg.get("train", {}).get("feature", "cls")
    train_features = load_feature(input_dir, "", feature_name)
    eval_features = load_feature(input_dir, "eval_", feature_name)
    train_labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    eval_labels = np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str)
    if train_features.shape[0] != len(train_labels):
        raise ValueError("Train feature and label lengths do not match.")
    if eval_features.shape[0] != len(eval_labels):
        raise ValueError("Eval feature and label lengths do not match.")

    split_masks = {}
    for item in METHODS:
        split_path = input_dir / item["split"]
        split_masks[item["method"]] = read_clean_mask(split_path, len(train_labels))
    return {
        "train_features": train_features,
        "eval_features": eval_features,
        "train_labels": train_labels,
        "eval_labels": eval_labels,
        "split_masks": split_masks,
    }


def load_feature(input_dir: Path, prefix: str, feature_name: str) -> np.ndarray:
    path = input_dir / f"{prefix}features_{feature_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Feature file is missing: {path}")
    return np.load(path)


def read_clean_mask(path: Path, n: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Split file is missing: {path}")
    mask = np.zeros(n, dtype=bool)
    rows = read_csv(path)
    if len(rows) != n:
        raise ValueError(f"{path} length {len(rows)} does not match expected train length {n}.")
    for row in rows:
        idx = int(row["index"])
        if idx < 0 or idx >= n:
            raise ValueError(f"Index {idx} in {path} is outside [0, {n}).")
        mask[idx] = row["state"] == "clean"
    return mask


def run_multiseed(data: dict[str, Any], cfg: dict[str, Any], seeds: list[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_logs: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for item in METHODS:
        method = item["method"]
        train_mask = data["split_masks"][method]
        for seed in seeds:
            seed_cfg = copy.deepcopy(cfg)
            seed_cfg.setdefault("train", {})["seed"] = int(seed)
            run_name = f"{method} seed={seed}"
            logs, _ = train_linear_eval(
                data["train_features"],
                data["train_labels"],
                data["eval_features"],
                data["eval_labels"],
                train_mask,
                seed_cfg,
                run_name,
            )
            for row in logs:
                out = dict(row)
                out["method"] = method
                out["seed"] = int(seed)
                train_logs.append(out)
            summary = summarize_epoch_logs(method, logs)
            summary["seed"] = int(seed)
            result_rows.append(summary)
    return train_logs, result_rows


def build_summary_rows(result_rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    rows = []
    by_method: dict[str, dict[int, float]] = {}
    for row in result_rows:
        by_method.setdefault(str(row["method"]), {})[int(row["seed"])] = float(row["best_top1"])

    for item in METHODS:
        method = item["method"]
        seed_values = by_method.get(method, {})
        values = np.array([seed_values[seed] for seed in seeds], dtype=np.float64)
        out: dict[str, Any] = {"method": method}
        for seed, value in zip(seeds, values):
            out[f"seed{seed}"] = float(value)
        out["mean"] = float(values.mean())
        out["std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(out)
    return rows


def write_summary(path: Path, input_dir: Path, seeds: list[int], summary_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> None:
    header = "| method | " + " | ".join(f"seed{seed}" for seed in seeds) + " | mean | std |"
    divider = "| --- | " + " | ".join("---:" for _ in seeds) + " | ---: | ---: |"
    lines = [
        "# Multi-Seed Verification Summary",
        "",
        "This run repeats only the three selected methods with different linear-classifier random seeds.",
        "Features, clean splits, evaluation set, scheduler, and other training hyperparameters are kept fixed.",
        "",
        f"- Source V1 output: {input_dir}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Best Top-1",
        header,
        divider,
    ]
    for row in summary_rows:
        values = " | ".join(f"{float(row[f'seed{seed}']):.6f}" for seed in seeds)
        lines.append(f"| {row['method']} | {values} | {float(row['mean']):.6f} | {float(row['std']):.6f} |")

    best = max(summary_rows, key=lambda row: float(row["mean"]))
    lines.extend(
        [
            "",
            "## Immediate Read",
            f"- Best mean method: {best['method']} with mean={float(best['mean']):.6f}, std={float(best['std']):.6f}.",
            "",
            "## Output Files",
            "- `multiseed_results.csv`: per-method, per-seed full metrics.",
            "- `multiseed_summary.csv`: compact table requested for mean/std comparison.",
            "- `train_log.csv`: epoch-level logs for all runs.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
