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


GATE_METHODS = {
    "gcdd_qgate": "GCDD + Q_same gate",
    "gcdd_pgate": "GCDD + centroid gate",
    "gcdd_qp_gate": "GCDD + Q_same AND centroid gate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally train V1.6 gated GCDD splits.")
    parser.add_argument("--input-dir", default="outputs/Web-Bird/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--output-dir", help="V1.6 output directory. Defaults to <input-dir>/v1_6_gated_splits.")
    parser.add_argument("--low-ratio", type=float, default=0.3, help="Per-class bottom ratio used for low Q_same and low centroid gates.")
    parser.add_argument("--train", action="store_true", help="Train the three gated splits after writing stats and split files.")
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
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "v1_6_gated_splits"
    ensure_dir(output_dir)

    validate_ratio(args.low_ratio)
    data = load_v1_outputs(input_dir)
    cfg = load_resolved_config(input_dir)
    apply_train_overrides(cfg, args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/4] Computing per-class low-Q and low-centroid flags.")
    flags = compute_low_flags(data["labels"], data["Q_same"], data["centroid_score"], args.low_ratio)
    groups = assign_overlap_groups(data["gcdd_clean"], data["centroid_clean"])
    masks = build_gated_masks(data["gcdd_clean"], flags)

    log_stage("[2/4] Writing suspicious-ratio stats and gated split files.")
    summary_rows = write_suspicious_ratio_summary(output_dir / "suspicious_ratio_summary.csv", data, flags, groups)
    per_class_rows = write_per_class_gate_summary(output_dir / "per_class_gate_summary.csv", data, flags, masks)
    write_gate_detail(output_dir / "gated_sample_flags.csv", data, flags, masks, groups)
    for key, mask in masks.items():
        write_split(output_dir / f"{key}_split.csv", data, mask)

    method_summaries: list[dict[str, Any]] = []
    train_logs: list[dict[str, Any]] = []
    if args.train:
        log_stage("[3/4] Training gated splits without supplementing removed samples.")
        train_logs, method_summaries = train_gated_splits(input_dir, output_dir, cfg, data, masks)
        write_csv(
            output_dir / "train_log.csv",
            train_logs,
            ["method", "epoch", "lr", "loss", "top1", "top5", "train_samples", "eval_samples"],
        )
        write_csv(
            output_dir / "gated_compare_web_bird.csv",
            method_summaries,
            ["method", "train_samples", "eval_samples", "best_epoch", "best_top1", "best_top5", "final_top1", "final_top5", "last10_mean", "last10_std"],
        )
        write_combined_compare(input_dir, output_dir, method_summaries)
    else:
        log_stage("[3/4] Training skipped. Re-run with --train to train the three gated splits.")

    log_stage("[4/4] Writing V1.6 summary.")
    write_summary(output_dir / "run_summary.md", input_dir, args.low_ratio, summary_rows, per_class_rows, method_summaries, train_enabled=args.train)
    write_json(
        output_dir / "v1_6_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "low_ratio": args.low_ratio,
            "train_enabled": args.train,
            "suspicious_ratio_summary": summary_rows,
            "gate_method_summaries": method_summaries,
        },
    )
    log_stage(f"V1.6 gated split analysis written to {output_dir}")


def validate_ratio(value: float) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError("--low-ratio must be in (0, 1).")


def load_v1_outputs(input_dir: Path) -> dict[str, Any]:
    required = [
        "full_gcdd_clean_split.csv",
        "centroid_filtering_split.csv",
        "gcdd_scores.csv",
        "centroid_scores.csv",
        "labels.npy",
        "paths.txt",
    ]
    for name in required:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required V1 output is missing: {path}")

    score_rows = sorted(read_csv(input_dir / "gcdd_scores.csv"), key=lambda row: int(row["index"]))
    centroid_rows = sorted(read_csv(input_dir / "centroid_scores.csv"), key=lambda row: int(row["index"]))
    labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    n = len(labels)
    if len(score_rows) != n or len(centroid_rows) != n or len(paths) != n:
        raise ValueError("V1 score, centroid, label, and path files have inconsistent lengths.")

    return {
        "index": np.array([int(row["index"]) for row in score_rows], dtype=np.int64),
        "path": np.array([row["path"] for row in score_rows], dtype=object),
        "labels": labels,
        "paths": paths,
        "S_clean": parse_float_array(score_rows, "S_clean"),
        "Q_same": parse_float_array(score_rows, "Q_same"),
        "centroid_score": parse_float_array(centroid_rows, "centroid_score"),
        "gcdd_clean": read_clean_mask(input_dir / "full_gcdd_clean_split.csv", n),
        "centroid_clean": read_clean_mask(input_dir / "centroid_filtering_split.csv", n),
    }


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


def parse_float_array(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([float(row[field]) for row in rows], dtype=np.float32)


def read_clean_mask(path: Path, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for row in read_csv(path):
        idx = int(row["index"])
        if idx >= n:
            raise ValueError(f"Index {idx} in {path} exceeds expected length {n}.")
        mask[idx] = row["state"] == "clean"
    return mask


def compute_low_flags(labels: np.ndarray, q_same: np.ndarray, centroid_score: np.ndarray, low_ratio: float) -> dict[str, np.ndarray]:
    low_q = np.zeros(len(labels), dtype=bool)
    low_centroid = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        keep = int(np.ceil(len(idx) * low_ratio))
        if keep <= 0:
            continue
        q_order = np.argsort(q_same[idx], kind="mergesort")
        c_order = np.argsort(centroid_score[idx], kind="mergesort")
        low_q[idx[q_order[:keep]]] = True
        low_centroid[idx[c_order[:keep]]] = True
    return {
        "low_Q_same": low_q,
        "low_centroid": low_centroid,
        "low_both": low_q & low_centroid,
        "low_either": low_q | low_centroid,
    }


def assign_overlap_groups(gcdd_clean: np.ndarray, centroid_clean: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(gcdd_clean), dtype=bool),
        "full_gcdd_clean": gcdd_clean,
        "both": gcdd_clean & centroid_clean,
        "gcdd_only": gcdd_clean & ~centroid_clean,
        "centroid_only": ~gcdd_clean & centroid_clean,
        "neither": ~gcdd_clean & ~centroid_clean,
    }


def build_gated_masks(gcdd_clean: np.ndarray, flags: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "gcdd_qgate": gcdd_clean & ~flags["low_Q_same"],
        "gcdd_pgate": gcdd_clean & ~flags["low_centroid"],
        "gcdd_qp_gate": gcdd_clean & ~flags["low_both"],
    }


def write_suspicious_ratio_summary(path: Path, data: dict[str, Any], flags: dict[str, np.ndarray], groups: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for name in ["full_gcdd_clean", "gcdd_only", "centroid_only", "both", "neither", "all"]:
        mask = groups[name]
        count = int(mask.sum())
        row = {"group": name, "count": count}
        for flag_name in ["low_Q_same", "low_centroid", "low_both", "low_either"]:
            flag_count = int(np.sum(mask & flags[flag_name]))
            row[f"{flag_name}_count"] = flag_count
            row[f"{flag_name}_ratio"] = safe_ratio(flag_count, count)
        rows.append(row)
    fieldnames = [
        "group",
        "count",
        "low_Q_same_count",
        "low_Q_same_ratio",
        "low_centroid_count",
        "low_centroid_ratio",
        "low_both_count",
        "low_both_ratio",
        "low_either_count",
        "low_either_ratio",
    ]
    write_csv(path, rows, fieldnames)
    return rows


def write_per_class_gate_summary(path: Path, data: dict[str, Any], flags: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    labels = data["labels"]
    base_clean = data["gcdd_clean"]
    rows = []
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        base_count = int(np.sum(base_clean[idx]))
        row = {
            "class_id": label,
            "num_total": int(len(idx)),
            "gcdd_clean": base_count,
            "gcdd_clean_low_Q_same": int(np.sum(base_clean[idx] & flags["low_Q_same"][idx])),
            "gcdd_clean_low_centroid": int(np.sum(base_clean[idx] & flags["low_centroid"][idx])),
            "gcdd_clean_low_both": int(np.sum(base_clean[idx] & flags["low_both"][idx])),
            "Q_same_mean": float(data["Q_same"][idx].mean()),
            "centroid_score_mean": float(data["centroid_score"][idx].mean()),
            "S_clean_mean": float(data["S_clean"][idx].mean()),
        }
        for key in ["gcdd_qgate", "gcdd_pgate", "gcdd_qp_gate"]:
            clean_count = int(np.sum(masks[key][idx]))
            row[f"{key}_clean"] = clean_count
            row[f"{key}_removed"] = base_count - clean_count
            row[f"{key}_removed_ratio"] = safe_ratio(base_count - clean_count, base_count)
        rows.append(row)
    fieldnames = [
        "class_id",
        "num_total",
        "gcdd_clean",
        "gcdd_clean_low_Q_same",
        "gcdd_clean_low_centroid",
        "gcdd_clean_low_both",
        "gcdd_qgate_clean",
        "gcdd_qgate_removed",
        "gcdd_qgate_removed_ratio",
        "gcdd_pgate_clean",
        "gcdd_pgate_removed",
        "gcdd_pgate_removed_ratio",
        "gcdd_qp_gate_clean",
        "gcdd_qp_gate_removed",
        "gcdd_qp_gate_removed_ratio",
        "Q_same_mean",
        "centroid_score_mean",
        "S_clean_mean",
    ]
    write_csv(path, rows, fieldnames)
    return rows


def write_gate_detail(path: Path, data: dict[str, Any], flags: dict[str, np.ndarray], masks: dict[str, np.ndarray], groups: dict[str, np.ndarray]) -> None:
    rows = []
    for i in range(len(data["labels"])):
        group = "neither"
        if groups["both"][i]:
            group = "both"
        elif groups["gcdd_only"][i]:
            group = "gcdd_only"
        elif groups["centroid_only"][i]:
            group = "centroid_only"
        rows.append(
            {
                "index": int(data["index"][i]),
                "path": data["path"][i],
                "web_label": data["labels"][i],
                "group": group,
                "full_gcdd_state": "clean" if data["gcdd_clean"][i] else "ignored",
                "centroid_state": "clean" if data["centroid_clean"][i] else "ignored",
                "low_Q_same": int(flags["low_Q_same"][i]),
                "low_centroid": int(flags["low_centroid"][i]),
                "low_both": int(flags["low_both"][i]),
                "Q_same": float(data["Q_same"][i]),
                "centroid_score": float(data["centroid_score"][i]),
                "S_clean": float(data["S_clean"][i]),
                "gcdd_qgate_state": "clean" if masks["gcdd_qgate"][i] else "ignored",
                "gcdd_pgate_state": "clean" if masks["gcdd_pgate"][i] else "ignored",
                "gcdd_qp_gate_state": "clean" if masks["gcdd_qp_gate"][i] else "ignored",
            }
        )
    write_csv(
        path,
        rows,
        [
            "index",
            "path",
            "web_label",
            "group",
            "full_gcdd_state",
            "centroid_state",
            "low_Q_same",
            "low_centroid",
            "low_both",
            "Q_same",
            "centroid_score",
            "S_clean",
            "gcdd_qgate_state",
            "gcdd_pgate_state",
            "gcdd_qp_gate_state",
        ],
    )


def write_split(path: Path, data: dict[str, Any], clean_mask: np.ndarray) -> None:
    rows = [
        {
            "index": int(data["index"][i]),
            "path": data["path"][i],
            "web_label": data["labels"][i],
            "state": "clean" if clean_mask[i] else "ignored",
        }
        for i in range(len(clean_mask))
    ]
    write_csv(path, rows, ["index", "path", "web_label", "state"])


def train_gated_splits(
    input_dir: Path,
    output_dir: Path,
    cfg: dict[str, Any],
    data: dict[str, Any],
    masks: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_name = cfg.get("train", {}).get("feature", "cls")
    train_features = load_feature(input_dir, "", feature_name)
    eval_features = load_feature(input_dir, "eval_", feature_name)
    train_labels = data["labels"]
    eval_labels = np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str)
    train_logs: list[dict[str, Any]] = []
    method_summaries: list[dict[str, Any]] = []
    for key, mask in masks.items():
        method = GATE_METHODS[key]
        logs, _ = train_linear_eval(train_features, train_labels, eval_features, eval_labels, mask, cfg, method)
        train_logs.extend(logs)
        method_summaries.append(summarize_epoch_logs(method, logs))
    return train_logs, method_summaries


def load_feature(input_dir: Path, prefix: str, feature_name: str) -> np.ndarray:
    path = input_dir / f"{prefix}features_{feature_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Feature file is missing: {path}")
    return np.load(path)


def write_combined_compare(input_dir: Path, output_dir: Path, method_summaries: list[dict[str, Any]]) -> None:
    existing_path = input_dir / "baseline_compare_web_bird.csv"
    rows: list[dict[str, Any]] = []
    if existing_path.exists():
        for row in read_csv(existing_path):
            if row["method"] in ["DINOv2 Linear all", "Centroid filtering", "Full GCDD-clean"]:
                row = dict(row)
                row["source"] = "existing_v1"
                rows.append(row)
    for row in method_summaries:
        out = dict(row)
        out["source"] = "v1_6_gated"
        rows.append(out)
    write_csv(
        output_dir / "combined_compare_web_bird.csv",
        rows,
        ["source", "method", "train_samples", "eval_samples", "best_epoch", "best_top1", "best_top5", "final_top1", "final_top5", "last10_mean", "last10_std"],
    )


def write_summary(
    path: Path,
    input_dir: Path,
    low_ratio: float,
    summary_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    method_summaries: list[dict[str, Any]],
    train_enabled: bool,
) -> None:
    summary_map = {row["group"]: row for row in summary_rows}
    full = summary_map["full_gcdd_clean"]
    gcdd_only = summary_map["gcdd_only"]
    centroid_only = summary_map["centroid_only"]
    lines = [
        "# V1.6 Gated GCDD Split Summary",
        "",
        f"- Source V1 output: {input_dir}",
        f"- Per-class low-ratio: {low_ratio:.2f}",
        "- Gate policy: remove clean samples only; do not supplement removed samples.",
        "",
        "## Suspicious Ratios",
        "| group | count | low_Q_same | low_centroid | low_both |",
        "| --- | ---: | ---: | ---: | ---: |",
        ratio_line(full),
        ratio_line(gcdd_only),
        ratio_line(centroid_only),
        "",
        "## Split Sizes",
        "| split | total clean | removed from Full GCDD |",
        "| --- | ---: | ---: |",
    ]
    totals = summarize_gate_totals(per_class_rows)
    for key, label in [("gcdd_qgate", "gcdd_qgate"), ("gcdd_pgate", "gcdd_pgate"), ("gcdd_qp_gate", "gcdd_qp_gate")]:
        lines.append(f"| {label} | {totals[key + '_clean']} | {totals[key + '_removed']} |")
    if train_enabled:
        lines.extend(["", "## Training Results", "| method | best_top1 | final_top1 | last10_mean | train_samples |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in method_summaries:
            lines.append(
                f"| {row['method']} | {float(row['best_top1']):.4f} | {float(row['final_top1']):.4f} | "
                f"{float(row['last10_mean']):.4f} | {int(row['train_samples'])} |"
            )
    else:
        lines.extend(["", "## Training Results", "Training was skipped. Run again with `--train` to train the three gated splits."])
    lines.extend(
        [
            "",
            "## Output Files",
            "- `suspicious_ratio_summary.csv`",
            "- `per_class_gate_summary.csv`",
            "- `gated_sample_flags.csv`",
            "- `gcdd_qgate_split.csv`",
            "- `gcdd_pgate_split.csv`",
            "- `gcdd_qp_gate_split.csv`",
            "- `gated_compare_web_bird.csv` if `--train` is used",
            "- `combined_compare_web_bird.csv` if `--train` is used",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def ratio_line(row: dict[str, Any]) -> str:
    return (
        f"| {row['group']} | {row['count']} | "
        f"{float(row['low_Q_same_ratio']):.4f} | "
        f"{float(row['low_centroid_ratio']):.4f} | "
        f"{float(row['low_both_ratio']):.4f} |"
    )


def summarize_gate_totals(per_class_rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for key in ["gcdd_qgate", "gcdd_pgate", "gcdd_qp_gate"]:
        totals[f"{key}_clean"] = int(sum(int(row[f"{key}_clean"]) for row in per_class_rows))
        totals[f"{key}_removed"] = int(sum(int(row[f"{key}_removed"]) for row in per_class_rows))
    return totals


def safe_ratio(num: int | float, denom: int | float) -> float:
    return float(num / denom) if denom else 0.0


if __name__ == "__main__":
    main()
