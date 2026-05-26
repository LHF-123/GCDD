from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json
from gcdd.progress import log_stage


GROUP_ORDER = ["both", "gcdd_proto_only", "centroid_only", "neither"]
METRIC_FIELDS = [
    "S_gcdd_proto",
    "S_clean",
    "centroid_score",
    "D_class",
    "R_class",
    "I_class_norm",
    "Q_same",
    "loss",
    "confidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GCDD+Proto clean split against centroid filtering.")
    parser.add_argument("--input-dir", default="outputs/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--proto-dir", help="Prototype-aware output directory. Defaults to <input-dir>/proto_gcdd.")
    parser.add_argument("--output-dir", help="Analysis output directory. Defaults to <input-dir>/proto_gcdd_vs_centroid_analysis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    proto_dir = Path(args.proto_dir) if args.proto_dir else input_dir / "proto_gcdd"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "proto_gcdd_vs_centroid_analysis"
    ensure_dir(output_dir)

    log_stage("[1/4] Loading centroid and GCDD+Proto split data.")
    data = load_data(input_dir, proto_dir)
    groups = assign_groups(data["gcdd_proto_clean"], data["centroid_clean"])
    data["group"] = groups

    log_stage("[2/4] Writing overlap and per-class tables.")
    overlap_row = write_overlap(output_dir / "proto_gcdd_vs_centroid_overlap.csv", data, groups)
    per_class_rows = write_per_class_overlap(output_dir, data, groups)

    log_stage("[3/4] Writing group metric summaries.")
    group_rows = write_group_metric_summary(output_dir / "group_metric_summary.csv", data, groups)
    write_sample_groups(output_dir / "sample_groups.csv", data, groups)

    log_stage("[4/4] Writing analysis summary.")
    write_summary(output_dir / "analysis_summary.md", input_dir, proto_dir, overlap_row, group_rows, per_class_rows)
    write_json(
        output_dir / "analysis_metadata.json",
        {
            "input_dir": str(input_dir),
            "proto_dir": str(proto_dir),
            "output_dir": str(output_dir),
            "metrics": METRIC_FIELDS,
        },
    )
    log_stage(f"GCDD+Proto vs centroid analysis written to {output_dir}")


def load_data(input_dir: Path, proto_dir: Path) -> dict[str, Any]:
    required = [
        input_dir / "centroid_filtering_split.csv",
        input_dir / "linear_all_train_scores.csv",
        proto_dir / "proto_gcdd_scores.csv",
        proto_dir / "splits" / "split_gcdd_proto.csv",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file is missing: {path}")

    score_rows = sorted(read_csv(proto_dir / "proto_gcdd_scores.csv"), key=lambda row: int(row["index"]))
    n = len(score_rows)
    if n == 0:
        raise ValueError("proto_gcdd_scores.csv is empty.")

    train_score_rows = sorted(read_csv(input_dir / "linear_all_train_scores.csv"), key=lambda row: int(row["index"]))
    if len(train_score_rows) != n:
        raise ValueError("linear_all_train_scores.csv length does not match proto_gcdd_scores.csv.")

    return {
        "index": np.array([int(row["index"]) for row in score_rows], dtype=np.int64),
        "path": np.array([row["path"] for row in score_rows], dtype=object),
        "web_label": np.array([row["web_label"] for row in score_rows], dtype=object),
        "S_gcdd_proto": float_array(score_rows, "S_gcdd_proto"),
        # Keep the user-facing name S_clean for the original GCDD score.
        "S_clean": float_array(score_rows, "S_gcdd"),
        "centroid_score": float_array(score_rows, "centroid_score"),
        "D_class": float_array(score_rows, "D_class"),
        "R_class": float_array(score_rows, "R_class"),
        "I_class_norm": float_array(score_rows, "I_class_norm"),
        "Q_same": float_array(score_rows, "Q_same"),
        "loss": float_array(train_score_rows, "loss"),
        "confidence": float_array(train_score_rows, "confidence"),
        "centroid_clean": read_clean_mask(input_dir / "centroid_filtering_split.csv", n),
        "gcdd_proto_clean": read_clean_mask(proto_dir / "splits" / "split_gcdd_proto.csv", n),
    }


def float_array(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([float(row[field]) for row in rows], dtype=np.float32)


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


def assign_groups(gcdd_proto_clean: np.ndarray, centroid_clean: np.ndarray) -> np.ndarray:
    """Four-way partition used to judge whether GCDD+Proto is just centroid in disguise."""
    groups = np.array(["neither"] * len(gcdd_proto_clean), dtype=object)
    groups[gcdd_proto_clean & centroid_clean] = "both"
    groups[gcdd_proto_clean & ~centroid_clean] = "gcdd_proto_only"
    groups[~gcdd_proto_clean & centroid_clean] = "centroid_only"
    return groups


def write_overlap(path: Path, data: dict[str, Any], groups: np.ndarray) -> dict[str, Any]:
    num_centroid = int(data["centroid_clean"].sum())
    num_gcdd_proto = int(data["gcdd_proto_clean"].sum())
    num_overlap = int(np.sum(groups == "both"))
    num_gcdd_proto_only = int(np.sum(groups == "gcdd_proto_only"))
    num_centroid_only = int(np.sum(groups == "centroid_only"))
    num_neither = int(np.sum(groups == "neither"))
    union = num_overlap + num_gcdd_proto_only + num_centroid_only
    row = {
        "num_centroid_clean": num_centroid,
        "num_gcdd_proto_clean": num_gcdd_proto,
        "num_overlap": num_overlap,
        "num_gcdd_proto_only": num_gcdd_proto_only,
        "num_centroid_only": num_centroid_only,
        "num_neither": num_neither,
        "jaccard": safe_ratio(num_overlap, union),
    }
    write_csv(path, [row], list(row.keys()))
    return row


def write_per_class_overlap(output_dir: Path, data: dict[str, Any], groups: np.ndarray) -> list[dict[str, Any]]:
    labels = data["web_label"]
    rows = []
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        both = int(np.sum(groups[idx] == "both"))
        proto_only = int(np.sum(groups[idx] == "gcdd_proto_only"))
        centroid_only = int(np.sum(groups[idx] == "centroid_only"))
        neither = int(np.sum(groups[idx] == "neither"))
        proto_clean = both + proto_only
        centroid_clean = both + centroid_only
        union = both + proto_only + centroid_only
        row = {
            "class_id": label,
            "num_total": int(len(idx)),
            "num_centroid_clean": centroid_clean,
            "num_gcdd_proto_clean": proto_clean,
            "num_overlap": both,
            "num_gcdd_proto_only": proto_only,
            "num_centroid_only": centroid_only,
            "num_neither": neither,
            "jaccard": safe_ratio(both, union),
            "gcdd_proto_only_ratio": safe_ratio(proto_only, len(idx)),
            "centroid_only_ratio": safe_ratio(centroid_only, len(idx)),
            "mean_S_gcdd_proto": float(np.mean(data["S_gcdd_proto"][idx])),
            "mean_S_clean": float(np.mean(data["S_clean"][idx])),
            "mean_centroid_score": float(np.mean(data["centroid_score"][idx])),
        }
        rows.append(row)
    fields = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "per_class_proto_gcdd_vs_centroid_overlap.csv", rows, fields)
    write_csv(output_dir / "per_class_low_jaccard_top20.csv", sorted(rows, key=lambda row: row["jaccard"])[:20], fields)
    write_csv(output_dir / "per_class_gcdd_proto_only_top20.csv", sorted(rows, key=lambda row: row["gcdd_proto_only_ratio"], reverse=True)[:20], fields)
    write_csv(output_dir / "per_class_centroid_only_top20.csv", sorted(rows, key=lambda row: row["centroid_only_ratio"], reverse=True)[:20], fields)
    return rows


def write_group_metric_summary(path: Path, data: dict[str, Any], groups: np.ndarray) -> list[dict[str, Any]]:
    fields = ["group", "count"]
    for metric in METRIC_FIELDS:
        fields.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_std", f"{metric}_p25", f"{metric}_p75"])
    rows = []
    for group in GROUP_ORDER:
        idx = groups == group
        row: dict[str, Any] = {"group": group, "count": int(np.sum(idx))}
        for metric in METRIC_FIELDS:
            row.update(metric_stats(metric, data[metric][idx]))
        rows.append(row)
    write_csv(path, rows, fields)
    return rows


def metric_stats(name: str, values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {f"{name}_{key}": np.nan for key in ["mean", "median", "std", "p25", "p75"]}
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return {f"{name}_{key}": np.nan for key in ["mean", "median", "std", "p25", "p75"]}
    return {
        f"{name}_mean": float(np.mean(valid)),
        f"{name}_median": float(np.median(valid)),
        f"{name}_std": float(np.std(valid)),
        f"{name}_p25": float(np.percentile(valid, 25)),
        f"{name}_p75": float(np.percentile(valid, 75)),
    }


def write_sample_groups(path: Path, data: dict[str, Any], groups: np.ndarray) -> None:
    rows = []
    fields = ["index", "path", "web_label", "group", "centroid_clean", "gcdd_proto_clean", *METRIC_FIELDS]
    for i in range(len(groups)):
        row = {
            "index": int(data["index"][i]),
            "path": data["path"][i],
            "web_label": data["web_label"][i],
            "group": groups[i],
            "centroid_clean": int(data["centroid_clean"][i]),
            "gcdd_proto_clean": int(data["gcdd_proto_clean"][i]),
        }
        for metric in METRIC_FIELDS:
            row[metric] = float(data[metric][i])
        rows.append(row)
    write_csv(path, rows, fields)


def write_summary(
    path: Path,
    input_dir: Path,
    proto_dir: Path,
    overlap: dict[str, Any],
    group_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
) -> None:
    group_map = {row["group"]: row for row in group_rows}
    lowest_j = sorted(per_class_rows, key=lambda row: row["jaccard"])[:5]
    top_proto = sorted(per_class_rows, key=lambda row: row["gcdd_proto_only_ratio"], reverse=True)[:5]
    top_centroid = sorted(per_class_rows, key=lambda row: row["centroid_only_ratio"], reverse=True)[:5]
    jaccard = float(overlap["jaccard"])
    if jaccard >= 0.90:
        judgment = "Jaccard >= 0.90: GCDD + Proto is very close to centroid filtering; graph contribution is weak in this split."
    elif jaccard >= 0.60:
        judgment = "Jaccard in [0.60, 0.90): the two splits differ meaningfully while performance is close."
    else:
        judgment = "Jaccard < 0.60: the two splits differ strongly; sample-quality analysis is required."

    lines = [
        "# GCDD+Proto vs Centroid Analysis",
        "",
        "This analysis does not train or change any split. It compares the selected clean samples only.",
        "",
        f"- Source V1 output: {input_dir}",
        f"- Source proto output: {proto_dir}",
        "",
        "## Overlap",
        f"- Centroid clean: {overlap['num_centroid_clean']}",
        f"- GCDD+Proto clean: {overlap['num_gcdd_proto_clean']}",
        f"- Overlap: {overlap['num_overlap']}",
        f"- GCDD+Proto only: {overlap['num_gcdd_proto_only']}",
        f"- Centroid only: {overlap['num_centroid_only']}",
        f"- Neither: {overlap['num_neither']}",
        f"- Jaccard: {jaccard:.4f}",
        f"- Judgment: {judgment}",
        "",
        "## Group Metric Summary",
        "| group | count | S_gcdd_proto | S_clean | centroid | R | I | Q | loss | confidence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUP_ORDER:
        row = group_map[group]
        lines.append(
            f"| {group} | {row['count']} | {fmt(row['S_gcdd_proto_mean'])} | {fmt(row['S_clean_mean'])} | "
            f"{fmt(row['centroid_score_mean'])} | {fmt(row['R_class_mean'])} | {fmt(row['I_class_norm_mean'])} | "
            f"{fmt(row['Q_same_mean'])} | {fmt(row['loss_mean'])} | {fmt(row['confidence_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Direct Comparisons",
            comparison_line(group_map, "gcdd_proto_only", "centroid_only", "loss_mean", "loss"),
            comparison_line(group_map, "gcdd_proto_only", "centroid_only", "confidence_mean", "confidence"),
            comparison_line(group_map, "gcdd_proto_only", "centroid_only", "R_class_mean", "R_class"),
            comparison_line(group_map, "gcdd_proto_only", "centroid_only", "I_class_norm_mean", "I_class_norm"),
            comparison_line(group_map, "gcdd_proto_only", "centroid_only", "Q_same_mean", "Q_same"),
            comparison_line(group_map, "gcdd_proto_only", "centroid_only", "centroid_score_mean", "centroid_score"),
            "",
            "## Per-Class Differences",
            "- Lowest Jaccard classes: " + ", ".join(row["class_id"] for row in lowest_j),
            "- Highest GCDD+Proto-only ratio classes: " + ", ".join(row["class_id"] for row in top_proto),
            "- Highest centroid-only ratio classes: " + ", ".join(row["class_id"] for row in top_centroid),
            "",
            "## Output Files",
            "- `proto_gcdd_vs_centroid_overlap.csv`",
            "- `group_metric_summary.csv`",
            "- `per_class_proto_gcdd_vs_centroid_overlap.csv`",
            "- `sample_groups.csv`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def comparison_line(group_map: dict[str, dict[str, Any]], left: str, right: str, field: str, label: str) -> str:
    return f"- {label}: {left}={fmt(group_map[left][field])}, {right}={fmt(group_map[right][field])}"


def safe_ratio(num: int | float, denom: int | float) -> float:
    return float(num / denom) if denom else 0.0


def fmt(value: Any) -> str:
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
