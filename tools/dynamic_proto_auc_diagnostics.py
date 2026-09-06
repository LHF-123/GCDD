"""Post-hoc AUC diagnostics from exact PGDF-DynamicProto selection snapshots.

This utility never instantiates a model or changes a training artifact.  It
reads the per-update ``selection_rows.csv`` written during the formal run,
whose ``proto_score`` column is the current LoRA-adapted prototype score at
that exact update.  Clean/noisy ground truth is joined only after the fact for
ROC-AUC reporting.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.selection_utils import path_key_candidates


SEEDS = (1, 42, 88)
UPDATE_EPOCHS = (5, 10, 15, 20, 25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=RUN_ROOT=NOISE_INDEX",
        help="Repeat once per dataset; RUN_ROOT contains pgdf_dynamic_proto/seed*/selection_rows.csv.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def parse_dataset(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("--dataset must be NAME=RUN_ROOT=NOISE_INDEX")
    return parts[0], Path(parts[1]), Path(parts[2])


def clean_by_path(noise_index: Path) -> dict[str, str]:
    rows = [row for row in read_csv(noise_index) if row.get("split", "train").lower() == "train"]
    result: dict[str, str] = {}
    for row in rows:
        for raw in (row.get("path", ""), row.get("abs_path", "")):
            for key in path_key_candidates(raw):
                result[key] = str(row["clean_label"])
    if not result:
        raise ValueError(f"No training clean labels found in {noise_index}")
    return result


def lookup_clean_label(path: str, values: dict[str, str]) -> str:
    for key in path_key_candidates(path):
        if key in values:
            return values[key]
    raise KeyError(f"Selection path is absent from noise index: {path}")


def roc_auc(scores: list[float], positives: list[bool]) -> float:
    """Tie-aware Mann-Whitney ROC-AUC without a sklearn dependency."""
    if len(scores) != len(positives) or not scores:
        return math.nan
    pos = sum(positives)
    neg = len(positives) - pos
    if pos == 0 or neg == 0:
        return math.nan
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for source_index, _ in ranked[start:end]:
            if positives[source_index]:
                rank_sum += average_rank
        start = end
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def snapshot_stats(rows: list[dict[str, str]], clean_labels: dict[str, str]) -> dict[str, Any]:
    scores: list[float] = []
    positives: list[bool] = []
    by_class: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        score = float(row["proto_score"])
        observed = str(row["web_label"])
        clean = lookup_clean_label(str(row["path"]), clean_labels)
        is_clean = observed == clean
        scores.append(score)
        positives.append(is_clean)
        by_class[observed].append((score, is_clean))

    class_aucs: list[tuple[float, int]] = []
    excluded = 0
    for values in by_class.values():
        auc = roc_auc([value[0] for value in values], [value[1] for value in values])
        if math.isnan(auc):
            excluded += 1
        else:
            class_aucs.append((auc, len(values)))
    auc_values = [item[0] for item in class_aucs]
    weighted = (
        sum(auc * count for auc, count in class_aucs) / sum(count for _, count in class_aucs)
        if class_aucs else math.nan
    )
    return {
        "pooled_auc": roc_auc(scores, positives),
        "macro_class_auc": statistics.mean(auc_values) if auc_values else math.nan,
        "weighted_class_auc": weighted,
        "valid_class_count": len(class_aucs),
        "excluded_class_count": excluded,
        "auc_lt_0p5_count": sum(value < 0.5 for value in auc_values),
        "auc_eq_0p5_count": sum(value == 0.5 for value in auc_values),
        "auc_gt_0p5_count": sum(value > 0.5 for value in auc_values),
        "median_class_auc": statistics.median(auc_values) if auc_values else math.nan,
        "training_pool_size": len(rows),
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnostic output: {output}")
    per_seed: list[dict[str, Any]] = []
    for raw in args.dataset:
        dataset, run_root, noise_index = parse_dataset(raw)
        run_index = read_csv(run_root / "run_index.csv")
        completed = {
            int(row["seed"]): row.get("status", "")
            for row in run_index if row.get("method_key") == "pgdf_dynamic_proto"
        }
        if completed != {seed: "complete" for seed in SEEDS}:
            raise ValueError(
                f"{run_root}: exact AUC diagnostic requires complete PGDF-DynamicProto seeds "
                f"{SEEDS}; found {completed}."
            )
        labels = clean_by_path(noise_index)
        for seed in SEEDS:
            selection_path = run_root / "pgdf_dynamic_proto" / f"seed{seed}" / "selection_rows.csv"
            if not selection_path.is_file():
                raise FileNotFoundError(f"Missing exact dynamic selection snapshot: {selection_path}")
            rows = read_csv(selection_path)
            grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                if row.get("prototype_mode") != "dynamic_lora":
                    raise ValueError(f"{selection_path} is not a dynamic-LoRA prototype artifact.")
                grouped[int(row["epoch"])].append(row)
            if sorted(grouped) != list(UPDATE_EPOCHS):
                raise ValueError(f"{selection_path} has update epochs {sorted(grouped)}, expected {list(UPDATE_EPOCHS)}")
            for epoch in UPDATE_EPOCHS:
                stats = snapshot_stats(grouped[epoch], labels)
                per_seed.append({"dataset": dataset, "seed": seed, "epoch": epoch, **stats})

    fields = [
        "dataset", "seed", "epoch", "pooled_auc", "macro_class_auc", "weighted_class_auc",
        "valid_class_count", "excluded_class_count", "auc_lt_0p5_count", "auc_eq_0p5_count",
        "auc_gt_0p5_count", "median_class_auc", "training_pool_size",
    ]
    write_csv(output / "dynamic_proto_auc_by_seed_epoch.csv", per_seed, fields)
    summaries: list[dict[str, Any]] = []
    metric_fields = [
        "pooled_auc", "macro_class_auc", "weighted_class_auc", "valid_class_count", "excluded_class_count",
        "auc_lt_0p5_count", "auc_eq_0p5_count", "auc_gt_0p5_count", "median_class_auc",
    ]
    for dataset in sorted({str(row["dataset"]) for row in per_seed}):
        for epoch in UPDATE_EPOCHS:
            rows = [row for row in per_seed if row["dataset"] == dataset and int(row["epoch"]) == epoch]
            item: dict[str, Any] = {"dataset": dataset, "epoch": epoch, "seeds": "1,42,88"}
            for metric in metric_fields:
                mean, std = mean_std([float(row[metric]) for row in rows])
                item[f"mean_{metric}"] = mean
                item[f"sample_std_{metric}"] = std
            summaries.append(item)
    summary_fields = ["dataset", "epoch", "seeds"] + [
        field for metric in metric_fields for field in (f"mean_{metric}", f"sample_std_{metric}")
    ]
    write_csv(output / "dynamic_proto_auc_summary.csv", summaries, summary_fields)
    lines = ["Dynamic Prototype AUC diagnostics", "", "Scores: exact saved dynamic-LoRA prototype scores at selection updates.", ""]
    for row in summaries:
        lines.append(
            f"{row['dataset']} epoch {row['epoch']}: pooled={row['mean_pooled_auc']:.6f} +/- "
            f"{row['sample_std_pooled_auc']:.6f}; macro={row['mean_macro_class_auc']:.6f} +/- "
            f"{row['sample_std_macro_class_auc']:.6f}; weighted={row['mean_weighted_class_auc']:.6f} +/- "
            f"{row['sample_std_weighted_class_auc']:.6f}."
        )
    (output / "dynamic_proto_auc_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"AUC diagnostic complete: {output}")


if __name__ == "__main__":
    main()
