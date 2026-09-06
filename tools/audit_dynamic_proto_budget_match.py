"""Verify and summarize Dynamic baseline budgets paired to PGDF-DynamicProto."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any


SEEDS = (1, 42, 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", required=True, metavar="NAME=PGDF_ROOT=BUDGET_ROOT",
        help="Repeat for CUB/Cars/Aircraft; roots contain per-method seed directories.",
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


def parse_dataset(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("--dataset must be NAME=PGDF_ROOT=BUDGET_ROOT")
    return parts[0], Path(parts[1]), Path(parts[2])


def result_by_seed(root: Path, method: str) -> dict[int, dict[str, str]]:
    rows = read_csv(root / "checkpoint_validation_results.csv")
    selected = {int(row["seed"]): row for row in rows if row["method_key"] == method}
    if sorted(selected) != list(SEEDS):
        raise ValueError(f"{root}: expected complete {method} seeds {SEEDS}, found {sorted(selected)}")
    return selected


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing summary output: {output}")
    cardinality_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for raw in args.dataset:
        dataset, pgdf_root, budget_root = parse_dataset(raw)
        pgdf_results = result_by_seed(pgdf_root, "pgdf_dynamic_proto")
        budget_results = result_by_seed(budget_root, "dynamic_budget_matched_dynamic_proto")
        for seed in SEEDS:
            pgdf_path = pgdf_root / "pgdf_dynamic_proto" / f"seed{seed}" / "selection_per_class.csv"
            budget_path = budget_root / "dynamic_budget_matched_dynamic_proto" / f"seed{seed}" / "selection_per_class.csv"
            pgdf = {(int(row["epoch"]), str(row["web_label"])): row for row in read_csv(pgdf_path)}
            budget = {(int(row["epoch"]), str(row["web_label"])): row for row in read_csv(budget_path)}
            if set(pgdf) != set(budget):
                raise RuntimeError(f"{dataset}/seed{seed}: budget class/update key set differs from PGDF-DynamicProto.")
            for key in sorted(pgdf):
                source = int(pgdf[key]["selected_count"])
                observed = int(budget[key]["selected_count"])
                cardinality_rows.append(
                    {
                        "dataset": dataset, "seed": seed, "epoch": key[0], "class": key[1],
                        "pgdf_dynamic_proto_count": source,
                        "budget_matched_dynamic_count": observed,
                        "match": "yes" if source == observed else "no",
                    }
                )
            if any(row["match"] != "yes" for row in cardinality_rows if row["dataset"] == dataset and row["seed"] == seed):
                raise RuntimeError(f"{dataset}/seed{seed}: cardinality audit failed.")
            pgdf_top1 = float(pgdf_results[seed]["validation_selected_test_top1"]) * 100.0
            budget_top1 = float(budget_results[seed]["validation_selected_test_top1"]) * 100.0
            paired_rows.append(
                {
                    "dataset": dataset, "seed": seed,
                    "pgdf_dynamic_proto_selected_checkpoint_epoch": int(pgdf_results[seed]["best_val_epoch"]),
                    "budget_matched_dynamic_selected_checkpoint_epoch": int(budget_results[seed]["best_val_epoch"]),
                    "pgdf_dynamic_proto_validation_top1_pct": float(pgdf_results[seed]["best_val_top1"]) * 100.0,
                    "budget_matched_dynamic_validation_top1_pct": float(budget_results[seed]["best_val_top1"]) * 100.0,
                    "pgdf_dynamic_proto_top1_pct": pgdf_top1,
                    "budget_matched_dynamic_top1_pct": budget_top1,
                    "paired_difference_pgdf_minus_budget_pp": pgdf_top1 - budget_top1,
                }
            )

    write_csv(
        output / "cardinality_audit.csv", cardinality_rows,
        ["dataset", "seed", "epoch", "class", "pgdf_dynamic_proto_count", "budget_matched_dynamic_count", "match"],
    )
    write_csv(
        output / "paired_results.csv", paired_rows,
        [
            "dataset", "seed", "pgdf_dynamic_proto_selected_checkpoint_epoch",
            "budget_matched_dynamic_selected_checkpoint_epoch", "pgdf_dynamic_proto_validation_top1_pct",
            "budget_matched_dynamic_validation_top1_pct", "pgdf_dynamic_proto_top1_pct",
            "budget_matched_dynamic_top1_pct", "paired_difference_pgdf_minus_budget_pp",
        ],
    )
    summary_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in paired_rows}):
        values = [float(row["paired_difference_pgdf_minus_budget_pp"]) for row in paired_rows if row["dataset"] == dataset]
        mean, std = mean_std(values)
        summary_rows.append(
            {
                "scope": "dataset", "dataset": dataset, "pairs": len(values),
                "mean_paired_difference_pp": mean, "sample_std_paired_difference_pp": std,
                "pgdf_higher_count": sum(value > 0.0 for value in values),
                "budget_higher_count": sum(value < 0.0 for value in values), "tie_count": sum(value == 0.0 for value in values),
            }
        )
    values = [float(row["paired_difference_pgdf_minus_budget_pp"]) for row in paired_rows]
    mean, std = mean_std(values)
    summary_rows.append(
        {
            "scope": "overall_descriptive", "dataset": "ALL_3_DATASETS", "pairs": len(values),
            "mean_paired_difference_pp": mean, "sample_std_paired_difference_pp": std,
            "pgdf_higher_count": sum(value > 0.0 for value in values),
            "budget_higher_count": sum(value < 0.0 for value in values), "tie_count": sum(value == 0.0 for value in values),
        }
    )
    fields = ["scope", "dataset", "pairs", "mean_paired_difference_pp", "sample_std_paired_difference_pp", "pgdf_higher_count", "budget_higher_count", "tie_count"]
    write_csv(output / "summary.csv", summary_rows, fields)
    lines = ["PGDF-DynamicProto vs exact class/update Budget-Matched Dynamic", ""]
    for row in summary_rows:
        lines.append(
            f"{row['dataset']}: {row['mean_paired_difference_pp']:.4f} +/- "
            f"{row['sample_std_paired_difference_pp']:.4f} pp; PGDF higher={row['pgdf_higher_count']}, "
            f"Budget higher={row['budget_higher_count']}, ties={row['tie_count']}."
        )
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Budget cardinality audit PASS: {len(cardinality_rows)} class/update rows")


if __name__ == "__main__":
    main()
