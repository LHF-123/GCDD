"""Summarize PGDF-DynamicProto noise-rate or one-dimensional sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

SEEDS = (1, 42, 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("noise_rates", "p_sensitivity", "r_sensitivity"), required=True)
    parser.add_argument("--entry", action="append", required=True, metavar="VALUE|DATASET|RESULT_ROOT")
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


def parse_entry(value: str) -> tuple[float, str, Path]:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("--entry must be VALUE|DATASET|RESULT_ROOT")
    return float(parts[0]), parts[1], Path(parts[2])


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    args = parse_args()
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing summary output: {output}")
    run_rows: list[dict[str, Any]] = []
    for raw in args.entry:
        value, dataset, root = parse_entry(raw)
        rows = [row for row in read_csv(root / "checkpoint_validation_results.csv") if row["method_key"] == "pgdf_dynamic_proto"]
        by_seed = {int(row["seed"]): row for row in rows}
        if sorted(by_seed) != list(SEEDS):
            raise ValueError(f"{root}: incomplete PGDF-DynamicProto seed set {sorted(by_seed)}")
        for seed in SEEDS:
            row = by_seed[seed]
            run_rows.append(
                {
                    "setting": value, "dataset": dataset, "seed": seed,
                    "selected_checkpoint_epoch": int(row["best_val_epoch"]),
                    "validation_top1_pct": float(row["best_val_top1"]) * 100.0,
                    "official_test_top1_pct": float(row["validation_selected_test_top1"]) * 100.0,
                    "scheduler_estimate": float(row["scheduler_retention_ratio"]),
                    "r": float(row["retention_ratio"]), "p": float(row["proto_keep_ratio"]),
                    "result_root": str(root),
                }
            )

    if args.mode == "noise_rates":
        run_name, summary_name = "noise_rate_runs.csv", "summary.csv"
    elif args.mode == "p_sensitivity":
        run_name, summary_name = "p_sensitivity_runs.csv", "p_sensitivity_summary.csv"
    else:
        run_name, summary_name = "r_sensitivity_runs.csv", "r_sensitivity_summary.csv"
    fields = ["setting", "dataset", "seed", "selected_checkpoint_epoch", "validation_top1_pct", "official_test_top1_pct", "scheduler_estimate", "r", "p", "result_root"]
    write_csv(output / run_name, run_rows, fields)

    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(float(row["setting"]), str(row["dataset"]))].append(row)
    summary_long: list[dict[str, Any]] = []
    for (setting, dataset), rows in sorted(grouped.items()):
        tests = [float(row["official_test_top1_pct"]) for row in rows]
        vals = [float(row["validation_top1_pct"]) for row in rows]
        test_mean, test_std = mean_std(tests)
        val_mean, val_std = mean_std(vals)
        summary_long.append(
            {
                "setting": setting, "dataset": dataset, "r": rows[0]["r"], "p": rows[0]["p"],
                "scheduler_estimate": rows[0]["scheduler_estimate"],
                "validation_mean_pct": val_mean, "validation_sample_std_pct": val_std,
                "official_test_mean_pct": test_mean, "official_test_sample_std_pct": test_std,
                "pre_specified_main_configuration": "yes" if rows[0]["r"] == 0.8 and rows[0]["p"] == 0.4 else "no",
            }
        )
    if args.mode == "noise_rates":
        write_csv(output / summary_name, summary_long, list(summary_long[0]) if summary_long else [])
        lines = ["PGDF-DynamicProto FGVC-Aircraft noise-rate summary", ""]
        for row in summary_long:
            lines.append(f"asym{int(row['setting'] * 100)}: {row['official_test_mean_pct']:.4f} +/- {row['official_test_sample_std_pct']:.4f}")
        (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    datasets = ("CUB-200-2011", "Stanford Cars", "FGVC-Aircraft")
    wide_rows: list[dict[str, Any]] = []
    for setting in sorted({float(row["setting"]) for row in summary_long}):
        subset = {str(row["dataset"]): row for row in summary_long if float(row["setting"]) == setting}
        if set(subset) != set(datasets):
            raise ValueError(f"setting {setting} is missing a dataset summary")
        row: dict[str, Any] = {
            "setting": setting, "r": subset[datasets[0]]["r"], "p": subset[datasets[0]]["p"],
            "scheduler_estimate": subset[datasets[0]]["scheduler_estimate"],
            "pre_specified_main_configuration": subset[datasets[0]]["pre_specified_main_configuration"],
        }
        means = []
        for dataset in datasets:
            prefix = {"CUB-200-2011": "cub", "Stanford Cars": "cars", "FGVC-Aircraft": "aircraft"}[dataset]
            row[f"{prefix}_mean_pct"] = subset[dataset]["official_test_mean_pct"]
            row[f"{prefix}_sample_std_pct"] = subset[dataset]["official_test_sample_std_pct"]
            means.append(float(subset[dataset]["official_test_mean_pct"]))
        row["avg_dataset_mean_pct"] = statistics.mean(means)
        wide_rows.append(row)
    write_csv(output / summary_name, wide_rows, list(wide_rows[0]) if wide_rows else [])
    lines = [f"PGDF-DynamicProto {args.mode} (descriptive; not parameter selection)", ""]
    for row in wide_rows:
        lines.append(
            f"setting={row['setting']:.1f}, r={row['r']:.1f}, p={row['p']:.1f}, scheduler={row['scheduler_estimate']:.1f}, "
            f"Avg={row['avg_dataset_mean_pct']:.4f}; pre-specified-main={row['pre_specified_main_configuration']}"
        )
    (output / "parameter_sensitivity_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
