"""Aggregate validation-selected Dynamic Prototype runs across three datasets."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Dynamic Prototype validation-selected test results across datasets."
    )
    parser.add_argument(
        "--method-key",
        required=True,
        choices=("dynamic_proto_only", "fixed_proto_warmup_matched", "pgdf_dynamic_proto"),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
        help="One checkpoint-validation output root. Pass once per dataset.",
    )
    parser.add_argument("--seeds", default="1,42,88", help="Expected comma-separated training seeds.")
    parser.add_argument("--output-dir", required=True, help="Directory for summary.csv, summary.txt, and summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = parse_datasets(args.dataset)
    seeds = parse_seeds(args.seeds)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty summary directory: {output_dir}")
    ensure_dir(output_dir)

    run_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for name, run_dir in datasets:
        rows = load_method_rows(name, run_dir, args.method_key, seeds)
        run_rows.extend(rows)
        validation = np.asarray([float(row["validation_top1_pct"]) for row in rows], dtype=np.float64)
        official_test = np.asarray([float(row["official_test_top1_pct"]) for row in rows], dtype=np.float64)
        dataset_rows.append(
            {
                "row_type": "dataset_summary",
                "dataset": name,
                "training_seed": "1,42,88",
                "selected_checkpoint_epoch": "",
                "validation_top1_pct": "",
                "official_test_top1_pct": "",
                "mean_validation_top1_pct": float(validation.mean()),
                "sample_std_validation_top1_pct": float(validation.std(ddof=1)),
                "mean_official_test_top1_pct": float(official_test.mean()),
                "sample_std_official_test_top1_pct": float(official_test.std(ddof=1)),
                "result_path": str(run_dir / "checkpoint_validation_results.csv"),
            }
        )

    avg_dataset_mean = float(
        np.asarray([row["mean_official_test_top1_pct"] for row in dataset_rows], dtype=np.float64).mean()
    )
    overall_row = {
        "row_type": "overall_summary",
        "dataset": "Avg.",
        "training_seed": "1,42,88",
        "selected_checkpoint_epoch": "",
        "validation_top1_pct": "",
        "official_test_top1_pct": "",
        "mean_validation_top1_pct": "",
        "sample_std_validation_top1_pct": "",
        "mean_official_test_top1_pct": avg_dataset_mean,
        "sample_std_official_test_top1_pct": "",
        "result_path": "",
    }
    csv_rows = [*run_rows, *dataset_rows, overall_row]
    fields = [
        "row_type",
        "dataset",
        "training_seed",
        "selected_checkpoint_epoch",
        "validation_top1_pct",
        "official_test_top1_pct",
        "mean_validation_top1_pct",
        "sample_std_validation_top1_pct",
        "mean_official_test_top1_pct",
        "sample_std_official_test_top1_pct",
        "result_path",
    ]
    write_csv(output_dir / "summary.csv", csv_rows, fields)
    write_json(
        output_dir / "summary.json",
        {
            "method_key": args.method_key,
            "seeds": seeds,
            "runs": run_rows,
            "datasets": dataset_rows,
            "avg_dataset_mean_official_test_top1_pct": avg_dataset_mean,
        },
    )
    write_text_summary(output_dir / "summary.txt", args.method_key, run_rows, dataset_rows, avg_dataset_mean)
    print(f"Dynamic Prototype summary written to {output_dir}", flush=True)


def parse_datasets(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    names: set[str] = set()
    for item in items:
        if "=" not in item:
            raise ValueError(f"--dataset must use NAME=RUN_DIR, got {item!r}.")
        name, raw_path = item.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"--dataset must include both NAME and RUN_DIR, got {item!r}.")
        if name in names:
            raise ValueError(f"Duplicate dataset name: {name}")
        names.add(name)
        parsed.append((name, Path(raw_path)))
    return parsed


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer values.")
    return seeds


def load_method_rows(dataset: str, run_dir: Path, method_key: str, seeds: list[int]) -> list[dict[str, Any]]:
    path = run_dir / "checkpoint_validation_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing result table for {dataset}: {path}")
    source_rows = [row for row in read_csv(path) if row.get("method_key") == method_key]
    by_seed = {int(row["seed"]): row for row in source_rows}
    if set(by_seed) != set(seeds):
        raise ValueError(
            f"{dataset} has {method_key} seeds {sorted(by_seed)}, expected exactly {seeds}: {path}"
        )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        row = by_seed[seed]
        if row.get("official_test_evaluation") != "validation_selected_only":
            raise ValueError(f"{dataset} seed {seed} did not use validation-selected-only test evaluation.")
        if row.get("validation_selected_test_top1", "") == "":
            raise ValueError(f"{dataset} seed {seed} is missing validation-selected official-test Top-1.")
        rows.append(
            {
                "row_type": "run",
                "dataset": dataset,
                "training_seed": int(seed),
                "selected_checkpoint_epoch": int(row["best_val_epoch"]),
                "validation_top1_pct": 100.0 * float(row["best_val_top1"]),
                "official_test_top1_pct": 100.0 * float(row["validation_selected_test_top1"]),
                "mean_validation_top1_pct": "",
                "sample_std_validation_top1_pct": "",
                "mean_official_test_top1_pct": "",
                "sample_std_official_test_top1_pct": "",
                "result_path": str(path),
            }
        )
    return rows


def write_text_summary(
    path: Path,
    method_key: str,
    run_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    avg_dataset_mean: float,
) -> None:
    lines = [
        f"Dynamic Prototype validation-selected summary: {method_key}",
        "Top-1 values below are percentages. Standard deviations are sample std (ddof=1).",
        "",
        "Dataset\tSeed\tSelected checkpoint epoch\tValidation Top-1\tOfficial-test Top-1",
    ]
    for row in run_rows:
        lines.append(
            f"{row['dataset']}\t{row['training_seed']}\t{row['selected_checkpoint_epoch']}\t"
            f"{float(row['validation_top1_pct']):.4f}\t{float(row['official_test_top1_pct']):.4f}"
        )
    lines.extend(["", "Dataset\tValidation Top-1 (mean +/- sample std)\tOfficial-test Top-1 (mean +/- sample std)"])
    for row in dataset_rows:
        lines.append(
            f"{row['dataset']}\t{float(row['mean_validation_top1_pct']):.4f} +/- "
            f"{float(row['sample_std_validation_top1_pct']):.4f}\t"
            f"{float(row['mean_official_test_top1_pct']):.4f} +/- "
            f"{float(row['sample_std_official_test_top1_pct']):.4f}"
        )
    lines.append(f"Avg.\t\t{avg_dataset_mean:.4f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
