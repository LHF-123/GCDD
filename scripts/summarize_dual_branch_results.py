"""Reconstruct Co-teaching/JoCoR Table-1 results from saved dual-branch runs.

This utility is intentionally read-only with respect to training assets.  For
each dataset, method, and seed it selects the earliest epoch maximizing the
mean validation Top-1 (as recorded by the run), then reports the official-test
Top-1 of the validation-better branch.  A validation tie is deterministically
resolved in favour of branch A.  The script writes a seed-level CSV and checks
the six Table-1 mean/sample-SD values after two-decimal display rounding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SEEDS = (1, 42, 88)
METHODS = ("coteaching", "jocor")
METHOD_DISPLAY = {"coteaching": "Co-teaching", "jocor": "JoCoR"}
PAPER_TABLE_1 = {
    ("coteaching", "CUB-200-2011"): (62.97, 1.03),
    ("coteaching", "Stanford Cars"): (61.11, 0.34),
    ("coteaching", "FGVC-Aircraft"): (60.08, 0.12),
    ("jocor", "CUB-200-2011"): (62.25, 1.18),
    ("jocor", "Stanford Cars"): (60.81, 0.70),
    ("jocor", "FGVC-Aircraft"): (60.27, 0.78),
}
CSV_FIELDS = [
    "Method",
    "Dataset",
    "Seed",
    "Best Epoch by Mean Val",
    "Mean-Val Max Epochs",
    "Val A",
    "Val B",
    "Selected Branch",
    "Test A",
    "Test B",
    "Selected Test",
    "Mean Test",
    "Branch Tie",
    "Result JSON",
    "Train Log",
    "Best Checkpoint",
]
TOLERANCE = 1e-12


@dataclass(frozen=True)
class SeedRecord:
    method: str
    dataset: str
    seed: int
    best_epoch: int
    max_epochs: tuple[int, ...]
    val_a: float
    val_b: float
    test_a: float
    test_b: float
    result_json: Path
    train_log: Path
    checkpoint: Path

    @property
    def branch_tie(self) -> bool:
        return math.isclose(self.val_a, self.val_b, rel_tol=0.0, abs_tol=TOLERANCE)

    @property
    def selected_branch(self) -> str:
        return "A" if self.val_a >= self.val_b else "B"

    @property
    def selected_test(self) -> float:
        return self.test_a if self.selected_branch == "A" else self.test_b

    @property
    def mean_test(self) -> float:
        return (self.test_a + self.test_b) / 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="Repeat once per dataset, e.g. CUB-200-2011=outputs/.../checkpoint_validation_missing9_s20250726.",
    )
    parser.add_argument("--seed-output", required=True, type=Path, help="Destination for the reconstructed seed-level CSV.")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated method keys; defaults to coteaching,jocor.")
    parser.add_argument("--seeds", default="1,42,88", help="Comma-separated training seeds; defaults to 1,42,88.")
    return parser.parse_args()


def parse_dataset_roots(items: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-root {item!r}; expected DATASET=PATH.")
        dataset, raw_path = item.split("=", 1)
        if not dataset or not raw_path:
            raise ValueError(f"Invalid --dataset-root {item!r}; both dataset and path are required.")
        if dataset in roots:
            raise ValueError(f"Duplicate dataset root for {dataset!r}.")
        path = Path(raw_path)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {path}")
        roots[dataset] = path
    expected = {dataset for _, dataset in PAPER_TABLE_1}
    if set(roots) != expected:
        raise ValueError(f"Dataset roots must be exactly {sorted(expected)}, got {sorted(roots)}.")
    return roots


def parse_csv_ints(raw: str, label: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must contain one or more unique integers.")
    return values


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def require_float(payload: dict[str, Any], key: str, path: Path) -> float:
    if key not in payload or payload[key] in (None, ""):
        raise ValueError(f"Missing required numeric field {key!r} in {path}.")
    return float(payload[key])


def require_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = require_float(payload, key, path)
    if not value.is_integer():
        raise ValueError(f"Expected integer field {key!r} in {path}, got {value}.")
    return int(value)


def read_log_maxima(train_log: Path) -> tuple[tuple[int, ...], dict[str, str]]:
    with train_log.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training log is empty: {train_log}")
    for key in ("epoch", "mean_ab_top1", "top1_a", "top1_b"):
        if key not in rows[0]:
            raise ValueError(f"Training log {train_log} is missing field {key!r}.")
    values = [float(row["mean_ab_top1"]) for row in rows]
    maximum = max(values)
    maxima = tuple(int(row["epoch"]) for row in rows if math.isclose(float(row["mean_ab_top1"]), maximum, rel_tol=0.0, abs_tol=TOLERANCE))
    if not maxima:
        raise RuntimeError(f"No maximum validation row found in {train_log}.")
    first_row = next(row for row in rows if int(row["epoch"]) == maxima[0])
    return maxima, first_row


def reconstruct_seed(dataset: str, root: Path, method: str, seed: int) -> SeedRecord:
    seed_dir = root / method / f"seed{seed}"
    result_json = seed_dir / "result.json"
    train_log = seed_dir / "train_log.csv"
    checkpoint = seed_dir / "checkpoints" / "best_val.pt"
    for path in (result_json, train_log, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact is absent: {path}")

    result = load_json(result_json)
    if str(result.get("method_key", "")) != method:
        raise ValueError(f"Method key mismatch in {result_json}: {result.get('method_key')!r} != {method!r}.")
    if require_int(result, "seed", result_json) != seed:
        raise ValueError(f"Seed mismatch in {result_json}.")
    if str(result.get("selection_mode", "")) != "two_branch_mean_validation_top1":
        raise ValueError(f"Unexpected selection_mode in {result_json}: {result.get('selection_mode')!r}.")

    max_epochs, log_row = read_log_maxima(train_log)
    best_epoch = require_int(result, "best_val_epoch", result_json)
    if best_epoch != max_epochs[0]:
        raise ValueError(f"best_val_epoch={best_epoch} is not the first maximum epoch {max_epochs} in {train_log}.")
    val_a = require_float(result, "best_model_a_top1", result_json)
    val_b = require_float(result, "best_model_b_top1", result_json)
    mean_val = require_float(result, "best_mean_ab_top1", result_json)
    if not math.isclose(mean_val, (val_a + val_b) / 2.0, rel_tol=0.0, abs_tol=TOLERANCE):
        raise ValueError(f"best_mean_ab_top1 is not the branch arithmetic mean in {result_json}.")
    for label, saved, logged in (("A", val_a, float(log_row["top1_a"])), ("B", val_b, float(log_row["top1_b"]))):
        if not math.isclose(saved, logged, rel_tol=0.0, abs_tol=TOLERANCE):
            raise ValueError(f"Validation branch {label} in {result_json} does not match {train_log} at epoch {best_epoch}.")

    test_a = require_float(result, "validation_selected_test_model_a_top1", result_json)
    test_b = require_float(result, "validation_selected_test_model_b_top1", result_json)
    saved_mean_test = require_float(result, "validation_selected_test_top1", result_json)
    if not math.isclose(saved_mean_test, (test_a + test_b) / 2.0, rel_tol=0.0, abs_tol=TOLERANCE):
        raise ValueError(f"validation_selected_test_top1 is not the branch arithmetic mean in {result_json}.")
    return SeedRecord(method, dataset, seed, best_epoch, max_epochs, val_a, val_b, test_a, test_b, result_json, train_log, checkpoint)


def format_percent(value: float) -> str:
    return f"{100.0 * value:.6f}"


def write_seed_csv(path: Path, records: list[SeedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "Method": METHOD_DISPLAY[record.method],
                    "Dataset": record.dataset,
                    "Seed": record.seed,
                    "Best Epoch by Mean Val": record.best_epoch,
                    "Mean-Val Max Epochs": "|".join(str(epoch) for epoch in record.max_epochs),
                    "Val A": format_percent(record.val_a),
                    "Val B": format_percent(record.val_b),
                    "Selected Branch": record.selected_branch,
                    "Test A": format_percent(record.test_a),
                    "Test B": format_percent(record.test_b),
                    "Selected Test": format_percent(record.selected_test),
                    "Mean Test": format_percent(record.mean_test),
                    "Branch Tie": "Yes" if record.branch_tie else "No",
                    "Result JSON": str(record.result_json),
                    "Train Log": str(record.train_log),
                    "Best Checkpoint": str(record.checkpoint),
                }
            )


def verify_table(records: list[SeedRecord]) -> list[str]:
    failures: list[str] = []
    grouped: dict[tuple[str, str], list[SeedRecord]] = {}
    for record in records:
        grouped.setdefault((record.method, record.dataset), []).append(record)
    for key, expected in sorted(PAPER_TABLE_1.items()):
        group = sorted(grouped.get(key, []), key=lambda record: record.seed)
        if tuple(record.seed for record in group) != SEEDS:
            failures.append(f"{key}: expected seeds {SEEDS}, got {[record.seed for record in group]}")
            continue
        values = [100.0 * record.selected_test for record in group]
        mean = statistics.fmean(values)
        sample_sd = statistics.stdev(values)
        mean_match = f"{mean:.2f}" == f"{expected[0]:.2f}"
        sd_match = f"{sample_sd:.2f}" == f"{expected[1]:.2f}"
        status = "PASS" if mean_match and sd_match else "FAIL"
        print(
            f"{METHOD_DISPLAY[key[0]]} | {key[1]} | {mean:.5f} ± {sample_sd:.5f} | "
            f"paper {expected[0]:.2f} ± {expected[1]:.2f} | {status}"
        )
        if status != "PASS":
            failures.append(f"{key}: reconstructed {mean:.8f} ± {sample_sd:.8f}, expected {expected}")
    return failures


def main() -> int:
    args = parse_args()
    roots = parse_dataset_roots(args.dataset_root)
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if set(methods) != set(METHODS) or len(methods) != len(METHODS):
        raise ValueError(f"--methods must be exactly {METHODS} for Table-1 verification.")
    seeds = parse_csv_ints(args.seeds, "--seeds")
    if seeds != SEEDS:
        raise ValueError(f"--seeds must be exactly {SEEDS} for Table-1 verification.")

    records = [
        reconstruct_seed(dataset, roots[dataset], method, seed)
        for method in METHODS
        for dataset in sorted(roots)
        for seed in seeds
    ]
    write_seed_csv(args.seed_output, records)
    failures = verify_table(records)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print(f"Wrote {len(records)} seed records to {args.seed_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, csv.Error, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
