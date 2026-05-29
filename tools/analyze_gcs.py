from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json


DEFAULT_DATASETS = [
    {
        "name": "Web-Bird",
        "scores": "outputs/Web-Bird/v1_web_bird/centroid_scores.csv",
        "best_strict": "yes",
        "best_high_ratio": "no",
    },
    {
        "name": "Web-Aircraft",
        "scores": "outputs/Web-Aircraft/v1_web_aircraft/centroid_scores.csv",
        "best_strict": "no",
        "best_high_ratio": "yes",
    },
    {
        "name": "Web-Car",
        "scores": "outputs/Web-Car/v1_web_car/centroid_scores.csv",
        "best_strict": "no",
        "best_high_ratio": "yes",
    },
]


SUMMARY_FIELDS = [
    "dataset",
    "scores_path",
    "num_samples",
    "num_classes",
    "GCS_sample_mean",
    "GCS_class_balanced",
    "class_mean_std",
    "class_mean_min",
    "class_mean_p25",
    "class_mean_median",
    "class_mean_p75",
    "class_mean_max",
    "best_strict",
    "best_high_ratio",
]

CLASS_FIELDS = ["dataset", "web_label", "count", "mean_centroid_score", "std_centroid_score"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze global compactness score from centroid score tables.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="Dataset centroid score CSV. Can be repeated. Defaults to current Web-Bird/Aircraft/Car outputs.",
    )
    parser.add_argument("--out-dir", default="outputs/gcs_analysis", help="Output directory for GCS tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_specs = parse_dataset_specs(args.dataset) if args.dataset else DEFAULT_DATASETS
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    summary_rows = []
    class_rows = []
    for spec in dataset_specs:
        summary, classes = analyze_dataset(spec)
        summary_rows.append(summary)
        class_rows.extend(classes)

    summary_rows = sorted(summary_rows, key=lambda row: float(row["GCS_class_balanced"]))
    write_csv(out_dir / "gcs_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "class_gcs.csv", class_rows, CLASS_FIELDS)
    write_json(out_dir / "gcs_summary.json", {"summary": summary_rows})
    write_summary(out_dir / "analysis_summary.md", summary_rows)
    print(f"GCS analysis written to {out_dir}", flush=True)


def parse_dataset_specs(items: list[str]) -> list[dict[str, str]]:
    specs = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--dataset must use NAME=CSV format, got: {item}")
        name, csv_path = item.split("=", 1)
        specs.append({"name": name, "scores": csv_path, "best_strict": "", "best_high_ratio": ""})
    return specs


def analyze_dataset(spec: dict[str, str]) -> tuple[dict[str, object], list[dict[str, object]]]:
    path = Path(spec["scores"])
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Empty centroid score file: {path}")
    required = {"web_label", "centroid_score"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing required fields: {sorted(missing)}")

    scores = [float(row["centroid_score"]) for row in rows]
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["web_label"], []).append(float(row["centroid_score"]))

    class_rows = []
    class_means = []
    for label in sorted(grouped):
        values = grouped[label]
        class_mean = mean(values)
        class_means.append(class_mean)
        class_rows.append(
            {
                "dataset": spec["name"],
                "web_label": label,
                "count": len(values),
                "mean_centroid_score": class_mean,
                "std_centroid_score": population_std(values),
            }
        )

    summary = {
        "dataset": spec["name"],
        "scores_path": str(path),
        "num_samples": len(scores),
        "num_classes": len(grouped),
        "GCS_sample_mean": mean(scores),
        # Class-balanced GCS gives every class equal weight and avoids large classes dominating.
        "GCS_class_balanced": mean(class_means),
        "class_mean_std": population_std(class_means),
        "class_mean_min": min(class_means),
        "class_mean_p25": percentile(class_means, 0.25),
        "class_mean_median": percentile(class_means, 0.50),
        "class_mean_p75": percentile(class_means, 0.75),
        "class_mean_max": max(class_means),
        "best_strict": spec.get("best_strict", ""),
        "best_high_ratio": spec.get("best_high_ratio", ""),
    }
    return summary, class_rows


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    value_mean = mean(values)
    return (sum((value - value_mean) ** 2 for value in values) / len(values)) ** 0.5


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    weight = pos - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    strict_values = [float(row["GCS_class_balanced"]) for row in rows if row.get("best_strict") == "yes"]
    high_values = [float(row["GCS_class_balanced"]) for row in rows if row.get("best_high_ratio") == "yes"]
    tau = ""
    if strict_values and high_values and max(strict_values) < min(high_values):
        tau = f"{(max(strict_values) + min(high_values)) / 2.0:.6f}"

    lines = [
        "# GCS Analysis",
        "",
        "GCS measures how compact samples are around their web-label class prototype.",
        "The main value is `GCS_class_balanced`, which averages per-class compactness so each class has equal weight.",
        "",
        "| dataset | samples | classes | GCS_sample_mean | GCS_class_balanced | best strict | best high-ratio |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {int(row['num_samples'])} | {int(row['num_classes'])} | "
            f"{float(row['GCS_sample_mean']):.6f} | {float(row['GCS_class_balanced']):.6f} | "
            f"{row.get('best_strict', '')} | {row.get('best_high_ratio', '')} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append("Current ordering by class-balanced GCS:")
    lines.append("")
    lines.append("```text")
    for row in rows:
        lines.append(f"{row['dataset']}: {float(row['GCS_class_balanced']):.6f}")
    lines.append("```")
    if tau:
        lines.extend(
            [
                "",
                "A simple routing threshold is possible for the current three datasets:",
                "",
                "```text",
                f"if GCS_class_balanced < {tau}: strict mode",
                f"else: relaxed mode",
                "```",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
