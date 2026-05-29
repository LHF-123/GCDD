from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json


METHOD_KEYS = {
    "DINOv2 LoRA all noisy samples": "all",
    "DINOv2 LoRA + Full GCDD-clean": "full_gcdd",
    "DINOv2 LoRA + both only": "both_only",
    "DINOv2 LoRA + GCDD+Proto-only added": "gcdd_proto",
    "DINOv2 LoRA + Centroid filtering": "centroid",
}

DEFAULT_RESULT_DIRS = [
    ("Web-Bird", "orig", "outputs/Web-Bird/v1_web_bird/lora"),
    ("Web-Bird", "orig", "outputs/Web-Bird/v1_web_bird/lora_42"),
    ("Web-Bird", "orig", "outputs/Web-Bird/v1_web_bird/lora_88"),
    ("Web-Bird", "0.8", "outputs/Web-Bird/v1_web_bird_0.8/lora_88"),
    ("Web-Bird", "0.9", "outputs/Web-Bird/v1_web_bird_0.9/lora_88"),
    ("Web-Aircraft", "orig", "outputs/Web-Aircraft/v1_web_aircraft/lora_1"),
    ("Web-Aircraft", "orig", "outputs/Web-Aircraft/v1_web_aircraft/lora_42"),
    ("Web-Aircraft", "orig", "outputs/Web-Aircraft/v1_web_aircraft/lora_88"),
    ("Web-Aircraft", "0.8", "outputs/Web-Aircraft/v1_web_aircraft_0.8/lora_88"),
    ("Web-Aircraft", "0.9", "outputs/Web-Aircraft/v1_web_aircraft_0.9/lora_1"),
    ("Web-Aircraft", "0.9", "outputs/Web-Aircraft/v1_web_aircraft_0.9/lora_42"),
    ("Web-Aircraft", "0.9", "outputs/Web-Aircraft/v1_web_aircraft_0.9/lora_88"),
    ("Web-Car", "orig", "outputs/Web-Car/v1_web_car/lora_1"),
    ("Web-Car", "orig", "outputs/Web-Car/v1_web_car/lora_42"),
    ("Web-Car", "orig", "outputs/Web-Car/v1_web_car/lora_88"),
    ("Web-Car", "0.8", "outputs/Web-Car/v1_web_car_0.8/lora_88"),
    ("Web-Car", "0.9", "outputs/Web-Car/v1_web_car_0.9/lora_1"),
    ("Web-Car", "0.9", "outputs/Web-Car/v1_web_car_0.9/lora_42"),
    ("Web-Car", "0.9", "outputs/Web-Car/v1_web_car_0.9/lora_88"),
]

ALL_METHOD_FIELDS = [
    "dataset",
    "ratio",
    "method_key",
    "method",
    "num_seeds",
    "seeds",
    "train_samples",
    "mean_best_top1",
    "std_best_top1",
    "mean_final_top1",
    "source_dirs",
]

ROUTING_FIELDS = [
    "strategy",
    "dataset",
    "GCS_class_balanced",
    "route",
    "selected_ratio",
    "selected_method_key",
    "selected_method",
    "num_seeds",
    "train_samples",
    "mean_best_top1",
    "std_best_top1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GCS hard routing from existing LoRA results.")
    parser.add_argument("--gcs-summary", default="outputs/gcs_analysis/gcs_summary.csv", help="GCS summary CSV.")
    parser.add_argument("--results", default="outputs/gcs_analysis/all_methods_results.csv", help="All-method LoRA result summary CSV.")
    parser.add_argument("--out-dir", default="outputs/gcs_analysis", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.53, help="GCS threshold for strict vs relaxed routing.")
    parser.add_argument("--rebuild-results", action="store_true", help="Rebuild all_methods_results.csv from default output directories.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    results_path = Path(args.results)
    if args.rebuild_results or not results_path.exists():
        all_method_rows = build_default_result_summary()
        ensure_dir(results_path.parent)
        write_csv(results_path, all_method_rows, ALL_METHOD_FIELDS)
    else:
        all_method_rows = read_csv(results_path)

    gcs_rows = read_csv(Path(args.gcs_summary))
    routing_rows = build_routing_rows(gcs_rows, all_method_rows, float(args.threshold))
    strategy_rows = build_strategy_summary(routing_rows)

    write_csv(out_dir / "gcs_routing_results.csv", routing_rows, ROUTING_FIELDS)
    write_csv(out_dir / "gcs_strategy_summary.csv", strategy_rows, strategy_fields(strategy_rows))
    write_json(
        out_dir / "gcs_routing_summary.json",
        {
            "threshold": float(args.threshold),
            "gcs_summary": str(args.gcs_summary),
            "all_methods_results": str(results_path),
            "routing_results": routing_rows,
            "strategy_summary": strategy_rows,
        },
    )
    write_markdown(out_dir / "gcs_routing_summary.md", args, routing_rows, strategy_rows)
    print(f"GCS routing summary written to {out_dir}", flush=True)


def build_default_result_summary() -> list[dict[str, Any]]:
    raw_rows = []
    for dataset, ratio, directory in DEFAULT_RESULT_DIRS:
        path = Path(directory) / "lora_results.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            method = row["method"]
            if method not in METHOD_KEYS:
                continue
            raw_rows.append(
                {
                    "dataset": dataset,
                    "ratio": ratio,
                    "method_key": METHOD_KEYS[method],
                    "method": method,
                    "seed": int(row["seed"]),
                    "train_samples": int(row["train_samples"]),
                    "best_top1": float(row["best_top1"]),
                    "final_top1": float(row["final_top1"]),
                    "source_dir": directory,
                }
            )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((row["dataset"], row["ratio"], row["method_key"]), []).append(row)

    out = []
    for (dataset, ratio, method_key), rows in sorted(grouped.items()):
        best = [float(row["best_top1"]) for row in rows]
        final = [float(row["final_top1"]) for row in rows]
        seeds = sorted(int(row["seed"]) for row in rows)
        out.append(
            {
                "dataset": dataset,
                "ratio": ratio,
                "method_key": method_key,
                "method": rows[0]["method"],
                "num_seeds": len(rows),
                "seeds": ",".join(str(seed) for seed in seeds),
                "train_samples": int(rows[0]["train_samples"]),
                "mean_best_top1": mean(best),
                "std_best_top1": population_std(best),
                "mean_final_top1": mean(final),
                "source_dirs": "|".join(sorted(set(str(row["source_dir"]) for row in rows))),
            }
        )
    return out


def build_routing_rows(gcs_rows: list[dict[str, str]], result_rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    rows = []
    for gcs in gcs_rows:
        dataset = gcs["dataset"]
        gcs_value = float(gcs["GCS_class_balanced"])
        rows.extend(
            [
                make_selection("LoRA all noisy", dataset, gcs_value, "baseline", "orig", "all", result_rows),
                make_selection("Always centroid 0.9", dataset, gcs_value, "relaxed", "0.9", "centroid", result_rows),
                make_selection("Always both only", dataset, gcs_value, "strict", "orig", "both_only", result_rows),
            ]
        )
        if gcs_value < threshold:
            rows.append(make_selection("GCS hard routing", dataset, gcs_value, "strict", "orig", "both_only", result_rows))
        else:
            rows.append(make_selection("GCS hard routing", dataset, gcs_value, "relaxed", "0.9", "centroid", result_rows))
    return rows


def make_selection(
    strategy: str,
    dataset: str,
    gcs_value: float,
    route: str,
    ratio: str,
    method_key: str,
    result_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    match = find_result(result_rows, dataset, ratio, method_key)
    return {
        "strategy": strategy,
        "dataset": dataset,
        "GCS_class_balanced": gcs_value,
        "route": route,
        "selected_ratio": ratio,
        "selected_method_key": method_key,
        "selected_method": match.get("method", ""),
        "num_seeds": match.get("num_seeds", ""),
        "train_samples": match.get("train_samples", ""),
        "mean_best_top1": match.get("mean_best_top1", ""),
        "std_best_top1": match.get("std_best_top1", ""),
    }


def find_result(result_rows: list[dict[str, Any]], dataset: str, ratio: str, method_key: str) -> dict[str, Any]:
    for row in result_rows:
        if row["dataset"] == dataset and row["ratio"] == ratio and row["method_key"] == method_key:
            return row
    return {}


def build_strategy_summary(routing_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = sorted(set(str(row["dataset"]) for row in routing_rows))
    strategies = []
    for row in routing_rows:
        if row["strategy"] not in strategies:
            strategies.append(row["strategy"])

    out = []
    for strategy in strategies:
        selected = [row for row in routing_rows if row["strategy"] == strategy]
        values = []
        item: dict[str, Any] = {"strategy": strategy}
        for dataset in datasets:
            match = next((row for row in selected if row["dataset"] == dataset), None)
            value = float(match["mean_best_top1"]) if match and match.get("mean_best_top1") != "" else None
            item[dataset] = value if value is not None else ""
            if value is not None:
                values.append(value)
        item["Avg"] = mean(values) if values else ""
        out.append(item)
    return out


def strategy_fields(rows: list[dict[str, Any]]) -> list[str]:
    dataset_fields = sorted({key for row in rows for key in row.keys()} - {"strategy", "Avg"})
    return ["strategy", *dataset_fields, "Avg"]


def write_markdown(path: Path, args: argparse.Namespace, routing_rows: list[dict[str, Any]], strategy_rows: list[dict[str, Any]]) -> None:
    fields = strategy_fields(strategy_rows)
    lines = [
        "# GCS Routing Summary",
        "",
        f"- GCS summary: `{args.gcs_summary}`",
        f"- All-method results: `{args.results}`",
        f"- Threshold: `{float(args.threshold):.6f}`",
        "",
        "## Strategy Summary",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" if field == "strategy" else "---:" for field in fields) + " |",
    ]
    for row in strategy_rows:
        lines.append("| " + " | ".join(format_cell(row.get(field, "")) for field in fields) + " |")

    lines.extend(["", "## Routing Decisions", "", "| dataset | GCS | route | selected ratio | selected method | Top-1 |", "| --- | ---: | --- | --- | --- | ---: |"])
    for row in routing_rows:
        if row["strategy"] != "GCS hard routing":
            continue
        lines.append(
            f"| {row['dataset']} | {float(row['GCS_class_balanced']):.6f} | {row['route']} | "
            f"{row['selected_ratio']} | {row['selected_method']} | {float(row['mean_best_top1']):.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_cell(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    value_mean = mean(values)
    return (sum((value - value_mean) ** 2 for value in values) / len(values)) ** 0.5


if __name__ == "__main__":
    main()
