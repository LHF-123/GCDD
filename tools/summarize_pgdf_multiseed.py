from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json
from tools.analyze_pgdf_gt_purity import (
    build_gt_index,
    compute_selection_metrics,
    display_path,
    find_all_results,
    read_all_training_top1,
    resolve_selection_source,
    safe_ratio,
    top1_to_percent,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_dir: Path
    noise_index: Path
    dynamic_dir: Path
    pgdf_auto_prefix: str
    all_results: Path | None = None


DEFAULT_DATASETS = [
    DatasetSpec(
        name="CUB asym40",
        input_dir=Path("outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42"),
        noise_index=Path("outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"),
        dynamic_dir=Path("outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/dynamic_loss_42"),
        pgdf_auto_prefix="proto_guided_dynamic_auto_",
        all_results=Path("outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/lora_30_42/lora_results.csv"),
    ),
    DatasetSpec(
        name="Stanford Cars asym40",
        input_dir=Path("outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42"),
        noise_index=Path("outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"),
        dynamic_dir=Path("outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/dynamic_loss_42"),
        pgdf_auto_prefix="proto_guided_dynamic_auto_",
        all_results=Path("outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/lora_30_42/lora_results.csv"),
    ),
    DatasetSpec(
        name="FGVC-Aircraft asym40",
        input_dir=Path("outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42"),
        noise_index=Path("outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"),
        dynamic_dir=Path("outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/dynamic_loss_42"),
        pgdf_auto_prefix="proto_guided_dynamic_auto_",
        all_results=Path("outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/lora42/lora_results.csv"),
    ),
]


SUMMARY_FIELDS = [
    "dataset",
    "base_train_samples",
    "base_clean_total",
    "base_clean_ratio_pct",
    "all_noisy_best_top1_pct",
    "dynamic_seed",
    "dynamic_best_top1_pct",
    "dynamic_final_top1_pct",
    "dynamic_selected",
    "dynamic_purity_pct",
    "dynamic_clean_recall_pct",
    "pgdf_seeds",
    "pgdf_auto_p_values",
    "pgdf_jaccard_mean",
    "pgdf_jaccard_std",
    "pgdf_best_top1_mean_pct",
    "pgdf_best_top1_std_pct",
    "pgdf_best_top1_min_pct",
    "pgdf_best_top1_max_pct",
    "pgdf_final_top1_mean_pct",
    "pgdf_final_top1_std_pct",
    "pgdf_selected_mean",
    "pgdf_selected_std",
    "pgdf_purity_mean_pct",
    "pgdf_purity_std_pct",
    "pgdf_clean_recall_mean_pct",
    "pgdf_clean_recall_std_pct",
    "pgdf_gain_vs_dynamic_best_mean_pct",
    "pgdf_gain_vs_dynamic_best_min_pct",
    "pgdf_gain_vs_dynamic_best_max_pct",
    "pgdf_purity_gain_mean_pct",
]

DETAIL_FIELDS = [
    "dataset",
    "method",
    "seed",
    "auto_p",
    "jaccard",
    "best_top1_pct",
    "final_top1_pct",
    "selected",
    "clean_selected",
    "purity_pct",
    "clean_recall_pct",
    "selection_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PGDF auto multiseed results with GT-purity metrics.")
    parser.add_argument("--out-dir", default="outputs/analysis/pgdf_3seed_summary", help="Output directory.")
    parser.add_argument("--seeds", default="1,42,88", help="Comma-separated PGDF seeds to summarize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    seeds = parse_int_list(args.seeds)

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for spec in DEFAULT_DATASETS:
        summary, details = summarize_dataset(spec, seeds)
        summary_rows.append(summary)
        detail_rows.extend(details)

    write_csv(out_dir / "pgdf_3seed_combined_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "pgdf_3seed_per_seed_details.csv", detail_rows, DETAIL_FIELDS)
    write_json(out_dir / "pgdf_3seed_summary.json", {"summary": summary_rows, "details": detail_rows})
    write_markdown(out_dir / "pgdf_3seed_combined_summary.md", summary_rows, detail_rows)
    print(f"PGDF 3seed summary written to {out_dir}", flush=True)


def summarize_dataset(spec: DatasetSpec, seeds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_dir = resolve_repo_path(spec.input_dir)
    noise_index = resolve_repo_path(spec.noise_index)
    dynamic_dir = resolve_repo_path(spec.dynamic_dir)

    gt_map, gt_samples = build_gt_index(read_csv(noise_index))
    train_count = len(gt_samples)
    base_clean_total = sum(1 for sample in gt_samples if sample.is_clean)
    base_clean_ratio_pct = safe_ratio(base_clean_total, train_count) * 100.0
    all_top1 = read_all_training_top1(resolve_optional_repo_path(spec.all_results) or find_all_results(input_dir))

    dynamic_result = read_dynamic_result(dynamic_dir)
    dynamic_selection = resolve_selection_source(dynamic_dir, "dynamic", "last", retention_ratio=0.8)
    dynamic_metrics = compute_selection_metrics(read_csv(dynamic_selection), gt_map, base_clean_total)
    dynamic_detail = {
        "dataset": spec.name,
        "method": "Dynamic r=0.8",
        "seed": dynamic_result.get("seed", ""),
        "auto_p": "",
        "jaccard": "",
        "best_top1_pct": top1_to_percent(dynamic_result["best_top1"]),
        "final_top1_pct": top1_to_percent(dynamic_result["final_top1"]),
        "selected": dynamic_metrics.selected,
        "clean_selected": dynamic_metrics.clean_selected,
        "purity_pct": dynamic_metrics.purity * 100.0,
        "clean_recall_pct": dynamic_metrics.clean_recall * 100.0,
        "selection_source": display_path(dynamic_selection),
    }

    pgdf_details = []
    for seed in seeds:
        pgdf_dir = input_dir / f"{spec.pgdf_auto_prefix}{seed}"
        if not pgdf_dir.exists():
            continue
        pgdf_details.append(read_pgdf_seed_detail(spec.name, pgdf_dir, gt_map, base_clean_total))
    if not pgdf_details:
        raise FileNotFoundError(f"No PGDF auto seed directories found for {spec.name}: {input_dir / spec.pgdf_auto_prefix}<seed>")

    best_values = numeric_values(pgdf_details, "best_top1_pct")
    final_values = numeric_values(pgdf_details, "final_top1_pct")
    selected_values = numeric_values(pgdf_details, "selected")
    purity_values = numeric_values(pgdf_details, "purity_pct")
    recall_values = numeric_values(pgdf_details, "clean_recall_pct")
    jaccard_values = numeric_values(pgdf_details, "jaccard")
    dynamic_best = float(dynamic_detail["best_top1_pct"])
    dynamic_purity = float(dynamic_detail["purity_pct"])

    summary = {
        "dataset": spec.name,
        "base_train_samples": train_count,
        "base_clean_total": base_clean_total,
        "base_clean_ratio_pct": base_clean_ratio_pct,
        "all_noisy_best_top1_pct": all_top1,
        "dynamic_seed": dynamic_detail["seed"],
        "dynamic_best_top1_pct": dynamic_detail["best_top1_pct"],
        "dynamic_final_top1_pct": dynamic_detail["final_top1_pct"],
        "dynamic_selected": dynamic_detail["selected"],
        "dynamic_purity_pct": dynamic_detail["purity_pct"],
        "dynamic_clean_recall_pct": dynamic_detail["clean_recall_pct"],
        "pgdf_seeds": "/".join(str(row["seed"]) for row in pgdf_details),
        "pgdf_auto_p_values": "/".join(format_short_float(row["auto_p"]) for row in pgdf_details),
        "pgdf_jaccard_mean": mean(jaccard_values),
        "pgdf_jaccard_std": sample_std(jaccard_values),
        "pgdf_best_top1_mean_pct": mean(best_values),
        "pgdf_best_top1_std_pct": sample_std(best_values),
        "pgdf_best_top1_min_pct": min(best_values),
        "pgdf_best_top1_max_pct": max(best_values),
        "pgdf_final_top1_mean_pct": mean(final_values),
        "pgdf_final_top1_std_pct": sample_std(final_values),
        "pgdf_selected_mean": mean(selected_values),
        "pgdf_selected_std": sample_std(selected_values),
        "pgdf_purity_mean_pct": mean(purity_values),
        "pgdf_purity_std_pct": sample_std(purity_values),
        "pgdf_clean_recall_mean_pct": mean(recall_values),
        "pgdf_clean_recall_std_pct": sample_std(recall_values),
        "pgdf_gain_vs_dynamic_best_mean_pct": mean(best_values) - dynamic_best,
        "pgdf_gain_vs_dynamic_best_min_pct": min(best_values) - dynamic_best,
        "pgdf_gain_vs_dynamic_best_max_pct": max(best_values) - dynamic_best,
        "pgdf_purity_gain_mean_pct": mean(purity_values) - dynamic_purity,
    }
    return summary, [dynamic_detail, *pgdf_details]


def read_dynamic_result(dynamic_dir: Path) -> dict[str, str]:
    rows = read_csv(dynamic_dir / "dynamic_loss_results.csv")
    for row in rows:
        if same_float_text(row.get("retention_ratio", ""), "0.8"):
            return row
    raise ValueError(f"No dynamic r=0.8 row found in {dynamic_dir / 'dynamic_loss_results.csv'}")


def read_pgdf_seed_detail(dataset: str, pgdf_dir: Path, gt_map: dict[str, Any], base_clean_total: int) -> dict[str, Any]:
    result_rows = read_csv(pgdf_dir / "pgdf_results.csv")
    if not result_rows:
        raise ValueError(f"Empty PGDF result file: {pgdf_dir / 'pgdf_results.csv'}")
    result = result_rows[0]
    selection = resolve_selection_source(pgdf_dir, "pgdf", "last", retention_ratio=0.8)
    metrics = compute_selection_metrics(read_csv(selection), gt_map, base_clean_total)
    return {
        "dataset": dataset,
        "method": "PGDF auto",
        "seed": result["seed"],
        "auto_p": float(result["proto_keep_ratio"]),
        "jaccard": float(result["auto_proto_jaccard"]),
        "best_top1_pct": top1_to_percent(result["best_top1"]),
        "final_top1_pct": top1_to_percent(result["final_top1"]),
        "selected": metrics.selected,
        "clean_selected": metrics.clean_selected,
        "purity_pct": metrics.purity * 100.0,
        "clean_recall_pct": metrics.clean_recall * 100.0,
        "selection_source": display_path(selection),
    }


def write_markdown(path: Path, summary_rows: list[dict[str, Any]], detail_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# PGDF Auto 3seed Combined Summary",
        "",
        "Dynamic is the current r=0.8 seed42 baseline. PGDF uses seeds 1/42/88.",
        "",
        "## Combined Table",
        "",
        (
            "| Dataset | Base purity | All noisy Top-1 | Dynamic Top-1 | Dynamic purity | "
            "PGDF p | PGDF J | PGDF Top-1 | PGDF final | PGDF purity | PGDF recall | Top-1 gain | Purity gain |"
        ),
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {fmt(row['base_clean_ratio_pct'])} | {fmt(row['all_noisy_best_top1_pct'])} | "
            f"{fmt(row['dynamic_best_top1_pct'])} | {fmt(row['dynamic_purity_pct'])} | "
            f"{row['pgdf_auto_p_values']} | {fmt(row['pgdf_jaccard_mean'])} +/- {fmt(row['pgdf_jaccard_std'])} | "
            f"{fmt(row['pgdf_best_top1_mean_pct'])} +/- {fmt(row['pgdf_best_top1_std_pct'])} | "
            f"{fmt(row['pgdf_final_top1_mean_pct'])} +/- {fmt(row['pgdf_final_top1_std_pct'])} | "
            f"{fmt(row['pgdf_purity_mean_pct'])} +/- {fmt(row['pgdf_purity_std_pct'])} | "
            f"{fmt(row['pgdf_clean_recall_mean_pct'])} +/- {fmt(row['pgdf_clean_recall_std_pct'])} | "
            f"{fmt_signed(row['pgdf_gain_vs_dynamic_best_mean_pct'])} | {fmt_signed(row['pgdf_purity_gain_mean_pct'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed Details",
            "",
            "| Dataset | Method | Seed | p | J | Best Top-1 | Final Top-1 | Selected | Purity | Clean recall |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in detail_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['seed']} | {fmt(row['auto_p'])} | {fmt(row['jaccard'])} | "
            f"{fmt(row['best_top1_pct'])} | {fmt(row['final_top1_pct'])} | {int(row['selected'])} | "
            f"{fmt(row['purity_pct'])} | {fmt(row['clean_recall_pct'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def resolve_optional_repo_path(path: Path | None) -> Path | None:
    return resolve_repo_path(path) if path is not None else None


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def same_float_text(left: str, right: str) -> bool:
    try:
        return abs(float(left) - float(right)) < 1.0e-9
    except ValueError:
        return left == right


def format_short_float(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):g}"


def fmt(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.2f}"


def fmt_signed(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):+.2f}"


if __name__ == "__main__":
    main()
