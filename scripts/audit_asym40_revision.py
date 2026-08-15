"""Read-only audit for formal cyclic-asym40 revision diagnostics.

The script reads saved validation-safe manifests, PGDF epoch-25 selection rows,
and validation-selected official-test result files.  It never imports torch,
loads a checkpoint, trains a model, or modifies any existing experiment file.
All outputs are newly created under ``outputs/analysis/asym40_revision_audit``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (1, 42, 88)
TARGET_EPOCH = 25
EPS = 1e-12
MAPPING_TYPE = "cyclic"
EXPERIMENT_ROOT: Path | None = None

EXPERIMENT_DATASET_DIRS = {
    "CUB-200-2011": "CUB_200_2011",
    "Stanford Cars": "Stanford_Cars",
    "FGVC-Aircraft": "FGVC_Aircraft",
}

DATASETS: dict[str, dict[str, Path]] = {
    "CUB-200-2011": {
        "root": ROOT / "outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42",
    },
    "Stanford Cars": {
        "root": ROOT / "outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42",
    },
    "FGVC-Aircraft": {
        "root": ROOT / "outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def number(value: str | float | int | None) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else math.nan


def fmt(value: float | int | str | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def observed_class_sort_key(label: str) -> tuple[int, int | str]:
    """Sort numeric observed labels numerically and class-name labels lexically."""
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def auc(scores: list[float], positive: list[bool]) -> float:
    """Tie-aware ROC AUC with larger score indicating a clean sample."""
    n_positive = sum(positive)
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda item: scores[item])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in order[start:end]:
            ranks[position] = average_rank
        start = end
    positive_rank_sum = sum(rank for rank, clean in zip(ranks, positive) if clean)
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def main_run_root(dataset: str) -> Path:
    if EXPERIMENT_ROOT is not None:
        return EXPERIMENT_ROOT / EXPERIMENT_DATASET_DIRS[dataset] / "checkpoint_validation_s20250726"
    return DATASETS[dataset]["root"] / "checkpoint_validation_s20250726"


def pgdf_selection_path(dataset: str, seed: int) -> Path:
    return main_run_root(dataset) / "pgdf_fixed" / f"seed{seed}" / "selection_rows.csv"


def manifest_rows(dataset: str) -> dict[int, dict[str, str]]:
    rows = read_csv(main_run_root(dataset) / "validation_manifest.csv")
    return {int(row["index"]): row for row in rows}


def reference_rows(dataset: str) -> dict[int, dict[str, str]]:
    rows = read_csv(main_run_root(dataset) / "pgdf_training_pool_reference.csv")
    return {int(row["index"]): row for row in rows}


def load_epoch_rows(
    dataset: str,
    seed: int,
    manifest: dict[int, dict[str, str]],
    reference: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    path = pgdf_selection_path(dataset, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    loaded: list[dict[str, Any]] = []
    for raw in read_csv(path):
        if int(raw["epoch"]) != TARGET_EPOCH:
            continue
        index = int(raw["index"])
        manifest_item = manifest.get(index)
        if manifest_item is None:
            raise RuntimeError(f"{path}: index {index} absent from validation manifest")
        if manifest_item["partition"] != "training_pool":
            raise RuntimeError(f"{path}: held-out validation index {index} appeared in selection rows")
        if raw["path"] != manifest_item["path"]:
            raise RuntimeError(f"{path}: path mismatch at index {index}")
        reference_item = reference.get(index)
        if reference_item is None:
            raise RuntimeError(f"{path}: index {index} absent from PGDF training-pool reference")
        if raw["web_label"] != reference_item["noisy_label"]:
            raise RuntimeError(f"{path}: observed-label mismatch at index {index}")
        proto_score = number(raw["proto_score"])
        if not math.isfinite(proto_score):
            raise RuntimeError(f"{path}: missing frozen prototype score at index {index}")
        if abs(proto_score - number(reference_item["prototype_score"])) > EPS:
            raise RuntimeError(f"{path}: prototype score mismatch with frozen training-pool reference at index {index}")
        loaded.append(
            {
                "index": index,
                "observed_class": raw["web_label"],
                "clean_label": manifest_item["clean_label"],
                "is_clean": raw["web_label"] == manifest_item["clean_label"],
                "loss": number(raw["loss"]),
                "prototype_score": proto_score,
                "loss_selected": raw["loss_selected"] == "yes",
                "prototype_gate": raw["proto_pass"] == "yes",
                "final_selected": raw["state"] == "clean",
            }
        )
    if not loaded:
        raise RuntimeError(f"{path}: epoch {TARGET_EPOCH} is absent")
    indexes = [row["index"] for row in loaded]
    if len(indexes) != len(set(indexes)):
        raise RuntimeError(f"{path}: duplicate epoch-{TARGET_EPOCH} indices")
    expected = {index for index, row in manifest.items() if row["partition"] == "training_pool"}
    if set(indexes) != expected:
        missing = len(expected.difference(indexes))
        extra = len(set(indexes).difference(expected))
        raise RuntimeError(f"{path}: epoch-{TARGET_EPOCH} scope mismatch (missing={missing}, extra={extra})")
    if any(not math.isfinite(float(row["loss"])) for row in loaded):
        raise RuntimeError(f"{path}: non-finite epoch-{TARGET_EPOCH} loss")
    return loaded


def score_summary(rows: list[dict[str, Any]], score_key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["observed_class"])].append(row)
    class_rows: list[dict[str, Any]] = []
    valid: list[tuple[float, int]] = []
    for observed_class, items in sorted(grouped.items(), key=lambda item: observed_class_sort_key(item[0])):
        values = [float(item[score_key]) for item in items]
        labels = [bool(item["is_clean"]) for item in items]
        class_auc = auc(values, labels)
        state = "valid" if math.isfinite(class_auc) else "excluded_single_clean_noisy_state"
        class_rows.append(
            {
                "observed_class": observed_class,
                "class_size": len(items),
                "clean_count": sum(labels),
                "noisy_count": len(items) - sum(labels),
                "clean_ratio": sum(labels) / len(items),
                "auc": class_auc,
                "auc_state": state,
            }
        )
        if math.isfinite(class_auc):
            valid.append((class_auc, len(items)))
    values = [item[0] for item in valid]
    valid_weight = sum(item[1] for item in valid)
    count_below = sum(value < 0.5 for value in values)
    count_equal = sum(math.isclose(value, 0.5, abs_tol=EPS) for value in values)
    count_above = sum(value > 0.5 for value in values)
    summary = {
        "pooled_auc": auc([float(item[score_key]) for item in rows], [bool(item["is_clean"]) for item in rows]),
        "macro_classwise_auc": statistics.mean(values) if values else math.nan,
        "weighted_classwise_auc": (
            sum(value * weight for value, weight in valid) / valid_weight if valid_weight else math.nan
        ),
        "total_observed_classes": len(grouped),
        "valid_classes": len(valid),
        "excluded_classes": len(grouped) - len(valid),
        "classes_auc_lt_05": count_below,
        "classes_auc_eq_05": count_equal,
        "classes_auc_gt_05": count_above,
        "pct_classes_auc_lt_05": 100.0 * count_below / len(valid) if valid else math.nan,
        "pct_classes_auc_gt_05": 100.0 * count_above / len(valid) if valid else math.nan,
        "median_classwise_auc": statistics.median(values) if values else math.nan,
    }
    return summary, class_rows


def composition_summary(rows: list[dict[str, Any]], dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["observed_class"])].append(row)
    details: list[dict[str, Any]] = []
    ratios: list[float] = []
    for observed_class, items in sorted(grouped.items(), key=lambda item: observed_class_sort_key(item[0])):
        clean_count = sum(bool(item["is_clean"]) for item in items)
        noisy_count = len(items) - clean_count
        if clean_count > noisy_count:
            majority = "clean-majority"
        elif clean_count < noisy_count:
            majority = "noisy-majority"
        else:
            majority = "tie"
        ratio = clean_count / len(items)
        ratios.append(ratio)
        details.append(
            {
                "dataset": dataset,
                "observed_class": observed_class,
                "class_size": len(items),
                "clean_count": clean_count,
                "noisy_count": noisy_count,
                "clean_ratio": ratio,
                "majority_state": majority,
            }
        )
    clean_majority = sum(row["majority_state"] == "clean-majority" for row in details)
    noisy_majority = sum(row["majority_state"] == "noisy-majority" for row in details)
    ties = sum(row["majority_state"] == "tie" for row in details)
    return {
        "dataset": dataset,
        "total_observed_classes": len(details),
        "nonempty_observed_classes": len(details),
        "clean_majority_classes": clean_majority,
        "noisy_majority_classes": noisy_majority,
        "tie_classes": ties,
        "clean_majority_pct": 100.0 * clean_majority / len(details),
        "noisy_majority_pct": 100.0 * noisy_majority / len(details),
        "median_clean_ratio": statistics.median(ratios),
        "min_clean_ratio": min(ratios),
        "max_clean_ratio": max(ratios),
    }, details


def selection_metrics(rows: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    if len(rows) != len(selected):
        raise RuntimeError("selection-mask length mismatch")
    clean_total = sum(bool(row["is_clean"]) for row in rows)
    noisy_total = len(rows) - clean_total
    selected_count = sum(selected)
    selected_clean = sum(flag and bool(row["is_clean"]) for row, flag in zip(rows, selected))
    selected_noisy = selected_count - selected_clean
    return {
        "selected_count": selected_count,
        "purity": selected_clean / selected_count if selected_count else math.nan,
        "clean_recall": selected_clean / clean_total if clean_total else math.nan,
        "noisy_retention": selected_noisy / noisy_total if noisy_total else math.nan,
    }


def get_main_result_top1(dataset: str, method_key: str, seed: int) -> tuple[float, Path]:
    path = main_run_root(dataset) / "checkpoint_validation_results.csv"
    candidates = [
        row for row in read_csv(path)
        if row["method_key"] == method_key and int(row["seed"]) == seed
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"{path}: expected one {method_key}/seed{seed} row, found {len(candidates)}")
    value = number(candidates[0]["validation_selected_test_top1"])
    if not math.isfinite(value):
        raise RuntimeError(f"{path}: {method_key}/seed{seed} lacks validation-selected official-test Top-1")
    return value, path


def get_budget_matched_top1(dataset: str, seed: int) -> tuple[float | None, Path]:
    root = (
        main_run_root(dataset)
        if EXPERIMENT_ROOT is not None
        else DATASETS[dataset]["root"] / "checkpoint_validation_dynamic_budget_matched_s20250726"
    )
    result = root / "dynamic_budget_matched" / f"seed{seed}" / "result.json"
    if not result.exists():
        return None, result
    payload = read_json(result)
    if payload.get("checkpoint_protocol") != "fixed_clean_validation_v1":
        raise RuntimeError(f"{result}: unexpected checkpoint protocol")
    if payload.get("official_test_evaluation") != "validation_selected_only":
        raise RuntimeError(f"{result}: expected validation-selected official-test evaluation")
    value = number(payload.get("validation_selected_test_top1"))
    if not math.isfinite(value):
        return None, result
    return value, result


def paired_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    comparisons = {
        "PGDF vs Budget-matched Dynamic": "budget_matched_dynamic",
        "PGDF vs Dynamic small-loss": "dynamic_r08" if EXPERIMENT_ROOT is not None else "dynamic",
        "PGDF vs JAL-CE": "jal_ce",
    }
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    missing: list[str] = []
    for comparison, baseline_method in comparisons.items():
        for dataset in DATASETS:
            for seed in SEEDS:
                pgdf_top1, pgdf_source = get_main_result_top1(dataset, "pgdf_fixed", seed)
                sources.append(rel(pgdf_source))
                if baseline_method == "budget_matched_dynamic":
                    baseline_top1, baseline_source = get_budget_matched_top1(dataset, seed)
                    sources.append(rel(baseline_source))
                else:
                    baseline_top1, baseline_source = get_main_result_top1(dataset, baseline_method, seed)
                    sources.append(rel(baseline_source))
                if baseline_top1 is None:
                    missing.append(
                        f"{comparison}: {dataset} seed {seed} lacks a saved validation-selected official-test Top-1 "
                        f"({rel(baseline_source)})."
                    )
                    delta = math.nan
                    status = "missing_baseline_official_test_top1"
                else:
                    delta = 100.0 * (pgdf_top1 - baseline_top1)
                    status = "complete"
                rows.append(
                    {
                        "comparison": comparison,
                        "dataset": dataset,
                        "training_seed": seed,
                        "pgdf_top1": pgdf_top1,
                        "baseline_top1": baseline_top1 if baseline_top1 is not None else "",
                        "paired_delta_pp": delta,
                        "pair_status": status,
                        "pgdf_source": rel(pgdf_source),
                        "baseline_source": rel(baseline_source),
                    }
                )

    summary: list[dict[str, Any]] = []
    for comparison in comparisons:
        comparison_rows = [row for row in rows if row["comparison"] == comparison]
        for dataset in DATASETS:
            dataset_rows = [row for row in comparison_rows if row["dataset"] == dataset]
            values = [float(row["paired_delta_pp"]) for row in dataset_rows if math.isfinite(float(row["paired_delta_pp"]))]
            summary.append(summarize_paired(comparison, dataset, values, expected=3, scope="dataset_seed_paired"))
        all_values = [float(row["paired_delta_pp"]) for row in comparison_rows if math.isfinite(float(row["paired_delta_pp"]))]
        summary.append(
            summarize_paired(
                comparison,
                "ALL_3_DATASETS",
                all_values,
                expected=9,
                scope="descriptive_pooled_paired_difference",
            )
        )
    return rows, summary, sorted(set(sources)), missing


def summarize_paired(comparison: str, dataset: str, values: list[float], expected: int, scope: str) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "summary_scope": scope,
        "dataset": dataset,
        "expected_pairs": expected,
        "available_pairs": len(values),
        "complete": "yes" if len(values) == expected else "no",
        "mean_paired_delta_pp": statistics.mean(values) if values else math.nan,
        "sample_std_paired_delta_pp": sample_std(values),
        "median_paired_delta_pp": statistics.median(values) if values else math.nan,
        "min_paired_delta_pp": min(values) if values else math.nan,
        "max_paired_delta_pp": max(values) if values else math.nan,
        "pgdf_higher_count": sum(value > 0.0 for value in values),
        "baseline_higher_count": sum(value < 0.0 for value in values),
        "tie_count": sum(math.isclose(value, 0.0, abs_tol=EPS) for value in values),
    }


def read_asym60_context() -> dict[str, Any]:
    prototype_path = ROOT / "outputs/analysis/analysis_auc_classwise/asym60_prototype_classwise.csv"
    negative_path = ROOT / "outputs/analysis/analysis_auc_classwise/asym60_negative_loss_summary.csv"
    composition_path = ROOT / "outputs/analysis/pgdf_existing_results_audit/asym60_observed_class_composition.csv"
    result: dict[str, Any] = {"available": False, "source_files": []}
    if not (prototype_path.exists() and negative_path.exists() and composition_path.exists()):
        return result
    prototype = read_csv(prototype_path)
    negative = [row for row in read_csv(negative_path) if int(row["epoch"]) == TARGET_EPOCH]
    composition = read_csv(composition_path)
    if len(prototype) != 1 or len(negative) != 1:
        return result
    noisy_majority = sum(int(row["noisy"]) > int(row["clean"]) for row in composition)
    result.update(
        {
            "available": True,
            "prototype": prototype[0],
            "negative_loss": negative[0],
            "composition_total": len(composition),
            "composition_noisy_majority": noisy_majority,
            "source_files": [rel(prototype_path), rel(negative_path), rel(composition_path)],
        }
    )
    return result


def make_report(
    output_dir: Path,
    composition: list[dict[str, Any]],
    prototype: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    selection_summary: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    paired_summary: list[dict[str, Any]],
    inputs: list[str],
    proto_invariance: dict[str, Any],
    asym60: dict[str, Any],
    missing: list[str],
) -> None:
    mapping_label = "cyclic" if MAPPING_TYPE == "cyclic" else "fixed-random-derangement"
    lines = [
        f"# Formal {mapping_label}-asym40 补充审计",
        "",
        "## 1. Scope and evidence",
        "",
        f"- 分析范围：CUB-200-2011、Stanford Cars、FGVC-Aircraft 的正式 `{mapping_label}-asym40`、noise seed 42、validation seed 20250726、`fixed_clean_validation_v1`。",
        "- 机制范围严格为 validation manifest 的 `partition=training_pool`；没有使用历史完整 source-train/A.5 范围。",
        "- PGDF 为 `r=0.8, p=0.4, warmup=5, selection_interval=5`，读取第 25 epoch（第 5 次 update）保存的完整 training-pool selection rows。",
        "- clean 指示为 `clean_label == observed/noisy_label`；所有 class-wise 项按 observed/noisy label 分组。ground truth 仅用于本报告的 post-hoc 审计。",
        "- 没有执行 offline inference；没有加载模型或 checkpoint；没有训练、没有 optimizer.step()、没有重新选择 checkpoint，也没有重新评估 official test。",
        "- Top-1 配对仅读取原始 validation-selected official-test 字段 `validation_selected_test_top1`，并直接用未四舍五入的原始数值计算。",
        "",
        "实际读取的正式工件：",
    ]
    lines.extend(f"- `{path}`" for path in inputs)

    lines += [
        "",
        "## 2. Formal asym40 training-pool composition",
        "",
        "| Dataset | Nonempty / total observed classes | Clean-majority | Noisy-majority | Tie | Median clean ratio | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in composition:
        lines.append(
            f"| {row['dataset']} | {row['nonempty_observed_classes']}/{row['total_observed_classes']} | "
            f"{row['clean_majority_classes']} ({fmt(row['clean_majority_pct'])}%) | "
            f"{row['noisy_majority_classes']} ({fmt(row['noisy_majority_pct'])}%) | {row['tie_classes']} | "
            f"{fmt(100*row['median_clean_ratio'])}% | {fmt(100*row['min_clean_ratio'])}% | {fmt(100*row['max_clean_ratio'])}% |"
        )
    lines += ["", "全部 observed class 均非空；逐类明细见 `asym40_observed_class_composition.csv`。"]

    lines += [
        "",
        "## 3. Formal asym40 prototype diagnostic",
        "",
        "该项使用 frozen prototype geometric score，且 scope 为 formal validation-safe training pool；它不同于历史完整 source-train/A.5 诊断。",
        "",
        "| Dataset | Pooled AUC | Macro class-wise AUC | Weighted class-wise AUC | Valid/excluded | AUC < 0.5 | AUC = 0.5 | AUC > 0.5 | Median class-wise AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in prototype:
        lines.append(
            f"| {row['dataset']} | {fmt(row['pooled_auc'], 4)} | {fmt(row['macro_classwise_auc'], 4)} | "
            f"{fmt(row['weighted_classwise_auc'], 4)} | {row['valid_classes']}/{row['excluded_classes']} | "
            f"{row['classes_auc_lt_05']} ({fmt(row['pct_classes_auc_lt_05'])}%) | {row['classes_auc_eq_05']} | "
            f"{row['classes_auc_gt_05']} ({fmt(row['pct_classes_auc_gt_05'])}%) | {fmt(row['median_classwise_auc'], 4)} |"
        )
    lines += ["", "Seed-invariance 核对："]
    for dataset, item in proto_invariance.items():
        lines.append(
            f"- {dataset}: scores identical across seeds = {item['scores_identical']}; prototype gate identical across seeds = {item['gates_identical']}; "
            f"max absolute score difference = {fmt(item['max_abs_score_difference'], 12)}。"
        )

    lines += [
        "",
        "## 4. Epoch-25 negative-loss diagnostic",
        "",
        "clean score 为 `- observed-label CE`。class-wise AUC 只对同时含 clean/noisy 的 observed class 计算；本设置中 valid/excluded class count 由共享 training-pool composition 决定。",
        "",
        "| Dataset | Seed | Pooled AUC | Macro AUC | Weighted AUC | Valid/excluded | AUC < 0.5 | AUC > 0.5 | Median class-wise AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in negative:
        lines.append(
            f"| {row['dataset']} | {row['training_seed']} | {fmt(row['pooled_auc'], 4)} | {fmt(row['macro_classwise_auc'], 4)} | "
            f"{fmt(row['weighted_classwise_auc'], 4)} | {row['valid_classes']}/{row['excluded_classes']} | "
            f"{row['classes_auc_lt_05']} ({fmt(row['pct_classes_auc_lt_05'])}%) | "
            f"{row['classes_auc_gt_05']} ({fmt(row['pct_classes_auc_gt_05'])}%) | {fmt(row['median_classwise_auc'], 4)} |"
        )

    lines += [
        "",
        "## 5. Epoch-25 selection-quality diagnostic",
        "",
        "`pgdf_dynamic_small_loss` 是同一 PGDF 运行的内部 `L^(25)`，不是 standalone Dynamic baseline。`pgdf_intersection` 是 strict loss∩prototype；如 strict intersection 为空类，`pgdf_final_selected_after_fallback` 单列。",
        "",
        "| Dataset | Set | Selected count | Purity | Clean recall | Noisy retention | Fallback count (per run) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selection_summary:
        lines.append(
            f"| {row['dataset']} | {row['set_name']} | {fmt(row['selected_count_mean'])} ± {fmt(row['selected_count_sample_std'])} | "
            f"{fmt(100*row['purity_mean'])}% ± {fmt(100*row['purity_sample_std'])}% | "
            f"{fmt(100*row['clean_recall_mean'])}% ± {fmt(100*row['clean_recall_sample_std'])}% | "
            f"{fmt(100*row['noisy_retention_mean'])}% ± {fmt(100*row['noisy_retention_sample_std'])}% | "
            f"{row['fallback_count_values']} |"
        )

    lines += ["", "## 6. asym40 vs Aircraft asym60 interpretation", ""]
    if asym60.get("available"):
        proto = asym60["prototype"]
        negative_loss = asym60["negative_loss"]
        lines += [
            f"- 已有同口径 Aircraft asym60 审计显示 {asym60['composition_noisy_majority']}/{asym60['composition_total']} 个 observed classes 为 noisy-majority。",
            f"- 同一既有审计中，prototype pooled/macro/weighted AUC 分别为 {fmt(proto['pooled_auc'], 4)}/{fmt(proto['macro_auc'], 4)}/{fmt(proto['weighted_auc'], 4)}；epoch-25 negative-loss pooled/macro/weighted AUC（seed mean）为 {fmt(negative_loss['pooled_mean'], 4)}/{fmt(negative_loss['macro_mean'], 4)}/{fmt(negative_loss['weighted_mean'], 4)}。",
            "- 本报告的 asym40 结果与该 high-noise 边界案例构成谨慎的机制对照：它们描述 observed-class composition 与排序信号的区分能力差异，提供机制层面的补充证据，但不能建立严格因果关系。",
        ]
    else:
        lines.append("- 未找到可读的既有 Aircraft asym60 同口径汇总，故不作数值对照。")

    lines += ["", "## 7. Paired effects", ""]
    for comparison in ("PGDF vs Budget-matched Dynamic", "PGDF vs Dynamic small-loss", "PGDF vs JAL-CE"):
        lines += [
            f"### {comparison}",
            "",
            "| Dataset | seed 1 Δ (pp) | seed 42 Δ (pp) | seed 88 Δ (pp) | Mean Δ ± sample std (pp) | PGDF higher |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for dataset in DATASETS:
            subset = [row for row in paired_rows if row["comparison"] == comparison and row["dataset"] == dataset]
            by_seed = {int(row["training_seed"]): row for row in subset}
            values = [float(row["paired_delta_pp"]) for row in subset if math.isfinite(float(row["paired_delta_pp"]))]
            row_summary = next(
                row for row in paired_summary
                if row["comparison"] == comparison and row["dataset"] == dataset
            )
            lines.append(
                f"| {dataset} | {fmt(float(by_seed[1]['paired_delta_pp']))} | {fmt(float(by_seed[42]['paired_delta_pp']))} | "
                f"{fmt(float(by_seed[88]['paired_delta_pp']))} | {fmt(row_summary['mean_paired_delta_pp'])} ± "
                f"{fmt(row_summary['sample_std_paired_delta_pp'])} | {row_summary['pgdf_higher_count']}/{row_summary['available_pairs']} |"
            )
        pooled = next(
            row for row in paired_summary
            if row["comparison"] == comparison and row["dataset"] == "ALL_3_DATASETS"
        )
        lines += [
            "",
            f"Descriptive pooled paired-difference summary: available {pooled['available_pairs']}/{pooled['expected_pairs']} pairs; "
            f"mean ± sample std = {fmt(pooled['mean_paired_delta_pp'])} ± {fmt(pooled['sample_std_paired_delta_pp'])} pp; "
            f"median/min/max = {fmt(pooled['median_paired_delta_pp'])}/{fmt(pooled['min_paired_delta_pp'])}/{fmt(pooled['max_paired_delta_pp'])} pp; "
            f"PGDF higher/baseline higher/ties = {pooled['pgdf_higher_count']}/{pooled['baseline_higher_count']}/{pooled['tie_count']}.",
            "",
        ]
    budget_pooled = next(
        row for row in paired_summary
        if row["comparison"] == "PGDF vs Budget-matched Dynamic" and row["dataset"] == "ALL_3_DATASETS"
    )
    lines += ["## 8. Paper-ready facts", ""]
    for row in composition:
        lines.append(
            f"- {row['dataset']} asym40 formal training pool 中 {row['clean_majority_classes']}/{row['total_observed_classes']} 个 observed classes 为 clean-majority，"
            f"{row['noisy_majority_classes']}/{row['total_observed_classes']} 为 noisy-majority。"
        )
    for row in prototype:
        lines.append(
            f"- {row['dataset']} asym40 frozen prototype 的 macro class-wise AUC = {fmt(row['macro_classwise_auc'], 4)}，"
            f"weighted class-wise AUC = {fmt(row['weighted_classwise_auc'], 4)}。"
        )
    for dataset in DATASETS:
        rows = [row for row in negative if row["dataset"] == dataset]
        weighted = [float(row["weighted_classwise_auc"]) for row in rows]
        lines.append(
            f"- {dataset} epoch-25 negative-loss weighted class-wise AUC（seeds 1/42/88 mean ± sample std）= "
            f"{fmt(statistics.mean(weighted), 4)} ± {fmt(sample_std(weighted), 4)}。"
        )
    if budget_pooled["complete"] == "yes":
        lines.append(
            f"- PGDF vs Budget-matched Dynamic 的 9-pair descriptive mean paired difference = {fmt(budget_pooled['mean_paired_delta_pp'])} pp；"
            f"PGDF 在 {budget_pooled['pgdf_higher_count']}/9 pairs 中更高。"
        )
    else:
        lines.append(
            f"- PGDF vs Budget-matched Dynamic 的完整 9-pair 结论当前无法验证：仅 {budget_pooled['available_pairs']}/9 个正式 pairs 有原始 validation-selected official-test Top-1。"
        )
    lines += [
        "",
        "## 9. Missing artifacts / blockers",
        "",
    ]
    if missing:
        lines.extend(f"- {item}" for item in missing)
    else:
        lines.append("- 无。")
    lines += [
        "",
        "审计边界：未执行显著性检验、未报告 p-value、未声称统计显著；n=3 的 seed 汇总仅为描述性 mean ± sample std，不作为总体分布估计。",
        "",
        "新输出只位于本目录；未修改原始实验工件、checkpoint、论文 LaTeX 或现有结果汇总。",
    ]
    (output_dir / "revision_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global EXPERIMENT_ROOT, MAPPING_TYPE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/analysis/asym40_revision_audit",
        help="New audit-only output directory; it must not already contain files.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        help=(
            "Optional experiment root containing CUB_200_2011, Stanford_Cars, and "
            "FGVC_Aircraft checkpoint_validation_s20250726 directories."
        ),
    )
    parser.add_argument(
        "--mapping-type",
        choices=("cyclic", "fixed_random_derangement"),
        default="cyclic",
        help="Label the audited class-transition mapping without changing any audit calculation.",
    )
    args = parser.parse_args()
    MAPPING_TYPE = str(args.mapping_type)
    if args.experiment_root is not None:
        EXPERIMENT_ROOT = (
            args.experiment_root
            if args.experiment_root.is_absolute()
            else ROOT / args.experiment_root
        )
        if not EXPERIMENT_ROOT.is_dir():
            raise FileNotFoundError(f"Experiment root does not exist: {EXPERIMENT_ROOT}")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    composition_rows: list[dict[str, Any]] = []
    composition_details: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    prototype_class_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    negative_class_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selection_summary: list[dict[str, Any]] = []
    inputs: list[str] = []
    missing: list[str] = []
    proto_invariance: dict[str, Any] = {}

    for dataset in DATASETS:
        manifest_path = main_run_root(dataset) / "validation_manifest.csv"
        reference_path = main_run_root(dataset) / "pgdf_training_pool_reference.csv"
        inputs.extend([rel(manifest_path), rel(reference_path)])
        manifest = manifest_rows(dataset)
        reference = reference_rows(dataset)
        epoch_by_seed: dict[int, list[dict[str, Any]]] = {}
        for seed in SEEDS:
            source = pgdf_selection_path(dataset, seed)
            inputs.append(rel(source))
            epoch_by_seed[seed] = load_epoch_rows(dataset, seed, manifest, reference)

        canonical = epoch_by_seed[SEEDS[0]]
        composition, details = composition_summary(canonical, dataset)
        composition_rows.append(composition)
        composition_details.extend(details)

        canonical_by_index = {int(row["index"]): row for row in canonical}
        max_score_difference = 0.0
        score_identical = True
        gate_identical = True
        for seed in SEEDS[1:]:
            current_by_index = {int(row["index"]): row for row in epoch_by_seed[seed]}
            if set(current_by_index) != set(canonical_by_index):
                raise RuntimeError(f"{dataset}: PGDF training-pool scope differs by seed")
            for index, canonical_row in canonical_by_index.items():
                current = current_by_index[index]
                difference = abs(float(canonical_row["prototype_score"]) - float(current["prototype_score"]))
                max_score_difference = max(max_score_difference, difference)
                score_identical = score_identical and difference <= EPS
                gate_identical = gate_identical and bool(canonical_row["prototype_gate"]) == bool(current["prototype_gate"])
                if canonical_row["observed_class"] != current["observed_class"]:
                    raise RuntimeError(f"{dataset}: observed labels differ by training seed")
        proto_invariance[dataset] = {
            "scores_identical": "yes" if score_identical else "no",
            "gates_identical": "yes" if gate_identical else "no",
            "max_abs_score_difference": max_score_difference,
        }
        prototype_summary, prototype_detail = score_summary(canonical, "prototype_score")
        prototype_rows.append({"dataset": dataset, **prototype_summary, **proto_invariance[dataset]})
        for detail in prototype_detail:
            prototype_class_rows.append({"dataset": dataset, "training_seed": "shared_frozen_score", **detail})

        for seed, rows in epoch_by_seed.items():
            negative_summary, negative_detail = score_summary(
                [{**row, "negative_loss_score": -float(row["loss"])} for row in rows],
                "negative_loss_score",
            )
            negative_rows.append({"dataset": dataset, "training_seed": seed, "epoch": TARGET_EPOCH, **negative_summary})
            for detail in negative_detail:
                negative_class_rows.append({"dataset": dataset, "training_seed": seed, "epoch": TARGET_EPOCH, **detail})

            strict_intersection = [bool(row["loss_selected"]) and bool(row["prototype_gate"]) for row in rows]
            final_selected = [bool(row["final_selected"]) for row in rows]
            if any(strict and not final for strict, final in zip(strict_intersection, final_selected)):
                raise RuntimeError(f"{dataset}/seed{seed}: final PGDF state omitted a strict-intersection sample")
            fallback_count = sum(final and not strict for strict, final in zip(strict_intersection, final_selected))
            set_masks: list[tuple[str, list[bool]]] = [
                ("full_pool", [True] * len(rows)),
                ("pgdf_dynamic_small_loss", [bool(row["loss_selected"]) for row in rows]),
                ("prototype_gate", [bool(row["prototype_gate"]) for row in rows]),
                ("pgdf_intersection", strict_intersection),
            ]
            if fallback_count:
                set_masks.append(("pgdf_final_selected_after_fallback", final_selected))
            for set_name, mask in set_masks:
                selection_rows.append(
                    {
                        "dataset": dataset,
                        "training_seed": seed,
                        "epoch": TARGET_EPOCH,
                        "set_name": set_name,
                        **selection_metrics(rows, mask),
                        "fallback_count": fallback_count,
                        "strict_intersection_count": sum(strict_intersection),
                        "final_selected_count": sum(final_selected),
                        "source_file": rel(pgdf_selection_path(dataset, seed)),
                    }
                )

    for dataset in DATASETS:
        for set_name in sorted({row["set_name"] for row in selection_rows if row["dataset"] == dataset}):
            rows = [row for row in selection_rows if row["dataset"] == dataset and row["set_name"] == set_name]
            selection_summary.append(
                {
                    "dataset": dataset,
                    "set_name": set_name,
                    "n_training_seeds": len(rows),
                    "selected_count_mean": statistics.mean(float(row["selected_count"]) for row in rows),
                    "selected_count_sample_std": sample_std([float(row["selected_count"]) for row in rows]),
                    "purity_mean": statistics.mean(float(row["purity"]) for row in rows),
                    "purity_sample_std": sample_std([float(row["purity"]) for row in rows]),
                    "clean_recall_mean": statistics.mean(float(row["clean_recall"]) for row in rows),
                    "clean_recall_sample_std": sample_std([float(row["clean_recall"]) for row in rows]),
                    "noisy_retention_mean": statistics.mean(float(row["noisy_retention"]) for row in rows),
                    "noisy_retention_sample_std": sample_std([float(row["noisy_retention"]) for row in rows]),
                    "fallback_count_values": ",".join(str(row["fallback_count"]) for row in sorted(rows, key=lambda row: int(row["training_seed"]))),
                }
            )

    paired_rows, paired_summary, paired_sources, paired_missing = paired_outputs()
    inputs.extend(paired_sources)
    missing.extend(paired_missing)
    asym60 = read_asym60_context()
    inputs.extend(asym60.get("source_files", []))

    write_csv(
        output_dir / "asym40_observed_class_composition.csv",
        composition_details,
        ["dataset", "observed_class", "class_size", "clean_count", "noisy_count", "clean_ratio", "majority_state"],
    )
    write_csv(
        output_dir / "asym40_prototype_auc.csv",
        prototype_rows,
        [
            "dataset", "pooled_auc", "macro_classwise_auc", "weighted_classwise_auc", "total_observed_classes",
            "valid_classes", "excluded_classes", "classes_auc_lt_05", "classes_auc_eq_05", "classes_auc_gt_05",
            "pct_classes_auc_lt_05", "pct_classes_auc_gt_05", "median_classwise_auc", "scores_identical",
            "gates_identical", "max_abs_score_difference",
        ],
    )
    write_csv(
        output_dir / "asym40_prototype_auc_by_class.csv",
        prototype_class_rows,
        ["dataset", "training_seed", "observed_class", "class_size", "clean_count", "noisy_count", "clean_ratio", "auc", "auc_state"],
    )
    write_csv(
        output_dir / "asym40_negative_loss_epoch25_auc.csv",
        negative_rows,
        [
            "dataset", "training_seed", "epoch", "pooled_auc", "macro_classwise_auc", "weighted_classwise_auc",
            "total_observed_classes", "valid_classes", "excluded_classes", "classes_auc_lt_05", "classes_auc_eq_05",
            "classes_auc_gt_05", "pct_classes_auc_lt_05", "pct_classes_auc_gt_05", "median_classwise_auc",
        ],
    )
    write_csv(
        output_dir / "asym40_negative_loss_epoch25_auc_by_class.csv",
        negative_class_rows,
        ["dataset", "training_seed", "epoch", "observed_class", "class_size", "clean_count", "noisy_count", "clean_ratio", "auc", "auc_state"],
    )
    write_csv(
        output_dir / "asym40_selection_quality_epoch25.csv",
        selection_rows,
        [
            "dataset", "training_seed", "epoch", "set_name", "selected_count", "purity", "clean_recall",
            "noisy_retention", "fallback_count", "strict_intersection_count", "final_selected_count", "source_file",
        ],
    )
    write_csv(
        output_dir / "asym40_selection_quality_epoch25_summary.csv",
        selection_summary,
        [
            "dataset", "set_name", "n_training_seeds", "selected_count_mean", "selected_count_sample_std",
            "purity_mean", "purity_sample_std", "clean_recall_mean", "clean_recall_sample_std", "noisy_retention_mean",
            "noisy_retention_sample_std", "fallback_count_values",
        ],
    )
    write_csv(
        output_dir / "paired_effects_asym40.csv",
        paired_rows,
        [
            "comparison", "dataset", "training_seed", "pgdf_top1", "baseline_top1", "paired_delta_pp", "pair_status",
            "pgdf_source", "baseline_source",
        ],
    )
    write_csv(
        output_dir / "paired_effects_summary.csv",
        paired_summary,
        [
            "comparison", "summary_scope", "dataset", "expected_pairs", "available_pairs", "complete", "mean_paired_delta_pp",
            "sample_std_paired_delta_pp", "median_paired_delta_pp", "min_paired_delta_pp", "max_paired_delta_pp",
            "pgdf_higher_count", "baseline_higher_count", "tie_count",
        ],
    )
    evidence = {
        "scope": {
            "noise_setting": f"{MAPPING_TYPE}-asym40",
            "noise_seed": 42,
            "validation_seed": 20250726,
            "training_seeds": list(SEEDS),
            "pgdf": {"r": 0.8, "p": 0.4, "warmup_epochs": 5, "selection_interval": 5, "epoch": TARGET_EPOCH},
            "pool": "formal validation-safe noisy training pool only",
        },
        "actions": {
            "offline_inference": False,
            "model_loading": False,
            "retraining": False,
            "optimizer_step": False,
            "official_test_reevaluation": False,
            "significance_test": False,
        },
        "input_files": sorted(set(inputs)),
        "prototype_seed_invariance": proto_invariance,
        "missing_artifacts": missing,
        "asym60_context_used": asym60,
    }
    (output_dir / "audit_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_report(
        output_dir,
        composition_rows,
        prototype_rows,
        negative_rows,
        selection_summary,
        paired_rows,
        paired_summary,
        sorted(set(inputs)),
        proto_invariance,
        asym60,
        missing,
    )
    print(json.dumps({"output_dir": rel(output_dir), "missing_artifacts": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
