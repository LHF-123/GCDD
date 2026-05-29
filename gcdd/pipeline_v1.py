from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .baselines import centroid_scores, per_class_keep_counts, select_top_per_class
from .data import ImageRecord, build_verified_index
from .features import FeatureExtractor
from .graph import build_rrf_graphs
from .io_utils import ensure_dir, write_csv, write_json, write_yaml
from .pipeline_v0 import class_clip_rate, top_bottom_summary, write_index, write_scores, write_split
from .progress import log_stage
from .scoring import compute_scores
from .training import predict_logits, summarize_epoch_logs, train_linear_eval, true_class_scores


V1_METHODS = [
    "DINOv2 Linear all",
    "Confidence filtering",
    "Loss filtering",
    "Centroid filtering",
    "Full GCDD-clean",
]


def run_v1_web_bird(cfg: dict) -> None:
    output_dir = make_output_dir(cfg)
    ensure_dir(output_dir)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/9] Building train/eval indexes and checking bad images...")
    train_records, train_bad = build_verified_index(
        cfg,
        split=cfg["dataset"].get("train_split", "train"),
        samples_per_class=cfg["dataset"].get("max_train_per_class"),
        max_classes=cfg["dataset"].get("max_classes"),
    )
    eval_records, eval_bad = build_verified_index(
        cfg,
        split=cfg["dataset"].get("eval_split", "val"),
        samples_per_class=cfg["dataset"].get("max_eval_per_class"),
        max_classes=cfg["dataset"].get("max_classes"),
    )
    if not train_records:
        raise RuntimeError("No valid train images found.")
    if not eval_records:
        raise RuntimeError("No valid eval images found. Check dataset.eval_split.")
    log_stage(f"[1/9] Train/eval valid images: {len(train_records)} / {len(eval_records)}. Bad images: {len(train_bad) + len(eval_bad)}.")

    log_stage("[2/9] Loading or extracting train/eval features...")
    feature_extractor: list[FeatureExtractor | None] = [None]
    train_features, train_records, train_feature_failures = load_or_extract_feature_set(output_dir, "train", train_records, cfg, feature_extractor)
    eval_features, eval_records, eval_feature_failures = load_or_extract_feature_set(output_dir, "eval", eval_records, cfg, feature_extractor)
    write_index(output_dir / "dataset_index.csv", train_records)
    write_index(output_dir / "eval_index.csv", eval_records)
    write_bad_images(output_dir, train_bad, eval_bad, train_feature_failures, eval_feature_failures)

    train_labels = np.array([record.label for record in train_records])
    eval_labels = np.array([record.label for record in eval_records])
    graph_features = {name: train_features[name] for name in ("cls", "gap", "top")}
    log_stage("[3/9] Building class/global RRF graphs...")
    graphs = build_rrf_graphs(graph_features, train_labels, cfg)
    save_graphs(output_dir, graphs)

    log_stage("[4/9] Computing GCDD scores and Adaptive-Otsu split...")
    metrics, split_info = compute_scores(train_labels, graphs, cfg)
    gcdd_mask = split_info["state"] == "clean"
    write_scores(output_dir / "gcdd_scores.csv", train_records, metrics, split_info)
    write_split(output_dir / "sample_split.csv", train_records, split_info["state"])
    write_clean_thresholds(output_dir / "clean_thresholds.csv", train_labels, metrics, split_info)
    write_q_same_top_bottom(output_dir / "q_same_top_bottom.csv", metrics)
    write_s_clean_distribution(output_dir / "s_clean_distribution.csv", train_labels, metrics, split_info)
    write_neighbor_sanity(output_dir / "neighbor_sanity_samples.csv", train_records, metrics, graphs)

    train_feature_name = cfg["train"].get("feature", "cls")
    train_x = train_features[train_feature_name]
    eval_x = eval_features[train_feature_name]
    keep_counts = per_class_keep_counts(train_labels, gcdd_mask)

    all_mask = np.ones(len(train_labels), dtype=bool)
    train_logs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    log_stage("[5/9] Training DINOv2 Linear all.")
    all_logs, all_model = train_linear_eval(train_x, train_labels, eval_x, eval_labels, all_mask, cfg, "DINOv2 Linear all")
    train_logs.extend(all_logs)
    summaries.append(summarize_epoch_logs("DINOv2 Linear all", all_logs))

    all_train_logits = predict_logits(all_model, train_x)
    confidence, loss = true_class_scores(all_train_logits, train_labels, all_model.classes)
    write_linear_all_scores(output_dir / "linear_all_train_scores.csv", train_records, confidence, loss)
    log_stage("[6/9] Building aligned baseline splits.")
    centroid_score = centroid_scores(train_x, train_labels)
    write_centroid_scores(output_dir / "centroid_scores.csv", train_records, centroid_score)
    baseline_masks = {
        "Confidence filtering": select_top_per_class(confidence, train_labels, keep_counts, largest=True),
        "Loss filtering": select_top_per_class(loss, train_labels, keep_counts, largest=False),
        "Centroid filtering": select_top_per_class(centroid_score, train_labels, keep_counts, largest=True),
        "Full GCDD-clean": gcdd_mask,
    }
    write_baseline_splits(output_dir, train_records, baseline_masks)

    log_stage("[7/9] Training aligned filtering baselines.")
    for method in V1_METHODS[1:]:
        logs, _ = train_linear_eval(train_x, train_labels, eval_x, eval_labels, baseline_masks[method], cfg, method)
        train_logs.extend(logs)
        summaries.append(summarize_epoch_logs(method, logs))

    log_stage("[8/9] Writing logs and comparison tables.")
    write_csv(output_dir / "train_log.csv", train_logs, ["method", "epoch", "lr", "loss", "top1", "top5", "train_samples", "eval_samples"])
    write_csv(
        output_dir / "eval_log.csv",
        train_logs,
        ["method", "epoch", "lr", "loss", "top1", "top5", "train_samples", "eval_samples"],
    )
    write_csv(
        output_dir / "baseline_compare_web_bird.csv",
        summaries,
        ["method", "train_samples", "eval_samples", "best_epoch", "best_top1", "best_top5", "final_top1", "final_top5", "last10_mean", "last10_std"],
    )

    summary = build_v1_summary(
        cfg,
        train_records,
        eval_records,
        train_bad,
        eval_bad,
        train_feature_failures,
        eval_feature_failures,
        train_features,
        eval_features,
        graphs,
        metrics,
        split_info,
        summaries,
    )
    write_json(output_dir / "v1_summary.json", summary)
    write_run_summary(output_dir / "run_summary.md", summary)
    log_stage(f"[9/9] V1 finished. Summary: {output_dir / 'run_summary.md'}")


def make_output_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["root"]) / cfg["dataset"]["name"] / cfg["output"]["version"]


def load_or_extract_feature_set(
    output_dir: Path,
    prefix: str,
    records: list[ImageRecord],
    cfg: dict,
    feature_extractor: list[FeatureExtractor | None],
) -> tuple[dict[str, np.ndarray], list[ImageRecord], list[dict[str, str]]]:
    if cfg["feature"].get("reuse", True):
        loaded = try_load_features(output_dir, prefix, records)
        if loaded is not None:
            log_stage(f"[features] Reusing cached {prefix} features from {output_dir}.")
            return loaded, records, []
    log_stage(f"[features] No valid cache for {prefix}; extracting features.")
    if feature_extractor[0] is None:
        feature_extractor[0] = FeatureExtractor(cfg)
    features, kept_records, failures = feature_extractor[0].extract(records, stage_name=prefix)
    kept_records = reindex_records(kept_records)
    save_features(output_dir, prefix, features, kept_records)
    return features, kept_records, failures


def try_load_features(output_dir: Path, prefix: str, records: list[ImageRecord]) -> dict[str, np.ndarray] | None:
    paths = feature_paths(output_dir, prefix)
    if not all(path.exists() for path in paths.values()):
        return None
    cached_paths = paths["paths"].read_text(encoding="utf-8").splitlines()
    current_paths = [str(record.path) for record in records]
    if cached_paths != current_paths:
        return None
    features = {name: np.load(paths[name]) for name in ("cls", "gap", "top")}
    if any(value.shape[0] != len(records) for value in features.values()):
        return None
    return features


def save_features(output_dir: Path, prefix: str, features: dict[str, np.ndarray], records: list[ImageRecord]) -> None:
    paths = feature_paths(output_dir, prefix)
    for name in ("cls", "gap", "top"):
        np.save(paths[name], features[name])
    np.save(paths["labels"], np.array([record.label for record in records]))
    paths["paths"].write_text("\n".join(str(record.path) for record in records), encoding="utf-8")


def feature_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    stem = "" if prefix == "train" else f"{prefix}_"
    return {
        "cls": output_dir / f"{stem}features_cls.npy",
        "gap": output_dir / f"{stem}features_gap.npy",
        "top": output_dir / f"{stem}features_top.npy",
        "labels": output_dir / f"{stem}labels.npy",
        "paths": output_dir / f"{stem}paths.txt",
    }


def save_graphs(output_dir: Path, graphs: dict[str, np.ndarray]) -> None:
    np.save(output_dir / "class_knn_indices.npy", graphs["class_indices"])
    np.save(output_dir / "class_knn_weights.npy", graphs["class_weights"])
    np.save(output_dir / "global_knn_indices.npy", graphs["global_indices"])
    np.save(output_dir / "global_knn_weights.npy", graphs["global_weights"])


def write_bad_images(
    output_dir: Path,
    train_bad: list[dict[str, str]],
    eval_bad: list[dict[str, str]],
    train_feature_failures: list[dict[str, str]],
    eval_feature_failures: list[dict[str, str]],
) -> None:
    rows = []
    for split, items in [("train", train_bad), ("eval", eval_bad)]:
        for item in items:
            rows.append({"stage": "verify", "split": split, **item})
    for split, items in [("train", train_feature_failures), ("eval", eval_feature_failures)]:
        for item in items:
            rows.append({"stage": "feature", "split": split, **item})
    write_csv(output_dir / "bad_images.csv", rows, ["stage", "split", "index", "path", "label", "reason"])


def write_clean_thresholds(path: Path, labels: np.ndarray, metrics: dict[str, np.ndarray], split_info: dict[str, np.ndarray]) -> None:
    rows = []
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        rows.append(
            {
                "web_label": label,
                "count": len(idx),
                "clean_count": int(np.sum(split_info["state"][idx] == "clean")),
                "S_clean_mean": float(metrics["S_clean"][idx].mean()),
                "S_clean_std": float(metrics["S_clean"][idx].std()),
                "otsu_threshold": float(split_info["otsu_threshold"][idx][0]),
                "clean_ratio_before_clip": float(split_info["clean_ratio_before_clip"][idx][0]),
                "clean_ratio_after_clip": float(split_info["clean_ratio_after_clip"][idx][0]),
                "clip_low": int(np.any(split_info["clip_low"][idx])),
                "clip_high": int(np.any(split_info["clip_high"][idx])),
            }
        )
    write_csv(
        path,
        rows,
        ["web_label", "count", "clean_count", "S_clean_mean", "S_clean_std", "otsu_threshold", "clean_ratio_before_clip", "clean_ratio_after_clip", "clip_low", "clip_high"],
    )


def write_q_same_top_bottom(path: Path, metrics: dict[str, np.ndarray]) -> None:
    summary = top_bottom_summary(metrics)
    rows = [{"metric": key, "value": value} for key, value in summary.items()]
    write_csv(path, rows, ["metric", "value"])


def write_s_clean_distribution(path: Path, labels: np.ndarray, metrics: dict[str, np.ndarray], split_info: dict[str, np.ndarray]) -> None:
    rows = []
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        rows.append(
            {
                "web_label": label,
                "S_clean_mean": float(metrics["S_clean"][idx].mean()),
                "S_clean_std": float(metrics["S_clean"][idx].std()),
                "S_clean_min": float(metrics["S_clean"][idx].min()),
                "S_clean_max": float(metrics["S_clean"][idx].max()),
                "otsu_threshold": float(split_info["otsu_threshold"][idx][0]),
                "clean_ratio_before_clip": float(split_info["clean_ratio_before_clip"][idx][0]),
                "clean_ratio_after_clip": float(split_info["clean_ratio_after_clip"][idx][0]),
            }
        )
    write_csv(
        path,
        rows,
        ["web_label", "S_clean_mean", "S_clean_std", "S_clean_min", "S_clean_max", "otsu_threshold", "clean_ratio_before_clip", "clean_ratio_after_clip"],
    )


def write_neighbor_sanity(path: Path, records: list[ImageRecord], metrics: dict[str, np.ndarray], graphs: dict[str, np.ndarray]) -> None:
    rows = []
    labels = np.array([record.label for record in records])
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        order = idx[np.argsort(metrics["S_clean"][idx], kind="mergesort")]
        selected = [("bottom", i) for i in order[:5]] + [("top", i) for i in order[-5:]]
        for group, i in selected:
            neighbors = [j for j in graphs["class_indices"][i, :5].tolist() if j >= 0]
            rows.append(
                {
                    "rank_group": group,
                    "index": records[i].index,
                    "path": str(records[i].path),
                    "web_label": records[i].label,
                    "S_clean": float(metrics["S_clean"][i]),
                    "Q_same": float(metrics["Q_same"][i]),
                    "class_neighbors": "|".join(str(records[j].path) for j in neighbors),
                }
            )
    write_csv(path, rows, ["rank_group", "index", "path", "web_label", "S_clean", "Q_same", "class_neighbors"])


def write_baseline_splits(output_dir: Path, records: list[ImageRecord], masks: dict[str, np.ndarray]) -> None:
    for method, mask in masks.items():
        filename = method.lower().replace(" ", "_").replace("-", "_") + "_split.csv"
        rows = [
            {"index": record.index, "path": str(record.path), "web_label": record.label, "state": "clean" if mask[i] else "ignored"}
            for i, record in enumerate(records)
        ]
        write_csv(output_dir / filename, rows, ["index", "path", "web_label", "state"])


def write_linear_all_scores(path: Path, records: list[ImageRecord], confidence: np.ndarray, loss: np.ndarray) -> None:
    rows = [
        {
            "index": record.index,
            "path": str(record.path),
            "web_label": record.label,
            "confidence": float(confidence[i]),
            "loss": float(loss[i]),
        }
        for i, record in enumerate(records)
    ]
    write_csv(path, rows, ["index", "path", "web_label", "confidence", "loss"])


def write_centroid_scores(path: Path, records: list[ImageRecord], scores: np.ndarray) -> None:
    rows = [
        {
            "index": record.index,
            "path": str(record.path),
            "web_label": record.label,
            "centroid_score": float(scores[i]),
        }
        for i, record in enumerate(records)
    ]
    write_csv(path, rows, ["index", "path", "web_label", "centroid_score"])


def build_v1_summary(
    cfg: dict,
    train_records: list[ImageRecord],
    eval_records: list[ImageRecord],
    train_bad: list[dict[str, str]],
    eval_bad: list[dict[str, str]],
    train_feature_failures: list[dict[str, str]],
    eval_feature_failures: list[dict[str, str]],
    train_features: dict[str, np.ndarray],
    eval_features: dict[str, np.ndarray],
    graphs: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    split_info: dict[str, np.ndarray],
    method_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = [record.label for record in train_records]
    clean_mask = split_info["state"] == "clean"
    method_map = {row["method"]: row for row in method_summaries}
    gcdd = method_map.get("Full GCDD-clean", {})
    linear_all = method_map.get("DINOv2 Linear all", {})
    return {
        "dataset": cfg["dataset"]["name"],
        "feature_backend": cfg["feature"]["backend"],
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "bad_images_skipped": len(train_bad) + len(eval_bad),
        "feature_failures_skipped": len(train_feature_failures) + len(eval_feature_failures),
        "classes": len(set(labels)),
        "clean_samples": int(clean_mask.sum()),
        "ignored_samples": int((~clean_mask).sum()),
        "clip_hit_rate_low": class_clip_rate(split_info["clip_low"], labels),
        "clip_hit_rate_high": class_clip_rate(split_info["clip_high"], labels),
        "train_feature_shapes": {name: list(value.shape) for name, value in train_features.items()},
        "eval_feature_shapes": {name: list(value.shape) for name, value in eval_features.items()},
        "class_knn_shape": list(graphs["class_indices"].shape),
        "global_knn_shape": list(graphs["global_indices"].shape),
        "s_clean_mean": float(metrics["S_clean"].mean()),
        "s_clean_std": float(metrics["S_clean"].std()),
        "top_bottom_20_percent": top_bottom_summary(metrics),
        "methods": method_summaries,
        "passes_minimum": passes_v1_minimum(method_map),
        "gcdd_minus_linear_all_best_top1": float(gcdd.get("best_top1", 0.0) - linear_all.get("best_top1", 0.0)) if gcdd and linear_all else None,
    }


def passes_v1_minimum(method_map: dict[str, dict[str, Any]]) -> bool:
    required = ["Full GCDD-clean", "Confidence filtering", "Loss filtering", "Centroid filtering", "DINOv2 Linear all"]
    if any(name not in method_map for name in required):
        return False
    gcdd = float(method_map["Full GCDD-clean"]["best_top1"])
    return (
        gcdd > float(method_map["Confidence filtering"]["best_top1"])
        and gcdd > float(method_map["Loss filtering"]["best_top1"])
        and gcdd > float(method_map["Centroid filtering"]["best_top1"])
        and gcdd >= float(method_map["DINOv2 Linear all"]["best_top1"])
    )


def write_run_summary(path: Path, summary: dict[str, Any]) -> None:
    top_bottom = summary["top_bottom_20_percent"]
    lines = [
        "# V1 Web-Bird Summary",
        "",
        f"- Dataset: {summary['dataset']}",
        f"- Feature backend: {summary['feature_backend']}",
        f"- Train / eval samples: {summary['train_samples']} / {summary['eval_samples']}",
        f"- Bad images skipped: {summary['bad_images_skipped']}",
        f"- Feature failures skipped: {summary['feature_failures_skipped']}",
        f"- Classes: {summary['classes']}",
        f"- Clean / ignored: {summary['clean_samples']} / {summary['ignored_samples']}",
        f"- clip_hit_rate_low: {summary['clip_hit_rate_low']:.4f}",
        f"- clip_hit_rate_high: {summary['clip_hit_rate_high']:.4f}",
        f"- Train feature shapes: {summary['train_feature_shapes']}",
        f"- Eval feature shapes: {summary['eval_feature_shapes']}",
        f"- Class KNN shape: {summary['class_knn_shape']}",
        f"- Global KNN shape: {summary['global_knn_shape']}",
        f"- S_clean mean/std: {summary['s_clean_mean']:.4f} / {summary['s_clean_std']:.4f}",
        f"- Q_same top20/bottom20 mean: {top_bottom['Q_same_top20_mean']:.4f} / {top_bottom['Q_same_bottom20_mean']:.4f}",
        f"- R_class top20/bottom20 mean: {top_bottom['R_class_top20_mean']:.4f} / {top_bottom['R_class_bottom20_mean']:.4f}",
        f"- I_class_norm top20/bottom20 mean: {top_bottom['I_class_norm_top20_mean']:.4f} / {top_bottom['I_class_norm_bottom20_mean']:.4f}",
        f"- V1 minimum pass: {summary['passes_minimum']}",
        f"- Full GCDD best_top1 minus Linear all: {summary['gcdd_minus_linear_all_best_top1']}",
        "",
        "## Method Results",
    ]
    for row in summary["methods"]:
        lines.append(
            f"- {row['method']}: best_top1={row['best_top1']:.4f}, final_top1={row['final_top1']:.4f}, "
            f"last10={row['last10_mean']:.4f}±{row['last10_std']:.4f}, train_samples={row['train_samples']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def reindex_records(records: list[ImageRecord]) -> list[ImageRecord]:
    return [ImageRecord(i, record.path, record.label, record.split) for i, record in enumerate(records)]
