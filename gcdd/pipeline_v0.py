from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import ImageRecord, build_debug_index
from .features import extract_features
from .graph import build_rrf_graphs
from .io_utils import ensure_dir, write_csv, write_json, write_yaml
from .scoring import compute_scores
from .training import train_linear_smoke


def run_v0_smoke(cfg: dict) -> None:
    output_dir = make_output_dir(cfg)
    ensure_dir(output_dir)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    records, bad_images = build_debug_index(cfg)
    if not records:
        raise RuntimeError("No valid images found after bad-image filtering and class sampling.")

    write_index(output_dir / "debug_index.csv", records)
    write_csv(output_dir / "bad_images.csv", bad_images, ["index", "path", "label", "split", "reason"])

    features, kept_records, feature_failures = extract_features(records, cfg)
    if feature_failures:
        write_csv(output_dir / "feature_failures.csv", feature_failures, ["index", "path", "label", "reason"])
    records = reindex_records(kept_records)
    labels = np.array([record.label for record in records])
    save_debug_features(output_dir, features)

    graphs = build_rrf_graphs(features, labels, cfg)
    np.save(output_dir / "debug_class_knn_indices.npy", graphs["class_indices"])
    np.save(output_dir / "debug_global_knn_indices.npy", graphs["global_indices"])

    metrics, split_info = compute_scores(labels, graphs, cfg)
    state = split_info["state"]
    write_scores(output_dir / "debug_scores.csv", records, metrics, split_info)
    write_split(output_dir / "debug_split.csv", records, state)

    train_logs = train_linear_smoke(features["cls"], labels, state == "clean", cfg)
    write_csv(output_dir / "debug_train_log.csv", train_logs, ["epoch", "loss", "top1", "top5", "train_samples"])

    summary = build_summary(records, bad_images, feature_failures, features, graphs, metrics, split_info, train_logs, cfg)
    write_json(output_dir / "debug_summary.json", summary)
    write_run_summary(output_dir / "run_summary.md", summary)


def make_output_dir(cfg: dict) -> Path:
    return Path(cfg["output"]["root"]) / cfg["dataset"]["name"] / cfg["output"]["version"]


def write_index(path: Path, records: list[ImageRecord]) -> None:
    rows = [
        {"index": record.index, "path": str(record.path), "web_label": record.label, "split": record.split}
        for record in records
    ]
    write_csv(path, rows, ["index", "path", "web_label", "split"])


def save_debug_features(output_dir: Path, features: dict[str, np.ndarray]) -> None:
    # The concatenated file is the V0 smoke artifact; modality files make shape debugging easier.
    concat = np.concatenate([features["cls"], features["gap"], features["top"]], axis=1)
    np.save(output_dir / "debug_features.npy", concat)
    np.save(output_dir / "debug_features_cls.npy", features["cls"])
    np.save(output_dir / "debug_features_gap.npy", features["gap"])
    np.save(output_dir / "debug_features_top.npy", features["top"])


def write_scores(path: Path, records: list[ImageRecord], metrics: dict[str, np.ndarray], split_info: dict[str, np.ndarray]) -> None:
    rows = []
    for i, record in enumerate(records):
        rows.append(
            {
                "index": record.index,
                "path": str(record.path),
                "web_label": record.label,
                "D_class": metrics["D_class"][i],
                "R_class": metrics["R_class"][i],
                "I_class_norm": metrics["I_class_norm"][i],
                "Q_same": metrics["Q_same"][i],
                "S_clean": metrics["S_clean"][i],
                "otsu_threshold": split_info["otsu_threshold"][i],
                "clean_ratio_before_clip": split_info["clean_ratio_before_clip"][i],
                "clean_ratio_after_clip": split_info["clean_ratio_after_clip"][i],
                "state": split_info["state"][i],
            }
        )
    write_csv(
        path,
        rows,
        [
            "index",
            "path",
            "web_label",
            "D_class",
            "R_class",
            "I_class_norm",
            "Q_same",
            "S_clean",
            "otsu_threshold",
            "clean_ratio_before_clip",
            "clean_ratio_after_clip",
            "state",
        ],
    )


def write_split(path: Path, records: list[ImageRecord], state: np.ndarray) -> None:
    rows = [
        {"index": record.index, "path": str(record.path), "web_label": record.label, "state": state[i]}
        for i, record in enumerate(records)
    ]
    write_csv(path, rows, ["index", "path", "web_label", "state"])


def build_summary(
    records: list[ImageRecord],
    bad_images: list[dict[str, str]],
    feature_failures: list[dict[str, str]],
    features: dict[str, np.ndarray],
    graphs: dict[str, np.ndarray],
    metrics: dict[str, np.ndarray],
    split_info: dict[str, np.ndarray],
    train_logs: list[dict],
    cfg: dict,
) -> dict:
    labels = [record.label for record in records]
    clean_mask = split_info["state"] == "clean"
    class_count = {label: labels.count(label) for label in sorted(set(labels))}
    clip_low_rate = class_clip_rate(split_info["clip_low"], labels)
    clip_high_rate = class_clip_rate(split_info["clip_high"], labels)
    return {
        "dataset": cfg["dataset"]["name"],
        "feature_backend": cfg["feature"]["backend"],
        "knn_backend": cfg["graph"].get("knn_backend", "auto"),
        "num_valid_samples": len(records),
        "num_bad_images": len(bad_images),
        "num_feature_failures": len(feature_failures),
        "num_classes": len(class_count),
        "samples_per_class": class_count,
        "clean_samples": int(clean_mask.sum()),
        "ignored_samples": int((~clean_mask).sum()),
        "clip_hit_rate_low": clip_low_rate,
        "clip_hit_rate_high": clip_high_rate,
        "feature_shapes": {name: list(value.shape) for name, value in features.items()},
        "combined_feature_shape": [len(records), int(sum(value.shape[1] for value in features.values()))],
        "class_knn_shape": list(graphs["class_indices"].shape),
        "global_knn_shape": list(graphs["global_indices"].shape),
        "s_clean_mean": float(metrics["S_clean"].mean()),
        "s_clean_std": float(metrics["S_clean"].std()),
        "q_same_mean": float(metrics["Q_same"].mean()),
        "top_bottom_20_percent": top_bottom_summary(metrics),
        "last_train_epoch": train_logs[-1] if train_logs else {},
    }


def class_clip_rate(flags: np.ndarray, labels: list[str]) -> float:
    hit = 0
    unique = sorted(set(labels))
    for label in unique:
        idx = [i for i, item in enumerate(labels) if item == label]
        if idx and np.any(flags[idx]):
            hit += 1
    return float(hit / len(unique)) if unique else 0.0


def top_bottom_summary(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    scores = metrics["S_clean"]
    count = max(1, len(scores) // 5)
    order = np.argsort(scores)
    bottom = order[:count]
    top = order[-count:]
    fields = ["Q_same", "D_class", "R_class", "I_class_norm"]
    summary: dict[str, float] = {}
    for field in fields:
        summary[f"{field}_top20_mean"] = float(metrics[field][top].mean())
        summary[f"{field}_bottom20_mean"] = float(metrics[field][bottom].mean())
    return summary


def write_run_summary(path: Path, summary: dict) -> None:
    top_bottom = summary["top_bottom_20_percent"]
    lines = [
        "# V0 Smoke Test Summary",
        "",
        f"- Dataset: {summary['dataset']}",
        f"- Feature backend: {summary['feature_backend']}",
        f"- KNN backend: {summary['knn_backend']}",
        f"- Valid samples: {summary['num_valid_samples']}",
        f"- Bad images skipped: {summary['num_bad_images']}",
        f"- Feature failures skipped: {summary['num_feature_failures']}",
        f"- Classes: {summary['num_classes']}",
        f"- Clean / ignored: {summary['clean_samples']} / {summary['ignored_samples']}",
        f"- clip_hit_rate_low: {summary['clip_hit_rate_low']:.4f}",
        f"- clip_hit_rate_high: {summary['clip_hit_rate_high']:.4f}",
        f"- Feature shapes: {summary['feature_shapes']}",
        f"- Combined feature shape: {summary['combined_feature_shape']}",
        f"- Class KNN shape: {summary['class_knn_shape']}",
        f"- Global KNN shape: {summary['global_knn_shape']}",
        f"- S_clean mean/std: {summary['s_clean_mean']:.4f} / {summary['s_clean_std']:.4f}",
        f"- Q_same mean: {summary['q_same_mean']:.4f}",
        f"- Q_same top20/bottom20 mean: {top_bottom['Q_same_top20_mean']:.4f} / {top_bottom['Q_same_bottom20_mean']:.4f}",
        f"- R_class top20/bottom20 mean: {top_bottom['R_class_top20_mean']:.4f} / {top_bottom['R_class_bottom20_mean']:.4f}",
        f"- I_class_norm top20/bottom20 mean: {top_bottom['I_class_norm_top20_mean']:.4f} / {top_bottom['I_class_norm_bottom20_mean']:.4f}",
        f"- Last train epoch: {summary['last_train_epoch']}",
        "",
        "V0 passes when there is no index mismatch, no KNN shape error, no label/path mismatch, and 1 epoch training plus eval completes.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def reindex_records(records: list[ImageRecord]) -> list[ImageRecord]:
    return [ImageRecord(i, record.path, record.label, record.split) for i, record in enumerate(records)]
