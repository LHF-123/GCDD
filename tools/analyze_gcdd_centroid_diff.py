from __future__ import annotations

import argparse
import csv
import html
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.baselines import centroid_scores
from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json


GROUP_ORDER = ["both", "gcdd_only", "centroid_only", "neither"]
METRIC_FIELDS = ["S_clean", "D_class", "R_class", "I_class_norm", "Q_same", "centroid_score", "confidence", "loss"]


class AssetContext:
    def __init__(self, asset_dir: Path, copy_assets: bool, path_maps: list[tuple[str, str]] | None = None):
        self.asset_dir = asset_dir
        self.copy_assets = copy_assets
        self.path_maps = path_maps or []
        self.cache: dict[str, Path] = {}
        ensure_dir(asset_dir)

    def html_src(self, image_path: str, html_file: Path, index: int | None = None) -> str:
        source = Path(image_path)
        if not self.copy_assets:
            return source.as_uri() if source.is_absolute() else image_path
        asset = self.copy_image(source, index)
        if asset is None:
            return source.as_uri() if source.is_absolute() else image_path
        return os.path.relpath(asset, html_file.parent).replace("\\", "/")

    def asset_path(self, image_path: str, index: int | None = None) -> str:
        source = Path(image_path)
        if not self.copy_assets:
            return ""
        asset = self.copy_image(source, index)
        return str(asset) if asset is not None else ""

    def copy_image(self, source: Path, index: int | None = None) -> Path | None:
        resolved_source = self.resolve_source(source)
        if resolved_source is None:
            return None
        key = str(resolved_source)
        if key in self.cache:
            return self.cache[key]
        prefix = f"{index:06d}_" if index is not None else ""
        target = self.asset_dir / f"{prefix}{safe_name(resolved_source.name)}"
        suffix_count = 1
        while target.exists() and not same_file(target, resolved_source):
            target = self.asset_dir / f"{prefix}{safe_name(resolved_source.stem)}_{suffix_count}{resolved_source.suffix}"
            suffix_count += 1
        if not target.exists():
            shutil.copy2(resolved_source, target)
        self.cache[key] = target
        return target

    def resolve_source(self, source: Path) -> Path | None:
        if source.exists():
            return source
        raw = str(source).replace("\\", "/")
        for old, new in self.path_maps:
            old_norm = old.replace("\\", "/").rstrip("/")
            if raw == old_norm or raw.startswith(old_norm + "/"):
                suffix = raw[len(old_norm) :].lstrip("/")
                mapped = Path(new).expanduser() / Path(*suffix.split("/"))
                if mapped.exists():
                    return mapped
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GCDD-clean vs centroid-clean split differences.")
    parser.add_argument("--input-dir", default="outputs/Web-Bird/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--output-dir", help="Analysis output directory.")
    parser.add_argument("--dataset", default="Web-Bird", help="Dataset name written to summary tables.")
    parser.add_argument("--neighbor-samples", type=int, default=20, help="Base sample count for neighbor HTML pages.")
    parser.add_argument("--no-figures", action="store_true", help="Skip matplotlib distribution figures.")
    parser.add_argument("--copy-assets", action="store_true", default=True, help="Copy visualization images into figures/assets and use relative HTML paths.")
    parser.add_argument("--no-copy-assets", action="store_false", dest="copy_assets", help="Do not copy images; HTML uses original file paths.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map original image root to local root before copying assets.")
    return parser.parse_args()


def parse_path_maps(items: list[str]) -> list[tuple[str, str]]:
    maps = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must use OLD=NEW format, got: {item}")
        old, new = item.split("=", 1)
        maps.append((old, new))
    return maps


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "gcdd_centroid_analysis"
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures" / "distributions")
    ensure_dir(output_dir / "figures" / "neighbors")
    ensure_dir(output_dir / "figures" / "assets")

    data = load_analysis_data(input_dir)
    groups = assign_groups(data["gcdd_clean"], data["centroid_clean"])
    data["group"] = groups

    write_overlap_summary(output_dir / "overlap_summary.csv", args.dataset, data, groups)
    per_class_rows = write_per_class_outputs(output_dir, data, groups)
    group_rows = write_group_metric_summary(output_dir / "group_metric_summary.csv", data, groups)
    write_hard_clean_distribution(output_dir / "hard_clean_group_distribution.csv", data, groups)
    asset_context = AssetContext(output_dir / "figures" / "assets", copy_assets=args.copy_assets, path_maps=parse_path_maps(args.path_map))
    write_visualization_index(output_dir / "visualization_index.csv", data, groups, args.neighbor_samples, asset_context)
    write_neighbor_html(output_dir / "figures" / "neighbors", data, groups, args.neighbor_samples, asset_context)
    write_class_visualizations(output_dir / "figures" / "classes", output_dir / "class_visualization_index.csv", data, groups, per_class_rows, asset_context)
    if not args.no_figures:
        write_distribution_figures(output_dir / "figures" / "distributions", data, groups)
    write_summary_md(output_dir / "analysis_summary.md", args.dataset, data, groups, per_class_rows, group_rows)
    write_json(output_dir / "analysis_metadata.json", {"input_dir": str(input_dir), "output_dir": str(output_dir), "has_confidence_loss": data["has_confidence_loss"]})
    print(f"Analysis written to {output_dir}", flush=True)


def load_analysis_data(input_dir: Path) -> dict[str, Any]:
    required = [
        "full_gcdd_clean_split.csv",
        "centroid_filtering_split.csv",
        "gcdd_scores.csv",
        "features_cls.npy",
        "labels.npy",
        "paths.txt",
        "class_knn_indices.npy",
        "global_knn_indices.npy",
    ]
    for name in required:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required V1 output is missing: {path}")

    score_rows = read_csv(input_dir / "gcdd_scores.csv")
    score_rows = sorted(score_rows, key=lambda row: int(row["index"]))
    labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    features = np.load(input_dir / "features_cls.npy")
    if len(score_rows) != len(labels) or len(paths) != len(labels):
        raise ValueError("gcdd_scores.csv, labels.npy, and paths.txt have inconsistent lengths.")

    metrics = {
        "index": np.array([int(row["index"]) for row in score_rows], dtype=np.int64),
        "path": np.array([row["path"] for row in score_rows], dtype=object),
        "web_label": np.array([row["web_label"] for row in score_rows], dtype=object),
        "D_class": parse_float_array(score_rows, "D_class"),
        "R_class": parse_float_array(score_rows, "R_class"),
        "I_class_norm": parse_float_array(score_rows, "I_class_norm"),
        "Q_same": parse_float_array(score_rows, "Q_same"),
        "S_clean": parse_float_array(score_rows, "S_clean"),
        "otsu_threshold": parse_float_array(score_rows, "otsu_threshold"),
    }
    metrics["centroid_score"] = load_or_compute_centroid_score(input_dir, features, labels, paths)
    confidence, loss, has_confidence_loss = load_optional_confidence_loss(input_dir, len(labels))
    metrics["confidence"] = confidence
    metrics["loss"] = loss
    metrics["has_confidence_loss"] = has_confidence_loss
    metrics["gcdd_clean"] = read_clean_mask(input_dir / "full_gcdd_clean_split.csv", len(labels))
    metrics["centroid_clean"] = read_clean_mask(input_dir / "centroid_filtering_split.csv", len(labels))
    metrics["class_knn"] = np.load(input_dir / "class_knn_indices.npy")
    metrics["global_knn"] = np.load(input_dir / "global_knn_indices.npy")
    metrics["labels"] = labels
    metrics["paths"] = paths
    return metrics


def parse_float_array(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([float(row[field]) for row in rows], dtype=np.float32)


def load_or_compute_centroid_score(input_dir: Path, features: np.ndarray, labels: np.ndarray, paths: list[str]) -> np.ndarray:
    path = input_dir / "centroid_scores.csv"
    if path.exists():
        rows = sorted(read_csv(path), key=lambda row: int(row["index"]))
        return parse_float_array(rows, "centroid_score")
    scores = centroid_scores(features, labels)
    rows = [{"index": i, "path": paths[i], "web_label": labels[i], "centroid_score": float(scores[i])} for i in range(len(scores))]
    write_csv(path, rows, ["index", "path", "web_label", "centroid_score"])
    return scores


def load_optional_confidence_loss(input_dir: Path, n: int) -> tuple[np.ndarray, np.ndarray, bool]:
    path = input_dir / "linear_all_train_scores.csv"
    if not path.exists():
        return np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32), False
    rows = sorted(read_csv(path), key=lambda row: int(row["index"]))
    if len(rows) != n:
        raise ValueError(f"{path} length does not match labels.npy.")
    return parse_float_array(rows, "confidence"), parse_float_array(rows, "loss"), True


def read_clean_mask(path: Path, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    rows = read_csv(path)
    for row in rows:
        idx = int(row["index"])
        if idx >= n:
            raise ValueError(f"Index {idx} in {path} exceeds expected length {n}.")
        mask[idx] = row["state"] == "clean"
    return mask


def assign_groups(gcdd_clean: np.ndarray, centroid_clean: np.ndarray) -> np.ndarray:
    groups = np.array(["neither"] * len(gcdd_clean), dtype=object)
    groups[gcdd_clean & centroid_clean] = "both"
    groups[gcdd_clean & ~centroid_clean] = "gcdd_only"
    groups[~gcdd_clean & centroid_clean] = "centroid_only"
    return groups


def write_overlap_summary(path: Path, dataset: str, data: dict[str, Any], groups: np.ndarray) -> None:
    num_gcdd = int(data["gcdd_clean"].sum())
    num_centroid = int(data["centroid_clean"].sum())
    num_overlap = int(np.sum(groups == "both"))
    num_gcdd_only = int(np.sum(groups == "gcdd_only"))
    num_centroid_only = int(np.sum(groups == "centroid_only"))
    num_neither = int(np.sum(groups == "neither"))
    union = num_overlap + num_gcdd_only + num_centroid_only
    row = {
        "dataset": dataset,
        "num_gcdd_clean": num_gcdd,
        "num_centroid_clean": num_centroid,
        "num_overlap": num_overlap,
        "num_gcdd_only": num_gcdd_only,
        "num_centroid_only": num_centroid_only,
        "num_neither": num_neither,
        "jaccard": safe_ratio(num_overlap, union),
        "gcdd_only_ratio": safe_ratio(num_gcdd_only, len(groups)),
        "centroid_only_ratio": safe_ratio(num_centroid_only, len(groups)),
    }
    write_csv(path, [row], list(row.keys()))


def write_per_class_outputs(output_dir: Path, data: dict[str, Any], groups: np.ndarray) -> list[dict[str, Any]]:
    labels = data["labels"]
    rows = []
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        both = int(np.sum(groups[idx] == "both"))
        gcdd_only = int(np.sum(groups[idx] == "gcdd_only"))
        centroid_only = int(np.sum(groups[idx] == "centroid_only"))
        neither = int(np.sum(groups[idx] == "neither"))
        gcdd_clean = both + gcdd_only
        centroid_clean = both + centroid_only
        union = both + gcdd_only + centroid_only
        row = {
            "class_id": label,
            "num_total": len(idx),
            "num_gcdd_clean": gcdd_clean,
            "num_centroid_clean": centroid_clean,
            "num_overlap": both,
            "num_gcdd_only": gcdd_only,
            "num_centroid_only": centroid_only,
            "num_neither": neither,
            "jaccard": safe_ratio(both, union),
            "gcdd_clean_ratio": safe_ratio(gcdd_clean, len(idx)),
            "centroid_clean_ratio": safe_ratio(centroid_clean, len(idx)),
            "gcdd_only_ratio": safe_ratio(gcdd_only, len(idx)),
            "centroid_only_ratio": safe_ratio(centroid_only, len(idx)),
            "mean_S_clean": float(np.mean(data["S_clean"][idx])),
            "mean_centroid_score": float(np.mean(data["centroid_score"][idx])),
            "otsu_threshold": float(data["otsu_threshold"][idx][0]),
        }
        rows.append(row)
    fields = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "per_class_overlap.csv", rows, fields)
    write_csv(output_dir / "per_class_overlap_top_low_jaccard.csv", sorted(rows, key=lambda r: r["jaccard"])[:20], fields)
    write_csv(output_dir / "per_class_gcdd_only_top_classes.csv", sorted(rows, key=lambda r: r["gcdd_only_ratio"], reverse=True)[:20], fields)
    write_csv(output_dir / "per_class_centroid_only_top_classes.csv", sorted(rows, key=lambda r: r["centroid_only_ratio"], reverse=True)[:20], fields)
    return rows


def write_group_metric_summary(path: Path, data: dict[str, Any], groups: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    fields = ["group", "count"]
    for metric in METRIC_FIELDS:
        fields.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_std", f"{metric}_p25", f"{metric}_p75"])
    for group in GROUP_ORDER:
        idx = groups == group
        row: dict[str, Any] = {"group": group, "count": int(np.sum(idx))}
        for metric in METRIC_FIELDS:
            values = data[metric][idx]
            row.update(metric_stats(metric, values))
        rows.append(row)
    write_csv(path, rows, fields)
    return rows


def metric_stats(name: str, values: np.ndarray) -> dict[str, float]:
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return {f"{name}_{key}": np.nan for key in ["mean", "median", "std", "p25", "p75"]}
    return {
        f"{name}_mean": float(np.mean(valid)),
        f"{name}_median": float(np.median(valid)),
        f"{name}_std": float(np.std(valid)),
        f"{name}_p25": float(np.percentile(valid, 25)),
        f"{name}_p75": float(np.percentile(valid, 75)),
    }


def write_hard_clean_distribution(path: Path, data: dict[str, Any], groups: np.ndarray) -> None:
    if not data["has_confidence_loss"]:
        rows = [{"group": group, "count": "", "ratio": "", "note": "linear_all_train_scores.csv not found; loss-based hard-clean analysis skipped"} for group in GROUP_ORDER]
        write_csv(path, rows, ["group", "count", "ratio", "note"])
        return
    loss_threshold = np.nanpercentile(data["loss"], 70)
    s_threshold = np.nanpercentile(data["S_clean"], 70)
    hard = (data["loss"] >= loss_threshold) & (data["S_clean"] >= s_threshold)
    total = int(np.sum(hard))
    rows = []
    for group in GROUP_ORDER:
        count = int(np.sum(hard & (groups == group)))
        rows.append({"group": group, "count": count, "ratio": safe_ratio(count, total), "note": ""})
    write_csv(path, rows, ["group", "count", "ratio", "note"])


def write_visualization_index(path: Path, data: dict[str, Any], groups: np.ndarray, base_count: int, asset_context: AssetContext) -> None:
    rows = visualization_rows(data, groups, base_count)
    for row in rows:
        row["asset_path"] = asset_context.asset_path(str(row["path"]), int(row["index"]))
    write_csv(
        path,
        rows,
        ["index", "path", "asset_path", "web_label", "group", "selection_rule", "S_clean", "centroid_score", "D_class", "R_class", "I_class_norm", "Q_same", "loss", "confidence"],
    )


def visualization_rows(data: dict[str, Any], groups: np.ndarray, base_count: int) -> list[dict[str, Any]]:
    selected: list[tuple[int, str]] = []
    selected.extend(select_by_rule(data, groups, "both", min(base_count, 20), "both_high_S_clean_centroid", ["S_clean", "centroid_score"]))
    selected.extend(select_by_rule(data, groups, "gcdd_only", min(base_count * 2, 40), "gcdd_only_high_S_clean", ["S_clean"]))
    selected.extend(select_by_rule(data, groups, "gcdd_only", min(base_count * 2, 40), "gcdd_only_low_centroid_high_graph", ["R_class", "I_class_norm", "Q_same", "-centroid_score"]))
    selected.extend(select_by_rule(data, groups, "centroid_only", min(base_count * 2, 40), "centroid_only_high_centroid", ["centroid_score"]))
    selected.extend(select_by_rule(data, groups, "centroid_only", min(base_count * 2, 40), "centroid_only_low_Q_same", ["-Q_same", "centroid_score"]))
    selected.extend(select_by_rule(data, groups, "centroid_only", min(base_count * 2, 40), "centroid_only_low_R_class", ["-R_class", "centroid_score"]))
    selected.extend(select_by_rule(data, groups, "neither", min(base_count, 20), "neither_low_S_clean_centroid", ["-S_clean", "-centroid_score"]))

    seen = set()
    rows = []
    for idx, rule in selected:
        key = (idx, rule)
        if key in seen:
            continue
        seen.add(key)
        rows.append(sample_row(data, groups, idx, rule))
    return rows


def select_by_rule(data: dict[str, Any], groups: np.ndarray, group: str, count: int, rule: str, sort_fields: list[str]) -> list[tuple[int, str]]:
    idx = np.where(groups == group)[0]
    if len(idx) == 0 or count <= 0:
        return []
    scores = np.zeros(len(idx), dtype=np.float64)
    for field in sort_fields:
        sign = -1.0 if field.startswith("-") else 1.0
        name = field[1:] if field.startswith("-") else field
        values = data[name][idx].astype(float)
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            continue
        low, high = float(np.min(valid)), float(np.max(valid))
        scaled = np.zeros_like(values, dtype=np.float64) if high == low else (values - low) / (high - low)
        scores += sign * scaled
    order = np.argsort(-scores, kind="mergesort")[: min(count, len(idx))]
    return [(int(idx[i]), rule) for i in order]


def sample_row(data: dict[str, Any], groups: np.ndarray, idx: int, rule: str) -> dict[str, Any]:
    return {
        "index": int(data["index"][idx]),
        "path": data["path"][idx],
        "web_label": data["web_label"][idx],
        "group": groups[idx],
        "selection_rule": rule,
        "S_clean": float(data["S_clean"][idx]),
        "centroid_score": float(data["centroid_score"][idx]),
        "D_class": float(data["D_class"][idx]),
        "R_class": float(data["R_class"][idx]),
        "I_class_norm": float(data["I_class_norm"][idx]),
        "Q_same": float(data["Q_same"][idx]),
        "loss": nan_to_empty(data["loss"][idx]),
        "confidence": nan_to_empty(data["confidence"][idx]),
    }


def write_neighbor_html(output_dir: Path, data: dict[str, Any], groups: np.ndarray, base_count: int, asset_context: AssetContext) -> None:
    rows = visualization_rows(data, groups, base_count)
    index_rows = []
    for row in rows:
        idx = int(row["index"])
        filename = f"{idx:06d}_{row['group']}_{row['selection_rule']}.html"
        write_sample_html(output_dir / filename, data, groups, idx, row["selection_rule"], asset_context)
        index_rows.append({"index": idx, "group": row["group"], "selection_rule": row["selection_rule"], "html": filename})
    write_csv(output_dir / "neighbor_html_index.csv", index_rows, ["index", "group", "selection_rule", "html"])


def write_class_visualizations(output_dir: Path, index_path: Path, data: dict[str, Any], groups: np.ndarray, per_class_rows: list[dict[str, Any]], asset_context: AssetContext) -> None:
    ensure_dir(output_dir)
    selected_classes = select_classes_for_visualization(per_class_rows)
    index_rows = []
    for class_id, reason in selected_classes:
        class_dir = output_dir / safe_name(class_id)
        ensure_dir(class_dir)
        for group in ["both", "gcdd_only", "centroid_only"]:
            filename = f"{group}_samples.html"
            write_class_group_html(class_dir / filename, data, groups, class_id, group, reason, asset_context)
            index_rows.append({"class_id": class_id, "selection_reason": reason, "group": group, "html": str(class_dir / filename)})
    write_csv(index_path, index_rows, ["class_id", "selection_reason", "group", "html"])


def select_classes_for_visualization(per_class_rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    selected: dict[str, str] = {}
    for row in sorted(per_class_rows, key=lambda r: r["jaccard"])[:5]:
        selected.setdefault(row["class_id"], "lowest_jaccard")
    for row in sorted(per_class_rows, key=lambda r: r["gcdd_only_ratio"], reverse=True)[:5]:
        selected.setdefault(row["class_id"], "highest_gcdd_only_ratio")
    for row in sorted(per_class_rows, key=lambda r: r["centroid_only_ratio"], reverse=True)[:5]:
        selected.setdefault(row["class_id"], "highest_centroid_only_ratio")
    return sorted(selected.items())


def write_class_group_html(path: Path, data: dict[str, Any], groups: np.ndarray, class_id: str, group: str, reason: str, asset_context: AssetContext) -> None:
    class_idx = np.where(data["labels"] == class_id)[0]
    idx = class_idx[groups[class_idx] == group]
    if group == "centroid_only":
        order = np.argsort(-data["centroid_score"][idx], kind="mergesort")
    else:
        order = np.argsort(-data["S_clean"][idx], kind="mergesort")
    idx = idx[order[:24]]
    body = [
        "<html><body>",
        f"<h1>{html.escape(class_id)} - {html.escape(group)}</h1>",
        f"<p>selection_reason: {html.escape(reason)}</p>",
        "<div style='display:flex;gap:14px;flex-wrap:wrap'>",
    ]
    for i in idx:
        src = asset_context.html_src(str(data["path"][i]), path, int(data["index"][i]))
        body.append(
            "<div style='width:190px'>"
            f"<img src=\"{html.escape(src)}\" style=\"max-width:170px\"><br>"
            f"idx={int(data['index'][i])}<br>"
            f"S={data['S_clean'][i]:.3f}, C={data['centroid_score'][i]:.3f}<br>"
            f"R={data['R_class'][i]:.3f}, I={data['I_class_norm'][i]:.3f}, Q={data['Q_same'][i]:.3f}"
            "</div>"
        )
    body.extend(["</div>", "</body></html>"])
    path.write_text("\n".join(body), encoding="utf-8")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def write_sample_html(path: Path, data: dict[str, Any], groups: np.ndarray, idx: int, rule: str, asset_context: AssetContext) -> None:
    class_neighbors = [j for j in data["class_knn"][idx, :5].tolist() if j >= 0]
    global_neighbors = [j for j in data["global_knn"][idx, :5].tolist() if j >= 0]
    body = [
        "<html><body>",
        f"<h1>{idx} - {html.escape(str(data['web_label'][idx]))}</h1>",
        "<ul>",
        f"<li>group: {groups[idx]}</li>",
        f"<li>selection_rule: {html.escape(rule)}</li>",
        f"<li>S_clean: {data['S_clean'][idx]:.4f}</li>",
        f"<li>centroid_score: {data['centroid_score'][idx]:.4f}</li>",
        f"<li>D/R/I/Q: {data['D_class'][idx]:.4f} / {data['R_class'][idx]:.4f} / {data['I_class_norm'][idx]:.4f} / {data['Q_same'][idx]:.4f}</li>",
        "</ul>",
        image_block("Query", data["path"][idx], path, asset_context, int(data["index"][idx])),
        neighbor_block("Top-5 Class Neighbors", class_neighbors, data, path, asset_context),
        neighbor_block("Top-5 Global Neighbors", global_neighbors, data, path, asset_context),
        "</body></html>",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def image_block(title: str, image_path: str, html_file: Path, asset_context: AssetContext, index: int) -> str:
    escaped = html.escape(image_path)
    src = html.escape(asset_context.html_src(image_path, html_file, index))
    return f"<h2>{html.escape(title)}</h2><div><img src=\"{src}\" style=\"max-width:240px\"><p>{escaped}</p></div>"


def neighbor_block(title: str, indices: list[int], data: dict[str, Any], html_file: Path, asset_context: AssetContext) -> str:
    parts = [f"<h2>{html.escape(title)}</h2><div style='display:flex;gap:12px;flex-wrap:wrap'>"]
    for j in indices:
        src = html.escape(asset_context.html_src(str(data["path"][j]), html_file, int(data["index"][j])))
        parts.append(
            "<div style='width:180px'>"
            f"<img src=\"{src}\" style=\"max-width:160px\"><br>"
            f"{j}<br>{html.escape(str(data['web_label'][j]))}<br>"
            f"S={data['S_clean'][j]:.3f}, Q={data['Q_same'][j]:.3f}"
            "</div>"
        )
    parts.append("</div>")
    return "\n".join(parts)


def write_distribution_figures(output_dir: Path, data: dict[str, Any], groups: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; distribution figures skipped.", flush=True)
        return
    for metric in ["S_clean", "centroid_score", "R_class", "I_class_norm", "Q_same", "D_class"]:
        plt.figure(figsize=(8, 5))
        for group in GROUP_ORDER:
            values = data[metric][groups == group]
            values = values[~np.isnan(values)]
            if len(values):
                plt.hist(values, bins=40, alpha=0.45, label=group, density=True)
        plt.title(f"{metric} by group")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_by_group.png", dpi=160)
        plt.close()
    plt.figure(figsize=(7, 6))
    for group in GROUP_ORDER:
        idx = groups == group
        plt.scatter(data["centroid_score"][idx], data["S_clean"][idx], s=6, alpha=0.35, label=group)
    plt.xlabel("centroid_score")
    plt.ylabel("S_clean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "S_clean_vs_centroid_score.png", dpi=160)
    plt.close()


def write_summary_md(
    path: Path,
    dataset: str,
    data: dict[str, Any],
    groups: np.ndarray,
    per_class_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> None:
    counts = {group: int(np.sum(groups == group)) for group in GROUP_ORDER}
    union = counts["both"] + counts["gcdd_only"] + counts["centroid_only"]
    lowest_j = sorted(per_class_rows, key=lambda r: r["jaccard"])[:5]
    top_g = sorted(per_class_rows, key=lambda r: r["gcdd_only_ratio"], reverse=True)[:5]
    top_c = sorted(per_class_rows, key=lambda r: r["centroid_only_ratio"], reverse=True)[:5]
    group_map = {row["group"]: row for row in group_rows}
    lines = [
        "# GCDD vs Centroid Analysis",
        "",
        "## 1. Overlap",
        f"- Dataset: {dataset}",
        f"- GCDD clean count: {int(data['gcdd_clean'].sum())}",
        f"- Centroid clean count: {int(data['centroid_clean'].sum())}",
        f"- Overlap: {counts['both']}",
        f"- GCDD only: {counts['gcdd_only']}",
        f"- Centroid only: {counts['centroid_only']}",
        f"- Neither: {counts['neither']}",
        f"- Jaccard: {safe_ratio(counts['both'], union):.4f}",
        "",
        "## 2. Group Metric Summary",
        "| group | count | S_clean mean | centroid mean | R mean | I mean | Q mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUP_ORDER:
        row = group_map[group]
        lines.append(
            f"| {group} | {row['count']} | {fmt(row['S_clean_mean'])} | {fmt(row['centroid_score_mean'])} | "
            f"{fmt(row['R_class_mean'])} | {fmt(row['I_class_norm_mean'])} | {fmt(row['Q_same_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 3. Key Observations",
            comparison_line(group_map, "gcdd_only", "centroid_only", "R_class_mean", "R_class"),
            comparison_line(group_map, "gcdd_only", "centroid_only", "I_class_norm_mean", "I_class_norm"),
            comparison_line(group_map, "gcdd_only", "centroid_only", "Q_same_mean", "Q_same"),
            comparison_line(group_map, "gcdd_only", "centroid_only", "centroid_score_mean", "centroid_score"),
            "",
            "## 4. Per-Class Differences",
            "- Lowest Jaccard classes: " + ", ".join(row["class_id"] for row in lowest_j),
            "- Highest GCDD-only ratio classes: " + ", ".join(row["class_id"] for row in top_g),
            "- Highest centroid-only ratio classes: " + ", ".join(row["class_id"] for row in top_c),
            "",
            "## 5. Interpretation",
            "- Use `group_metric_summary.csv` to judge whether GCDD-only has stronger graph connectivity than centroid-only.",
            "- Use `S_clean_vs_centroid_score.png` to inspect whether the methods select different score regions.",
            "- Loss/confidence analysis is skipped unless `linear_all_train_scores.csv` exists.",
            "",
            "## 6. Next Action",
            "- If GCDD-only has higher R/I/Q but lower centroid_score, GCDD is providing information beyond centroid.",
            "- If GCDD-only is weak on graph metrics or visually noisy, inspect clean score components before changing training.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def comparison_line(group_map: dict[str, dict[str, Any]], left: str, right: str, field: str, label: str) -> str:
    return f"- {label}: {left}={fmt(group_map[left][field])}, {right}={fmt(group_map[right][field])}"


def safe_ratio(num: int | float, denom: int | float) -> float:
    return float(num / denom) if denom else 0.0


def nan_to_empty(value: float) -> str | float:
    return "" if np.isnan(value) else float(value)


def fmt(value: Any) -> str:
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
