from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    input_dir: Path
    noise_index: Path
    dynamic_source: Path
    pgdf_source: Path
    all_results: Path | None = None
    dynamic_retention_ratio: float | None = 0.8
    pgdf_retention_ratio: float | None = 0.8


@dataclass(frozen=True)
class GTSample:
    row_id: str
    path: str
    clean_label: str
    web_label: str
    is_clean: bool


@dataclass(frozen=True)
class SelectionMetrics:
    selected: int
    clean_selected: int
    purity: float
    clean_recall: float


DEFAULT_DATASETS = [
    DatasetSpec(
        name="CUB asym40",
        input_dir=Path("outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42"),
        noise_index=Path("outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv"),
        dynamic_source=Path("outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/dynamic_loss_42"),
        pgdf_source=Path("outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/proto_guided_dynamic_auto_42"),
        all_results=Path("outputs/CUB_200_2011/CUB-200-2011-asym40-s42/v1_cub_asym_r0p4_s42/lora_30_42/lora_results.csv"),
    ),
    DatasetSpec(
        name="Stanford Cars asym40",
        input_dir=Path("outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42"),
        noise_index=Path("outputs/Stanford_Cars/noise_indices/stanford_cars_asym_r0p4_s42_index.csv"),
        dynamic_source=Path("outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/dynamic_loss_42"),
        pgdf_source=Path("outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/proto_guided_dynamic_auto_42"),
        all_results=Path("outputs/Stanford_Cars/Stanford-Cars-asym40-s42/v1_cars_asym_r0p4_s42/lora_30_42/lora_results.csv"),
    ),
    DatasetSpec(
        name="FGVC-Aircraft asym40",
        input_dir=Path("outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42"),
        noise_index=Path("outputs/FGVC_Aircraft/noise_indices/fgvc_aircraft_asym_r0p4_s42_index.csv"),
        dynamic_source=Path("outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/dynamic_loss_42"),
        pgdf_source=Path("outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/proto_guided_dynamic_auto_42"),
        all_results=Path("outputs/FGVC_Aircraft/FGVC-Aircraft-asym40-s42/v1_aircraft_asym_r0p4_s42/lora42/lora_results.csv"),
    ),
]

DEFAULT_WEB_JACCARD_GAIN_ROWS = [
    {"dataset": "Web-Bird", "jaccard": 0.56, "pgdf_gain_pct": 0.52},
    {"dataset": "Web-Car", "jaccard": 0.81, "pgdf_gain_pct": -0.25},
    {"dataset": "Web-Aircraft", "jaccard": 0.82, "pgdf_gain_pct": -0.54},
]


PURITY_FIELDS = [
    "dataset",
    "method",
    "selection_source",
    "selected",
    "clean_selected",
    "base_clean_total",
    "base_clean_ratio_pct",
    "purity_pct",
    "clean_recall_pct",
    "top1_pct",
]

PROTO_FIELDS = [
    "dataset",
    "score_mode",
    "num_clean",
    "num_noisy",
    "clean_mean",
    "noisy_mean",
    "mean_gap",
    "clean_median",
    "noisy_median",
    "auc_clean_vs_noisy",
]

JACCARD_FIELDS = [
    "dataset",
    "jaccard",
    "dynamic_top1_pct",
    "pgdf_top1_pct",
    "pgdf_gain_pct",
    "dynamic_purity_pct",
    "pgdf_purity_pct",
    "purity_gain_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze GT purity and prototype-score separability for PGDF on asym40 datasets.")
    parser.add_argument("--out-dir", default="outputs/analysis/pgdf_gt_purity", help="Output directory.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=INPUT_DIR,NOISE_INDEX,DYNAMIC_SOURCE,PGDF_SOURCE[,ALL_RESULTS]",
        help="Override datasets. Paths may be selection CSVs or selection directories.",
    )
    parser.add_argument("--selection-policy", choices=["last"], default="last", help="How to pick a CSV when a selection source is a directory.")
    parser.add_argument("--leave-one-out", action="store_true", help="Use leave-one-out class prototypes for prototype score analysis.")
    parser.add_argument("--no-plots", action="store_true", help="Write CSV/JSON/Markdown only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_dataset_specs(args.dataset) if args.dataset else DEFAULT_DATASETS
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    purity_rows: list[dict[str, Any]] = []
    proto_rows: list[dict[str, Any]] = []
    jaccard_rows: list[dict[str, Any]] = []
    distributions: dict[str, dict[str, list[float]]] = {}

    for spec in specs:
        result = analyze_dataset(spec, selection_policy=args.selection_policy, leave_one_out=args.leave_one_out)
        purity_rows.extend(result["purity_rows"])
        proto_rows.append(result["prototype_row"])
        jaccard_rows.append(result["jaccard_row"])
        distributions[spec.name] = result["score_distribution"]
    jaccard_rows.extend(build_web_jaccard_gain_rows(DEFAULT_WEB_JACCARD_GAIN_ROWS))

    write_csv(out_dir / "gt_purity_summary.csv", purity_rows, PURITY_FIELDS)
    write_csv(out_dir / "prototype_score_summary.csv", proto_rows, PROTO_FIELDS)
    write_csv(out_dir / "jaccard_gain_summary.csv", jaccard_rows, JACCARD_FIELDS)
    write_json(
        out_dir / "pgdf_gt_purity_summary.json",
        {"gt_purity": purity_rows, "prototype_score": proto_rows, "jaccard_gain": jaccard_rows},
    )
    write_summary(out_dir / "analysis_summary.md", purity_rows, proto_rows, jaccard_rows)
    if not args.no_plots:
        write_plots(out_dir, purity_rows, distributions)
    print(f"PGDF GT purity analysis written to {out_dir}", flush=True)


def parse_dataset_specs(items: list[str]) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--dataset must use NAME=... format, got: {item}")
        name, rest = item.split("=", 1)
        parts = [Path(part.strip()) for part in rest.split(",")]
        if len(parts) not in {4, 5}:
            raise ValueError(
                "--dataset must use NAME=INPUT_DIR,NOISE_INDEX,DYNAMIC_SOURCE,PGDF_SOURCE[,ALL_RESULTS], "
                f"got {len(parts)} paths in: {item}"
            )
        all_results = parts[4] if len(parts) == 5 and str(parts[4]) else None
        specs.append(DatasetSpec(name=name.strip(), input_dir=parts[0], noise_index=parts[1], dynamic_source=parts[2], pgdf_source=parts[3], all_results=all_results))
    return specs


def analyze_dataset(spec: DatasetSpec, selection_policy: str = "last", leave_one_out: bool = False) -> dict[str, Any]:
    input_dir = resolve_repo_path(spec.input_dir)
    noise_index = resolve_repo_path(spec.noise_index)
    dynamic_selection = resolve_selection_source(resolve_repo_path(spec.dynamic_source), "dynamic", selection_policy, spec.dynamic_retention_ratio)
    pgdf_selection = resolve_selection_source(resolve_repo_path(spec.pgdf_source), "pgdf", selection_policy, spec.pgdf_retention_ratio)

    gt_map, gt_samples = build_gt_index(read_csv(noise_index))
    train_count = len(gt_samples)
    base_clean_total = sum(1 for sample in gt_samples if sample.is_clean)
    base_clean_ratio = safe_ratio(base_clean_total, train_count)

    all_top1 = read_all_training_top1(resolve_optional_repo_path(spec.all_results) or find_all_results(input_dir))
    dynamic_rows = read_csv(dynamic_selection)
    pgdf_rows = read_csv(pgdf_selection)
    dynamic_metrics = compute_selection_metrics(dynamic_rows, gt_map, base_clean_total)
    pgdf_metrics = compute_selection_metrics(pgdf_rows, gt_map, base_clean_total)

    dynamic_top1 = read_selection_top1(dynamic_selection.parent / "dynamic_loss_results.csv", dynamic_rows)
    pgdf_result_path = pgdf_selection.parent / "pgdf_results.csv"
    pgdf_top1 = read_selection_top1(pgdf_result_path, pgdf_rows)
    jaccard = read_pgdf_jaccard(pgdf_result_path, pgdf_rows)

    score_values, score_mode = load_prototype_scores(input_dir, leave_one_out)
    prototype_row, distribution = summarize_prototype_scores(spec.name, score_values, gt_map, score_mode)

    purity_rows = [
        build_purity_row(
            spec.name,
            "All training set",
            "all_train",
            train_count,
            base_clean_total,
            base_clean_total,
            base_clean_ratio,
            safe_ratio(base_clean_total, train_count),
            1.0,
            all_top1,
        ),
        build_purity_row(
            spec.name,
            "Dynamic r=0.8",
            display_path(dynamic_selection),
            dynamic_metrics.selected,
            dynamic_metrics.clean_selected,
            base_clean_total,
            base_clean_ratio,
            dynamic_metrics.purity,
            dynamic_metrics.clean_recall,
            dynamic_top1,
        ),
        build_purity_row(
            spec.name,
            "PGDF auto",
            display_path(pgdf_selection),
            pgdf_metrics.selected,
            pgdf_metrics.clean_selected,
            base_clean_total,
            base_clean_ratio,
            pgdf_metrics.purity,
            pgdf_metrics.clean_recall,
            pgdf_top1,
        ),
    ]

    jaccard_row = {
        "dataset": spec.name,
        "jaccard": jaccard,
        "dynamic_top1_pct": dynamic_top1,
        "pgdf_top1_pct": pgdf_top1,
        "pgdf_gain_pct": pct_difference(pgdf_top1, dynamic_top1),
        "dynamic_purity_pct": dynamic_metrics.purity * 100.0,
        "pgdf_purity_pct": pgdf_metrics.purity * 100.0,
        "purity_gain_pct": (pgdf_metrics.purity - dynamic_metrics.purity) * 100.0,
    }
    return {
        "purity_rows": purity_rows,
        "prototype_row": prototype_row,
        "jaccard_row": jaccard_row,
        "score_distribution": distribution,
    }


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def resolve_optional_repo_path(path: Path | None) -> Path | None:
    return resolve_repo_path(path) if path is not None else None


def resolve_selection_source(source: Path, kind: str, selection_policy: str, retention_ratio: float | None = None) -> Path:
    if source.is_file():
        return source
    if not source.exists():
        raise FileNotFoundError(f"Selection source does not exist: {source}")
    if selection_policy != "last":
        raise ValueError(f"Unsupported selection policy: {selection_policy}")
    prefix = "dynamic_loss_selection_" if kind == "dynamic" else "pgdf_selection_"
    candidates = sorted(source.glob(f"{prefix}*.csv"))
    if retention_ratio is not None:
        token = f"r{ratio_to_text(retention_ratio)}"
        candidates = [path for path in candidates if token in path.name]
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.csv files found under {source}")
    return max(candidates, key=lambda path: (extract_epoch(path), path.name))


def extract_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def ratio_to_text(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def build_gt_index(rows: list[dict[str, str]]) -> tuple[dict[str, GTSample], list[GTSample]]:
    key_to_sample: dict[str, GTSample] = {}
    samples: list[GTSample] = []
    for row_number, row in enumerate(rows):
        if row.get("split", "train").lower() != "train":
            continue
        clean_label = str(row.get("clean_label", ""))
        web_label = str(row.get("web_label", ""))
        if "is_noisy" in row and row["is_noisy"] != "":
            is_clean = str(row["is_noisy"]) in {"0", "false", "False", "no", "clean"}
        else:
            is_clean = clean_label == web_label
        sample = GTSample(
            row_id=str(row.get("index", row_number)),
            path=str(row.get("abs_path") or row.get("path") or ""),
            clean_label=clean_label,
            web_label=web_label,
            is_clean=is_clean,
        )
        samples.append(sample)
        paths = [row.get("path", ""), row.get("abs_path", "")]
        for raw_path in paths:
            for key in path_key_candidates(raw_path):
                existing = key_to_sample.get(key)
                if existing is None:
                    key_to_sample[key] = sample
                elif existing.row_id != sample.row_id:
                    raise ValueError(
                        f"Ambiguous path key '{key}' maps to multiple GT rows: "
                        f"{existing.row_id} and {sample.row_id}. Use a more specific path key."
                    )
    if not samples:
        raise ValueError("No train samples found in noise index.")
    return key_to_sample, samples


def path_key_candidates(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    normalized = normalize_path_text(path)
    if not normalized:
        return []
    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip("/")
        if value and value not in candidates:
            candidates.append(value)

    add(normalized)
    for prefix in ("images/", "train/", "test/", "val/"):
        if normalized.startswith(prefix):
            add(normalized)
            if prefix == "images/":
                add(normalized[len(prefix) :])
    for marker in ("/images/", "/train/", "/test/", "/val/"):
        if marker in f"/{normalized}":
            before, after = f"/{normalized}".split(marker, 1)
            if marker == "/images/":
                add(after)
                add(f"images/{after}")
            else:
                add(f"{marker.strip('/')}/{after}")
                add(after)
    parts = [part for part in normalized.split("/") if part]
    for count in (3, 2):
        if len(parts) >= count:
            add("/".join(parts[-count:]))
    return candidates


def normalize_path_text(path: str | Path) -> str:
    text = str(path).strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    return text.strip("/")


def resolve_gt_sample(path: str, gt_map: dict[str, GTSample]) -> GTSample | None:
    for key in path_key_candidates(path):
        sample = gt_map.get(key)
        if sample is not None:
            return sample
    return None


def compute_selection_metrics(selection_rows: list[dict[str, str]], gt_map: dict[str, GTSample], base_clean_total: int) -> SelectionMetrics:
    require_fields(selection_rows, {"path", "state"}, "selection CSV")
    selected = 0
    clean_selected = 0
    missing: list[str] = []
    for row in selection_rows:
        sample = resolve_gt_sample(row["path"], gt_map)
        if sample is None:
            missing.append(row["path"])
            continue
        if row["state"] == "clean":
            selected += 1
            clean_selected += int(sample.is_clean)
    if missing:
        preview = "; ".join(path_key_candidates(path)[0] if path_key_candidates(path) else path for path in missing[:5])
        raise ValueError(f"{len(missing)} selection rows could not be matched to GT noise index. Examples: {preview}")
    return SelectionMetrics(
        selected=selected,
        clean_selected=clean_selected,
        purity=safe_ratio(clean_selected, selected),
        clean_recall=safe_ratio(clean_selected, base_clean_total),
    )


def require_fields(rows: list[dict[str, str]], required: set[str], source: str) -> None:
    if not rows:
        raise ValueError(f"Empty {source}.")
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"{source} is missing required fields: {sorted(missing)}")


def build_purity_row(
    dataset: str,
    method: str,
    selection_source: str,
    selected: int,
    clean_selected: int,
    base_clean_total: int,
    base_clean_ratio: float,
    purity: float,
    clean_recall: float,
    top1: float | str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method": method,
        "selection_source": selection_source,
        "selected": int(selected),
        "clean_selected": int(clean_selected),
        "base_clean_total": int(base_clean_total),
        "base_clean_ratio_pct": base_clean_ratio * 100.0,
        "purity_pct": purity * 100.0,
        "clean_recall_pct": clean_recall * 100.0,
        "top1_pct": top1,
    }


def build_web_jaccard_gain_rows(rows: list[dict[str, float | str]]) -> list[dict[str, Any]]:
    """Add WebFG routing rows without GT-purity fields, because WebFG has no clean-label GT."""
    output = []
    for row in rows:
        output.append(
            {
                "dataset": row["dataset"],
                "jaccard": row["jaccard"],
                "dynamic_top1_pct": "",
                "pgdf_top1_pct": "",
                "pgdf_gain_pct": row["pgdf_gain_pct"],
                "dynamic_purity_pct": "",
                "pgdf_purity_pct": "",
                "purity_gain_pct": "",
            }
        )
    return output


def load_prototype_scores(input_dir: Path, leave_one_out: bool) -> tuple[list[dict[str, str]], str]:
    if leave_one_out:
        return build_leave_one_out_score_rows(input_dir), "leave_one_out_centroid"
    centroid_path = input_dir / "centroid_scores.csv"
    if not centroid_path.exists():
        raise FileNotFoundError(f"Missing centroid score file: {centroid_path}")
    return read_csv(centroid_path), "centroid_score"


def build_leave_one_out_score_rows(input_dir: Path) -> list[dict[str, str]]:
    feature_path = input_dir / "features_cls.npy"
    labels_path = input_dir / "labels.npy"
    paths_path = input_dir / "paths.txt"
    for path in [feature_path, labels_path, paths_path]:
        if not path.exists():
            raise FileNotFoundError(f"Leave-one-out prototype scores require {path}")
    features = np.load(feature_path)
    labels = np.load(labels_path, allow_pickle=True).astype(str)
    paths = paths_path.read_text(encoding="utf-8").splitlines()
    if len(features) != len(labels) or len(paths) != len(labels):
        raise ValueError("features_cls.npy, labels.npy, and paths.txt lengths do not match.")

    normalized = normalize_rows(features.astype(np.float32))
    scores = np.zeros(len(labels), dtype=np.float32)
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        if len(idx) <= 1:
            scores[idx] = np.nan
            continue
        class_sum = normalized[idx].sum(axis=0)
        for sample_idx in idx:
            proto = (class_sum - normalized[sample_idx]) / float(len(idx) - 1)
            norm = np.linalg.norm(proto)
            if norm <= 0.0:
                scores[sample_idx] = np.nan
            else:
                scores[sample_idx] = float(normalized[sample_idx] @ (proto / norm))
    return [{"index": str(i), "path": paths[i], "web_label": labels[i], "centroid_score": str(float(scores[i]))} for i in range(len(labels))]


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def summarize_prototype_scores(
    dataset: str,
    score_rows: list[dict[str, str]],
    gt_map: dict[str, GTSample],
    score_mode: str,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    require_fields(score_rows, {"path", "centroid_score"}, "prototype score CSV")
    clean_scores: list[float] = []
    noisy_scores: list[float] = []
    missing: list[str] = []
    for row in score_rows:
        sample = resolve_gt_sample(row["path"], gt_map)
        if sample is None:
            missing.append(row["path"])
            continue
        score = float(row["centroid_score"])
        if math.isnan(score):
            continue
        if sample.is_clean:
            clean_scores.append(score)
        else:
            noisy_scores.append(score)
    if missing:
        preview = "; ".join(path_key_candidates(path)[0] if path_key_candidates(path) else path for path in missing[:5])
        raise ValueError(f"{len(missing)} prototype score rows could not be matched to GT noise index. Examples: {preview}")
    row = {
        "dataset": dataset,
        "score_mode": score_mode,
        "num_clean": len(clean_scores),
        "num_noisy": len(noisy_scores),
        "clean_mean": mean(clean_scores),
        "noisy_mean": mean(noisy_scores),
        "mean_gap": mean(clean_scores) - mean(noisy_scores),
        "clean_median": median(clean_scores),
        "noisy_median": median(noisy_scores),
        "auc_clean_vs_noisy": binary_auc([1] * len(clean_scores) + [0] * len(noisy_scores), clean_scores + noisy_scores),
    }
    return row, {"clean": clean_scores, "noisy": noisy_scores}


def binary_auc(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")
    positives = sum(1 for label in labels if label == 1)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        average_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def find_all_results(input_dir: Path) -> Path | None:
    for relative in ["lora_30_42/lora_results.csv", "lora42/lora_results.csv", "lora/lora_results.csv"]:
        path = input_dir / relative
        if path.exists():
            return path
    return None


def read_all_training_top1(path: Path | None) -> float | str:
    if path is None or not path.exists():
        return ""
    rows = read_csv(path)
    for row in rows:
        if "all noisy" in row.get("method", "").lower():
            return top1_to_percent(row.get("best_top1", ""))
    return top1_to_percent(rows[0].get("best_top1", "")) if rows else ""


def read_selection_top1(results_path: Path, selection_rows: list[dict[str, str]]) -> float | str:
    if not results_path.exists():
        return ""
    rows = read_csv(results_path)
    if not rows:
        return ""
    selected_method = selection_rows[0].get("method", "") if selection_rows else ""
    selected_seed = selection_rows[0].get("seed", "") if selection_rows else ""
    selected_ratio = selection_rows[0].get("retention_ratio", "") if selection_rows else ""
    selected_proto = selection_rows[0].get("proto_keep_ratio", "") if selection_rows else ""

    for row in rows:
        if selected_method and row.get("method") == selected_method and seeds_match(row.get("seed", ""), selected_seed):
            return top1_to_percent(row.get("best_top1", ""))
    for row in rows:
        if selected_ratio and same_float_text(row.get("retention_ratio", ""), selected_ratio):
            if selected_proto and row.get("proto_keep_ratio", "") and not same_float_text(row.get("proto_keep_ratio", ""), selected_proto):
                continue
            return top1_to_percent(row.get("best_top1", ""))
    return top1_to_percent(rows[0].get("best_top1", ""))


def read_pgdf_jaccard(results_path: Path, selection_rows: list[dict[str, str]]) -> float | str:
    if not results_path.exists():
        return ""
    rows = read_csv(results_path)
    selected_method = selection_rows[0].get("method", "") if selection_rows else ""
    selected_seed = selection_rows[0].get("seed", "") if selection_rows else ""
    for row in rows:
        if selected_method and row.get("method") == selected_method and seeds_match(row.get("seed", ""), selected_seed):
            return parse_optional_float(row.get("auto_proto_jaccard", ""))
    return parse_optional_float(rows[0].get("auto_proto_jaccard", "")) if rows else ""


def top1_to_percent(value: str) -> float | str:
    parsed = parse_optional_float(value)
    if parsed == "":
        return ""
    return parsed * 100.0 if parsed <= 1.0 else parsed


def parse_optional_float(value: str) -> float | str:
    if value is None or str(value).strip() == "":
        return ""
    return float(value)


def same_float_text(left: str, right: str) -> bool:
    try:
        return abs(float(left) - float(right)) < 1.0e-9
    except ValueError:
        return left == right


def seeds_match(left: str, right: str) -> bool:
    return not right or left == right


def pct_difference(left: float | str, right: float | str) -> float | str:
    if left == "" or right == "":
        return ""
    return float(left) - float(right)


def safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def write_plots(out_dir: Path, purity_rows: list[dict[str, Any]], distributions: dict[str, dict[str, list[float]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_distribution_plots(out_dir, distributions, plt)
    write_metric_bar_plot(out_dir / "selected_purity_bar.png", purity_rows, "purity_pct", "Selected Purity (%)", include_all=True, plt=plt)
    write_metric_bar_plot(out_dir / "clean_recall_bar.png", purity_rows, "clean_recall_pct", "Clean Recall (%)", include_all=False, plt=plt)


def write_distribution_plots(out_dir: Path, distributions: dict[str, dict[str, list[float]]], plt: Any) -> None:
    all_scores = [score for groups in distributions.values() for values in groups.values() for score in values]
    if not all_scores:
        return
    x_min = min(all_scores)
    x_max = max(all_scores)
    bins = np.linspace(x_min, x_max, 51)
    for dataset, groups in distributions.items():
        plt.figure(figsize=(6, 4))
        plt.hist(groups["clean"], bins=bins, alpha=0.6, density=True, label="Clean samples")
        plt.hist(groups["noisy"], bins=bins, alpha=0.6, density=True, label="Noisy samples")
        plt.xlim(x_min, x_max)
        plt.xlabel("Prototype score")
        plt.ylabel("Density")
        plt.title(f"Prototype Score Distribution on {dataset}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"prototype_score_distribution_{slugify(dataset)}.png", dpi=300)
        plt.close()


def write_metric_bar_plot(path: Path, purity_rows: list[dict[str, Any]], metric: str, ylabel: str, include_all: bool, plt: Any) -> None:
    datasets = sorted({row["dataset"] for row in purity_rows})
    methods = ["All training set", "Dynamic r=0.8", "PGDF auto"] if include_all else ["Dynamic r=0.8", "PGDF auto"]
    width = 0.8 / len(methods)
    x = np.arange(len(datasets))

    plt.figure(figsize=(7, 4))
    for i, method in enumerate(methods):
        values = [float(next(row for row in purity_rows if row["dataset"] == dataset and row["method"] == method)[metric]) for dataset in datasets]
        plt.bar(x + (i - (len(methods) - 1) / 2.0) * width, values, width=width, label=method)
    plt.xticks(x, datasets, rotation=15, ha="right")
    plt.ylabel(ylabel)
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def write_summary(
    path: Path,
    purity_rows: list[dict[str, Any]],
    proto_rows: list[dict[str, Any]],
    jaccard_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# PGDF GT Purity Analysis",
        "",
        "GT purity is computed only for synthetic asym40 datasets where clean labels are known.",
        "",
        "## Selected Set Purity",
        "",
        "| dataset | method | selected | clean selected | purity | clean recall | top1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in purity_rows:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {int(row['selected'])} | {int(row['clean_selected'])} | "
            f"{format_float(row['purity_pct'])} | {format_float(row['clean_recall_pct'])} | {format_float(row['top1_pct'])} |"
        )
    lines.extend(
        [
            "",
            "## Prototype Score Separability",
            "",
            "| dataset | clean mean | noisy mean | mean gap | clean median | noisy median | AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in proto_rows:
        lines.append(
            f"| {row['dataset']} | {format_float(row['clean_mean'])} | {format_float(row['noisy_mean'])} | "
            f"{format_float(row['mean_gap'])} | {format_float(row['clean_median'])} | "
            f"{format_float(row['noisy_median'])} | {format_float(row['auc_clean_vs_noisy'])} |"
        )
    lines.extend(
        [
            "",
            "## Jaccard, Purity, and Top-1 Gain",
            "",
            "| dataset | Jaccard | Dynamic Top-1 | PGDF Top-1 | PGDF gain | Dynamic purity | PGDF purity | Purity gain |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in jaccard_rows:
        lines.append(
            f"| {row['dataset']} | {format_float(row['jaccard'])} | {format_float(row['dynamic_top1_pct'])} | "
            f"{format_float(row['pgdf_top1_pct'])} | {format_signed_float(row['pgdf_gain_pct'])} | "
            f"{format_float(row['dynamic_purity_pct'])} | {format_float(row['pgdf_purity_pct'])} | "
            f"{format_signed_float(row['purity_gain_pct'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):.2f}"


def format_signed_float(value: Any) -> str:
    if value == "":
        return ""
    return f"{float(value):+.2f}"


if __name__ == "__main__":
    main()
