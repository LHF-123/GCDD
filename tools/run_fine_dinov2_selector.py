from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json
from gcdd.selection_utils import build_gt_clean_mask_from_noise_rows


SCORE_FIELDS = ["index", "path", "web_label", "score_mode", "fine_score", "class_size", "small_class"]
SELECTION_FIELDS = ["index", "path", "web_label", "fine_score", "state"]
SUMMARY_FIELDS = [
    "score_mode",
    "keep_ratio",
    "selected",
    "selected_ratio",
    "selected_clean",
    "purity",
    "clean_recall",
    "small_class_count",
    "small_class_samples",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FINE-style DINOv2 feature filtering selections.")
    parser.add_argument("--input-dir", required=True, help="V1 output directory containing features_cls.npy, labels.npy, and paths.txt.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/fine_dinov2.")
    parser.add_argument("--feature-file", default="features_cls.npy", help="Feature npy relative to input-dir.")
    parser.add_argument("--labels-file", default="labels.npy", help="Labels npy relative to input-dir.")
    parser.add_argument("--paths-file", default="paths.txt", help="Paths txt relative to input-dir.")
    parser.add_argument("--keep-ratios", default="0.6,0.8", help="Comma-separated class-wise keep ratios.")
    parser.add_argument("--score-modes", default="center,nocenter", help="Comma-separated score modes: center,nocenter.")
    parser.add_argument("--min-class-size", type=int, default=3, help="Classes smaller than this use --small-class-policy.")
    parser.add_argument("--small-class-policy", choices=["keep_all"], default="keep_all", help="Policy for tiny noisy-label classes.")
    parser.add_argument("--noise-index", help="Optional synthetic-noise index CSV for selected purity / clean recall.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "fine_dinov2"
    ensure_dir(output_dir)

    keep_ratios = parse_float_list(args.keep_ratios, "--keep-ratios")
    score_modes = parse_score_modes(args.score_modes)
    features = np.load(input_dir / args.feature_file)
    labels = np.load(input_dir / args.labels_file, allow_pickle=True).astype(str)
    paths = (input_dir / args.paths_file).read_text(encoding="utf-8").splitlines()
    validate_inputs(features, labels, paths, keep_ratios, args.min_class_size)

    gt_clean = load_gt_clean_mask(Path(args.noise_index), paths) if args.noise_index else None
    all_score_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for mode in score_modes:
        scores, class_size, small_class = compute_fine_scores(features, labels, center=(mode == "center"), min_class_size=args.min_class_size)
        all_score_rows.extend(build_score_rows(paths, labels, scores, class_size, small_class, mode))
        for keep_ratio in keep_ratios:
            selected = select_classwise(scores, labels, keep_ratio, small_class)
            selection_path = output_dir / f"fine_selection_{mode}_p{ratio_to_text(keep_ratio)}.csv"
            write_csv(selection_path, build_selection_rows(paths, labels, scores, selected), SELECTION_FIELDS)
            summary_rows.append(build_summary_row(mode, keep_ratio, selected, gt_clean, small_class, labels))

    write_csv(output_dir / "fine_scores.csv", all_score_rows, SCORE_FIELDS)
    write_csv(output_dir / "fine_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_json(
        output_dir / "fine_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "keep_ratios": keep_ratios,
            "score_modes": score_modes,
            "min_class_size": int(args.min_class_size),
            "small_class_policy": args.small_class_policy,
            "summary": summary_rows,
        },
    )
    print(f"FINE-style DINOv2 selections written to {output_dir}", flush=True)


def compute_fine_scores(features: np.ndarray, labels: np.ndarray, *, center: bool, min_class_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = l2_normalize(features.astype(np.float32))
    scores = np.zeros(len(labels), dtype=np.float32)
    class_size = np.zeros(len(labels), dtype=np.int32)
    small_class = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        class_size[idx] = len(idx)
        if len(idx) < min_class_size:
            small_class[idx] = True
            scores[idx] = 0.0
            continue
        x = normalized[idx]
        if center:
            mu = x.mean(axis=0, keepdims=True)
            x_svd = x - mu
        else:
            mu = None
            x_svd = x
        try:
            _, _, vt = np.linalg.svd(x_svd, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(f"SVD failed for class {label} with {len(idx)} samples.") from exc
        v1 = vt[0]
        class_scores = np.abs((x - mu) @ v1) if center and mu is not None else np.abs(x @ v1)
        scores[idx] = class_scores.astype(np.float32)
    return scores, class_size, small_class


def select_classwise(scores: np.ndarray, labels: np.ndarray, keep_ratio: float, small_class: np.ndarray) -> np.ndarray:
    selected = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        if len(idx) == 0:
            continue
        if np.all(small_class[idx]):
            selected[idx] = True
            continue
        keep = len(idx) if keep_ratio >= 1.0 else max(1, int(math.floor(len(idx) * keep_ratio)))
        order = np.argsort(-scores[idx], kind="mergesort")
        selected[idx[order[:keep]]] = True
    return selected


def build_score_rows(paths: list[str], labels: np.ndarray, scores: np.ndarray, class_size: np.ndarray, small_class: np.ndarray, mode: str) -> list[dict[str, Any]]:
    return [
        {
            "index": int(i),
            "path": paths[i],
            "web_label": labels[i],
            "score_mode": mode,
            "fine_score": float(scores[i]),
            "class_size": int(class_size[i]),
            "small_class": "yes" if small_class[i] else "no",
        }
        for i in range(len(paths))
    ]


def build_selection_rows(paths: list[str], labels: np.ndarray, scores: np.ndarray, selected: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "index": int(i),
            "path": paths[i],
            "web_label": labels[i],
            "fine_score": float(scores[i]),
            "state": "clean" if selected[i] else "ignored",
        }
        for i in range(len(paths))
    ]


def build_summary_row(mode: str, keep_ratio: float, selected: np.ndarray, gt_clean: np.ndarray | None, small_class: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    selected_count = int(selected.sum())
    selected_clean = int(np.sum(selected & gt_clean)) if gt_clean is not None else ""
    total_clean = int(gt_clean.sum()) if gt_clean is not None else 0
    return {
        "score_mode": mode,
        "keep_ratio": float(keep_ratio),
        "selected": selected_count,
        "selected_ratio": selected_count / max(1, len(selected)),
        "selected_clean": selected_clean,
        "purity": float(selected_clean) / selected_count if gt_clean is not None and selected_count else "",
        "clean_recall": float(selected_clean) / total_clean if gt_clean is not None and total_clean else "",
        "small_class_count": int(len(set(labels[small_class].tolist()))),
        "small_class_samples": int(small_class.sum()),
    }


def load_gt_clean_mask(noise_index: Path, train_paths: list[str]) -> np.ndarray:
    return build_gt_clean_mask_from_noise_rows(read_csv(noise_index), train_paths)


def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def validate_inputs(features: np.ndarray, labels: np.ndarray, paths: list[str], keep_ratios: list[float], min_class_size: int) -> None:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, d], got {features.shape}.")
    if len(features) != len(labels) or len(paths) != len(labels):
        raise ValueError("features, labels, and paths lengths do not match.")
    if min_class_size < 1:
        raise ValueError("--min-class-size must be >= 1.")
    for ratio in keep_ratios:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"keep ratio must satisfy 0 < p <= 1, got {ratio}.")


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def parse_score_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in {"center", "nocenter"}]
    if unknown:
        raise ValueError(f"Unknown score modes: {unknown}. Available: center,nocenter.")
    if not modes:
        raise ValueError("At least one score mode is required.")
    return modes


def ratio_to_text(value: float) -> str:
    return f"{value:g}".replace(".", "p")


if __name__ == "__main__":
    main()
