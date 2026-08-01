from __future__ import annotations

import math

import numpy as np


def per_class_keep_counts(labels: np.ndarray, clean_mask: np.ndarray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in sorted(set(labels.tolist())):
        idx = labels == label
        counts[label] = int(np.sum(clean_mask[idx]))
    return counts


def select_top_per_class(scores: np.ndarray, labels: np.ndarray, keep_counts: dict[str, int], largest: bool = True) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=bool)
    for label, keep in keep_counts.items():
        idx = np.where(labels == label)[0]
        if len(idx) == 0 or keep <= 0:
            continue
        class_scores = scores[idx]
        order = np.argsort(-class_scores if largest else class_scores, kind="mergesort")
        mask[idx[order[: min(keep, len(idx))]]] = True
    return mask


def centroid_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    normalized = normalize_rows(features)
    centroids: dict[str, np.ndarray] = {}
    for label in sorted(set(labels.tolist())):
        idx = labels == label
        centroid = normalized[idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[label] = centroid / norm if norm > 0 else centroid
    return np.array([float(normalized[i] @ centroids[labels[i]]) for i in range(len(labels))], dtype=np.float32)


def compute_fine_scores(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    center: bool,
    min_class_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the repository's FINE-style first-singular-vector scores."""
    normalized = normalize_rows(features.astype(np.float32))
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


def select_fine_classwise(
    scores: np.ndarray,
    labels: np.ndarray,
    keep_ratio: float,
    small_class: np.ndarray,
) -> np.ndarray:
    """Select FINE samples class-wise, retaining all configured tiny classes."""
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


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms
