from __future__ import annotations

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


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms

