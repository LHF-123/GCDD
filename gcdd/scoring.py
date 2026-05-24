from __future__ import annotations

import numpy as np


def compute_scores(labels: np.ndarray, graphs: dict[str, np.ndarray], cfg: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    class_indices = graphs["class_indices"]
    class_weights = graphs["class_weights"]
    global_indices = graphs["global_indices"]

    d_class = class_density(class_indices, class_weights)
    r_class = reciprocal_ratio(class_indices)
    i_class_norm = normalized_indegree(class_indices, int(cfg["graph"]["k_class"]))
    q_same = global_label_purity(global_indices, labels)

    p_d = percentile_by_class(d_class, labels)
    p_r = percentile_by_class(r_class, labels)
    p_i = percentile_by_class(i_class_norm, labels)
    p_q = percentile_by_class(q_same, labels)

    eps = float(cfg["selection"]["epsilon"])
    s_clean = np.power((p_d + eps) * (p_r + eps) * (p_i + eps) * (p_q + eps), 0.25)

    metrics = {
        "D_class": d_class,
        "R_class": r_class,
        "I_class_norm": i_class_norm,
        "Q_same": q_same,
        "P_Dclass": p_d,
        "P_Rclass": p_r,
        "P_Iclass_norm": p_i,
        "P_Qsame": p_q,
        "S_clean": s_clean,
    }
    split, threshold_stats = adaptive_otsu_split(s_clean, labels, cfg)
    return metrics, {"state": split, **threshold_stats}


def class_density(indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = indices >= 0
    counts = valid.sum(axis=1)
    totals = (weights * valid).sum(axis=1)
    return safe_divide(totals, counts)


def reciprocal_ratio(indices: np.ndarray) -> np.ndarray:
    n = indices.shape[0]
    neighbor_sets = [set(row[row >= 0].tolist()) for row in indices]
    ratios = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neighbors = neighbor_sets[i]
        if not neighbors:
            continue
        reciprocal = sum(1 for j in neighbors if i in neighbor_sets[j])
        ratios[i] = reciprocal / len(neighbors)
    return ratios


def normalized_indegree(indices: np.ndarray, k_class: int) -> np.ndarray:
    n = indices.shape[0]
    indegree = np.zeros(n, dtype=np.float32)
    for row in indices:
        for j in row:
            if j >= 0:
                indegree[j] += 1.0
    denom = max(1, k_class)
    return np.minimum(1.0, indegree / denom)


def global_label_purity(indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    q_same = np.zeros(len(labels), dtype=np.float32)
    for i, row in enumerate(indices):
        valid = row[row >= 0]
        if len(valid) == 0:
            continue
        q_same[i] = np.mean(labels[valid] == labels[i])
    return q_same


def percentile_by_class(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    percentiles = np.zeros_like(values, dtype=np.float32)
    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        if len(idx) == 1:
            percentiles[idx] = 1.0
            continue
        order = np.argsort(values[idx], kind="mergesort")
        ranks = np.empty(len(idx), dtype=np.float32)
        ranks[order] = np.arange(len(idx), dtype=np.float32) / (len(idx) - 1)
        percentiles[idx] = ranks
    return percentiles


def adaptive_otsu_split(scores: np.ndarray, labels: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    bins = int(cfg["selection"]["otsu_bins"])
    low_clip, high_clip = [float(x) for x in cfg["selection"]["clean_ratio_clip"]]

    state = np.array(["ignored"] * len(scores), dtype=object)
    thresholds = np.zeros(len(scores), dtype=np.float32)
    clean_before = np.zeros(len(scores), dtype=np.float32)
    clean_after = np.zeros(len(scores), dtype=np.float32)
    clip_low = np.zeros(len(scores), dtype=np.int8)
    clip_high = np.zeros(len(scores), dtype=np.int8)

    for label in sorted(set(labels.tolist())):
        idx = np.where(labels == label)[0]
        class_scores = scores[idx]
        threshold = otsu_threshold(class_scores, bins)
        raw_clean = class_scores >= threshold
        before_ratio = float(raw_clean.mean()) if len(raw_clean) else 0.0

        target_clean = raw_clean.copy()
        if before_ratio < low_clip:
            keep = max(1, int(np.ceil(len(idx) * low_clip)))
            target_clean = topk_mask(class_scores, keep)
            clip_low[idx] = 1
        elif before_ratio > high_clip:
            keep = max(1, int(np.floor(len(idx) * high_clip)))
            target_clean = topk_mask(class_scores, keep)
            clip_high[idx] = 1

        state[idx[target_clean]] = "clean"
        thresholds[idx] = threshold
        clean_before[idx] = before_ratio
        clean_after[idx] = float(target_clean.mean()) if len(target_clean) else 0.0

    return state, {
        "otsu_threshold": thresholds,
        "clean_ratio_before_clip": clean_before,
        "clean_ratio_after_clip": clean_after,
        "clip_low": clip_low,
        "clip_high": clip_high,
    }


def otsu_threshold(values: np.ndarray, bins: int) -> float:
    if len(values) == 0:
        return 0.0
    if np.allclose(values, values[0]):
        return float(values[0])
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    sum_total = (hist * centers).sum()
    weight_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_threshold = centers[0]

    for count, center in zip(hist, centers):
        weight_bg += count
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += count * center
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_threshold = center
    return float(best_threshold)


def topk_mask(values: np.ndarray, k: int) -> np.ndarray:
    keep = min(k, len(values))
    order = np.argsort(-values, kind="mergesort")
    mask = np.zeros(len(values), dtype=bool)
    mask[order[:keep]] = True
    return mask


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out

