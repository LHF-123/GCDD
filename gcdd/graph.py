from __future__ import annotations

import numpy as np

from .progress import log_stage, progress_iter


def build_rrf_graphs(features: dict[str, np.ndarray], labels: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    graph_cfg = cfg["graph"]
    backend = graph_cfg.get("knn_backend", "auto")
    if backend not in {"auto", "numpy"}:
        raise ValueError("V0 currently supports exact numpy KNN only; use graph.knn_backend=auto or numpy.")
    log_stage("[graph] Building class-wise RRF graph.")
    class_indices, class_weights = build_rrf_graph(
        features,
        labels,
        mode="class",
        k_pool=int(graph_cfg["k_pool_class"]),
        k=int(graph_cfg["k_class"]),
        k0=float(graph_cfg["rrf_k0"]),
    )
    log_stage("[graph] Building global RRF graph.")
    global_indices, global_weights = build_rrf_graph(
        features,
        labels,
        mode="global",
        k_pool=int(graph_cfg["k_pool_global"]),
        k=int(graph_cfg["k_global"]),
        k0=float(graph_cfg["rrf_k0"]),
    )
    return {
        "class_indices": class_indices,
        "class_weights": class_weights,
        "global_indices": global_indices,
        "global_weights": global_weights,
    }


def build_rrf_graph(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    mode: str,
    k_pool: int,
    k: int,
    k0: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(labels)
    indices = np.full((n, k), -1, dtype=np.int64)
    weights = np.zeros((n, k), dtype=np.float32)
    normalized = {name: normalize_rows(value.astype(np.float32)) for name, value in features.items()}
    if mode == "class":
        pools = {name: class_topk_pool(value, labels, k_pool, desc=f"class pool {name}") for name, value in normalized.items()}
    elif mode == "global":
        pools = {name: global_topk_pool(value, k_pool, desc=f"global pool {name}") for name, value in normalized.items()}
    else:
        raise ValueError(f"Unsupported graph mode: {mode}")

    for i in progress_iter(range(n), total=n, desc=f"RRF fuse {mode}"):
        score_map: dict[int, float] = {}
        for pool in pools.values():
            for rank, j in enumerate(pool[i], start=1):
                if j < 0:
                    continue
                score_map[int(j)] = score_map.get(int(j), 0.0) + 1.0 / (k0 + rank)
        if not score_map:
            continue
        ranked = sorted(score_map.items(), key=lambda item: (-item[1], item[0]))[:k]
        final = np.array([item[0] for item in ranked], dtype=np.int64)
        final_scores = np.array([item[1] for item in ranked], dtype=np.float64)
        max_score = final_scores.max()
        indices[i, : len(final)] = final
        if max_score > 0:
            weights[i, : len(final)] = (final_scores / max_score).astype(np.float32)
    return indices, weights


def class_topk_pool(features: np.ndarray, labels: np.ndarray, k_pool: int, desc: str) -> np.ndarray:
    n = len(labels)
    pool = np.full((n, k_pool), -1, dtype=np.int64)
    unique_labels = sorted(set(labels.tolist()))
    for label in progress_iter(unique_labels, total=len(unique_labels), desc=desc):
        idx = np.where(labels == label)[0]
        if len(idx) <= 1:
            continue
        sim = features[idx] @ features[idx].T
        np.fill_diagonal(sim, -np.inf)
        top = sorted_topk(sim, min(k_pool, len(idx) - 1))
        pool[idx, : top.shape[1]] = idx[top]
    return pool


def global_topk_pool(features: np.ndarray, k_pool: int, desc: str, block_size: int = 256) -> np.ndarray:
    n = features.shape[0]
    pool = np.full((n, k_pool), -1, dtype=np.int64)
    take = min(k_pool, max(0, n - 1))
    if take == 0:
        return pool
    total_blocks = (n + block_size - 1) // block_size
    for start in progress_iter(range(0, n, block_size), total=total_blocks, desc=desc):
        end = min(start + block_size, n)
        sim = features[start:end] @ features.T
        rows = np.arange(end - start)
        sim[rows, np.arange(start, end)] = -np.inf
        top = sorted_topk(sim, take)
        pool[start:end, : top.shape[1]] = top
    return pool


def sorted_topk(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64)
    partial = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    partial_scores = np.take_along_axis(scores, partial, axis=1)
    order = np.argsort(-partial_scores, axis=1, kind="mergesort")
    return np.take_along_axis(partial, order, axis=1)


def rank_candidates(scores: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    if len(allowed) == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    take = min(k, len(allowed))
    allowed_scores = scores[allowed]
    order = np.argsort(-allowed_scores, kind="mergesort")[:take]
    return allowed[order]


def normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms
