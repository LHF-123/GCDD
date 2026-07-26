"""Fixed clean-validation protocol helpers for synthetic noisy-label experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import centroid_scores, per_class_keep_counts, select_top_per_class
from .graph import build_rrf_graphs
from .io_utils import read_csv, write_csv, write_json
from .scoring import compute_scores
from .selection_utils import path_key_candidates


PROTOCOL_NAME = "fixed_clean_validation_v1"


@dataclass(frozen=True)
class FixedValidationSplit:
    """A deterministic held-out clean validation split over the original train pool."""

    train_mask: np.ndarray
    validation_mask: np.ndarray
    clean_labels: np.ndarray
    metadata: dict[str, Any]


def load_clean_labels_from_noise_index(noise_index_path: Path, train_paths: list[str]) -> np.ndarray:
    """Match clean labels from a synthetic-noise index to V1 train paths.

    The labels returned here are used only to create/evaluate the held-out validation
    split. They must never be supplied to the model training or PGDF selection code.
    """
    rows = read_csv(noise_index_path)
    key_to_label: dict[str, str] = {}
    for row in rows:
        if str(row.get("split", "train")).lower() != "train":
            continue
        clean_label = str(row.get("clean_label", ""))
        if not clean_label:
            raise ValueError(f"{noise_index_path} is missing clean_label for a training row.")
        for raw_path in (row.get("path", ""), row.get("abs_path", "")):
            for key in path_key_candidates(raw_path):
                existing = key_to_label.get(key)
                if existing is not None and existing != clean_label:
                    raise ValueError(f"Conflicting clean labels for path key '{key}' in {noise_index_path}.")
                key_to_label[key] = clean_label

    labels: list[str] = []
    missing: list[str] = []
    for path in train_paths:
        matched_label = None
        for key in path_key_candidates(path):
            if key in key_to_label:
                matched_label = key_to_label[key]
                break
        if matched_label is None:
            missing.append(path)
        else:
            labels.append(matched_label)
    if missing:
        preview = "; ".join(missing[:5])
        raise ValueError(f"Noise index is missing {len(missing)} V1 training paths. Examples: {preview}")
    return np.asarray(labels, dtype=str)


def build_or_load_fixed_validation_split(
    output_dir: Path,
    train_paths: list[str],
    clean_labels: np.ndarray,
    validation_ratio: float,
    validation_seed: int,
) -> FixedValidationSplit:
    """Create once, then strictly reuse a class-stratified clean validation manifest."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must satisfy 0 < validation_ratio < 1.")
    clean_labels = np.asarray(clean_labels, dtype=str)
    if clean_labels.shape != (len(train_paths),):
        raise ValueError("clean_labels must align with train_paths.")

    manifest_path = output_dir / "validation_manifest.csv"
    metadata_path = output_dir / "validation_manifest.json"
    path_hash = hash_paths(train_paths)
    if manifest_path.exists() or metadata_path.exists():
        if not manifest_path.exists() or not metadata_path.exists():
            raise ValueError("Validation manifest CSV and JSON must either both exist or both be absent.")
        return load_existing_validation_split(
            manifest_path,
            metadata_path,
            train_paths,
            clean_labels,
            validation_ratio,
            validation_seed,
            path_hash,
        )

    validation_mask = np.zeros(len(train_paths), dtype=bool)
    rng = np.random.default_rng(validation_seed)
    for clean_label in sorted(set(clean_labels.tolist())):
        class_idx = np.where(clean_labels == clean_label)[0]
        if len(class_idx) < 2:
            raise ValueError(f"Cannot reserve validation data for clean class {clean_label}: only {len(class_idx)} sample(s).")
        keep = max(1, int(np.floor(len(class_idx) * validation_ratio)))
        keep = min(keep, len(class_idx) - 1)
        selected = rng.permutation(np.sort(class_idx))[:keep]
        validation_mask[selected] = True

    train_mask = ~validation_mask
    metadata = {
        "protocol": PROTOCOL_NAME,
        "validation_ratio": float(validation_ratio),
        "validation_seed": int(validation_seed),
        "train_paths_sha256": path_hash,
        "source_train_samples": int(len(train_paths)),
        "training_pool_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "stratification_label": "clean_label",
        "validation_label": "clean_label",
    }
    rows = [
        {
            "index": int(index),
            "path": train_paths[index],
            "clean_label": str(clean_labels[index]),
            "partition": "validation" if validation_mask[index] else "training_pool",
        }
        for index in range(len(train_paths))
    ]
    write_csv(manifest_path, rows, ["index", "path", "clean_label", "partition"])
    write_json(metadata_path, metadata)
    return FixedValidationSplit(train_mask=train_mask, validation_mask=validation_mask, clean_labels=clean_labels, metadata=metadata)


def load_existing_validation_split(
    manifest_path: Path,
    metadata_path: Path,
    train_paths: list[str],
    clean_labels: np.ndarray,
    validation_ratio: float,
    validation_seed: int,
    path_hash: str,
) -> FixedValidationSplit:
    import json

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "protocol": PROTOCOL_NAME,
        "validation_seed": int(validation_seed),
        "train_paths_sha256": path_hash,
        "source_train_samples": len(train_paths),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Existing validation manifest has incompatible {key}: {metadata.get(key)!r} != {value!r}.")
    if not np.isclose(float(metadata.get("validation_ratio", -1.0)), validation_ratio):
        raise ValueError("Existing validation manifest uses a different validation_ratio.")

    rows = read_csv(manifest_path)
    if len(rows) != len(train_paths):
        raise ValueError("Existing validation manifest does not cover every training sample.")
    validation_mask = np.zeros(len(train_paths), dtype=bool)
    for expected_index, row in enumerate(rows):
        if int(row.get("index", -1)) != expected_index or row.get("path") != train_paths[expected_index]:
            raise ValueError("Existing validation manifest no longer aligns with paths.txt.")
        if row.get("clean_label") != str(clean_labels[expected_index]):
            raise ValueError("Existing validation manifest no longer aligns with the synthetic-noise index.")
        partition = row.get("partition", "")
        if partition not in {"training_pool", "validation"}:
            raise ValueError(f"Unknown validation partition '{partition}' at index {expected_index}.")
        validation_mask[expected_index] = partition == "validation"
    if not np.any(validation_mask) or np.all(validation_mask):
        raise ValueError("Existing validation manifest must contain both validation and training-pool samples.")
    return FixedValidationSplit(
        train_mask=~validation_mask,
        validation_mask=validation_mask,
        clean_labels=np.asarray(clean_labels, dtype=str),
        metadata=metadata,
    )


def build_validation_safe_pgdf_reference(
    features: dict[str, np.ndarray],
    noisy_labels: np.ndarray,
    training_pool_mask: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Recompute PGDF's graph budget and centroid reference using training-pool samples only."""
    noisy_labels = np.asarray(noisy_labels, dtype=str)
    training_pool_mask = np.asarray(training_pool_mask, dtype=bool)
    if noisy_labels.shape != training_pool_mask.shape:
        raise ValueError("training_pool_mask must align with noisy_labels.")
    required = {"cls", "gap", "top"}
    missing = required - set(features)
    if missing:
        raise ValueError(f"Validation-safe PGDF reference is missing feature sets: {sorted(missing)}")
    for name, values in features.items():
        if values.shape[0] != len(noisy_labels):
            raise ValueError(f"Feature array {name} does not align with noisy_labels.")

    pool_idx = np.where(training_pool_mask)[0]
    pool_labels = noisy_labels[pool_idx]
    pool_features = {name: np.asarray(values[pool_idx], dtype=np.float32) for name, values in features.items()}
    graphs = build_rrf_graphs(pool_features, pool_labels, cfg)
    _, split_info = compute_scores(pool_labels, graphs, cfg)
    gcdd_clean_pool = np.asarray(split_info["state"] == "clean", dtype=bool)
    keep_counts = per_class_keep_counts(pool_labels, gcdd_clean_pool)
    pool_proto_scores = centroid_scores(pool_features["cls"], pool_labels)
    centroid_reference_pool = select_top_per_class(pool_proto_scores, pool_labels, keep_counts, largest=True)

    proto_scores = np.full(len(noisy_labels), np.nan, dtype=np.float32)
    centroid_reference = np.zeros(len(noisy_labels), dtype=bool)
    gcdd_clean = np.zeros(len(noisy_labels), dtype=bool)
    proto_scores[pool_idx] = pool_proto_scores
    centroid_reference[pool_idx] = centroid_reference_pool
    gcdd_clean[pool_idx] = gcdd_clean_pool

    return {
        "proto_scores": proto_scores,
        "centroid_reference_mask": centroid_reference,
        "gcdd_clean_mask": gcdd_clean,
        "per_class_keep_counts": keep_counts,
        "pool_indices": pool_idx,
    }


def hash_paths(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
