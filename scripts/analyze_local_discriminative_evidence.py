"""Read-only local discriminative-evidence audit for Margin-Rank checkpoints.

The script is intentionally separate from training.  It reuses the production
DINOv2+LoRA checkpoint construction, reads historical Margin-Rank memberships,
and executes inference only on noisy-training-pool images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analyze_ambiguous_consistency_gradients import (  # noqa: E402
    CheckpointSpec,
    assert_model_state_unchanged,
    load_audited_model,
    parse_path_maps,
    read_yaml,
    reconstruct_index_aligned_inputs,
    selection_epoch_for_checkpoint,
    snapshot_model_state,
)
from scripts.analyze_ambiguous_consistency_offline import (  # noqa: E402
    SelectionGroups,
    build_groups,
    selection_rows_by_epoch,
)


DATASETS = ("cub", "aircraft")
VARIANT = "margin_rank"
DEFAULT_VARIANTS_ROOT = Path(
    "outputs/neighbor_margin/cyclic_asym40_noise42/fixedval_s20250726/r08_p04_w5_u5/variants"
)
EPSILON = 1.0e-12


@dataclass(frozen=True)
class ReferenceImage:
    original_index: int
    observed_label: str
    cls: np.ndarray
    patches: np.ndarray
    patch_positions: tuple[int, ...]


@dataclass(frozen=True)
class QueryDescriptor:
    original_index: int
    group: str
    observed_label: str
    historical_competitor: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only local Patch-token evidence audit for Margin-Rank.")
    parser.add_argument("--variants-root", type=Path, default=DEFAULT_VARIANTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/local_discriminative_evidence_audit"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--diagnostic-seed", type=int, default=20261009)
    parser.add_argument("--reference-images-per-class", type=int, default=4)
    parser.add_argument("--reference-patches-per-image", type=int, default=16)
    parser.add_argument("--max-reference-patches-per-class", type=int, default=64)
    parser.add_argument("--queries-per-group", type=int, default=200)
    parser.add_argument("--top-patch-fraction", type=float, default=0.05)
    parser.add_argument("--random-patch-repeats", type=int, default=5)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--occlusion-query-count", type=int, default=24)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--dry-run", action="store_true", help="Audit metadata and manifests only; do not load images/model.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    materialized = list(rows)
    if fields is None:
        if not materialized:
            raise ValueError(f"Refusing to write schema-less empty CSV: {path}")
        # Summaries can legitimately combine methods with a few method-specific
        # columns. Keep a deterministic union rather than silently dropping them.
        fields = list(dict.fromkeys(key for row in materialized for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(*parts: Any) -> int:
    text = ":".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little", signed=False)


def normalise_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= EPSILON:
        raise FloatingPointError("Cannot L2-normalise a non-finite or zero feature vector.")
    return values / norm


def normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a matrix for row normalisation, got {values.shape}.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= EPSILON):
        raise FloatingPointError("Patch features include a non-finite or zero-norm row.")
    return values / norms


def find_spec(variants_root: Path, dataset: str, seed: int) -> CheckpointSpec:
    run_dir = variants_root / VARIANT / dataset / f"seed{seed}"
    checkpoint = run_dir / "checkpoints" / "best_val.pt"
    rows_path = run_dir / "selection_rows.csv"
    cfg_path = run_dir.parent / "resolved_config.yaml"
    missing = [str(path) for path in (checkpoint, rows_path, cfg_path, run_dir / "result.json") if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete Margin-Rank run for {dataset}/seed{seed}: {missing}")
    import torch

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, dict) or not isinstance(checkpoint_payload.get("best_epoch"), int):
        raise ValueError(f"Missing best_epoch in {checkpoint}")
    update_epoch = selection_epoch_for_checkpoint(
        int(checkpoint_payload["best_epoch"]), selection_rows_by_epoch(rows_path)
    )
    return CheckpointSpec(
        dataset=dataset,
        method=VARIANT,
        seed=seed,
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        checkpoint_epoch=int(checkpoint_payload["best_epoch"]),
        selection_update_epoch=update_epoch,
        cfg_path=cfg_path,
    )


def history_rows_and_groups(spec: CheckpointSpec) -> tuple[list[dict[str, str]], SelectionGroups]:
    epochs = selection_rows_by_epoch(spec.run_dir / "selection_rows.csv")
    rows = epochs.get(spec.selection_update_epoch)
    if not rows:
        raise ValueError(f"No historical selection rows for update epoch {spec.selection_update_epoch}.")
    groups = build_groups(rows)
    if groups.actual_active & groups.ambiguous_for_consistency:
        raise RuntimeError("Historical active and Ambiguous sets overlap.")
    return rows, groups


def audited_run_inputs(torch: Any, spec: CheckpointSpec) -> dict[str, Any]:
    """Load and cross-check historical metadata without loading model/image data."""
    cfg = read_yaml(spec.cfg_path)
    pgdf = cfg.get("pgdf", {})
    if pgdf.get("geometry_mode") != "neighbor_margin" or bool(pgdf.get("neighbor_margin_positive_only", True)):
        raise ValueError(f"{spec.run_dir} is not a Margin-Rank (neighbor-margin, positive-only=false) run.")
    result = read_json(spec.run_dir / "result.json")
    rows, groups = history_rows_and_groups(spec)
    noise_index = Path(str(result.get("noise_index", "")))
    if not noise_index.exists():
        raise FileNotFoundError(f"Saved noise index is unavailable: {noise_index}")
    dataset_root = str(cfg.get("dataset", {}).get("root", ""))
    truth_by_path = parse_truth_by_path(noise_index, dataset_root)
    validate_noise_alignment(groups, truth_by_path)
    checkpoint_payload: dict[str, Any] = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
    classes = [str(value) for value in checkpoint_payload.get("classes", [])]
    if not classes:
        raise ValueError(f"Checkpoint classes are missing: {spec.checkpoint_path}")
    label_to_id = {label: position for position, label in enumerate(classes)}
    if any(groups.labels[index] not in label_to_id for index in groups.pool):
        raise ValueError("Historical observed labels are not aligned with checkpoint classifier classes.")
    validation_manifest = spec.run_dir.parent / "validation_manifest.json"
    if not validation_manifest.exists():
        raise FileNotFoundError(f"Saved validation manifest is unavailable: {validation_manifest}")
    return {
        "cfg": cfg,
        "result": result,
        "rows": rows,
        "groups": groups,
        "noise_index": noise_index,
        "truth_by_path": truth_by_path,
        "checkpoint_payload": checkpoint_payload,
        "classes": classes,
        "label_to_id": label_to_id,
        "validation_manifest": validation_manifest,
    }


def canonical_path(path: str) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def parse_truth_by_path(noise_index: Path, dataset_root: str) -> dict[str, dict[str, str]]:
    records = read_csv(noise_index)
    required = {"clean_label", "web_label", "is_noisy"}
    missing = required - set(records[0] if records else [])
    if missing:
        raise ValueError(f"Noise index lacks required post-hoc fields: {sorted(missing)}")
    key_field = "abs_path" if "abs_path" in records[0] else "path" if "path" in records[0] else None
    if key_field is None:
        raise ValueError(f"Noise index requires either abs_path or path for sample alignment: {noise_index}")
    lookup: dict[str, dict[str, str]] = {}

    def add_key(key: str, row: dict[str, str]) -> None:
        existing = lookup.get(key)
        if existing is not None and existing != row:
            raise ValueError(f"Ambiguous path mapping in noise index: {key}")
        lookup[key] = row

    root = canonical_path(dataset_root) if dataset_root else ""
    for row in records:
        key = canonical_path(str(row[key_field]))
        add_key(key, row)
        # Official noise indices may keep a dataset-relative path while the
        # training selection CSV stores the resolved absolute image path.
        if root and not key.startswith("/") and not (len(key) >= 3 and key[1:3] == ":/"):
            add_key(f"{root}/{key.lstrip('./')}", row)
    return lookup


def validate_noise_alignment(groups: SelectionGroups, truth_by_path: dict[str, dict[str, str]]) -> None:
    for index in groups.pool:
        path = canonical_path(groups.paths[index])
        record = truth_by_path.get(path)
        if record is None:
            raise ValueError(f"Selection sample {index} is absent from its saved noise index: {path}")
        if str(record["web_label"]) != groups.labels[index]:
            raise ValueError(f"Observed-label mismatch for sample {index}: {record['web_label']!r} vs {groups.labels[index]!r}")


def deterministic_class_sample(indices: Iterable[int], labels: dict[int, str], per_class: int, seed: int) -> list[int]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for index in sorted(set(int(value) for value in indices)):
        by_label[labels[index]].append(index)
    chosen: list[int] = []
    for label in sorted(by_label):
        candidates = list(by_label[label])
        random.Random(stable_int(seed, "reference", label)).shuffle(candidates)
        chosen.extend(sorted(candidates[: min(per_class, len(candidates))]))
    return chosen


def deterministic_stratified_sample(indices: Iterable[int], labels: dict[int, str], count: int, seed: int) -> list[int]:
    """Round-robin class coverage without replacement or truth-label access."""
    values = sorted(set(int(value) for value in indices))
    target = min(int(count), len(values))
    if target <= 0:
        return []
    by_label: dict[str, list[int]] = defaultdict(list)
    for index in values:
        by_label[labels[index]].append(index)
    order = sorted(by_label)
    random.Random(seed).shuffle(order)
    for label in order:
        random.Random(stable_int(seed, "query", label)).shuffle(by_label[label])
    cursors = {label: 0 for label in order}
    chosen: list[int] = []
    while len(chosen) < target:
        progressed = False
        for label in order:
            cursor = cursors[label]
            if cursor >= len(by_label[label]):
                continue
            chosen.append(by_label[label][cursor])
            cursors[label] = cursor + 1
            progressed = True
            if len(chosen) == target:
                break
        if not progressed:
            raise RuntimeError("Stratified sampler exhausted unexpectedly.")
    return chosen


def patch_grid_shape(token_count: int, image_size: int, patch_size: int | None) -> tuple[int, int]:
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError(f"Patch-token count {token_count} is not a square grid.")
    if patch_size is not None and side * patch_size != image_size:
        raise ValueError(f"Patch grid {side} with patch size {patch_size} does not match input {image_size}.")
    return side, side


def uniform_patch_positions(grid_shape: tuple[int, int], count: int) -> tuple[int, ...]:
    height, width = grid_shape
    if count < 1 or count > height * width:
        raise ValueError(f"Cannot sample {count} positions from grid {grid_shape}.")
    side = int(math.ceil(math.sqrt(count)))
    ys = np.rint(np.linspace(0, height - 1, side)).astype(int)
    xs = np.rint(np.linspace(0, width - 1, side)).astype(int)
    positions = [int(y * width + x) for y in ys for x in xs]
    return tuple(positions[:count])


def extract_reference_bank(
    torch: Any,
    model: Any,
    dataset: Any,
    index_to_position: dict[int, int],
    reference_indices: Sequence[int],
    labels: dict[int, str],
    *,
    batch_size: int,
    device: str,
    patch_per_image: int,
    input_size: int,
) -> tuple[dict[str, list[ReferenceImage]], tuple[int, int], int, int]:
    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, [index_to_position[index] for index in reference_indices])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.startswith("cuda"))
    bank: dict[str, list[ReferenceImage]] = defaultdict(list)
    grid: tuple[int, int] | None = None
    feature_dim: int | None = None
    patch_size = getattr(model.backbone, "patch_size", None)
    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    was_training = bool(model.training)
    try:
        model.eval()
        with torch.inference_mode():
            for images, _, original_indices in loader:
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                    output = model.backbone.forward_features(images)
                cls = output.get("x_norm_clstoken")
                patches = output.get("x_norm_patchtokens")
                if cls is None or patches is None or cls.ndim != 2 or patches.ndim != 3:
                    raise ValueError("DINOv2 forward_features did not expose [B,D] CLS and [B,N,D] Patch tokens.")
                if cls.shape[0] != patches.shape[0] or cls.shape[1] != patches.shape[2]:
                    raise ValueError("CLS/Patch feature shapes are inconsistent.")
                batch_grid = patch_grid_shape(int(patches.shape[1]), int(images.shape[-1]), int(patch_size) if patch_size else None)
                if grid is None:
                    grid = batch_grid
                    feature_dim = int(cls.shape[1])
                elif grid != batch_grid or feature_dim != int(cls.shape[1]):
                    raise ValueError("Patch grid or feature dimension changed within one checkpoint extraction.")
                positions = uniform_patch_positions(batch_grid, patch_per_image)
                cls_cpu = cls.detach().float().cpu().numpy()
                patch_cpu = patches[:, list(positions)].detach().float().cpu().numpy()
                for offset, original_index in enumerate(original_indices.tolist()):
                    label = labels[int(original_index)]
                    bank[label].append(
                        ReferenceImage(
                            original_index=int(original_index),
                            observed_label=label,
                            cls=normalise_vector(cls_cpu[offset]),
                            patches=normalise_rows(patch_cpu[offset]),
                            patch_positions=positions,
                        )
                    )
    finally:
        model.train(was_training)
    if grid is None or feature_dim is None:
        raise RuntimeError("Reference extraction produced no features.")
    return dict(bank), grid, feature_dim, int(patch_size or 0)


def eligible_reference_images(bank: dict[str, list[ReferenceImage]], label: str, query_index: int) -> list[ReferenceImage]:
    return [record for record in bank.get(label, []) if record.original_index != query_index]


def balanced_patch_banks(
    bank: dict[str, list[ReferenceImage]],
    observed_label: str,
    competitor_label: str,
    query_index: int,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if observed_label == competitor_label:
        raise ValueError("Observed and competitor labels must differ.")
    left = eligible_reference_images(bank, observed_label, query_index)
    right = eligible_reference_images(bank, competitor_label, query_index)
    left_values = np.concatenate([record.patches for record in left], axis=0) if left else np.empty((0, 0), dtype=np.float32)
    right_values = np.concatenate([record.patches for record in right], axis=0) if right else np.empty((0, 0), dtype=np.float32)
    count = min(len(left_values), len(right_values), maximum)
    if count < 1:
        raise ValueError("Unavailable balanced Patch reference bank after leave-one-image-out exclusion.")
    return left_values[:count], right_values[:count], count


def balanced_cls_prototypes(
    bank: dict[str, list[ReferenceImage]],
    observed_label: str,
    competitor_label: str,
    query_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    left = eligible_reference_images(bank, observed_label, query_index)
    right = eligible_reference_images(bank, competitor_label, query_index)
    count = min(len(left), len(right))
    if count < 1:
        raise ValueError("Unavailable balanced CLS reference after leave-one-image-out exclusion.")
    return (
        normalise_vector(np.mean(np.stack([record.cls for record in left[:count]]), axis=0)),
        normalise_vector(np.mean(np.stack([record.cls for record in right[:count]]), axis=0)),
        count,
    )


def local_evidence(patches: np.ndarray, bank_a: np.ndarray, bank_b: np.ndarray, top_fraction: float) -> dict[str, Any]:
    patches = normalise_rows(patches)
    bank_a = normalise_rows(bank_a)
    bank_b = normalise_rows(bank_b)
    if patches.shape[1] != bank_a.shape[1] or patches.shape[1] != bank_b.shape[1]:
        raise ValueError("Query and reference Patch feature dimensions differ.")
    evidence = np.max(patches @ bank_a.T, axis=1) - np.max(patches @ bank_b.T, axis=1)
    if not np.all(np.isfinite(evidence)):
        raise FloatingPointError("Local Patch evidence contains NaN/Inf.")
    top_count = max(1, int(math.ceil(len(evidence) * top_fraction)))
    top_positions = np.argsort(-evidence, kind="mergesort")[:top_count]
    return {
        "evidence": evidence.astype(np.float32),
        "local_margin": float(np.mean(evidence[top_positions])),
        "patch_mean_evidence": float(np.mean(evidence)),
        "top_positions": tuple(int(position) for position in top_positions.tolist()),
        "top_count": top_count,
        "positive_patch_ratio": float(np.mean(evidence > 0.0)),
        "strongest_a_evidence": float(np.max(evidence)),
        "strongest_b_evidence": float(np.min(evidence)),
    }


def random_patch_margin(evidence: np.ndarray, count: int, repeats: int, seed: int) -> tuple[float, float]:
    if count < 1 or count > len(evidence) or repeats < 1:
        raise ValueError("Invalid random-Patch control request.")
    rng = np.random.default_rng(seed)
    scores = [float(np.mean(evidence[rng.choice(len(evidence), size=count, replace=False)])) for _ in range(repeats)]
    return float(np.mean(scores)), float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0


def diagnostic_competitor(cls: np.ndarray, observed_label: str, query_index: int, bank: dict[str, list[ReferenceImage]]) -> str | None:
    candidates: list[tuple[float, str]] = []
    for label in sorted(bank):
        if label == observed_label:
            continue
        records = eligible_reference_images(bank, label, query_index)
        if not records:
            continue
        proto = normalise_vector(np.mean(np.stack([record.cls for record in records]), axis=0))
        candidates.append((float(np.dot(cls, proto)), label))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[1]


def auc_clean_positive(scores: Sequence[float], clean_flags: Sequence[bool]) -> float | None:
    """Mann-Whitney AUROC where positive means synthetic-noise clean."""
    if len(scores) != len(clean_flags) or not scores:
        return None
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(clean_flags, dtype=bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if float(a.std()) <= EPSILON or float(b.std()) <= EPSILON:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def ab_correct(margin: float, observed_label: str, competitor_label: str, true_label: str) -> int | None:
    if true_label not in {observed_label, competitor_label}:
        return None
    prediction = observed_label if margin >= 0.0 else competitor_label
    return int(prediction == true_label)


def score_query(
    descriptor: QueryDescriptor,
    cls: np.ndarray,
    patches: np.ndarray,
    bank: dict[str, list[ReferenceImage]],
    truth: dict[str, str],
    *,
    max_reference_patches: int,
    top_fraction: float,
    random_repeats: int,
    random_seed: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    observed = descriptor.observed_label
    competitor = descriptor.historical_competitor
    base = {
        "original_index": descriptor.original_index,
        "stratum": descriptor.group,
        "observed_label": observed,
        "historical_competitor": competitor,
        "current_diagnostic_competitor": diagnostic_competitor(cls, observed, descriptor.original_index, bank) or "",
    }
    base["historical_current_competitor_match"] = "yes" if base["current_diagnostic_competitor"] == competitor else "no"
    if not competitor or competitor == observed:
        base.update({"available": "no", "unavailable_reason": "missing_or_self_competitor"})
        return base, None
    try:
        cls_a, cls_b, cls_reference_count = balanced_cls_prototypes(bank, observed, competitor, descriptor.original_index)
        patches_a, patches_b, patch_reference_count = balanced_patch_banks(
            bank, observed, competitor, descriptor.original_index, max_reference_patches
        )
    except ValueError as exc:
        base.update({"available": "no", "unavailable_reason": str(exc)})
        return base, None
    cls_margin = float(np.dot(cls, cls_a) - np.dot(cls, cls_b))
    patch_mean = normalise_vector(np.mean(patches, axis=0))
    patch_mean_a = normalise_vector(np.mean(patches_a, axis=0))
    patch_mean_b = normalise_vector(np.mean(patches_b, axis=0))
    patch_mean_margin = float(np.dot(patch_mean, patch_mean_a) - np.dot(patch_mean, patch_mean_b))
    local = local_evidence(patches, patches_a, patches_b, top_fraction)
    random_margin, random_std = random_patch_margin(
        local["evidence"], local["top_count"], random_repeats,
        stable_int(random_seed, descriptor.original_index, descriptor.group, "random_patch"),
    )
    # All geometry is now fixed. Clean/noisy data enter only the post-hoc
    # evaluation fields below, never references, competitors, or Patch ranks.
    true_label = truth["true_label"]
    base.update(truth)
    base.update(
        {
            "available": "yes",
            "unavailable_reason": "",
            "cls_margin": cls_margin,
            "patch_mean_margin": patch_mean_margin,
            "random_patch_margin": random_margin,
            "random_patch_std": random_std,
            "local_margin": local["local_margin"],
            "full_patch_evidence_mean": local["patch_mean_evidence"],
            "strongest_a_patch_evidence": local["strongest_a_evidence"],
            "strongest_b_patch_evidence": local["strongest_b_evidence"],
            "positive_patch_ratio": local["positive_patch_ratio"],
            "top_patch_count": local["top_count"],
            "top_patch_positions": ";".join(str(value) for value in local["top_positions"]),
            "cls_reference_images_per_side": cls_reference_count,
            "patch_reference_patches_per_side": patch_reference_count,
            "ab_candidate_covered": "yes" if true_label in {observed, competitor} else "no",
            "cls_ab_correct": ab_correct(cls_margin, observed, competitor, true_label),
            "patch_mean_ab_correct": ab_correct(patch_mean_margin, observed, competitor, true_label),
            "random_patch_ab_correct": ab_correct(random_margin, observed, competitor, true_label),
            "local_ab_correct": ab_correct(local["local_margin"], observed, competitor, true_label),
        }
    )
    return base, local["evidence"]


def query_descriptors(
    groups: SelectionGroups,
    rows: list[dict[str, str]],
    *,
    count: int,
    seed: int,
) -> list[QueryDescriptor]:
    by_index = {int(row["index"]): row for row in rows}
    output: list[QueryDescriptor] = []
    # Reliable is the strict dual-evidence intersection. Historical fallback,
    # if any, remains an auditable active-training state but must not be
    # promoted into the local reference bank or the Reliable query stratum.
    for group_name, indices in (("reliable", groups.strict_reliable), ("ambiguous_l", groups.ambiguous_l)):
        sampled = deterministic_stratified_sample(indices, groups.labels, count, stable_int(seed, group_name))
        for index in sampled:
            row = by_index[index]
            output.append(QueryDescriptor(index, group_name, groups.labels[index], row["competitor_class"]))
    return output


def extract_and_score_queries(
    torch: Any,
    model: Any,
    dataset: Any,
    index_to_position: dict[int, int],
    descriptors: Sequence[QueryDescriptor],
    bank: dict[str, list[ReferenceImage]],
    truth_by_path: dict[str, dict[str, str]],
    paths: list[str],
    *,
    batch_size: int,
    device: str,
    max_reference_patches: int,
    top_fraction: float,
    random_repeats: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    from torch.utils.data import DataLoader, Subset

    descriptor_by_index = {item.original_index: item for item in descriptors}
    if len(descriptor_by_index) != len(descriptors):
        raise ValueError("Diagnostic query indices are unexpectedly duplicated.")
    subset = Subset(dataset, [index_to_position[index] for index in descriptor_by_index])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.startswith("cuda"))
    rows: list[dict[str, Any]] = []
    evidence_by_index: dict[int, np.ndarray] = {}
    was_training = bool(model.training)
    try:
        model.eval()
        with torch.inference_mode():
            for images, _, original_indices in loader:
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                    output = model.backbone.forward_features(images)
                cls = output.get("x_norm_clstoken")
                patches = output.get("x_norm_patchtokens")
                if cls is None or patches is None:
                    raise ValueError("DINOv2 Patch tokens are unavailable for query extraction.")
                cls_cpu = cls.detach().float().cpu().numpy()
                patch_cpu = patches.detach().float().cpu().numpy()
                for offset, original_index in enumerate(original_indices.tolist()):
                    index = int(original_index)
                    path = paths[index].replace("\\", "/")
                    record = truth_by_path.get(path)
                    if record is None:
                        raise ValueError(f"Missing post-hoc truth record for query {index}.")
                    truth = {
                        "path": paths[index],
                        "true_label": str(record["clean_label"]),
                        "posthoc_is_clean": "yes" if str(record["is_noisy"]) in {"0", "no", "false"} else "no",
                        "posthoc_is_noisy": "yes" if str(record["is_noisy"]) in {"1", "yes", "true"} else "no",
                    }
                    row, evidence = score_query(
                        descriptor_by_index[index], normalise_vector(cls_cpu[offset]), normalise_rows(patch_cpu[offset]),
                        bank, truth, max_reference_patches=max_reference_patches, top_fraction=top_fraction,
                        random_repeats=random_repeats, random_seed=random_seed,
                    )
                    rows.append(row)
                    if evidence is not None:
                        evidence_by_index[index] = evidence
    finally:
        model.train(was_training)
    return rows, evidence_by_index


def attach_dataset_identity(rows: Iterable[dict[str, Any]], dataset: str) -> None:
    """Attach the run identity required by every downstream audit artifact."""
    if not dataset:
        raise ValueError("Dataset identity cannot be empty.")
    for row in rows:
        row["dataset"] = dataset


def mean_or_na(values: Sequence[float]) -> float | str:
    return float(statistics.mean(values)) if values else ""


def accuracy_or_na(values: Sequence[int | None]) -> float | str:
    usable = [int(value) for value in values if value is not None and value != ""]
    return float(statistics.mean(usable)) if usable else ""


def build_summaries(dataset: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    global_rows: list[dict[str, Any]] = []
    ambiguous_rows: list[dict[str, Any]] = []
    competitor_rows: list[dict[str, Any]] = []
    available = [row for row in rows if row["available"] == "yes"]
    for group in ("reliable", "ambiguous_l"):
        subset = [row for row in available if row["stratum"] == group]
        covered = [row for row in subset if row["ab_candidate_covered"] == "yes"]
        methods = ("cls", "patch_mean", "random_patch", "local")
        for method in methods:
            margins = [float(row[f"{method}_margin"]) for row in subset]
            correct = [row[f"{method}_ab_correct"] for row in covered]
            global_rows.append(
                {
                    "dataset": dataset,
                    "stratum": group,
                    "method": method,
                    "query_count": len(subset),
                    "ab_covered_count": len(covered),
                    "ab_coverage_ratio": len(covered) / len(subset) if subset else "",
                    "ab_accuracy": accuracy_or_na(correct),
                    "margin_mean": mean_or_na(margins),
                    "margin_std": float(statistics.stdev(margins)) if len(margins) >= 2 else "",
                }
            )
        local_correct = {int(row["original_index"]): row["local_ab_correct"] for row in covered}
        cls_correct = {int(row["original_index"]): row["cls_ab_correct"] for row in covered}
        global_rows.append(
            {
                "dataset": dataset,
                "stratum": group,
                "method": "local_vs_cls_transition",
                "query_count": len(subset),
                "ab_covered_count": len(covered),
                "ab_coverage_ratio": len(covered) / len(subset) if subset else "",
                "ab_accuracy": "",
                "margin_mean": "",
                "margin_std": "",
                "cls_wrong_local_correct": sum(cls_correct[index] == 0 and local_correct[index] == 1 for index in cls_correct),
                "cls_correct_local_wrong": sum(cls_correct[index] == 1 and local_correct[index] == 0 for index in cls_correct),
            }
        )
    ambiguous = [row for row in available if row["stratum"] == "ambiguous_l"]
    clean_flags = [row["posthoc_is_clean"] == "yes" for row in ambiguous]
    for method in ("cls", "patch_mean", "random_patch", "local"):
        scores = [float(row[f"{method}_margin"]) for row in ambiguous]
        ambiguous_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "positive_class": "clean",
                "query_count": len(ambiguous),
                "clean_count": sum(clean_flags),
                "noisy_count": len(clean_flags) - sum(clean_flags),
                "clean_margin_mean": mean_or_na([score for score, clean in zip(scores, clean_flags) if clean]),
                "noisy_margin_mean": mean_or_na([score for score, clean in zip(scores, clean_flags) if not clean]),
                "clean_positive_auroc": auc_clean_positive(scores, clean_flags),
                "pearson_with_cls_margin": pearson(scores, [float(row["cls_margin"]) for row in ambiguous]),
            }
        )
    noisy = [row for row in ambiguous if row["posthoc_is_noisy"] == "yes"]
    hit = [row for row in noisy if row["true_label"] == row["historical_competitor"]]
    competitor_rows.append(
        {
            "dataset": dataset,
            "stratum": "ambiguous_l_noisy",
            "noisy_query_count": len(noisy),
            "historical_competitor_hit_count": len(hit),
            "historical_competitor_hit_rate": len(hit) / len(noisy) if noisy else "",
            "local_supports_true_competitor_count": sum(float(row["local_margin"]) < 0.0 for row in hit),
            "cls_supports_true_competitor_count": sum(float(row["cls_margin"]) < 0.0 for row in hit),
        }
    )
    return global_rows, ambiguous_rows, competitor_rows


def choose_occlusion_rows(rows: list[dict[str, Any]], labels: dict[int, str], count: int, seed: int) -> list[dict[str, Any]]:
    available = [row for row in rows if row["available"] == "yes"]
    chosen: list[dict[str, Any]] = []
    for group in ("reliable", "ambiguous_l"):
        candidates = [row for row in available if row["stratum"] == group]
        target = min(len(candidates), max(0, count // 2))
        lookup = {int(row["original_index"]): row for row in candidates}
        indices = deterministic_stratified_sample(lookup, labels, target, stable_int(seed, "occlusion", group))
        chosen.extend(lookup[index] for index in indices)
    return chosen


def occlude_patch_cells(torch: Any, image: Any, positions: Sequence[int], grid: tuple[int, int]) -> Any:
    height, width = grid
    image_height, image_width = int(image.shape[-2]), int(image.shape[-1])
    if image_height % height or image_width % width:
        raise ValueError("Image dimensions are not divisible by Patch grid dimensions.")
    block_h, block_w = image_height // height, image_width // width
    result = image.clone()
    for position in positions:
        y, x = divmod(int(position), width)
        result[..., y * block_h : (y + 1) * block_h, x * block_w : (x + 1) * block_w] = 0.0
    return result


def parse_patch_positions(value: str) -> tuple[int, ...]:
    positions = tuple(int(item) for item in str(value).split(";") if item != "")
    if not positions:
        raise ValueError("A selected local-evidence row has no Patch positions.")
    return positions


def cls_margin_and_probabilities(
    torch: Any,
    model: Any,
    image: Any,
    cls_a: np.ndarray,
    cls_b: np.ndarray,
    observed_id: int,
    competitor_id: int,
    *,
    device: str,
) -> tuple[float, float, float]:
    """Forward one image in eval/inference mode; this has no train-state mutation."""
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            features = model.backbone.forward_features(image.unsqueeze(0).to(device, non_blocking=True))
            cls = features.get("x_norm_clstoken")
            if cls is None:
                raise ValueError("DINOv2 forward_features did not return x_norm_clstoken during occlusion.")
            logits = model.head(cls)
        cls_np = normalise_vector(cls.detach().float().cpu().numpy()[0])
        probs = torch.softmax(logits.float(), dim=-1).detach().cpu().numpy()[0]
    return (
        float(np.dot(cls_np, cls_a) - np.dot(cls_np, cls_b)),
        float(probs[observed_id]),
        float(probs[competitor_id]),
    )


def run_occlusion_control(
    torch: Any,
    model: Any,
    dataset: Any,
    index_to_position: dict[int, int],
    rows: list[dict[str, Any]],
    labels: dict[int, str],
    label_to_id: dict[str, int],
    bank: dict[str, list[ReferenceImage]],
    grid: tuple[int, int],
    *,
    count: int,
    seed: int,
    device: str,
) -> list[dict[str, Any]]:
    """Compare masking top local-evidence cells with equally sized random masks."""
    selected = choose_occlusion_rows(rows, labels, count, seed)
    results: list[dict[str, Any]] = []
    was_training = bool(model.training)
    try:
        model.eval()
        for row in selected:
            index = int(row["original_index"])
            observed, competitor = str(row["observed_label"]), str(row["historical_competitor"])
            if observed not in label_to_id or competitor not in label_to_id:
                raise ValueError(f"Occlusion labels are absent from the classifier mapping: {observed!r}, {competitor!r}")
            cls_a, cls_b, _ = balanced_cls_prototypes(bank, observed, competitor, index)
            image, _, returned_index = dataset[index_to_position[index]]
            if int(returned_index) != index:
                raise RuntimeError("Occlusion dataset returned a mismatched original sample index.")
            top_positions = parse_patch_positions(str(row["top_patch_positions"]))
            if any(position < 0 or position >= grid[0] * grid[1] for position in top_positions):
                raise ValueError("Local-evidence Patch positions are outside the runtime grid.")
            rng = np.random.default_rng(stable_int(seed, "occlusion_random", index))
            random_positions = tuple(sorted(int(item) for item in rng.choice(grid[0] * grid[1], size=len(top_positions), replace=False)))
            before_margin, before_observed, before_competitor = cls_margin_and_probabilities(
                torch, model, image, cls_a, cls_b, label_to_id[observed], label_to_id[competitor], device=device
            )
            for name, positions in (("high_local_evidence", top_positions), ("random_equal_area", random_positions)):
                after_margin, after_observed, after_competitor = cls_margin_and_probabilities(
                    torch,
                    model,
                    occlude_patch_cells(torch, image, positions, grid),
                    cls_a,
                    cls_b,
                    label_to_id[observed],
                    label_to_id[competitor],
                    device=device,
                )
                results.append(
                    {
                        "dataset": row["dataset"],
                        "stratum": row["stratum"],
                        "original_index": index,
                        "observed_label": observed,
                        "historical_competitor": competitor,
                        "occlusion_type": name,
                        "cell_count": len(positions),
                        "patch_positions": ";".join(str(value) for value in positions),
                        "cls_margin_before": before_margin,
                        "cls_margin_after": after_margin,
                        "cls_margin_delta_after_minus_before": after_margin - before_margin,
                        "observed_probability_before": before_observed,
                        "observed_probability_after": after_observed,
                        "observed_probability_delta_after_minus_before": after_observed - before_observed,
                        "competitor_probability_before": before_competitor,
                        "competitor_probability_after": after_competitor,
                        "competitor_probability_delta_after_minus_before": after_competitor - before_competitor,
                    }
                )
    finally:
        model.train(was_training)
    return results


def summarise_occlusion(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["stratum"]), str(row["occlusion_type"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, stratum, kind), values in sorted(grouped.items()):
        output.append(
            {
                "dataset": dataset,
                "stratum": stratum,
                "occlusion_type": kind,
                "query_count": len(values),
                "cell_count": values[0]["cell_count"] if len({int(row["cell_count"]) for row in values}) == 1 else "variable",
                "mean_cls_margin_delta_after_minus_before": mean_or_na([float(row["cls_margin_delta_after_minus_before"]) for row in values]),
                "mean_observed_probability_delta_after_minus_before": mean_or_na([float(row["observed_probability_delta_after_minus_before"]) for row in values]),
                "mean_competitor_probability_delta_after_minus_before": mean_or_na([float(row["competitor_probability_delta_after_minus_before"]) for row in values]),
            }
        )
    return output


def save_case_figures(
    output_dir: Path,
    rows: list[dict[str, Any]],
    evidence_by_index: dict[int, np.ndarray],
    paths: list[str],
    path_maps: list[tuple[str, str]],
    grid: tuple[int, int],
) -> list[dict[str, Any]]:
    """Write deterministic success/failure illustrations after all scores are fixed.

    Post-hoc clean/noisy status is only used here to label/select report cases;
    it does not affect references, competitors, scores, or Patch selection.
    """
    from PIL import Image, ImageDraw, ImageFont

    from gcdd.lora_training import resolve_image_path

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    available = [row for row in rows if row.get("available") == "yes" and int(row["original_index"]) in evidence_by_index]
    categories = {
        "reliable_local_correct": lambda row: row["stratum"] == "reliable" and row.get("local_ab_correct") == 1,
        "reliable_local_wrong": lambda row: row["stratum"] == "reliable" and row.get("local_ab_correct") == 0,
        "ambiguous_l_clean_support_observed": lambda row: row["stratum"] == "ambiguous_l" and row["posthoc_is_clean"] == "yes" and float(row["local_margin"]) >= 0.0,
        "ambiguous_l_clean_support_competitor": lambda row: row["stratum"] == "ambiguous_l" and row["posthoc_is_clean"] == "yes" and float(row["local_margin"]) < 0.0,
        "ambiguous_l_noisy_support_competitor": lambda row: row["stratum"] == "ambiguous_l" and row["posthoc_is_noisy"] == "yes" and float(row["local_margin"]) < 0.0,
        "ambiguous_l_noisy_support_observed": lambda row: row["stratum"] == "ambiguous_l" and row["posthoc_is_noisy"] == "yes" and float(row["local_margin"]) >= 0.0,
    }
    font = ImageFont.load_default()
    manifest: list[dict[str, Any]] = []
    for category, predicate in categories.items():
        matches = sorted((row for row in available if predicate(row)), key=lambda row: int(row["original_index"]))
        if not matches:
            manifest.append({"category": category, "status": "unavailable", "reason": "no_matching_query"})
            continue
        row = matches[0]
        index = int(row["original_index"])
        bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        with Image.open(resolve_image_path(paths[index], path_maps)) as source:
            image = source.convert("RGB").resize((448, 448), bilinear)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        evidence = evidence_by_index[index]
        if evidence.size != grid[0] * grid[1]:
            raise ValueError("Cannot visualise evidence because its token count differs from the Patch grid.")
        maximum = max(float(np.max(np.abs(evidence))), EPSILON)
        cell_h, cell_w = 448 // grid[0], 448 // grid[1]
        for position, value in enumerate(evidence.tolist()):
            y, x = divmod(position, grid[1])
            intensity = int(min(220, 35 + 185 * abs(float(value)) / maximum))
            colour = (230, 50, 50, intensity) if value >= 0.0 else (45, 85, 230, intensity)
            draw.rectangle((x * cell_w, y * cell_h, (x + 1) * cell_w, (y + 1) * cell_h), fill=colour)
        for position in parse_patch_positions(str(row["top_patch_positions"])):
            y, x = divmod(position, grid[1])
            draw.rectangle((x * cell_w, y * cell_h, (x + 1) * cell_w - 1, (y + 1) * cell_h - 1), outline=(255, 235, 0, 255), width=2)
        composite = Image.blend(image, overlay, 0.45)
        panel = Image.new("RGB", (448, 516), color="white")
        panel.paste(composite, (0, 68))
        panel_draw = ImageDraw.Draw(panel)
        text = (
            f"{category}; idx={index}; observed={row['observed_label']}; competitor={row['historical_competitor']}\n"
            f"CLS={float(row['cls_margin']):+.3f}; Local={float(row['local_margin']):+.3f}; true(post-hoc)={row['true_label']}"
        )
        panel_draw.multiline_text((5, 4), text, fill="black", font=font, spacing=2)
        file_name = f"{row['dataset']}_{category}_idx{index}.png"
        panel.save(figures_dir / file_name)
        manifest.append({"category": category, "status": "written", "dataset": row["dataset"], "original_index": index, "file": str(Path("figures") / file_name)})
    return manifest


def write_markdown(output_dir: Path, inventory: list[dict[str, Any]], schema: dict[str, Any], global_rows: list[dict[str, Any]], ambiguous_rows: list[dict[str, Any]], competitor_rows: list[dict[str, Any]], occlusion_rows: list[dict[str, Any]]) -> None:
    def table(rows: list[dict[str, Any]], columns: Sequence[str]) -> str:
        if not rows:
            return "Unavailable.\n"
        result = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
        for row in rows:
            result.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
        return "\n".join(result) + "\n"

    report = [
        "# Local Discriminative Evidence Audit for Neighbor-Margin Ranking",
        "",
        "This is a frozen-checkpoint, noisy-training-pool, post-hoc diagnostic. No training or parameter update is performed.",
        "",
        "## Checkpoint inventory",
        "",
        table(inventory, ("dataset", "checkpoint_epoch", "selection_update_epoch", "fallback_count", "noise_index_sha256", "validation_manifest_sha256")),
        "## Feature schema",
        "",
        "```json",
        json.dumps(schema, indent=2, ensure_ascii=False),
        "```",
        "",
        "CLS and Patch tokens are each L2-normalised before cosine comparison. Historical competitors define the primary A/B task; current Reliable-CLS competitors are retained only as a diagnostic agreement field.",
        "Patch references use only historical strict Reliable samples (L ∩ G). A Reliable query removes all of its own reference Patch/CLS features before scoring. Fallback samples are reported separately and are never silently promoted to strict Reliable.",
        "",
        "## Local evidence definition",
        "",
        "For observed class A and historical competitor B, a query Patch has evidence `max cos(h, Q_A) - max cos(h, Q_B)`. `local` is the mean of the fixed top 5% evidence Patches; `random_patch` averages the same number of uniformly selected Patch evidences across fixed random repeats. `patch_mean` averages all query Patches before A/B comparison. All four methods use the same query and A/B pair.",
        "",
        "## Global A/B evidence",
        "",
        table(global_rows, ("dataset", "stratum", "method", "query_count", "ab_covered_count", "ab_coverage_ratio", "ab_accuracy", "margin_mean", "cls_wrong_local_correct", "cls_correct_local_wrong")),
        "## Ambiguous-L clean/noisy post-hoc diagnostic",
        "",
        table(ambiguous_rows, ("dataset", "method", "query_count", "clean_count", "noisy_count", "clean_positive_auroc", "clean_margin_mean", "noisy_margin_mean", "pearson_with_cls_margin")),
        "## Noisy Ambiguous-L competitor coverage",
        "",
        table(competitor_rows, ("dataset", "noisy_query_count", "historical_competitor_hit_count", "historical_competitor_hit_rate", "local_supports_true_competitor_count", "cls_supports_true_competitor_count")),
        "## Occlusion control",
        "",
        table(occlusion_rows, ("dataset", "stratum", "occlusion_type", "query_count", "cell_count", "mean_cls_margin_delta_after_minus_before", "mean_observed_probability_delta_after_minus_before", "mean_competitor_probability_delta_after_minus_before")),
        "## Interpretation boundary",
        "",
        "This pilot is descriptive. Higher A/B accuracy or clean-positive AUROC for local evidence would support further confirmation, but does not prove a local loss will improve Top-1. Conversely, an A/B task is evaluated only when the post-hoc true class is covered by {A, B}; uncovered noisy examples are not misreported as binary errors.",
        "## Limits",
        "",
        "AUROC uses clean=positive and is post-hoc only. The diagnostic does not establish causal training benefit; occlusion is an auxiliary perturbation result and may introduce distribution shift. Heatmaps are generated from already-computed local evidence; their post-hoc labels are annotations only and were not used to select reference features, competitors, or regions.",
    ]
    (output_dir / "local_evidence_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_dataset(
    torch: Any,
    spec: CheckpointSpec,
    args: argparse.Namespace,
    path_maps: list[tuple[str, str]],
    output_dir: Path,
) -> dict[str, Any]:
    from torchvision import transforms

    from gcdd.lora_training import ImageSplitDataset, build_transforms

    audit_inputs = audited_run_inputs(torch, spec)
    cfg = audit_inputs["cfg"]
    rows = audit_inputs["rows"]
    groups = audit_inputs["groups"]
    noise_index = audit_inputs["noise_index"]
    truth_by_path = audit_inputs["truth_by_path"]
    label_to_id = audit_inputs["label_to_id"]
    validation_manifest = audit_inputs["validation_manifest"]
    paths, labels = reconstruct_index_aligned_inputs(groups)
    input_size = int(cfg["feature"]["input_size"])
    _, eval_transform = build_transforms(transforms, input_size)
    reference_indices = deterministic_class_sample(groups.strict_reliable, groups.labels, args.reference_images_per_class, stable_int(args.diagnostic_seed, spec.dataset))
    descriptors = query_descriptors(groups, rows, count=args.queries_per_group, seed=stable_int(args.diagnostic_seed, spec.dataset, "queries"))
    needed = sorted(set(reference_indices) | {item.original_index for item in descriptors})
    dataset = ImageSplitDataset(paths, labels, np.asarray(needed, dtype=np.int64), label_to_id, eval_transform, path_maps)
    position = {index: offset for offset, index in enumerate(needed)}
    model, _ = load_audited_model(torch, spec, cfg, args.device)
    before = snapshot_model_state(model)
    start = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bank, grid, feature_dim, patch_size = extract_reference_bank(
        torch, model, dataset, position, reference_indices, groups.labels,
        batch_size=args.feature_batch_size, device=args.device, patch_per_image=args.reference_patches_per_image, input_size=input_size,
    )
    query_rows, evidence = extract_and_score_queries(
        torch, model, dataset, position, descriptors, bank, truth_by_path, paths,
        batch_size=args.feature_batch_size, device=args.device, max_reference_patches=args.max_reference_patches_per_class,
        top_fraction=args.top_patch_fraction, random_repeats=args.random_patch_repeats, random_seed=args.diagnostic_seed,
    )
    # score_query is deliberately dataset-agnostic; attach the resolved run
    # identity once here so all downstream audit artifacts carry it.
    attach_dataset_identity(query_rows, spec.dataset)
    occlusion_rows = run_occlusion_control(
        torch,
        model,
        dataset,
        position,
        query_rows,
        groups.labels,
        label_to_id,
        bank,
        grid,
        count=args.occlusion_query_count,
        seed=stable_int(args.diagnostic_seed, spec.dataset, "occlusion"),
        device=args.device,
    )
    assert_model_state_unchanged(model, before)
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0
    reference_manifest: list[dict[str, Any]] = []
    for label in sorted(bank):
        for record in bank[label]:
            for patch_position in record.patch_positions:
                reference_manifest.append({"dataset": spec.dataset, "observed_label": label, "original_index": record.original_index, "patch_position": patch_position, "reference_source": "historical_strict_reliable"})
    query_manifest = [
        {"dataset": spec.dataset, "original_index": item.original_index, "stratum": item.group, "observed_label": item.observed_label, "historical_competitor": item.historical_competitor}
        for item in descriptors
    ]
    global_rows, ambiguous_rows, competitor_rows = build_summaries(spec.dataset, query_rows)
    inventory = [{
        "dataset": spec.dataset, "checkpoint": str(spec.checkpoint_path), "checkpoint_epoch": spec.checkpoint_epoch,
        "selection_update_epoch": spec.selection_update_epoch, "fallback_count": len(groups.fallback),
        "noise_index": str(noise_index), "noise_index_sha256": sha256_file(noise_index),
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "training_pool": len(groups.pool), "strict_reliable_count": len(groups.strict_reliable), "actual_active_count": len(groups.actual_active),
        "reference_images": len(reference_indices), "reliable_queries": sum(item.group == "reliable" for item in descriptors),
        "ambiguous_l_queries": sum(item.group == "ambiguous_l" for item in descriptors), "elapsed_seconds": elapsed, "peak_cuda_bytes": peak,
    }]
    schema = {
        "dataset": spec.dataset, "feature_source": "backbone.forward_features after injected LoRA",
        "cls_key": "x_norm_clstoken", "patch_key": "x_norm_patchtokens", "feature_dim": feature_dim,
        "patch_grid": list(grid), "patch_size": patch_size, "input_size": input_size,
        "normalisation": "independent L2 normalisation of CLS vectors and each Patch token",
        "reference_images_per_class": args.reference_images_per_class, "reference_patches_per_image": args.reference_patches_per_image,
        "max_reference_patches_per_class": args.max_reference_patches_per_class, "top_patch_fraction": args.top_patch_fraction,
    }
    figure_manifest = save_case_figures(output_dir, query_rows, evidence, paths, path_maps, grid)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return {
        "inventory": inventory,
        "reference_manifest": reference_manifest,
        "query_manifest": query_manifest,
        "sample_rows": query_rows,
        "global_rows": global_rows,
        "ambiguous_rows": ambiguous_rows,
        "competitor_rows": competitor_rows,
        "occlusion_rows": occlusion_rows,
        "figure_manifest": figure_manifest,
        "schema": schema,
    }


def main() -> None:
    args = parse_args()
    if args.reference_images_per_class < 1 or args.reference_patches_per_image < 1 or args.max_reference_patches_per_class < 1:
        raise ValueError("Reference-bank sizes must be positive.")
    if args.max_reference_patches_per_class < args.reference_patches_per_image:
        raise ValueError("Maximum reference patches per class cannot be smaller than per-image sample count.")
    if not 0.0 < args.top_patch_fraction <= 1.0:
        raise ValueError("--top-patch-fraction must lie in (0, 1].")
    if args.feature_batch_size < 1 or args.queries_per_group < 1 or args.random_patch_repeats < 1 or args.occlusion_query_count < 2:
        raise ValueError("Batch/query/repeat sizes must be positive and --occlusion-query-count must be at least 2.")
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path_maps = parse_path_maps(args.path_map)
    specs = [find_spec(args.variants_root, dataset, args.seed) for dataset in args.datasets]
    if args.dry_run:
        dry_inventory = []
        for spec in specs:
            audit_inputs = audited_run_inputs(torch, spec)
            groups = audit_inputs["groups"]
            rows = audit_inputs["rows"]
            noise_index = audit_inputs["noise_index"]
            validation_manifest = audit_inputs["validation_manifest"]
            dry_inventory.append(
                {
                    "dataset": spec.dataset,
                    "checkpoint": str(spec.checkpoint_path),
                    "checkpoint_epoch": spec.checkpoint_epoch,
                    "selection_update_epoch": spec.selection_update_epoch,
                    "fallback_count": len(groups.fallback),
                    "training_pool": len(groups.pool),
                    "selection_rows": len(rows),
                    "noise_index_sha256": sha256_file(noise_index),
                    "validation_manifest_sha256": sha256_file(validation_manifest),
                    "noise_alignment": "passed",
                }
            )
        write_csv(output_dir / "checkpoint_inventory.csv", dry_inventory)
        (output_dir / "checkpoint_inventory.md").write_text("# Local evidence audit preflight\n\nHistorical selection, noise-index/sample alignment, checkpoint metadata, class mapping, and validation manifest were verified. No model or image was loaded.\n", encoding="utf-8")
        print(f"Dry-run completed: {output_dir}")
        return
    all_inventory: list[dict[str, Any]] = []
    all_reference: list[dict[str, Any]] = []
    all_queries: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    all_global: list[dict[str, Any]] = []
    all_ambiguous: list[dict[str, Any]] = []
    all_competitor: list[dict[str, Any]] = []
    all_occlusion: list[dict[str, Any]] = []
    all_figures: list[dict[str, Any]] = []
    schemas: dict[str, Any] = {}
    for spec in specs:
        artifacts = run_dataset(torch, spec, args, path_maps, output_dir)
        all_inventory.extend(artifacts["inventory"])
        all_reference.extend(artifacts["reference_manifest"])
        all_queries.extend(artifacts["query_manifest"])
        all_samples.extend(artifacts["sample_rows"])
        all_global.extend(artifacts["global_rows"])
        all_ambiguous.extend(artifacts["ambiguous_rows"])
        all_competitor.extend(artifacts["competitor_rows"])
        all_occlusion.extend(artifacts["occlusion_rows"])
        all_figures.extend(artifacts["figure_manifest"])
        schemas[spec.dataset] = artifacts["schema"]
    write_csv(output_dir / "checkpoint_inventory.csv", all_inventory)
    write_csv(output_dir / "reference_bank_manifest.csv", all_reference)
    write_csv(output_dir / "diagnostic_query_manifest.csv", all_queries)
    write_csv(output_dir / "local_evidence_per_sample.csv", all_samples)
    write_csv(output_dir / "global_vs_local_summary.csv", all_global)
    write_csv(output_dir / "ambiguous_clean_noisy_summary.csv", all_ambiguous)
    write_csv(output_dir / "competitor_hit_summary.csv", all_competitor)
    write_csv(output_dir / "occlusion_per_sample.csv", all_occlusion)
    occlusion_summary = summarise_occlusion(all_occlusion)
    write_csv(output_dir / "occlusion_summary.csv", occlusion_summary)
    write_csv(output_dir / "visualization_cases.csv", all_figures)
    inventory_md = ["# Local evidence checkpoint inventory", "", "| Dataset | Checkpoint epoch | Active selection update | Strict Reliable | Actual active | Fallback |", "|---|---:|---:|---:|---:|---:|"]
    for row in all_inventory:
        inventory_md.append(
            f"| {row['dataset']} | {row['checkpoint_epoch']} | {row['selection_update_epoch']} | "
            f"{row['strict_reliable_count']} | {row['actual_active_count']} | {row['fallback_count']} |"
        )
    inventory_md.extend(["", "Checkpoint source paths and hashes are retained in `checkpoint_inventory.csv`."])
    (output_dir / "checkpoint_inventory.md").write_text("\n".join(inventory_md) + "\n", encoding="utf-8")
    (output_dir / "local_feature_schema.md").write_text("# Local feature schema\n\n```json\n" + json.dumps(schemas, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8")
    write_markdown(output_dir, all_inventory, schemas, all_global, all_ambiguous, all_competitor, occlusion_summary)
    print(f"Local discriminative evidence audit completed: {output_dir}")


if __name__ == "__main__":
    main()
