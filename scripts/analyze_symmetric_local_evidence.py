"""Read-only Phase-2 audit for symmetric local Patch-token evidence.

This script deliberately does not modify the Phase-1 local-evidence audit.
It reuses its frozen Margin-Rank checkpoints, historical strict-Reliable
membership, reference bank, and pilot manifest; it only changes how the
already-defined signed Patch evidence is aggregated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    assert_model_state_unchanged,
    load_audited_model,
    parse_path_maps,
    reconstruct_index_aligned_inputs,
    snapshot_model_state,
)
from scripts.analyze_local_discriminative_evidence import (  # noqa: E402
    DEFAULT_VARIANTS_ROOT,
    EPSILON,
    ReferenceImage,
    attach_dataset_identity,
    auc_clean_positive,
    audited_run_inputs,
    balanced_cls_prototypes,
    balanced_patch_banks,
    deterministic_stratified_sample,
    eligible_reference_images,
    extract_reference_bank,
    find_spec,
    history_rows_and_groups,
    normalise_rows,
    normalise_vector,
    patch_grid_shape,
    random_patch_margin,
    stable_int,
    uniform_patch_positions,
    write_csv,
)


DATASETS = ("cub", "aircraft")
PHASE1_ROOT = Path("outputs/analysis/local_discriminative_evidence_audit")
DEFAULT_OUTPUT_DIR = PHASE1_ROOT / "symmetric_local_v2"
TOP_FRACTION = 0.05
METHODS = (
    "cls",
    "patch_mean",
    "random_signed",
    "original_top_positive",
    "all_patch_signed_mean",
    "sle",
)
METHOD_LABELS = {
    "cls": "CLS A/B margin",
    "patch_mean": "Patch mean",
    "random_signed": "Random signed Patch",
    "original_top_positive": "Original Local top-positive",
    "all_patch_signed_mean": "All-Patch signed mean",
    "sle": "Symmetric Local Evidence (top-|e|)",
}
DIRECTION_EPSILON = 1.0e-12


@dataclass(frozen=True)
class V2Query:
    original_index: int
    split: str
    stratum: str
    observed_label: str
    historical_competitor: str


@dataclass(frozen=True)
class SymmetricPatchEvidence:
    s_a: np.ndarray
    s_b: np.ndarray
    evidence: np.ndarray
    top_positive_positions: tuple[int, ...]
    top_absolute_positions: tuple[int, ...]
    top_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Phase-2 symmetric local discriminative-evidence audit."
    )
    parser.add_argument("--variants-root", type=Path, default=DEFAULT_VARIANTS_ROOT)
    parser.add_argument("--phase1-output-dir", type=Path, default=PHASE1_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--independent-seed", type=int, default=20261010)
    parser.add_argument("--independent-queries-per-group", type=int, default=200)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--random-patch-repeats", type=int, default=5)
    parser.add_argument("--phase1-regression-atol", type=float, default=5.0e-4)
    parser.add_argument("--antisymmetry-checks-per-group", type=int, default=4)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifests and metadata only; no image/model load.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def direction(score: float) -> str:
    if score > DIRECTION_EPSILON:
        return "A"
    if score < -DIRECTION_EPSILON:
        return "B"
    return "tie"


def quantile_or_na(values: Sequence[float], quantile: float) -> float | str:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile)) if values else ""


def mean_or_na(values: Sequence[float]) -> float | str:
    return float(statistics.mean(values)) if values else ""


def sample_std_or_na(values: Sequence[float]) -> float | str:
    return float(statistics.stdev(values)) if len(values) >= 2 else ""


def average_rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and array[order[end]] == array[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks, right_ranks = average_rank(left), average_rank(right)
    if float(left_ranks.std()) <= EPSILON or float(right_ranks.std()) <= EPSILON:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def top_k_count(token_count: int, fraction: float = TOP_FRACTION) -> int:
    if token_count < 1 or not 0.0 < fraction <= 1.0:
        raise ValueError("Top-K requires a non-empty token sequence and fraction in (0, 1].")
    return max(1, int(math.ceil(token_count * fraction)))


def patch_competition(patches: np.ndarray, bank_a: np.ndarray, bank_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return max-cosine similarities and their signed A-vs-B difference."""
    patches = normalise_rows(patches)
    bank_a = normalise_rows(bank_a)
    bank_b = normalise_rows(bank_b)
    if patches.shape[1] != bank_a.shape[1] or patches.shape[1] != bank_b.shape[1]:
        raise ValueError("Query and A/B reference Patch features have incompatible dimensions.")
    if len(bank_a) != len(bank_b):
        raise ValueError("A/B Patch reference banks must be cardinality-balanced before max similarity.")
    s_a = np.max(patches @ bank_a.T, axis=1)
    s_b = np.max(patches @ bank_b.T, axis=1)
    evidence = s_a - s_b
    if not (np.all(np.isfinite(s_a)) and np.all(np.isfinite(s_b)) and np.all(np.isfinite(evidence))):
        raise FloatingPointError("Patch competition contains NaN/Inf.")
    return s_a.astype(np.float32), s_b.astype(np.float32), evidence.astype(np.float32)


def symmetric_patch_evidence(
    patches: np.ndarray,
    bank_a: np.ndarray,
    bank_b: np.ndarray,
    fraction: float = TOP_FRACTION,
) -> SymmetricPatchEvidence:
    """Select discriminative Patch tokens by |e| while retaining signed e."""
    s_a, s_b, evidence = patch_competition(patches, bank_a, bank_b)
    count = top_k_count(len(evidence), fraction)
    # mergesort is stable and therefore defines deterministic tie-breaking by
    # the original row-major Patch-token position.
    top_positive = np.argsort(-evidence, kind="mergesort")[:count]
    top_absolute = np.argsort(-np.abs(evidence), kind="mergesort")[:count]
    return SymmetricPatchEvidence(
        s_a=s_a,
        s_b=s_b,
        evidence=evidence,
        top_positive_positions=tuple(int(value) for value in top_positive.tolist()),
        top_absolute_positions=tuple(int(value) for value in top_absolute.tolist()),
        top_count=count,
    )


def signed_score(evidence: np.ndarray, positions: Sequence[int]) -> float:
    if not positions:
        raise ValueError("A signed Patch score requires at least one selected Patch.")
    return float(np.mean(np.asarray(evidence, dtype=np.float32)[list(positions)]))


def evidence_statistics(bundle: SymmetricPatchEvidence) -> dict[str, Any]:
    evidence = bundle.evidence
    selected = evidence[list(bundle.top_absolute_positions)]
    positive_mask = evidence > DIRECTION_EPSILON
    negative_mask = evidence < -DIRECTION_EPSILON
    selected_positive = selected > DIRECTION_EPSILON
    selected_negative = selected < -DIRECTION_EPSILON
    return {
        "positive_patch_count": int(positive_mask.sum()),
        "negative_patch_count": int(negative_mask.sum()),
        "zero_patch_count": int(len(evidence) - positive_mask.sum() - negative_mask.sum()),
        "selected_positive_count": int(selected_positive.sum()),
        "selected_negative_count": int(selected_negative.sum()),
        "selected_zero_count": int(len(selected) - selected_positive.sum() - selected_negative.sum()),
        "selected_positive_evidence_sum": float(selected[selected_positive].sum()) if bool(selected_positive.any()) else 0.0,
        "selected_negative_evidence_sum": float(selected[selected_negative].sum()) if bool(selected_negative.any()) else 0.0,
        "selected_signed_mean": float(selected.mean()),
        "selected_absolute_mean": float(np.abs(selected).mean()),
        "max_positive_evidence": float(evidence.max()),
        "max_negative_evidence": float(evidence.min()),
    }


def method_correct(score: float, observed: str, competitor: str, true_label: str) -> int | None:
    if true_label not in {observed, competitor}:
        return None
    predicted = direction(score)
    return int((predicted == "A" and true_label == observed) or (predicted == "B" and true_label == competitor))


def phase1_paths(root: Path) -> tuple[Path, Path, Path]:
    references = root / "reference_bank_manifest.csv"
    queries = root / "diagnostic_query_manifest.csv"
    sample_scores = root / "local_evidence_per_sample.csv"
    missing = [str(path) for path in (references, queries, sample_scores) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase-1 audit artifacts are incomplete: {missing}")
    return references, queries, sample_scores


def load_phase1_reference_indices(path: Path, dataset: str) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    rows = [row for row in read_csv(path) if row.get("dataset") == dataset]
    if not rows:
        raise ValueError(f"Phase-1 reference bank has no rows for {dataset}.")
    positions: dict[int, set[int]] = defaultdict(set)
    seen_rows: set[tuple[int, int]] = set()
    for row in rows:
        if row.get("reference_source") != "historical_strict_reliable":
            raise ValueError(f"Unexpected Phase-1 reference source: {row.get('reference_source')!r}")
        identity = (int(row["original_index"]), int(row["patch_position"]))
        if identity in seen_rows:
            raise ValueError(f"Duplicate Phase-1 reference Patch row for {dataset}: {identity}")
        seen_rows.add(identity)
        positions[identity[0]].add(identity[1])
    return sorted(positions), {index: tuple(sorted(values)) for index, values in positions.items()}


def load_phase1_pilot_queries(path: Path, dataset: str) -> list[V2Query]:
    rows = [row for row in read_csv(path) if row.get("dataset") == dataset]
    if not rows:
        raise ValueError(f"Phase-1 query manifest has no rows for {dataset}.")
    seen: set[int] = set()
    output: list[V2Query] = []
    for row in rows:
        index = int(row["original_index"])
        if index in seen:
            raise ValueError(f"Duplicate Phase-1 query index for {dataset}: {index}")
        seen.add(index)
        stratum = row["stratum"]
        if stratum not in {"reliable", "ambiguous_l"}:
            raise ValueError(f"Unexpected Phase-1 query stratum: {stratum}")
        output.append(V2Query(index, "pilot", stratum, row["observed_label"], row["historical_competitor"]))
    return output


def verify_pilot_queries_against_history(queries: Sequence[V2Query], rows: list[dict[str, str]], labels: dict[int, str]) -> None:
    history = {int(row["index"]): row for row in rows}
    for query in queries:
        row = history.get(query.original_index)
        if row is None:
            raise ValueError(f"Phase-1 query {query.original_index} is absent from historical selection rows.")
        if labels[query.original_index] != query.observed_label or row["competitor_class"] != query.historical_competitor:
            raise ValueError(f"Phase-1 query {query.original_index} no longer matches its saved selection identity.")


def independent_queries(
    groups: Any,
    rows: list[dict[str, str]],
    pilot: Sequence[V2Query],
    *,
    dataset: str,
    count_per_group: int,
    seed: int,
) -> tuple[list[V2Query], list[dict[str, Any]]]:
    history = {int(row["index"]): row for row in rows}
    pilot_indices = {query.original_index for query in pilot}
    output: list[V2Query] = []
    availability: list[dict[str, Any]] = []
    for stratum, source in (("reliable", groups.strict_reliable), ("ambiguous_l", groups.ambiguous_l)):
        remaining = sorted(set(source) - pilot_indices)
        selected = deterministic_stratified_sample(
            remaining,
            groups.labels,
            count_per_group,
            stable_int(seed, dataset, "independent", stratum),
        )
        availability.append(
            {
                "dataset": dataset,
                "stratum": stratum,
                "pool_size": len(source),
                "pilot_excluded_count": len(set(source) & pilot_indices),
                "remaining_pool_size": len(remaining),
                "requested_count": count_per_group,
                "actual_count": len(selected),
                "shortfall": max(0, count_per_group - len(selected)),
            }
        )
        for index in selected:
            row = history[index]
            output.append(V2Query(index, "independent", stratum, groups.labels[index], row["competitor_class"]))
    independent_indices = {query.original_index for query in output}
    if pilot_indices & independent_indices:
        raise RuntimeError("Independent query manifest overlaps the Phase-1 pilot.")
    return output, availability


def truth_for_index(index: int, paths: list[str], truth_by_path: dict[str, dict[str, str]]) -> dict[str, str]:
    path = paths[index].replace("\\", "/")
    record = truth_by_path.get(path)
    if record is None:
        raise ValueError(f"Missing post-hoc noise-index record for query {index}.")
    is_noisy = str(record["is_noisy"]).strip().lower() in {"1", "yes", "true"}
    return {
        "path": paths[index],
        "true_label": str(record["clean_label"]),
        "posthoc_is_clean": "no" if is_noisy else "yes",
        "posthoc_is_noisy": "yes" if is_noisy else "no",
    }


def score_query(
    query: V2Query,
    cls: np.ndarray,
    patches: np.ndarray,
    bank: dict[str, list[ReferenceImage]],
    truth: dict[str, str],
    *,
    max_reference_patches: int,
    random_repeats: int,
    random_seed: int,
) -> tuple[dict[str, Any], SymmetricPatchEvidence | None]:
    observed, competitor = query.observed_label, query.historical_competitor
    base: dict[str, Any] = {
        "original_index": query.original_index,
        "split": query.split,
        "stratum": query.stratum,
        "observed_label": observed,
        "historical_competitor": competitor,
    }
    if not competitor or competitor == observed:
        base.update({"available": "no", "unavailable_reason": "missing_or_self_competitor"})
        base.update(truth)
        return base, None
    try:
        source_a = eligible_reference_images(bank, observed, query.original_index)
        source_b = eligible_reference_images(bank, competitor, query.original_index)
        cls_a, cls_b, cls_reference_count = balanced_cls_prototypes(bank, observed, competitor, query.original_index)
        patches_a, patches_b, patch_reference_count = balanced_patch_banks(
            bank, observed, competitor, query.original_index, max_reference_patches
        )
    except ValueError as exc:
        base.update({"available": "no", "unavailable_reason": str(exc)})
        base.update(truth)
        return base, None
    cls = normalise_vector(cls)
    patches = normalise_rows(patches)
    bundle = symmetric_patch_evidence(patches, patches_a, patches_b)
    patch_mean = normalise_vector(np.mean(patches, axis=0))
    patch_mean_a = normalise_vector(np.mean(patches_a, axis=0))
    patch_mean_b = normalise_vector(np.mean(patches_b, axis=0))
    random_signed, random_signed_std = random_patch_margin(
        bundle.evidence,
        bundle.top_count,
        random_repeats,
        stable_int(random_seed, query.original_index, query.split, query.stratum, "random_signed"),
    )
    original_top_positive = signed_score(bundle.evidence, bundle.top_positive_positions)
    all_patch_signed = float(np.mean(bundle.evidence))
    sle = signed_score(bundle.evidence, bundle.top_absolute_positions)
    # Post-hoc fields enter only after every reference, similarity, Patch rank,
    # and score above has been fixed without their use.
    true_label = truth["true_label"]
    stats = evidence_statistics(bundle)
    scores = {
        "cls": float(np.dot(cls, cls_a) - np.dot(cls, cls_b)),
        "patch_mean": float(np.dot(patch_mean, patch_mean_a) - np.dot(patch_mean, patch_mean_b)),
        "random_signed": random_signed,
        "original_top_positive": original_top_positive,
        "all_patch_signed_mean": all_patch_signed,
        "sle": sle,
    }
    base.update(truth)
    base.update(
        {
            "available": "yes",
            "unavailable_reason": "",
            "cls_reference_images_per_side": cls_reference_count,
            "available_reference_images_a": len(source_a),
            "available_reference_images_b": len(source_b),
            "available_reference_patches_a": sum(len(record.patches) for record in source_a),
            "available_reference_patches_b": sum(len(record.patches) for record in source_b),
            "patch_reference_patches_per_side": patch_reference_count,
            "top_patch_count": bundle.top_count,
            "top_positive_positions": ";".join(str(value) for value in bundle.top_positive_positions),
            "top_absolute_positions": ";".join(str(value) for value in bundle.top_absolute_positions),
            "ab_candidate_covered": "yes" if true_label in {observed, competitor} else "no",
            **stats,
            "random_signed_std": random_signed_std,
        }
    )
    for method, score in scores.items():
        base[f"{method}_margin"] = score
        base[f"{method}_prediction"] = direction(score)
        base[f"{method}_ab_correct"] = method_correct(score, observed, competitor, true_label)
    return base, bundle


def reference_bank_summary(bank: dict[str, list[ReferenceImage]], dataset: str) -> list[dict[str, Any]]:
    """Validate and describe the fixed Phase-1 Reliable reference source."""
    output: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for label in sorted(bank):
        records = bank[label]
        indices = [record.original_index for record in records]
        if len(indices) != len(set(indices)):
            raise RuntimeError(f"Duplicate reference image within class {label!r}.")
        if seen_indices & set(indices):
            raise RuntimeError("A Phase-1 reference image appears in more than one observed-label bank.")
        seen_indices.update(indices)
        if any(record.observed_label != label for record in records):
            raise RuntimeError(f"Reference bank label mismatch for {label!r}.")
        patch_counts = [len(record.patches) for record in records]
        if not patch_counts or min(patch_counts) < 1:
            raise RuntimeError(f"Reference bank has no Patch tokens for {label!r}.")
        output.append(
            {
                "dataset": dataset,
                "observed_label": label,
                "reference_image_count": len(records),
                "reference_patch_count": sum(patch_counts),
                "patches_per_reference_image_min": min(patch_counts),
                "patches_per_reference_image_max": max(patch_counts),
                "reference_source": "historical_strict_reliable",
            }
        )
    return output


def score_queries(
    torch: Any,
    model: Any,
    dataset: Any,
    index_to_position: dict[int, int],
    queries: Sequence[V2Query],
    bank: dict[str, list[ReferenceImage]],
    paths: list[str],
    truth_by_path: dict[str, dict[str, str]],
    *,
    batch_size: int,
    device: str,
    max_reference_patches: int,
    random_repeats: int,
    random_seed: int,
    capture_patch_indices: set[int],
) -> tuple[list[dict[str, Any]], dict[int, SymmetricPatchEvidence], dict[int, np.ndarray]]:
    from torch.utils.data import DataLoader, Subset

    query_by_index = {query.original_index: query for query in queries}
    if len(query_by_index) != len(queries):
        raise ValueError("Pilot and independent query indices must be globally unique in one run.")
    subset = Subset(dataset, [index_to_position[index] for index in query_by_index])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.startswith("cuda"))
    rows: list[dict[str, Any]] = []
    evidence: dict[int, SymmetricPatchEvidence] = {}
    captured_patches: dict[int, np.ndarray] = {}
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
                    raise ValueError("DINOv2 forward_features did not expose CLS/Patch tokens.")
                cls_cpu = cls.detach().float().cpu().numpy()
                patch_cpu = patches.detach().float().cpu().numpy()
                for offset, raw_index in enumerate(original_indices.tolist()):
                    index = int(raw_index)
                    row, bundle = score_query(
                        query_by_index[index],
                        cls_cpu[offset],
                        patch_cpu[offset],
                        bank,
                        truth_for_index(index, paths, truth_by_path),
                        max_reference_patches=max_reference_patches,
                        random_repeats=random_repeats,
                        random_seed=random_seed,
                    )
                    rows.append(row)
                    if bundle is not None:
                        evidence[index] = bundle
                    if index in capture_patch_indices:
                        captured_patches[index] = normalise_rows(patch_cpu[offset])
    finally:
        model.train(was_training)
    if set(captured_patches) != capture_patch_indices:
        missing = sorted(capture_patch_indices - set(captured_patches))
        raise RuntimeError(f"Could not capture Patch tokens for antisymmetry audit: {missing}")
    return rows, evidence, captured_patches


def antisymmetry_check(
    patches: np.ndarray,
    bank_a: np.ndarray,
    bank_b: np.ndarray,
) -> dict[str, Any]:
    forward = symmetric_patch_evidence(patches, bank_a, bank_b)
    reverse = symmetric_patch_evidence(patches, bank_b, bank_a)
    position_match = forward.top_absolute_positions == reverse.top_absolute_positions
    evidence_error = float(np.max(np.abs(forward.evidence + reverse.evidence)))
    sle_error = abs(
        signed_score(forward.evidence, forward.top_absolute_positions)
        + signed_score(reverse.evidence, reverse.top_absolute_positions)
    )
    if not position_match or evidence_error > 1.0e-5 or sle_error > 1.0e-5:
        raise RuntimeError(
            f"A/B antisymmetry failed: positions_match={position_match}, "
            f"max_e_error={evidence_error}, sle_error={sle_error}"
        )
    return {
        "positions_match": "yes",
        "max_abs_e_plus_swapped_e": evidence_error,
        "abs_sle_plus_swapped_sle": sle_error,
    }


def select_antisymmetry_queries(rows: Sequence[dict[str, Any]], count_per_group: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split in ("pilot", "independent"):
        for stratum in ("reliable", "ambiguous_l"):
            candidates = sorted(
                (row for row in rows if row["split"] == split and row["stratum"] == stratum and row["available"] == "yes"),
                key=lambda row: int(row["original_index"]),
            )
            selected.extend(candidates[:count_per_group])
    return selected


def select_antisymmetry_indices(queries: Sequence[V2Query], count_per_group: int) -> set[int]:
    selected: set[int] = set()
    for split in ("pilot", "independent"):
        for stratum in ("reliable", "ambiguous_l"):
            candidates = sorted(
                query.original_index for query in queries if query.split == split and query.stratum == stratum
            )
            selected.update(candidates[:count_per_group])
    return selected


def method_metrics(
    rows: Sequence[dict[str, Any]],
    dataset: str,
    split: str,
    stratum: str,
    method: str,
) -> dict[str, Any]:
    available = [row for row in rows if row["dataset"] == dataset and row["split"] == split and row["stratum"] == stratum and row["available"] == "yes"]
    eligible = [row for row in available if row["ab_candidate_covered"] == "yes"]
    # Directional support is defined for every available query. A/B
    # correctness is meaningful only when the post-hoc true class is in the
    # fixed candidate pair, so keep their denominators separate.
    scores = [float(row[f"{method}_margin"]) for row in available]
    all_predictions = [direction(value) for value in scores]
    eligible_scores = [float(row[f"{method}_margin"]) for row in eligible]
    eligible_predictions = [direction(value) for value in eligible_scores]
    true_a = [row for row in eligible if row["true_label"] == row["observed_label"]]
    true_b = [row for row in eligible if row["true_label"] == row["historical_competitor"]]
    correct = [int(row[f"{method}_ab_correct"]) for row in eligible]
    recall_a = mean_or_na([int(row[f"{method}_ab_correct"]) for row in true_a])
    recall_b = mean_or_na([int(row[f"{method}_ab_correct"]) for row in true_b])
    balanced = (float(recall_a) + float(recall_b)) / 2.0 if recall_a != "" and recall_b != "" else ""
    return {
        "dataset": dataset,
        "split": split,
        "stratum": stratum,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "query_count": len([row for row in rows if row["dataset"] == dataset and row["split"] == split and row["stratum"] == stratum]),
        "available_count": len(available),
        "eligible_ab_count": len(eligible),
        "candidate_coverage": len(eligible) / len(available) if available else "",
        "true_a_count": len(true_a),
        "true_b_count": len(true_b),
        "true_a_recall": recall_a,
        "true_b_recall": recall_b,
        "ab_accuracy": mean_or_na(correct),
        "balanced_accuracy": balanced,
        "score_positive_count": all_predictions.count("A"),
        "score_negative_count": all_predictions.count("B"),
        "score_tie_count": all_predictions.count("tie"),
        "supports_a_ratio": all_predictions.count("A") / len(available) if available else "",
        "supports_b_ratio": all_predictions.count("B") / len(available) if available else "",
        "score_tie_ratio": all_predictions.count("tie") / len(available) if available else "",
        "eligible_predicted_a_count": eligible_predictions.count("A"),
        "eligible_predicted_b_count": eligible_predictions.count("B"),
        "eligible_tie_count": eligible_predictions.count("tie"),
        "eligible_predicted_a_ratio": eligible_predictions.count("A") / len(eligible) if eligible else "",
        "eligible_predicted_b_ratio": eligible_predictions.count("B") / len(eligible) if eligible else "",
        "eligible_tie_ratio": eligible_predictions.count("tie") / len(eligible) if eligible else "",
        "signed_mean": mean_or_na(scores),
        "signed_median": quantile_or_na(scores, 0.5),
        "signed_q05": quantile_or_na(scores, 0.05),
        "signed_q25": quantile_or_na(scores, 0.25),
        "signed_q75": quantile_or_na(scores, 0.75),
        "signed_q95": quantile_or_na(scores, 0.95),
    }


def prediction_direction_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    identities = sorted({(row["dataset"], row["split"], row["stratum"]) for row in rows})
    for dataset, split, stratum in identities:
        for method in METHODS:
            output.append(method_metrics(rows, dataset, split, stratum, method))
    return output


def ab_classification_summary(direction_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: row[key]
            for key in (
                "dataset", "split", "stratum", "method", "method_label", "available_count", "eligible_ab_count",
                "candidate_coverage", "true_a_count", "true_b_count", "true_a_recall", "true_b_recall", "ab_accuracy",
                "balanced_accuracy", "eligible_predicted_a_count", "eligible_predicted_b_count", "eligible_tie_count",
                "eligible_predicted_a_ratio", "eligible_predicted_b_ratio", "eligible_tie_ratio",
            )
        }
        for row in direction_rows
    ]


def auroc_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, split in sorted({(row["dataset"], row["split"]) for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset and row["split"] == split and row["stratum"] == "ambiguous_l" and row["available"] == "yes"]
        clean_flags = [row["posthoc_is_clean"] == "yes" for row in subset]
        cls_auc = auc_clean_positive([float(row["cls_margin"]) for row in subset], clean_flags)
        for method in METHODS:
            auc = auc_clean_positive([float(row[f"{method}_margin"]) for row in subset], clean_flags)
            output.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "available_count": len(subset),
                    "clean_count": sum(clean_flags),
                    "noisy_count": len(clean_flags) - sum(clean_flags),
                    "clean_positive_auroc": auc if auc is not None else "",
                    "auroc_delta_vs_cls": (auc - cls_auc) if auc is not None and cls_auc is not None else "",
                }
            )
    return output


def competitor_recovery_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, split in sorted({(row["dataset"], row["split"]) for row in rows}):
        ambiguous = [row for row in rows if row["dataset"] == dataset and row["split"] == split and row["stratum"] == "ambiguous_l" and row["available"] == "yes"]
        noisy = [row for row in ambiguous if row["posthoc_is_noisy"] == "yes"]
        true_b = [row for row in noisy if row["true_label"] == row["historical_competitor"]]
        outside = [row for row in noisy if row["true_label"] not in {row["observed_label"], row["historical_competitor"]}]
        for method in METHODS:
            supports_b = [row for row in true_b if row[f"{method}_prediction"] == "B"]
            output.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "noisy_ambiguous_l_count": len(noisy),
                    "true_b_count": len(true_b),
                    "true_outside_ab_count": len(outside),
                    "true_b_supports_competitor_count": len(supports_b),
                    "true_b_supports_competitor_recall": len(supports_b) / len(true_b) if true_b else "",
                }
            )
    return output


def complementarity_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, split, stratum in sorted({(row["dataset"], row["split"], row["stratum"]) for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset and row["split"] == split and row["stratum"] == stratum and row["available"] == "yes" and row["ab_candidate_covered"] == "yes"]
        cls_correct = [int(row["cls_ab_correct"]) for row in subset]
        sle_correct = [int(row["sle_ab_correct"]) for row in subset]
        output.append(
            {
                "dataset": dataset,
                "split": split,
                "stratum": stratum,
                "eligible_ab_count": len(subset),
                "cls_correct_sle_correct": sum(left == 1 and right == 1 for left, right in zip(cls_correct, sle_correct)),
                "cls_correct_sle_wrong": sum(left == 1 and right == 0 for left, right in zip(cls_correct, sle_correct)),
                "cls_wrong_sle_correct": sum(left == 0 and right == 1 for left, right in zip(cls_correct, sle_correct)),
                "cls_wrong_sle_wrong": sum(left == 0 and right == 0 for left, right in zip(cls_correct, sle_correct)),
                "sle_correct_given_cls_wrong": mean_or_na([right for left, right in zip(cls_correct, sle_correct) if left == 0]),
                "sle_wrong_given_cls_correct": mean_or_na([1 - right for left, right in zip(cls_correct, sle_correct) if left == 1]),
                "spearman_cls_sle_margin": spearman(
                    [float(row["cls_margin"]) for row in subset],
                    [float(row["sle_margin"]) for row in subset],
                ),
            }
        )
    return output


def patch_sign_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset, split in sorted({(row["dataset"], row["split"]) for row in rows}):
        available = [row for row in rows if row["dataset"] == dataset and row["split"] == split and row["available"] == "yes"]
        groups = {
            "reliable": [row for row in available if row["stratum"] == "reliable"],
            "ambiguous_l_clean": [row for row in available if row["stratum"] == "ambiguous_l" and row["posthoc_is_clean"] == "yes"],
            "ambiguous_l_noisy": [row for row in available if row["stratum"] == "ambiguous_l" and row["posthoc_is_noisy"] == "yes"],
            "ambiguous_l_true_b_noisy": [
                row for row in available if row["stratum"] == "ambiguous_l" and row["posthoc_is_noisy"] == "yes" and row["true_label"] == row["historical_competitor"]
            ],
        }
        for name, subset in groups.items():
            output.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "posthoc_group": name,
                    "sample_count": len(subset),
                    "mean_positive_patch_count": mean_or_na([float(row["positive_patch_count"]) for row in subset]),
                    "mean_negative_patch_count": mean_or_na([float(row["negative_patch_count"]) for row in subset]),
                    "mean_selected_positive_count": mean_or_na([float(row["selected_positive_count"]) for row in subset]),
                    "mean_selected_negative_count": mean_or_na([float(row["selected_negative_count"]) for row in subset]),
                    "mean_selected_positive_evidence_sum": mean_or_na([float(row["selected_positive_evidence_sum"]) for row in subset]),
                    "mean_selected_negative_evidence_sum": mean_or_na([float(row["selected_negative_evidence_sum"]) for row in subset]),
                    "mean_selected_signed_mean": mean_or_na([float(row["selected_signed_mean"]) for row in subset]),
                    "mean_selected_absolute_mean": mean_or_na([float(row["selected_absolute_mean"]) for row in subset]),
                    "mean_max_positive_evidence": mean_or_na([float(row["max_positive_evidence"]) for row in subset]),
                    "mean_max_negative_evidence": mean_or_na([float(row["max_negative_evidence"]) for row in subset]),
                }
            )
    return output


def compare_phase1_pilot(rows: Sequence[dict[str, Any]], phase1_scores_path: Path, dataset: str, tolerance: float) -> list[dict[str, Any]]:
    previous = {
        int(row["original_index"]): row
        for row in read_csv(phase1_scores_path)
        if row.get("dataset") == dataset and row.get("available") == "yes"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        if row["dataset"] != dataset or row["split"] != "pilot" or row["available"] != "yes":
            continue
        previous_row = previous.get(int(row["original_index"]))
        if previous_row is None:
            raise ValueError(f"Pilot query {row['original_index']} is absent from Phase-1 scores.")
        local_difference = abs(float(row["original_top_positive_margin"]) - float(previous_row["local_margin"]))
        if local_difference > tolerance:
            raise RuntimeError(
                f"Phase-1 regression mismatch for {dataset} index={row['original_index']}: "
                f"{local_difference} > {tolerance}"
            )
        output.append(
            {
                "dataset": dataset,
                "original_index": row["original_index"],
                "phase1_local_margin": previous_row["local_margin"],
                "v2_original_top_positive_margin": row["original_top_positive_margin"],
                "absolute_difference": local_difference,
                "status": "passed",
            }
        )
    return output


def save_case_figures(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    evidence_by_index: dict[int, SymmetricPatchEvidence],
    paths: list[str],
    path_maps: list[tuple[str, str]],
    grid: tuple[int, int],
) -> list[dict[str, Any]]:
    """Save both successes and failures after all scores are fixed."""
    from PIL import Image, ImageDraw, ImageFont

    from gcdd.lora_training import resolve_image_path

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    categories = {
        "cls_correct_sle_correct": lambda row: row["cls_ab_correct"] == 1 and row["sle_ab_correct"] == 1,
        "cls_correct_sle_wrong": lambda row: row["cls_ab_correct"] == 1 and row["sle_ab_correct"] == 0,
        "cls_wrong_sle_correct": lambda row: row["cls_ab_correct"] == 0 and row["sle_ab_correct"] == 1,
        "cls_wrong_sle_wrong": lambda row: row["cls_ab_correct"] == 0 and row["sle_ab_correct"] == 0,
    }
    font = ImageFont.load_default()
    manifest: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in rows}):
        candidates = [
            row for row in rows
            if row["dataset"] == dataset and row["split"] == "independent" and row["available"] == "yes"
            and row["ab_candidate_covered"] == "yes" and int(row["original_index"]) in evidence_by_index
        ]
        for category, predicate in categories.items():
            matches = sorted((row for row in candidates if predicate(row)), key=lambda row: int(row["original_index"]))
            if not matches:
                manifest.append({"dataset": dataset, "category": category, "status": "unavailable", "reason": "no_matching_independent_query"})
                continue
            row = matches[0]
            index = int(row["original_index"])
            bundle = evidence_by_index[index]
            bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            with Image.open(resolve_image_path(paths[index], path_maps)) as source:
                original = source.convert("RGB").resize((448, 448), bilinear)
            overlay = original.copy()
            draw = ImageDraw.Draw(overlay, "RGBA")
            magnitude = max(float(np.abs(bundle.evidence).max()), EPSILON)
            cell_h, cell_w = 448 // grid[0], 448 // grid[1]
            for position, value in enumerate(bundle.evidence.tolist()):
                y, x = divmod(position, grid[1])
                intensity = int(35 + 185 * abs(float(value)) / magnitude)
                colour = (220, 45, 45, intensity) if value > 0.0 else (40, 85, 220, intensity) if value < 0.0 else (230, 230, 230, 25)
                draw.rectangle((x * cell_w, y * cell_h, (x + 1) * cell_w, (y + 1) * cell_h), fill=colour)
            for position in bundle.top_absolute_positions:
                y, x = divmod(position, grid[1])
                draw.rectangle((x * cell_w, y * cell_h, (x + 1) * cell_w - 1, (y + 1) * cell_h - 1), outline=(255, 235, 0, 255), width=2)
            composite = Image.blend(original, overlay, 0.45)
            panel = Image.new("RGB", (448, 520), "white")
            panel.paste(composite, (0, 72))
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.multiline_text(
                (5, 4),
                f"{category}; split=independent; idx={index}; A={row['observed_label']}; B={row['historical_competitor']}\n"
                f"CLS={float(row['cls_margin']):+.3f}; SLE={float(row['sle_margin']):+.3f}; true(post-hoc)={row['true_label']}\n"
                "red=A-support, blue=B-support, yellow=top-|e| Patch",
                fill="black",
                font=font,
                spacing=2,
            )
            name = f"{dataset}_{category}_idx{index}.png"
            panel.save(figures / name)
            manifest.append({"dataset": dataset, "category": category, "status": "written", "original_index": index, "file": str(Path("figures") / name)})
    return manifest


def table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "Unavailable.\n"
    output = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(output) + "\n"


def write_preflight_report(output_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Symmetric Local Evidence v2 preflight",
        "",
        "## Phase-1 reuse audit",
        "",
        table(rows, ("dataset", "checkpoint_epoch", "selection_update_epoch", "strict_reliable_count", "fallback_count", "phase1_pilot_count", "phase1_reference_images", "saved_per_patch_scores", "reference_query_self_match", "a_b_reference_balancing")),
        "Phase 1 persisted only aggregate margins, selected top-positive positions, and reference/query manifests; it did not persist per-Patch s_A, s_B, or e. v2 therefore re-extracts frozen Patch tokens from the same checkpoint. The historical competitor from the saved selection row remains the primary A/B competitor.",
        "",
        "Reliable references are read from Phase-1's `historical_strict_reliable` manifest. The v2 code validates original sample identity, observed label, historical competitor, reference source, uniform Patch positions, and leave-one-image-out scoring.",
    ]
    (output_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_implementation_diff(output_dir: Path) -> None:
    text = """# Phase-2 implementation difference from Phase 1

Phase 1 and v2 use the same frozen Margin-Rank checkpoint, historical strict-Reliable reference source, reference image identities, reference Patch positions, preprocessing, observed class A, historical competitor B, and query identity for the pilot regression set.

The only scoring addition is the aggregation of the same signed per-Patch evidence `e = s_A - s_B`:

- `original_top_positive`: Phase-1 top-5% by descending `e` (kept as a regression control).
- `all_patch_signed_mean`: mean over all signed `e`.
- `random_signed`: mean over a fixed-size random Patch set of the same size as Top-K.
- `sle`: select the fixed top-5% by descending `|e|` with stable Patch-index tie-breaking, then average the original signed `e` values.

No clean/noisy or true-class field is read until after references, competitors, Patch similarities, rankings, and scores are fixed. No model parameter, buffer, checkpoint, selection CSV, noise index, or validation manifest is modified.
"""
    (output_dir / "implementation_diff.md").write_text(text, encoding="utf-8")


def write_final_report(
    output_dir: Path,
    preflight: Sequence[dict[str, Any]],
    direction_rows: Sequence[dict[str, Any]],
    auc_rows: Sequence[dict[str, Any]],
    recovery_rows: Sequence[dict[str, Any]],
    complement_rows: Sequence[dict[str, Any]],
    patch_rows: Sequence[dict[str, Any]],
) -> None:
    pilot_direction = [row for row in direction_rows if row["split"] == "pilot"]
    independent_direction = [row for row in direction_rows if row["split"] == "independent"]
    report = [
        "# Symmetric Local Discriminative Evidence Audit — Phase 2",
        "",
        "This is a frozen-checkpoint, noisy-training-pool post-hoc diagnostic. It does not run backward, an optimizer, a scheduler, or any training epoch.",
        "",
        "## Preflight",
        "",
        table(preflight, ("dataset", "checkpoint_epoch", "selection_update_epoch", "strict_reliable_count", "fallback_count", "phase1_pilot_count", "phase1_reference_images", "saved_per_patch_scores")),
        "## Pilot: prediction direction and A/B classification",
        "",
        table(pilot_direction, ("dataset", "stratum", "method", "available_count", "score_positive_count", "score_negative_count", "score_tie_count", "supports_a_ratio", "supports_b_ratio", "eligible_ab_count", "true_a_recall", "true_b_recall", "ab_accuracy", "balanced_accuracy")),
        "## Independent confirmation: prediction direction and A/B classification",
        "",
        table(independent_direction, ("dataset", "stratum", "method", "available_count", "score_positive_count", "score_negative_count", "score_tie_count", "supports_a_ratio", "supports_b_ratio", "eligible_ab_count", "true_a_recall", "true_b_recall", "ab_accuracy", "balanced_accuracy")),
        "## Ambiguous-L clean/noisy AUROC",
        "",
        table(auc_rows, ("dataset", "split", "method", "available_count", "clean_count", "noisy_count", "clean_positive_auroc", "auroc_delta_vs_cls")),
        "## Noisy Ambiguous-L true-B recovery",
        "",
        table(recovery_rows, ("dataset", "split", "method", "noisy_ambiguous_l_count", "true_b_count", "true_outside_ab_count", "true_b_supports_competitor_count", "true_b_supports_competitor_recall")),
        "## CLS/SLE complementarity",
        "",
        table(complement_rows, ("dataset", "split", "stratum", "eligible_ab_count", "cls_correct_sle_correct", "cls_correct_sle_wrong", "cls_wrong_sle_correct", "cls_wrong_sle_wrong", "sle_correct_given_cls_wrong", "sle_wrong_given_cls_correct", "spearman_cls_sle_margin")),
        "## Patch-sign distribution",
        "",
        table(patch_rows, ("dataset", "split", "posthoc_group", "sample_count", "mean_positive_patch_count", "mean_negative_patch_count", "mean_selected_positive_count", "mean_selected_negative_count", "mean_selected_signed_mean", "mean_selected_absolute_mean")),
        "## Interpretation boundary",
        "",
        "Balanced accuracy and true-B recall are required because an always-observed predictor can have high ordinary accuracy. SLE is only a diagnostic score; it does not establish that a local loss will improve test accuracy. The independent manifest is disjoint from the Phase-1 pilot but remains one seed per dataset, not a cross-seed claim.",
    ]
    (output_dir / "symmetric_local_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def run_dataset(torch: Any, spec: Any, args: argparse.Namespace, path_maps: list[tuple[str, str]]) -> dict[str, Any]:
    from torchvision import transforms

    from gcdd.lora_training import ImageSplitDataset, build_transforms

    audit = audited_run_inputs(torch, spec)
    cfg, groups, history_rows = audit["cfg"], audit["groups"], audit["rows"]
    reference_path, pilot_path, phase1_scores_path = phase1_paths(args.phase1_output_dir)
    reference_indices, saved_positions = load_phase1_reference_indices(reference_path, spec.dataset)
    pilot = load_phase1_pilot_queries(pilot_path, spec.dataset)
    verify_pilot_queries_against_history(pilot, history_rows, groups.labels)
    independent, independent_availability = independent_queries(
        groups,
        history_rows,
        pilot,
        dataset=spec.dataset,
        count_per_group=args.independent_queries_per_group,
        seed=args.independent_seed,
    )
    if not set(reference_indices).issubset(groups.strict_reliable):
        raise RuntimeError("Phase-1 reference manifest includes a non-strict-Reliable sample.")
    all_queries = pilot + independent
    if len({query.original_index for query in all_queries}) != len(all_queries):
        raise RuntimeError("Pilot and independent query manifests overlap.")
    paths, labels = reconstruct_index_aligned_inputs(groups)
    label_to_id = audit["label_to_id"]
    needed = sorted(set(reference_indices) | {query.original_index for query in all_queries})
    input_size = int(cfg["feature"]["input_size"])
    _, eval_transform = build_transforms(transforms, input_size)
    dataset = ImageSplitDataset(paths, labels, np.asarray(needed, dtype=np.int64), label_to_id, eval_transform, path_maps)
    index_to_position = {index: position for position, index in enumerate(needed)}
    model, _ = load_audited_model(torch, spec, cfg, args.device)
    before = snapshot_model_state(model)
    start = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bank, grid, feature_dim, patch_size = extract_reference_bank(
        torch,
        model,
        dataset,
        index_to_position,
        reference_indices,
        groups.labels,
        batch_size=args.feature_batch_size,
        device=args.device,
        patch_per_image=len(next(iter(saved_positions.values()))),
        input_size=input_size,
    )
    expected_positions = uniform_patch_positions(grid, len(next(iter(saved_positions.values()))))
    for label, records in bank.items():
        for record in records:
            if record.patch_positions != expected_positions or saved_positions[record.original_index] != expected_positions:
                raise RuntimeError("v2 reference Patch positions do not exactly match the Phase-1 manifest.")
            if record.observed_label != label:
                raise RuntimeError("Reference-bank class identity mismatch.")
    bank_summary = reference_bank_summary(bank, spec.dataset)
    max_reference_patches = max(sum(len(record.patches) for record in records) for records in bank.values())
    antisymmetry_indices = select_antisymmetry_indices(all_queries, args.antisymmetry_checks_per_group)
    rows, evidence_by_index, captured_patches = score_queries(
        torch,
        model,
        dataset,
        index_to_position,
        all_queries,
        bank,
        paths,
        audit["truth_by_path"],
        batch_size=args.feature_batch_size,
        device=args.device,
        max_reference_patches=max_reference_patches,
        random_repeats=args.random_patch_repeats,
        random_seed=20261009,  # Phase-1 diagnostic seed for paired random control.
        capture_patch_indices=antisymmetry_indices,
    )
    attach_dataset_identity(rows, spec.dataset)
    for row in rows:
        row["seed"] = spec.seed
        row["checkpoint_epoch"] = spec.checkpoint_epoch
        row["selection_update_epoch"] = spec.selection_update_epoch
    antisymmetry_rows: list[dict[str, Any]] = []
    for row in select_antisymmetry_queries(rows, args.antisymmetry_checks_per_group):
        index = int(row["original_index"])
        left, right, _ = balanced_patch_banks(
            bank, row["observed_label"], row["historical_competitor"], index,
            max_reference_patches,
        )
        result = antisymmetry_check(
            captured_patches[index],
            left,
            right,
        )
        antisymmetry_rows.append({"dataset": spec.dataset, "split": row["split"], "stratum": row["stratum"], "original_index": index, **result})
    # The checks above use actual frozen query tokens with A/B reference banks
    # swapped only. Unit tests additionally exercise all-positive, all-negative,
    # mixed, and all-zero synthetic evidence cases.
    assert_model_state_unchanged(model, before)
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else 0
    phase1_regression = compare_phase1_pilot(rows, phase1_scores_path, spec.dataset, args.phase1_regression_atol)
    pilot_manifest = [
        {"dataset": spec.dataset, "seed": spec.seed, "checkpoint_epoch": spec.checkpoint_epoch, "selection_update_epoch": spec.selection_update_epoch, "original_index": query.original_index, "stratum": query.stratum, "observed_label": query.observed_label, "historical_competitor": query.historical_competitor}
        for query in pilot
    ]
    independent_manifest = [
        {"dataset": spec.dataset, "seed": spec.seed, "checkpoint_epoch": spec.checkpoint_epoch, "selection_update_epoch": spec.selection_update_epoch, "original_index": query.original_index, "stratum": query.stratum, "observed_label": query.observed_label, "historical_competitor": query.historical_competitor, "diagnostic_seed": args.independent_seed}
        for query in independent
    ]
    preflight = {
        "dataset": spec.dataset,
        "seed": spec.seed,
        "checkpoint": str(spec.checkpoint_path),
        "checkpoint_epoch": spec.checkpoint_epoch,
        "selection_update_epoch": spec.selection_update_epoch,
        "strict_reliable_count": len(groups.strict_reliable),
        "actual_active_count": len(groups.actual_active),
        "fallback_count": len(groups.fallback),
        "phase1_pilot_count": len(pilot),
        "phase1_reference_images": len(reference_indices),
        "saved_per_patch_scores": "not_saved_frozen_forward_recomputed",
        "reference_query_self_match": "leave_one_image_out_verified",
        "a_b_reference_balancing": "equal_patch_count_per_side",
        "feature_dim": feature_dim,
        "patch_grid": f"{grid[0]}x{grid[1]}",
        "patch_size": patch_size,
        "input_size": input_size,
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": peak,
    }
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return {
        "preflight": preflight,
        "pilot_manifest": pilot_manifest,
        "independent_manifest": independent_manifest,
        "independent_availability": independent_availability,
        "rows": rows,
        "evidence": evidence_by_index,
        "paths": paths,
        "grid": grid,
        "phase1_regression": phase1_regression,
        "antisymmetry": antisymmetry_rows,
        "reference_bank_summary": bank_summary,
    }


def main() -> None:
    args = parse_args()
    if args.independent_queries_per_group < 1 or args.feature_batch_size < 1 or args.random_patch_repeats < 1 or args.antisymmetry_checks_per_group < 1:
        raise ValueError("Independent query, batch, random-repeat, and antisymmetry sizes must be positive.")
    if args.phase1_regression_atol < 0.0:
        raise ValueError("--phase1-regression-atol must be non-negative.")
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path_maps = parse_path_maps(args.path_map)
    specs = [find_spec(args.variants_root, dataset, args.seed) for dataset in args.datasets]
    if args.dry_run:
        preflight_rows: list[dict[str, Any]] = []
        all_availability: list[dict[str, Any]] = []
        for spec in specs:
            audit = audited_run_inputs(torch, spec)
            references, queries, _ = phase1_paths(args.phase1_output_dir)
            reference_indices, _ = load_phase1_reference_indices(references, spec.dataset)
            pilot = load_phase1_pilot_queries(queries, spec.dataset)
            verify_pilot_queries_against_history(pilot, audit["rows"], audit["groups"].labels)
            independent, availability = independent_queries(audit["groups"], audit["rows"], pilot, dataset=spec.dataset, count_per_group=args.independent_queries_per_group, seed=args.independent_seed)
            preflight_rows.append({
                "dataset": spec.dataset, "seed": spec.seed, "checkpoint": str(spec.checkpoint_path), "checkpoint_epoch": spec.checkpoint_epoch,
                "selection_update_epoch": spec.selection_update_epoch, "strict_reliable_count": len(audit["groups"].strict_reliable),
                "actual_active_count": len(audit["groups"].actual_active), "fallback_count": len(audit["groups"].fallback),
                "phase1_pilot_count": len(pilot), "phase1_reference_images": len(reference_indices), "independent_query_count": len(independent),
                "saved_per_patch_scores": "not_saved_frozen_forward_recomputed", "reference_query_self_match": "will_use_leave_one_image_out", "a_b_reference_balancing": "will_equalize_patch_counts",
            })
            all_availability.extend(availability)
        write_csv(output_dir / "preflight_inventory.csv", preflight_rows)
        write_csv(output_dir / "independent_query_availability.csv", all_availability)
        write_preflight_report(output_dir, preflight_rows)
        write_implementation_diff(output_dir)
        print(f"Symmetric local v2 dry-run completed: {output_dir}")
        return
    all_preflight: list[dict[str, Any]] = []
    all_pilot: list[dict[str, Any]] = []
    all_independent: list[dict[str, Any]] = []
    all_availability: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_regression: list[dict[str, Any]] = []
    all_antisymmetry: list[dict[str, Any]] = []
    all_reference_bank_summary: list[dict[str, Any]] = []
    all_figures: list[dict[str, Any]] = []
    for spec in specs:
        artifacts = run_dataset(torch, spec, args, path_maps)
        all_preflight.append(artifacts["preflight"])
        all_pilot.extend(artifacts["pilot_manifest"])
        all_independent.extend(artifacts["independent_manifest"])
        all_availability.extend(artifacts["independent_availability"])
        all_rows.extend(artifacts["rows"])
        all_regression.extend(artifacts["phase1_regression"])
        all_antisymmetry.extend(artifacts["antisymmetry"])
        all_reference_bank_summary.extend(artifacts["reference_bank_summary"])
        all_figures.extend(save_case_figures(output_dir, artifacts["rows"], artifacts["evidence"], artifacts["paths"], path_maps, artifacts["grid"]))
    pilot_rows = [row for row in all_rows if row["split"] == "pilot"]
    independent_rows = [row for row in all_rows if row["split"] == "independent"]
    directions = prediction_direction_summary(all_rows)
    classifications = ab_classification_summary(directions)
    aucs = auroc_summary(all_rows)
    recovery = competitor_recovery_summary(all_rows)
    complementarity = complementarity_summary(all_rows)
    signs = patch_sign_summary(all_rows)
    write_csv(output_dir / "preflight_inventory.csv", all_preflight)
    write_preflight_report(output_dir, all_preflight)
    write_implementation_diff(output_dir)
    write_csv(output_dir / "pilot_query_manifest.csv", all_pilot)
    write_csv(output_dir / "independent_query_manifest.csv", all_independent)
    write_csv(output_dir / "independent_query_availability.csv", all_availability)
    write_csv(output_dir / "local_evidence_per_sample_pilot.csv", pilot_rows)
    write_csv(output_dir / "local_evidence_per_sample_independent.csv", independent_rows)
    write_csv(output_dir / "phase1_regression.csv", all_regression)
    write_csv(output_dir / "antisymmetry_checks.csv", all_antisymmetry)
    write_csv(output_dir / "reference_bank_balance_summary.csv", all_reference_bank_summary)
    (output_dir / "antisymmetry_test_report.md").write_text(
        "# A/B antisymmetry checks\n\n"
        "For each checked frozen-query/reference pair, the script recomputed the score after swapping only A/B references. "
        "All rows in `antisymmetry_checks.csv` must have identical top-|e| positions and near-zero `e(A,B)+e(B,A)` and `SLE(A,B)+SLE(B,A)`.\n\n"
        + table(all_antisymmetry, ("dataset", "split", "stratum", "original_index", "positions_match", "max_abs_e_plus_swapped_e", "abs_sle_plus_swapped_sle")),
        encoding="utf-8",
    )
    write_csv(output_dir / "prediction_direction_summary.csv", directions)
    write_csv(output_dir / "ab_classification_summary.csv", classifications)
    write_csv(output_dir / "ambiguous_clean_noisy_auroc.csv", aucs)
    write_csv(output_dir / "competitor_recovery_summary.csv", recovery)
    write_csv(output_dir / "cls_local_complementarity.csv", complementarity)
    write_csv(output_dir / "patch_sign_distribution.csv", signs)
    write_csv(output_dir / "visualization_cases.csv", all_figures)
    write_final_report(output_dir, all_preflight, directions, aucs, recovery, complementarity, signs)
    print(f"Symmetric local v2 audit completed: {output_dir}")


if __name__ == "__main__":
    main()
