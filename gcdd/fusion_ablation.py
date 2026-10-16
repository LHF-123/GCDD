"""Deterministic, budget-matched PGDF fusion ablations.

The submitted PGDF strict-intersection implementation remains in
``gcdd.lora_dynamic``.  This module only implements the three new comparison
rules.  It consumes the loss/prototype arrays already computed by the training
loop; it never invokes a model, reads oracle labels, or inspects test metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FUSION_MODES = frozenset({"weighted_ranking", "union_relaxed", "soft_weighting"})
SOFT_WEIGHT_FLOOR = 1.0e-6


@dataclass(frozen=True)
class FusionSelection:
    """One class-budgeted fusion decision derived from already available scores."""

    final_selected_mask: np.ndarray
    sample_weights: np.ndarray | None
    sample_rows: list[dict[str, Any]]
    per_class_rows: list[dict[str, Any]]
    update_row: dict[str, Any]


def _validate_inputs(
    mode: str,
    losses: np.ndarray,
    prototype_scores: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    loss_candidate_mask: np.ndarray,
    prototype_candidate_mask: np.ndarray,
    reference_budgets: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mode not in FUSION_MODES:
        raise ValueError(f"Unsupported fusion mode {mode!r}; expected one of {sorted(FUSION_MODES)}")
    labels = np.asarray(labels).astype(str)
    candidate = np.asarray(candidate_mask, dtype=bool)
    losses = np.asarray(losses, dtype=np.float64)
    proto = np.asarray(prototype_scores, dtype=np.float64)
    loss_candidate = np.asarray(loss_candidate_mask, dtype=bool)
    proto_candidate = np.asarray(prototype_candidate_mask, dtype=bool)
    expected_shape = labels.shape
    if not (
        losses.shape == proto.shape == candidate.shape == loss_candidate.shape == proto_candidate.shape == expected_shape
    ):
        raise ValueError("Fusion arrays must all align with the full training-pool labels.")
    if np.any(~np.isfinite(losses[candidate])) or np.any(~np.isfinite(proto[candidate])):
        raise ValueError("Fusion requires finite loss and prototype scores for every candidate.")
    if np.any(loss_candidate & ~candidate) or np.any(proto_candidate & ~candidate):
        raise ValueError("Candidate gate masks must be subsets of candidate_mask.")
    expected_labels = {str(label) for label in labels[candidate]}
    normalized_budgets = {str(label): int(count) for label, count in reference_budgets.items()}
    if set(normalized_budgets) != expected_labels:
        raise ValueError(
            "Reference budget classes do not match current observed-label classes: "
            f"expected={sorted(expected_labels)}, found={sorted(normalized_budgets)}"
        )
    for label, budget in normalized_budgets.items():
        class_count = int(np.sum(candidate & (labels == label)))
        if not 1 <= budget <= class_count:
            raise ValueError(f"Invalid reference budget for class {label!r}: {budget} of {class_count}")
    return losses, proto, labels, candidate, loss_candidate, proto_candidate


def _reliability_from_rank(values: np.ndarray, indices: np.ndarray, *, high_is_reliable: bool) -> np.ndarray:
    """Return deterministic within-class percentile reliability in [0, 1].

    Value ties are resolved by original full-pool index.  This makes the rank
    itself deterministic rather than relying on a platform sort implementation.
    """
    if len(indices) == 0:
        return np.asarray([], dtype=np.float64)
    primary = -values if high_is_reliable else values
    order = np.lexsort((indices, primary))
    scores = np.empty(len(indices), dtype=np.float64)
    if len(indices) == 1:
        scores[order[0]] = 1.0
    else:
        scores[order] = 1.0 - np.arange(len(indices), dtype=np.float64) / float(len(indices) - 1)
    return scores


def _ordered_by_score(indices: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Higher score first, then smaller original full-pool index."""
    return indices[np.lexsort((indices, -score))]


def soft_weighted_mean(per_sample_loss: Any, sample_weights: Any) -> Any:
    """Return a differentiable non-negative weighted mean for the soft branch."""
    if tuple(per_sample_loss.shape) != tuple(sample_weights.shape):
        raise ValueError("per_sample_loss and sample_weights must have the same shape")
    denominator = sample_weights.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("Soft sample weights must have a positive sum in every optimizer batch")
    return (per_sample_loss * sample_weights).sum() / denominator


def select_budget_matched_fusion(
    *,
    mode: str,
    losses: np.ndarray,
    prototype_scores: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    loss_candidate_mask: np.ndarray,
    prototype_candidate_mask: np.ndarray,
    reference_budgets: dict[str, int],
    epoch: int,
) -> FusionSelection:
    """Apply one deterministic, class-budgeted fusion rule.

    ``weighted_ranking`` ranks all class members by the equal-weight mean of
    loss and prototype percentile reliabilities.  ``union_relaxed`` first
    restricts to the union of the two original candidate sets and ranks by the
    better of the two reliabilities; if that union is smaller than the reference
    PGDF budget, the same rule fills from the remaining class members.  The
    ``soft_weighting`` branch retains all candidates and normalizes positive
    continuous equal-weight reliability scores so every class sums to the
    PGDF budget.
    """
    losses, proto, labels, candidate, loss_candidate, proto_candidate = _validate_inputs(
        mode,
        losses,
        prototype_scores,
        labels,
        candidate_mask,
        loss_candidate_mask,
        prototype_candidate_mask,
        reference_budgets,
    )
    normalized_budgets = {str(label): int(count) for label, count in reference_budgets.items()}
    selected = np.zeros(len(labels), dtype=bool)
    weights = np.zeros(len(labels), dtype=np.float64) if mode == "soft_weighting" else None
    sample_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []

    for label in sorted(normalized_budgets):
        idx = np.where(candidate & (labels == label))[0]
        budget = normalized_budgets[label]
        loss_reliability = _reliability_from_rank(losses[idx], idx, high_is_reliable=False)
        proto_reliability = _reliability_from_rank(proto[idx], idx, high_is_reliable=True)
        equal_weight_reliability = 0.5 * loss_reliability + 0.5 * proto_reliability
        strict_mask = loss_candidate[idx] & proto_candidate[idx]
        union_mask = loss_candidate[idx] | proto_candidate[idx]
        strict_idx = idx[strict_mask]
        union_idx = idx[union_mask]
        strict_count = int(len(strict_idx))
        union_count = int(len(union_idx))
        budget_fill_count = 0
        raw_weight_sum: float | str = ""
        normalized_weight_sum: float | str = ""
        nonzero_weight_count: int | str = ""

        if mode == "weighted_ranking":
            chosen = _ordered_by_score(idx, equal_weight_reliability)[:budget]
        elif mode == "union_relaxed":
            union_reliability = np.maximum(loss_reliability, proto_reliability)
            chosen = _ordered_by_score(union_idx, union_reliability[union_mask])[:budget]
            if len(chosen) < budget:
                remaining_mask = ~union_mask
                remaining_idx = idx[remaining_mask]
                supplement = _ordered_by_score(remaining_idx, union_reliability[remaining_mask])[: budget - len(chosen)]
                chosen = np.concatenate((chosen, supplement))
                budget_fill_count = int(len(supplement))
        else:
            # A positive floor preserves a continuous, nonzero contribution for
            # every training-pool sample, including a class's worst-ranked item.
            raw_weights = np.maximum(equal_weight_reliability, SOFT_WEIGHT_FLOOR)
            class_weights = raw_weights * (float(budget) / float(raw_weights.sum()))
            assert weights is not None
            weights[idx] = class_weights
            chosen = idx
            raw_weight_sum = float(raw_weights.sum())
            normalized_weight_sum = float(class_weights.sum())
            nonzero_weight_count = int(np.count_nonzero(class_weights > 0.0))

        selected[chosen] = True
        final_idx = idx[selected[idx]]
        final_intersection_count = int(np.sum(np.isin(final_idx, strict_idx)))
        final_union_count = int(len(np.union1d(final_idx, strict_idx)))
        class_rows.append(
            {
                "epoch": int(epoch),
                "fusion_mode": mode,
                "web_label": str(label),
                "total_count": int(len(idx)),
                "loss_candidate_count": int(np.sum(loss_candidate[idx])),
                "prototype_candidate_count": int(np.sum(proto_candidate[idx])),
                "strict_intersection_count": strict_count,
                "union_candidate_count": union_count,
                "reference_budget": int(budget),
                "final_selected_count": int(len(final_idx)),
                "budget_fill_count": int(budget_fill_count),
                "final_strict_intersection_count": final_intersection_count,
                "final_strict_union_count": final_union_count,
                "final_strict_jaccard": (
                    float(final_intersection_count / final_union_count) if final_union_count else 1.0
                ),
                "raw_weight_sum": raw_weight_sum,
                "normalized_weight_sum": normalized_weight_sum,
                "nonzero_weight_count": nonzero_weight_count,
            }
        )
        union_reliability = np.maximum(loss_reliability, proto_reliability)
        for local_position, original_index in enumerate(idx):
            sample_rows.append(
                {
                    "epoch": int(epoch),
                    "fusion_mode": mode,
                    "index": int(original_index),
                    "web_label": str(label),
                    "loss_reliability": float(loss_reliability[local_position]),
                    "prototype_reliability": float(proto_reliability[local_position]),
                    "equal_weight_reliability": float(equal_weight_reliability[local_position]),
                    "union_reliability": float(union_reliability[local_position]),
                    "loss_candidate": "yes" if loss_candidate[original_index] else "no",
                    "prototype_candidate": "yes" if proto_candidate[original_index] else "no",
                    "strict_intersection_candidate": "yes" if strict_mask[local_position] else "no",
                    "union_candidate": "yes" if union_mask[local_position] else "no",
                    "final_selected": "yes" if selected[original_index] else "no",
                    "sample_weight": float(weights[original_index]) if weights is not None else "",
                }
            )

    update_row = {
        "epoch": int(epoch),
        "fusion_mode": mode,
        "reference_budget_total": int(sum(normalized_budgets.values())),
        "final_selected_total": int(selected.sum()),
        "strict_intersection_total": int(sum(row["strict_intersection_count"] for row in class_rows)),
        "union_candidate_total": int(sum(row["union_candidate_count"] for row in class_rows)),
        "budget_fill_total": int(sum(row["budget_fill_count"] for row in class_rows)),
        "final_strict_intersection_total": int(sum(row["final_strict_intersection_count"] for row in class_rows)),
        "final_strict_union_total": int(sum(row["final_strict_union_count"] for row in class_rows)),
        "effective_weight_budget": (
            float(sum(float(row["normalized_weight_sum"]) for row in class_rows))
            if mode == "soft_weighting"
            else ""
        ),
        "nonzero_weight_total": (
            int(sum(int(row["nonzero_weight_count"]) for row in class_rows))
            if mode == "soft_weighting"
            else ""
        ),
    }
    return FusionSelection(selected, weights, sample_rows, class_rows, update_row)
