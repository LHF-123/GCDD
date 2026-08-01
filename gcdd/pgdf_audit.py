"""Deterministic, data-free audits for PGDF selection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lora_dynamic import (
    choose_auto_proto_keep_ratio,
    combine_loss_and_proto_classwise,
    mask_jaccard,
    select_small_loss_classwise,
    select_top_proto_classwise,
)


@dataclass
class AutoFixedEquivalenceAudit:
    """Selection masks produced from one shared loss/prototype snapshot."""

    auto_p: float
    fixed_p: float
    jaccard: float
    loss_selected_mask: np.ndarray
    auto_proto_pass_mask: np.ndarray
    fixed_proto_pass_mask: np.ndarray
    auto_selected_mask: np.ndarray
    fixed_selected_mask: np.ndarray

    @property
    def p_matches(self) -> bool:
        return bool(np.isclose(self.auto_p, self.fixed_p, rtol=0.0, atol=1.0e-12))

    @property
    def proto_masks_match(self) -> bool:
        return bool(np.array_equal(self.auto_proto_pass_mask, self.fixed_proto_pass_mask))

    @property
    def selected_masks_match(self) -> bool:
        return bool(np.array_equal(self.auto_selected_mask, self.fixed_selected_mask))

    @property
    def passed(self) -> bool:
        return self.p_matches and self.proto_masks_match and self.selected_masks_match

    def summary(self) -> dict[str, Any]:
        return {
            "auto_p": float(self.auto_p),
            "fixed_p": float(self.fixed_p),
            "jaccard": float(self.jaccard),
            "p_matches": self.p_matches,
            "proto_masks_match": self.proto_masks_match,
            "selected_masks_match": self.selected_masks_match,
            "proto_mismatch_count": int(np.count_nonzero(self.auto_proto_pass_mask != self.fixed_proto_pass_mask)),
            "selected_mismatch_count": int(np.count_nonzero(self.auto_selected_mask != self.fixed_selected_mask)),
            "auto_selected_count": int(self.auto_selected_mask.sum()),
            "fixed_selected_count": int(self.fixed_selected_mask.sum()),
            "status": "PASS" if self.passed else "FAIL",
        }


def audit_auto_fixed_selection_equivalence(
    losses: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    centroid_mask: np.ndarray,
    proto_scores: np.ndarray,
    retention_ratio: float,
    auto_rule: dict[str, float],
    fixed_p: float,
) -> AutoFixedEquivalenceAudit:
    """Recompute both paths from identical arrays without training or randomness."""
    losses = np.asarray(losses, dtype=np.float32)
    labels = np.asarray(labels)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    centroid_mask = np.asarray(centroid_mask, dtype=bool)
    proto_scores = np.asarray(proto_scores, dtype=np.float32)
    if not (losses.shape == labels.shape == candidate_mask.shape == centroid_mask.shape == proto_scores.shape):
        raise ValueError("All audit arrays must have the same one-dimensional shape.")
    if not np.any(candidate_mask):
        raise ValueError("candidate_mask must contain at least one sample.")
    if np.any(np.isnan(losses[candidate_mask])) or np.any(np.isnan(proto_scores[candidate_mask])):
        raise ValueError("The shared snapshot has missing loss or prototype scores.")

    loss_selected_mask = select_small_loss_classwise(losses, labels, candidate_mask, retention_ratio)
    jaccard = mask_jaccard(loss_selected_mask, centroid_mask)
    auto_p = choose_auto_proto_keep_ratio(jaccard, auto_rule)

    auto_proto_pass_mask = select_top_proto_classwise(proto_scores, labels, candidate_mask, auto_p)
    fixed_proto_pass_mask = select_top_proto_classwise(proto_scores, labels, candidate_mask, fixed_p)
    auto_selected_mask = combine_loss_and_proto_classwise(
        loss_selected_mask, auto_proto_pass_mask, losses, labels, candidate_mask
    )
    fixed_selected_mask = combine_loss_and_proto_classwise(
        loss_selected_mask, fixed_proto_pass_mask, losses, labels, candidate_mask
    )
    return AutoFixedEquivalenceAudit(
        auto_p=float(auto_p),
        fixed_p=float(fixed_p),
        jaccard=float(jaccard),
        loss_selected_mask=loss_selected_mask,
        auto_proto_pass_mask=auto_proto_pass_mask,
        fixed_proto_pass_mask=fixed_proto_pass_mask,
        auto_selected_mask=auto_selected_mask,
        fixed_selected_mask=fixed_selected_mask,
    )
