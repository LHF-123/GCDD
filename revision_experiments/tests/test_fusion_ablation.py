"""CPU-only checks for the new PGDF fusion-ablation decision rules."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from gcdd.fusion_ablation import select_budget_matched_fusion, soft_weighted_mean
from gcdd.lora_dynamic import combine_loss_and_proto_classwise


class FusionAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.asarray(["a", "a", "a", "a", "b", "b", "b"], dtype=str)
        self.candidate = np.ones(len(self.labels), dtype=bool)
        self.losses = np.asarray([0.10, 0.20, 0.30, 0.40, 0.40, 0.20, 0.10], dtype=np.float64)
        self.proto = np.asarray([0.10, 0.90, 0.80, 0.20, 0.10, 0.80, 0.90], dtype=np.float64)
        self.loss_gate = np.asarray([True, True, True, False, False, True, True])
        self.proto_gate = np.asarray([False, True, True, False, False, True, True])
        self.budgets = {"a": 2, "b": 2}

    def test_01_existing_strict_intersection_and_fallback_remain_unchanged(self) -> None:
        # Class a has a nonempty strict intersection (1, 2).  Class b uses its
        # pre-existing strict intersection (5, 6); this directly exercises the
        # submitted function rather than a replacement implementation.
        selected = combine_loss_and_proto_classwise(
            self.loss_gate, self.proto_gate, self.losses, self.labels, self.candidate
        )
        self.assertEqual(np.where(selected)[0].tolist(), [1, 2, 5, 6])

    def test_02_weighted_ranking_uses_equal_percentiles_and_original_index_ties(self) -> None:
        result = select_budget_matched_fusion(
            mode="weighted_ranking",
            losses=self.losses,
            prototype_scores=self.proto,
            labels=self.labels,
            candidate_mask=self.candidate,
            loss_candidate_mask=self.loss_gate,
            prototype_candidate_mask=self.proto_gate,
            reference_budgets=self.budgets,
            epoch=5,
        )
        # For class a, index 1 has the highest fused score.  Indices 0 and 2
        # tie at 0.5, so the smaller original index 0 must win deterministically.
        self.assertEqual(np.where(result.final_selected_mask)[0].tolist(), [0, 1, 5, 6])
        self.assertEqual(result.update_row["reference_budget_total"], 4)
        self.assertEqual(result.update_row["final_selected_total"], 4)

    def test_03_union_relaxed_uses_union_then_declared_class_internal_fill(self) -> None:
        budgets = {"a": 3, "b": 2}
        loss_gate = self.loss_gate.copy()
        proto_gate = self.proto_gate.copy()
        # Make class a union only {0, 1}; a fixed budget of three requires one
        # deterministic fill from the remaining class members.
        loss_gate[:4] = np.asarray([True, False, False, False])
        proto_gate[:4] = np.asarray([False, True, False, False])
        result = select_budget_matched_fusion(
            mode="union_relaxed",
            losses=self.losses,
            prototype_scores=self.proto,
            labels=self.labels,
            candidate_mask=self.candidate,
            loss_candidate_mask=loss_gate,
            prototype_candidate_mask=proto_gate,
            reference_budgets=budgets,
            epoch=10,
        )
        class_a = next(row for row in result.per_class_rows if row["web_label"] == "a")
        self.assertEqual(class_a["union_candidate_count"], 2)
        self.assertEqual(class_a["budget_fill_count"], 1)
        self.assertEqual(class_a["final_selected_count"], 3)
        self.assertTrue({0, 1}.issubset(set(np.where(result.final_selected_mask)[0].tolist())))

    def test_04_soft_weighting_is_positive_class_normalized_and_differentiable(self) -> None:
        result = select_budget_matched_fusion(
            mode="soft_weighting",
            losses=self.losses,
            prototype_scores=self.proto,
            labels=self.labels,
            candidate_mask=self.candidate,
            loss_candidate_mask=self.loss_gate,
            prototype_candidate_mask=self.proto_gate,
            reference_budgets=self.budgets,
            epoch=15,
        )
        self.assertTrue(np.all(result.final_selected_mask))
        assert result.sample_weights is not None
        self.assertTrue(np.all(result.sample_weights > 0.0))
        for row in result.per_class_rows:
            idx = np.where(self.labels == row["web_label"])[0]
            self.assertAlmostEqual(float(result.sample_weights[idx].sum()), float(row["reference_budget"]), places=10)
            self.assertEqual(row["nonzero_weight_count"], len(idx))

        logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], requires_grad=True)
        targets = torch.tensor([0, 1])
        per_sample = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        weights = torch.tensor([0.25, 1.75])
        weighted = soft_weighted_mean(per_sample, weights)
        expected = (per_sample * weights).sum() / weights.sum()
        self.assertTrue(torch.allclose(weighted, expected))
        weighted.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_05_reference_budget_must_match_all_observed_classes(self) -> None:
        with self.assertRaises(ValueError):
            select_budget_matched_fusion(
                mode="weighted_ranking",
                losses=self.losses,
                prototype_scores=self.proto,
                labels=self.labels,
                candidate_mask=self.candidate,
                loss_candidate_mask=self.loss_gate,
                prototype_candidate_mask=self.proto_gate,
                reference_budgets={"a": 2},
                epoch=5,
            )


if __name__ == "__main__":
    unittest.main()
