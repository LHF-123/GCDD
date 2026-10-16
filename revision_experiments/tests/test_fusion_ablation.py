"""CPU-only checks for the new PGDF fusion-ablation decision rules."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from gcdd.budget_matching import load_pgdf_class_budget_schedule
from gcdd.fusion_ablation import select_budget_matched_fusion, soft_weighted_mean
from gcdd.lora_dynamic import build_dynamic_prototype_snapshot, combine_loss_and_proto_classwise
from scripts import run_lora_checkpoint_validation as checkpoint_runner


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

    def test_06_dynamic_prototype_snapshot_is_a_valid_fusion_score_source(self) -> None:
        # These are deliberately current synthetic CLS vectors, not cached
        # frozen features.  The same snapshot scores can feed the unchanged
        # fusion decision routine without an additional model pass.
        features = np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
        )
        labels = np.asarray(["a", "a", "b", "b"], dtype=str)
        snapshot = build_dynamic_prototype_snapshot(
            features,
            labels,
            np.ones(len(labels), dtype=bool),
            proto_keep_ratio=0.5,
            model_state_checksum="synthetic-current-lora-state",
        )
        self.assertTrue(np.isfinite(snapshot.scores).all())
        self.assertEqual(int(snapshot.gate_mask.sum()), 2)
        self.assertEqual(snapshot.model_state_checksum, "synthetic-current-lora-state")
        fusion = select_budget_matched_fusion(
            mode="weighted_ranking",
            losses=np.asarray([0.1, 0.2, 0.2, 0.1]),
            prototype_scores=snapshot.scores,
            labels=labels,
            candidate_mask=np.ones(len(labels), dtype=bool),
            loss_candidate_mask=np.asarray([True, False, False, True]),
            prototype_candidate_mask=snapshot.gate_mask,
            reference_budgets={"a": 1, "b": 1},
            epoch=5,
        )
        self.assertEqual(int(fusion.final_selected_mask.sum()), 2)

    def test_07_dynamic_fusion_budget_rejects_fixed_prototype_source(self) -> None:
        labels = np.asarray(["a", "a", "b", "b"], dtype=str)
        candidate = np.ones(len(labels), dtype=bool)
        fields = [
            "seed", "epoch", "web_label", "total_count", "selected_count",
            "retention_ratio", "proto_keep_ratio", "selection_strategy", "prototype_mode", "method",
        ]
        base_rows = [
            {
                "seed": 1, "epoch": 5, "web_label": label, "total_count": 2, "selected_count": 1,
                "retention_ratio": 0.8, "proto_keep_ratio": 0.4,
                "selection_strategy": "loss_and_proto", "prototype_mode": "fixed", "method": "PGDF fixed",
            }
            for label in ("a", "b")
        ]
        with tempfile.TemporaryDirectory() as temporary:
            budget_path = Path(temporary) / "selection_per_class.csv"
            with budget_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(base_rows)
            with self.assertRaisesRegex(ValueError, "prototype_mode mismatch"):
                load_pgdf_class_budget_schedule(
                    budget_path,
                    expected_seed=1,
                    expected_update_epochs=[5],
                    labels=labels,
                    candidate_mask=candidate,
                    expected_selection_strategy="loss_and_proto",
                    expected_prototype_mode="dynamic_lora",
                )
            for row in base_rows:
                row["prototype_mode"] = "dynamic_lora"
                row["method"] = "PGDF-DynamicProto"
            with budget_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(base_rows)
            schedule = load_pgdf_class_budget_schedule(
                budget_path,
                expected_seed=1,
                expected_update_epochs=[5],
                labels=labels,
                candidate_mask=candidate,
                expected_selection_strategy="loss_and_proto",
                expected_prototype_mode="dynamic_lora",
            )
        self.assertEqual(schedule.budgets[5], {"a": 1, "b": 1})

    def test_08_dynamic_fusion_method_keys_are_distinct_from_fixed_fusion_keys(self) -> None:
        self.assertTrue(
            set(checkpoint_runner.DYNAMIC_FUSION_ABLATION_METHODS).isdisjoint(
                checkpoint_runner.FIXED_FUSION_ABLATION_METHODS
            )
        )
        self.assertEqual(
            checkpoint_runner.DYNAMIC_FUSION_SPECIFICATION["pgdf_dynamic_fusion_soft"][1],
            "soft_weighting",
        )

    def test_09_dynamic_budget_provenance_records_exact_or_legacy_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary) / "dynamic_reference"
            budget_root = source_root / "pgdf_dynamic_proto"
            budget_root.mkdir(parents=True)
            config_path = source_root / "resolved_config.yaml"
            config_path.write_text(
                "dataset:\n"
                "  name: CUB-200-2011-asym40-s42\n"
                "  index_file: outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv\n"
                "checkpoint_validation:\n"
                "  noise_realization:\n"
                "    noise_strategy: asymmetric\n"
                "    noise_ratio: 0.4\n"
                "    noise_seed: 42\n"
                "    mapping_type: cyclic\n"
                "    mapping_seed: ''\n"
                "  noise_index_sha256: expected-hash\n",
                encoding="utf-8",
            )
            metadata = {
                "noise_strategy": "asymmetric",
                "noise_ratio": 0.4,
                "noise_seed": 42,
                "mapping_type": "cyclic",
                "mapping_seed": "",
            }
            status = checkpoint_runner.validate_dynamic_pgdf_budget_source_provenance(
                budget_root,
                dataset_name="CUB-200-2011-asym40-s42",
                noise_metadata=metadata,
                noise_index_sha256="expected-hash",
                noise_index_name="cub_asym_r0p4_s42_index.csv",
            )
            self.assertEqual(status, "exact_noise_index_sha256")
            config_path.write_text(
                "dataset:\n"
                "  name: CUB-200-2011-asym40-s42\n"
                "  index_file: outputs/CUB_200_2011/noise_indices/cub_asym_r0p4_s42_index.csv\n"
                "checkpoint_validation:\n"
                "  validation_seed: 20250726\n",
                encoding="utf-8",
            )
            legacy_status = checkpoint_runner.validate_dynamic_pgdf_budget_source_provenance(
                budget_root,
                dataset_name="CUB-200-2011-asym40-s42",
                noise_metadata=metadata,
                noise_index_sha256="current-hash",
                noise_index_name="cub_asym_r0p4_s42_index.csv",
            )
            self.assertEqual(legacy_status, "legacy_manifest_and_noise_index_filename")


if __name__ == "__main__":
    unittest.main()
