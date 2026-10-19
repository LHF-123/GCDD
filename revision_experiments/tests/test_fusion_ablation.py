"""CPU-only checks for the new PGDF fusion-ablation decision rules."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from gcdd.budget_matching import load_pgdf_class_budget_schedule
from gcdd.fusion_ablation import select_budget_matched_fusion, soft_weighted_mean
from gcdd.lora_dynamic import (
    BUDGET_MATCHED_PROTO_STRATEGY,
    BUDGET_MATCHED_RANDOM_STRATEGY,
    RANDOM_BUDGET_SELECTION_SCHEME,
    build_dynamic_prototype_snapshot,
    combine_loss_and_proto_classwise,
    estimate_dynamic_total_steps,
    select_random_classwise_by_budget,
    select_top_proto_classwise_by_budget,
    train_dynamic_loss_lora,
)
from scripts import run_lora_checkpoint_validation as checkpoint_runner
from tools.audit_dynamic_proto_budget_match import verify_control_budget_provenance


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

    def test_10_budget_matched_prototype_only_selects_exact_class_counts(self) -> None:
        selected = select_top_proto_classwise_by_budget(
            self.proto,
            self.labels,
            self.candidate,
            self.budgets,
        )
        # Highest prototype scores per class are selected independently of the
        # submitted PGDF sample identities and loss values.
        self.assertEqual(np.where(selected)[0].tolist(), [1, 2, 5, 6])
        for label, expected_count in self.budgets.items():
            self.assertEqual(int(np.sum(selected & (self.labels == label))), expected_count)

    def test_11_budget_matched_random_is_stateless_reproducible_and_classwise_exact(self) -> None:
        first = select_random_classwise_by_budget(
            self.labels,
            self.candidate,
            self.budgets,
            dataset="synthetic",
            training_seed=42,
            epoch=10,
        )
        repeated = select_random_classwise_by_budget(
            self.labels,
            self.candidate,
            self.budgets,
            dataset="synthetic",
            training_seed=42,
            epoch=10,
        )
        self.assertTrue(np.array_equal(first, repeated))
        self.assertIn("sha256-v1", RANDOM_BUDGET_SELECTION_SCHEME)
        for label, expected_count in self.budgets.items():
            self.assertEqual(int(np.sum(first & (self.labels == label))), expected_count)

    def test_12_runner_registers_all_dynamic_pgdf_actual_budget_controls(self) -> None:
        expected = {
            "pgdf_exact_budget_proto",
            "pgdf_exact_budget_random",
        }
        self.assertEqual(set(checkpoint_runner.EXACT_BUDGET_CONTROL_METHODS), expected)
        self.assertTrue(expected.issubset(set(checkpoint_runner.ABLATION_METHODS)))
        self.assertTrue(expected.issubset(set(checkpoint_runner.DYNAMIC_METHODS)))
        self.assertIn("dynamic_budget_matched_dynamic_proto", checkpoint_runner.DYNAMIC_METHODS)

    def test_13_budget_schedule_loader_fails_closed_for_missing_or_illegal_class_budgets(self) -> None:
        fields = [
            "seed", "epoch", "web_label", "total_count", "selected_count",
            "retention_ratio", "proto_keep_ratio", "selection_strategy", "prototype_mode", "method",
        ]

        def rows(*, seed: int = 1, selected_a: int = 2, include_b: bool = True) -> list[dict[str, object]]:
            values: list[dict[str, object]] = [
                {
                    "seed": seed, "epoch": 5, "web_label": "a", "total_count": 4,
                    "selected_count": selected_a, "retention_ratio": 0.8, "proto_keep_ratio": 0.4,
                    "selection_strategy": "loss_and_proto", "prototype_mode": "dynamic_lora", "method": "PGDF-DynamicProto",
                }
            ]
            if include_b:
                values.append(
                    {
                        "seed": seed, "epoch": 5, "web_label": "b", "total_count": 3,
                        "selected_count": 2, "retention_ratio": 0.8, "proto_keep_ratio": 0.4,
                        "selection_strategy": "loss_and_proto", "prototype_mode": "dynamic_lora", "method": "PGDF-DynamicProto",
                    }
                )
            return values

        def write(path: Path, values: list[dict[str, object]]) -> None:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)

        def load(path: Path, *, updates: list[int] = [5]):
            return load_pgdf_class_budget_schedule(
                path,
                expected_seed=1,
                expected_update_epochs=updates,
                labels=self.labels,
                candidate_mask=self.candidate,
                expected_selection_strategy="loss_and_proto",
                expected_prototype_mode="dynamic_lora",
            )

        with tempfile.TemporaryDirectory() as temporary:
            budget_path = Path(temporary) / "selection_per_class.csv"
            write(budget_path, rows())
            self.assertEqual(load(budget_path).budgets[5], {"a": 2, "b": 2})
            write(budget_path, rows(include_b=False))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load(budget_path)
            write(budget_path, rows(selected_a=0))
            with self.assertRaisesRegex(ValueError, "Invalid PGDF selected_count"):
                load(budget_path)
            write(budget_path, rows(selected_a=5))
            with self.assertRaisesRegex(ValueError, "Invalid PGDF selected_count"):
                load(budget_path)
            write(budget_path, rows(seed=42))
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                load(budget_path)
            write(budget_path, rows())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load(budget_path, updates=[5, 10])

    def test_14_budget_source_validation_manifest_mismatch_fails_closed(self) -> None:
        manifest = {
            "protocol": "fixed_clean_validation_v1",
            "validation_seed": 20250726,
            "train_paths_sha256": "paths-a",
            "source_train_samples": 10,
            "training_pool_samples": 9,
            "validation_samples": 1,
            "stratification_label": "clean_label",
            "validation_label": "clean_label",
            "validation_ratio": 0.1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary) / "source"
            budget_root = source_root / "pgdf_dynamic_proto"
            budget_root.mkdir(parents=True)
            mismatched = dict(manifest)
            mismatched["train_paths_sha256"] = "paths-b"
            (source_root / "validation_manifest.json").write_text(
                json.dumps(mismatched), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "validation manifest"):
                checkpoint_runner.validate_budget_source_manifest(budget_root, manifest)

    def test_15_source_active_counts_and_budget_sha256_are_audited(self) -> None:
        fields = [
            "seed", "epoch", "web_label", "total_count", "selected_count",
            "retention_ratio", "proto_keep_ratio", "selection_strategy", "prototype_mode", "method",
        ]
        values = [
            {
                "seed": 1, "epoch": 5, "web_label": "a", "total_count": 4, "selected_count": 2,
                "retention_ratio": 0.8, "proto_keep_ratio": 0.4,
                "selection_strategy": "loss_and_proto", "prototype_mode": "dynamic_lora", "method": "PGDF-DynamicProto",
            },
            {
                "seed": 1, "epoch": 5, "web_label": "b", "total_count": 3, "selected_count": 2,
                "retention_ratio": 0.8, "proto_keep_ratio": 0.4,
                "selection_strategy": "loss_and_proto", "prototype_mode": "dynamic_lora", "method": "PGDF-DynamicProto",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "pgdf_dynamic_proto"
            source_seed = source_root / "seed1"
            source_seed.mkdir(parents=True)
            budget_path = source_seed / "selection_per_class.csv"
            with budget_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)
            with (source_seed / "selection_updates.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "num_selected", "fallback_count"])
                writer.writeheader()
                writer.writerow({"epoch": 5, "num_selected": 4, "fallback_count": 1})
            schedule = load_pgdf_class_budget_schedule(
                budget_path,
                expected_seed=1,
                expected_update_epochs=[5],
                labels=self.labels,
                candidate_mask=self.candidate,
                expected_selection_strategy="loss_and_proto",
                expected_prototype_mode="dynamic_lora",
            )
            audit = checkpoint_runner.inspect_pgdf_active_budget_source(source_root, schedule)
            self.assertEqual(audit["updates"], [{
                "epoch": 5, "active_selected_count": 4, "fallback_count": 1,
                "derived_strict_intersection_count": 3,
            }])

            run_dir = root / "control" / "seed1"
            run_dir.mkdir(parents=True)
            sha = hashlib.sha256(budget_path.read_bytes()).hexdigest()
            (run_dir / "budget_source.json").write_text(json.dumps({"source_sha256": sha}), encoding="utf-8")
            (run_dir / "budget_match_validation.json").write_text(
                json.dumps({
                    "status": "PASS", "source_budget_sha256": sha,
                    "observed_selection_counts_verified": True,
                    "membership_rule": {
                        "uses_pgdf_sample_identity": False,
                        "uses_clean_noisy_oracle": False,
                        "uses_official_test_for_selection": False,
                    },
                }),
                encoding="utf-8",
            )
            verify_control_budget_provenance(run_dir, budget_path)
            (run_dir / "budget_match_validation.json").write_text(
                json.dumps({"status": "PASS", "source_budget_sha256": "incorrect"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                verify_control_budget_provenance(run_dir, budget_path)

    def test_16_prototype_and_random_controls_are_criterion_isolated_and_deterministic(self) -> None:
        # The top prototype item in class a has deliberately poor synthetic CE
        # loss; it still wins, proving direct score ranking is not p-gate or
        # small-loss membership reuse.
        scores = np.asarray([0.10, 0.20, 0.30, 0.99, 0.50, 0.40, 0.30])
        selected_proto = select_top_proto_classwise_by_budget(
            scores, self.labels, self.candidate, {"a": 1, "b": 1}
        )
        self.assertEqual(np.where(selected_proto)[0].tolist(), [3, 4])

        reversed_budget_mapping = {"b": 2, "a": 2}
        first = select_random_classwise_by_budget(
            self.labels, self.candidate, reversed_budget_mapping,
            dataset="synthetic", training_seed=1, epoch=5,
        )
        same_inputs = select_random_classwise_by_budget(
            self.labels, self.candidate, self.budgets,
            dataset="synthetic", training_seed=1, epoch=5,
        )
        changed_seed = next(
            (
                select_random_classwise_by_budget(
                    self.labels, self.candidate, self.budgets,
                    dataset="synthetic", training_seed=other_seed, epoch=5,
                )
                for other_seed in range(2, 101)
                if not np.array_equal(
                    first,
                    select_random_classwise_by_budget(
                        self.labels, self.candidate, self.budgets,
                        dataset="synthetic", training_seed=other_seed, epoch=5,
                    ),
                )
            ),
            None,
        )
        self.assertTrue(np.array_equal(first, same_inputs))
        self.assertIsNotNone(changed_seed)

    def test_17_scheduler_horizon_remains_the_pgdf_declared_min_r_p_protocol(self) -> None:
        # r=0.8, p=0.4 fixes the declared scheduler retention estimate at
        # min(r, p)=0.4, irrespective of a particular update's exact count.
        self.assertEqual(
            estimate_dynamic_total_steps(
                candidate_count=11, batch_size=4, epochs=30, warmup_epochs=5, retention_ratio=min(0.8, 0.4)
            ),
            40,
        )
        self.assertIn("scheduler_estimated_total_steps", checkpoint_runner.result_fields())
        self.assertIn("optimizer_steps_actual_since_previous_update", checkpoint_runner.update_fields())

    def test_18_exact_budget_controls_keep_warmup_updates_active_masks_and_scheduler_protocol(self) -> None:
        class FakeBackbone(torch.nn.Module):
            embed_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(3, 4)
                self.qkv = torch.nn.Linear(4, 4)

            def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"x_norm_clstoken": self.qkv(self.stem(images.mean(dim=(2, 3))))}

        cfg = {
            "dataset": {"name": "synthetic-exact-budget"},
            "feature": {"device": "cpu", "input_size": 16},
            "lora": {"rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "qkv"},
            "lora_train": {
                "epochs": 2, "batch_size": 2, "eval_batch_size": 2, "num_workers": 0, "pin_memory": False,
                "lora_lr": 1.0e-3, "head_lr": 1.0e-3, "weight_decay": 0.0,
                "scheduler": "cosine", "warmup_ratio": 0.1, "amp": False,
            },
        }
        labels = np.asarray(["a", "a", "b", "b"], dtype=str)
        schedule = {1: {"a": 2, "b": 2}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths: list[str] = []
            for index, color in enumerate(((255, 0, 0), (200, 0, 0), (0, 255, 0), (0, 200, 0))):
                path = root / f"sample_{index}.png"
                Image.new("RGB", (20, 20), color).save(path)
                paths.append(str(path))
            with mock.patch("gcdd.lora_training.load_dinov2_model", side_effect=lambda *args, **kwargs: FakeBackbone()):
                proto = train_dynamic_loss_lora(
                    paths, labels, paths, labels, np.ones(len(labels), dtype=bool), cfg,
                    method="pgdf_exact_budget_proto", seed=42, retention_ratio=0.8,
                    warmup_epochs=1, update_interval=1, class_budget_schedule=schedule,
                    scheduler_retention_ratio=0.4, selection_strategy=BUDGET_MATCHED_PROTO_STRATEGY,
                    prototype_mode="dynamic_lora", proto_keep_ratio=0.4,
                )
                random = train_dynamic_loss_lora(
                    paths, labels, paths, labels, np.ones(len(labels), dtype=bool), cfg,
                    method="pgdf_exact_budget_random", seed=42, retention_ratio=0.8,
                    warmup_epochs=1, update_interval=1, class_budget_schedule=schedule,
                    scheduler_retention_ratio=0.4, selection_strategy=BUDGET_MATCHED_RANDOM_STRATEGY,
                    prototype_mode="fixed",
                )
        for result in (proto, random):
            self.assertEqual([4, 4], [int(row["train_samples"]) for row in result.logs])
            self.assertEqual(1, result.summary["selection_updates"])
            self.assertEqual(0.4, result.summary["scheduler_retention_ratio"])
            self.assertEqual(3, result.summary["scheduler_estimated_total_steps"])
            self.assertEqual(4, result.summary["optimizer_steps_attempted"])
            self.assertEqual(4, result.summary["optimizer_steps_actual"])
            self.assertEqual(4, result.summary["scheduler_steps_actual"])
            self.assertEqual(4, result.update_rows[0]["num_selected"])
            self.assertEqual(2, result.update_rows[0]["optimizer_steps_actual_since_previous_update"])
        self.assertEqual(BUDGET_MATCHED_PROTO_STRATEGY, proto.summary["selection_strategy"])
        self.assertEqual(BUDGET_MATCHED_RANDOM_STRATEGY, random.summary["selection_strategy"])


if __name__ == "__main__":
    unittest.main()
