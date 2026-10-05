from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from gcdd.budget_matching import load_pgdf_class_budget_schedule
from gcdd.checkpoint_validation import (
    PROTOCOL_NAME,
    build_or_load_fixed_validation_split,
    build_validation_safe_pgdf_reference,
    build_validation_safe_static_selections,
)
from gcdd.lora_noisy_baselines import select_best_dual_validation_row, train_coteaching_lora
from gcdd.lora_dynamic import (
    build_neighbor_margin_candidates,
    combine_loss_and_proto_classwise,
    count_classwise_intersection_fallbacks,
    select_small_loss_classwise_by_budget,
    select_top_proto_classwise,
    selection_update_epochs,
    stratify_neighbor_margin_samples,
)
from gcdd.lora_training import summarize_lora_logs
from gcdd.lora_training import train_dinov2_lora
from gcdd.lora_dynamic import train_dynamic_loss_lora
from scripts.run_lora_checkpoint_validation import (
    ABLATION_METHODS,
    METHODS,
    apply_lora_defaults,
    apply_overrides,
    parse_args,
    parse_methods,
    resolve_pgdf_geometry_config,
    resolve_retention_ratio,
    result_fields,
    validate_official_test_request,
    verify_observed_budget_match,
)
from scripts.run_neighbor_margin_asym40 import (
    DEFAULT_CONFIG as NEIGHBOR_MARGIN_STANDALONE_CONFIG,
    apply_machine_overrides as apply_neighbor_margin_standalone_overrides,
    load_standalone_config,
    validate_standalone_config,
)


class CheckpointValidationTests(unittest.TestCase):
    def test_neighbor_margin_cli_override_preserves_strict_default_and_enables_margin_rank(self) -> None:
        base = {"feature": {}, "lora": {}, "lora_train": {}, "pgdf": {"geometry_mode": "neighbor_margin"}}
        apply_lora_defaults(base)
        self.assertEqual(("neighbor_margin", False, True), resolve_pgdf_geometry_config(base))

        with mock.patch.object(
            sys,
            "argv",
            [
                "run_lora_checkpoint_validation.py",
                "--input-dir", "fixture-input",
                "--noise-index", "fixture-noise.csv",
                "--geometry-mode", "neighbor_margin",
                "--no-neighbor-margin-positive-only",
                "--neighbor-margin-use-fallback",
            ],
        ):
            args = parse_args()
        apply_overrides(base, args)
        self.assertEqual(("neighbor_margin", True, False), resolve_pgdf_geometry_config(base))

    def test_margin_rank_keeps_negative_margin_top_p_and_reuses_pgdf_fallback(self) -> None:
        """Margin-Rank changes only the geometry ranking score of PGDF."""
        labels = np.asarray(["A", "A", "A", "B", "B", "B"], dtype=str)
        training_pool = np.ones(6, dtype=bool)
        # A has only negative margins.  Its loss candidate (2) does not
        # intersect margin top-p (0, 1), so original PGDF fallback must take
        # the lowest-loss margin candidate (1), even though it is negative.
        margins = np.asarray([-0.1, -0.2, -0.3, 0.3, 0.2, 0.1], dtype=np.float32)
        losses = np.asarray([0.3, 0.1, 0.05, 0.1, 0.2, 0.3], dtype=np.float32)
        loss_selected = np.asarray([False, False, True, True, False, False])

        margin_top_p = build_neighbor_margin_candidates(
            margins, labels, training_pool, 0.8, positive_only=False
        )
        prototype_top_p = select_top_proto_classwise(
            np.asarray([0.9, 0.8, 0.1, 0.9, 0.8, 0.1], dtype=np.float32),
            labels,
            training_pool,
            0.8,
        )
        selected = combine_loss_and_proto_classwise(
            loss_selected, margin_top_p, losses, labels, training_pool
        )
        reliable, ambiguous, suspicious = stratify_neighbor_margin_samples(
            selected, margins, training_pool, positive_only=False
        )

        np.testing.assert_array_equal(np.where(margin_top_p)[0], np.asarray([0, 1, 3, 4]))
        self.assertTrue(np.all(margins[margin_top_p & (labels == "A")] < 0.0))
        self.assertEqual(2, int(margin_top_p[labels == "A"].sum()))
        self.assertEqual(2, int(margin_top_p[labels == "B"].sum()))
        self.assertEqual(2, int(prototype_top_p[labels == "A"].sum()))
        self.assertEqual(2, int(prototype_top_p[labels == "B"].sum()))
        self.assertEqual(
            1,
            count_classwise_intersection_fallbacks(loss_selected, margin_top_p, labels, training_pool),
        )
        np.testing.assert_array_equal(np.where(selected)[0], np.asarray([1, 3]))
        self.assertTrue(selected[1] and suspicious[1])
        self.assertFalse(reliable[1])
        self.assertFalse(np.any(reliable & ambiguous))
        self.assertFalse(np.any(reliable & suspicious))
        self.assertFalse(np.any(ambiguous & suspicious))
        np.testing.assert_array_equal(reliable | ambiguous | suspicious, training_pool)

    def test_standalone_launcher_routes_selection_methods_to_separate_variants(self) -> None:
        def make_args(
            *,
            positive_only: bool | None,
            use_fallback: bool | None,
            selection_method: str | None = None,
        ) -> SimpleNamespace:
            return SimpleNamespace(
                cub_root=None,
                cars_root=None,
                aircraft_root=None,
                device=None,
                local_repo=None,
                seeds="1,42,88",
                neighbor_margin_positive_only=positive_only,
                neighbor_margin_use_fallback=use_fallback,
                selection_method=selection_method,
            )

        strict_cfg = load_standalone_config(NEIGHBOR_MARGIN_STANDALONE_CONFIG)
        apply_neighbor_margin_standalone_overrides(strict_cfg, make_args(positive_only=None, use_fallback=None))
        validate_standalone_config(strict_cfg)
        self.assertEqual("reliable_only", strict_cfg["protocol"]["variant_id"])
        self.assertTrue(strict_cfg["pgdf"]["neighbor_margin_positive_only"])
        self.assertFalse(strict_cfg["pgdf"]["neighbor_margin_use_fallback"])

        margin_rank_cfg = load_standalone_config(NEIGHBOR_MARGIN_STANDALONE_CONFIG)
        apply_neighbor_margin_standalone_overrides(margin_rank_cfg, make_args(positive_only=False, use_fallback=True))
        validate_standalone_config(margin_rank_cfg)
        self.assertEqual("margin_rank", margin_rank_cfg["protocol"]["variant_id"])
        self.assertEqual("neighbor_margin_margin_rank_cyclic_asym40", margin_rank_cfg["protocol"]["name"])
        self.assertFalse(margin_rank_cfg["pgdf"]["neighbor_margin_positive_only"])
        self.assertTrue(margin_rank_cfg["pgdf"]["neighbor_margin_use_fallback"])

        prototype_cfg = load_standalone_config(NEIGHBOR_MARGIN_STANDALONE_CONFIG)
        apply_neighbor_margin_standalone_overrides(
            prototype_cfg,
            make_args(positive_only=None, use_fallback=None, selection_method="prototype_similarity"),
        )
        validate_standalone_config(prototype_cfg)
        self.assertEqual("prototype_similarity", prototype_cfg["protocol"]["variant_id"])
        self.assertEqual("prototype_similarity", prototype_cfg["pgdf"]["geometry_mode"])

    def test_methods_all_expands_all_thirteen_once(self) -> None:
        methods = parse_methods("all")

        self.assertEqual(list(METHODS), methods)
        self.assertEqual(13, len(methods))
        self.assertEqual(13, len(set(methods)))
        self.assertEqual("all_noisy", methods[0])

    def test_legacy_core_method_list_and_dynamic_alias_still_parse(self) -> None:
        methods = parse_methods("all_noisy,dynamic,jal_ce,pgdf_auto,pgdf_fixed")

        self.assertEqual(["all_noisy", "dynamic", "jal_ce", "pgdf_auto", "pgdf_fixed"], methods)
        self.assertEqual(parse_methods("dynamic"), ["dynamic"])
        self.assertEqual(0.8, resolve_retention_ratio("dynamic", 0.3))
        self.assertEqual(0.8, resolve_retention_ratio("dynamic_r08", 0.3))
        self.assertEqual(0.9, resolve_retention_ratio("dynamic_r09", 0.3))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_methods("dynamic,dynamic_r08")

    def test_proto_only_is_explicit_ablation_and_requires_selected_only_test(self) -> None:
        self.assertNotIn("proto_only", METHODS)
        self.assertIn("proto_only", ABLATION_METHODS)
        self.assertNotIn("dynamic_budget_matched", METHODS)
        self.assertIn("dynamic_budget_matched", ABLATION_METHODS)
        self.assertEqual(["proto_only"], parse_methods("proto_only"))
        validate_official_test_request(["proto_only"], True)
        validate_official_test_request(["dynamic_budget_matched"], True)
        validate_official_test_request(["dynamic", "jal_ce", "pgdf_fixed"], True)
        with self.assertRaisesRegex(ValueError, "requires --official-test-selected-only"):
            validate_official_test_request(["proto_only"], False)
        with self.assertRaisesRegex(ValueError, "requires --official-test-selected-only"):
            validate_official_test_request(["dynamic_budget_matched"], False)
        with self.assertRaisesRegex(ValueError, "two branches"):
            validate_official_test_request(["coteaching"], True)

    def test_unified_result_schema_contains_required_fields(self) -> None:
        required = {
            "method_key", "method", "seed", "checkpoint_protocol", "train_samples", "validation_samples",
            "test_samples", "best_val_epoch", "best_val_top1", "validation_selected_test_top1",
            "final_test_top1", "last5_test_mean", "selection_mode", "selected_count", "selection_ratio",
        }

        self.assertTrue(required.issubset(result_fields()))

    def test_fixed_validation_manifest_is_reused_and_class_stratified(self) -> None:
        paths = [f"/dataset/images/{label}/{index}.jpg" for label in ("A", "B") for index in range(10)]
        clean_labels = np.array(["A"] * 10 + ["B"] * 10)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first = build_or_load_fixed_validation_split(output_dir, paths, clean_labels, validation_ratio=0.2, validation_seed=17)
            second = build_or_load_fixed_validation_split(output_dir, paths, clean_labels, validation_ratio=0.2, validation_seed=17)

        self.assertEqual(PROTOCOL_NAME, first.metadata["protocol"])
        self.assertTrue(np.array_equal(first.validation_mask, second.validation_mask))
        self.assertTrue(np.array_equal(first.train_mask, ~first.validation_mask))
        self.assertEqual(2, int(first.validation_mask[clean_labels == "A"].sum()))
        self.assertEqual(2, int(first.validation_mask[clean_labels == "B"].sum()))

    def test_pgdf_reference_excludes_validation_rows(self) -> None:
        labels = np.array(["A", "A", "A", "B", "B", "B"])
        training_pool = np.array([True, True, False, True, True, False])
        base = np.array(
            [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [0.0, 1.0], [0.1, 0.9], [0.0, -1.0]],
            dtype=np.float32,
        )
        cfg = {
            "graph": {"knn_backend": "numpy", "k_pool_class": 2, "k_pool_global": 3, "k_class": 1, "k_global": 2, "rrf_k0": 20},
            "selection": {"otsu_bins": 8, "clean_ratio_clip": [0.3, 0.9], "epsilon": 1.0e-8},
        }
        reference = build_validation_safe_pgdf_reference({"cls": base, "gap": base, "top": base}, labels, training_pool, cfg)

        self.assertTrue(np.isnan(reference["proto_scores"][2]))
        self.assertTrue(np.isnan(reference["proto_scores"][5]))
        self.assertFalse(reference["centroid_reference_mask"][2])
        self.assertFalse(reference["gcdd_clean_mask"][5])
        self.assertEqual({"A", "B"}, set(reference["per_class_keep_counts"]))

    def test_all_static_masks_exclude_validation_rows(self) -> None:
        labels = np.array(["A"] * 4 + ["B"] * 4)
        training_pool = np.array([True, True, True, False, True, True, True, False])
        base = np.array(
            [
                [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [-1.0, 0.0],
                [0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.0, -1.0],
            ],
            dtype=np.float32,
        )
        cfg = {
            "graph": {"knn_backend": "numpy", "k_pool_class": 3, "k_pool_global": 4, "k_class": 2, "k_global": 3, "rrf_k0": 20},
            "selection": {"otsu_bins": 8, "clean_ratio_clip": [0.3, 0.9], "epsilon": 1.0e-8},
        }
        selections = build_validation_safe_static_selections(
            {"cls": base, "gap": base, "top": base}, labels, training_pool, cfg
        )

        self.assertEqual({"full_gcdd", "centroid", "proto_only", "both_only", "gcdd_proto", "fine"}, set(selections))
        for item in selections.values():
            self.assertFalse(np.any(np.asarray(item["mask"]) & ~training_pool))
        proto_mask = np.asarray(selections["proto_only"]["mask"], dtype=bool)
        expected_proto_mask = select_top_proto_classwise(
            np.asarray(selections["proto_only"]["score"]), labels, training_pool, 0.4
        )
        np.testing.assert_array_equal(expected_proto_mask, proto_mask)
        self.assertEqual(2, int(proto_mask.sum()))
        self.assertEqual(1, int(proto_mask[labels == "A"].sum()))
        self.assertEqual(1, int(proto_mask[labels == "B"].sum()))
        self.assertEqual("static_training_pool_prototype_only_p0.4", selections["proto_only"]["selection_mode"])

    def test_pgdf_class_budget_loader_requires_complete_seed_matched_schedule(self) -> None:
        labels = np.array(["A", "A", "B", "B"])
        training_pool = np.ones(4, dtype=bool)
        rows = [
            "method,seed,retention_ratio,proto_keep_ratio,epoch,web_label,total_count,selected_count",
            "DINOv2 LoRA PGDF fixed-p r=0.8 p=0.4,42,0.8,0.4,5,A,2,1",
            "DINOv2 LoRA PGDF fixed-p r=0.8 p=0.4,42,0.8,0.4,5,B,2,2",
            "DINOv2 LoRA PGDF fixed-p r=0.8 p=0.4,42,0.8,0.4,10,A,2,2",
            "DINOv2 LoRA PGDF fixed-p r=0.8 p=0.4,42,0.8,0.4,10,B,2,1",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection_per_class.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            schedule = load_pgdf_class_budget_schedule(
                path,
                expected_seed=42,
                expected_update_epochs=[5, 10],
                labels=labels,
                candidate_mask=training_pool,
            )
            with self.assertRaisesRegex(ValueError, "seed mismatch"):
                load_pgdf_class_budget_schedule(
                    path,
                    expected_seed=1,
                    expected_update_epochs=[5, 10],
                    labels=labels,
                    candidate_mask=training_pool,
                )

        self.assertEqual({"A": 1, "B": 2}, schedule.budgets[5])
        self.assertEqual({"A": 2, "B": 1}, schedule.budgets[10])
        self.assertEqual(0.8, schedule.source_retention_ratio)
        self.assertEqual(0.4, schedule.source_proto_keep_ratio)
        observed = [
            {"epoch": epoch, "web_label": label, "selected_count": count}
            for epoch, counts in schedule.budgets.items()
            for label, count in counts.items()
        ]
        verify_observed_budget_match(observed, schedule)
        observed[0]["selected_count"] = int(observed[0]["selected_count"]) + 1
        with self.assertRaisesRegex(RuntimeError, "do not exactly match"):
            verify_observed_budget_match(observed, schedule)

    def test_budget_matched_small_loss_uses_exact_counts_and_excludes_validation(self) -> None:
        labels = np.array(["A", "A", "A", "B", "B", "B"])
        training_pool = np.array([True, True, False, True, True, False])
        losses = np.array([0.2, 0.1, 0.0, 0.3, 0.1, 0.0], dtype=np.float32)

        selected = select_small_loss_classwise_by_budget(
            losses,
            labels,
            training_pool,
            {"A": 1, "B": 1},
        )

        np.testing.assert_array_equal(
            np.array([False, True, False, False, True, False]),
            selected,
        )
        self.assertFalse(np.any(selected & ~training_pool))
        self.assertEqual([5, 10, 15, 20, 25], selection_update_epochs(30, 5, 5))

    def test_budget_matched_small_loss_ties_use_original_index(self) -> None:
        labels = np.array(["A", "A", "B", "B"])
        losses = np.ones(4, dtype=np.float32)

        selected = select_small_loss_classwise_by_budget(
            losses,
            labels,
            np.ones(4, dtype=bool),
            {"A": 1, "B": 1},
        )

        np.testing.assert_array_equal(np.array([True, False, True, False]), selected)

    def test_dual_checkpoint_uses_validation_branch_mean_only(self) -> None:
        rows = [
            {"epoch": 1, "top1_a": 0.80, "top1_b": 0.60, "mean_ab_top1": 0.70, "official_test_top1": 0.99},
            {"epoch": 2, "top1_a": 0.77, "top1_b": 0.75, "mean_ab_top1": 0.76, "official_test_top1": 0.10},
            {"epoch": 3, "top1_a": 0.79, "top1_b": 0.69, "mean_ab_top1": 0.74, "official_test_top1": 1.00},
        ]

        selected = select_best_dual_validation_row(rows)

        self.assertEqual(2, selected["epoch"])
        self.assertAlmostEqual(0.76, (selected["top1_a"] + selected["top1_b"]) / 2.0)

    def test_coteaching_protocol_saves_validation_selected_and_final_states(self) -> None:
        import torch

        class FakeBackbone(torch.nn.Module):
            embed_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(3, 4)
                self.qkv = torch.nn.Linear(4, 4)

            def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"x_norm_clstoken": self.qkv(self.stem(images.mean(dim=(2, 3))))}

        cfg = {
            "feature": {"device": "cpu", "input_size": 16},
            "lora": {"rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "qkv"},
            "lora_train": {
                "epochs": 2, "batch_size": 2, "eval_batch_size": 2, "num_workers": 0, "pin_memory": False,
                "lora_lr": 1.0e-3, "head_lr": 1.0e-3, "weight_decay": 0.0,
                "scheduler": "none", "warmup_ratio": 0.0, "amp": False,
            },
        }
        # First four calls are validation A/B for epochs 1/2. Remaining calls
        # are post-training official-test evaluations and cannot alter best epoch.
        eval_metrics = [
            (0.9, 1.0), (0.1, 0.8),
            (0.6, 0.9), (0.6, 0.9),
            (0.2, 0.7), (0.4, 0.9),
            (0.8, 1.0), (0.6, 0.8),
            (0.3, 0.8), (0.5, 0.8),
            (0.8, 1.0), (0.6, 0.8),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
                path = root / f"dual_{index}.png"
                Image.new("RGB", (20, 20), color).save(path)
                paths.append(str(path))
            labels = np.array(["a", "b"])
            with (
                mock.patch("gcdd.lora_training.load_dinov2_model", side_effect=[FakeBackbone(), FakeBackbone()]),
                mock.patch("gcdd.lora_noisy_baselines.evaluate_lora", side_effect=eval_metrics),
            ):
                result = train_coteaching_lora(
                    paths, labels, paths, labels, np.ones(2, dtype=bool), cfg, "coteaching", 42,
                    remember_mode="fixed", remember_rate=0.8, final_remember_rate=0.8, warmup_epochs=1,
                    checkpoint_path=root / "best_val.pt", test_paths=paths, test_labels=labels,
                    final_checkpoint_path=root / "last.pt", last5_checkpoint_dir=root / "last5",
                    checkpoint_protocol=PROTOCOL_NAME,
                )

            self.assertEqual(2, result.summary["best_val_epoch"])
            self.assertAlmostEqual(0.6, result.summary["best_val_top1"])
            self.assertAlmostEqual(0.3, result.summary["validation_selected_test_top1"])
            self.assertAlmostEqual(0.7, result.summary["final_test_top1"])
            self.assertTrue((root / "best_val.pt").exists())
            self.assertTrue((root / "last.pt").exists())
            self.assertEqual(2, len(list((root / "last5").glob("*.pt"))))

    def test_dynamic_logs_can_use_shared_summary_helper(self) -> None:
        logs = [
            {"epoch": 1, "top1": 0.4, "top5": 0.7, "train_samples": 10, "eval_samples": 2, "trainable_params": 3, "total_params": 5},
            {"epoch": 2, "top1": 0.5, "top5": 0.8, "train_samples": 10, "eval_samples": 2, "trainable_params": 3, "total_params": 5},
        ]
        summary = summarize_lora_logs("dynamic", 42, logs)

        self.assertEqual("ce", summary["loss_type"])
        self.assertEqual("dynamic_or_provided_mask", summary["selection_mode"])
        self.assertEqual(2, summary["best_epoch"])

    def test_lora_and_dynamic_protocols_select_on_validation_then_test(self) -> None:
        import torch

        class FakeBackbone(torch.nn.Module):
            embed_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(3, 4)
                self.qkv = torch.nn.Linear(4, 4)

            def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"x_norm_clstoken": self.qkv(self.stem(images.mean(dim=(2, 3))))}

        cfg = {
            "feature": {"device": "cpu", "input_size": 16},
            "lora": {"rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "qkv"},
            "lora_train": {
                "epochs": 1, "batch_size": 2, "eval_batch_size": 2, "num_workers": 0, "pin_memory": False,
                "lora_lr": 1.0e-3, "head_lr": 1.0e-3, "weight_decay": 0.0, "scheduler": "none", "warmup_ratio": 0.0, "amp": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
                path = root / f"{index}.png"
                Image.new("RGB", (20, 20), color).save(path)
                paths.append(str(path))
            labels = np.array(["a", "b"])
            with mock.patch("gcdd.lora_training.load_dinov2_model", return_value=FakeBackbone()):
                lora = train_dinov2_lora(
                    paths, labels, paths, labels, np.ones(2, dtype=bool), cfg, "ce", 42,
                    checkpoint_path=root / "best_val.pt", test_paths=paths, test_labels=labels,
                    final_checkpoint_path=root / "last.pt", last5_checkpoint_dir=root / "last5",
                    checkpoint_protocol=PROTOCOL_NAME, posthoc_oracle_test=True,
                )
                dynamic = train_dynamic_loss_lora(
                    paths, labels, paths, labels, np.ones(2, dtype=bool), cfg, "dynamic", 42,
                    retention_ratio=0.8, warmup_epochs=1, update_interval=1,
                    checkpoint_path=root / "dynamic_best_val.pt", test_paths=paths, test_labels=labels,
                    final_checkpoint_path=root / "dynamic_last.pt", last5_checkpoint_dir=root / "dynamic_last5",
                    checkpoint_protocol=PROTOCOL_NAME, posthoc_oracle_test=True,
                )

            for result in (lora, dynamic):
                self.assertEqual(PROTOCOL_NAME, result.summary["checkpoint_protocol"])
                self.assertIn("validation_selected_test_top1", result.summary)
                self.assertIn("final_test_top1", result.summary)
                self.assertIn("last5_test_mean", result.summary)
                self.assertIn("oracle_best_test_top1", result.summary)
            self.assertTrue((root / "best_val.pt").exists())
            self.assertTrue((root / "last.pt").exists())

    def test_selected_only_official_test_evaluates_best_state_once(self) -> None:
        import torch

        class FakeBackbone(torch.nn.Module):
            embed_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(3, 4)
                self.qkv = torch.nn.Linear(4, 4)

            def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"x_norm_clstoken": self.qkv(self.stem(images.mean(dim=(2, 3))))}

        cfg = {
            "feature": {"device": "cpu", "input_size": 16},
            "lora": {"rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "qkv"},
            "lora_train": {
                "epochs": 1, "batch_size": 2, "eval_batch_size": 2, "num_workers": 0, "pin_memory": False,
                "lora_lr": 1.0e-3, "head_lr": 1.0e-3, "weight_decay": 0.0,
                "scheduler": "none", "warmup_ratio": 0.0, "amp": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
                path = root / f"selected_only_{index}.png"
                Image.new("RGB", (20, 20), color).save(path)
                paths.append(str(path))
            labels = np.array(["a", "b"])
            with (
                mock.patch("gcdd.lora_training.load_dinov2_model", return_value=FakeBackbone()),
                mock.patch("gcdd.lora_training.evaluate_lora", return_value=(0.75, 1.0)),
                mock.patch("gcdd.lora_training.evaluate_state_lora", return_value=(0.5, 0.9)) as official_eval,
            ):
                result = train_dinov2_lora(
                    paths, labels, paths, labels, np.ones(2, dtype=bool), cfg, "proto_only", 42,
                    checkpoint_path=root / "best_val.pt", test_paths=paths, test_labels=labels,
                    final_checkpoint_path=root / "last.pt", last5_checkpoint_dir=root / "last5",
                    checkpoint_protocol=PROTOCOL_NAME, posthoc_oracle_test=False,
                    official_test_selected_only=True,
                )

            self.assertEqual(1, official_eval.call_count)
            self.assertEqual("validation_selected_only", result.summary["official_test_evaluation"])
            self.assertAlmostEqual(0.5, result.summary["validation_selected_test_top1"])
            self.assertEqual("", result.summary["final_test_top1"])
            self.assertEqual("", result.summary["last5_test_mean"])
            self.assertTrue((root / "best_val.pt").exists())
            self.assertTrue((root / "last.pt").exists())

    def test_budget_matched_dynamic_selects_on_validation_and_tests_best_once(self) -> None:
        import torch

        class FakeBackbone(torch.nn.Module):
            embed_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(3, 4)
                self.qkv = torch.nn.Linear(4, 4)

            def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"x_norm_clstoken": self.qkv(self.stem(images.mean(dim=(2, 3))))}

        cfg = {
            "feature": {"device": "cpu", "input_size": 16},
            "lora": {"rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "qkv"},
            "lora_train": {
                "epochs": 2, "batch_size": 2, "eval_batch_size": 2, "num_workers": 0,
                "pin_memory": False, "lora_lr": 1.0e-3, "head_lr": 1.0e-3,
                "weight_decay": 0.0, "scheduler": "none", "warmup_ratio": 0.0, "amp": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            colors = ((255, 0, 0), (220, 0, 0), (0, 255, 0), (0, 220, 0))
            for index, color in enumerate(colors):
                path = root / f"budget_matched_{index}.png"
                Image.new("RGB", (20, 20), color).save(path)
                paths.append(str(path))
            labels = np.array(["a", "a", "b", "b"])
            with (
                mock.patch("gcdd.lora_training.load_dinov2_model", return_value=FakeBackbone()),
                mock.patch("gcdd.lora_dynamic.evaluate_lora", side_effect=[(0.8, 1.0), (0.7, 1.0)]),
                mock.patch("gcdd.lora_dynamic.evaluate_state_lora", return_value=(0.99, 1.0)) as official_eval,
            ):
                result = train_dynamic_loss_lora(
                    paths,
                    labels,
                    paths,
                    labels,
                    np.ones(4, dtype=bool),
                    cfg,
                    "dynamic_budget_matched",
                    42,
                    retention_ratio=0.8,
                    warmup_epochs=1,
                    update_interval=1,
                    checkpoint_path=root / "best_val.pt",
                    test_paths=paths,
                    test_labels=labels,
                    final_checkpoint_path=root / "last.pt",
                    last5_checkpoint_dir=root / "last5",
                    checkpoint_protocol=PROTOCOL_NAME,
                    class_budget_schedule={1: {"a": 1, "b": 1}},
                    scheduler_retention_ratio=0.4,
                    official_test_selected_only=True,
                )

            self.assertEqual(1, official_eval.call_count)
            self.assertEqual(1, result.summary["best_val_epoch"])
            self.assertAlmostEqual(0.8, result.summary["best_val_top1"])
            self.assertAlmostEqual(0.99, result.summary["validation_selected_test_top1"])
            self.assertEqual("", result.summary["final_test_top1"])
            self.assertEqual("", result.summary["last5_test_mean"])
            self.assertEqual(2, result.summary["final_selected_samples"])
            self.assertTrue(all(int(row["selected_count"]) == 1 for row in result.per_class_rows))


if __name__ == "__main__":
    unittest.main()
