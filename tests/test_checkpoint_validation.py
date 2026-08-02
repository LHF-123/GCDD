from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from gcdd.checkpoint_validation import (
    PROTOCOL_NAME,
    build_or_load_fixed_validation_split,
    build_validation_safe_pgdf_reference,
    build_validation_safe_static_selections,
)
from gcdd.lora_noisy_baselines import select_best_dual_validation_row, train_coteaching_lora
from gcdd.lora_dynamic import select_top_proto_classwise
from gcdd.lora_training import summarize_lora_logs
from gcdd.lora_training import train_dinov2_lora
from gcdd.lora_dynamic import train_dynamic_loss_lora
from scripts.run_lora_checkpoint_validation import (
    ABLATION_METHODS,
    METHODS,
    parse_methods,
    resolve_retention_ratio,
    result_fields,
    validate_official_test_request,
)


class CheckpointValidationTests(unittest.TestCase):
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
        self.assertEqual(["proto_only"], parse_methods("proto_only"))
        validate_official_test_request(["proto_only"], True)
        with self.assertRaisesRegex(ValueError, "requires --official-test-selected-only"):
            validate_official_test_request(["proto_only"], False)
        with self.assertRaisesRegex(ValueError, "restricted"):
            validate_official_test_request(["dynamic_r08"], True)

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


if __name__ == "__main__":
    unittest.main()
