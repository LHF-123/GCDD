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
)
from gcdd.lora_training import summarize_lora_logs
from gcdd.lora_training import train_dinov2_lora
from gcdd.lora_dynamic import train_dynamic_loss_lora
from scripts.run_lora_checkpoint_validation import METHODS, parse_methods


class CheckpointValidationTests(unittest.TestCase):
    def test_all_noisy_is_an_explicit_checkpoint_validation_method(self) -> None:
        methods = parse_methods("all_noisy,dynamic,jal_ce,pgdf_auto,pgdf_fixed")

        self.assertEqual(list(METHODS), methods)
        self.assertEqual("all_noisy", methods[0])

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


if __name__ == "__main__":
    unittest.main()
