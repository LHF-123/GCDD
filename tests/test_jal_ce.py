from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from gcdd.lora_training import build_criterion, resolve_loss_config, train_dinov2_lora
from losses.jal import AMSELoss, JALCELoss, NCELoss
from scripts.run_lora_web_bird import validate_loss_method_combination


class JALCELossTest(unittest.TestCase):
    def test_nce_and_amse_match_manual_formula(self) -> None:
        import torch
        import torch.nn.functional as F

        logits = torch.tensor([[2.0, -1.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float64)
        target = torch.tensor([0, 2])

        log_prob = F.log_softmax(logits.float(), dim=1)
        expected_nce = (
            -log_prob.gather(1, target.view(-1, 1)).squeeze(1) / (-log_prob).sum(dim=1).clamp_min(1.0e-8)
        ).mean()
        prob = F.softmax(logits.float(), dim=1)
        target_vec = 30.0 * F.one_hot(target, num_classes=3).to(prob.dtype)
        expected_amse = (prob - target_vec).pow(2).mean()

        self.assertTrue(torch.allclose(NCELoss()(logits, target), expected_nce))
        self.assertTrue(torch.allclose(AMSELoss()(logits, target), expected_amse))
        self.assertTrue(torch.allclose(JALCELoss()(logits, target), expected_nce + expected_amse))

    def test_jal_returns_fp32_scalar_and_backward_is_finite(self) -> None:
        import torch

        logits = torch.randn(4, 10, dtype=torch.float16, requires_grad=True)
        target = torch.randint(0, 10, (4,))
        loss = JALCELoss()(logits, target)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_factory_defaults_to_ce_and_builds_jal(self) -> None:
        import torch

        self.assertEqual(resolve_loss_config({})["loss_type"], "ce")
        self.assertIsInstance(build_criterion(torch, {}), torch.nn.CrossEntropyLoss)
        criterion = build_criterion(torch, {"loss_type": "jal_ce", "jal_a": 12.0})
        self.assertIsInstance(criterion, JALCELoss)
        self.assertEqual(criterion.amse.a, 12.0)

    def test_entry_rejects_jal_with_filtered_method(self) -> None:
        cfg = {"loss_type": "jal_ce"}
        validate_loss_method_combination(cfg, ["all"])
        with self.assertRaisesRegex(ValueError, "full noisy"):
            validate_loss_method_combination(cfg, ["centroid"])

    def test_minimal_jal_lora_train_uses_full_noisy_mask(self) -> None:
        import torch

        class FakeBackbone(torch.nn.Module):
            embed_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(3, 4)
                self.qkv = torch.nn.Linear(4, 4)

            def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                pooled = images.mean(dim=(2, 3))
                return {"x_norm_clstoken": self.qkv(self.stem(pooled))}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for idx, color in enumerate([(255, 0, 0), (0, 255, 0)]):
                path = root / f"{idx}.png"
                Image.new("RGB", (20, 20), color).save(path)
                paths.append(str(path))

            cfg = {
                "loss_type": "jal_ce",
                "jal_alpha": 1.0,
                "jal_beta": 1.0,
                "jal_a": 30.0,
                "jal_eps": 1.0e-8,
                "feature": {"device": "cpu", "input_size": 16},
                "lora": {"rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "qkv"},
                "lora_train": {
                    "epochs": 1,
                    "batch_size": 2,
                    "eval_batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "lora_lr": 1.0e-3,
                    "head_lr": 1.0e-3,
                    "weight_decay": 0.0,
                    "scheduler": "none",
                    "warmup_ratio": 0.0,
                    "amp": False,
                },
            }
            labels = np.array(["a", "b"], dtype=object)
            with mock.patch("gcdd.lora_training.load_dinov2_model", return_value=FakeBackbone()):
                result = train_dinov2_lora(
                    paths,
                    labels,
                    paths,
                    labels,
                    np.ones(2, dtype=bool),
                    cfg,
                    method="JAL-CE-DINOv2+LoRA (full noisy)",
                    seed=42,
                )

            self.assertEqual(result.summary["loss_type"], "jal_ce")
            self.assertEqual(result.summary["selection_mode"], "full_noisy")
            self.assertEqual(result.summary["train_samples"], 2)
            self.assertTrue(np.isfinite(result.logs[0]["loss"]))

            with self.assertRaisesRegex(ValueError, "full-noisy"):
                train_dinov2_lora(
                    paths,
                    labels,
                    paths,
                    labels,
                    np.array([True, False]),
                    cfg,
                    method="invalid filtered JAL",
                    seed=42,
                )

    def test_minimal_ce_lora_train_still_runs(self) -> None:
        import torch

        class FakeBackbone(torch.nn.Module):
            embed_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(3, 4)
                self.qkv = torch.nn.Linear(4, 4)

            def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                pooled = images.mean(dim=(2, 3))
                return {"x_norm_clstoken": self.qkv(self.stem(pooled))}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for idx, color in enumerate([(0, 0, 255), (255, 255, 0)]):
                path = root / f"ce_{idx}.png"
                Image.new("RGB", (20, 20), color).save(path)
                paths.append(str(path))
            labels = np.array(["a", "b"], dtype=object)
            cfg = {
                "feature": {"device": "cpu", "input_size": 16},
                "lora": {"rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "qkv"},
                "lora_train": {
                    "epochs": 1,
                    "batch_size": 2,
                    "eval_batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "lora_lr": 1.0e-3,
                    "head_lr": 1.0e-3,
                    "weight_decay": 0.0,
                    "scheduler": "none",
                    "warmup_ratio": 0.0,
                    "amp": False,
                },
            }
            with mock.patch("gcdd.lora_training.load_dinov2_model", return_value=FakeBackbone()):
                result = train_dinov2_lora(
                    paths,
                    labels,
                    paths,
                    labels,
                    np.ones(2, dtype=bool),
                    cfg,
                    method="DINOv2 LoRA all noisy samples",
                    seed=42,
                )
            self.assertEqual(result.summary["loss_type"], "ce")
            self.assertTrue(np.isfinite(result.logs[0]["train_loss"]))


if __name__ == "__main__":
    unittest.main()
