from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.lora_noisy_baselines import get_remember_rate, select_small_loss_indices, symmetric_kl_each
from gcdd.selection_utils import build_gt_clean_mask_from_noise_rows, build_mask_from_selection_rows
from tools.run_fine_dinov2_selector import compute_fine_scores, select_classwise


class ExternalBaselineTest(unittest.TestCase):
    def test_remember_rate_fixed_and_schedule(self) -> None:
        self.assertEqual(get_remember_rate(1, 5, 30, mode="fixed", fixed_rate=0.8, final_rate=0.6), 1.0)
        self.assertEqual(get_remember_rate(6, 5, 30, mode="fixed", fixed_rate=0.8, final_rate=0.6), 0.8)
        self.assertAlmostEqual(get_remember_rate(30, 5, 30, mode="schedule", fixed_rate=0.8, final_rate=0.6), 0.6)

    def test_small_loss_indices_keep_floor_at_least_one(self) -> None:
        import torch

        losses = torch.tensor([3.0, 1.0, 2.0])
        idx = select_small_loss_indices(torch, losses, 0.5)

        self.assertEqual(idx.tolist(), [1])

    def test_symmetric_kl_has_gradients_for_both_models(self) -> None:
        import torch

        logits_a = torch.tensor([[2.0, -1.0], [0.5, 1.0]], requires_grad=True)
        logits_b = torch.tensor([[1.0, 0.0], [1.5, -0.5]], requires_grad=True)
        loss = symmetric_kl_each(torch, logits_a, logits_b).sum()
        loss.backward()

        self.assertIsNotNone(logits_a.grad)
        self.assertIsNotNone(logits_b.grad)
        self.assertGreater(float(logits_a.grad.abs().sum()), 0.0)
        self.assertGreater(float(logits_b.grad.abs().sum()), 0.0)

    def test_fine_centered_scores_and_small_class_keep_all(self) -> None:
        features = np.array(
            [
                [10.0, 0.0],
                [11.0, 0.0],
                [12.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        )
        labels = np.array(["a", "a", "a", "b", "b"], dtype=object)

        scores, _, small = compute_fine_scores(features, labels, center=True, min_class_size=3)
        selected = select_classwise(scores, labels, 0.6, small)

        self.assertTrue(np.all(small[labels == "b"]))
        self.assertTrue(np.all(selected[labels == "b"]))
        self.assertEqual(int(selected[labels == "a"].sum()), 1)

    def test_static_selection_path_join_and_gt_clean_mask(self) -> None:
        train_paths = ["/mnt/data/train/A/1.jpg", "/mnt/data/train/A/2.jpg", "/mnt/data/train/B/3.jpg"]
        selection_rows = [
            {"path": "train/A/1.jpg", "state": "clean"},
            {"path": "train/A/2.jpg", "state": "ignored"},
            {"path": "train/B/3.jpg", "state": "clean"},
        ]
        result = build_mask_from_selection_rows(selection_rows, train_paths)

        self.assertEqual(result.selected_count, 2)
        self.assertEqual(result.mask.tolist(), [True, False, True])

        noise_rows = [
            {"path": "train/A/1.jpg", "split": "train", "clean_label": "A", "web_label": "A", "is_noisy": "0"},
            {"path": "train/A/2.jpg", "split": "train", "clean_label": "A", "web_label": "B", "is_noisy": "1"},
            {"path": "train/B/3.jpg", "split": "train", "clean_label": "B", "web_label": "B", "is_noisy": "0"},
        ]
        clean = build_gt_clean_mask_from_noise_rows(noise_rows, train_paths)

        self.assertEqual(clean.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
