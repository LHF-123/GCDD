from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.lora_snscl import (
    ClassWiseQueue,
    ProjectionHead,
    SNSCLHealthError,
    StochasticHead,
    assert_finite_tensors,
    assert_finite_parameters,
    check_and_clip_gradients,
    compute_noise_metrics,
    evaluate_snscl_health,
    fit_gmm_reliability,
    forward_stochastic_fp32,
    gaussian_kl_loss,
    is_better_healthy_checkpoint,
    paper_ntcl_loss,
    reliability_weights,
    soft_cross_entropy,
    update_soft_labels,
)
from scripts.run_lora_snscl import build_epoch_callback, resolve_snscl_config, write_run_status


class SNSCLComponentsTest(unittest.TestCase):
    def test_projection_stochastic_shapes_normalization_and_kl(self) -> None:
        torch.manual_seed(2)
        features = torch.randn(4, 6)
        projection = ProjectionHead(6, 3)
        stochastic = StochasticHead(3, hidden_dim=8, output_dim=3)

        projected = projection(features)
        sampled, mu, logvar = stochastic(projected)
        loss = gaussian_kl_loss(mu, logvar)
        loss.backward()

        self.assertEqual(projected.shape, (4, 3))
        self.assertEqual(sampled.shape, (4, 3))
        self.assertTrue(torch.allclose(projected.norm(dim=1), torch.ones(4), atol=1.0e-5))
        self.assertTrue(torch.allclose(sampled.norm(dim=1), torch.ones(4), atol=1.0e-5))
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertIsNotNone(stochastic.mu.weight.grad)

    def test_stochastic_branch_forces_fp32(self) -> None:
        projection = ProjectionHead(4, 3)
        stochastic = StochasticHead(3, hidden_dim=8, output_dim=3)

        projected, sampled, mu, logvar = forward_stochastic_fp32(
            projection,
            stochastic,
            torch.randn(2, 4, dtype=torch.bfloat16),
        )

        for tensor in [projected, sampled, mu, logvar]:
            self.assertEqual(tensor.dtype, torch.float32)

    def test_gmm_reliability_direction_and_fallback(self) -> None:
        losses = np.array([0.05, 0.10, 0.15, 2.0, 2.2, 2.4], dtype=np.float32)
        result = fit_gmm_reliability(losses, seed=3)

        self.assertTrue(result.success)
        self.assertGreater(float(result.gamma[:3].mean()), float(result.gamma[3:].mean()))
        previous = np.linspace(0.1, 0.9, 6, dtype=np.float32)
        fallback = fit_gmm_reliability(np.ones(6, dtype=np.float32), previous_gamma=previous, seed=3)
        self.assertFalse(fallback.success)
        np.testing.assert_allclose(fallback.gamma, previous)
        self.assertIn("constant", fallback.reason)

    def test_soft_label_update_boundaries_and_soft_ce(self) -> None:
        previous = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        probabilities = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        noisy = previous.clone()
        omega = torch.tensor([1.0, 0.0])

        updated = update_soft_labels(previous, probabilities, noisy, omega, alpha=0.0)
        self.assertTrue(torch.allclose(updated[0], noisy[0]))
        self.assertTrue(torch.allclose(updated[1], probabilities[1]))
        self.assertTrue(torch.allclose(updated.sum(dim=1), torch.ones(2)))
        weights = reliability_weights(np.array([0.4, 0.5, 0.9], dtype=np.float32), 0.5)
        np.testing.assert_allclose(weights, np.array([0.4, 1.0, 1.0], dtype=np.float32))

        logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
        loss = soft_cross_entropy(logits, torch.tensor([[0.25, 0.75]]))
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        self.assertIsNotNone(logits.grad)

    def test_soft_label_alpha_point_nine_can_change_corrected_label(self) -> None:
        soft_label = torch.tensor([[1.0, 0.0]])
        probabilities = torch.tensor([[0.05, 0.95]])
        noisy = torch.tensor([[1.0, 0.0]])
        omega = torch.tensor([0.0])

        for _ in range(10):
            soft_label = update_soft_labels(soft_label, probabilities, noisy, omega, alpha=0.9)

        self.assertEqual(int(soft_label.argmax(dim=1).item()), 1)

    def test_queue_weighted_update_fifo_and_state_restore(self) -> None:
        queue = ClassWiseQueue(num_classes=2, queue_size=2, embedding_dim=2)
        embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0]])
        labels = torch.tensor([0, 0, 0, 1])
        weights = torch.tensor([1.0, 1.0, 1.0, 0.0])
        inserted = queue.enqueue(embeddings, labels, weights, random_values=torch.zeros(4))

        self.assertEqual(inserted, 3)
        self.assertEqual(queue.counts.tolist(), [2, 0])
        self.assertEqual(queue.pointers.tolist(), [1, 0])
        self.assertTrue(torch.allclose(queue.features[0, 0], torch.tensor([-1.0, 0.0])))
        self.assertTrue(torch.allclose(queue.features[0, 1], torch.tensor([0.0, 1.0])))

        restored = ClassWiseQueue(num_classes=2, queue_size=2, embedding_dim=2)
        restored.load_state_dict(queue.state_dict())
        self.assertTrue(torch.equal(restored.features, queue.features))
        self.assertTrue(torch.equal(restored.valid, queue.valid))
        self.assertTrue(torch.equal(restored.pointers, queue.pointers))

    def test_ntcl_matches_manual_paper_formula(self) -> None:
        queue = ClassWiseQueue(num_classes=2, queue_size=2, embedding_dim=2)
        queue.enqueue(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            torch.tensor([0, 0, 1]),
            torch.ones(3),
            random_values=torch.zeros(3),
        )
        anchor = torch.tensor([[1.0, 0.0]], requires_grad=True)
        loss, stats = paper_ntcl_loss(anchor, torch.tensor([0]), queue, temperature=1.0)

        denominator = torch.logsumexp(torch.tensor([1.0, 0.0, -1.0]), dim=0)
        expected = -torch.tensor([1.0 - denominator, 0.0 - denominator]).mean()
        self.assertTrue(torch.allclose(loss, expected, atol=1.0e-6))
        self.assertEqual(stats["num_valid_ntcl_anchors"], 1.0)
        self.assertEqual(stats["mean_positive_count"], 2.0)
        self.assertEqual(stats["mean_negative_count"], 1.0)

    def test_ntcl_empty_queue_and_no_valid_anchor_return_grad_zero(self) -> None:
        anchors = torch.randn(2, 3, requires_grad=True)
        queue = ClassWiseQueue(num_classes=2, queue_size=2, embedding_dim=3)
        loss, stats = paper_ntcl_loss(anchors, torch.tensor([0, 1]), queue, temperature=0.1)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(stats["num_valid_ntcl_anchors"], 0.0)
        self.assertIsNotNone(anchors.grad)

        queue.enqueue(torch.randn(1, 3), torch.tensor([0]), torch.ones(1), random_values=torch.zeros(1))
        loss, stats = paper_ntcl_loss(anchors, torch.tensor([0, 0]), queue, temperature=0.1)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(stats["num_valid_ntcl_anchors"], 0.0)

    def test_config_merge_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v1 = root / "v1.yaml"
            method = root / "method.yaml"
            v1.write_text(yaml.safe_dump({"feature": {"input_size": 224}, "lora_train": {"epochs": 10, "batch_size": 8}}), encoding="utf-8")
            method.write_text(yaml.safe_dump({"feature": {"input_size": 448}, "lora_train": {"epochs": 20}}), encoding="utf-8")
            args = argparse.Namespace(epochs=7, batch_size=None)

            cfg = resolve_snscl_config(v1, method, args)

        self.assertEqual(cfg["feature"]["input_size"], 448)
        self.assertEqual(cfg["lora_train"]["epochs"], 7)
        self.assertEqual(cfg["lora_train"]["batch_size"], 8)
        self.assertEqual(cfg["snscl"]["queue_size"], 32)
        self.assertEqual(cfg["snscl"]["label_ma_alpha"], 0.9)
        self.assertEqual(cfg["snscl"]["projection_lr"], 1.0e-4)
        self.assertEqual(cfg["snscl"]["stochastic_lr"], 1.0e-4)
        self.assertEqual(cfg["snscl"]["max_grad_norm"], 1.0)

    def test_noise_metrics_are_analysis_only_and_do_not_mutate_state(self) -> None:
        gamma = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        omega = np.array([1.0, 1.0, 0.2, 0.1], dtype=np.float32)
        gamma_before = gamma.copy()
        omega_before = omega.copy()

        metrics = compute_noise_metrics(gamma, omega, np.array([True, True, False, False]))

        self.assertGreater(float(metrics["gamma_clean_auc"]), 0.5)
        np.testing.assert_array_equal(gamma, gamma_before)
        np.testing.assert_array_equal(omega, omega_before)

    def test_non_finite_tensor_stops_training(self) -> None:
        with self.assertRaisesRegex(SNSCLHealthError, "loss_total"):
            assert_finite_tensors(3, 4, loss_total=torch.tensor(float("nan")), logits=torch.ones(2))

    def test_gradient_check_rejects_non_finite_and_clips_finite_norm(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        optimizer = torch.optim.SGD([{"name": "test", "params": [parameter]}], lr=0.1)
        parameter.grad = torch.tensor([3.0, 4.0])

        norm = check_and_clip_gradients(torch, optimizer, max_grad_norm=1.0, epoch_id=1, batch_id=2)

        self.assertAlmostEqual(norm, 5.0, places=5)
        self.assertLessEqual(float(parameter.grad.norm()), 1.00001)
        parameter.grad = torch.tensor([float("nan"), 0.0])
        with self.assertRaisesRegex(SNSCLHealthError, "test"):
            check_and_clip_gradients(torch, optimizer, max_grad_norm=1.0, epoch_id=1, batch_id=3)

    def test_parameter_check_detects_optimizer_pollution(self) -> None:
        module = torch.nn.Linear(2, 2)
        assert_finite_parameters(1, 1, classifier=module)
        with torch.no_grad():
            module.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(SNSCLHealthError, "classifier"):
            assert_finite_parameters(1, 2, classifier=module)

    def test_health_check_detects_silent_mechanism_failures(self) -> None:
        row = {
            "epoch": 7,
            "loss_total": 1.0,
            "loss_cls": 1.0,
            "loss_ntcl": 0.0,
            "loss_kl": 0.1,
            "mean_gamma": 1.0,
            "gamma_std": 0.0,
            "mean_omega": 1.0,
            "queue_fill_ratio": 0.0,
            "num_valid_ntcl_anchors": 0,
            "mean_grad_norm": 1.0,
            "max_mu_abs": 1.0,
            "min_logvar": -1.0,
            "max_logvar": 1.0,
            "max_model_param_abs": 1.0,
            "max_projection_param_abs": 1.0,
            "max_stochastic_param_abs": 1.0,
            "val_top1": 0.5,
            "val_top5": 0.8,
        }
        cfg = {
            "health_check_epoch": 7,
            "gmm_failure_patience": 2,
            "min_queue_fill_ratio": 1.0e-6,
            "min_valid_ntcl_anchors": 1,
            "min_gamma_std": 1.0e-6,
        }
        reliability = {"gmm_reason": "constant losses"}

        reasons = evaluate_snscl_health(row, reliability, cfg, consecutive_gmm_failures=2)

        self.assertTrue(any("GMM fallback" in reason for reason in reasons))
        self.assertTrue(any("queue_fill_ratio" in reason for reason in reasons))
        self.assertTrue(any("num_valid_ntcl_anchors" in reason for reason in reasons))
        self.assertTrue(any("gamma_std" in reason for reason in reasons))

    def test_epoch_callback_writes_live_progress_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            (output_dir / "reliability").mkdir()
            callback = build_epoch_callback(output_dir)
            callback(
                {
                    "method": "SNSCL-DINOv2+LoRA (adapted)",
                    "seed": 42,
                    "epoch": 1,
                    "logs": [{"method": "m", "seed": 42, "epoch": 1, "health_status": "ok"}],
                    "queue_rows": [{"method": "m", "seed": 42, "epoch": 1}],
                    "reliability_summary_rows": [],
                    "reliability_rows": [],
                    "health_status": "ok",
                    "health_reasons": "",
                }
            )

            self.assertTrue((output_dir / "snscl_seed42_train_log.csv").exists())
            self.assertTrue((output_dir / "snscl_seed42_queue_stats.csv").exists())
            self.assertTrue((output_dir / "snscl_seed42_reliability_summary.csv").exists())
            status = (output_dir / "run_status_seed42.json").read_text(encoding="utf-8")
            self.assertIn('"last_completed_epoch": 1', status)
            self.assertIn('"health_status": "ok"', status)

    def test_unhealthy_epoch_cannot_replace_best_checkpoint(self) -> None:
        best = {"val_top1": 0.7, "health_status": "ok"}
        unhealthy_better = {"val_top1": 0.9, "health_status": "failed"}
        healthy_better = {"val_top1": 0.8, "health_status": "ok"}

        self.assertFalse(is_better_healthy_checkpoint(unhealthy_better, best))
        self.assertTrue(is_better_healthy_checkpoint(healthy_better, best))

    def test_new_or_completed_run_clears_stale_exception_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_run_status(
                output_dir,
                42,
                status="failed",
                exception_type="SNSCLHealthError",
                exception="old failure",
                health_status="failed",
                health_reasons="old reason",
            )
            write_run_status(output_dir, 42, status="running", last_completed_epoch=0)
            running = json.loads((output_dir / "run_status_seed42.json").read_text(encoding="utf-8"))
            self.assertNotIn("exception_type", running)
            self.assertNotIn("exception", running)
            self.assertNotIn("health_status", running)
            self.assertNotIn("health_reasons", running)

            write_run_status(output_dir, 42, status="failed", exception_type="RuntimeError", exception="new failure")
            write_run_status(output_dir, 42, status="complete", last_completed_epoch=30)
            complete = json.loads((output_dir / "run_status_seed42.json").read_text(encoding="utf-8"))
            self.assertNotIn("exception_type", complete)
            self.assertNotIn("exception", complete)


if __name__ == "__main__":
    unittest.main()
