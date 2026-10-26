"""Regression tests for the isolated R4-C4 CLIP ViT-B/16 adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from gcdd.clip_vit import (
    CLIP_VITB16_BACKEND,
    CLIP_VITB16_CLS_DIM,
    extract_clip_final_ln_pre_projection_cls,
)
from gcdd.lora_training import (
    DINOv2LoRAClassifier,
    LoRAMultiheadAttention,
    freeze_all,
    inject_lora,
    lora_microbatch_slices,
    parse_target_modules,
    resolve_lora_batching,
)
from gcdd.r4_c4_clip_vitb16 import FORMAL_RUNS, FROZEN


WEIGHTS = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
WEIGHTS_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def clip_config(device: str = "cpu") -> dict:
    return {
        "feature": {
            "backend": CLIP_VITB16_BACKEND,
            "device": device,
            "input_size": 224,
            "clip_architecture": "ViT-B-16",
            "clip_pretrained": "openai",
            "clip_weights_path": str(WEIGHTS),
            "clip_weights_sha256": WEIGHTS_SHA256,
            "cls_feature_space": "final_ln_pre_projection",
        },
        "lora": {"rank": 8, "alpha": 16.0, "dropout": 0.05, "target_modules": "qkv"},
        "lora_train": {"batch_size": 80, "microbatch_size": 16},
    }


class ClipBackboneTest(unittest.TestCase):
    def test_fused_qkv_wrapper_is_identity_before_lora_training(self) -> None:
        base = torch.nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True).eval()
        values = torch.randn(3, 5, 8)
        with torch.no_grad():
            expected = base(values, values, values, need_weights=False)[0]
        freeze_all(base)
        wrapped = LoRAMultiheadAttention.make(torch, base, rank=2, alpha=4.0, dropout=0.0).eval()
        with torch.no_grad():
            actual = wrapped(values, values, values, need_weights=False)[0]
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapped.base.parameters()))
        self.assertEqual(6, sum(parameter.requires_grad for parameter in wrapped.parameters()))

    def test_effective_batch_microbatch_partition_preserves_last_remainder(self) -> None:
        self.assertEqual((80, 16), resolve_lora_batching({"batch_size": 80, "microbatch_size": 16}))
        self.assertEqual((slice(0, 16), slice(16, 32), slice(32, 48), slice(48, 64), slice(64, 80)), lora_microbatch_slices(80, 16))
        self.assertEqual((slice(0, 16), slice(16, 31)), lora_microbatch_slices(31, 16))

    def test_formal_matrix_is_fixed_36_runs(self) -> None:
        self.assertEqual(36, FORMAL_RUNS)
        self.assertEqual((1, 42, 88), FROZEN["training_seeds"])
        self.assertEqual(0.4, FROZEN["noise_rate"])
        self.assertEqual(80, FROZEN["effective_batch_size"])
        self.assertEqual(16, FROZEN["microbatch_size"])

    @unittest.skipUnless(WEIGHTS.is_file(), "local CLIP ViT-B/16 weights are not installed")
    def test_local_openai_clip_visual_has_preprojection_cls_and_qkv_lora_gradients(self) -> None:
        cfg = clip_config()
        model = DINOv2LoRAClassifier.make(torch, cfg, num_classes=3)
        freeze_all(model.backbone)
        modules = inject_lora(torch, model.backbone, parse_target_modules("qkv"), rank=8, alpha=16.0, dropout=0.05)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
        self.assertEqual(12, len(modules))
        self.assertTrue(all(name.endswith(".attn.qkv") for name in modules))
        self.assertTrue(all(not parameter.requires_grad for name, parameter in model.backbone.named_parameters() if "lora_" not in name))
        images = torch.randn(1, 3, 224, 224)
        cls = model.extract_cls_features(images)
        self.assertEqual((1, CLIP_VITB16_CLS_DIM), tuple(cls.shape))
        self.assertEqual((1, CLIP_VITB16_CLS_DIM), tuple(extract_clip_final_ln_pre_projection_cls(model.backbone, images).shape))
        model(images).sum().backward()
        self.assertGreater(sum(int(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))) for name, parameter in model.backbone.named_parameters() if "lora_" in name), 0)
        self.assertGreater(sum(int(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))) for parameter in model.head.parameters()), 0)

    def test_rejects_substituted_or_bad_hash_weights(self) -> None:
        cfg = clip_config()
        cfg["feature"]["clip_architecture"] = "ViT-B-32"
        with self.assertRaises(ValueError):
            DINOv2LoRAClassifier.make(torch, cfg, num_classes=2)
        with tempfile.TemporaryDirectory() as temporary:
            cfg = clip_config()
            path = Path(temporary) / "wrong.pt"
            path.write_bytes(b"not-a-clip-checkpoint")
            cfg["feature"]["clip_weights_path"] = str(path)
            with self.assertRaises(ValueError):
                DINOv2LoRAClassifier.make(torch, cfg, num_classes=2)


if __name__ == "__main__":
    unittest.main()
