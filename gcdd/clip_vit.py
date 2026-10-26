"""Strict local OpenCLIP ViT-B/16 visual-backbone adapter for R4-C4.

The adapter deliberately accepts only the pre-registered OpenAI CLIP
``ViT-B-16`` checkpoint.  It returns the visual tower only: the text tower is
discarded immediately after loading and is never used by classification,
selection, or prototype construction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


CLIP_VITB16_BACKEND = "clip_vitb16_openai"
CLIP_VITB16_ARCHITECTURE = "ViT-B-16"
CLIP_VITB16_PRETRAINED_SOURCE = "openai"
CLIP_VITB16_INPUT_SIZE = 224
CLIP_VITB16_CLS_DIM = 768
CLIP_VITB16_PROJECTED_DIM = 512
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLIP_VITB16_OPENAI_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CLIP_VITB16_OPENAI_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    f"{CLIP_VITB16_OPENAI_SHA256}/ViT-B-16.pt"
)


def sha256_file(path: Path) -> str:
    """Return the byte-level SHA-256 of a required local checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_clip_vitb16_weights(cache_dir: Path) -> Path:
    """Fetch exactly the registered OpenAI ViT-B/16 file and verify its hash.

    This is intentionally an explicit setup action rather than a training
    worker side effect. It pins the direct OpenAI URL and never accepts an
    alternate model or a Hugging Face mirror.
    """
    try:
        import open_clip.pretrained as pretrained
    except ImportError as exc:  # pragma: no cover - deployment dependency error
        raise RuntimeError("R4-C4 CLIP weight retrieval requires open_clip_torch==3.2.0.") from exc
    cfg = pretrained.get_pretrained_cfg(CLIP_VITB16_ARCHITECTURE, CLIP_VITB16_PRETRAINED_SOURCE)
    if not isinstance(cfg, dict) or cfg.get("url") != CLIP_VITB16_OPENAI_URL:
        raise RuntimeError("OpenCLIP's registered OpenAI ViT-B/16 URL does not match the frozen R4-C4 protocol.")
    cache_dir = cache_dir.expanduser()
    existing = cache_dir / "ViT-B-16.pt"
    if existing.is_file() and sha256_file(existing) == CLIP_VITB16_OPENAI_SHA256:
        return existing.resolve()
    target = Path(pretrained.download_pretrained(cfg, prefer_hf_hub=False, cache_dir=str(cache_dir))).resolve()
    if target.name != "ViT-B-16.pt":
        raise RuntimeError(f"Unexpected CLIP ViT-B/16 cache filename: {target.name}")
    actual_sha = sha256_file(target)
    if actual_sha != CLIP_VITB16_OPENAI_SHA256:
        raise RuntimeError(
            "Downloaded OpenAI CLIP ViT-B/16 SHA-256 does not match the frozen protocol: "
            f"{actual_sha} != {CLIP_VITB16_OPENAI_SHA256}"
        )
    return target


def validate_clip_vitb16_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the exact R4-C4 CLIP identity is declared."""
    feature = cfg.get("feature", {})
    if feature.get("backend") != CLIP_VITB16_BACKEND:
        raise ValueError(
            f"CLIP adapter requires feature.backend={CLIP_VITB16_BACKEND!r}."
        )
    if feature.get("clip_architecture") != CLIP_VITB16_ARCHITECTURE:
        raise ValueError("R4-C4 only permits the CLIP ViT-B/16 architecture.")
    if feature.get("clip_pretrained") != CLIP_VITB16_PRETRAINED_SOURCE:
        raise ValueError("R4-C4 only permits the declared OpenAI CLIP pretraining source.")
    if int(feature.get("input_size", 0)) != CLIP_VITB16_INPUT_SIZE:
        raise ValueError("CLIP ViT-B/16 input_size must remain 224.")
    if feature.get("cls_feature_space") != "final_ln_pre_projection":
        raise ValueError(
            "R4-C4 CLIP prototypes must use final-LN, pre-projection CLS features."
        )
    weight_text = str(feature.get("clip_weights_path", "")).strip()
    expected_sha = str(feature.get("clip_weights_sha256", "")).strip().lower()
    if not weight_text or not expected_sha:
        raise ValueError("CLIP weights path and SHA-256 must both be explicitly configured.")
    path = Path(weight_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Configured CLIP ViT-B/16 weights are unavailable: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            "Configured CLIP ViT-B/16 checkpoint SHA-256 does not match the frozen protocol: "
            f"{actual_sha} != {expected_sha}"
        )
    return {
        "architecture": CLIP_VITB16_ARCHITECTURE,
        "pretrained_source": CLIP_VITB16_PRETRAINED_SOURCE,
        "weights_path": str(path.resolve()),
        "weights_sha256": actual_sha,
        "input_size": CLIP_VITB16_INPUT_SIZE,
        "cls_feature_space": "final_ln_pre_projection",
        "cls_feature_dim": CLIP_VITB16_CLS_DIM,
        "projected_embedding_dim_not_used": CLIP_VITB16_PROJECTED_DIM,
        "text_encoder_used": False,
    }


def load_clip_vitb16_visual(torch: Any, cfg: dict[str, Any]) -> Any:
    """Load exactly one local OpenAI CLIP ViT-B/16 visual tower.

    OpenCLIP constructs a complete CLIP object while parsing the original
    checkpoint.  The returned object is *only* ``model.visual``; the caller
    does not retain the text tower.
    """
    identity = validate_clip_vitb16_config(cfg)
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - exercised on deployment hosts
        raise RuntimeError(
            "R4-C4 CLIP requires open_clip_torch. Install the pinned dependency "
            "before attempting a formal run."
        ) from exc
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_VITB16_ARCHITECTURE,
        pretrained=identity["weights_path"],
        device="cpu",
        # PyTorch 2.6 defaults torch.load(weights_only=True), which rejects
        # the original OpenAI TorchScript archive. This is safe here only
        # because validate_clip_vitb16_config already verified the exact
        # frozen OpenAI ViT-B/16 SHA-256 above.
        weights_only=False,
    )
    visual = model.visual
    if tuple(int(value) for value in visual.image_size) != (224, 224):
        raise RuntimeError(f"Loaded CLIP visual image size is not 224x224: {visual.image_size!r}")
    if int(getattr(visual, "output_dim", -1)) != CLIP_VITB16_PROJECTED_DIM:
        raise RuntimeError("Loaded CLIP visual projected embedding dimension is not 512.")
    if not hasattr(visual, "_embeds") or not hasattr(visual, "transformer") or not hasattr(visual, "ln_post"):
        raise RuntimeError("OpenCLIP VisionTransformer lacks the audited CLS extraction interface.")
    visual.r4_c4_clip_identity = identity
    # Do not retain the full CLIP module (and hence the unused text tower).
    del model
    return visual


def extract_clip_final_ln_pre_projection_cls(visual: Any, images: Any) -> Any:
    """Return the visual transformer's final-LN CLS token before projection.

    This is intentionally not ``visual(images)``: that public forward applies
    the 768->512 CLIP projection.  PGDF uses the current LoRA-adapted visual
    representation, matching the role of DINOv2's pre-head CLS feature.
    """
    tokens = visual._embeds(images)
    tokens = visual.transformer(tokens)
    if bool(getattr(visual, "final_ln_after_pool", False)):
        raise RuntimeError("Unsupported CLIP final_ln_after_pool layout for the frozen ViT-B/16 protocol.")
    tokens = visual.ln_post(tokens)
    cls = tokens[:, 0]
    if cls.ndim != 2 or int(cls.shape[1]) != CLIP_VITB16_CLS_DIM:
        raise RuntimeError(f"Unexpected CLIP final-LN CLS shape: {tuple(cls.shape)}")
    return cls


def clip_backbone_protocol_record(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a serializable pre-load protocol record for audit artifacts."""
    identity = validate_clip_vitb16_config(cfg)
    try:
        import open_clip

        identity["open_clip_version"] = str(getattr(open_clip, "__version__", "unknown"))
    except ImportError:
        identity["open_clip_version"] = "unavailable"
    identity.update(
        {
            "train_preprocess": {
                "resize_crop": "RandomResizedCrop(224, scale=(0.6,1.0))",
                "horizontal_flip": True,
                "normalization_mean": list(CLIP_MEAN),
                "normalization_std": list(CLIP_STD),
            },
            "eval_preprocess": {
                "resize": "Resize(224, bicubic)",
                "crop": "CenterCrop(224)",
                "normalization_mean": list(CLIP_MEAN),
                "normalization_std": list(CLIP_STD),
            },
            "lora_mapping": "PyTorch MultiheadAttention.in_proj_weight split into Q/K/V; rank-8 LoRA is applied to all three projections in each of 12 visual Transformer blocks.",
        }
    )
    return identity
