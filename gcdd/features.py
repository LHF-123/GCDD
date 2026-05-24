from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .data import ImageRecord
from .progress import log_stage, progress_iter


def extract_features(records: list[ImageRecord], cfg: dict, stage_name: str = "features") -> tuple[dict[str, np.ndarray], list[ImageRecord], list[dict[str, str]]]:
    backend = cfg["feature"]["backend"]
    if backend == "random":
        log_stage(f"[features] Building random {stage_name} features for {len(records)} images.")
        return extract_random_features(records, cfg), records, []
    if backend == "dinov2_vitb14":
        return extract_dinov2_features(records, cfg, stage_name)
    raise ValueError(f"Unsupported feature backend: {backend}")


def extract_random_features(records: list[ImageRecord], cfg: dict) -> dict[str, np.ndarray]:
    dim = int(cfg["feature"]["random_dim"])
    seed = int(cfg["train"]["seed"])
    features = {"cls": [], "gap": [], "top": []}
    for record in records:
        rng = np.random.default_rng(stable_seed(record.path, seed))
        for name in features:
            features[name].append(normalize(rng.normal(size=dim).astype(np.float32)))
    return {name: np.vstack(values).astype(np.float32) for name, values in features.items()}


def extract_dinov2_features(records: list[ImageRecord], cfg: dict, stage_name: str) -> tuple[dict[str, np.ndarray], list[ImageRecord], list[dict[str, str]]]:
    torch, transforms = import_torch_stack()
    device = resolve_device(torch, cfg["feature"]["device"])
    log_stage(f"[features] Loading DINOv2 ViT-B/14 on {device}.")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((int(cfg["feature"]["input_size"]), int(cfg["feature"]["input_size"]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    batch_size = int(cfg["feature"]["batch_size"])
    top_ratio = float(cfg["feature"]["top_patch_ratio"])
    collected: dict[str, list[np.ndarray]] = {"cls": [], "gap": [], "top": []}
    kept_all: list[ImageRecord] = []
    failures: list[dict[str, str]] = []
    total_batches = (len(records) + batch_size - 1) // batch_size
    log_stage(f"[features] Extracting {stage_name}: {len(records)} images, batch_size={batch_size}.")

    for start in progress_iter(range(0, len(records), batch_size), total=total_batches, desc=f"Extracting {stage_name}"):
        batch_records = records[start : start + batch_size]
        images = []
        kept_records = []
        for record in batch_records:
            try:
                with Image.open(record.path) as img:
                    images.append(transform(img.convert("RGB")))
                kept_records.append(record)
            except Exception as exc:  # noqa: BLE001 - log and keep V0 running.
                failures.append(
                    {
                        "index": str(record.index),
                        "path": str(record.path),
                        "label": record.label,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
        if not images:
            continue

        batch = torch.stack(images, dim=0).to(device)
        with torch.no_grad():
            cls, gap, top = forward_dinov2(model, batch, top_ratio)
        collected["cls"].append(cls.cpu().numpy().astype(np.float32))
        collected["gap"].append(gap.cpu().numpy().astype(np.float32))
        collected["top"].append(top.cpu().numpy().astype(np.float32))
        kept_all.extend(kept_records)

    if not collected["cls"]:
        raise RuntimeError("No features were extracted. Check dataset paths and image validity.")
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}, kept_all, failures


def forward_dinov2(model: Any, batch: Any, top_ratio: float) -> tuple[Any, Any, Any]:
    """Return CLS/GAP/Top-patch features while tolerating DINOv2 API differences."""
    import torch
    import torch.nn.functional as F

    if hasattr(model, "forward_features"):
        output = model.forward_features(batch)
        cls = output["x_norm_clstoken"]
        patches = output.get("x_norm_patchtokens")
    else:
        cls = model(batch)
        patches = None

    if patches is None:
        cls = F.normalize(cls, dim=1)
        return cls, cls, cls

    gap = patches.mean(dim=1)
    top_k = max(1, int(patches.shape[1] * top_ratio))
    response = torch.einsum("bld,bd->bl", patches, cls)
    top_idx = response.topk(top_k, dim=1).indices
    gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
    top = patches.gather(1, gather_idx).mean(dim=1)
    return F.normalize(cls, dim=1), F.normalize(gap, dim=1), F.normalize(top, dim=1)


def import_torch_stack() -> tuple[Any, Any]:
    try:
        import torch
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("DINOv2 backend requires torch and torchvision. Install requirements.txt first.") from exc
    return torch, transforms


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def stable_seed(path: Path, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)
