from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .io_utils import read_csv


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "test", "val", "valid", "validation"}


@dataclass(frozen=True)
class ImageRecord:
    index: int
    path: Path
    label: str
    split: str


def build_debug_index(cfg: dict) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    """Build a small verified index and return bad images separately."""
    return build_verified_index(
        cfg,
        split=cfg["dataset"].get("split", "train"),
        samples_per_class=int(cfg["dataset"]["samples_per_class"]),
        max_classes=cfg["dataset"].get("max_classes"),
    )


def build_verified_index(
    cfg: dict,
    split: str,
    samples_per_class: int | None = None,
    max_classes: int | None = None,
) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    """Load a split, verify images, skip bad files, and optionally sample per class."""
    dataset_cfg = cfg["dataset"]
    root_text = str(dataset_cfg.get("root") or "")
    root = Path(root_text).expanduser()
    if not root_text and not dataset_cfg.get("index_file"):
        raise ValueError("dataset.root or dataset.index_file must be provided.")

    records = load_records(root, dataset_cfg.get("index_file", ""))
    split = normalize_split(split)
    records = [r for r in records if r.split == split]

    valid_records: list[ImageRecord] = []
    bad_images: list[dict[str, str]] = []
    for record in records:
        ok, reason = verify_image(record.path) if dataset_cfg.get("verify_images", True) else (True, "")
        if ok:
            valid_records.append(record)
        else:
            bad_images.append(
                {
                    "index": str(record.index),
                    "path": str(record.path),
                    "label": record.label,
                    "split": record.split,
                    "reason": reason,
                }
            )

    if samples_per_class is None:
        sampled = valid_records
        if max_classes is not None:
            allowed = set(sorted({record.label for record in sampled})[: int(max_classes)])
            sampled = [record for record in sampled if record.label in allowed]
    else:
        sampled = sample_per_class(
            valid_records,
            samples_per_class=samples_per_class,
            max_classes=max_classes,
            seed=int(cfg["train"]["seed"]),
        )
    return reindex(sampled), bad_images


def load_records(root: Path, index_file: str) -> list[ImageRecord]:
    if index_file:
        return load_records_from_csv(Path(index_file).expanduser(), root)
    return discover_records(root)


def load_records_from_csv(index_path: Path, root: Path) -> list[ImageRecord]:
    rows = read_csv(index_path)
    records: list[ImageRecord] = []
    for i, row in enumerate(rows):
        raw_path = row.get("path") or row.get("filepath") or row.get("image")
        label = row.get("web_label") or row.get("label") or row.get("class")
        if not raw_path or not label:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        split = normalize_split(row.get("split") or infer_split(path))
        records.append(ImageRecord(i, path, str(label), split))
    return records


def discover_records(root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
            continue
        split = infer_split(path)
        label = infer_label(path)
        records.append(ImageRecord(len(records), path, label, split))
    return records


def infer_split(path: Path) -> str:
    for part in path.parts:
        lower = part.lower()
        if lower in SPLIT_NAMES:
            return normalize_split(lower)
    return "train"


def infer_label(path: Path) -> str:
    # Prefer the class folder under a split folder, otherwise use the parent folder.
    parts = list(path.parts)
    for i, part in enumerate(parts[:-1]):
        if part.lower() in SPLIT_NAMES and i + 1 < len(parts) - 1:
            return parts[i + 1]
    return path.parent.name


def normalize_split(split: str) -> str:
    split = split.lower()
    if split in {"valid", "validation"}:
        return "val"
    return split


def verify_image(path: Path) -> tuple[bool, str]:
    """Detect WebFG bad images before they can break feature extraction."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB")
        return True, ""
    except Exception as exc:  # noqa: BLE001 - the reason is written to bad_images.csv.
        return False, f"{type(exc).__name__}: {exc}"


def sample_per_class(records: list[ImageRecord], samples_per_class: int, max_classes: int | None, seed: int) -> list[ImageRecord]:
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.label, []).append(record)

    labels = sorted(grouped)
    if max_classes is not None:
        labels = labels[: int(max_classes)]

    rng = random.Random(seed)
    sampled: list[ImageRecord] = []
    for label in labels:
        class_records = grouped[label][:]
        rng.shuffle(class_records)
        sampled.extend(class_records[:samples_per_class])
    return sorted(sampled, key=lambda r: (r.label, str(r.path)))


def reindex(records: list[ImageRecord]) -> list[ImageRecord]:
    return [ImageRecord(i, record.path, record.label, record.split) for i, record in enumerate(records)]
