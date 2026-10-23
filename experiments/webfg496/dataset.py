"""Read-only WebFG-496 list-file adapter and data-integrity inventory."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class WebFGRecord:
    """A single list-file sample with an observed web label."""

    relative_path: str
    label: int
    effective_path: str | None = None


@dataclass(frozen=True)
class WebFGSplit:
    """Absolute sample paths and labels ready for the public training APIs."""

    paths: list[str]
    labels: list[str]


def parse_list_file(path: Path) -> list[WebFGRecord]:
    """Parse the official ``relative/path.jpg: integer_label`` list format."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing WebFG list file: {path}")
    records: list[WebFGRecord] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number} is not '<relative path>: <integer label>'.")
        relative_text, label_text = line.rsplit(":", 1)
        relative_text = relative_text.strip().replace("\\", "/")
        try:
            label = int(label_text.strip())
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} has a non-integer label: {label_text!r}") from exc
        pure = PurePosixPath(relative_text)
        if not relative_text or pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"{path}:{line_number} has unsafe relative path: {relative_text!r}")
        if pure.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{path}:{line_number} is not an image path: {relative_text!r}")
        records.append(WebFGRecord(relative_text, label))
    if not records:
        raise ValueError(f"WebFG list file is empty: {path}")
    return records


def load_webfg_splits(
    data_root: Path,
    dataset: str,
    *,
    train_manifest: Path | None = None,
) -> tuple[WebFGSplit, WebFGSplit]:
    """Return the verified train manifest and immutable official ``val`` split."""
    dataset_root = Path(data_root) / dataset
    source_train_records = parse_list_file(dataset_root / "train-list.txt")
    train_records = (
        load_effective_train_manifest(Path(train_manifest), source_train_records)
        if train_manifest is not None
        else source_train_records
    )
    eval_records = parse_list_file(dataset_root / "val-list.txt")
    return _to_split(dataset_root, train_records), _to_split(dataset_root, eval_records)


def _to_split(dataset_root: Path, records: Iterable[WebFGRecord]) -> WebFGSplit:
    records = list(records)
    paths = [record.effective_path or str(dataset_root / record.relative_path) for record in records]
    labels = [str(record.label) for record in records]
    return WebFGSplit(paths=paths, labels=labels)


def inventory_webfg(
    data_root: Path,
    expected_classes: dict[str, int],
    *,
    verify_images: bool,
    train_manifests: dict[str, Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit list files, directory contents, labels, and image readability.

    The source data are only read.  Every listed image is checked when
    ``verify_images`` is true; this intentionally detects corrupt files before
    a long remote training matrix is scheduled.
    """
    rows: list[dict[str, Any]] = []
    mappings: dict[str, Any] = {}
    for dataset, expected_count in expected_classes.items():
        root = Path(data_root) / dataset
        source_train_records = parse_list_file(root / "train-list.txt")
        manifest_path = None if train_manifests is None else train_manifests.get(dataset)
        train_records = (
            load_effective_train_manifest(manifest_path, source_train_records)
            if manifest_path is not None
            else source_train_records
        )
        val_records = parse_list_file(root / "val-list.txt")
        train_row, train_map = _inventory_split(
            root,
            dataset,
            "train",
            train_records,
            verify_images,
            source_records=source_train_records,
            manifest_path=manifest_path,
        )
        val_row, val_map = _inventory_split(root, dataset, "val", val_records, verify_images)
        merged = _merge_class_mapping(dataset, train_map, val_map)
        all_labels = sorted({record.label for record in [*source_train_records, *val_records]})
        contiguous = all_labels == list(range(int(expected_count)))
        class_count_matches = len(all_labels) == int(expected_count)
        for row in (train_row, val_row):
            row["expected_class_count"] = int(expected_count)
            row["observed_class_count"] = len(all_labels)
            row["labels_contiguous_zero_based"] = contiguous
            row["class_count_matches_expected"] = class_count_matches
            row["dataset_status"] = "PASS" if _row_passes(row) and contiguous and class_count_matches else "FAIL"
            rows.append(row)
        mappings[dataset] = {
            "dataset_root": str(root),
            "expected_class_count": int(expected_count),
            "observed_class_count": len(all_labels),
            "labels_contiguous_zero_based": contiguous,
            "classes": merged,
        }
    return rows, mappings


def _inventory_split(
    root: Path,
    dataset: str,
    split: str,
    records: list[WebFGRecord],
    verify_images: bool,
    *,
    source_records: list[WebFGRecord] | None = None,
    manifest_path: Path | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing WebFG split directory: {split_dir}")
    source_records = records if source_records is None else source_records
    listed_paths = [record.relative_path for record in source_records]
    listed_counter = Counter(listed_paths)
    listed_set = set(listed_paths)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    missing = sorted(path for path in listed_set if not (root / path).is_file())
    unlisted = sorted(actual_paths - listed_set)
    unreadable: list[str] = []
    if verify_images:
        unreadable = _verify_records(root, records)
    label_map: dict[int, dict[str, Any]] = {}
    for record in records:
        parts = PurePosixPath(record.relative_path).parts
        class_name = parts[1] if len(parts) >= 3 and parts[0] == split else ""
        entry = label_map.setdefault(record.label, {"label": record.label, "class_names": set(), "count": 0})
        entry["class_names"].add(class_name)
        entry["count"] += 1
    class_dir_count = sum(1 for path in split_dir.iterdir() if path.is_dir())
    row: dict[str, Any] = {
        "dataset": dataset,
        "dataset_root": str(root),
        "split": split,
        "list_file": str(root / f"{split}-list.txt"),
        "list_entries": len(source_records),
        "effective_list_entries": len(records),
        "excluded_from_effective_train": len(source_records) - len(records) if split == "train" else 0,
        "effective_train_manifest": str(manifest_path) if manifest_path is not None else "",
        "unique_list_entries": len(listed_set),
        "duplicate_list_entries": sum(count - 1 for count in listed_counter.values() if count > 1),
        "actual_image_files": len(actual_paths),
        "class_directory_count": class_dir_count,
        "missing_listed_images": len(missing),
        "unlisted_images": len(unlisted),
        "unreadable_images": len(unreadable),
        "min_label": min(record.label for record in records),
        "max_label": max(record.label for record in records),
        "label_count_in_split": len(label_map),
        "missing_examples": " | ".join(missing[:5]),
        "unlisted_examples": " | ".join(unlisted[:5]),
        "unreadable_examples": " | ".join(unreadable[:5]),
    }
    return row, label_map


def _verify_records(root: Path, records: Iterable[WebFGRecord]) -> list[str]:
    from PIL import Image

    unreadable: list[str] = []
    for record in sorted(records, key=lambda item: item.relative_path):
        path = Path(record.effective_path) if record.effective_path else root / record.relative_path
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - inventory reports offending sample paths.
            unreadable.append(f"{record.relative_path} ({type(exc).__name__}: {exc})")
    return unreadable


def load_effective_train_manifest(path: Path, source_records: list[WebFGRecord]) -> list[WebFGRecord]:
    """Load a WebFG-only manifest while proving labels and source IDs are unchanged."""
    if not path.is_file():
        raise FileNotFoundError(f"Verified WebFG train manifest is missing: {path}")
    source_labels = {record.relative_path: record.label for record in source_records}
    records: list[WebFGRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "observed_label", "effective_path", "resolution"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}.")
        for line_number, row in enumerate(reader, start=2):
            relative_path = str(row["relative_path"]).replace("\\", "/")
            if relative_path not in source_labels:
                raise ValueError(f"{path}:{line_number} is not in the official training list: {relative_path}")
            if relative_path in seen:
                raise ValueError(f"{path}:{line_number} duplicates {relative_path}")
            try:
                label = int(str(row["observed_label"]))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} has invalid observed_label") from exc
            if label != source_labels[relative_path]:
                raise ValueError(f"{path}:{line_number} changes observed label for {relative_path}")
            effective_path = Path(str(row["effective_path"]))
            if not effective_path.is_file():
                raise FileNotFoundError(f"{path}:{line_number} effective path is missing: {effective_path}")
            records.append(WebFGRecord(relative_path, label, str(effective_path)))
            seen.add(relative_path)
    if not records:
        raise ValueError(f"Verified WebFG train manifest is empty: {path}")
    return records


def _merge_class_mapping(
    dataset: str,
    train_map: dict[int, dict[str, Any]],
    val_map: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = sorted(set(train_map) | set(val_map))
    merged: list[dict[str, Any]] = []
    for label in labels:
        train_entry = train_map.get(label, {"class_names": set(), "count": 0})
        val_entry = val_map.get(label, {"class_names": set(), "count": 0})
        train_names = sorted(train_entry["class_names"])
        val_names = sorted(val_entry["class_names"])
        merged.append(
            {
                "label": int(label),
                "train_class_names": train_names,
                "evaluation_class_names": val_names,
                "mapping_consistent": len(train_names) == 1 and train_names == val_names,
                "train_images": int(train_entry["count"]),
                "evaluation_images": int(val_entry["count"]),
                "dataset": dataset,
            }
        )
    return merged


def _row_passes(row: dict[str, Any]) -> bool:
    return (
        int(row["duplicate_list_entries"]) == 0
        and int(row["missing_listed_images"]) == 0
        and int(row["unlisted_images"]) == 0
        and int(row["unreadable_images"]) == 0
        and int(row["list_entries"]) == int(row["actual_image_files"])
    )
