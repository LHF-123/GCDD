from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SelectionMaskResult:
    mask: np.ndarray
    matched_count: int
    selected_count: int


def normalize_path_text(path: str | Path) -> str:
    text = str(path).strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    return text.strip("/")


def path_key_candidates(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    normalized = normalize_path_text(path)
    if not normalized:
        return []

    candidates: list[str] = []

    def add(value: str) -> None:
        value = value.strip("/")
        if value and value not in candidates:
            candidates.append(value)

    add(normalized)
    for prefix in ("images/", "train/", "test/", "val/"):
        if normalized.startswith(prefix):
            add(normalized)
            if prefix == "images/":
                add(normalized[len(prefix) :])

    wrapped = f"/{normalized}"
    for marker in ("/images/", "/train/", "/test/", "/val/"):
        if marker not in wrapped:
            continue
        _, after = wrapped.split(marker, 1)
        if marker == "/images/":
            add(after)
            add(f"images/{after}")
        else:
            add(f"{marker.strip('/')}/{after}")
            add(after)

    parts = [part for part in normalized.split("/") if part]
    for count in (3, 2):
        if len(parts) >= count:
            add("/".join(parts[-count:]))
    return candidates


def build_unique_path_map(paths: list[str], source_name: str = "paths") -> dict[str, int]:
    key_to_index: dict[str, int] = {}
    for idx, path in enumerate(paths):
        for key in path_key_candidates(path):
            existing = key_to_index.get(key)
            if existing is None:
                key_to_index[key] = idx
            elif existing != idx:
                raise ValueError(f"Ambiguous path key '{key}' in {source_name}: rows {existing} and {idx}.")
    return key_to_index


def build_mask_from_selection_rows(
    rows: list[dict[str, str]],
    train_paths: list[str],
    *,
    selected_state: str = "clean",
    require_full_coverage: bool = True,
) -> SelectionMaskResult:
    if not rows:
        raise ValueError("Selection CSV is empty.")
    required = {"path", "state"}
    missing_fields = required - set(rows[0].keys())
    if missing_fields:
        raise ValueError(f"Selection CSV is missing required fields: {sorted(missing_fields)}")

    key_to_train = build_unique_path_map(train_paths, "train paths")
    matched = np.zeros(len(train_paths), dtype=bool)
    mask = np.zeros(len(train_paths), dtype=bool)
    seen_rows: set[int] = set()

    for row_num, row in enumerate(rows):
        match_idx = resolve_path_index(row["path"], key_to_train)
        if match_idx is None:
            if require_full_coverage:
                raise ValueError(f"Selection row {row_num} does not match any train path: {row['path']}")
            continue
        if match_idx in seen_rows:
            raise ValueError(f"Selection CSV contains duplicate train path for index {match_idx}: {row['path']}")
        seen_rows.add(match_idx)
        matched[match_idx] = True
        mask[match_idx] = row["state"] == selected_state

    if require_full_coverage and not np.all(matched):
        missing = np.where(~matched)[0]
        preview = ", ".join(str(int(idx)) for idx in missing[:10])
        raise ValueError(f"Selection CSV is missing {len(missing)} train samples. Missing train indices: {preview}")

    return SelectionMaskResult(mask=mask, matched_count=int(matched.sum()), selected_count=int(mask.sum()))


def resolve_path_index(path: str, key_to_index: dict[str, int]) -> int | None:
    for key in path_key_candidates(path):
        if key in key_to_index:
            return key_to_index[key]
    return None


def build_gt_clean_mask_from_noise_rows(rows: list[dict[str, str]], train_paths: list[str]) -> np.ndarray:
    key_to_clean: dict[str, bool] = {}
    for row in rows:
        if row.get("split", "train").lower() != "train":
            continue
        if row.get("is_noisy", "") != "":
            is_clean = row["is_noisy"] == "0"
        else:
            is_clean = row.get("clean_label") == row.get("web_label")
        for raw in [row.get("path", ""), row.get("abs_path", "")]:
            for key in path_key_candidates(raw):
                key_to_clean.setdefault(key, bool(is_clean))

    clean = np.zeros(len(train_paths), dtype=bool)
    missing = []
    for idx, path in enumerate(train_paths):
        matched = False
        for key in path_key_candidates(path):
            if key in key_to_clean:
                clean[idx] = key_to_clean[key]
                matched = True
                break
        if not matched:
            missing.append(path)
    if missing:
        preview = "; ".join(str(path) for path in missing[:5])
        raise ValueError(f"Noise index missing {len(missing)} train paths. Examples: {preview}")
    return clean
