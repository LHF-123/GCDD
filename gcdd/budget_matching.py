from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ClassBudgetSchedule:
    """Frozen per-update, per-noisy-class sample counts from one PGDF run."""

    seed: int
    source_path: str
    source_sha256: str
    source_retention_ratio: float
    source_proto_keep_ratio: float
    budgets: dict[int, dict[str, int]]
    rows: list[dict[str, Any]]


def load_pgdf_class_budget_schedule(
    path: Path,
    *,
    expected_seed: int,
    expected_update_epochs: Iterable[int],
    labels: np.ndarray,
    candidate_mask: np.ndarray,
) -> ClassBudgetSchedule:
    """Load only the actual PGDF class counts needed by budget matching.

    Sample identities and prototype/graph scores are deliberately discarded.
    The returned rows contain only the update, noisy class, and count budget.
    """

    if not path.exists():
        raise FileNotFoundError(f"PGDF per-class budget file does not exist: {path}")
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.shape != labels.shape:
        raise ValueError("candidate_mask must match labels shape.")

    expected_epochs = sorted({int(epoch) for epoch in expected_update_epochs})
    expected_classes = sorted(set(labels[candidate_mask].tolist()))
    class_totals = {
        label: int(np.sum(candidate_mask & (labels == label)))
        for label in expected_classes
    }
    required = {"seed", "epoch", "web_label", "total_count", "selected_count"}
    parsed: dict[tuple[int, str], dict[str, Any]] = {}
    retention_ratios: set[float] = set()
    proto_keep_ratios: set[float] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"PGDF budget file is missing columns {missing}: {path}")
        for line_number, row in enumerate(reader, start=2):
            try:
                seed = int(row["seed"])
                epoch = int(row["epoch"])
                noisy_label = str(row["web_label"])
                total_count = int(row["total_count"])
                selected_count = int(row["selected_count"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid PGDF budget value at {path}:{line_number}.") from exc
            if seed != int(expected_seed):
                raise ValueError(
                    f"PGDF budget seed mismatch at {path}:{line_number}: "
                    f"found {seed}, expected {expected_seed}."
                )
            if epoch not in expected_epochs:
                raise ValueError(
                    f"Unexpected PGDF budget update epoch {epoch}; expected {expected_epochs}."
                )
            if noisy_label not in class_totals:
                raise ValueError(f"Unknown noisy-label class {noisy_label!r} in {path}.")
            if total_count != class_totals[noisy_label]:
                raise ValueError(
                    f"PGDF class total mismatch for class {noisy_label!r}: "
                    f"found {total_count}, expected {class_totals[noisy_label]}."
                )
            if not 1 <= selected_count <= total_count:
                raise ValueError(
                    f"Invalid PGDF selected_count={selected_count} for class "
                    f"{noisy_label!r} with total_count={total_count}."
                )
            key = (epoch, noisy_label)
            if key in parsed:
                raise ValueError(f"Duplicate PGDF budget for epoch={epoch}, class={noisy_label!r}.")
            parsed[key] = {
                "seed": seed,
                "epoch": epoch,
                "noisy_label": noisy_label,
                "total_count": total_count,
                "selected_count": selected_count,
                "selected_ratio": selected_count / total_count,
            }
            if row.get("retention_ratio", "") != "":
                retention_ratios.add(float(row["retention_ratio"]))
            if row.get("proto_keep_ratio", "") != "":
                proto_keep_ratios.add(float(row["proto_keep_ratio"]))
            method = str(row.get("method", ""))
            if method and "PGDF" not in method:
                raise ValueError(f"Budget source is not labelled as a PGDF run at {path}:{line_number}.")

    expected_keys = {(epoch, label) for epoch in expected_epochs for label in expected_classes}
    missing_keys = sorted(expected_keys - set(parsed))
    if missing_keys:
        preview = ", ".join(f"({epoch}, {label})" for epoch, label in missing_keys[:5])
        raise ValueError(f"PGDF budget is incomplete; missing epoch/class entries: {preview}.")
    if len(retention_ratios) != 1 or len(proto_keep_ratios) != 1:
        raise ValueError(
            "PGDF budget must contain one consistent retention_ratio and one "
            "consistent proto_keep_ratio."
        )

    rows = [parsed[key] for key in sorted(parsed, key=lambda item: (item[0], item[1]))]
    budgets = {
        epoch: {label: int(parsed[(epoch, label)]["selected_count"]) for label in expected_classes}
        for epoch in expected_epochs
    }
    return ClassBudgetSchedule(
        seed=int(expected_seed),
        source_path=str(path),
        source_sha256=file_sha256(path),
        source_retention_ratio=next(iter(retention_ratios)),
        source_proto_keep_ratio=next(iter(proto_keep_ratios)),
        budgets=budgets,
        rows=rows,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
