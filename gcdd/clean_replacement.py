"""Fail-closed construction of R4-C2 fixed-mask replay interventions.

This module reads only saved, instrumented PGDF selection facts.  It never
recomputes a loss, prototype score, or selection member.  Oracle clean labels
are used solely to construct the pre-registered O-group replacements.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPLACEMENT_RATE = 0.20
REPLACEMENT_RULE_ID = "r4_c2_oracle_clean_replacement_sha256_v1"


@dataclass(frozen=True)
class ReplacementTrajectory:
    """One dataset/seed's frozen R and O masks plus auditable derivation rows."""

    dataset: str
    training_seed: int
    replay_masks: dict[int, np.ndarray]
    oracle_masks: dict[int, np.ndarray]
    membership_rows: list[dict[str, Any]]
    classwise_rows: list[dict[str, Any]]
    summary: dict[str, Any]


class ReplacementInputError(ValueError):
    """Raised when a historical selection asset cannot safely define a replay."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one immutable source asset."""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_oracle_clean_replacement(
    *,
    dataset: str,
    training_seed: int,
    train_paths: list[str],
    observed_labels: np.ndarray,
    clean_labels: np.ndarray,
    training_pool_mask: np.ndarray,
    selection_samples_path: Path,
    selection_per_class_path: Path,
    selection_updates_path: Path,
    expected_update_epochs: Iterable[int],
    replacement_rate: float = REPLACEMENT_RATE,
) -> ReplacementTrajectory:
    """Freeze R masks and deterministically create budget-preserving O masks.

    ``training_pool_mask`` is the fixed clean-validation training pool.  Saved
    stable ids must cover that pool exactly at every update.  Any identity,
    class-count, fallback, or oracle-alignment discrepancy aborts before a
    training run can be created.
    """
    if not 0.0 <= float(replacement_rate) <= 1.0:
        raise ReplacementInputError("replacement_rate must lie in [0, 1].")
    observed = np.asarray(observed_labels, dtype=str)
    clean = np.asarray(clean_labels, dtype=str)
    pool = np.asarray(training_pool_mask, dtype=bool)
    if not (observed.shape == clean.shape == pool.shape == (len(train_paths),)):
        raise ReplacementInputError("Paths, observed labels, clean labels, and pool mask must align.")
    expected_epochs = tuple(sorted({int(epoch) for epoch in expected_update_epochs}))
    if not expected_epochs:
        raise ReplacementInputError("At least one historical selection update is required.")

    sample_rows = _read_csv(selection_samples_path)
    class_rows = _read_csv(selection_per_class_path)
    update_rows = _read_csv(selection_updates_path)
    _require_columns(
        sample_rows,
        {
            "training_seed",
            "epoch",
            "stable_sample_id",
            "sample_path",
            "observed_label",
            "loss_candidate",
            "prototype_candidate",
            "final_selected",
            "selection_strategy",
        },
        selection_samples_path,
    )
    _require_columns(
        class_rows,
        {
            "training_seed",
            "epoch",
            "observed_label",
            "total_count",
            "loss_candidate_count",
            "prototype_candidate_count",
            "intersection_count",
            "final_selected_count",
            "fallback_triggered",
            "selection_strategy",
        },
        selection_per_class_path,
    )
    _require_columns(
        update_rows,
        {
            "training_seed",
            "update_epoch",
            "full_training_pool_size",
            "selected_total",
            "fallback_count",
            "selection_strategy",
        },
        selection_updates_path,
    )

    candidate_indices = np.flatnonzero(pool)
    candidate_set = {int(index) for index in candidate_indices}
    classes = tuple(sorted(set(observed[pool].tolist())))
    clean_mask = pool & (clean == observed)
    sample_by_epoch = _index_sample_rows(
        sample_rows,
        dataset=dataset,
        training_seed=training_seed,
        expected_epochs=expected_epochs,
        train_paths=train_paths,
        observed_labels=observed,
        candidate_indices=candidate_indices,
    )
    class_by_epoch = _index_class_rows(
        class_rows,
        training_seed=training_seed,
        expected_epochs=expected_epochs,
        expected_classes=classes,
    )
    update_by_epoch = _index_update_rows(
        update_rows,
        training_seed=training_seed,
        expected_epochs=expected_epochs,
    )

    replay_masks: dict[int, np.ndarray] = {}
    oracle_masks: dict[int, np.ndarray] = {}
    membership_rows: list[dict[str, Any]] = []
    classwise_rows: list[dict[str, Any]] = []
    total_replacements = 0

    for epoch in expected_epochs:
        rows_by_index = sample_by_epoch[epoch]
        source_mask = np.zeros(len(train_paths), dtype=bool)
        loss_mask = np.zeros(len(train_paths), dtype=bool)
        proto_mask = np.zeros(len(train_paths), dtype=bool)
        for index, row in rows_by_index.items():
            source_mask[index] = _parse_yes_no(row["final_selected"], "final_selected", selection_samples_path)
            loss_mask[index] = _parse_yes_no(row["loss_candidate"], "loss_candidate", selection_samples_path)
            proto_mask[index] = _parse_yes_no(row["prototype_candidate"], "prototype_candidate", selection_samples_path)
        if np.any(source_mask & ~pool):
            raise ReplacementInputError(f"Epoch {epoch} source selection leaves the fixed training pool.")
        if not np.array_equal(source_mask[pool], (loss_mask & proto_mask)[pool]):
            raise ReplacementInputError(
                f"Epoch {epoch} final selection is not the strict saved L∩P mask; fallback or identity drift is present."
            )
        _validate_source_update(
            epoch=epoch,
            source_mask=source_mask,
            pool=pool,
            update_row=update_by_epoch[epoch],
            selection_updates_path=selection_updates_path,
        )

        oracle_mask = source_mask.copy()
        removed = np.zeros(len(train_paths), dtype=bool)
        added = np.zeros(len(train_paths), dtype=bool)
        class_replacements: dict[str, int] = {}

        for label in classes:
            class_mask = pool & (observed == label)
            source_class = class_by_epoch[epoch][label]
            _validate_source_class(
                epoch=epoch,
                label=label,
                class_mask=class_mask,
                source_mask=source_mask,
                loss_mask=loss_mask,
                proto_mask=proto_mask,
                source_class=source_class,
                selection_per_class_path=selection_per_class_path,
            )
            a_indices = np.flatnonzero(class_mask & clean_mask & loss_mask & proto_mask)
            b_indices = np.flatnonzero(class_mask & clean_mask & loss_mask & ~proto_mask)
            replacement_count = min(int(np.floor(float(replacement_rate) * len(a_indices))), len(b_indices))
            class_replacements[label] = replacement_count
            removed_indices = _deterministic_choose(
                a_indices,
                count=replacement_count,
                dataset=dataset,
                seed=training_seed,
                epoch=epoch,
                observed_label=label,
                role="remove_A",
            )
            added_indices = _deterministic_choose(
                b_indices,
                count=replacement_count,
                dataset=dataset,
                seed=training_seed,
                epoch=epoch,
                observed_label=label,
                role="add_B",
            )
            if np.any(~source_mask[removed_indices]) or np.any(source_mask[added_indices]):
                raise ReplacementInputError(f"Epoch {epoch}, class {label}: invalid A/B replacement membership.")
            oracle_mask[removed_indices] = False
            oracle_mask[added_indices] = True
            removed[removed_indices] = True
            added[added_indices] = True
            total_replacements += replacement_count

        _validate_replacement_invariants(
            epoch=epoch,
            source_mask=source_mask,
            oracle_mask=oracle_mask,
            removed=removed,
            added=added,
            clean_mask=clean_mask,
            pool=pool,
        )
        replay_masks[epoch] = source_mask
        oracle_masks[epoch] = oracle_mask

        for label in classes:
            class_mask = pool & (observed == label)
            r_active = int(np.sum(source_mask & class_mask))
            o_active = int(np.sum(oracle_mask & class_mask))
            r_clean = int(np.sum(source_mask & class_mask & clean_mask))
            o_clean = int(np.sum(oracle_mask & class_mask & clean_mask))
            r_noisy = r_active - r_clean
            o_noisy = o_active - o_clean
            h = int(class_replacements[label])
            classwise_rows.append(
                {
                    "dataset": dataset,
                    "training_seed": int(training_seed),
                    "selection_epoch": int(epoch),
                    "observed_label": label,
                    "source_total_count": int(np.sum(class_mask)),
                    "source_final_selected_count": int(class_by_epoch[epoch][label]["final_selected_count"]),
                    "r_active_count": r_active,
                    "o_active_count": o_active,
                    "r_clean_active_count": r_clean,
                    "o_clean_active_count": o_clean,
                    "r_noisy_active_count": r_noisy,
                    "o_noisy_active_count": o_noisy,
                    "a_clean_l_and_p_count": int(np.sum(class_mask & clean_mask & loss_mask & proto_mask)),
                    "b_clean_l_not_p_count": int(np.sum(class_mask & clean_mask & loss_mask & ~proto_mask)),
                    "replacement_count": h,
                    "replacement_rate_target": float(replacement_rate),
                    "replacement_rate_realized_of_a": float(h / np.sum(class_mask & clean_mask & loss_mask & proto_mask))
                    if np.any(class_mask & clean_mask & loss_mask & proto_mask)
                    else 0.0,
                    "r_o_active_count_match": r_active == o_active,
                    "r_o_clean_count_match": r_clean == o_clean,
                    "r_o_noisy_count_match": r_noisy == o_noisy,
                    "unmodified_members_match": bool(
                        np.array_equal(
                            source_mask[class_mask & ~removed & ~added],
                            oracle_mask[class_mask & ~removed & ~added],
                        )
                    ),
                    "status": "PASS",
                }
            )

        for index in candidate_indices:
            row = rows_by_index[int(index)]
            source_group = _source_group(
                is_clean=bool(clean_mask[index]),
                loss_candidate=bool(loss_mask[index]),
                prototype_candidate=bool(proto_mask[index]),
            )
            intervention_role = "removed_A" if removed[index] else "added_B" if added[index] else "unchanged"
            membership_rows.append(
                {
                    "dataset": dataset,
                    "training_seed": int(training_seed),
                    "selection_epoch": int(epoch),
                    "stable_sample_id": row["stable_sample_id"],
                    "train_index": int(index),
                    "sample_path": row["sample_path"],
                    "observed_label": str(observed[index]),
                    "clean_label_oracle": str(clean[index]),
                    "is_clean_oracle": bool(clean_mask[index]),
                    "loss_candidate": bool(loss_mask[index]),
                    "prototype_candidate": bool(proto_mask[index]),
                    "source_group": source_group,
                    "r_active": bool(source_mask[index]),
                    "o_active": bool(oracle_mask[index]),
                    "intervention_role": intervention_role,
                    "replacement_rule": REPLACEMENT_RULE_ID,
                }
            )

    summary = {
        "dataset": dataset,
        "training_seed": int(training_seed),
        "update_epochs": list(expected_epochs),
        "training_pool_count": int(pool.sum()),
        "total_replacements": int(total_replacements),
        "replacement_rate_target": float(replacement_rate),
        "replacement_rule": REPLACEMENT_RULE_ID,
        "selection_samples_sha256": file_sha256(selection_samples_path),
        "selection_per_class_sha256": file_sha256(selection_per_class_path),
        "selection_updates_sha256": file_sha256(selection_updates_path),
    }
    return ReplacementTrajectory(
        dataset=dataset,
        training_seed=int(training_seed),
        replay_masks=replay_masks,
        oracle_masks=oracle_masks,
        membership_rows=membership_rows,
        classwise_rows=classwise_rows,
        summary=summary,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ReplacementInputError(f"Required immutable source asset is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _require_columns(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    if not rows:
        raise ReplacementInputError(f"Historical selection asset is empty: {path}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ReplacementInputError(f"Historical selection asset lacks {missing}: {path}")


def _parse_yes_no(value: Any, field: str, path: Path) -> bool:
    text = str(value).strip().lower()
    if text == "yes":
        return True
    if text == "no":
        return False
    raise ReplacementInputError(f"Expected yes/no for {field} in {path}, found {value!r}.")


def _parse_train_index(stable_sample_id: str, path: Path) -> int:
    prefix = "train_index:"
    if not str(stable_sample_id).startswith(prefix):
        raise ReplacementInputError(f"Unsupported stable sample id {stable_sample_id!r} in {path}.")
    try:
        return int(str(stable_sample_id)[len(prefix) :])
    except ValueError as exc:
        raise ReplacementInputError(f"Invalid stable sample id {stable_sample_id!r} in {path}.") from exc


def _index_sample_rows(
    rows: list[dict[str, str]],
    *,
    dataset: str,
    training_seed: int,
    expected_epochs: tuple[int, ...],
    train_paths: list[str],
    observed_labels: np.ndarray,
    candidate_indices: np.ndarray,
) -> dict[int, dict[int, dict[str, str]]]:
    del dataset  # The filename/run mapping binds dataset; row schema has no mandatory canonical spelling.
    result: dict[int, dict[int, dict[str, str]]] = {epoch: {} for epoch in expected_epochs}
    candidate_set = {int(index) for index in candidate_indices}
    for row in rows:
        try:
            seed = int(row["training_seed"])
            epoch = int(row["epoch"])
        except (TypeError, ValueError) as exc:
            raise ReplacementInputError("Historical sample row has invalid training_seed or epoch.") from exc
        if seed != int(training_seed):
            raise ReplacementInputError(f"Historical sample seed {seed} does not match requested {training_seed}.")
        if epoch not in result:
            raise ReplacementInputError(f"Unexpected historical sample update epoch {epoch}.")
        if row.get("selection_strategy") != "loss_and_proto":
            raise ReplacementInputError("Historical masks must originate from PGDF loss_and_proto selection.")
        index = _parse_train_index(row["stable_sample_id"], Path("selection_samples.csv"))
        if index not in candidate_set:
            raise ReplacementInputError(f"Historical stable id {index} is outside the fixed validation training pool.")
        if index in result[epoch]:
            raise ReplacementInputError(f"Duplicate historical stable id {index} at epoch {epoch}.")
        if row.get("sample_path") != train_paths[index]:
            raise ReplacementInputError(f"Stable id {index} maps to a different path at historical epoch {epoch}.")
        if row.get("observed_label") != str(observed_labels[index]):
            raise ReplacementInputError(f"Stable id {index} maps to a different observed label at historical epoch {epoch}.")
        result[epoch][index] = row
    for epoch, indexed_rows in result.items():
        if set(indexed_rows) != candidate_set:
            missing = sorted(candidate_set - set(indexed_rows))
            unexpected = sorted(set(indexed_rows) - candidate_set)
            raise ReplacementInputError(
                f"Historical sample identities at epoch {epoch} do not exactly cover the training pool; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}."
            )
    return result


def _index_class_rows(
    rows: list[dict[str, str]],
    *,
    training_seed: int,
    expected_epochs: tuple[int, ...],
    expected_classes: tuple[str, ...],
) -> dict[int, dict[str, dict[str, str]]]:
    result: dict[int, dict[str, dict[str, str]]] = {epoch: {} for epoch in expected_epochs}
    expected_class_set = set(expected_classes)
    for row in rows:
        try:
            seed = int(row["training_seed"])
            epoch = int(row["epoch"])
        except (TypeError, ValueError) as exc:
            raise ReplacementInputError("Historical per-class row has invalid training_seed or epoch.") from exc
        if seed != int(training_seed) or epoch not in result:
            raise ReplacementInputError("Historical per-class rows do not match the requested seed/update schedule.")
        label = str(row["observed_label"])
        if label not in expected_class_set or label in result[epoch]:
            raise ReplacementInputError(f"Invalid or duplicate historical per-class label {label!r} at epoch {epoch}.")
        if row.get("selection_strategy") != "loss_and_proto":
            raise ReplacementInputError("Historical per-class source is not PGDF loss_and_proto.")
        if _parse_yes_no(row["fallback_triggered"], "fallback_triggered", Path("selection_per_class.csv")):
            raise ReplacementInputError("Historical PGDF fallback is nonzero; A is not a strict L∩P set.")
        result[epoch][label] = row
    for epoch, by_class in result.items():
        if set(by_class) != expected_class_set:
            raise ReplacementInputError(f"Historical per-class source is incomplete at epoch {epoch}.")
    return result


def _index_update_rows(
    rows: list[dict[str, str]],
    *,
    training_seed: int,
    expected_epochs: tuple[int, ...],
) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            seed = int(row["training_seed"])
            epoch = int(row["update_epoch"])
        except (TypeError, ValueError) as exc:
            raise ReplacementInputError("Historical update row has invalid training_seed or update_epoch.") from exc
        if seed != int(training_seed) or epoch not in expected_epochs or epoch in result:
            raise ReplacementInputError("Historical update rows do not match the requested seed/update schedule.")
        if row.get("selection_strategy") != "loss_and_proto":
            raise ReplacementInputError("Historical update source is not PGDF loss_and_proto.")
        if int(row["fallback_count"]) != 0:
            raise ReplacementInputError("Historical PGDF fallback_count is nonzero; fixed strict replay is not defined.")
        result[epoch] = row
    if set(result) != set(expected_epochs):
        raise ReplacementInputError("Historical update source is incomplete.")
    return result


def _validate_source_update(
    *,
    epoch: int,
    source_mask: np.ndarray,
    pool: np.ndarray,
    update_row: dict[str, str],
    selection_updates_path: Path,
) -> None:
    if int(update_row["full_training_pool_size"]) != int(pool.sum()):
        raise ReplacementInputError(f"Epoch {epoch} historical pool count differs from the fixed validation pool.")
    if int(update_row["selected_total"]) != int(source_mask.sum()):
        raise ReplacementInputError(f"Epoch {epoch} selection update total differs from sample identities: {selection_updates_path}")


def _validate_source_class(
    *,
    epoch: int,
    label: str,
    class_mask: np.ndarray,
    source_mask: np.ndarray,
    loss_mask: np.ndarray,
    proto_mask: np.ndarray,
    source_class: dict[str, str],
    selection_per_class_path: Path,
) -> None:
    checks = {
        "total_count": int(class_mask.sum()),
        "loss_candidate_count": int(np.sum(class_mask & loss_mask)),
        "prototype_candidate_count": int(np.sum(class_mask & proto_mask)),
        "intersection_count": int(np.sum(class_mask & loss_mask & proto_mask)),
        "final_selected_count": int(np.sum(class_mask & source_mask)),
    }
    for field, actual in checks.items():
        if int(source_class[field]) != actual:
            raise ReplacementInputError(
                f"Epoch {epoch}, observed class {label}: {field} disagrees with source identities in {selection_per_class_path}."
            )


def _deterministic_choose(
    candidates: np.ndarray,
    *,
    count: int,
    dataset: str,
    seed: int,
    epoch: int,
    observed_label: str,
    role: str,
) -> np.ndarray:
    if count < 0 or count > len(candidates):
        raise ReplacementInputError("Deterministic replacement requested an invalid count.")
    ranked = sorted(
        (int(index) for index in candidates),
        key=lambda index: sha256(
            f"{REPLACEMENT_RULE_ID}|{dataset}|{seed}|{epoch}|{observed_label}|{role}|train_index:{index}".encode("utf-8")
        ).hexdigest(),
    )
    return np.asarray(ranked[:count], dtype=np.int64)


def _validate_replacement_invariants(
    *,
    epoch: int,
    source_mask: np.ndarray,
    oracle_mask: np.ndarray,
    removed: np.ndarray,
    added: np.ndarray,
    clean_mask: np.ndarray,
    pool: np.ndarray,
) -> None:
    if np.any(removed & added):
        raise ReplacementInputError(f"Epoch {epoch}: one member was both removed and added.")
    if int(removed.sum()) != int(added.sum()):
        raise ReplacementInputError(f"Epoch {epoch}: replacement counts are not balanced.")
    if not np.array_equal(source_mask & ~(removed | added), oracle_mask & ~(removed | added)):
        raise ReplacementInputError(f"Epoch {epoch}: an unmodified member changed state.")
    for mask, name in ((source_mask, "R"), (oracle_mask, "O")):
        if np.any(mask & ~pool):
            raise ReplacementInputError(f"Epoch {epoch}: {name} contains a validation/non-pool sample.")
    if int(source_mask.sum()) != int(oracle_mask.sum()):
        raise ReplacementInputError(f"Epoch {epoch}: R/O total active count differs.")
    if int(np.sum(source_mask & clean_mask)) != int(np.sum(oracle_mask & clean_mask)):
        raise ReplacementInputError(f"Epoch {epoch}: R/O clean active count differs.")
    source_noisy = int(np.sum(source_mask & pool & ~clean_mask))
    oracle_noisy = int(np.sum(oracle_mask & pool & ~clean_mask))
    if source_noisy != oracle_noisy:
        raise ReplacementInputError(f"Epoch {epoch}: R/O noisy active count differs.")


def _source_group(*, is_clean: bool, loss_candidate: bool, prototype_candidate: bool) -> str:
    if is_clean and loss_candidate and prototype_candidate:
        return "A_clean_l_and_p"
    if is_clean and loss_candidate and not prototype_candidate:
        return "B_clean_l_not_p"
    if is_clean:
        return "clean_other"
    return "noisy"
