"""Training-time selection facts, deliberately separated from oracle diagnostics.

This module records only information available to a selector during training.
It has no clean-label or clean/noisy field, and consequently cannot make an
oracle label available to loss ranking, prototype ranking, fusion, or checkpoint
selection. Synthetic ground truth belongs exclusively in
``offline_diagnostics.py``.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ._paths import experiment_directory


SCHEMA_VERSION = "revision_selection_logging_v2"
NA = "N/A"

# Only these explicitly named strategies assert PGDF's strict class-wise
# intersection semantics.  Other strategies (for example ``union`` or a
# single-signal selector) must not have fallback inferred from an empty
# intersection that is merely a diagnostic by-product.
STRICT_INTERSECTION_STRATEGIES = frozenset({"loss_and_proto", "strict_intersection"})
# The submitted PGDF fallback is the minimum-loss member of the prototype
# candidate set.  The logger can verify this named rule from its recorded
# scores; other explicitly named rules are retained as declarations but not
# silently claimed to be semantically checked.
PGDF_FALLBACK_RULE = "prototype_candidate_lowest_loss"

SAMPLE_FIELDS = (
    "schema_version",
    "experiment_id",
    "dataset",
    "noise_protocol",
    "noise_seed",
    "training_seed",
    "epoch",
    "stable_sample_id",
    "sample_path",
    "observed_label",
    "loss_score",
    "prototype_score",
    "loss_candidate",
    "prototype_candidate",
    "final_selected",
    "selection_strategy",
    "model_state_identifier",
)
CLASS_FIELDS = (
    "schema_version",
    "experiment_id",
    "dataset",
    "training_seed",
    "epoch",
    "observed_label",
    "total_count",
    "loss_candidate_count",
    "prototype_candidate_count",
    "intersection_count",
    "final_selected_count",
    "actual_retention_ratio",
    "fallback_triggered",
    "fallback_status",
    "selection_strategy",
    "model_state_identifier",
)
UPDATE_FIELDS = (
    "schema_version",
    "experiment_id",
    "dataset",
    "noise_protocol",
    "noise_seed",
    "training_seed",
    "update_epoch",
    "full_training_pool_size",
    "selected_total",
    "selected_ratio",
    "number_of_observed_classes",
    "fallback_count",
    "fallback_rule",
    "selection_strategy",
    "model_state_identifier",
)


class SelectionLogError(ValueError):
    """Raised when training-time selection facts are incomplete or inconsistent."""


def normalise_path(path: str) -> str:
    return path.replace("\\", "/").strip().rstrip("/")


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise SelectionLogError("experiment_id must be a non-empty path-safe component")
    return value


def _optional_score(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SelectionLogError(f"{field_name} must be a finite numeric value or None")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise SelectionLogError(f"{field_name} must be finite")
    return result


def _optional_bool(value: bool | None, field_name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise SelectionLogError(f"{field_name} must be bool or None, not {value!r}")
    return value


def _csv_value(value: Any) -> str | int | float:
    if value is None:
        return NA
    if isinstance(value, bool):
        return "yes" if value else "no"
    return value


def _parse_optional_float(value: str, field_name: str) -> float | None:
    if value == NA:
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise SelectionLogError(f"Invalid {field_name} value {value!r}") from exc
    return _optional_score(result, field_name)


def _parse_optional_bool(value: str, field_name: str) -> bool | None:
    if value == NA:
        return None
    if value == "yes":
        return True
    if value == "no":
        return False
    raise SelectionLogError(f"Invalid {field_name} flag {value!r}")


def _is_strict_intersection_strategy(strategy: str) -> bool:
    """Whether a strategy promises final membership equals strict intersection."""
    return strategy in STRICT_INTERSECTION_STRATEGIES


def _normalise_fallback_rule(fallback_rule: str | None) -> str | None:
    if fallback_rule is None:
        return None
    if not isinstance(fallback_rule, str) or not fallback_rule.strip() or fallback_rule == NA:
        raise SelectionLogError("fallback_rule must be a non-empty string or None")
    return fallback_rule.strip()


def _ratio_equal(actual: float | None, expected: float | None) -> bool:
    """Strict enough for CSV round-trips while permitting decimal serialization."""
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def _validate_declared_fallback_rule(
    samples: list["SelectionSampleRecord"], fallback_rule: str
) -> str:
    """Validate the one fallback rule representable by the current schema.

    Other rules remain explicitly *declared but not semantically verified*;
    pretending that an empty intersection proves their fallback would be less
    reliable than reporting that boundary.
    """
    if fallback_rule != PGDF_FALLBACK_RULE:
        return "strict_intersection_empty_fallback_declared_rule_not_semantically_verified"
    selected = [record for record in samples if record.final_selected]
    eligible = [
        record
        for record in samples
        if record.prototype_candidate is True and record.loss_score is not None
    ]
    if len(selected) != 1 or not eligible:
        raise SelectionLogError(
            "prototype_candidate_lowest_loss fallback requires exactly one selected eligible prototype sample"
        )
    minimum_loss = min(record.loss_score for record in eligible)
    selected_record = selected[0]
    if (
        selected_record.prototype_candidate is not True
        or selected_record.loss_score is None
        or not math.isclose(selected_record.loss_score, minimum_loss, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise SelectionLogError(
            "prototype_candidate_lowest_loss fallback did not select a minimum-loss prototype candidate"
        )
    return "strict_intersection_empty_fallback_pgdf_rule_verified"


@dataclass(frozen=True)
class SelectionSampleRecord:
    """One training-time sample decision; intentionally contains no oracle state."""

    experiment_id: str
    dataset: str
    noise_protocol: str
    noise_seed: int | None
    training_seed: int
    epoch: int
    stable_sample_id: str
    sample_path: str
    observed_label: str
    loss_score: float | None
    prototype_score: float | None
    loss_candidate: bool | None
    prototype_candidate: bool | None
    final_selected: bool
    selection_strategy: str
    model_state_identifier: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _safe_component(self.experiment_id)
        if not self.dataset or not self.noise_protocol or not self.observed_label or not self.selection_strategy:
            raise SelectionLogError("dataset, noise_protocol, observed_label, and selection_strategy are required")
        if self.training_seed < 0 or self.epoch < 1 or not self.stable_sample_id:
            raise SelectionLogError("training_seed must be non-negative; epoch >= 1; stable_sample_id is required")
        if self.noise_seed is not None and self.noise_seed < 0:
            raise SelectionLogError("noise_seed must be non-negative or None")
        if not isinstance(self.final_selected, bool):
            raise SelectionLogError("final_selected must be bool")
        _optional_score(self.loss_score, "loss_score")
        _optional_score(self.prototype_score, "prototype_score")
        _optional_bool(self.loss_candidate, "loss_candidate")
        _optional_bool(self.prototype_candidate, "prototype_candidate")
        if self.loss_candidate is True and self.loss_score is None:
            raise SelectionLogError("loss_candidate=True requires loss_score")
        if self.prototype_candidate is True and self.prototype_score is None:
            raise SelectionLogError("prototype_candidate=True requires prototype_score")
        if self.schema_version != SCHEMA_VERSION:
            raise SelectionLogError(f"Unsupported schema_version {self.schema_version!r}")

    @property
    def key(self) -> tuple[str, str]:
        return self.stable_sample_id, normalise_path(self.sample_path)

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["sample_path"] = normalise_path(self.sample_path)
        return {field: _csv_value(row[field]) for field in SAMPLE_FIELDS}

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "SelectionSampleRecord":
        if tuple(row.keys()) != SAMPLE_FIELDS:
            raise SelectionLogError("selection_samples.csv has an unexpected schema")
        try:
            final_selected = _parse_optional_bool(row["final_selected"], "final_selected")
            if final_selected is None:
                raise SelectionLogError("final_selected may not be N/A")
            return cls(
                schema_version=row["schema_version"],
                experiment_id=row["experiment_id"],
                dataset=row["dataset"],
                noise_protocol=row["noise_protocol"],
                noise_seed=None if row["noise_seed"] == NA else int(row["noise_seed"]),
                training_seed=int(row["training_seed"]),
                epoch=int(row["epoch"]),
                stable_sample_id=row["stable_sample_id"],
                sample_path=row["sample_path"],
                observed_label=row["observed_label"],
                loss_score=_parse_optional_float(row["loss_score"], "loss_score"),
                prototype_score=_parse_optional_float(row["prototype_score"], "prototype_score"),
                loss_candidate=_parse_optional_bool(row["loss_candidate"], "loss_candidate"),
                prototype_candidate=_parse_optional_bool(row["prototype_candidate"], "prototype_candidate"),
                final_selected=final_selected,
                selection_strategy=row["selection_strategy"],
                model_state_identifier=row["model_state_identifier"],
            )
        except (KeyError, ValueError) as exc:
            raise SelectionLogError("Malformed selection_samples.csv row") from exc


@dataclass(frozen=True)
class SelectionBundle:
    """Validated training-time records returned by :func:`read_selection_bundle`."""

    samples: tuple[SelectionSampleRecord, ...]
    class_rows: tuple[dict[str, Any], ...]
    update_rows: tuple[dict[str, Any], ...]
    run_dir: Path


class SelectionLogger:
    """Accumulates complete update snapshots and writes an isolated CSV bundle."""

    def __init__(self, experiment_id: str) -> None:
        self.experiment_id = _safe_component(experiment_id)
        self._samples: list[SelectionSampleRecord] = []
        self._class_rows: list[dict[str, Any]] = []
        self._update_rows: list[dict[str, Any]] = []
        self._recorded_updates: set[tuple[int, int]] = set()

    @staticmethod
    def _class_row(
        samples: list[SelectionSampleRecord],
        fallback_labels: set[str] | None,
        fallback_rule: str | None,
    ) -> dict[str, Any]:
        first = samples[0]
        label = first.observed_label
        for record in samples:
            if (
                record.epoch != first.epoch
                or record.training_seed != first.training_seed
                or record.observed_label != label
                or record.selection_strategy != first.selection_strategy
                or record.model_state_identifier != first.model_state_identifier
            ):
                raise SelectionLogError("class rows require homogeneous update metadata")
        loss_values = [record.loss_candidate for record in samples]
        proto_values = [record.prototype_candidate for record in samples]
        loss_count = None if any(value is None for value in loss_values) else sum(value is True for value in loss_values)
        proto_count = None if any(value is None for value in proto_values) else sum(value is True for value in proto_values)
        intersection_count = (
            None
            if loss_count is None or proto_count is None
            else sum(record.loss_candidate is True and record.prototype_candidate is True for record in samples)
        )
        final_count = sum(record.final_selected for record in samples)
        strict = _is_strict_intersection_strategy(first.selection_strategy)
        fallback_triggered: bool | None
        fallback_status: str
        if not strict:
            if fallback_labels:
                raise SelectionLogError(
                    "fallback_observed_labels applies only to explicitly strict-intersection strategies"
                )
            if fallback_rule is not None:
                raise SelectionLogError("fallback_rule applies only to explicitly strict-intersection strategies")
            fallback_triggered = None
            fallback_status = "not_applicable_non_strict_strategy"
        elif intersection_count is None:
            if fallback_labels:
                raise SelectionLogError(
                    "Cannot validate declared fallback without loss and prototype candidate membership"
                )
            fallback_triggered = None
            fallback_status = "not_applicable_candidate_membership_missing"
        elif intersection_count > 0:
            if fallback_labels is not None and label in fallback_labels:
                raise SelectionLogError(
                    f"Fallback was declared for non-empty strict intersection in observed label {label!r}"
                )
            expected_final = [
                record.loss_candidate is True and record.prototype_candidate is True for record in samples
            ]
            actual_final = [record.final_selected for record in samples]
            if actual_final != expected_final:
                raise SelectionLogError(
                    f"Strict-intersection final membership differs from intersection for observed label {label!r}"
                )
            fallback_triggered = False
            fallback_status = "strict_intersection_nonempty_no_fallback"
        elif fallback_labels is None:
            # An empty intersection with selected samples is compatible with a
            # fallback, but this API was not told whether one was declared.
            # Keep that fact explicitly unknown rather than fabricating zero.
            fallback_triggered = None if final_count > 0 else False
            fallback_status = (
                "strict_intersection_empty_fallback_unknown"
                if final_count > 0
                else "strict_intersection_empty_no_selection"
            )
        elif label in fallback_labels:
            if fallback_rule is None:
                raise SelectionLogError("A declared strict-intersection fallback requires fallback_rule")
            if final_count == 0:
                raise SelectionLogError(
                    f"Declared fallback selected no sample for empty strict intersection in observed label {label!r}"
                )
            fallback_triggered = True
            fallback_status = _validate_declared_fallback_rule(samples, fallback_rule)
        else:
            if final_count != 0:
                raise SelectionLogError(
                    f"Empty strict intersection selected samples without a declared fallback for observed label {label!r}"
                )
            fallback_triggered = False
            fallback_status = "strict_intersection_empty_no_fallback_declared"
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": first.experiment_id,
            "dataset": first.dataset,
            "training_seed": first.training_seed,
            "epoch": first.epoch,
            "observed_label": label,
            "total_count": len(samples),
            "loss_candidate_count": loss_count,
            "prototype_candidate_count": proto_count,
            "intersection_count": intersection_count,
            "final_selected_count": final_count,
            "actual_retention_ratio": final_count / len(samples),
            "fallback_triggered": fallback_triggered,
            "fallback_status": fallback_status,
            "selection_strategy": first.selection_strategy,
            "model_state_identifier": first.model_state_identifier,
        }

    def record_update(
        self,
        records: Iterable[SelectionSampleRecord],
        *,
        full_training_pool_size: int,
        fallback_observed_labels: Iterable[str] | None = None,
        fallback_rule: str | None = None,
    ) -> None:
        """Validate and add one whole-pool update snapshot.

        The method derives class and update counts from sample records, so a
        caller cannot silently write inconsistent aggregate counts.  Strict
        intersection fallback is valid only for named strict strategies.  When
        it occurs, callers must declare both the affected labels and the rule
        that selected their replacement samples.  For non-strict strategies it
        is explicitly ``N/A`` rather than inferred from empty intersections.
        For methods without a loss/prototype signal, use ``None`` for that
        field; aggregates are written as ``N/A`` rather than forged zeroes.
        """
        samples = list(records)
        if not samples:
            raise SelectionLogError("A selection update must contain at least one sample")
        first = samples[0]
        if first.experiment_id != self.experiment_id:
            raise SelectionLogError("record experiment_id does not match logger experiment_id")
        if full_training_pool_size != len(samples):
            raise SelectionLogError("full_training_pool_size must equal number of update sample records")
        update_key = first.training_seed, first.epoch
        if update_key in self._recorded_updates:
            raise SelectionLogError(f"Duplicate selection update for seed={first.training_seed}, epoch={first.epoch}")
        keys: set[tuple[str, str]] = set()
        by_label: dict[str, list[SelectionSampleRecord]] = defaultdict(list)
        for record in samples:
            if record.experiment_id != self.experiment_id:
                raise SelectionLogError("All records must have logger experiment_id")
            if (
                record.dataset != first.dataset
                or record.noise_protocol != first.noise_protocol
                or record.noise_seed != first.noise_seed
                or record.training_seed != first.training_seed
                or record.epoch != first.epoch
                or record.selection_strategy != first.selection_strategy
                or record.model_state_identifier != first.model_state_identifier
            ):
                raise SelectionLogError("All records for an update must share run/update metadata")
            if record.key in keys:
                raise SelectionLogError(f"Duplicate stable sample key in update: {record.key!r}")
            keys.add(record.key)
            by_label[record.observed_label].append(record)
        fallback_labels = (
            None if fallback_observed_labels is None else {str(label) for label in fallback_observed_labels}
        )
        fallback_rule = _normalise_fallback_rule(fallback_rule)
        if fallback_labels is not None:
            unknown_labels = fallback_labels - set(by_label)
            if unknown_labels:
                raise SelectionLogError(f"Fallback labels absent from update: {sorted(unknown_labels)}")
        class_rows = [
            self._class_row(group, fallback_labels, fallback_rule) for _, group in sorted(by_label.items())
        ]
        fallback_count = (
            None
            if any(row["fallback_triggered"] is None for row in class_rows)
            else sum(row["fallback_triggered"] is True for row in class_rows)
        )
        update_row = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "dataset": first.dataset,
            "noise_protocol": first.noise_protocol,
            "noise_seed": first.noise_seed,
            "training_seed": first.training_seed,
            "update_epoch": first.epoch,
            "full_training_pool_size": full_training_pool_size,
            "selected_total": sum(record.final_selected for record in samples),
            "selected_ratio": sum(record.final_selected for record in samples) / full_training_pool_size,
            "number_of_observed_classes": len(class_rows),
            "fallback_count": fallback_count,
            "fallback_rule": fallback_rule,
            "selection_strategy": first.selection_strategy,
            "model_state_identifier": first.model_state_identifier,
        }
        self._recorded_updates.add(update_key)
        self._samples.extend(samples)
        self._class_rows.extend(class_rows)
        self._update_rows.append(update_row)

    @staticmethod
    def _write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row.get(field)) for field in fields})

    def finalize(self, output_root: Path) -> Path:
        """Write a complete, non-overwriting bundle beneath ``output_root``.

        ``run_status.json`` is written last. A missing or non-complete status is
        treated as a partial/corrupt bundle by :func:`read_selection_bundle`.
        """
        self.assert_ready_to_finalize()
        try:
            run_dir = experiment_directory(output_root, self.experiment_id) / "selection_records"
        except ValueError as exc:
            raise SelectionLogError(str(exc)) from exc
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing selection records: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        sample_rows = [record.to_row() for record in self._samples]
        self._write_csv(run_dir / "selection_samples.csv", SAMPLE_FIELDS, sample_rows)
        self._write_csv(run_dir / "selection_per_class.csv", CLASS_FIELDS, self._class_rows)
        self._write_csv(run_dir / "selection_updates.csv", UPDATE_FIELDS, self._update_rows)
        temporary = run_dir / ".run_status.json.tmp"
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": self.experiment_id,
                    "status": "complete",
                    "sample_record_count": len(self._samples),
                    "class_record_count": len(self._class_rows),
                    "update_record_count": len(self._update_rows),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, run_dir / "run_status.json")
        return run_dir

    def assert_ready_to_finalize(self) -> None:
        """Reject an empty logger before any caller writes a complete result."""
        if not self._update_rows:
            raise SelectionLogError("Cannot finalize an empty selection logger")


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.is_file():
        raise SelectionLogError(f"Missing required selection-log file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise SelectionLogError(f"Unexpected schema in {path.name}")
        return list(reader)


def _parse_count(value: str, field: str) -> int | None:
    if value == NA:
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise SelectionLogError(f"Invalid integer {field}={value!r}") from exc
    if result < 0:
        raise SelectionLogError(f"Negative count {field}={value!r}")
    return result


def _parse_ratio(value: str, field: str) -> float | None:
    if value == NA:
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise SelectionLogError(f"Invalid ratio {field}={value!r}") from exc
    if not 0.0 <= result <= 1.0:
        raise SelectionLogError(f"Ratio outside [0, 1] for {field}={value!r}")
    return result


def _decode_class_or_update(row: dict[str, str], fields: tuple[str, ...]) -> dict[str, Any]:
    decoded: dict[str, Any] = dict(row)
    for name in ("training_seed", "epoch", "update_epoch", "full_training_pool_size", "selected_total", "number_of_observed_classes", "fallback_count", "total_count", "loss_candidate_count", "prototype_candidate_count", "intersection_count", "final_selected_count"):
        if name in decoded:
            decoded[name] = _parse_count(decoded[name], name)
    for name in ("selected_ratio", "actual_retention_ratio"):
        if name in decoded:
            decoded[name] = _parse_ratio(decoded[name], name)
    if "noise_seed" in decoded:
        decoded["noise_seed"] = _parse_count(decoded["noise_seed"], "noise_seed")
    if "fallback_triggered" in decoded:
        decoded["fallback_triggered"] = _parse_optional_bool(decoded["fallback_triggered"], "fallback_triggered")
    if "fallback_rule" in decoded:
        decoded["fallback_rule"] = None if decoded["fallback_rule"] == NA else _normalise_fallback_rule(decoded["fallback_rule"])
    if "fallback_status" in decoded and not decoded["fallback_status"]:
        raise SelectionLogError("fallback_status may not be empty")
    return decoded


def read_selection_bundle(run_dir: Path) -> SelectionBundle:
    """Read and cross-check a completed bundle written by :class:`SelectionLogger`."""
    path = Path(run_dir)
    status_path = path / "run_status.json"
    if not status_path.is_file():
        raise SelectionLogError(f"Selection bundle lacks completion marker: {status_path}")
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SelectionLogError(f"Malformed completion marker: {status_path}") from exc
    if status.get("schema_version") != SCHEMA_VERSION or status.get("status") != "complete":
        raise SelectionLogError("Selection bundle is incomplete or has an unsupported schema")
    sample_rows = _read_csv(path / "selection_samples.csv", SAMPLE_FIELDS)
    samples = tuple(SelectionSampleRecord.from_row(row) for row in sample_rows)
    class_rows = tuple(_decode_class_or_update(row, CLASS_FIELDS) for row in _read_csv(path / "selection_per_class.csv", CLASS_FIELDS))
    update_rows = tuple(_decode_class_or_update(row, UPDATE_FIELDS) for row in _read_csv(path / "selection_updates.csv", UPDATE_FIELDS))
    if status.get("sample_record_count") != len(samples) or status.get("class_record_count") != len(class_rows) or status.get("update_record_count") != len(update_rows):
        raise SelectionLogError("Selection bundle completion counts disagree with data files")
    if len({(record.training_seed, record.epoch, record.key) for record in samples}) != len(samples):
        raise SelectionLogError("Duplicate sample key found while reading selection bundle")
    by_update: dict[tuple[int, int], list[SelectionSampleRecord]] = defaultdict(list)
    for record in samples:
        by_update[(record.training_seed, record.epoch)].append(record)
    update_index = {(row["training_seed"], row["update_epoch"]): row for row in update_rows}
    if len(update_index) != len(update_rows) or set(update_index) != set(by_update):
        raise SelectionLogError("Sample/update record coverage differs")
    class_index = {(row["training_seed"], row["epoch"], row["observed_label"]): row for row in class_rows}
    if len(class_index) != len(class_rows):
        raise SelectionLogError("Duplicate class update row")
    class_update_keys = {(row["training_seed"], row["epoch"]) for row in class_rows}
    if class_update_keys != set(by_update):
        raise SelectionLogError("Class/update record coverage differs")
    for update_key, records in by_update.items():
        update = update_index[update_key]
        first = records[0]
        if (
            update["schema_version"] != SCHEMA_VERSION
            or update["experiment_id"] != first.experiment_id
            or update["dataset"] != first.dataset
            or update["noise_protocol"] != first.noise_protocol
            or update["noise_seed"] != first.noise_seed
            or update["selection_strategy"] != first.selection_strategy
            or update["model_state_identifier"] != first.model_state_identifier
        ):
            raise SelectionLogError("Update metadata disagrees with sample records")
        if update["full_training_pool_size"] != len(records):
            raise SelectionLogError("full_training_pool_size disagrees with sample records")
        selected_total = sum(record.final_selected for record in records)
        if update["selected_total"] != selected_total:
            raise SelectionLogError("selected_total disagrees with sample records")
        if not _ratio_equal(update["selected_ratio"], selected_total / len(records)):
            raise SelectionLogError("selected_ratio disagrees with sample records")
        labels = {record.observed_label for record in records}
        if update["number_of_observed_classes"] != len(labels):
            raise SelectionLogError("number_of_observed_classes disagrees with sample records")
        actual_class_keys = {(update_key[0], update_key[1], label) for label in labels}
        logged_class_keys = {
            key for key in class_index if key[0] == update_key[0] and key[1] == update_key[1]
        }
        if actual_class_keys != logged_class_keys:
            raise SelectionLogError("Class rows do not exactly cover observed-label classes")
        recorded_flags = [
            class_index[(update_key[0], update_key[1], label)]["fallback_triggered"] for label in labels
        ]
        fallback_labels = None if any(flag is None for flag in recorded_flags) else {
            label
            for label in labels
            if class_index[(update_key[0], update_key[1], label)]["fallback_triggered"] is True
        }
        expected_class_rows: list[dict[str, Any]] = []
        for label in labels:
            row = class_index.get((update_key[0], update_key[1], label))
            if row is None:
                raise SelectionLogError("Missing class row for recorded sample class")
            members = [record for record in records if record.observed_label == label]
            expected = SelectionLogger._class_row(members, fallback_labels, update["fallback_rule"])
            expected_class_rows.append(expected)
            for field in (
                "schema_version",
                "experiment_id",
                "dataset",
                "training_seed",
                "epoch",
                "observed_label",
                "total_count",
                "loss_candidate_count",
                "prototype_candidate_count",
                "intersection_count",
                "final_selected_count",
                "fallback_triggered",
                "fallback_status",
                "selection_strategy",
                "model_state_identifier",
            ):
                if row[field] != expected[field]:
                    raise SelectionLogError(f"Class field {field} disagrees with sample records")
            if not _ratio_equal(row["actual_retention_ratio"], expected["actual_retention_ratio"]):
                raise SelectionLogError("actual_retention_ratio disagrees with sample records")
        class_fallbacks = (
            None
            if any(row["fallback_triggered"] is None for row in expected_class_rows)
            else sum(row["fallback_triggered"] is True for row in expected_class_rows)
        )
        if update["fallback_count"] != class_fallbacks:
            raise SelectionLogError("fallback_count disagrees with class rows")
    return SelectionBundle(samples, class_rows, update_rows, path)
