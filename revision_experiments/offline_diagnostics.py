"""Oracle-separated, offline selection diagnostics for future revision runs.

Training-time records come from :mod:`selection_logging` and contain no oracle
state.  This module accepts a separately supplied oracle only after a run has
finished.  When no trustworthy oracle is supplied (for example, real noisy
data), truth-dependent quantities are represented as ``None``/``N/A`` rather
than guessed.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ._paths import protect_output_root
from .selection_logging import (
    SelectionBundle,
    SelectionLogError,
    SelectionSampleRecord,
    _is_strict_intersection_strategy,
    _ratio_equal,
    normalise_path,
)


class DiagnosticError(ValueError):
    """Raised when offline inputs cannot support a trustworthy diagnostic."""


@dataclass(frozen=True)
class OracleLabelRecord:
    """A separately provided synthetic oracle record for one stable training key."""

    stable_sample_id: str
    sample_path: str
    observed_label: str
    clean_label: str
    partition: str = "training_pool"

    def __post_init__(self) -> None:
        if not self.stable_sample_id or not self.observed_label or not self.clean_label:
            raise DiagnosticError("oracle stable_sample_id, observed_label, and clean_label are required")
        if self.partition not in {"training_pool", "validation", "test"}:
            raise DiagnosticError(f"Unsupported oracle partition {self.partition!r}")

    @property
    def key(self) -> tuple[str, str]:
        return self.stable_sample_id, normalise_path(self.sample_path)


@dataclass(frozen=True)
class RunDiagnostics:
    """Traceable diagnostics for exactly one dataset/seed/selection epoch."""

    experiment_id: str
    dataset: str
    training_seed: int
    epoch: int
    oracle_status: str
    micro_metrics: dict[str, dict[str, Any]]
    class_rows: tuple[dict[str, Any], ...]
    macro_metrics: dict[str, dict[str, Any]]
    fallback_count: int | None
    fallback_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def _sample_std(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return statistics.stdev(usable) if len(usable) >= 2 else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[float | None]) -> dict[str, Any]:
    """Mean/median/sample-SD/Q25/Q75, excluding undefined class metrics."""
    usable = [value for value in values if value is not None]
    return {
        "defined_count": len(usable),
        "mean": _mean(usable),
        "median": statistics.median(usable) if usable else None,
        "sample_std": _sample_std(usable),
        "q25": _quantile(usable, 0.25),
        "q75": _quantile(usable, 0.75),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def tie_aware_auc(scores: Sequence[float], clean_positive: Sequence[bool]) -> float | None:
    """ROC-AUC with clean positive and exact 0.5 credit for tied pairs."""
    if len(scores) != len(clean_positive):
        raise DiagnosticError("AUC scores and labels have different lengths")
    positive_count = sum(clean_positive)
    negative_count = len(clean_positive) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum = sum(rank for rank, positive in zip(ranks, clean_positive) if positive)
    mann_whitney_u = rank_sum - positive_count * (positive_count + 1) / 2.0
    return mann_whitney_u / (positive_count * negative_count)


def _oracle_map(records: Iterable[OracleLabelRecord]) -> dict[tuple[str, str], OracleLabelRecord]:
    mapping: dict[tuple[str, str], OracleLabelRecord] = {}
    for record in records:
        if record.key in mapping:
            raise DiagnosticError(f"Duplicate oracle key {record.key!r}")
        mapping[record.key] = record
    return mapping


def _selection_metrics(
    samples: Sequence[SelectionSampleRecord],
    membership: Sequence[bool] | None,
    clean_flags: Sequence[bool] | None,
    status: str,
) -> dict[str, Any]:
    if membership is None:
        return {
            "status": status,
            "selected_count": None,
            "selected_ratio": None,
            "purity": None,
            "clean_recall": None,
            "noisy_retention": None,
        }
    selected = [sample for sample, flag in zip(samples, membership) if flag]
    basic = {
        "status": "recorded_no_oracle" if clean_flags is None else "recorded",
        "selected_count": len(selected),
        "selected_ratio": safe_ratio(len(selected), len(samples)),
        "purity": None,
        "clean_recall": None,
        "noisy_retention": None,
    }
    if clean_flags is None:
        return basic
    selected_clean = sum(flag and clean for flag, clean in zip(membership, clean_flags))
    selected_noisy = sum(flag and not clean for flag, clean in zip(membership, clean_flags))
    total_clean = sum(clean_flags)
    total_noisy = len(clean_flags) - total_clean
    basic.update(
        {
            "purity": safe_ratio(selected_clean, len(selected)),
            "clean_recall": safe_ratio(selected_clean, total_clean),
            "noisy_retention": safe_ratio(selected_noisy, total_noisy),
        }
    )
    return basic


def _validate_external_aggregates(
    samples: Sequence[SelectionSampleRecord],
    class_rows: Sequence[dict[str, Any]] | None,
    update_row: dict[str, Any] | None,
) -> None:
    """Cross-check caller-provided logger aggregates without trusting them blindly."""
    by_class: dict[str, list[SelectionSampleRecord]] = {}
    for sample in samples:
        by_class.setdefault(sample.observed_label, []).append(sample)
    derived_fallback_count: int | None = None
    if class_rows is not None:
        index = {str(row["observed_label"]): row for row in class_rows}
        if len(index) != len(class_rows) or set(index) != set(by_class):
            raise DiagnosticError("Provided class rows do not bijectively cover observed-label classes")
        fallback_flags: list[bool | None] = []
        for label, members in by_class.items():
            row = index[label]
            loss_count = (
                None
                if any(item.loss_candidate is None for item in members)
                else sum(item.loss_candidate is True for item in members)
            )
            prototype_count = (
                None
                if any(item.prototype_candidate is None for item in members)
                else sum(item.prototype_candidate is True for item in members)
            )
            intersection_count = (
                None
                if loss_count is None or prototype_count is None
                else sum(item.loss_candidate is True and item.prototype_candidate is True for item in members)
            )
            final_count = sum(item.final_selected for item in members)
            if (
                row.get("total_count") != len(members)
                or row.get("loss_candidate_count") != loss_count
                or row.get("prototype_candidate_count") != prototype_count
                or row.get("intersection_count") != intersection_count
                or row.get("final_selected_count") != final_count
                or not _ratio_equal(row.get("actual_retention_ratio"), safe_ratio(final_count, len(members)))
            ):
                raise DiagnosticError(f"Provided class row counts disagree for {label!r}")
            fallback_flags.append(row.get("fallback_triggered"))
        derived_fallback_count = (
            None if any(flag is None for flag in fallback_flags) else sum(flag is True for flag in fallback_flags)
        )
    if update_row is not None:
        if update_row.get("full_training_pool_size") != len(samples):
            raise DiagnosticError("Provided update pool size disagrees with sample records")
        if update_row.get("selected_total") != sum(item.final_selected for item in samples):
            raise DiagnosticError("Provided update selected count disagrees with sample records")
        selected_total = sum(item.final_selected for item in samples)
        if not _ratio_equal(update_row.get("selected_ratio"), safe_ratio(selected_total, len(samples))):
            raise DiagnosticError("Provided update selected ratio disagrees with sample records")
        if update_row.get("number_of_observed_classes") != len(by_class):
            raise DiagnosticError("Provided update class count disagrees with sample records")
        if class_rows is not None and update_row.get("fallback_count") != derived_fallback_count:
            raise DiagnosticError("Provided update fallback count disagrees with sample flags")


def diagnose_selection_update(
    samples: Sequence[SelectionSampleRecord],
    *,
    oracle_records: Iterable[OracleLabelRecord] | None = None,
    logged_class_rows: Sequence[dict[str, Any]] | None = None,
    logged_update_row: dict[str, Any] | None = None,
) -> RunDiagnostics:
    """Diagnose one selection update, optionally using a separately supplied oracle.

    A missing oracle is valid: selected ratios and non-oracle counts remain
    available, while Purity/Recall/Retention/Lift/AUC are ``None``. An oracle
    must match all sample keys, labels, and the ``training_pool`` partition.
    """
    if not samples:
        raise DiagnosticError("Cannot diagnose an empty selection update")
    first = samples[0]
    seen_keys: set[tuple[str, str]] = set()
    for sample in samples:
        if sample.key in seen_keys:
            raise DiagnosticError(f"Duplicate sample key in diagnostic input: {sample.key!r}")
        seen_keys.add(sample.key)
        if (
            sample.experiment_id != first.experiment_id
            or sample.dataset != first.dataset
            or sample.training_seed != first.training_seed
            or sample.epoch != first.epoch
        ):
            raise DiagnosticError("Diagnostic input must represent exactly one dataset/seed/epoch")
    _validate_external_aggregates(samples, logged_class_rows, logged_update_row)
    clean_flags: list[bool] | None = None
    if oracle_records is not None:
        mapping = _oracle_map(oracle_records)
        if set(mapping) != seen_keys:
            missing = seen_keys - set(mapping)
            extra = set(mapping) - seen_keys
            details: list[str] = []
            if missing:
                details.append(f"missing={len(missing)}")
            if extra:
                details.append(f"extra={len(extra)}")
            raise DiagnosticError("Oracle keys must exactly equal training selection keys (" + ", ".join(details) + ")")
        clean_flags = []
        for sample in samples:
            oracle = mapping.get(sample.key)
            if oracle is None:
                raise DiagnosticError(f"Missing oracle key for sample {sample.key!r}")
            if oracle.partition != "training_pool":
                raise DiagnosticError(f"Non-training oracle sample was supplied for selection key {sample.key!r}")
            if oracle.observed_label != sample.observed_label:
                raise DiagnosticError(f"Observed label mismatch for oracle key {sample.key!r}")
            clean_flags.append(oracle.clean_label == oracle.observed_label)
        oracle_status = "complete_synthetic_oracle"
    else:
        oracle_status = "not_provided_truth_dependent_metrics_not_computable"
    loss_membership = None if any(sample.loss_candidate is None for sample in samples) else [bool(sample.loss_candidate) for sample in samples]
    proto_membership = None if any(sample.prototype_candidate is None for sample in samples) else [bool(sample.prototype_candidate) for sample in samples]
    final_membership = [sample.final_selected for sample in samples]
    micro_metrics = {
        "loss_candidate": _selection_metrics(samples, loss_membership, clean_flags, "not_applicable_loss_membership_missing"),
        "prototype_candidate": _selection_metrics(samples, proto_membership, clean_flags, "not_applicable_prototype_membership_missing"),
        "final_selected": _selection_metrics(samples, final_membership, clean_flags, "recorded"),
    }
    rows: list[dict[str, Any]] = []
    labels = sorted({sample.observed_label for sample in samples})
    logged_fallbacks = (
        {str(row["observed_label"]): row.get("fallback_triggered") for row in logged_class_rows}
        if logged_class_rows is not None
        else {}
    )
    logged_fallback_statuses = (
        {str(row["observed_label"]): row.get("fallback_status") for row in logged_class_rows}
        if logged_class_rows is not None
        else {}
    )
    fallback_count: int | None = 0
    for label in labels:
        indices = [index for index, sample in enumerate(samples) if sample.observed_label == label]
        members = [samples[index] for index in indices]
        class_clean = [clean_flags[index] for index in indices] if clean_flags is not None else None
        loss_count = None if loss_membership is None else sum(loss_membership[index] for index in indices)
        proto_count = None if proto_membership is None else sum(proto_membership[index] for index in indices)
        final_count = sum(final_membership[index] for index in indices)
        intersection_count = (
            None
            if loss_membership is None or proto_membership is None
            else sum(loss_membership[index] and proto_membership[index] for index in indices)
        )
        strategy_is_strict = _is_strict_intersection_strategy(first.selection_strategy)
        if label in logged_fallbacks:
            fallback = logged_fallbacks[label]
            fallback_status = logged_fallback_statuses.get(label, "recorded_by_selection_log")
        elif not strategy_is_strict:
            fallback = None
            fallback_status = "not_applicable_non_strict_strategy"
        elif intersection_count is None:
            fallback = None
            fallback_status = "not_applicable_candidate_membership_missing"
        elif intersection_count > 0:
            expected_final = [member.loss_candidate is True and member.prototype_candidate is True for member in members]
            if [member.final_selected for member in members] != expected_final:
                raise DiagnosticError("Strict-intersection final membership differs from intersection")
            fallback = False
            fallback_status = "strict_intersection_nonempty_no_fallback"
        elif final_count == 0:
            fallback = False
            fallback_status = "strict_intersection_empty_no_selection"
        else:
            fallback = None
            fallback_status = "strict_intersection_empty_fallback_unknown"
        if fallback is None:
            fallback_count = None
        elif fallback_count is not None:
            fallback_count += int(fallback)
        row: dict[str, Any] = {
            "observed_label": label,
            "total_count": len(members),
            "loss_candidate_count": loss_count,
            "prototype_candidate_count": proto_count,
            "intersection_count": intersection_count,
            "final_selected_count": final_count,
            "actual_retention_ratio": safe_ratio(final_count, len(members)),
            "fallback_triggered": fallback,
            "fallback_status": fallback_status,
            "clean_count": None,
            "noisy_count": None,
            "class_clean_ratio": None,
            "prototype_purity": None,
            "final_purity": None,
            "prototype_purity_lift": None,
            "final_purity_lift": None,
            "prototype_auc": None,
            "auc_status": "not_computable_without_oracle",
        }
        if class_clean is not None:
            clean_count = sum(class_clean)
            noisy_count = len(class_clean) - clean_count
            class_ratio = safe_ratio(clean_count, len(class_clean))
            proto_clean = (
                None if proto_membership is None else sum(proto_membership[index] and clean_flags[index] for index in indices)
            )
            final_clean = sum(final_membership[index] and clean_flags[index] for index in indices)
            prototype_purity = safe_ratio(proto_clean, proto_count) if proto_clean is not None and proto_count is not None else None
            final_purity = safe_ratio(final_clean, final_count)
            score_values = [member.prototype_score for member in members]
            if any(score is None for score in score_values):
                auc = None
                auc_status = "not_computable_missing_prototype_score"
            else:
                auc = tie_aware_auc([float(score) for score in score_values], class_clean)
                auc_status = "defined" if auc is not None else "not_defined_single_clean_or_noisy_outcome"
            row.update(
                {
                    "clean_count": clean_count,
                    "noisy_count": noisy_count,
                    "class_clean_ratio": class_ratio,
                    "prototype_purity": prototype_purity,
                    "final_purity": final_purity,
                    "prototype_purity_lift": (
                        prototype_purity - class_ratio if prototype_purity is not None and class_ratio is not None else None
                    ),
                    "final_purity_lift": final_purity - class_ratio if final_purity is not None and class_ratio is not None else None,
                    "prototype_auc": auc,
                    "auc_status": auc_status,
                }
            )
        rows.append(row)
    macro_fields = (
        "actual_retention_ratio",
        "prototype_purity",
        "final_purity",
        "prototype_purity_lift",
        "final_purity_lift",
        "prototype_auc",
    )
    macro_metrics = {field: distribution([row[field] for row in rows]) for field in macro_fields}
    return RunDiagnostics(
        experiment_id=first.experiment_id,
        dataset=first.dataset,
        training_seed=first.training_seed,
        epoch=first.epoch,
        oracle_status=oracle_status,
        micro_metrics=micro_metrics,
        class_rows=tuple(rows),
        macro_metrics=macro_metrics,
        fallback_count=fallback_count,
        fallback_status="recorded" if fallback_count is not None else "not_recorded_or_not_applicable",
    )


def diagnose_bundle_update(
    bundle: SelectionBundle,
    *,
    training_seed: int,
    epoch: int,
    oracle_records: Iterable[OracleLabelRecord] | None = None,
) -> RunDiagnostics:
    """Convenience adapter that diagnoses one validated logger bundle update."""
    samples = [record for record in bundle.samples if record.training_seed == training_seed and record.epoch == epoch]
    class_rows = [
        row for row in bundle.class_rows if row["training_seed"] == training_seed and row["epoch"] == epoch
    ]
    update_rows = [
        row for row in bundle.update_rows if row["training_seed"] == training_seed and row["update_epoch"] == epoch
    ]
    if len(update_rows) != 1:
        raise DiagnosticError("Expected exactly one matching update row in bundle")
    return diagnose_selection_update(
        samples,
        oracle_records=oracle_records,
        logged_class_rows=class_rows,
        logged_update_row=update_rows[0],
    )


def aggregate_across_seeds(results: Sequence[RunDiagnostics]) -> dict[str, Any]:
    """Aggregate per-seed results without pooling samples across seeds.

    The input must represent one experiment/dataset/epoch and unique training
    seeds. For macro metrics, this computes each seed's class-equal metric first
    (already stored in ``RunDiagnostics``), then mean and sample SD across seeds.
    """
    if not results:
        raise DiagnosticError("No diagnostics supplied for cross-seed aggregation")
    first = results[0]
    seeds = {result.training_seed for result in results}
    if len(seeds) != len(results):
        raise DiagnosticError("Cross-seed aggregation requires unique training_seed values")
    for result in results:
        if (result.experiment_id, result.dataset, result.epoch) != (
            first.experiment_id,
            first.dataset,
            first.epoch,
        ):
            raise DiagnosticError("Cross-seed aggregation requires same experiment/dataset/epoch")
    macro = {
        name: {
            "seed_count": len(results),
            "mean": _mean([result.macro_metrics[name]["mean"] for result in results]),
            "sample_std": _sample_std([result.macro_metrics[name]["mean"] for result in results]),
            "defined_seed_count": sum(result.macro_metrics[name]["mean"] is not None for result in results),
        }
        for name in results[0].macro_metrics
    }
    micro: dict[str, dict[str, Any]] = {}
    for set_name in results[0].micro_metrics:
        metric_names = ("selected_ratio", "purity", "clean_recall", "noisy_retention")
        micro[set_name] = {
            metric: {
                "seed_count": len(results),
                "mean": _mean([result.micro_metrics[set_name][metric] for result in results]),
                "sample_std": _sample_std([result.micro_metrics[set_name][metric] for result in results]),
            }
            for metric in metric_names
        }
    return {
        "experiment_id": first.experiment_id,
        "dataset": first.dataset,
        "epoch": first.epoch,
        "training_seeds": sorted(seeds),
        "aggregation": "per-seed metrics first; sample SD (ddof=1) across seeds; no pooled cross-seed AUC",
        "macro": macro,
        "micro": micro,
    }


def write_diagnostics_json(result: RunDiagnostics | dict[str, Any], path: Path) -> Path:
    """Atomically write a derived diagnostic file; callers choose an isolated output path."""
    requested = Path(path)
    try:
        # Resolve the parent before writing, so direct historical paths and
        # existing symlink aliases into that tree receive the same protection as
        # profiling and selection logging.
        output = protect_output_root(requested.parent) / requested.name
    except ValueError as exc:
        raise DiagnosticError(str(exc)) from exc
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict() if isinstance(result, RunDiagnostics) else result
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    return output
