"""Small adapters that attach approved D1/D2 observers to PGDF training.

The adapters only consume arrays already produced by the training loop.  They
never run a model, rank candidates, or access oracle clean/noisy labels.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .profiling import RunProfiler
from .selection_logging import SelectionLogger, SelectionSampleRecord
from ._paths import experiment_directory, protect_output_root


PGDF_ACTUAL_FALLBACK_RULE = "prototype_candidate_or_class_lowest_loss"


def build_experiment_id(
    *, dataset: str, noise_protocol: str, noise_seed: int | None, method: str, training_seed: int
) -> str:
    """Build a path-safe, metadata-bearing identifier without consulting metrics."""
    def component(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
        if not normalized:
            raise ValueError("experiment metadata contains an empty path component")
        return normalized

    return (
        f"{component(dataset)}_{component(noise_protocol)}_noise{noise_seed if noise_seed is not None else 'na'}_"
        f"{component(method)}_seed{int(training_seed)}"
    )


def protect_monitored_training_output_dir(output_dir: Path, *, instrumentation_requested: bool) -> Path:
    """Protect the primary training result root only for opt-in revision runs.

    Historical invocations without D1/D2 monitoring retain their existing
    output-dir behavior.  Monitored runs must keep both their observer output
    and their main training result outside the protected historical tree.
    """
    if not instrumentation_requested:
        return Path(output_dir)
    return protect_output_root(Path(output_dir))


def _git_output(repository_root: Path, *arguments: str) -> str | None:
    """Read a Git value for provenance without changing repository state."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_code_provenance(repository_root: Path) -> dict[str, Any]:
    """Collect read-only code identity for a future monitored result.

    The values are written only after a successful training return and never
    affect model construction, data order, random state, or optimization.
    """
    root = Path(repository_root).resolve()
    training_function = root / "gcdd" / "lora_dynamic.py"
    training_entry = root / "scripts" / "run_lora_checkpoint_validation.py"
    d1_d2_modules = {
        "revision_experiments/_paths.py": root / "revision_experiments" / "_paths.py",
        "revision_experiments/profiling.py": root / "revision_experiments" / "profiling.py",
        "revision_experiments/selection_logging.py": root / "revision_experiments" / "selection_logging.py",
        "revision_experiments/offline_diagnostics.py": root / "revision_experiments" / "offline_diagnostics.py",
        "revision_experiments/training_integration.py": root / "revision_experiments" / "training_integration.py",
    }
    ido_modules = {
        "gcdd/ido.py": root / "gcdd" / "ido.py",
        "gcdd/ido_batch.py": root / "gcdd" / "ido_batch.py",
        "gcdd/features.py": root / "gcdd" / "features.py",
        "gcdd/data.py": root / "gcdd" / "data.py",
        "gcdd/lora_training.py": root / "gcdd" / "lora_training.py",
        "gcdd/checkpoint_validation.py": root / "gcdd" / "checkpoint_validation.py",
    }
    required = [training_function, training_entry, *d1_d2_modules.values(), *ido_modules.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot build revision provenance; missing source files: {missing}")
    dirty_output = _git_output(root, "status", "--porcelain")
    git_head = _git_output(root, "rev-parse", "HEAD")
    return {
        "git_head": git_head if git_head else "unavailable",
        "git_dirty": None if dirty_output is None else bool(dirty_output),
        "training_function_sha256": _sha256_file(training_function),
        "training_entry_sha256": _sha256_file(training_entry),
        "d1_d2_module_sha256": {
            relative_path: _sha256_file(path) for relative_path, path in d1_d2_modules.items()
        },
        # This is intentionally present for every monitored method.  It makes
        # a dirty-worktree IDO result self-identifying without changing the
        # training path for the pre-existing methods.
        "ido_module_sha256": {
            relative_path: _sha256_file(path) for relative_path, path in ido_modules.items()
        },
    }


@dataclass(frozen=True)
class SelectionRunMetadata:
    """Non-oracle metadata required for one future selection-log run."""

    dataset: str
    noise_protocol: str
    noise_seed: int | None
    training_seed: int

    def __post_init__(self) -> None:
        if not self.dataset or not self.noise_protocol:
            raise ValueError("dataset and noise_protocol are required for selection logging")
        if self.noise_seed is not None and self.noise_seed < 0:
            raise ValueError("noise_seed must be non-negative or None")
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")


def profiling_enabled(profiler: RunProfiler | None) -> bool:
    return profiler is not None and profiler.enabled


@contextmanager
def profiling_stage(
    profiler: RunProfiler | None,
    name: str,
    *,
    phase: str | None = None,
    data_loading_included: bool = False,
) -> Iterator[None]:
    """No-op when monitoring is absent/off; otherwise close stages on errors."""
    if not profiling_enabled(profiler):
        with nullcontext():
            yield
        return
    assert profiler is not None
    with profiler.stage(name, phase=phase, data_loading_included=data_loading_included):
        yield


@contextmanager
def profiling_memory_phase(profiler: RunProfiler | None, phase: str) -> Iterator[None]:
    """No-op CPU/off-safe wrapper around a non-nestable RunProfiler phase."""
    if not profiling_enabled(profiler):
        with nullcontext():
            yield
        return
    assert profiler is not None
    with profiler.memory_phase(phase):
        yield


def record_optimizer_and_scheduler_step(
    profiler: RunProfiler | None,
    *,
    epoch: int,
    amp_enabled: bool,
    scale_before: float | None,
    scale_after: float | None,
    scheduler_stepped: bool = True,
) -> bool:
    """Record existing step calls and return whether AMP applied an update.

    A disabled scaler always updates.  With AMP, a reduced post-update scale is
    the public GradScaler signal for an overflow-skipped optimizer update.  The
    ``scheduler_stepped`` lets a caller record a corrected AMP path that does
    not advance its scheduler for a skipped optimizer update.  The default
    preserves existing callers' record-only behavior.
    """
    actual_update = not amp_enabled or scale_before is None or scale_after is None or scale_after >= scale_before
    if profiling_enabled(profiler):
        assert profiler is not None
        profiler.record_optimizer_step(actual_update=actual_update, epoch=epoch)
        if scheduler_stepped:
            profiler.record_scheduler_step()
    return actual_update


def _as_aligned_vector(value: np.ndarray | None, length: int, name: str, *, bool_vector: bool = False) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=bool if bool_vector else None)
    if array.shape != (length,):
        raise ValueError(f"{name} must be aligned to the full training pool with shape ({length},)")
    return array


def _optional_score(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def build_selection_sample_records(
    *,
    train_paths: Sequence[str],
    observed_labels: np.ndarray,
    candidate_mask: np.ndarray,
    losses: np.ndarray | None,
    prototype_scores: np.ndarray | None,
    loss_candidate_mask: np.ndarray | None,
    prototype_candidate_mask: np.ndarray | None,
    final_selected_mask: np.ndarray,
    epoch: int,
    selection_strategy: str,
    model_state_identifier: str,
    metadata: SelectionRunMetadata,
    experiment_id: str,
) -> list[SelectionSampleRecord]:
    """Build full-pool-aligned records from already-computed selection arrays."""
    length = len(train_paths)
    labels = np.asarray(observed_labels)
    candidate = np.asarray(candidate_mask, dtype=bool)
    final_selected = np.asarray(final_selected_mask, dtype=bool)
    if labels.shape != (length,) or candidate.shape != (length,) or final_selected.shape != (length,):
        raise ValueError("paths, labels, candidate_mask, and final_selected_mask must share full-pool length")
    loss_values = _as_aligned_vector(losses, length, "losses")
    prototype_values = _as_aligned_vector(prototype_scores, length, "prototype_scores")
    loss_membership = _as_aligned_vector(loss_candidate_mask, length, "loss_candidate_mask", bool_vector=True)
    prototype_membership = _as_aligned_vector(
        prototype_candidate_mask, length, "prototype_candidate_mask", bool_vector=True
    )
    if loss_membership is not None and loss_values is None:
        raise ValueError("loss_candidate_mask requires losses")
    if prototype_membership is not None and prototype_values is None:
        raise ValueError("prototype_candidate_mask requires prototype_scores")
    records: list[SelectionSampleRecord] = []
    for index in np.flatnonzero(candidate):
        loss_score = None if loss_values is None else _optional_score(float(loss_values[index]))
        prototype_score = None if prototype_values is None else _optional_score(float(prototype_values[index]))
        loss_candidate = None if loss_membership is None else bool(loss_membership[index])
        prototype_candidate = None if prototype_membership is None else bool(prototype_membership[index])
        if loss_candidate is True and loss_score is None:
            raise ValueError(f"Candidate {index} has loss membership but no finite loss score")
        if prototype_candidate is True and prototype_score is None:
            raise ValueError(f"Candidate {index} has prototype membership but no finite prototype score")
        records.append(
            SelectionSampleRecord(
                experiment_id=experiment_id,
                dataset=metadata.dataset,
                noise_protocol=metadata.noise_protocol,
                noise_seed=metadata.noise_seed,
                training_seed=metadata.training_seed,
                epoch=int(epoch),
                stable_sample_id=f"train_index:{int(index)}",
                sample_path=str(train_paths[int(index)]),
                observed_label=str(labels[int(index)]),
                loss_score=loss_score,
                prototype_score=prototype_score,
                loss_candidate=loss_candidate,
                prototype_candidate=prototype_candidate,
                final_selected=bool(final_selected[int(index)]),
                selection_strategy=selection_strategy,
                model_state_identifier=model_state_identifier,
            )
        )
    return records


def actual_pgdf_fallback_labels(
    observed_labels: np.ndarray,
    candidate_mask: np.ndarray,
    loss_candidate_mask: np.ndarray,
    prototype_candidate_mask: np.ndarray,
) -> set[str]:
    """Derive only the classes for which existing PGDF entered fallback.

    The existing code first uses prototype candidates, then falls back to the
    whole observed-label class if that prototype subset is empty.  This exact
    rule is declared in the log; it is not rewritten to fit the logger.
    """
    labels = np.asarray(observed_labels).astype(str)
    candidate = np.asarray(candidate_mask, dtype=bool)
    loss_mask = np.asarray(loss_candidate_mask, dtype=bool)
    prototype_mask = np.asarray(prototype_candidate_mask, dtype=bool)
    if not (labels.shape == candidate.shape == loss_mask.shape == prototype_mask.shape):
        raise ValueError("PGDF fallback inputs must share full-pool shape")
    intersection = candidate & loss_mask & prototype_mask
    return {
        str(label)
        for label in sorted(set(labels[candidate].tolist()))
        if not np.any(intersection[candidate & (labels == label)])
    }


def record_selection_update(
    logger: SelectionLogger,
    *,
    train_paths: Sequence[str],
    observed_labels: np.ndarray,
    candidate_mask: np.ndarray,
    losses: np.ndarray | None,
    prototype_scores: np.ndarray | None,
    loss_candidate_mask: np.ndarray | None,
    prototype_candidate_mask: np.ndarray | None,
    final_selected_mask: np.ndarray,
    epoch: int,
    selection_strategy: str,
    model_state_identifier: str,
    metadata: SelectionRunMetadata,
) -> None:
    """Record one already-finalized selection update without recomputation."""
    records = build_selection_sample_records(
        train_paths=train_paths,
        observed_labels=observed_labels,
        candidate_mask=candidate_mask,
        losses=losses,
        prototype_scores=prototype_scores,
        loss_candidate_mask=loss_candidate_mask,
        prototype_candidate_mask=prototype_candidate_mask,
        final_selected_mask=final_selected_mask,
        epoch=epoch,
        selection_strategy=selection_strategy,
        model_state_identifier=model_state_identifier,
        metadata=metadata,
        experiment_id=logger.experiment_id,
    )
    fallback_labels: set[str] | None = None
    fallback_rule: str | None = None
    if selection_strategy == "loss_and_proto":
        if loss_candidate_mask is None or prototype_candidate_mask is None:
            raise ValueError("loss_and_proto selection logging requires both candidate masks")
        fallback_labels = actual_pgdf_fallback_labels(
            observed_labels, candidate_mask, loss_candidate_mask, prototype_candidate_mask
        )
        fallback_rule = PGDF_ACTUAL_FALLBACK_RULE if fallback_labels else None
    logger.record_update(
        records,
        full_training_pool_size=len(records),
        fallback_observed_labels=fallback_labels,
        fallback_rule=fallback_rule,
    )


def write_instrumentation_outputs(
    *,
    output_root: Path,
    profiler: RunProfiler | None,
    selection_logger: SelectionLogger | None,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Persist only completed observer outputs after a successful train return."""
    paths: dict[str, Path] = {}
    experiment_ids = {
        observer.experiment_id
        for observer in (profiler, selection_logger)
        if observer is not None
    }
    if len(experiment_ids) > 1:
        raise ValueError("profiler and selection_logger must use the same experiment_id")
    if run_metadata is not None and not experiment_ids:
        raise ValueError("run_metadata requires at least one observer with an experiment_id")
    if not experiment_ids:
        return paths

    experiment_id = next(iter(experiment_ids))
    destination = experiment_directory(output_root, experiment_id)

    # Validate every enabled observer and all known destinations *before* the
    # first complete file is written. This is intentionally a small preflight,
    # not a cross-file transaction framework.
    if profiler is not None and profiler.enabled:
        profiler.to_dict()
        profiling_path = destination / "profiling.json"
        if profiling_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing profiling result: {profiling_path}")
    if selection_logger is not None:
        selection_logger.assert_ready_to_finalize()
        selection_path = destination / "selection_records"
        if selection_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing selection records: {selection_path}")
    if run_metadata is not None:
        payload = dict(run_metadata)
        if payload.get("experiment_id", experiment_id) != experiment_id:
            raise ValueError("run_metadata experiment_id disagrees with observer experiment_id")
        metadata_path = destination / "run_metadata.json"
        if metadata_path.exists():
            raise FileExistsError(f"Refusing to overwrite instrumentation metadata: {metadata_path}")

    if profiler is not None and profiler.enabled:
        paths["profiling"] = profiler.write_json(output_root)
    if selection_logger is not None:
        paths["selection_records"] = selection_logger.finalize(output_root)
    if run_metadata is not None:
        payload["experiment_id"] = experiment_id
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / ".run_metadata.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, metadata_path)
        paths["run_metadata"] = metadata_path
    return paths
