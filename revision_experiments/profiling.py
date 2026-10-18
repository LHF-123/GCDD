"""Opt-in wall-clock, CUDA-memory, and step-count monitoring.

The caller owns every measurement boundary.  In particular, the PGDF dynamic
selection update must call separate stages for ``selection_loss_forward`` and
``selection_cls_forward``: the submitted implementation performs two distinct
no-gradient full-training-pool forwards at the same model parameter state.

No model call is made by this module.  ``basic`` stores aggregate stage totals;
``detailed`` additionally retains individual timing events.  CUDA timing
synchronizes immediately before and after a requested CUDA stage.  Therefore a
stage duration includes work dispatched between those two boundaries, plus the
caller-owned CPU work inside the scope, but excludes work outside the scope.
Data loading is included only when the caller deliberately wraps it.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from ._paths import experiment_directory


class ProfilingMode(str, Enum):
    """Monitoring levels; ``OFF`` has no timer or memory side effects."""

    OFF = "off"
    BASIC = "basic"
    DETAILED = "detailed"


STAGE_NAMES = frozenset(
    {
        "total_training",
        "fit_time",
        "ido_stage1_training",
        "ido_stage2_training",
        "bmm_fit",
        "official_test",
        "epoch",
        "optimizer_training",
        "validation",
        "checkpoint_save",
        "selection_update_total",
        "selection_loss_forward",
        "selection_cls_forward",
        "prototype_construction",
        "fusion_selection",
        "inference_only",
    }
)


class ProfilingError(RuntimeError):
    """Raised for invalid explicit profiling calls."""


@dataclass(frozen=True)
class StageToken:
    """Opaque token returned by :meth:`RunProfiler.start_stage`."""

    owner_id: int
    token_id: int
    name: str
    phase: str | None
    started_at: float | None
    cuda_synchronized: bool
    synchronize_cuda_requested: bool
    active: bool


@dataclass(frozen=True)
class StageEvent:
    name: str
    phase: str | None
    duration_seconds: float
    timing_method: str
    cuda_synchronized: bool
    data_loading_included: bool
    status: str = "recorded"


@dataclass(frozen=True)
class MemoryMetrics:
    """CUDA peak bytes for a phase, or an explicit non-measured status."""

    allocated_bytes: int | None
    reserved_bytes: int | None
    status: str


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("experiment_id must be a non-empty path-safe component")
    return value


class RunProfiler:
    """Explicit stage profiler with optional CUDA peak-memory observation.

    The profiler does not infer optimizer updates.  Call
    :meth:`record_optimizer_step` only after an actual update; for AMP, pass
    ``actual_update=False`` if GradScaler skipped the optimizer step.  Scheduler
    steps are likewise recorded only after the actual ``scheduler.step()`` call.

    ``memory_phase`` is the only API that resets CUDA peak counters. Before a
    reset it captures the process peak into an internal all-run maximum, and it
    merges the phase peak back into that maximum when the scope closes. This
    preserves ``overall_run_peak`` despite per-phase reset operations. CUDA
    memory is deliberately ``N/A`` on CPU or unavailable CUDA.
    """

    def __init__(
        self,
        experiment_id: str,
        mode: ProfilingMode | str = ProfilingMode.BASIC,
        *,
        device_type: str = "cpu",
        torch_module: Any | None = None,
    ) -> None:
        self.experiment_id = _safe_component(experiment_id)
        self.mode = ProfilingMode(mode)
        self.device_type = device_type
        self._torch_module = torch_module
        self._token_owner_id = id(self)
        self._next_token_id = 1
        self._active: dict[int, StageToken] = {}
        self._active_names: set[str] = set()
        self._stage_totals: dict[str, dict[str, float | int]] = {}
        self._events: list[StageEvent] = []
        self._optimizer_steps_attempted = 0
        self._optimizer_steps_actual = 0
        self._scheduler_steps_actual = 0
        self._training_batches_seen = 0
        self._optimizer_steps_by_epoch: dict[int, int] = {}
        self._selection_interval_steps: dict[int, int] = {}
        self._memory_phases: dict[str, MemoryMetrics] = {}
        self._overall_allocated_peak: int | None = None
        self._overall_reserved_peak: int | None = None
        self._active_memory_phase: str | None = None

    @property
    def enabled(self) -> bool:
        return self.mode is not ProfilingMode.OFF

    def _cuda_available(self) -> bool:
        if self.device_type != "cuda":
            return False
        torch_module = self._torch_module
        if torch_module is None:
            try:
                import torch as torch_module_imported
            except ImportError:
                return False
            torch_module = torch_module_imported
            self._torch_module = torch_module
        return bool(getattr(torch_module, "cuda", None) and torch_module.cuda.is_available())

    def _synchronize_if_cuda(self) -> bool:
        if not self._cuda_available():
            return False
        self._torch_module.cuda.synchronize()
        return True

    def _read_cuda_peaks(self) -> MemoryMetrics:
        if not self._cuda_available():
            return MemoryMetrics(None, None, "not_applicable_cpu_or_cuda_unavailable")
        return MemoryMetrics(
            int(self._torch_module.cuda.max_memory_allocated()),
            int(self._torch_module.cuda.max_memory_reserved()),
            "recorded",
        )

    def _merge_overall_peaks(self, metrics: MemoryMetrics) -> None:
        if metrics.status != "recorded":
            return
        self._overall_allocated_peak = max(self._overall_allocated_peak or 0, metrics.allocated_bytes or 0)
        self._overall_reserved_peak = max(self._overall_reserved_peak or 0, metrics.reserved_bytes or 0)

    def _record_phase_peak(self, phase: str, metrics: MemoryMetrics) -> None:
        """Merge repeated phase observations without losing an earlier peak."""
        previous = self._memory_phases.get(phase)
        if previous is None:
            self._memory_phases[phase] = metrics
            return
        if previous.status == "recorded" and metrics.status == "recorded":
            self._memory_phases[phase] = MemoryMetrics(
                max(previous.allocated_bytes or 0, metrics.allocated_bytes or 0),
                max(previous.reserved_bytes or 0, metrics.reserved_bytes or 0),
                "recorded",
            )
            return
        # A recorded observation is more informative than a CPU/unavailable
        # placeholder. Otherwise retain the existing equivalent status.
        if metrics.status == "recorded":
            self._memory_phases[phase] = metrics

    def start_stage(
        self,
        name: str,
        *,
        phase: str | None = None,
        synchronize_cuda: bool = True,
        data_loading_included: bool = False,
    ) -> StageToken:
        """Start one explicit stage; same-name re-entry is rejected.

        Different stage names may nest. Parent and child durations intentionally
        overlap and are never summed to represent a total duration.
        """
        if name not in STAGE_NAMES:
            raise ValueError(f"Unknown profiling stage {name!r}; use one of {sorted(STAGE_NAMES)}")
        if not self.enabled:
            return StageToken(self._token_owner_id, 0, name, phase, None, False, False, False)
        if name in self._active_names:
            raise ProfilingError(f"Stage {name!r} is already active")
        cuda_synchronized = synchronize_cuda and self._synchronize_if_cuda()
        token = StageToken(
            self._token_owner_id,
            self._next_token_id,
            name,
            phase,
            time.perf_counter(),
            cuda_synchronized,
            synchronize_cuda,
            True,
        )
        self._next_token_id += 1
        self._active[token.token_id] = token
        self._active_names.add(name)
        # Retain caller-declared boundary semantics in detailed events only.
        if self.mode is ProfilingMode.DETAILED:
            self._events.append(
                StageEvent(name, phase, 0.0, "pending", cuda_synchronized, data_loading_included, "started")
            )
        return token

    def stop_stage(self, token: StageToken, *, data_loading_included: bool = False) -> float | None:
        """Stop a stage and return duration, or ``None`` while profiling is off."""
        if token.owner_id != self._token_owner_id:
            raise ProfilingError("Stage token belongs to a different RunProfiler instance")
        if not token.active:
            return None
        current = self._active.pop(token.token_id, None)
        if current is None:
            raise ProfilingError(f"Stage token {token.token_id} is not active or was already stopped")
        self._active_names.remove(current.name)
        # CUDA is asynchronous.  A requested measurement synchronizes at both
        # boundaries; an explicitly disabled measurement must never acquire an
        # accidental stop-side synchronization merely because start-side sync
        # was absent.
        cuda_synchronized_at_stop = (
            self._synchronize_if_cuda() if current.synchronize_cuda_requested else False
        )
        cuda_synchronized = current.cuda_synchronized or cuda_synchronized_at_stop
        duration = time.perf_counter() - (current.started_at or time.perf_counter())
        totals = self._stage_totals.setdefault(current.name, {"count": 0, "duration_seconds": 0.0})
        totals["count"] = int(totals["count"]) + 1
        totals["duration_seconds"] = float(totals["duration_seconds"]) + duration
        if self.mode is ProfilingMode.DETAILED:
            # Replace the matching pending start event with a completed event.
            for index in range(len(self._events) - 1, -1, -1):
                event = self._events[index]
                if event.status == "started" and event.name == current.name and event.phase == current.phase:
                    self._events[index] = StageEvent(
                        current.name,
                        current.phase,
                        duration,
                        "perf_counter_with_cuda_synchronization" if cuda_synchronized else "perf_counter",
                        cuda_synchronized,
                        data_loading_included,
                    )
                    break
        return duration

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        phase: str | None = None,
        synchronize_cuda: bool = True,
        data_loading_included: bool = False,
    ) -> Iterator[None]:
        """Context-manager form of :meth:`start_stage` / :meth:`stop_stage`."""
        token = self.start_stage(
            name,
            phase=phase,
            synchronize_cuda=synchronize_cuda,
            data_loading_included=data_loading_included,
        )
        try:
            yield
        finally:
            self.stop_stage(token, data_loading_included=data_loading_included)

    @contextmanager
    def memory_phase(self, phase: str) -> Iterator[None]:
        """Record one phase peak without losing the profiler's all-run peak."""
        if not self.enabled:
            yield
            return
        if self._active_memory_phase is not None:
            raise ProfilingError("memory_phase scopes cannot nest because they reset CUDA peak counters")
        if not phase:
            raise ValueError("memory phase must be non-empty")
        self._active_memory_phase = phase
        if not self._cuda_available():
            try:
                yield
            finally:
                self._record_phase_peak(
                    phase, MemoryMetrics(None, None, "not_applicable_cpu_or_cuda_unavailable")
                )
                self._active_memory_phase = None
            return
        self._synchronize_if_cuda()
        self._merge_overall_peaks(self._read_cuda_peaks())
        self._torch_module.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            self._synchronize_if_cuda()
            metrics = self._read_cuda_peaks()
            self._record_phase_peak(phase, metrics)
            self._merge_overall_peaks(metrics)
            self._active_memory_phase = None

    def observe_overall_memory_peak(self) -> None:
        """Merge current CUDA high-water marks without resetting counters."""
        if self.enabled:
            self._merge_overall_peaks(self._read_cuda_peaks())

    def record_training_batch(self) -> None:
        if self.enabled:
            self._training_batches_seen += 1

    def record_optimizer_step(self, *, actual_update: bool = True, epoch: int | None = None) -> None:
        """Record an attempted AMP/optimizer step and whether it updated parameters."""
        if self.enabled:
            if epoch is not None and epoch < 1:
                raise ValueError("epoch must be >=1 when provided")
            self._optimizer_steps_attempted += 1
            self._optimizer_steps_actual += int(actual_update)
            if actual_update and epoch is not None:
                self._optimizer_steps_by_epoch[epoch] = self._optimizer_steps_by_epoch.get(epoch, 0) + 1

    def record_scheduler_step(self) -> None:
        if self.enabled:
            self._scheduler_steps_actual += 1

    def record_selection_interval_steps(self, epoch: int, steps: int) -> None:
        if not self.enabled:
            return
        if epoch < 1 or steps < 0:
            raise ValueError("selection interval epoch must be >=1 and steps must be >=0")
        self._selection_interval_steps[epoch] = steps

    def _stage_summary(self, name: str) -> dict[str, Any]:
        data = self._stage_totals.get(name)
        if data is None:
            return {"status": "not_recorded", "count": 0, "duration_seconds": None}
        return {"status": "recorded", "count": int(data["count"]), "duration_seconds": float(data["duration_seconds"])}

    def to_dict(self) -> dict[str, Any]:
        """Return structured results; it is invalid to export with active stages."""
        if self._active or self._active_memory_phase is not None:
            raise ProfilingError("Cannot export while profiling stages or memory phases remain active")
        if not self.enabled:
            return {
                "experiment_id": self.experiment_id,
                "profiling_mode": self.mode.value,
                "device_type": self.device_type,
                "recording_status": "off",
                "stage_totals": {},
                "memory": {"overall_run_peak": asdict(MemoryMetrics(None, None, "not_recorded")), "phases": {}},
                "optimizer_steps_attempted": None,
                "optimizer_steps_actual": None,
                "scheduler_steps_actual": None,
                "training_batches_seen": None,
                "optimizer_steps_by_epoch": {},
                "selection_interval_steps": {},
                "detailed_events": [],
            }
        self.observe_overall_memory_peak()
        overall_status = "recorded" if self._overall_allocated_peak is not None else "not_applicable_cpu_or_cuda_unavailable"
        default_phase_status = "recorded" if self._cuda_available() else "not_applicable_cpu_or_cuda_unavailable"
        phase_metrics = {
            name: asdict(
                self._memory_phases.get(name, MemoryMetrics(None, None, "not_recorded" if default_phase_status == "recorded" else default_phase_status))
            )
            for name in sorted({"training", "selection", *self._memory_phases})
        }
        return {
            "experiment_id": self.experiment_id,
            "profiling_mode": self.mode.value,
            "device_type": self.device_type,
            "recording_status": "complete",
            "stage_totals": {name: self._stage_summary(name) for name in sorted(STAGE_NAMES)},
            "memory": {
                "overall_run_peak": asdict(
                    MemoryMetrics(self._overall_allocated_peak, self._overall_reserved_peak, overall_status)
                ),
                "phases": phase_metrics,
                "reserved_interpretation": "CUDA allocator reserved bytes; not a tensor-occupancy estimate.",
            },
            "optimizer_steps_attempted": self._optimizer_steps_attempted,
            "optimizer_steps_actual": self._optimizer_steps_actual,
            "scheduler_steps_actual": self._scheduler_steps_actual,
            "training_batches_seen": self._training_batches_seen,
            "optimizer_steps_by_epoch": dict(sorted(self._optimizer_steps_by_epoch.items())),
            "selection_interval_steps": dict(sorted(self._selection_interval_steps.items())),
            "detailed_events": [asdict(event) for event in self._events] if self.mode is ProfilingMode.DETAILED else [],
            "measurement_notes": {
                "stage_boundaries": "Caller-owned; nested stage durations may overlap and are not summed as total time.",
                "total_training": "For the PGDF integration, this is the broad train_dynamic_loss_lora() interval from after device resolution through return. It includes pre-loop setup, model fitting, post-fit evaluation, and checkpoint serialization when those operations occur; argument validation and device resolution precede its timing boundary.",
                "fit_time": "For the PGDF integration, this is the epoch-loop model-fitting interval: optimization, validation, and selection updates. It excludes pre-loop setup, post-fit official-test evaluation, and checkpoint serialization.",
                "ido_stage1_training": "IDO Stage 1 full epoch-loop cost: training dataloading, optimization, label-wave accounting, and history snapshots. It excludes Stage 2, post-fit official-test evaluation, and checkpoint serialization.",
                "ido_stage2_training": "IDO Stage 2 full epoch-loop cost: training dataloading, weak/strong optimization, epoch validation, and non-terminal BMM refreshes. It excludes post-fit official-test evaluation and checkpoint serialization.",
                "bmm_fit": "IDO class-wise BMM fit calls, timed separately from optimizer loops.",
                "official_test": "IDO evaluation of the validation-selected checkpoint on the official test split. It is separate from epoch validation and never used to select a checkpoint.",
                "cuda_timing": "CUDA stages synchronize at start and stop when CUDA is available.",
                "profiling_overhead": "basic uses timer boundaries only; detailed retains per-stage events. GPU overhead requires future GPU validation.",
            },
        }

    def write_json(self, output_root: Path) -> Path:
        """Write ``profiling.json`` under an isolated future-experiment directory.

        Existing complete or partial directories are rejected rather than
        overwritten. This method never targets the historical ``outputs/`` tree
        unless a caller explicitly passes it, which future runners must not do.
        """
        # Validate completion before creating any result directory or temporary
        # file, so an active stage cannot leave an incomplete profiling output.
        payload = self.to_dict()
        destination = experiment_directory(output_root, self.experiment_id)
        if destination.exists() and not destination.is_dir():
            raise FileExistsError(f"Experiment path exists but is not a directory: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "profiling.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing profiling result: {path}")
        temporary = destination / ".profiling.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        return path
