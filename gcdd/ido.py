"""IDO-style DINOv2-LoRA controlled adaptation.

This module adapts the two-stage IDO procedure to the repository's fixed
clean-validation protocol.  It deliberately does not claim an architecture-
equivalent reproduction of the official ViT-B/16, ResNet, or ConvNeXt code.

The implementation preserves the paper's observed-label class-wise BMM path
and the official WELoss terms.  It is deliberately not an unmodified
official-source reproduction: WELoss fixes the official cross-sample entropy
broadcast, and a degenerate class-wise BMM is recorded and aborts the run
rather than silently changing the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .features import resolve_device
from .lora_training import (
    DINOv2LoRAClassifier,
    build_scheduler,
    build_dinov2_vitb14_lora_structure_contract,
    build_transforms,
    canonical_structure_sha256,
    capture_lora_runtime_structure,
    count_total_params,
    count_trainable_params,
    evaluate_lora,
    evaluate_state_lora,
    freeze_all,
    inject_lora,
    lora_parameters,
    parse_target_modules,
    resolve_image_path,
    safe_ratio,
    set_torch_seed,
)
from .progress import log_stage, progress_iter
from revision_experiments.profiling import RunProfiler
from revision_experiments.training_integration import (
    profiling_enabled,
    profiling_stage,
    record_optimizer_and_scheduler_step,
)


IDO_ADAPTATION_NAME = "IDO-style DINOv2-B/14+LoRA controlled adaptation"
IDO_GLOBAL_BMM_ADAPTATION_NAME = "IDO-style global-BMM DINOv2-B/14+LoRA controlled adaptation"
IDO_BMM_MODE_CLASSWISE = "classwise"
IDO_BMM_MODE_GLOBAL = "global"
IDO_BMM_MODES = (IDO_BMM_MODE_CLASSWISE, IDO_BMM_MODE_GLOBAL)
IDO_BMM_SCOPE = "observed_label_classwise"
IDO_GLOBAL_BMM_SCOPE = "all_observed_training_pool"
IDO_DEGENERATE_POLICY = "fail_closed_record_and_abort"
IDO_WAVE_SNAPSHOT_POLICY = "deep_clone_trainable_state"
IDO_WELoss_SOURCE = "official_repository_WELoss_logits_mse_with_per_sample_entropy_broadcast_fix"


def resolve_ido_bmm_mode(mode: str | None) -> str:
    """Return an explicit supported BMM mode without changing the legacy default.

    ``None`` deliberately resolves to ``classwise`` so existing IDO command
    lines retain their prior, observed-label class-wise semantics.  The global
    variant is opt-in and carries its own method identity and experiment
    signature downstream.
    """
    resolved = IDO_BMM_MODE_CLASSWISE if mode in (None, "") else str(mode).strip().lower()
    if resolved not in IDO_BMM_MODES:
        raise ValueError(f"ido_bmm_mode must be one of {IDO_BMM_MODES}, got {mode!r}.")
    return resolved


def ido_bmm_scope(mode: str | None) -> str:
    """Return the auditable fit scope for a resolved BMM mode."""
    return IDO_GLOBAL_BMM_SCOPE if resolve_ido_bmm_mode(mode) == IDO_BMM_MODE_GLOBAL else IDO_BMM_SCOPE


def ido_adaptation_name(mode: str | None) -> str:
    """Return the non-interchangeable controlled-adaptation method identity."""
    return (
        IDO_GLOBAL_BMM_ADAPTATION_NAME
        if resolve_ido_bmm_mode(mode) == IDO_BMM_MODE_GLOBAL
        else IDO_ADAPTATION_NAME
    )


@dataclass
class IDOLoraRunResult:
    """Artifacts returned from one IDO-style seed run."""

    logs: list[dict[str, Any]]
    summary: dict[str, Any]
    bmm_fit_rows: list[dict[str, Any]]
    stage1_wrong_event: np.ndarray
    wrong_event: np.ndarray
    trainable_modules: list[str]
    trainable_params: int
    total_params: int


class BMMDegenerateError(RuntimeError):
    """Fail closed when a class cannot support a trustworthy two-component BMM.

    ``record`` is serializable and is intentionally carried to the entry point
    so the failed observed-label class, stage, reason, and input statistics are
    written before the run exits.  No clean/noisy oracle is accepted here.
    """

    def __init__(self, reason: str, values: np.ndarray) -> None:
        raw = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = raw[np.isfinite(raw)]
        unique_count = int(np.unique(raw).size) if np.all(np.isfinite(raw)) else 0
        self.record: dict[str, Any] = {
            "fit_round": "",
            "stage2_epoch": "",
            "observed_label_id": "",
            "observed_label": "",
            "sample_count": int(len(raw)),
            "unique_wrong_event_count": unique_count,
            "wrong_event_min": float(finite.min()) if len(finite) else "",
            "wrong_event_max": float(finite.max()) if len(finite) else "",
            "bmm_status": "degenerate_fail_closed",
            "bmm_reason": str(reason),
            "bmm_fit_iterations": 0,
            "alpha_1": "",
            "beta_1": "",
            "weight_1": "",
            "alpha_2": "",
            "beta_2": "",
            "weight_2": "",
            "component_1_normalized_mean": "",
            "component_2_normalized_mean": "",
        }
        super().__init__(f"IDO class-wise BMM degenerated: {reason}; samples={len(raw)}")

    def add_context(
        self,
        *,
        observed_label_id: int,
        observed_label: str,
        stage2_epoch: int,
        fit_round: str,
    ) -> "BMMDegenerateError":
        self.record.update(
            {
                "fit_round": fit_round,
                "stage2_epoch": int(stage2_epoch),
                "observed_label_id": int(observed_label_id),
                "observed_label": str(observed_label),
            }
        )
        return self

    def add_fit_parameters(self, *, alphas: np.ndarray, betas: np.ndarray, weights: np.ndarray) -> "BMMDegenerateError":
        """Attach the last EM parameters when degeneracy was detected post-fit.

        Constant/non-finite input necessarily has no meaningful fit parameters,
        but a boundary or component-collapse diagnosis must leave its actual
        final values in the fail-closed audit record.
        """
        self.record.update(
            {
                "alpha_1": float(alphas[0]) if len(alphas) > 0 and np.isfinite(alphas[0]) else "",
                "beta_1": float(betas[0]) if len(betas) > 0 and np.isfinite(betas[0]) else "",
                "weight_1": float(weights[0]) if len(weights) > 0 and np.isfinite(weights[0]) else "",
                "alpha_2": float(alphas[1]) if len(alphas) > 1 and np.isfinite(alphas[1]) else "",
                "beta_2": float(betas[1]) if len(betas) > 1 and np.isfinite(betas[1]) else "",
                "weight_2": float(weights[1]) if len(weights) > 1 and np.isfinite(weights[1]) else "",
                "component_1_normalized_mean": float(alphas[0] / (alphas[0] + betas[0])) if len(alphas) > 0 and len(betas) > 0 and np.isfinite(alphas[0]) and np.isfinite(betas[0]) and alphas[0] + betas[0] > 0.0 else "",
                "component_2_normalized_mean": float(alphas[1] / (alphas[1] + betas[1])) if len(alphas) > 1 and len(betas) > 1 and np.isfinite(alphas[1]) and np.isfinite(betas[1]) and alphas[1] + betas[1] > 0.0 else "",
            }
        )
        return self


class IDOTwoViewDataset:
    """Return weak/strong views while retaining original training-pool IDs."""

    def __init__(
        self,
        paths: list[str],
        labels: np.ndarray,
        indices: np.ndarray,
        label_to_id: dict[str, int],
        weak_transform: Any,
        strong_transform: Any,
        path_maps: list[tuple[str, str]],
    ) -> None:
        self.paths = paths
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_to_id = label_to_id
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.path_maps = path_maps

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> tuple[Any, Any, int, int]:
        from PIL import Image

        index = int(self.indices[position])
        image_path = resolve_image_path(self.paths[index], self.path_maps)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            weak = self.weak_transform(rgb)
            strong = self.strong_transform(rgb)
        return weak, strong, int(self.label_to_id[str(self.labels[index])]), index


class ClassConditionalBetaMixture:
    """Official-style two-component BMM with fail-closed degeneracy detection.

    The equations intentionally retain the official BMM initialization and EM
    updates.  Unlike the official code, unsupported or numerically collapsed
    fits never receive a fabricated reliability posterior: a
    :class:`BMMDegenerateError` stops the run.
    """

    def __init__(self, *, epsilon: float = 1.0e-6) -> None:
        self.epsilon = float(epsilon)
        self.component_boundary = 10.0 * self.epsilon
        self.alphas = np.asarray([1.0, 2.0], dtype=np.float64)
        self.betas = np.asarray([2.0, 1.0], dtype=np.float64)
        self.weights = np.asarray([0.5, 0.5], dtype=np.float64)
        self.min_value: float | None = None
        self.max_value: float | None = None
        self.status = "unfitted"
        self.reason = ""

    def fit(self, values: np.ndarray, *, max_iter: int) -> dict[str, Any]:
        """Fit the class BMM or terminate instead of altering the loss silently."""
        raw = np.asarray(values, dtype=np.float64).reshape(-1)
        self._reset_parameters()
        unique_count = int(np.unique(raw).size) if np.all(np.isfinite(raw)) else 0
        if len(raw) < 2:
            raise BMMDegenerateError("fewer_than_two_samples", raw)
        if not np.all(np.isfinite(raw)):
            raise BMMDegenerateError("nonfinite_wrong_event", raw)
        if unique_count < 2:
            raise BMMDegenerateError("constant_wrong_event", raw)

        self.min_value = float(np.min(raw))
        self.max_value = float(np.max(raw))
        if not np.isfinite(self.min_value) or not np.isfinite(self.max_value) or self.max_value <= self.min_value:
            raise BMMDegenerateError("invalid_normalization_range", raw)

        x = self._normalize(raw)
        try:
            for _ in range(int(max_iter)):
                responsibilities = self._responsibilities(x)
                self.alphas[0], self.betas[0] = self._fit_beta_weighted(x, responsibilities[0])
                if self.betas[0] < 1.02:
                    self.betas[0] = 1.1
                self.alphas[1], self.betas[1] = self._fit_beta_weighted(x, responsibilities[1])
                if self.alphas[1] < 1.02:
                    self.alphas[1] = 1.1
                self.weights = responsibilities.sum(axis=1)
                self.weights /= self.weights.sum()
        except (FloatingPointError, OverflowError, ValueError) as exc:
            raise BMMDegenerateError(f"fit_exception:{type(exc).__name__}", raw) from exc

        shape_parameters = np.concatenate((self.alphas, self.betas))
        if not np.all(np.isfinite(shape_parameters)) or np.any(shape_parameters <= 0.0):
            raise self._post_fit_error("nonfinite_or_nonpositive_fit", raw)
        # Check mixture mass first: this makes an explicit collapse observable
        # even if the zero-mass component also produced a boundary beta value.
        if not np.all(np.isfinite(self.weights)):
            raise self._post_fit_error("nonfinite_component_weight", raw)
        if np.any(self.weights <= self.component_boundary) or np.any(self.weights >= 1.0 - self.component_boundary):
            raise self._post_fit_error("component_weight_collapse", raw)
        if np.any(self.alphas <= self.component_boundary) or np.any(self.betas <= self.component_boundary):
            raise self._post_fit_error("component_parameter_at_numeric_boundary", raw)
        # The official initialization assigns component 1 to the lower
        # wrong-event mode (Beta(1, 2)) and component 2 to the higher mode
        # (Beta(2, 1)); WELoss attaches different semantics to prob1/prob2.
        # EM is permutation-invariant, so silently accepting a strict label
        # swap would invert those semantics.  Treat a verified swap as an
        # auditable fail-closed condition rather than relabelling components
        # or fabricating an alternative reliability rule.
        component_means = self._component_means()
        if component_means[0] > component_means[1]:
            raise self._post_fit_error("component_order_violation", raw)
        self.status = "fitted"
        self.reason = ""
        return self.fit_metadata(raw, unique_count, int(max_iter))

    def predict(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Return posterior/CDF terms, or fail closed on a prediction anomaly."""
        raw = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(raw)):
            raise BMMDegenerateError("nonfinite_prediction_input", raw)
        if self.status != "fitted" or self.min_value is None or self.max_value is None or self.max_value <= self.min_value:
            raise BMMDegenerateError("prediction_without_valid_fit_range", raw)

        normalized = (raw - self.min_value) / (self.max_value - self.min_value)
        clipped = int(np.count_nonzero((normalized <= 0.0) | (normalized >= 1.0)))
        normalized = np.clip(normalized, self.epsilon, 1.0 - self.epsilon)
        try:
            from scipy import stats

            posterior_terms = np.vstack(
                [
                    self.weights[component]
                    * stats.beta.pdf(normalized, self.alphas[component], self.betas[component])
                    for component in range(2)
                ]
            )
            posterior = posterior_terms / (posterior_terms.sum(axis=0, keepdims=True) + self.epsilon)
            cdf = np.vstack(
                [stats.beta.cdf(normalized, self.alphas[component], self.betas[component]) for component in range(2)]
            )
        except (FloatingPointError, OverflowError, ValueError) as exc:
            raise BMMDegenerateError(f"prediction_exception:{type(exc).__name__}", raw) from exc
        if not np.all(np.isfinite(posterior)) or not np.all(np.isfinite(cdf)):
            raise BMMDegenerateError("nonfinite_prediction", raw)
        return posterior[0], posterior[1], cdf[0], cdf[1], clipped

    def fit_metadata(self, raw: np.ndarray, unique_count: int, max_iter: int) -> dict[str, Any]:
        return {
            "sample_count": int(len(raw)),
            "unique_wrong_event_count": int(unique_count),
            "wrong_event_min": float(np.min(raw)) if len(raw) else "",
            "wrong_event_max": float(np.max(raw)) if len(raw) else "",
            "bmm_status": self.status,
            "bmm_reason": self.reason,
            "bmm_fit_iterations": int(max_iter),
            "alpha_1": float(self.alphas[0]),
            "beta_1": float(self.betas[0]),
            "weight_1": float(self.weights[0]),
            "alpha_2": float(self.alphas[1]),
            "beta_2": float(self.betas[1]),
            "weight_2": float(self.weights[1]),
            "component_1_normalized_mean": float(self._component_means()[0]),
            "component_2_normalized_mean": float(self._component_means()[1]),
        }

    def _reset_parameters(self) -> None:
        self.alphas = np.asarray([1.0, 2.0], dtype=np.float64)
        self.betas = np.asarray([2.0, 1.0], dtype=np.float64)
        self.weights = np.asarray([0.5, 0.5], dtype=np.float64)
        self.min_value = None
        self.max_value = None
        self.status = "unfitted"
        self.reason = ""

    def _post_fit_error(self, reason: str, raw: np.ndarray) -> BMMDegenerateError:
        return BMMDegenerateError(reason, raw).add_fit_parameters(
            alphas=self.alphas,
            betas=self.betas,
            weights=self.weights,
        )

    def _component_means(self) -> np.ndarray:
        return self.alphas / (self.alphas + self.betas)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        assert self.min_value is not None and self.max_value is not None
        normalized = (values - self.min_value) / (self.max_value - self.min_value)
        return normalized * (1.0 - 2.0 * self.epsilon) + self.epsilon

    def _weighted_mean(self, values: np.ndarray, weights: np.ndarray) -> float:
        return float(np.sum(weights * values) / (np.sum(weights) + self.epsilon))

    def _fit_beta_weighted(self, values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        mean = self._weighted_mean(values, weights)
        variance = self._weighted_mean((values - mean) ** 2, weights)
        variance = max(variance, self.epsilon)
        alpha = mean * ((mean * (1.0 - mean)) / variance - 1.0)
        beta = alpha * (1.0 - mean) / mean
        return max(float(alpha), self.epsilon), max(float(beta), self.epsilon)

    def _responsibilities(self, values: np.ndarray) -> np.ndarray:
        from scipy import stats

        likelihood = np.vstack(
            [
                self.weights[component] * stats.beta.pdf(values, self.alphas[component], self.betas[component])
                for component in range(2)
            ]
        )
        likelihood[likelihood <= self.epsilon] = self.epsilon
        return likelihood / (likelihood.sum(axis=0, keepdims=True) + self.epsilon)


def _prediction_diagnostics(
    model: ClassConditionalBetaMixture,
    values: np.ndarray,
) -> dict[str, Any]:
    """Describe fitted BMM outputs without changing their training semantics.

    The official BMM divides component densities by ``density_sum + epsilon``.
    Consequently ``prob1 + prob2`` is not asserted to be exactly one.  The
    deficit is recorded rather than normalized away because WELoss retains the
    official ``1 - prob1`` and ``1 - prob2`` expressions.
    """
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    prob1, prob2, cdf1, cdf2, clipped = model.predict(raw)
    posterior_mass = prob1 + prob2

    def component_support(responsibility: np.ndarray, component: int) -> dict[str, Any]:
        soft = float(np.sum(responsibility))
        squared = float(np.sum(np.square(responsibility)))
        return {
            f"component_{component}_soft_assignment": soft,
            f"component_{component}_kish_ess": float((soft * soft) / squared) if squared > 0.0 else 0.0,
            f"component_{component}_weight_times_n": float(model.weights[component - 1] * len(raw)),
            f"component_{component}_max_responsibility": float(np.max(responsibility)),
            f"component_{component}_responsibility_ge_half_count": int(np.count_nonzero(responsibility >= 0.5)),
        }

    return {
        **component_support(prob1, 1),
        **component_support(prob2, 2),
        "posterior_1_min": float(np.min(prob1)),
        "posterior_1_max": float(np.max(prob1)),
        "posterior_2_min": float(np.min(prob2)),
        "posterior_2_max": float(np.max(prob2)),
        "cdf_1_min": float(np.min(cdf1)),
        "cdf_1_max": float(np.max(cdf1)),
        "cdf_2_min": float(np.min(cdf2)),
        "cdf_2_max": float(np.max(cdf2)),
        "posterior_mass_sum": float(np.sum(posterior_mass)),
        "posterior_mass_deficit": float(len(raw) - np.sum(posterior_mass)),
        "posterior_mass_min": float(np.min(posterior_mass)),
        "posterior_mass_max": float(np.max(posterior_mass)),
        "bmm_prediction_clipped_count": int(clipped),
    }


class ClasswiseBMM:
    """One BMM per observed-label class, never a hidden global fallback."""

    def __init__(self, observed_label_names: list[str], *, epsilon: float = 1.0e-6) -> None:
        self.observed_label_names = list(observed_label_names)
        self.models = {label_id: ClassConditionalBetaMixture(epsilon=epsilon) for label_id in range(len(observed_label_names))}

    def fit(
        self,
        wrong_event: np.ndarray,
        label_ids: np.ndarray,
        *,
        stage2_epoch: int,
        fit_round: str,
        max_iter: int,
    ) -> list[dict[str, Any]]:
        wrong_event = np.asarray(wrong_event, dtype=np.float64)
        label_ids = np.asarray(label_ids, dtype=np.int64)
        if wrong_event.shape != label_ids.shape:
            raise ValueError("wrong_event and label_ids must share the same shape.")
        rows: list[dict[str, Any]] = []
        for label_id, observed_label in enumerate(self.observed_label_names):
            values = wrong_event[label_ids == label_id]
            try:
                metadata = self.models[label_id].fit(values, max_iter=max_iter)
            except BMMDegenerateError as exc:
                exc.record.update({"bmm_mode": IDO_BMM_MODE_CLASSWISE, "bmm_scope": IDO_BMM_SCOPE})
                raise exc.add_context(
                    observed_label_id=label_id,
                    observed_label=observed_label,
                    stage2_epoch=stage2_epoch,
                    fit_round=fit_round,
                ) from exc
            rows.append(
                {
                    "bmm_mode": IDO_BMM_MODE_CLASSWISE,
                    "bmm_scope": IDO_BMM_SCOPE,
                    "fit_round": fit_round,
                    "stage2_epoch": int(stage2_epoch),
                    "observed_label_id": int(label_id),
                    "observed_label": observed_label,
                    **metadata,
                    **_prediction_diagnostics(self.models[label_id], values),
                }
            )
        return rows

    def predict(
        self,
        wrong_event: np.ndarray,
        label_ids: np.ndarray,
        *,
        stage2_epoch: int,
    ) -> dict[str, np.ndarray | int]:
        wrong_event = np.asarray(wrong_event, dtype=np.float64)
        label_ids = np.asarray(label_ids, dtype=np.int64)
        if wrong_event.shape != label_ids.shape:
            raise ValueError("wrong_event and label_ids must share the same shape.")
        outputs = {name: np.empty(len(wrong_event), dtype=np.float64) for name in ("prob1", "prob2", "cdf1", "cdf2")}
        clipped_predictions = 0
        for label_id, model in self.models.items():
            positions = np.where(label_ids == label_id)[0]
            if not len(positions):
                continue
            try:
                prob1, prob2, cdf1, cdf2, clipped = model.predict(wrong_event[positions])
            except BMMDegenerateError as exc:
                exc.record.update({"bmm_mode": IDO_BMM_MODE_CLASSWISE, "bmm_scope": IDO_BMM_SCOPE})
                raise exc.add_context(
                    observed_label_id=label_id,
                    observed_label=self.observed_label_names[label_id],
                    stage2_epoch=stage2_epoch,
                    fit_round="stage2_prediction",
                ) from exc
            outputs["prob1"][positions] = prob1
            outputs["prob2"][positions] = prob2
            outputs["cdf1"][positions] = cdf1
            outputs["cdf2"][positions] = cdf2
            clipped_predictions += int(clipped)
        posterior_mass_deficit = 1.0 - (outputs["prob1"] + outputs["prob2"])
        return {
            **outputs,
            "fallback_predictions": 0,
            "clipped_predictions": clipped_predictions,
            "posterior_mass_deficit_sum": float(np.sum(posterior_mass_deficit)),
            "posterior_mass_deficit_max": float(np.max(posterior_mass_deficit)),
        }

    def fallback_class_count(self) -> int:
        """Retained for result-schema compatibility; fail-closed has no fallback."""
        return 0


class GlobalBMM:
    """One official-style BMM over the full observed training pool.

    This is an opt-in controlled adaptation, never a hidden fallback from the
    class-wise branch.  ``label_ids`` are accepted only to verify alignment
    with the training pool; no observed-label partitioning occurs here.
    """

    def __init__(self, observed_label_names: list[str], *, epsilon: float = 1.0e-6) -> None:
        self.observed_label_names = list(observed_label_names)
        self.model = ClassConditionalBetaMixture(epsilon=epsilon)

    def _validate_inputs(self, wrong_event: np.ndarray, label_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(wrong_event, dtype=np.float64).reshape(-1)
        ids = np.asarray(label_ids, dtype=np.int64).reshape(-1)
        if values.shape != ids.shape:
            raise ValueError("wrong_event and label_ids must share the same shape.")
        if np.any(ids < 0) or np.any(ids >= len(self.observed_label_names)):
            raise ValueError("label_ids contain an observed class outside the configured vocabulary.")
        return values, ids

    def fit(
        self,
        wrong_event: np.ndarray,
        label_ids: np.ndarray,
        *,
        stage2_epoch: int,
        fit_round: str,
        max_iter: int,
    ) -> list[dict[str, Any]]:
        values, _ = self._validate_inputs(wrong_event, label_ids)
        try:
            metadata = self.model.fit(values, max_iter=max_iter)
        except BMMDegenerateError as exc:
            exc.record.update({"bmm_mode": IDO_BMM_MODE_GLOBAL, "bmm_scope": IDO_GLOBAL_BMM_SCOPE})
            raise exc.add_context(
                observed_label_id=-1,
                observed_label="__global_observed_training_pool__",
                stage2_epoch=stage2_epoch,
                fit_round=fit_round,
            ) from exc
        return [
            {
                "bmm_mode": IDO_BMM_MODE_GLOBAL,
                "bmm_scope": IDO_GLOBAL_BMM_SCOPE,
                "fit_round": fit_round,
                "stage2_epoch": int(stage2_epoch),
                "observed_label_id": "",
                "observed_label": "",
                **metadata,
                **_prediction_diagnostics(self.model, values),
            }
        ]

    def predict(
        self,
        wrong_event: np.ndarray,
        label_ids: np.ndarray,
        *,
        stage2_epoch: int,
    ) -> dict[str, np.ndarray | int | float]:
        values, _ = self._validate_inputs(wrong_event, label_ids)
        try:
            prob1, prob2, cdf1, cdf2, clipped = self.model.predict(values)
        except BMMDegenerateError as exc:
            exc.record.update({"bmm_mode": IDO_BMM_MODE_GLOBAL, "bmm_scope": IDO_GLOBAL_BMM_SCOPE})
            raise exc.add_context(
                observed_label_id=-1,
                observed_label="__global_observed_training_pool__",
                stage2_epoch=stage2_epoch,
                fit_round="stage2_prediction",
            ) from exc
        posterior_mass_deficit = 1.0 - (prob1 + prob2)
        return {
            "prob1": prob1,
            "prob2": prob2,
            "cdf1": cdf1,
            "cdf2": cdf2,
            "fallback_predictions": 0,
            "clipped_predictions": int(clipped),
            "posterior_mass_deficit_sum": float(np.sum(posterior_mass_deficit)),
            "posterior_mass_deficit_max": float(np.max(posterior_mass_deficit)),
        }

    def fallback_class_count(self) -> int:
        return 0


def resolve_stage_budget(total_epochs: int, stage1_epochs: int | None = None) -> tuple[int, int]:
    """Scale IDO's official 5:10 stage ratio to the controlled total budget."""
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive.")
    if stage1_epochs is None:
        if total_epochs % 3:
            raise ValueError(
                "IDO controlled adaptation preserves the official 1:2 stage ratio; total epochs must be divisible by 3 "
                "or stage1_epochs must be explicitly pre-registered."
            )
        stage1_epochs = total_epochs // 3
    stage2_epochs = total_epochs - int(stage1_epochs)
    if stage1_epochs <= 0 or stage2_epochs <= 0:
        raise ValueError("IDO requires positive Stage 1 and Stage 2 budgets.")
    return int(stage1_epochs), int(stage2_epochs)


def snapshot_trainable_state(model: Any) -> dict[str, Any]:
    """Deep-copy LoRA/head state so label-wave history cannot share storage."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if "lora_" in name or name.startswith("head.")
    }


def build_ido_lora_structure_metadata(
    *,
    torch: Any,
    model: Any,
    trainable_modules: list[str],
    cfg: dict[str, Any],
    num_classes: int,
) -> dict[str, Any]:
    """Record frozen and live LoRA structure evidence for later GPU-pilot audit.

    The frozen contract is derived from the controlled DINOv2-B/14 catalog and
    the effective target configuration.  The runtime record is separately
    captured from the instantiated model after ``inject_lora``.  Completion
    auditing relies on the former; the latter makes a subsequent GPU pilot
    able to prove that the real hub model matched that contract.
    """
    target_modules = parse_target_modules(str(cfg["lora"].get("target_modules", "qkv")))
    rank = int(cfg["lora"].get("rank", 8))
    frozen_contract = build_dinov2_vitb14_lora_structure_contract(
        target_modules=target_modules,
        rank=rank,
        num_classes=num_classes,
    )
    runtime_structure = capture_lora_runtime_structure(
        model,
        injected_modules=trainable_modules,
        target_modules=target_modules,
        rank=rank,
    )
    local_repo = str(cfg.get("feature", {}).get("local_repo") or "").strip()
    if local_repo:
        pretrained_source = {
            "loader": "torch.hub",
            "entrypoint": "dinov2_vitb14",
            "source_mode": "configured_local_repo",
            "configured_local_repo": local_repo,
        }
    else:
        pretrained_source = {
            "loader": "torch.hub",
            "entrypoint": "dinov2_vitb14",
            "source_mode": "local_cache_or_github_fallback",
            "repository": "facebookresearch/dinov2",
        }
    try:
        import torchvision

        torchvision_version = str(torchvision.__version__)
    except Exception:  # optional provenance only; training must not depend on torchvision metadata import.
        torchvision_version = "unavailable"
    return {
        "lora_structure_contract": frozen_contract,
        "lora_structure_contract_sha256": canonical_structure_sha256(frozen_contract),
        "lora_runtime_structure": runtime_structure,
        "lora_runtime_structure_sha256": canonical_structure_sha256(runtime_structure),
        "dinov2_pretrained_source": pretrained_source,
        "runtime_environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": str(getattr(torch, "__version__", "unknown")),
            "torchvision": torchvision_version,
        },
    }


def update_label_wave_state(
    *,
    previous_prediction: np.ndarray | None,
    previous_change: int,
    label_wave_saved: bool,
    current_prediction: np.ndarray,
    train_idx: np.ndarray,
    stage_epoch: int,
    stage1_epochs: int,
) -> tuple[int | str, bool, np.ndarray, int, bool]:
    """Apply the official first-non-decrease label-wave rule without I/O.

    The initial epoch is the deterministic fallback base.  Thereafter, a
    strictly smaller adjacent prediction-change count replaces the candidate;
    the first non-decrease locks the previous candidate.  A strictly decreasing
    sequence through the final Stage-1 epoch uses that final epoch, matching
    the official control flow while making the off-by-one rule unit-testable.
    """
    if previous_prediction is None:
        return "", True, current_prediction, int(len(train_idx)), False
    prediction_changes = int(np.count_nonzero(current_prediction[train_idx] != previous_prediction[train_idx]))
    if label_wave_saved:
        return prediction_changes, False, previous_prediction, int(previous_change), True
    if prediction_changes < previous_change:
        return (
            prediction_changes,
            True,
            current_prediction,
            prediction_changes,
            bool(stage_epoch == stage1_epochs),
        )
    return prediction_changes, False, previous_prediction, int(previous_change), True


def _assert_finite_trainable_gradients(torch: Any, model: Any) -> None:
    """Stop before an optimizer update if any LoRA/head gradient is non-finite."""
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"IDO non-finite gradient for trainable parameter {name!r}.")


def _ido_d1_summary(
    profiler: RunProfiler | None,
    *,
    stage1_processed_samples: int,
    stage2_processed_samples: int,
) -> dict[str, Any]:
    """Expose IDO D1 metrics without inventing measurements while profiling is off."""
    if not profiling_enabled(profiler):
        return {
            "d1_recording_status": "off",
            "d1_stage1_wall_seconds": "",
            "d1_stage2_wall_seconds": "",
            "d1_optimizer_loop_wall_seconds": "",
            "d1_bmm_fit_seconds": "",
            "d1_checkpoint_seconds": "",
            "d1_official_test_seconds": "",
            "d1_end_to_end_wall_seconds": "",
            "d1_total_wall_seconds": "",
            "d1_optimizer_steps_attempted": "",
            "d1_optimizer_steps_actual": "",
            "d1_peak_cuda_allocated_bytes": "",
            "d1_peak_cuda_reserved_bytes": "",
            "d1_validation_seconds": "",
            "d1_stage1_processed_samples": int(stage1_processed_samples),
            "d1_stage2_processed_samples": int(stage2_processed_samples),
            "d1_stage2_extra_view_forwards": int(stage2_processed_samples),
        }
    assert profiler is not None
    payload = profiler.to_dict()
    stages = payload["stage_totals"]
    overall_memory = payload["memory"]["overall_run_peak"]

    def duration(name: str) -> float | str:
        value = stages[name]["duration_seconds"]
        return "" if value is None else float(value)

    return {
        "d1_recording_status": str(payload["recording_status"]),
        "d1_stage1_wall_seconds": duration("ido_stage1_training"),
        "d1_stage2_wall_seconds": duration("ido_stage2_training"),
        "d1_optimizer_loop_wall_seconds": duration("optimizer_training"),
        "d1_bmm_fit_seconds": duration("bmm_fit"),
        "d1_checkpoint_seconds": duration("checkpoint_save"),
        "d1_official_test_seconds": duration("official_test"),
        # ``total_training`` is deliberately retained for existing analysis
        # readers.  It is an end-to-end function interval, not a pure
        # optimizer metric; use the explicitly named fields above for a
        # like-for-like training-cost comparison.
        "d1_end_to_end_wall_seconds": duration("total_training"),
        "d1_total_wall_seconds": duration("total_training"),
        "d1_optimizer_steps_attempted": int(payload["optimizer_steps_attempted"]),
        "d1_optimizer_steps_actual": int(payload["optimizer_steps_actual"]),
        "d1_peak_cuda_allocated_bytes": overall_memory["allocated_bytes"],
        "d1_peak_cuda_reserved_bytes": overall_memory["reserved_bytes"],
        "d1_validation_seconds": duration("validation"),
        "d1_stage1_processed_samples": int(stage1_processed_samples),
        "d1_stage2_processed_samples": int(stage2_processed_samples),
        "d1_stage2_extra_view_forwards": int(stage2_processed_samples),
    }


def _weloss_batch_vector(torch: Any, value: np.ndarray, *, batch_size: int, dtype: Any, device: Any, name: str) -> Any:
    """Validate a BMM term and preserve its one-value-per-sample semantics."""
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (batch_size,):
        raise ValueError(f"{name} must have shape ({batch_size},), got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return torch.as_tensor(array, dtype=dtype, device=device)


def official_we_loss_per_sample(
    torch: Any,
    output_s: Any,
    output_w: Any,
    prob1: np.ndarray,
    prob2: np.ndarray,
    cdf1: np.ndarray,
    cdf2: np.ndarray,
    labels: Any,
) -> dict[str, Any]:
    """Return corrected official-source WELoss contributions, each shaped ``[B]``.

    The official implementation computes logits MSE and the weight
    ``p1*cdf1 + (1-p2)*(1-cdf2)``.  Those definitions are retained.  Its
    ``confidence.unsqueeze(1)`` produced a cross-sample ``[B,B]`` entropy term;
    this controlled correction keeps confidence as ``[B]`` so all reductions
    are per sample before the final mean.
    """
    functional = torch.nn.functional
    if output_s.ndim != 2 or output_w.ndim != 2 or output_s.shape != output_w.shape:
        raise ValueError("output_s and output_w must both have shape [B, C] with identical shapes.")
    batch_size = int(output_s.shape[0])
    if batch_size <= 0:
        raise ValueError("WELoss requires a non-empty batch.")
    if labels.ndim != 1 or int(labels.shape[0]) != batch_size:
        raise ValueError(f"labels must have shape ({batch_size},), got {tuple(labels.shape)}.")
    dtype = output_s.dtype
    device = output_s.device
    prob_tensor = _weloss_batch_vector(torch, prob1, batch_size=batch_size, dtype=dtype, device=device, name="prob1")
    prob2_tensor = _weloss_batch_vector(torch, prob2, batch_size=batch_size, dtype=dtype, device=device, name="prob2")
    cdf1_tensor = _weloss_batch_vector(torch, cdf1, batch_size=batch_size, dtype=dtype, device=device, name="cdf1")
    cdf2_tensor = _weloss_batch_vector(torch, cdf2, batch_size=batch_size, dtype=dtype, device=device, name="cdf2")
    y_pred_s = functional.softmax(output_s, dim=1).clamp(1.0e-4, 1.0 - 1.0e-4)
    y_pred_w = functional.softmax(output_w, dim=1).clamp(1.0e-4, 1.0 - 1.0e-4)
    ce_s = functional.cross_entropy(output_s, labels, reduction="none")
    ce_w = functional.cross_entropy(output_w, labels, reduction="none")
    mse = functional.mse_loss(output_s, output_w, reduction="none").mean(dim=1)
    mse_weight = prob_tensor * cdf1_tensor + (1.0 - prob2_tensor) * (1.0 - cdf2_tensor)
    y_pred_avg = (y_pred_s + y_pred_w) / 2.0
    confidence = torch.max(y_pred_avg, dim=1)[0]
    entropy = -confidence * torch.sum(y_pred_avg * torch.log(y_pred_avg), dim=1)
    terms = {
        "ce_per_sample": prob_tensor * (ce_s + ce_w),
        "mse_per_sample": mse_weight * mse,
        "entropy_per_sample": (1.0 - prob_tensor) * entropy,
        "clean_posterior": prob_tensor,
        "noisy_posterior": prob2_tensor,
        "cdf_clean": cdf1_tensor,
        "cdf_noisy": cdf2_tensor,
        "consistency_weight": mse_weight,
        "confidence": confidence,
        "logits_mse": mse,
        "confidence_weighted_entropy": entropy,
    }
    for name, value in terms.items():
        if tuple(value.shape) != (batch_size,):
            raise RuntimeError(f"WELoss {name} must be per-sample [B], got {tuple(value.shape)}.")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"WELoss {name} is non-finite.")
    return terms


def official_we_loss(
    torch: Any,
    output_s: Any,
    output_w: Any,
    prob1: np.ndarray,
    prob2: np.ndarray,
    cdf1: np.ndarray,
    cdf2: np.ndarray,
    labels: Any,
) -> tuple[Any, dict[str, float]]:
    """Mean the corrected per-sample official-source WELoss terms."""
    terms = official_we_loss_per_sample(torch, output_s, output_w, prob1, prob2, cdf1, cdf2, labels)
    ce_term = terms["ce_per_sample"].mean()
    mse_term = terms["mse_per_sample"].mean()
    entropy_term = terms["entropy_per_sample"].mean()
    total = ce_term + mse_term + entropy_term
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("WELoss total is non-finite.")
    return total, {
        "ce_term": float(ce_term.detach().cpu()),
        "mse_term": float(mse_term.detach().cpu()),
        "entropy_term": float(entropy_term.detach().cpu()),
        "clean_posterior_mean": float(terms["clean_posterior"].detach().mean().cpu()),
    }


def build_ido_transforms(transforms: Any, input_size: int) -> tuple[Any, Any, Any]:
    """Use common weak/strong views while retaining the repository evaluation view."""
    weak_transform, eval_transform = build_transforms(transforms, input_size)
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    strong_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return weak_transform, strong_transform, eval_transform


def train_ido_lora(
    train_paths: list[str],
    train_labels: np.ndarray,
    eval_paths: list[str],
    eval_labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: dict[str, Any],
    *,
    method: str,
    seed: int,
    stage1_epochs: int | None = None,
    bmm_mode: str | None = None,
    path_maps: list[tuple[str, str]] | None = None,
    checkpoint_path: Path | None = None,
    test_paths: list[str] | None = None,
    test_labels: np.ndarray | None = None,
    final_checkpoint_path: Path | None = None,
    last5_checkpoint_dir: Path | None = None,
    checkpoint_protocol: str = "legacy_test_selected",
    posthoc_oracle_test: bool = False,
    official_test_selected_only: bool = False,
    profiler: RunProfiler | None = None,
    task_contract_sha256: str = "",
    task_id: str = "",
    experiment_signature: str = "",
) -> IDOLoraRunResult:
    """Run a validation-selected IDO-style DINOv2-LoRA controlled adaptation."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    resolved_bmm_mode = resolve_ido_bmm_mode(bmm_mode)
    bmm_scope = ido_bmm_scope(resolved_bmm_mode)
    if (test_paths is None) != (test_labels is None):
        raise ValueError("test_paths and test_labels must be provided together.")
    if official_test_selected_only and posthoc_oracle_test:
        raise ValueError("official_test_selected_only cannot be combined with posthoc_oracle_test.")
    path_maps = path_maps or []
    train_labels = np.asarray(train_labels, dtype=str)
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape != (len(train_labels),):
        raise ValueError(f"train_mask must have shape ({len(train_labels)},), got {train_mask.shape}.")
    if not np.any(train_mask):
        raise ValueError("IDO selected no training images.")

    feature_cfg = cfg["feature"]
    train_cfg = cfg["lora_train"]
    device = resolve_device(torch, feature_cfg.get("device", "auto"))
    if profiling_enabled(profiler):
        assert profiler is not None
        profiler.device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    total_training_token = (
        profiler.start_stage("total_training", phase="train_ido_lora")
        if profiling_enabled(profiler)
        else None
    )
    set_torch_seed(torch, seed)
    classes = sorted(set(train_labels.tolist()))
    label_to_id = {label: index for index, label in enumerate(classes)}
    train_idx = np.where(train_mask)[0]
    eval_known = np.asarray([label in label_to_id for label in eval_labels], dtype=bool)
    eval_idx = np.where(eval_known)[0]
    if not len(eval_idx):
        raise ValueError("Eval split has no labels that appear in the train split.")

    input_size = int(feature_cfg["input_size"])
    weak_transform, strong_transform, eval_transform = build_ido_transforms(transforms, input_size)
    from .lora_training import ImageSplitDataset

    stage1_dataset = ImageSplitDataset(train_paths, train_labels, train_idx, label_to_id, weak_transform, path_maps)
    stage2_dataset = IDOTwoViewDataset(
        train_paths, train_labels, train_idx, label_to_id, weak_transform, strong_transform, path_maps
    )
    eval_dataset = ImageSplitDataset(eval_paths, np.asarray(eval_labels, dtype=str), eval_idx, label_to_id, eval_transform, path_maps)
    loader_kwargs = {
        "batch_size": int(train_cfg["batch_size"]),
        "num_workers": int(train_cfg.get("num_workers", 4)),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
        "drop_last": False,
    }
    stage1_loader = DataLoader(stage1_dataset, shuffle=True, **loader_kwargs)
    stage2_loader = DataLoader(stage2_dataset, shuffle=True, **loader_kwargs)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    test_loader = _build_optional_test_loader(
        torch, DataLoader, ImageSplitDataset, test_paths, test_labels, label_to_id, eval_transform, path_maps, train_cfg
    )

    total_epochs = int(train_cfg["epochs"])
    resolved_stage1_epochs, resolved_stage2_epochs = resolve_stage_budget(total_epochs, stage1_epochs)
    model, trainable_modules = _build_ido_model(torch, cfg, len(classes), device)
    lora_structure_metadata = build_ido_lora_structure_metadata(
        torch=torch,
        model=model,
        trainable_modules=trainable_modules,
        cfg=cfg,
        num_classes=len(classes),
    )
    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)
    amp = bool(train_cfg.get("amp", True)) and device.startswith("cuda")
    wrong_event = np.zeros(len(train_labels), dtype=np.int64)
    logs: list[dict[str, Any]] = []

    log_stage(
        f"[ido] seed={seed}: Stage 1={resolved_stage1_epochs}, Stage 2={resolved_stage2_epochs}, "
        f"BMM={bmm_scope}, train_images={len(train_idx)}, trainable_params={trainable_params}"
    )
    base_state, base_stage_epoch = _run_stage1(
        torch=torch,
        model=model,
        loader=stage1_loader,
        train_cfg=train_cfg,
        device=device,
        amp=amp,
        wrong_event=wrong_event,
        train_idx=train_idx,
        stage1_epochs=resolved_stage1_epochs,
        method=method,
        seed=seed,
        logs=logs,
        trainable_params=trainable_params,
        total_params=total_params,
        profiler=profiler,
    )
    # Preserve the exact Stage-1 statistic used by the first BMM fit.  Stage 2
    # continues to accumulate into ``wrong_event``; exporting that mutable final
    # array under a Stage-1 filename would make the audit artifact misleading.
    stage1_wrong_event = np.asarray(wrong_event, dtype=np.int64).copy()
    stage1_base_checkpoint_path = checkpoint_path.parent / "stage1_base.pt" if checkpoint_path is not None else None
    stage1_checkpoint_token = (
        profiler.start_stage("checkpoint_save", phase="ido_stage1_base")
        if profiling_enabled(profiler) and stage1_base_checkpoint_path is not None
        else None
    )
    if stage1_base_checkpoint_path is not None:
        _save_stage1_base_checkpoint(
            torch=torch,
            path=stage1_base_checkpoint_path,
            method=method,
            seed=seed,
            classes=classes,
            base_state=base_state,
            wrong_event=stage1_wrong_event,
            stage1_epochs=resolved_stage1_epochs,
            stage2_epochs=resolved_stage2_epochs,
            base_stage_epoch=base_stage_epoch,
            checkpoint_protocol=checkpoint_protocol,
            task_contract_sha256=task_contract_sha256,
            task_id=task_id,
            experiment_signature=experiment_signature,
            lora_structure_metadata=lora_structure_metadata,
            bmm_mode=resolved_bmm_mode,
            bmm_scope=bmm_scope,
        )
    if stage1_checkpoint_token is not None:
        profiler.stop_stage(stage1_checkpoint_token)
    model.load_state_dict(base_state, strict=False)
    model.to(device)

    pool_label_ids = np.asarray([label_to_id[str(train_labels[index])] for index in train_idx], dtype=np.int64)
    bmm = (
        GlobalBMM(classes)
        if resolved_bmm_mode == IDO_BMM_MODE_GLOBAL
        else ClasswiseBMM(classes)
    )
    with profiling_stage(profiler, "bmm_fit", phase="stage2_initial"):
        bmm_fit_rows = bmm.fit(
            wrong_event[train_idx], pool_label_ids, stage2_epoch=0, fit_round="stage2_initial", max_iter=20
        )
    optimizer, scheduler = _build_optimizer_and_scheduler(
        torch, model, train_cfg, len(stage2_loader), resolved_stage2_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_row: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    last5_states: list[tuple[int, dict[str, Any]]] = []
    oracle_states: list[tuple[int, dict[str, Any]]] = []

    stage2_training_token = (
        profiler.start_stage("ido_stage2_training", phase="stage2_epoch_loop", data_loading_included=True)
        if profiling_enabled(profiler)
        else None
    )
    for stage_epoch in range(1, resolved_stage2_epochs + 1):
        epoch_token = (
            profiler.start_stage("epoch", phase=f"ido_stage2_epoch_{stage_epoch}", data_loading_included=True)
            if profiling_enabled(profiler)
            else None
        )
        optimizer_token = (
            profiler.start_stage("optimizer_training", phase=f"ido_stage2_epoch_{stage_epoch}", data_loading_included=True)
            if profiling_enabled(profiler)
            else None
        )
        model.train()
        loss_sum = 0.0
        ce_sum = 0.0
        mse_sum = 0.0
        entropy_sum = 0.0
        posterior_sum = 0.0
        posterior_mass_deficit_sum = 0.0
        posterior_mass_deficit_max = 0.0
        seen = 0
        fallback_predictions = 0
        clipped_predictions = 0
        progress = progress_iter(
            stage2_loader,
            total=len(stage2_loader),
            desc=f"IDO Stage2 seed={seed} epoch {stage_epoch}/{resolved_stage2_epochs}",
        )
        for weak_images, strong_images, labels, original_indices in progress:
            if profiling_enabled(profiler):
                assert profiler is not None
                profiler.record_training_batch()
            weak_images = weak_images.to(device, non_blocking=True)
            strong_images = strong_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            index_np = original_indices.detach().cpu().numpy().astype(np.int64)
            label_np = labels.detach().cpu().numpy().astype(np.int64)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                weak_logits = model(weak_images)
                strong_logits = model(strong_images)
                weak_predicted = torch.argmax(weak_logits.detach(), dim=1).cpu().numpy()
                strong_predicted = torch.argmax(strong_logits.detach(), dim=1).cpu().numpy()
                wrong_event[index_np[weak_predicted != label_np]] += 1
                wrong_event[index_np[strong_predicted != label_np]] += 1
                bmm_terms = bmm.predict(wrong_event[index_np], label_np, stage2_epoch=stage_epoch)
                loss, parts = official_we_loss(
                    torch,
                    strong_logits,
                    weak_logits,
                    bmm_terms["prob1"],
                    bmm_terms["prob2"],
                    bmm_terms["cdf1"],
                    bmm_terms["cdf2"],
                    labels,
                )
            scaler.scale(loss).backward()
            _assert_finite_trainable_gradients(torch, model)
            scale_before = float(scaler.get_scale()) if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            scale_after = float(scaler.get_scale()) if scaler.is_enabled() else None
            record_optimizer_and_scheduler_step(
                profiler,
                epoch=resolved_stage1_epochs + stage_epoch,
                amp_enabled=bool(scaler.is_enabled()),
                scale_before=scale_before,
                scale_after=scale_after,
            )
            batch_size = int(labels.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            ce_sum += parts["ce_term"] * batch_size
            mse_sum += parts["mse_term"] * batch_size
            entropy_sum += parts["entropy_term"] * batch_size
            posterior_sum += parts["clean_posterior_mean"] * batch_size
            posterior_mass_deficit_sum += float(bmm_terms["posterior_mass_deficit_sum"])
            posterior_mass_deficit_max = max(
                posterior_mass_deficit_max,
                float(bmm_terms["posterior_mass_deficit_max"]),
            )
            seen += batch_size
            fallback_predictions += int(bmm_terms["fallback_predictions"])
            clipped_predictions += int(bmm_terms["clipped_predictions"])

        if optimizer_token is not None:
            profiler.stop_stage(optimizer_token)
        with profiling_stage(profiler, "validation", phase=f"ido_stage2_epoch_{stage_epoch}", data_loading_included=True):
            top1, top5 = evaluate_lora(torch, model, eval_loader, device, len(classes), amp)
        global_epoch = resolved_stage1_epochs + stage_epoch
        row = _base_log_row(
            method=method,
            seed=seed,
            epoch=global_epoch,
            stage="stage2",
            stage_epoch=stage_epoch,
            optimizer=optimizer,
            loss=safe_ratio(loss_sum, seen),
            top1=top1,
            top5=top5,
            train_samples=len(train_idx),
            eval_samples=len(eval_idx),
            trainable_params=trainable_params,
            total_params=total_params,
            wrong_event_total=int(wrong_event[train_idx].sum()),
        )
        row.update(
            {
                "ce_term": safe_ratio(ce_sum, seen),
                "mse_term": safe_ratio(mse_sum, seen),
                "entropy_term": safe_ratio(entropy_sum, seen),
                "bmm_clean_posterior_mean": safe_ratio(posterior_sum, seen),
                "bmm_fallback_predictions": int(fallback_predictions),
                "bmm_clipped_predictions": int(clipped_predictions),
                "bmm_fallback_classes": int(bmm.fallback_class_count()),
                "bmm_mode": resolved_bmm_mode,
                "bmm_scope": bmm_scope,
                "bmm_posterior_mass_deficit_sum": float(posterior_mass_deficit_sum),
                "bmm_posterior_mass_deficit_max": float(posterior_mass_deficit_max),
            }
        )
        if best_row is None or float(row["top1"]) > float(best_row["top1"]):
            best_row = row
            best_state = snapshot_trainable_state(model)
        row["best_top1"] = float(best_row["top1"])
        row["best_epoch"] = int(best_row["epoch"])
        logs.append(row)
        epoch_state = snapshot_trainable_state(model)
        last5_states.append((global_epoch, epoch_state))
        if len(last5_states) > 5:
            last5_states.pop(0)
        if posthoc_oracle_test and test_loader is not None:
            oracle_states.append((global_epoch, epoch_state))
        if stage_epoch < resolved_stage2_epochs:
            with profiling_stage(profiler, "bmm_fit", phase=f"stage2_epoch_{stage_epoch}"):
                bmm_fit_rows.extend(
                    bmm.fit(
                        wrong_event[train_idx],
                        pool_label_ids,
                        stage2_epoch=stage_epoch,
                        fit_round="stage2_epoch_update",
                        max_iter=2,
                    )
                )
        else:
            log_stage(
                "[ido] Stage2 terminal BMM refit skipped: no subsequent optimization step consumes it; "
                "the last fitted BMM remains the final training-state reliability model."
            )
        log_stage(
            f"[ido] seed={seed} Stage2 {stage_epoch}/{resolved_stage2_epochs}: loss={row['loss']:.4f}, "
            f"val_top1={top1:.4f}, fallback_classes={bmm.fallback_class_count()}"
        )
        if epoch_token is not None:
            profiler.stop_stage(epoch_token)

    if stage2_training_token is not None:
        profiler.stop_stage(stage2_training_token)

    if best_row is None or best_state is None:
        raise RuntimeError("IDO Stage 2 did not produce a validation checkpoint.")
    final_state = snapshot_trainable_state(model)
    official_test_token = (
        profiler.start_stage("official_test", phase="validation_selected_checkpoint")
        if profiling_enabled(profiler) and test_loader is not None
        else None
    )
    protocol_metrics = _evaluate_protocol(
        torch=torch,
        model=model,
        best_state=best_state,
        final_state=final_state,
        best_row=best_row,
        last5_states=last5_states,
        oracle_states=oracle_states,
        test_loader=test_loader,
        device=device,
        num_classes=len(classes),
        amp=amp,
        checkpoint_protocol=checkpoint_protocol,
        posthoc_oracle_test=posthoc_oracle_test,
        official_test_selected_only=official_test_selected_only,
    )
    if official_test_token is not None:
        profiler.stop_stage(official_test_token)
    checkpoint_save_token = (
        profiler.start_stage("checkpoint_save", phase="ido_training_checkpoints")
        if profiling_enabled(profiler)
        and (checkpoint_path is not None or final_checkpoint_path is not None or last5_checkpoint_dir is not None)
        else None
    )
    _save_checkpoints(
        torch=torch,
        model=model,
        method=method,
        seed=seed,
        classes=classes,
        best_state=best_state,
        final_state=final_state,
        best_row=best_row,
        last5_states=last5_states,
        checkpoint_path=checkpoint_path,
        final_checkpoint_path=final_checkpoint_path,
        last5_checkpoint_dir=last5_checkpoint_dir,
        checkpoint_protocol=checkpoint_protocol,
        protocol_metrics=protocol_metrics,
        stage1_epochs=resolved_stage1_epochs,
        stage2_epochs=resolved_stage2_epochs,
        base_stage_epoch=base_stage_epoch,
        task_contract_sha256=task_contract_sha256,
        task_id=task_id,
        experiment_signature=experiment_signature,
        lora_structure_metadata=lora_structure_metadata,
        bmm_mode=resolved_bmm_mode,
        bmm_scope=bmm_scope,
    )
    if checkpoint_save_token is not None:
        profiler.stop_stage(checkpoint_save_token)
    summary = _summarize_ido_logs(
        method=method,
        seed=seed,
        logs=logs,
        stage1_epochs=resolved_stage1_epochs,
        stage2_epochs=resolved_stage2_epochs,
        base_stage_epoch=base_stage_epoch,
        bmm=bmm,
        bmm_mode=resolved_bmm_mode,
        bmm_scope=bmm_scope,
    )
    summary["stage1_base_checkpoint"] = str(stage1_base_checkpoint_path) if stage1_base_checkpoint_path is not None else ""
    summary.update(protocol_metrics)
    summary.update(
        {
            "lora_structure_contract_sha256": lora_structure_metadata["lora_structure_contract_sha256"],
            "lora_runtime_structure_sha256": lora_structure_metadata["lora_runtime_structure_sha256"],
            "dinov2_pretrained_source": lora_structure_metadata["dinov2_pretrained_source"],
            "runtime_environment": lora_structure_metadata["runtime_environment"],
        }
    )
    if total_training_token is not None:
        profiler.stop_stage(total_training_token)
    summary.update(
        _ido_d1_summary(
            profiler,
            stage1_processed_samples=len(train_idx) * resolved_stage1_epochs,
            stage2_processed_samples=len(train_idx) * resolved_stage2_epochs,
        )
    )
    return IDOLoraRunResult(
        logs=logs,
        summary=summary,
        bmm_fit_rows=bmm_fit_rows,
        stage1_wrong_event=stage1_wrong_event,
        wrong_event=wrong_event,
        trainable_modules=trainable_modules,
        trainable_params=trainable_params,
        total_params=total_params,
    )


def _build_ido_model(torch: Any, cfg: dict[str, Any], num_classes: int, device: str) -> tuple[Any, list[str]]:
    model = DINOv2LoRAClassifier.make(torch, cfg, num_classes).to(device)
    freeze_all(model.backbone)
    modules = inject_lora(
        torch,
        model.backbone,
        target_modules=parse_target_modules(str(cfg["lora"].get("target_modules", "qkv"))),
        rank=int(cfg["lora"].get("rank", 8)),
        alpha=float(cfg["lora"].get("alpha", 16.0)),
        dropout=float(cfg["lora"].get("dropout", 0.05)),
    )
    model.to(device)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    return model, modules


def _build_optimizer_and_scheduler(
    torch: Any,
    model: Any,
    train_cfg: dict[str, Any],
    loader_length: int,
    stage_epochs: int,
) -> tuple[Any, Any]:
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters(model), "lr": float(train_cfg["lora_lr"])},
            {"params": model.head.parameters(), "lr": float(train_cfg["head_lr"])},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    total_steps = max(1, int(stage_epochs) * max(1, loader_length))
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
    scheduler = build_scheduler(torch, optimizer, total_steps, warmup_steps, str(train_cfg.get("scheduler", "cosine")))
    return optimizer, scheduler


def _run_stage1(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    train_cfg: dict[str, Any],
    device: str,
    amp: bool,
    wrong_event: np.ndarray,
    train_idx: np.ndarray,
    stage1_epochs: int,
    method: str,
    seed: int,
    logs: list[dict[str, Any]],
    trainable_params: int,
    total_params: int,
    profiler: RunProfiler | None,
) -> tuple[dict[str, Any], int]:
    optimizer, scheduler = _build_optimizer_and_scheduler(torch, model, train_cfg, len(loader), stage1_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    previous_prediction: np.ndarray | None = None
    previous_change = int(len(train_idx))
    base_state: dict[str, Any] | None = None
    base_stage_epoch = 0
    label_wave_saved = False

    stage1_training_token = (
        profiler.start_stage("ido_stage1_training", phase="stage1_epoch_loop", data_loading_included=True)
        if profiling_enabled(profiler)
        else None
    )
    for stage_epoch in range(1, stage1_epochs + 1):
        epoch_token = (
            profiler.start_stage("epoch", phase=f"ido_stage1_epoch_{stage_epoch}", data_loading_included=True)
            if profiling_enabled(profiler)
            else None
        )
        optimizer_token = (
            profiler.start_stage("optimizer_training", phase=f"ido_stage1_epoch_{stage_epoch}", data_loading_included=True)
            if profiling_enabled(profiler)
            else None
        )
        model.train()
        current_prediction = np.full(len(wrong_event), -1, dtype=np.int64)
        loss_sum = 0.0
        seen = 0
        progress = progress_iter(loader, total=len(loader), desc=f"IDO Stage1 seed={seed} epoch {stage_epoch}/{stage1_epochs}")
        for images, labels, original_indices in progress:
            if profiling_enabled(profiler):
                assert profiler is not None
                profiler.record_training_batch()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
            predicted = torch.argmax(logits.detach(), dim=1)
            indices_np = original_indices.detach().cpu().numpy().astype(np.int64)
            predicted_np = predicted.detach().cpu().numpy().astype(np.int64)
            labels_np = labels.detach().cpu().numpy().astype(np.int64)
            wrong_event[indices_np[predicted_np != labels_np]] += 1
            current_prediction[indices_np] = predicted_np
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("IDO Stage 1 loss is non-finite.")
            scaler.scale(loss).backward()
            _assert_finite_trainable_gradients(torch, model)
            scale_before = float(scaler.get_scale()) if scaler.is_enabled() else None
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            scale_after = float(scaler.get_scale()) if scaler.is_enabled() else None
            record_optimizer_and_scheduler_step(
                profiler,
                epoch=stage_epoch,
                amp_enabled=bool(scaler.is_enabled()),
                scale_before=scale_before,
                scale_after=scale_after,
            )
            batch_size = int(labels.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            seen += batch_size

        if optimizer_token is not None:
            profiler.stop_stage(optimizer_token)

        prediction_changes, selected_this_epoch, next_prediction, next_change, label_wave_saved = update_label_wave_state(
            previous_prediction=previous_prediction,
            previous_change=previous_change,
            label_wave_saved=label_wave_saved,
            current_prediction=current_prediction,
            train_idx=train_idx,
            stage_epoch=stage_epoch,
            stage1_epochs=stage1_epochs,
        )
        if selected_this_epoch:
            base_state = snapshot_trainable_state(model)
            base_stage_epoch = stage_epoch
        previous_prediction = next_prediction
        previous_change = next_change
        row = _base_log_row(
            method=method,
            seed=seed,
            epoch=stage_epoch,
            stage="stage1",
            stage_epoch=stage_epoch,
            optimizer=optimizer,
            loss=safe_ratio(loss_sum, seen),
            top1="",
            top5="",
            train_samples=len(train_idx),
            eval_samples=0,
            trainable_params=trainable_params,
            total_params=total_params,
            wrong_event_total=int(wrong_event[train_idx].sum()),
        )
        row.update(
            {
                "stage1_prediction_changes": prediction_changes,
                "stage1_base_snapshot_selected": "yes" if selected_this_epoch or stage_epoch == 1 else "no",
            }
        )
        logs.append(row)
        log_stage(
            f"[ido] seed={seed} Stage1 {stage_epoch}/{stage1_epochs}: loss={row['loss']:.4f}, "
            f"label_wave_changes={prediction_changes if prediction_changes != '' else 'initial'}"
        )
        if epoch_token is not None:
            profiler.stop_stage(epoch_token)
    if base_state is None:
        raise RuntimeError("IDO Stage 1 did not produce a base-model snapshot.")
    if stage1_training_token is not None:
        profiler.stop_stage(stage1_training_token)
    return base_state, base_stage_epoch


def _build_optional_test_loader(
    torch: Any,
    data_loader_cls: Any,
    dataset_cls: Any,
    test_paths: list[str] | None,
    test_labels: np.ndarray | None,
    label_to_id: dict[str, int],
    eval_transform: Any,
    path_maps: list[tuple[str, str]],
    train_cfg: dict[str, Any],
) -> Any | None:
    del torch
    if test_paths is None or test_labels is None:
        return None
    labels = np.asarray(test_labels, dtype=str)
    test_idx = np.where(np.asarray([label in label_to_id for label in labels], dtype=bool))[0]
    if not len(test_idx):
        raise ValueError("Test split has no labels that appear in the train split.")
    dataset = dataset_cls(test_paths, labels, test_idx, label_to_id, eval_transform, path_maps)
    return data_loader_cls(
        dataset,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=False,
    )


def _base_log_row(
    *,
    method: str,
    seed: int,
    epoch: int,
    stage: str,
    stage_epoch: int,
    optimizer: Any,
    loss: float,
    top1: float | str,
    top5: float | str,
    train_samples: int,
    eval_samples: int,
    trainable_params: int,
    total_params: int,
    wrong_event_total: int,
) -> dict[str, Any]:
    return {
        "method": method,
        "seed": int(seed),
        "epoch": int(epoch),
        "stage": stage,
        "stage_epoch": int(stage_epoch),
        "lr_lora": float(optimizer.param_groups[0]["lr"]),
        "lr_head": float(optimizer.param_groups[1]["lr"]),
        "loss": float(loss),
        "train_loss": float(loss),
        "top1": top1,
        "top5": top5,
        "val_top1": top1,
        "val_top5": top5,
        "best_top1": "",
        "best_epoch": "",
        "train_samples": int(train_samples),
        "candidate_samples": int(train_samples),
        "selected_ratio": 1.0,
        "eval_samples": int(eval_samples),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "wrong_event_total": int(wrong_event_total),
        "loss_type": "ido_official_weloss",
        "selection_mode": "ido_classwise_bmm_all_training_pool",
        "selection_strategy": "classwise_bmm_weighted_loss",
        "prototype_mode": "",
    }


def _evaluate_protocol(
    *,
    torch: Any,
    model: Any,
    best_state: dict[str, Any],
    final_state: dict[str, Any],
    best_row: dict[str, Any],
    last5_states: list[tuple[int, dict[str, Any]]],
    oracle_states: list[tuple[int, dict[str, Any]]],
    test_loader: Any | None,
    device: str,
    num_classes: int,
    amp: bool,
    checkpoint_protocol: str,
    posthoc_oracle_test: bool,
    official_test_selected_only: bool,
) -> dict[str, Any]:
    if test_loader is None:
        return {}
    selected_top1, selected_top5 = evaluate_state_lora(torch, model, best_state, test_loader, device, num_classes, amp)
    metrics: dict[str, Any] = {
        "checkpoint_protocol": checkpoint_protocol,
        "official_test_evaluation": "validation_selected_only" if official_test_selected_only else "validation_selected_final_last5",
        "best_val_epoch": int(best_row["epoch"]),
        "best_val_top1": float(best_row["top1"]),
        "best_val_top5": float(best_row["top5"]),
        "validation_selected_test_top1": float(selected_top1),
        "validation_selected_test_top5": float(selected_top5),
    }
    if official_test_selected_only:
        metrics.update({"final_test_top1": "", "final_test_top5": "", "last5_test_mean": "", "last5_test_std": ""})
        return metrics
    final_top1, final_top5 = evaluate_state_lora(torch, model, final_state, test_loader, device, num_classes, amp)
    if posthoc_oracle_test:
        epoch_metrics = {
            epoch: evaluate_state_lora(torch, model, state, test_loader, device, num_classes, amp)
            for epoch, state in oracle_states
        }
        last5 = np.asarray([epoch_metrics[epoch][0] for epoch, _ in last5_states], dtype=np.float32)
        oracle_epoch, (oracle_top1, oracle_top5) = max(epoch_metrics.items(), key=lambda item: item[1][0])
        metrics["official_test_evaluation"] = "posthoc_oracle_curve"
        metrics.update(
            {
                "oracle_best_test_epoch": int(oracle_epoch),
                "oracle_best_test_top1": float(oracle_top1),
                "oracle_best_test_top5": float(oracle_top5),
                "oracle_best_to_final_drop": float(oracle_top1 - final_top1),
            }
        )
    else:
        last5 = np.asarray(
            [evaluate_state_lora(torch, model, state, test_loader, device, num_classes, amp)[0] for _, state in last5_states],
            dtype=np.float32,
        )
    metrics.update(
        {
            "final_test_top1": float(final_top1),
            "final_test_top5": float(final_top5),
            "last5_test_mean": float(last5.mean()),
            "last5_test_std": float(last5.std()),
        }
    )
    return metrics


def _save_stage1_base_checkpoint(
    *,
    torch: Any,
    path: Path,
    method: str,
    seed: int,
    classes: list[str],
    base_state: dict[str, Any],
    wrong_event: np.ndarray,
    stage1_epochs: int,
    stage2_epochs: int,
    base_stage_epoch: int,
    checkpoint_protocol: str,
    task_contract_sha256: str,
    task_id: str,
    experiment_signature: str,
    lora_structure_metadata: dict[str, Any],
    bmm_mode: str,
    bmm_scope: str,
) -> None:
    """Persist the selected Stage-1 boundary for audit, not in-place resume.

    The runner intentionally does not advertise epoch-level resume because that
    would additionally require optimizer, scheduler, scaler, RNG, BMM, and
    label-wave state.  The batch launcher treats a partial attempt as blocked
    and starts only an explicit new attempt directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "method": method,
            "ido_adaptation": ido_adaptation_name(bmm_mode),
            "seed": int(seed),
            "classes": classes,
            "state_dict": base_state,
            "wrong_event": np.asarray(wrong_event, dtype=np.int64).copy(),
            "stage1_epochs": int(stage1_epochs),
            "stage2_epochs": int(stage2_epochs),
            "base_stage_epoch": int(base_stage_epoch),
            "checkpoint_protocol": checkpoint_protocol,
            "resume_support": "not_implemented_fail_closed_new_attempt_required",
            "label_wave_snapshot_policy": IDO_WAVE_SNAPSHOT_POLICY,
            "bmm_degenerate_policy": IDO_DEGENERATE_POLICY,
            "bmm_mode": bmm_mode,
            "bmm_scope": bmm_scope,
            "task_contract_sha256": task_contract_sha256,
            "task_id": task_id,
            "experiment_signature": experiment_signature,
            **lora_structure_metadata,
        },
        path,
    )


def _save_checkpoints(
    *,
    torch: Any,
    model: Any,
    method: str,
    seed: int,
    classes: list[str],
    best_state: dict[str, Any],
    final_state: dict[str, Any],
    best_row: dict[str, Any],
    last5_states: list[tuple[int, dict[str, Any]]],
    checkpoint_path: Path | None,
    final_checkpoint_path: Path | None,
    last5_checkpoint_dir: Path | None,
    checkpoint_protocol: str,
    protocol_metrics: dict[str, Any],
    stage1_epochs: int,
    stage2_epochs: int,
    base_stage_epoch: int,
    task_contract_sha256: str,
    task_id: str,
    experiment_signature: str,
    lora_structure_metadata: dict[str, Any],
    bmm_mode: str,
    bmm_scope: str,
) -> None:
    del model
    metadata = {
        "method": method,
        "seed": int(seed),
        "classes": classes,
        "checkpoint_protocol": checkpoint_protocol,
        "ido_adaptation": ido_adaptation_name(bmm_mode),
        "stage1_epochs": int(stage1_epochs),
        "stage2_epochs": int(stage2_epochs),
        "base_stage1_epoch": int(base_stage_epoch),
        "bmm_mode": bmm_mode,
        "bmm_scope": bmm_scope,
        "bmm_degenerate_policy": IDO_DEGENERATE_POLICY,
        "label_wave_snapshot_policy": IDO_WAVE_SNAPSHOT_POLICY,
        "weloss_source": IDO_WELoss_SOURCE,
        "task_contract_sha256": task_contract_sha256,
        "task_id": task_id,
        "experiment_signature": experiment_signature,
        **lora_structure_metadata,
        **protocol_metrics,
    }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**metadata, "state_dict": best_state, "best_epoch": int(best_row["epoch"]), "best_top1": float(best_row["top1"])}, checkpoint_path)
    if final_checkpoint_path is not None:
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**metadata, "state_dict": final_state, "final_epoch": int(stage1_epochs + stage2_epochs)}, final_checkpoint_path)
    if last5_checkpoint_dir is not None:
        last5_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch, state in last5_states:
            torch.save({**metadata, "state_dict": state, "epoch": int(epoch)}, last5_checkpoint_dir / f"epoch_{epoch:03d}.pt")


def _summarize_ido_logs(
    *,
    method: str,
    seed: int,
    logs: list[dict[str, Any]],
    stage1_epochs: int,
    stage2_epochs: int,
    base_stage_epoch: int,
    bmm: ClasswiseBMM | GlobalBMM,
    bmm_mode: str,
    bmm_scope: str,
) -> dict[str, Any]:
    stage2_logs = [row for row in logs if row["stage"] == "stage2"]
    if not stage2_logs:
        raise ValueError("IDO has no Stage 2 log rows.")
    best = max(stage2_logs, key=lambda row: float(row["top1"]))
    final = stage2_logs[-1]
    return {
        "method": method,
        "seed": int(seed),
        "train_samples": int(final["train_samples"]),
        "candidate_samples": int(final["candidate_samples"]),
        "eval_samples": int(final["eval_samples"]),
        "best_epoch": int(best["epoch"]),
        "best_top1": float(best["top1"]),
        "best_top5": float(best["top5"]),
        "final_top1": float(final["top1"]),
        "final_top5": float(final["top5"]),
        "last5_mean": float(np.asarray([row["top1"] for row in stage2_logs[-5:]], dtype=np.float32).mean()),
        "last5_std": float(np.asarray([row["top1"] for row in stage2_logs[-5:]], dtype=np.float32).std()),
        "trainable_params": int(final["trainable_params"]),
        "total_params": int(final["total_params"]),
        "selection_mode": f"ido_{bmm_mode}_bmm_all_training_pool",
        "selection_strategy": f"{bmm_mode}_bmm_weighted_loss",
        "prototype_mode": "",
        "loss_type": "ido_official_weloss",
        "stage1_epochs": int(stage1_epochs),
        "stage2_epochs": int(stage2_epochs),
        "base_stage1_epoch": int(base_stage_epoch),
        "bmm_mode": bmm_mode,
        "bmm_scope": bmm_scope,
        "bmm_degenerate_policy": IDO_DEGENERATE_POLICY,
        "label_wave_snapshot_policy": IDO_WAVE_SNAPSHOT_POLICY,
        "weloss_source": IDO_WELoss_SOURCE,
        "final_bmm_fallback_classes": int(bmm.fallback_class_count()),
        "backbone_frozen": True,
    }
