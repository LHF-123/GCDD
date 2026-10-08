"""Checkpoint-level CE/KL gradient audit for completed consistency runs.

This is an analysis-only entry point.  It never creates an optimizer or a
scheduler and therefore cannot train or persistently update a model.  It
reconstructs the historical Reliable and Ambiguous memberships from the
run's ``selection_rows.csv`` and uses only noisy-training-pool images.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.analyze_ambiguous_consistency_offline import SelectionGroups, build_groups, selection_rows_by_epoch


DATASETS = ("cub", "cars", "aircraft")
CONSISTENCY_VARIANT = "margin_rank_ambiguous_consistency_sequential"
BASELINE_VARIANT = "margin_rank"
DEFAULT_VARIANTS_ROOT = Path(
    "outputs/neighbor_margin/cyclic_asym40_noise42/fixedval_s20250726/r08_p04_w5_u5/variants"
)
EPSILON = 1.0e-12


@dataclass(frozen=True)
class CheckpointSpec:
    dataset: str
    method: str
    seed: int
    run_dir: Path
    checkpoint_path: Path
    checkpoint_epoch: int
    selection_update_epoch: int
    cfg_path: Path


@dataclass(frozen=True)
class DiagnosticBatch:
    batch_id: int
    reliable_indices: tuple[int, ...]
    ambiguous_indices: tuple[int, ...]
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analysis-only checkpoint CE/KL gradient contribution and conflict audit."
    )
    parser.add_argument("--variants-root", type=Path, default=DEFAULT_VARIANTS_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/ambiguous_consistency_gradient_audit"),
    )
    parser.add_argument("--seed", type=int, default=42, help="Completed training seed to audit.")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--diagnostic-seed", type=int, default=20261008)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--ambiguous-micro-batch-size", type=int, default=20)
    parser.add_argument(
        "--reliable-gradient-mode",
        choices=("auto", "full", "micro"),
        default="auto",
        help=(
            "Reliable CE physical-batch policy. auto first uses a complete AMP batch, then retries "
            "with weighted micro-batches only after CUDA OOM; full never falls back; micro always micro-batches."
        ),
    )
    parser.add_argument(
        "--reliable-micro-batch-size",
        type=int,
        default=20,
        help="Physical Reliable CE micro-batch size used by auto fallback or --reliable-gradient-mode micro.",
    )
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Map a logged Linux path prefix to a local dataset root; repeatable.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also audit available Margin-Rank best checkpoints as counterfactual KL diagnostics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit files and sampling only; do not load images, models, or checkpoints into CUDA.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    materialized = list(rows)
    if fieldnames is None:
        if not materialized:
            raise ValueError(f"Cannot infer an empty CSV schema for {path}.")
        fieldnames = list(materialized[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)


def parse_path_maps(items: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must be OLD=NEW, found {item!r}.")
        old, new = item.split("=", 1)
        if not old or not new:
            raise ValueError(f"--path-map must have non-empty OLD and NEW: {item!r}.")
        parsed.append((old, new))
    return parsed


def selection_epoch_for_checkpoint(checkpoint_epoch: int, available_updates: Iterable[int]) -> int:
    """Return the latest update active while a checkpoint epoch was trained.

    Production code evaluates and snapshots an epoch *before* executing that
    epoch's selection update.  Therefore epoch ``e`` uses the greatest update
    strictly smaller than ``e``.
    """
    eligible = [epoch for epoch in available_updates if epoch < checkpoint_epoch]
    if not eligible:
        raise ValueError(
            f"Checkpoint epoch {checkpoint_epoch} predates every saved selection update; "
            "the requested post-warmup diagnostic cannot be reconstructed."
        )
    return max(eligible)


def find_checkpoint_spec(variants_root: Path, variant: str, dataset: str, seed: int) -> CheckpointSpec:
    run_dir = variants_root / variant / dataset / f"seed{seed}"
    checkpoint_path = run_dir / "checkpoints" / "best_val.pt"
    rows_path = run_dir / "selection_rows.csv"
    cfg_path = run_dir.parent / "resolved_config.yaml"
    missing = [str(path) for path in (checkpoint_path, rows_path, cfg_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete {variant} run for {dataset}/seed{seed}: {missing}")
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("best_epoch"), int):
        raise ValueError(f"Checkpoint lacks an integer best_epoch: {checkpoint_path}")
    checkpoint_epoch = int(payload["best_epoch"])
    update_epoch = selection_epoch_for_checkpoint(checkpoint_epoch, selection_rows_by_epoch(rows_path))
    return CheckpointSpec(
        dataset=dataset,
        method=variant,
        seed=seed,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
        selection_update_epoch=update_epoch,
        cfg_path=cfg_path,
    )


def history_groups(spec: CheckpointSpec) -> SelectionGroups:
    rows_by_epoch = selection_rows_by_epoch(spec.run_dir / "selection_rows.csv")
    if spec.selection_update_epoch not in rows_by_epoch:
        raise ValueError(f"Missing selected update {spec.selection_update_epoch} for {spec.run_dir}")
    return build_groups(rows_by_epoch[spec.selection_update_epoch])


def validate_groups_for_diagnostic(groups: SelectionGroups) -> None:
    if not groups.actual_active:
        raise ValueError("Historical actual active subset is empty.")
    if not groups.ambiguous_for_consistency:
        raise ValueError("Historical Ambiguous-for-consistency subset is empty.")
    if groups.actual_active & groups.ambiguous_for_consistency:
        raise ValueError("A sample appears in both Reliable CE and Ambiguous KL pools.")
    if not groups.strict_reliable <= groups.actual_active:
        raise ValueError("Strict Reliable must be included in actual active training.")
    if not groups.fallback <= groups.actual_active:
        raise ValueError("Fallback must be included in actual active training.")


def stratified_indices(indices: Iterable[int], labels: dict[int, str], size: int, seed: int) -> tuple[int, ...]:
    """Deterministic label-spread sample without duplicating items to fill a batch."""
    candidate = sorted(set(int(index) for index in indices))
    if not candidate:
        return ()
    target = min(int(size), len(candidate))
    if target < 1:
        return ()
    per_label: dict[str, list[int]] = defaultdict(list)
    for index in candidate:
        per_label[labels[index]].append(index)
    rng = random.Random(seed)
    label_order = sorted(per_label)
    rng.shuffle(label_order)
    for values in per_label.values():
        rng.shuffle(values)
    selected: list[int] = []
    cursor = {label: 0 for label in label_order}
    while len(selected) < target:
        progressed = False
        for label in label_order:
            position = cursor[label]
            values = per_label[label]
            if position >= len(values):
                continue
            selected.append(values[position])
            cursor[label] = position + 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            raise RuntimeError("Stratified sampler exhausted before reaching its requested batch size.")
    return tuple(selected)


def build_diagnostic_batches(
    groups: SelectionGroups,
    *,
    num_batches: int,
    batch_size: int,
    diagnostic_seed: int,
) -> list[DiagnosticBatch]:
    validate_groups_for_diagnostic(groups)
    if num_batches < 1 or batch_size < 1:
        raise ValueError("num_batches and batch_size must both be positive.")
    batches: list[DiagnosticBatch] = []
    for batch_id in range(num_batches):
        batch_seed = int(diagnostic_seed + 1009 * batch_id)
        reliable = stratified_indices(groups.actual_active, groups.labels, batch_size, batch_seed + 1)
        ambiguous = stratified_indices(groups.ambiguous_for_consistency, groups.labels, batch_size, batch_seed + 2)
        if not reliable or not ambiguous:
            raise ValueError(f"Diagnostic batch {batch_id} unexpectedly lacks Reliable or Ambiguous samples.")
        if set(reliable) & set(ambiguous):
            raise ValueError("Reliable and Ambiguous diagnostics overlap.")
        batches.append(DiagnosticBatch(batch_id, reliable, ambiguous, batch_seed))
    return batches


def split_parameter_groups(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, list[tuple[str, Any]]]:
    groups = {"lora": [], "classifier_head": [], "all_trainable": []}
    unexpected: list[str] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if "lora_a" in name or "lora_b" in name:
            groups["lora"].append((name, parameter))
        elif name.startswith("head."):
            groups["classifier_head"].append((name, parameter))
        else:
            unexpected.append(name)
        groups["all_trainable"].append((name, parameter))
    if unexpected:
        raise RuntimeError(f"Unexpected trainable non-LoRA/non-head parameters: {unexpected}")
    if not groups["lora"] or not groups["classifier_head"]:
        raise RuntimeError("Expected both LoRA and classifier-head trainable parameters.")
    return groups


def gradients_to_cpu(torch: Any, grads: Sequence[Any | None], parameters: Sequence[Any]) -> list[Any]:
    """Detach gradients as float32 CPU tensors; map None to exact parameter-shaped zero."""
    if len(grads) != len(parameters):
        raise ValueError("Gradient and parameter sequences have different lengths.")
    result: list[Any] = []
    for grad, parameter in zip(grads, parameters):
        if grad is None:
            result.append(torch.zeros_like(parameter, dtype=torch.float32, device="cpu"))
        else:
            if not bool(torch.isfinite(grad).all().item()):
                raise FloatingPointError("Encountered NaN/Inf gradient.")
            result.append(grad.detach().to(device="cpu", dtype=torch.float32).clone())
    return result


def autograd_snapshot(torch: Any, loss: Any, parameters: Sequence[Any]) -> list[Any]:
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Diagnostic loss is NaN or Inf.")
    grads = torch.autograd.grad(loss, parameters, retain_graph=False, create_graph=False, allow_unused=True)
    return gradients_to_cpu(torch, grads, parameters)


def add_gradient_snapshots(torch: Any, accumulator: list[Any], addition: Sequence[Any]) -> list[Any]:
    if len(accumulator) != len(addition):
        raise ValueError("Cannot combine differently shaped gradient snapshots.")
    return [left + right for left, right in zip(accumulator, addition)]


def zeros_for_parameters(torch: Any, parameters: Sequence[Any]) -> list[Any]:
    return [torch.zeros_like(parameter, dtype=torch.float32, device="cpu") for parameter in parameters]


def flatten_group(torch: Any, snapshot: Sequence[Any], all_parameters: Sequence[Any], group_parameters: Sequence[Any]) -> Any:
    by_id = {id(parameter): gradient for parameter, gradient in zip(all_parameters, snapshot)}
    missing = [parameter for parameter in group_parameters if id(parameter) not in by_id]
    if missing:
        raise ValueError("Parameter group is not a subset of the gradient snapshot.")
    return torch.cat([by_id[id(parameter)].reshape(-1) for parameter in group_parameters])


def gradient_metrics(torch: Any, ce_vector: Any, kl_vector: Any, epsilon: float = EPSILON) -> dict[str, float | None]:
    ce_norm = float(torch.linalg.vector_norm(ce_vector).item())
    kl_norm = float(torch.linalg.vector_norm(kl_vector).item())
    total_vector = ce_vector + kl_vector
    combined_norm = float(torch.linalg.vector_norm(total_vector).item())
    dot = float(torch.dot(ce_vector, kl_vector).item())
    cosine: float | None = None
    if ce_norm > epsilon and kl_norm > epsilon:
        cosine = dot / (ce_norm * kl_norm)
    return {
        "ce_grad_norm": ce_norm,
        "kl_grad_norm": kl_norm,
        "grad_norm_ratio": kl_norm / (ce_norm + epsilon),
        "grad_cosine": cosine,
        "combined_grad_norm": combined_norm,
        "grad_dot_product": dot,
    }


def microbatch_scale(micro_batch_size: int, effective_batch_size: int) -> float:
    if micro_batch_size < 1 or effective_batch_size < 1 or micro_batch_size > effective_batch_size:
        raise ValueError("Invalid micro-batch/effective-batch size pair.")
    return float(micro_batch_size) / float(effective_batch_size)


def snapshot_model_state(model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()},
        {name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()},
    )


def assert_model_state_unchanged(model: Any, snapshot: tuple[dict[str, Any], dict[str, Any]]) -> None:
    before_parameters, before_buffers = snapshot
    for name, parameter in model.named_parameters():
        if not bool(np.array_equal(parameter.detach().cpu().numpy(), before_parameters[name].numpy())):
            raise RuntimeError(f"Diagnostic changed model parameter: {name}")
    for name, buffer in model.named_buffers():
        if not bool(np.array_equal(buffer.detach().cpu().numpy(), before_buffers[name].numpy())):
            raise RuntimeError(f"Diagnostic changed model buffer: {name}")


def set_all_seeds(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stack_reliable_samples(dataset: Any, index_to_position: dict[int, int], indices: Sequence[int], torch: Any) -> tuple[Any, Any]:
    materialized = [dataset[index_to_position[index]] for index in indices]
    images = torch.stack([item[0] for item in materialized])
    labels = torch.tensor([item[1] for item in materialized], dtype=torch.long)
    returned_indices = [int(item[2]) for item in materialized]
    if returned_indices != list(indices):
        raise ValueError("Reliable Dataset did not preserve original sample indices.")
    return images, labels


def stack_ambiguous_samples(dataset: Any, index_to_position: dict[int, int], indices: Sequence[int], torch: Any) -> tuple[Any, Any]:
    materialized = [dataset[index_to_position[index]] for index in indices]
    weak_images = torch.stack([item[0] for item in materialized])
    strong_images = torch.stack([item[1] for item in materialized])
    returned_indices = [int(item[2]) for item in materialized]
    if returned_indices != list(indices):
        raise ValueError("Weak/strong Dataset did not preserve original sample indices.")
    return weak_images, strong_images


def reconstruct_index_aligned_inputs(groups: SelectionGroups) -> tuple[list[str], np.ndarray]:
    largest = max(groups.pool)
    paths = [""] * (largest + 1)
    labels = np.full(largest + 1, "", dtype=str)
    for index in groups.pool:
        paths[index] = groups.paths[index]
        labels[index] = groups.labels[index]
    if any(not paths[index] or not labels[index] for index in groups.pool):
        raise ValueError("Selection rows do not retain complete paths/observed labels.")
    return paths, labels


def load_audited_model(torch: Any, spec: CheckpointSpec, cfg: dict[str, Any], device: str) -> tuple[Any, dict[str, Any]]:
    from gcdd.lora_dynamic import assert_lora_backbone_invariants
    from gcdd.lora_training import DINOv2LoRAClassifier, freeze_all, inject_lora, parse_target_modules

    payload = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Checkpoint has no trainable state_dict: {spec.checkpoint_path}")
    checkpoint_state = payload["state_dict"]
    classes = payload.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"Checkpoint has no valid classes list: {spec.checkpoint_path}")
    local_cfg = json.loads(json.dumps(cfg))
    local_cfg.setdefault("feature", {})["device"] = device
    model = DINOv2LoRAClassifier.make(torch, local_cfg, len(classes)).to(device)
    freeze_all(model.backbone)
    lora_cfg = local_cfg["lora"]
    inject_lora(
        torch,
        model.backbone,
        target_modules=parse_target_modules(str(lora_cfg.get("target_modules", "qkv"))),
        rank=int(lora_cfg.get("rank", 8)),
        alpha=float(lora_cfg.get("alpha", 16.0)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
    )
    model.to(device)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    assert_lora_backbone_invariants(model)
    expected = {name: value for name, value in model.state_dict().items() if "lora_" in name or name.startswith("head.")}
    missing = sorted(set(expected) - set(checkpoint_state))
    unexpected = sorted(set(checkpoint_state) - set(expected))
    mismatched = sorted(
        name for name in set(expected) & set(checkpoint_state) if tuple(expected[name].shape) != tuple(checkpoint_state[name].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Checkpoint/trainable-state mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    unexpected_after_load = [name for name in incompatible.unexpected_keys if name in expected]
    missing_trainable_after_load = [name for name in incompatible.missing_keys if name in expected]
    if unexpected_after_load or missing_trainable_after_load:
        raise RuntimeError(
            "Checkpoint did not load every expected trainable tensor: "
            f"missing={missing_trainable_after_load}, unexpected={unexpected_after_load}"
        )
    return model, payload


def compute_weighted_kl_snapshot(
    torch: Any,
    model: Any,
    weak_images: Any,
    strong_images: Any,
    parameters: Sequence[Any],
    *,
    consistency_weight: float,
    micro_batch_size: int,
    amp_enabled: bool,
) -> tuple[list[Any], float, float, list[Any], list[Any]]:
    """Return weighted effective-batch KL gradient, using bounded physical batches."""
    from gcdd.lora_dynamic import forward_ambiguous_consistency

    total = int(weak_images.shape[0])
    if total != int(strong_images.shape[0]) or total < 1:
        raise ValueError("Weak and strong effective batches must be non-empty and equally sized.")
    accumulator = zeros_for_parameters(torch, parameters)
    loss_sum = 0.0
    weighted_loss_sum = 0.0
    all_weak_prob: list[Any] = []
    all_strong_logits: list[Any] = []
    for start in range(0, total, micro_batch_size):
        end = min(total, start + micro_batch_size)
        count = end - start
        kl_loss, weak_prob, strong_logits = forward_ambiguous_consistency(
            torch,
            model,
            weak_images[start:end],
            strong_images[start:end],
            amp_enabled=amp_enabled,
        )
        scaled_loss = float(consistency_weight) * microbatch_scale(count, total) * kl_loss
        addition = autograd_snapshot(torch, scaled_loss, parameters)
        accumulator = add_gradient_snapshots(torch, accumulator, addition)
        loss_sum += float(kl_loss.detach().cpu().item()) * microbatch_scale(count, total)
        weighted_loss_sum += float(scaled_loss.detach().cpu().item())
        all_weak_prob.append(weak_prob.detach().cpu())
        all_strong_logits.append(strong_logits.detach().cpu())
    return accumulator, loss_sum, weighted_loss_sum, all_weak_prob, all_strong_logits


def compute_weighted_ce_snapshot(
    torch: Any,
    model: Any,
    images: Any,
    labels: Any,
    criterion: Any,
    parameters: Sequence[Any],
    *,
    micro_batch_size: int,
    amp_enabled: bool,
) -> tuple[list[Any], float]:
    """CE gradient for one effective Reliable batch with no retained prior graph.

    CrossEntropyLoss defaults to a mean reduction.  Scaling each physical
    micro-batch by ``n_micro / N_effective`` therefore preserves the gradient
    of the mean CE over the original Reliable effective batch.
    """
    total = int(images.shape[0])
    if total != int(labels.shape[0]) or total < 1:
        raise ValueError("Reliable images and labels must have the same non-zero effective batch size.")
    if micro_batch_size < 1 or micro_batch_size > total:
        raise ValueError(f"Reliable micro-batch size must be within [1, {total}], got {micro_batch_size}.")
    accumulator = zeros_for_parameters(torch, parameters)
    loss_sum = 0.0
    for start in range(0, total, micro_batch_size):
        end = min(total, start + micro_batch_size)
        count = end - start
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images[start:end])
            ce_loss = criterion(logits, labels[start:end])
        scaled_loss = microbatch_scale(count, total) * ce_loss
        accumulator = add_gradient_snapshots(
            torch,
            accumulator,
            autograd_snapshot(torch, scaled_loss, parameters),
        )
        loss_sum += float(ce_loss.detach().cpu().item()) * microbatch_scale(count, total)
        del logits, ce_loss, scaled_loss
    return accumulator, loss_sum


def restore_buffers(torch: Any, model: Any, buffers: dict[str, Any]) -> None:
    """Restore persistent buffers after a failed full-batch forward attempt."""
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name not in buffers:
                raise RuntimeError(f"Model buffer inventory changed after failed diagnostic forward: {name}")
            buffer.copy_(buffers[name].to(device=buffer.device, dtype=buffer.dtype))


def compute_reliable_ce_snapshot(
    torch: Any,
    model: Any,
    images: Any,
    labels: Any,
    criterion: Any,
    parameters: Sequence[Any],
    *,
    requested_mode: str,
    micro_batch_size: int,
    amp_enabled: bool,
    retry_seed: int,
) -> tuple[list[Any], float, str]:
    """Compute Reliable CE gradients while preserving the effective batch size.

    In ``auto`` mode, a full effective batch is attempted first under AMP to
    match production training.  CUDA OOM alone triggers a state-safe retry in
    weighted micro-batches; no optimizer state or model tensor is updated.
    """
    total = int(images.shape[0])
    if requested_mode not in {"auto", "full", "micro"}:
        raise ValueError(f"Unsupported Reliable gradient mode: {requested_mode}")
    full_label = "full_amp" if amp_enabled else "full_fp32"
    micro_label = "microbatch_amp" if amp_enabled else "microbatch_fp32"
    if requested_mode in {"auto", "full"}:
        buffer_snapshot = {name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()}
        try:
            snapshot, loss = compute_weighted_ce_snapshot(
                torch,
                model,
                images,
                labels,
                criterion,
                parameters,
                micro_batch_size=total,
                amp_enabled=amp_enabled,
            )
            return snapshot, loss, full_label
        except torch.OutOfMemoryError:
            if requested_mode == "full":
                raise
            restore_buffers(torch, model, buffer_snapshot)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            set_all_seeds(torch, retry_seed)
            print(
                "[gradient-audit] Reliable full effective batch OOM under AMP; "
                f"retrying with weighted Reliable micro-batches of {micro_batch_size}."
            )
    snapshot, loss = compute_weighted_ce_snapshot(
        torch,
        model,
        images,
        labels,
        criterion,
        parameters,
        micro_batch_size=micro_batch_size,
        amp_enabled=amp_enabled,
    )
    return snapshot, loss, micro_label


def gradient_rows_for_batch(
    torch: Any,
    spec: CheckpointSpec,
    diagnostic_batch: DiagnosticBatch,
    model: Any,
    cfg: dict[str, Any],
    groups: SelectionGroups,
    path_maps: list[tuple[str, str]],
    device: str,
    output_sampling_rows: list[dict[str, Any]],
    ambiguous_micro_batch_size: int,
    reliable_gradient_mode: str,
    reliable_micro_batch_size: int,
) -> tuple[list[dict[str, Any]], str]:
    from torchvision import transforms

    from gcdd.lora_training import ImageSplitDataset, WeakStrongImageSplitDataset, build_criterion, build_transforms, build_weak_strong_transforms

    paths, labels = reconstruct_index_aligned_inputs(groups)
    checkpoint_payload = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint_payload["classes"]
    label_to_id = {str(label): position for position, label in enumerate(classes)}
    selected_labels = [groups.labels[index] for index in (*diagnostic_batch.reliable_indices, *diagnostic_batch.ambiguous_indices)]
    unknown = sorted(set(selected_labels) - set(label_to_id))
    if unknown:
        raise ValueError(f"Historical observed labels do not occur in checkpoint classes: {unknown[:5]}")
    input_size = int(cfg["feature"]["input_size"])
    train_transform, _ = build_transforms(transforms, input_size)
    weak_transform, strong_transform = build_weak_strong_transforms(transforms, input_size)
    reliable_array = np.asarray(diagnostic_batch.reliable_indices, dtype=np.int64)
    ambiguous_array = np.asarray(diagnostic_batch.ambiguous_indices, dtype=np.int64)
    reliable_dataset = ImageSplitDataset(paths, labels, reliable_array, label_to_id, train_transform, path_maps)
    ambiguous_dataset = WeakStrongImageSplitDataset(paths, ambiguous_array, weak_transform, strong_transform, path_maps)
    reliable_pos = {index: position for position, index in enumerate(reliable_array.tolist())}
    ambiguous_pos = {index: position for position, index in enumerate(ambiguous_array.tolist())}
    for group_name, indices in (("reliable", diagnostic_batch.reliable_indices), ("ambiguous", diagnostic_batch.ambiguous_indices)):
        for sample_order, index in enumerate(indices):
            output_sampling_rows.append(
                {
                    "dataset": spec.dataset,
                    "method": spec.method,
                    "seed": spec.seed,
                    "checkpoint_epoch": spec.checkpoint_epoch,
                    "selection_update_epoch": spec.selection_update_epoch,
                    "diagnostic_batch_id": diagnostic_batch.batch_id,
                    "diagnostic_seed": diagnostic_batch.seed,
                    "group": group_name,
                    "sample_order": sample_order,
                    "original_index": index,
                    "observed_label": groups.labels[index],
                    "path": groups.paths[index],
                }
            )
    set_all_seeds(torch, diagnostic_batch.seed)
    reliable_images, reliable_labels = stack_reliable_samples(reliable_dataset, reliable_pos, diagnostic_batch.reliable_indices, torch)
    weak_images, strong_images = stack_ambiguous_samples(ambiguous_dataset, ambiguous_pos, diagnostic_batch.ambiguous_indices, torch)
    reliable_images = reliable_images.to(device)
    reliable_labels = reliable_labels.to(device)
    weak_images = weak_images.to(device)
    strong_images = strong_images.to(device)
    parameter_groups = split_parameter_groups(model.named_parameters())
    all_parameters = [parameter for _, parameter in parameter_groups["all_trainable"]]
    model.train(True)
    reliable_forward_seed = diagnostic_batch.seed + 10_000
    set_all_seeds(torch, reliable_forward_seed)
    criterion = build_criterion(torch, cfg)
    amp_enabled = bool(cfg.get("lora_train", {}).get("amp", True)) and device.startswith("cuda")
    ce_snapshot, ce_loss_value, reliable_mode_used = compute_reliable_ce_snapshot(
        torch,
        model,
        reliable_images,
        reliable_labels,
        criterion,
        all_parameters,
        requested_mode=reliable_gradient_mode,
        micro_batch_size=reliable_micro_batch_size,
        amp_enabled=amp_enabled,
        retry_seed=reliable_forward_seed,
    )
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    consistency_weight = float(cfg.get("pgdf", {}).get("consistency_weight", 0.5))
    micro_batch_size = int(ambiguous_micro_batch_size)
    kl_snapshot, kl_loss, weighted_kl_loss, weak_prob, strong_logits = compute_weighted_kl_snapshot(
        torch,
        model,
        weak_images,
        strong_images,
        all_parameters,
        consistency_weight=consistency_weight,
        micro_batch_size=micro_batch_size,
        amp_enabled=amp_enabled,
    )
    if not all(bool(torch.isfinite(value).all().item()) for value in (*weak_prob, *strong_logits)):
        raise FloatingPointError("Weak/strong diagnostic outputs are not finite.")
    rows: list[dict[str, Any]] = []
    for group_name, named_parameters in parameter_groups.items():
        parameters = [parameter for _, parameter in named_parameters]
        metrics = gradient_metrics(
            torch,
            flatten_group(torch, ce_snapshot, all_parameters, parameters),
            flatten_group(torch, kl_snapshot, all_parameters, parameters),
        )
        rows.append(
            {
                "dataset": spec.dataset,
                "method": spec.method,
                "seed": spec.seed,
                "checkpoint_epoch": spec.checkpoint_epoch,
                "selection_update_epoch": spec.selection_update_epoch,
                "diagnostic_batch_id": diagnostic_batch.batch_id,
                "parameter_group": group_name,
                "reliable_batch_size": len(diagnostic_batch.reliable_indices),
                "reliable_gradient_mode": reliable_mode_used,
                "reliable_micro_batch_size": reliable_micro_batch_size if reliable_mode_used.startswith("microbatch") else len(diagnostic_batch.reliable_indices),
                "ambiguous_batch_size": len(diagnostic_batch.ambiguous_indices),
                "ambiguous_micro_batch_size": micro_batch_size,
                "consistency_weight": consistency_weight,
                "amp_enabled": "yes" if amp_enabled else "no",
                "ce_loss": ce_loss_value,
                "kl_loss": kl_loss,
                "weighted_kl_loss": weighted_kl_loss,
                **metrics,
                "finite_gradients": "yes",
                "diagnostic_seed": diagnostic_batch.seed,
                "gradient_dtype": "float32_cpu",
            }
        )
    del reliable_images, reliable_labels, weak_images, strong_images, weak_prob, strong_logits
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, reliable_mode_used


def finite_or_na(value: float | None) -> str:
    return "NA" if value is None or not math.isfinite(value) else f"{value:.6g}"


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method"], int(row["checkpoint_epoch"]), row["parameter_group"])].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, method, epoch, parameter_group), values in sorted(groups.items()):
        numerical = ("ce_grad_norm", "kl_grad_norm", "grad_norm_ratio", "combined_grad_norm", "grad_dot_product")
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "checkpoint_epoch": epoch,
            "parameter_group": parameter_group,
            "diagnostic_batches": len(values),
            "negative_cosine_batches": sum(float(item["grad_cosine"]) < 0 for item in values if item["grad_cosine"] not in (None, "")),
        }
        for field in numerical:
            data = [float(item[field]) for item in values]
            row[f"{field}_mean"] = statistics.mean(data)
            row[f"{field}_std"] = sample_std(data)
        cosines = [float(item["grad_cosine"]) for item in values if item["grad_cosine"] not in (None, "")]
        row["grad_cosine_mean"] = statistics.mean(cosines) if cosines else None
        row["grad_cosine_std"] = sample_std(cosines)
        row["grad_cosine_min"] = min(cosines) if cosines else None
        row["grad_cosine_max"] = max(cosines) if cosines else None
        summary.append(row)
    return summary


def write_inventory(output_dir: Path, specs: Sequence[CheckpointSpec], dry_run: bool) -> None:
    lines = [
        "# Checkpoint inventory",
        "",
        "All source checkpoints are read-only.  A checkpoint epoch is paired with the latest selection update strictly before it, because production code snapshots validation state before executing that epoch's update.",
        "",
        "| Dataset | Method | Checkpoint | Best epoch | Active selection update | Mode |",
        "|---|---|---|---:|---:|---|",
    ]
    for spec in specs:
        lines.append(
            f"| {spec.dataset} | {spec.method} | `{spec.checkpoint_path}` | {spec.checkpoint_epoch} | {spec.selection_update_epoch} | {'dry-run' if dry_run else 'gradient diagnostic'} |"
        )
    (output_dir / "checkpoint_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def image_access_preflight(specs: Sequence[CheckpointSpec], path_maps: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Check only a representative recorded training image for each required pool.

    This is deliberately a path-access check, not model inference.  It makes a
    missing server dataset explicit before an expensive model load is attempted.
    """
    from gcdd.lora_training import resolve_image_path

    rows: list[dict[str, Any]] = []
    for spec in specs:
        groups = history_groups(spec)
        for group_name, indices in (("reliable", groups.actual_active), ("ambiguous", groups.ambiguous_for_consistency)):
            index = min(indices)
            source_path = groups.paths[index]
            try:
                resolved = resolve_image_path(source_path, path_maps)
                status = "accessible"
                detail = str(resolved)
            except FileNotFoundError as exc:
                status = "missing"
                detail = str(exc)
            rows.append(
                {
                    "dataset": spec.dataset,
                    "method": spec.method,
                    "seed": spec.seed,
                    "checkpoint_epoch": spec.checkpoint_epoch,
                    "selection_update_epoch": spec.selection_update_epoch,
                    "group": group_name,
                    "representative_original_index": index,
                    "logged_path": source_path,
                    "status": status,
                    "detail": detail,
                }
            )
    return rows


def write_integrity_report(
    output_dir: Path,
    *,
    device: str,
    ambiguous_micro_batch_size: int,
    reliable_gradient_mode: str,
    reliable_micro_batch_size: int,
    completed: bool,
    peak_bytes: int | None,
) -> None:
    peak = "NA" if peak_bytes is None else f"{peak_bytes} bytes ({peak_bytes / 1024**3:.3f} GiB)"
    body = f"""# Gradient integrity check

- Device: `{device}`.
- Statistics use detached CPU `float32` gradient snapshots.
- KL uses production `forward_ambiguous_consistency` and production `compute_soft_consistency_loss`.
- Weak view is evaluated under `eval() + no_grad()` by that helper; strong view runs in restored train state.
- Each physical Ambiguous micro-batch is multiplied by `n_micro / N_effective`; therefore the accumulated gradient corresponds to one effective-batch mean KL.
- Reliable CE policy: `{reliable_gradient_mode}`.  In auto mode, a complete AMP Reliable batch is attempted first and only CUDA OOM triggers the weighted `{reliable_micro_batch_size}`-sample micro-batch fallback.  Actual per-batch policy is recorded in `gradient_per_batch.csv`.
- Ambiguous physical micro-batch size: {ambiguous_micro_batch_size}.
- No optimizer or scheduler is constructed by this script, and no model state is saved.
- Parameter and persistent-buffer equality is asserted after every checkpoint diagnostic.
- Peak CUDA allocation: {peak}.
- Completed GPU/CPU gradient pass: {'yes' if completed else 'no (dry-run)'}.
"""
    (output_dir / "gradient_integrity_check.md").write_text(body, encoding="utf-8")


def write_reports(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    rows = [row for row in summary if row["method"] == CONSISTENCY_VARIANT and row["parameter_group"] in {"lora", "classifier_head"}]
    lines = [
        "# CE/KL gradient comparison",
        "",
        "Each value is the mean across three fixed diagnostic batches at one validation-selected checkpoint. KL gradients already include the historical consistency weight 0.5. These are local, post-hoc gradient measurements, not historical step-by-step training gradients or causal evidence for Top-1 changes.",
        "",
        "| Dataset | Group | CE norm | Weighted KL norm | KL/CE | Cosine | Negative cosine batches |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {group} | {ce} | {kl} | {ratio} | {cosine} | {negative} |".format(
                dataset=row["dataset"],
                group=row["parameter_group"],
                ce=finite_or_na(row["ce_grad_norm_mean"]),
                kl=finite_or_na(row["kl_grad_norm_mean"]),
                ratio=finite_or_na(row["grad_norm_ratio_mean"]),
                cosine=finite_or_na(row["grad_cosine_mean"]),
                negative=row["negative_cosine_batches"],
            )
        )
    text = "\n".join(lines) + "\n"
    (output_dir / "gradient_comparison.md").write_text(text, encoding="utf-8")
    (output_dir / "gradient_audit_report.md").write_text(
        "# Checkpoint-level CE/KL Gradient Contribution and Conflict Audit\n\n"
        "## Scope\n\n"
        "This report uses only saved consistency-run checkpoints and noisy-training-pool samples. It does not update a model, perform optimizer/scheduler operations, or use validation/test images or clean/noisy identities for sampling.\n\n"
        "## Main result table\n\n" + text +
        "## Interpretation limits\n\n"
        "Norm ratios quantify local loss-gradient magnitude within a parameter group; they are not performance evidence. Cosine is a local directional relationship, not proof that consistency is helpful or harmful. Three diagnostic batches are descriptive rather than seed-level uncertainty.\n",
        encoding="utf-8",
    )


def run_checkpoint(
    torch: Any,
    spec: CheckpointSpec,
    path_maps: list[tuple[str, str]],
    device: str,
    num_batches: int,
    batch_size: int,
    diagnostic_seed: int,
    ambiguous_micro_batch_size: int,
    reliable_gradient_mode: str,
    reliable_micro_batch_size: int,
    sampling_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    cfg = read_yaml(spec.cfg_path)
    groups = history_groups(spec)
    batches = build_diagnostic_batches(
        groups,
        num_batches=num_batches,
        batch_size=batch_size,
        diagnostic_seed=diagnostic_seed,
    )
    model, payload = load_audited_model(torch, spec, cfg, device)
    if int(payload["best_epoch"]) != spec.checkpoint_epoch:
        raise RuntimeError("Checkpoint metadata changed between inventory and load.")
    state = snapshot_model_state(model)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, Any]] = []
    runtime_reliable_mode = reliable_gradient_mode
    for batch in batches:
        batch_rows, mode_used = gradient_rows_for_batch(
            torch,
            spec,
            batch,
            model,
            cfg,
            groups,
            path_maps,
            device,
            sampling_rows,
            ambiguous_micro_batch_size,
            runtime_reliable_mode,
            reliable_micro_batch_size,
        )
        rows.extend(batch_rows)
        if reliable_gradient_mode == "auto" and mode_used.startswith("microbatch"):
            runtime_reliable_mode = "micro"
        assert_model_state_unchanged(model, state)
    peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows, peak


def main() -> None:
    args = parse_args()
    if args.ambiguous_micro_batch_size < 1:
        raise ValueError("--ambiguous-micro-batch-size must be positive.")
    if args.reliable_micro_batch_size < 1:
        raise ValueError("--reliable-micro-batch-size must be positive.")
    if args.reliable_micro_batch_size > args.batch_size:
        raise ValueError("--reliable-micro-batch-size cannot exceed --batch-size.")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    path_maps = parse_path_maps(args.path_map)
    variants = [CONSISTENCY_VARIANT] + ([BASELINE_VARIANT] if args.include_baseline else [])
    specs = [find_checkpoint_spec(args.variants_root, variant, dataset, args.seed) for variant in variants for dataset in args.datasets]
    write_inventory(output_dir, specs, args.dry_run)
    image_access_rows = image_access_preflight(specs, path_maps)
    write_csv(output_dir / "image_access_preflight.csv", image_access_rows)
    image_access_missing = any(row["status"] != "accessible" for row in image_access_rows)
    sampling_rows: list[dict[str, Any]] = []
    if args.dry_run:
        for spec in specs:
            groups = history_groups(spec)
            for batch in build_diagnostic_batches(groups, num_batches=args.num_batches, batch_size=args.batch_size, diagnostic_seed=args.diagnostic_seed):
                for group_name, indices in (("reliable", batch.reliable_indices), ("ambiguous", batch.ambiguous_indices)):
                    for sample_order, index in enumerate(indices):
                        sampling_rows.append(
                            {
                                "dataset": spec.dataset, "method": spec.method, "seed": spec.seed,
                                "checkpoint_epoch": spec.checkpoint_epoch, "selection_update_epoch": spec.selection_update_epoch,
                                "diagnostic_batch_id": batch.batch_id, "diagnostic_seed": batch.seed,
                                "group": group_name, "sample_order": sample_order, "original_index": index,
                                "observed_label": groups.labels[index], "path": groups.paths[index],
                            }
                        )
        write_csv(output_dir / "diagnostic_sampling_manifest.csv", sampling_rows)
        write_integrity_report(
            output_dir,
            device=args.device,
            ambiguous_micro_batch_size=args.ambiguous_micro_batch_size,
            reliable_gradient_mode=args.reliable_gradient_mode,
            reliable_micro_batch_size=args.reliable_micro_batch_size,
            completed=False,
            peak_bytes=None,
        )
        if image_access_missing:
            (output_dir / "preflight_status.md").write_text(
                "# Gradient audit preflight status\n\n"
                "The checkpoint and historical-membership preflight completed, but one or more representative noisy-training-pool images are inaccessible. "
                "No model was loaded and no gradient was computed. Supply the original dataset through `--path-map OLD=NEW` or run this script on the original training server. "
                "See `image_access_preflight.csv` for exact paths.\n",
                encoding="utf-8",
            )
        print(f"Dry-run completed: {output_dir}")
        return
    if image_access_missing:
        raise FileNotFoundError(
            "Required noisy-training-pool images are inaccessible. Refusing to load a model or compute a partial gradient audit; "
            "see image_access_preflight.csv and provide --path-map OLD=NEW or run on the original training server."
        )
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    all_rows: list[dict[str, Any]] = []
    peak_bytes = 0
    for spec in specs:
        rows, peak = run_checkpoint(
            torch,
            spec,
            path_maps,
            args.device,
            args.num_batches,
            args.batch_size,
            args.diagnostic_seed,
            args.ambiguous_micro_batch_size,
            args.reliable_gradient_mode,
            args.reliable_micro_batch_size,
            sampling_rows,
        )
        all_rows.extend(rows)
        peak_bytes = max(peak_bytes, peak)
    write_csv(output_dir / "diagnostic_sampling_manifest.csv", sampling_rows)
    write_csv(output_dir / "gradient_per_batch.csv", all_rows)
    summary = summarize_rows(all_rows)
    write_csv(output_dir / "gradient_summary.csv", summary)
    write_reports(output_dir, summary)
    write_integrity_report(
        output_dir,
        device=args.device,
        ambiguous_micro_batch_size=args.ambiguous_micro_batch_size,
        reliable_gradient_mode=args.reliable_gradient_mode,
        reliable_micro_batch_size=args.reliable_micro_batch_size,
        completed=True,
        peak_bytes=peak_bytes,
    )
    print(f"Gradient audit completed: {output_dir}")


if __name__ == "__main__":
    main()
