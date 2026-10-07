from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import resolve_device
from .lora_training import (
    DINOv2LoRAClassifier,
    ImageSplitDataset,
    WeakStrongImageSplitDataset,
    build_scheduler,
    build_transforms,
    build_weak_strong_transforms,
    count_total_params,
    count_trainable_params,
    evaluate_lora,
    evaluate_state_lora,
    freeze_all,
    inject_lora,
    lora_parameters,
    parse_target_modules,
    safe_ratio,
    set_torch_seed,
    summarize_lora_logs,
    trainable_state_dict,
)
from .progress import log_stage, progress_iter


CONSISTENCY_BACKWARD_MODES = ("joint", "sequential", "microbatch")


@dataclass
class DynamicLossRunResult:
    logs: list[dict[str, Any]]
    summary: dict[str, Any]
    trainable_modules: list[str]
    trainable_params: int
    total_params: int
    selection_rows: list[dict[str, Any]]
    update_rows: list[dict[str, Any]]
    per_class_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class DynamicPrototypeSnapshot:
    """One deterministic, current-model prototype-gate snapshot.

    Feature tensors deliberately remain in memory only.  The compact statistics
    and hashes are written to the selection audit so a run can prove that the
    gate was rebuilt without persisting all representations.
    """

    scores: np.ndarray
    gate_mask: np.ndarray
    class_retained_counts: dict[str, int]
    feature_dim: int
    prototype_checksum: str
    gate_membership_hash: str
    prototype_mean: float
    prototype_std: float
    model_state_checksum: str
    # These are update-local C x D values, never persisted to a selection
    # artifact.  They let alternative geometry gates reuse the exact observed-
    # label prototype construction used by the original dynamic PGDF gate.
    prototype_labels: tuple[str, ...]
    prototypes: np.ndarray


@dataclass(frozen=True)
class NeighborMarginSnapshot:
    """Compact per-sample diagnostics for one Neighbor-Margin update.

    The temporary all-class similarity matrix is deliberately not retained.
    Every vector below is aligned to the original full training-array index;
    non-candidate positions are left empty/NaN and never become trainable.
    """

    margin: np.ndarray
    observed_similarity: np.ndarray
    competitor_similarity: np.ndarray
    competitor_class: np.ndarray
    candidate_mask: np.ndarray


@dataclass(frozen=True)
class DualEvidenceStrata:
    """Update-local Margin-Rank strata derived solely from the L/G gates.

    ``strict_reliable`` remains the mathematical L intersection G.  The
    original PGDF fallback may enlarge ``actual_active`` for supervised
    training, but those fallback samples are intentionally excluded from
    ``ambiguous_for_consistency`` so a sample never receives both objectives.
    """

    strict_reliable: np.ndarray
    ambiguous_l: np.ndarray
    ambiguous_g: np.ndarray
    ambiguous: np.ndarray
    suspicious: np.ndarray
    actual_active: np.ndarray
    fallback: np.ndarray
    ambiguous_for_consistency: np.ndarray


def train_dynamic_loss_lora(
    train_paths: list[str],
    train_labels: np.ndarray,
    eval_paths: list[str],
    eval_labels: np.ndarray,
    candidate_mask: np.ndarray,
    cfg: dict[str, Any],
    method: str,
    seed: int,
    retention_ratio: float,
    warmup_epochs: int,
    update_interval: int,
    path_maps: list[tuple[str, str]] | None = None,
    centroid_mask: np.ndarray | None = None,
    proto_scores: np.ndarray | None = None,
    proto_keep_ratio: float | None = None,
    auto_proto_keep: dict[str, float] | None = None,
    checkpoint_path: Path | None = None,
    test_paths: list[str] | None = None,
    test_labels: np.ndarray | None = None,
    final_checkpoint_path: Path | None = None,
    last5_checkpoint_dir: Path | None = None,
    checkpoint_protocol: str = "legacy_test_selected",
    posthoc_oracle_test: bool = False,
    class_budget_schedule: dict[int, dict[str, int]] | None = None,
    scheduler_retention_ratio: float | None = None,
    official_test_selected_only: bool = False,
    selection_strategy: str = "auto",
    prototype_mode: str = "fixed",
    geometry_mode: str = "prototype_similarity",
    neighbor_margin_use_fallback: bool = False,
    neighbor_margin_positive_only: bool = True,
    ambiguous_consistency: bool = False,
    consistency_weight: float = 0.5,
    consistency_backward_mode: str = "joint",
    ambiguous_micro_batch_size: int = 20,
) -> DynamicLossRunResult:
    """Train DINOv2-LoRA with periodically updated class-wise small-loss selection.

    With an explicit test split, ``eval_paths`` is reserved for validation-only
    checkpoint selection and test metrics are evaluated after fitting.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Preserve the original public call pattern: callers that supplied static
    # prototype scores previously received PGDF intersection selection without
    # an extra strategy argument.
    if selection_strategy == "auto":
        selection_strategy = (
            "loss_and_proto"
            if proto_scores is not None or auto_proto_keep is not None or prototype_mode == "dynamic_lora"
            else "loss_only"
        )
    validate_dynamic_args(
        retention_ratio,
        warmup_epochs,
        update_interval,
        proto_keep_ratio,
        auto_proto_keep,
        selection_strategy=selection_strategy,
        prototype_mode=prototype_mode,
        geometry_mode=geometry_mode,
        neighbor_margin_use_fallback=neighbor_margin_use_fallback,
        neighbor_margin_positive_only=neighbor_margin_positive_only,
        ambiguous_consistency=ambiguous_consistency,
        consistency_weight=consistency_weight,
        consistency_backward_mode=consistency_backward_mode,
        ambiguous_micro_batch_size=ambiguous_micro_batch_size,
    )
    if (test_paths is None) != (test_labels is None):
        raise ValueError("test_paths and test_labels must be provided together.")
    if official_test_selected_only and posthoc_oracle_test:
        raise ValueError("official_test_selected_only cannot be combined with posthoc_oracle_test.")
    path_maps = path_maps or []
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.shape != (len(train_labels),):
        raise ValueError(f"candidate_mask must have shape ({len(train_labels)},), got {candidate_mask.shape}.")
    if not np.any(candidate_mask):
        raise ValueError(f"{method} has no candidate training images.")
    if centroid_mask is not None:
        centroid_mask = np.asarray(centroid_mask, dtype=bool)
        if centroid_mask.shape != candidate_mask.shape:
            raise ValueError("centroid_mask must match candidate_mask shape.")
    if proto_scores is not None:
        proto_scores = np.asarray(proto_scores, dtype=np.float32)
        if proto_scores.shape != candidate_mask.shape:
            raise ValueError("proto_scores must match candidate_mask shape.")
        if np.any(np.isnan(proto_scores[candidate_mask])):
            raise ValueError("proto_scores contains NaN values for candidate samples.")
    if auto_proto_keep is not None and centroid_mask is None:
        raise ValueError("auto_proto_keep requires centroid_mask to compute dynamic/prototype overlap.")
    if class_budget_schedule is not None and any(
        value is not None for value in (centroid_mask, proto_scores, proto_keep_ratio, auto_proto_keep)
    ):
        raise ValueError("class_budget_schedule cannot be combined with prototype, centroid, or graph selection inputs.")
    if selection_strategy == "proto_only" and class_budget_schedule is not None:
        raise ValueError("proto_only selection cannot use a dynamic small-loss class budget.")
    if prototype_mode == "dynamic_lora":
        if proto_scores is not None:
            raise ValueError("dynamic_lora prototypes must not receive frozen prototype scores.")
        if auto_proto_keep is not None:
            raise ValueError("dynamic_lora prototypes require a pre-declared fixed proto_keep_ratio.")
        if proto_keep_ratio is None:
            raise ValueError("dynamic_lora prototypes require proto_keep_ratio.")
    elif prototype_mode == "fixed" and selection_strategy in {"proto_only", "loss_and_proto"} and proto_scores is None:
        raise ValueError("fixed prototype selection requires proto_scores.")
    if geometry_mode == "neighbor_margin" and not (
        selection_strategy == "loss_and_proto" and prototype_mode == "dynamic_lora"
    ):
        raise ValueError(
            "neighbor_margin geometry is defined only for dynamic_lora PGDF "
            "loss_and_proto selection."
        )
    if ambiguous_consistency and not (
        geometry_mode == "neighbor_margin" and not neighbor_margin_positive_only
    ):
        raise ValueError(
            "ambiguous_consistency is supported only for Margin-Rank "
            "(geometry_mode='neighbor_margin' with neighbor_margin_positive_only=False)."
        )

    lora_cfg = cfg["lora"]
    train_cfg = cfg["lora_train"]
    feature_cfg = cfg["feature"]
    dataset_name = str(cfg.get("dataset", {}).get("name", ""))
    device = resolve_device(torch, feature_cfg.get("device", "auto"))
    set_torch_seed(torch, seed)

    classes = sorted(set(train_labels.tolist()))
    label_to_id = {label: i for i, label in enumerate(classes)}
    eval_known = np.array([label in label_to_id for label in eval_labels], dtype=bool)
    eval_idx = np.where(eval_known)[0]
    if len(eval_idx) == 0:
        raise ValueError("Eval split has no labels that appear in the train split.")

    input_size = int(feature_cfg["input_size"])
    train_transform, eval_transform = build_transforms(transforms, input_size)
    weak_transform = None
    strong_transform = None
    if ambiguous_consistency:
        # Construct these only for the opt-in branch.  The default Margin-Rank
        # path therefore keeps its original transform/randomness lifecycle.
        weak_transform, strong_transform = build_weak_strong_transforms(transforms, input_size)
    eval_dataset = ImageSplitDataset(eval_paths, eval_labels, eval_idx, label_to_id, eval_transform, path_maps)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    test_loader = None
    test_idx = np.array([], dtype=np.int64)
    if test_paths is not None and test_labels is not None:
        test_known = np.array([label in label_to_id for label in test_labels], dtype=bool)
        test_idx = np.where(test_known)[0]
        if len(test_idx) == 0:
            raise ValueError("Test split has no labels that appear in the train split.")
        test_dataset = ImageSplitDataset(test_paths, test_labels, test_idx, label_to_id, eval_transform, path_maps)
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            drop_last=False,
        )

    candidate_idx = np.where(candidate_mask)[0]
    loss_dataset = ImageSplitDataset(train_paths, train_labels, candidate_idx, label_to_id, eval_transform, path_maps)
    loss_loader = DataLoader(
        loss_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=False,
    )

    model = DINOv2LoRAClassifier.make(torch, cfg, len(classes)).to(device)
    freeze_all(model.backbone)
    trainable_modules = inject_lora(
        torch,
        model.backbone,
        target_modules=parse_target_modules(str(lora_cfg.get("target_modules", "qkv"))),
        rank=int(lora_cfg.get("rank", 8)),
        alpha=float(lora_cfg.get("alpha", 16.0)),
        dropout=float(lora_cfg.get("dropout", 0.05)),
    )
    model.to(device)
    for param in model.head.parameters():
        param.requires_grad_(True)
    assert_lora_backbone_invariants(model)

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters(model), "lr": float(train_cfg["lora_lr"])},
            {"params": model.head.parameters(), "lr": float(train_cfg["head_lr"])},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )

    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    expected_budget_epochs = selection_update_epochs(epochs, warmup_epochs, update_interval)
    if class_budget_schedule is not None:
        validate_class_budget_schedule(
            class_budget_schedule,
            train_labels,
            candidate_mask,
            expected_budget_epochs,
        )
    if scheduler_retention_ratio is not None:
        effective_scheduler_ratio = float(scheduler_retention_ratio)
    elif selection_strategy == "proto_only":
        # The dynamic-prototype-only protocol has the same full-pool warm-up
        # and post-update class-wise p gate used by the requested experiment.
        effective_scheduler_ratio = float(proto_keep_ratio)
    else:
        effective_scheduler_ratio = estimate_selection_retention_ratio(
            retention_ratio,
            proto_keep_ratio,
            auto_proto_keep,
        )
    if not 0.0 < effective_scheduler_ratio <= 1.0:
        raise ValueError("scheduler_retention_ratio must satisfy 0 < ratio <= 1.")
    total_steps = estimate_dynamic_total_steps(
        candidate_count=int(candidate_mask.sum()),
        batch_size=batch_size,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        retention_ratio=effective_scheduler_ratio,
    )
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
    scheduler = build_scheduler(torch, optimizer, total_steps, warmup_steps, str(train_cfg.get("scheduler", "cosine")))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.startswith("cuda"))
    criterion = torch.nn.CrossEntropyLoss()
    logs: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    last5_states: list[tuple[int, dict[str, Any]]] = []
    oracle_states: list[tuple[int, dict[str, Any]]] = []
    selected_proto_keep_ratio = proto_keep_ratio
    auto_proto_jaccard: float | None = None

    selected_mask = candidate_mask.copy()
    ambiguous_training_mask = np.zeros(len(train_labels), dtype=bool)
    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)
    initial_trainable_checksum = trainable_model_checksum(model)
    selection_split_name = "validation" if test_loader is not None else "eval"
    log_stage(
        f"[dynamic-loss] {method} seed={seed}: candidates={int(candidate_mask.sum())}, "
        f"{selection_split_name}_images={len(eval_idx)}, retention_ratio="
        f"{'not_used' if selection_strategy == 'proto_only' else f'{retention_ratio:.3f}'}, "
        f"proto_keep_ratio={format_optional_ratio(proto_keep_ratio)}, "
        f"selection_strategy={selection_strategy}, prototype_mode={prototype_mode}, geometry_mode={geometry_mode}, "
        f"auto_proto_keep={'yes' if auto_proto_keep is not None else 'no'}, trainable_params={trainable_params}"
    )

    for epoch in range(1, epochs + 1):
        epoch_train_idx = np.where(selected_mask)[0]
        if len(epoch_train_idx) == 0:
            raise RuntimeError(
                "Dynamic selection produced an empty active subset; training cannot continue. "
                "This does not alter fallback semantics."
            )
        train_dataset = ImageSplitDataset(train_paths, train_labels, epoch_train_idx, label_to_id, train_transform, path_maps)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            drop_last=False,
        )

        consistency_reason = "disabled"
        ambiguous_loader = None
        ambiguous_train_idx = np.asarray([], dtype=np.int64)
        if ambiguous_consistency:
            if epoch <= warmup_epochs:
                consistency_reason = "warmup"
            elif consistency_weight == 0.0:
                consistency_reason = "weight_zero"
            else:
                ambiguous_train_idx = np.where(ambiguous_training_mask)[0]
                if len(ambiguous_train_idx) == 0:
                    consistency_reason = "empty_ambiguous"
                else:
                    if weak_transform is None or strong_transform is None:
                        raise RuntimeError("Ambiguous consistency transforms were not initialized.")
                    ambiguous_dataset = WeakStrongImageSplitDataset(
                        train_paths,
                        ambiguous_train_idx,
                        weak_transform,
                        strong_transform,
                        path_maps,
                    )
                    ambiguous_loader = DataLoader(
                        ambiguous_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=int(train_cfg.get("num_workers", 4)),
                        pin_memory=bool(train_cfg.get("pin_memory", True)),
                        drop_last=False,
                    )
                    consistency_reason = "active"

        model.train()
        loss_sum = 0.0
        supervised_loss_sum = 0.0
        consistency_loss_sum = 0.0
        weak_confidence_sum = 0.0
        weak_entropy_sum = 0.0
        weak_strong_agreement_sum = 0.0
        seen = 0
        ambiguous_sample_exposures = 0
        ambiguous_batch_count = 0
        consistency_micro_batch_count = 0
        optimizer_steps = 0
        scheduler_steps = 0
        ambiguous_iter = iter(ambiguous_loader) if ambiguous_loader is not None else None
        progress = progress_iter(train_loader, total=len(train_loader), desc=f"Dynamic LoRA {method} seed={seed} epoch {epoch}/{epochs}")
        for images, labels, _ in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                supervised_loss = criterion(logits, labels)
            if not bool(torch.isfinite(supervised_loss).item()):
                raise FloatingPointError("Supervised CE loss is NaN or Inf.")

            # ``sequential`` and ``microbatch`` deliberately release the
            # Reliable CE graph before allocating an Ambiguous strong graph.
            # No parameter/scheduler update occurs between the two backwards.
            if consistency_backward_mode in {"sequential", "microbatch"}:
                scaler.scale(supervised_loss).backward()

            logical_consistency_loss = 0.0
            logical_ambiguous_size = 0
            if ambiguous_iter is not None:
                try:
                    weak_images_cpu, strong_images_cpu, ambiguous_indices = next(ambiguous_iter)
                except StopIteration:
                    ambiguous_iter = iter(ambiguous_loader)
                    weak_images_cpu, strong_images_cpu, ambiguous_indices = next(ambiguous_iter)
                expected_ambiguous = ambiguous_training_mask[ambiguous_indices.cpu().numpy().astype(np.int64)]
                if not np.all(expected_ambiguous):
                    raise RuntimeError("Ambiguous loader emitted a sample outside the current Ambiguous consistency set.")
                logical_ambiguous_size = int(weak_images_cpu.shape[0])
                if logical_ambiguous_size <= 0:
                    raise RuntimeError("Ambiguous loader emitted an empty logical batch.")
                micro_batch_size = (
                    min(int(ambiguous_micro_batch_size), logical_ambiguous_size)
                    if consistency_backward_mode == "microbatch"
                    else logical_ambiguous_size
                )
                joint_loss = None
                for start in range(0, logical_ambiguous_size, micro_batch_size):
                    stop = min(start + micro_batch_size, logical_ambiguous_size)
                    micro_size = int(stop - start)
                    # Keep the logical batch on CPU and move only the current
                    # micro-batch to CUDA; moving all 80 first would defeat
                    # the purpose of the memory-safe path.
                    weak_images = weak_images_cpu[start:stop].to(device, non_blocking=True)
                    strong_images = strong_images_cpu[start:stop].to(device, non_blocking=True)
                    consistency_loss, weak_prob, strong_logits = forward_ambiguous_consistency(
                        torch,
                        model,
                        weak_images,
                        strong_images,
                        amp_enabled=scaler.is_enabled(),
                    )
                    if not bool(torch.isfinite(consistency_loss).item()):
                        raise FloatingPointError("Ambiguous consistency loss is NaN or Inf.")
                    fraction = float(micro_size) / float(logical_ambiguous_size)
                    if consistency_backward_mode == "joint":
                        # joint mode has exactly one micro-batch and keeps the
                        # original single-backward implementation intact.
                        joint_loss = supervised_loss + float(consistency_weight) * consistency_loss
                    else:
                        scaler.scale(float(consistency_weight) * fraction * consistency_loss).backward()
                    logical_consistency_loss += fraction * float(consistency_loss.detach().cpu())
                    weak_confidence_sum += float(weak_prob.max(dim=-1).values.sum().detach().cpu())
                    weak_entropy_sum += float((-(weak_prob * weak_prob.clamp_min(1.0e-12).log()).sum(dim=-1)).sum().detach().cpu())
                    weak_strong_agreement_sum += float(
                        (weak_prob.argmax(dim=-1) == strong_logits.detach().float().argmax(dim=-1)).sum().detach().cpu()
                    )
                    consistency_micro_batch_count += 1
                if consistency_backward_mode == "joint":
                    if joint_loss is None:
                        raise RuntimeError("Joint consistency path did not build a loss.")
                    if not bool(torch.isfinite(joint_loss).item()):
                        raise FloatingPointError("Dynamic training loss is NaN or Inf.")
                    scaler.scale(joint_loss).backward()
                ambiguous_batch_count += 1
                ambiguous_sample_exposures += logical_ambiguous_size
                total_loss_value = float(supervised_loss.detach().cpu()) + float(consistency_weight) * logical_consistency_loss
            else:
                if consistency_backward_mode == "joint":
                    scaler.scale(supervised_loss).backward()
                total_loss_value = float(supervised_loss.detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            current_batch = int(images.shape[0])
            loss_sum += total_loss_value * current_batch
            supervised_loss_sum += float(supervised_loss.detach().cpu()) * current_batch
            seen += current_batch
            optimizer_steps += 1
            scheduler_steps += 1
            if logical_ambiguous_size:
                consistency_loss_sum += logical_consistency_loss

        top1, top5 = evaluate_lora(torch, model, eval_loader, device, len(classes), bool(train_cfg.get("amp", True)))
        row = {
            "method": method,
            "seed": int(seed),
            "epoch": int(epoch),
            "lr_lora": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
            "loss": safe_ratio(loss_sum, seen),
            "supervised_ce_loss": safe_ratio(supervised_loss_sum, seen),
            "consistency_loss": safe_ratio(consistency_loss_sum, ambiguous_batch_count),
            "total_loss": safe_ratio(loss_sum, seen),
            "ambiguous_consistency_enabled": "yes" if ambiguous_consistency else "no",
            "consistency_computed": "yes" if ambiguous_batch_count else "no",
            "consistency_reason": consistency_reason,
            "consistency_weight": float(consistency_weight),
            "consistency_backward_mode": consistency_backward_mode,
            "ambiguous_micro_batch_size": int(ambiguous_micro_batch_size),
            "optimizer_steps": int(optimizer_steps),
            "scheduler_steps": int(scheduler_steps),
            "ambiguous_batch_count": int(ambiguous_batch_count),
            "consistency_micro_batch_count": int(consistency_micro_batch_count),
            "ambiguous_sample_exposures": int(ambiguous_sample_exposures),
            "weak_prediction_max_probability_mean": safe_ratio(weak_confidence_sum, ambiguous_sample_exposures),
            "weak_prediction_entropy_mean": safe_ratio(weak_entropy_sum, ambiguous_sample_exposures),
            "weak_strong_prediction_agreement": safe_ratio(weak_strong_agreement_sum, ambiguous_sample_exposures),
            "top1": float(top1),
            "top5": float(top5),
            "train_samples": int(len(epoch_train_idx)),
            "candidate_samples": int(candidate_mask.sum()),
            "selected_ratio": safe_ratio(len(epoch_train_idx), int(candidate_mask.sum())),
            "eval_samples": int(len(eval_idx)),
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
        }
        logs.append(row)
        if best_row is None or float(row["top1"]) > float(best_row["top1"]):
            best_row = row
            best_state = trainable_state_dict(model)
        epoch_state = trainable_state_dict(model)
        last5_states.append((epoch, epoch_state))
        if len(last5_states) > 5:
            last5_states.pop(0)
        if posthoc_oracle_test and test_loader is not None:
            oracle_states.append((epoch, epoch_state))
        log_stage(
            f"[dynamic-loss] {method} seed={seed} epoch {epoch}/{epochs}: "
            f"loss={row['loss']:.4f}, top1={top1:.4f}, selected={len(epoch_train_idx)}"
        )

        if epoch < epochs and should_update_selection(epoch, warmup_epochs, update_interval):
            previous_mask = selected_mask.copy()
            selection_model_checksum = trainable_model_checksum(model)
            losses: np.ndarray | None = None
            confidence: np.ndarray | None = None
            loss_selected_mask: np.ndarray | None = None
            proto_pass_mask = None
            current_proto_scores: np.ndarray | None = proto_scores
            prototype_snapshot: DynamicPrototypeSnapshot | None = None
            neighbor_margin_snapshot: NeighborMarginSnapshot | None = None
            neighbor_margin_candidate_mask: np.ndarray | None = None
            reliable_mask: np.ndarray | None = None
            ambiguous_mask: np.ndarray | None = None
            suspicious_mask: np.ndarray | None = None
            dual_evidence_strata: DualEvidenceStrata | None = None
            fallback_count = 0

            # Dynamic Prototype Gate Only intentionally never runs this block:
            # no CE ranking is computed or used for its selection decision.
            if selection_strategy in {"loss_only", "loss_and_proto"}:
                losses, confidence = compute_train_losses(
                    torch,
                    model,
                    loss_loader,
                    device,
                    len(train_labels),
                    bool(train_cfg.get("amp", True)),
                )
                if trainable_model_checksum(model) != selection_model_checksum:
                    raise RuntimeError("Model parameters changed during deterministic small-loss evaluation.")
                if class_budget_schedule is not None:
                    loss_selected_mask = select_small_loss_classwise_by_budget(
                        losses,
                        train_labels,
                        candidate_mask,
                        class_budget_schedule[epoch],
                    )
                else:
                    loss_selected_mask = select_small_loss_classwise(
                        losses, train_labels, candidate_mask, retention_ratio
                    )

            if selection_strategy in {"proto_only", "loss_and_proto"}:
                if prototype_mode == "dynamic_lora":
                    dynamic_features = extract_current_lora_cls_features(
                        torch,
                        model,
                        loss_loader,
                        device,
                        len(train_labels),
                        bool(train_cfg.get("amp", True)),
                    )
                    if trainable_model_checksum(model) != selection_model_checksum:
                        raise RuntimeError("Model parameters changed during dynamic prototype feature extraction.")
                    prototype_snapshot = build_dynamic_prototype_snapshot(
                        dynamic_features,
                        train_labels,
                        candidate_mask,
                        float(selected_proto_keep_ratio),
                        selection_model_checksum,
                        build_gate=geometry_mode == "prototype_similarity",
                    )
                    current_proto_scores = prototype_snapshot.scores
                    if geometry_mode == "neighbor_margin":
                        neighbor_margin_snapshot = compute_neighbor_margin(
                            dynamic_features,
                            train_labels,
                            candidate_mask,
                            prototype_snapshot,
                        )
                        neighbor_margin_candidate_mask = build_neighbor_margin_candidates(
                            neighbor_margin_snapshot.margin,
                            train_labels,
                            candidate_mask,
                            float(selected_proto_keep_ratio),
                            positive_only=neighbor_margin_positive_only,
                        )
                        proto_pass_mask = neighbor_margin_candidate_mask
                    else:
                        proto_pass_mask = prototype_snapshot.gate_mask
                else:
                    if auto_proto_keep is not None and selected_proto_keep_ratio is None:
                        if loss_selected_mask is None:
                            raise RuntimeError("Auto prototype routing requires dynamic small-loss selection.")
                        auto_proto_jaccard = mask_jaccard(loss_selected_mask, centroid_mask)
                        selected_proto_keep_ratio = choose_auto_proto_keep_ratio(auto_proto_jaccard, auto_proto_keep)
                        log_stage(
                            f"[dynamic-loss] auto proto_keep_ratio selected: jaccard={auto_proto_jaccard:.4f}, "
                            f"p={selected_proto_keep_ratio:.3f}"
                        )
                    if current_proto_scores is None or selected_proto_keep_ratio is None:
                        raise RuntimeError("Prototype selection was requested without scores and a keep ratio.")
                    proto_pass_mask = select_top_proto_classwise(
                        current_proto_scores,
                        train_labels,
                        candidate_mask,
                        selected_proto_keep_ratio,
                    )

            if selection_strategy == "loss_only":
                if loss_selected_mask is None:
                    raise RuntimeError("loss_only selection did not produce a small-loss mask.")
                selected_mask = loss_selected_mask
            elif selection_strategy == "proto_only":
                if proto_pass_mask is None:
                    raise RuntimeError("proto_only selection did not produce a prototype gate.")
                selected_mask = proto_pass_mask
            elif selection_strategy == "loss_and_proto":
                if loss_selected_mask is None or proto_pass_mask is None or losses is None:
                    raise RuntimeError("loss_and_proto selection requires loss and prototype masks.")
                if geometry_mode == "neighbor_margin":
                    if neighbor_margin_snapshot is None or neighbor_margin_candidate_mask is None:
                        raise RuntimeError("neighbor_margin selection requires a current margin snapshot and candidate mask.")
                    if neighbor_margin_positive_only:
                        # Preserve the existing strict Neighbor-Margin path,
                        # including its explicitly opt-in positive-only fallback.
                        selected_mask, fallback_count = combine_loss_and_neighbor_margin_classwise(
                            loss_selected_mask,
                            neighbor_margin_candidate_mask,
                            neighbor_margin_snapshot.margin,
                            losses,
                            train_labels,
                            candidate_mask,
                            use_fallback=neighbor_margin_use_fallback,
                        )
                    else:
                        # Margin-Rank changes only the class-wise geometry
                        # ordering.  Its intersection fallback is the original
                        # PGDF implementation, applied to margin candidates.
                        fallback_count = (
                            count_classwise_intersection_fallbacks(
                                loss_selected_mask,
                                neighbor_margin_candidate_mask,
                                train_labels,
                                candidate_mask,
                            )
                            if neighbor_margin_use_fallback
                            else 0
                        )
                        selected_mask = (
                            combine_loss_and_proto_classwise(
                                loss_selected_mask,
                                neighbor_margin_candidate_mask,
                                losses,
                                train_labels,
                                candidate_mask,
                            )
                            if neighbor_margin_use_fallback
                            else loss_selected_mask & neighbor_margin_candidate_mask
                        )
                    reliable_mask, ambiguous_mask, suspicious_mask = stratify_neighbor_margin_samples(
                        selected_mask,
                        neighbor_margin_snapshot.margin,
                        candidate_mask,
                        positive_only=neighbor_margin_positive_only,
                    )
                    if not neighbor_margin_positive_only:
                        dual_evidence_strata = build_dual_evidence_strata(
                            loss_selected_mask,
                            neighbor_margin_candidate_mask,
                            candidate_mask,
                            selected_mask,
                        )
                        if ambiguous_consistency:
                            # Fallback samples remain supervised active but
                            # are removed from consistency to prevent dual use.
                            ambiguous_training_mask = dual_evidence_strata.ambiguous_for_consistency.copy()
                else:
                    # This is the original PGDF path.  Keep its selection and
                    # class-preserving fallback byte-for-byte independent of
                    # the Neighbor-Margin implementation.
                    fallback_count = count_classwise_intersection_fallbacks(
                        loss_selected_mask,
                        proto_pass_mask,
                        train_labels,
                        candidate_mask,
                    )
                    selected_mask = combine_loss_and_proto_classwise(
                        loss_selected_mask,
                        proto_pass_mask,
                        losses,
                        train_labels,
                        candidate_mask,
                    )
            else:
                raise RuntimeError(f"Unsupported selection_strategy: {selection_strategy}")
            update_rows.append(
                build_update_row(
                    method,
                    dataset_name,
                    seed,
                    retention_ratio,
                    selected_proto_keep_ratio,
                    epoch,
                    candidate_mask,
                    train_labels,
                    selected_mask,
                    previous_mask,
                    centroid_mask,
                    losses,
                    loss_selected_mask=loss_selected_mask,
                    proto_pass_mask=proto_pass_mask,
                    proto_scores=current_proto_scores,
                    auto_proto_jaccard=auto_proto_jaccard,
                    selection_strategy=selection_strategy,
                    prototype_mode=prototype_mode,
                    prototype_snapshot=prototype_snapshot,
                    selection_model_checksum=selection_model_checksum,
                    lora_updated_since_initial=(selection_model_checksum != initial_trainable_checksum),
                    fallback_count=fallback_count,
                    geometry_mode=geometry_mode,
                    neighbor_margin_use_fallback=neighbor_margin_use_fallback,
                    neighbor_margin_positive_only=neighbor_margin_positive_only,
                    neighbor_margin_snapshot=neighbor_margin_snapshot,
                    reliable_mask=reliable_mask,
                    ambiguous_mask=ambiguous_mask,
                    suspicious_mask=suspicious_mask,
                    dual_evidence_strata=dual_evidence_strata,
                )
            )
            selection_rows.extend(
                build_selection_rows(
                    method,
                    dataset_name,
                    seed,
                    retention_ratio,
                    selected_proto_keep_ratio,
                    epoch,
                    train_paths,
                    train_labels,
                    candidate_mask,
                    selected_mask,
                    losses,
                    confidence,
                    loss_selected_mask=loss_selected_mask,
                    proto_pass_mask=proto_pass_mask,
                    proto_scores=current_proto_scores,
                    selection_strategy=selection_strategy,
                    prototype_mode=prototype_mode,
                    geometry_mode=geometry_mode,
                    neighbor_margin_positive_only=neighbor_margin_positive_only,
                    neighbor_margin_snapshot=neighbor_margin_snapshot,
                    reliable_mask=reliable_mask,
                    ambiguous_mask=ambiguous_mask,
                    suspicious_mask=suspicious_mask,
                )
            )
            per_class_rows.extend(
                build_per_class_rows(
                    method,
                    dataset_name,
                    seed,
                    retention_ratio,
                    selected_proto_keep_ratio,
                    epoch,
                    train_labels,
                    candidate_mask,
                    selected_mask,
                    losses,
                    loss_selected_mask=loss_selected_mask,
                    proto_pass_mask=proto_pass_mask,
                    proto_scores=current_proto_scores,
                    selection_strategy=selection_strategy,
                    prototype_mode=prototype_mode,
                    geometry_mode=geometry_mode,
                    neighbor_margin_positive_only=neighbor_margin_positive_only,
                    neighbor_margin_snapshot=neighbor_margin_snapshot,
                    reliable_mask=reliable_mask,
                    ambiguous_mask=ambiguous_mask,
                    suspicious_mask=suspicious_mask,
                )
            )
            neighbor_summary = (
                f", reliable={int(reliable_mask.sum())}, ambiguous={int(ambiguous_mask.sum())}, "
                f"suspicious={int(suspicious_mask.sum())}, reliable_zero_classes="
                f"{update_rows[-1]['reliable_zero_class_count']}, "
                f"strict_intersection={update_rows[-1]['strict_intersection_count']}, "
                f"fallback_classes={fallback_count}, "
                f"selected_negative_margin={update_rows[-1]['selected_negative_margin_count']}, "
                f"selected_zero_classes={update_rows[-1]['selected_zero_class_count']}, "
                f"margin(mean/std/min/max/median)="
                f"{update_rows[-1]['margin_mean']:.4f}/"
                f"{update_rows[-1]['margin_std']:.4f}/"
                f"{update_rows[-1]['margin_min']:.4f}/"
                f"{update_rows[-1]['margin_max']:.4f}/"
                f"{update_rows[-1]['margin_median']:.4f}"
                if geometry_mode == "neighbor_margin"
                and reliable_mask is not None
                and ambiguous_mask is not None
                and suspicious_mask is not None
                else ""
            )
            log_stage(
                f"[dynamic-loss] selection update epoch={epoch}: "
                f"selected={int(selected_mask.sum())}, proto_mode={prototype_mode}, "
                f"prev_jaccard={update_rows[-1]['overlap_with_previous_selection']:.4f}{neighbor_summary}"
            )

    final_state = trainable_state_dict(model)
    protocol_metrics: dict[str, Any] = {}
    if test_loader is not None and best_state is not None and best_row is not None:
        epoch_test_metrics: dict[int, tuple[float, float]] = {}
        if posthoc_oracle_test:
            epoch_test_metrics = {
                epoch: evaluate_state_lora(torch, model, state, test_loader, device, len(classes), bool(train_cfg.get("amp", True)))
                for epoch, state in oracle_states
            }
            validation_selected_test_top1, validation_selected_test_top5 = epoch_test_metrics[int(best_row["epoch"])]
            final_test_top1, final_test_top5 = epoch_test_metrics[int(epochs)]
            last5_test_top1 = np.asarray([epoch_test_metrics[epoch][0] for epoch, _ in last5_states], dtype=np.float32)
            last5_test_mean: float | str = float(last5_test_top1.mean())
            last5_test_std: float | str = float(last5_test_top1.std())
        elif official_test_selected_only:
            validation_selected_test_top1, validation_selected_test_top5 = evaluate_state_lora(
                torch, model, best_state, test_loader, device, len(classes), bool(train_cfg.get("amp", True))
            )
            final_test_top1 = ""
            final_test_top5 = ""
            last5_test_mean = ""
            last5_test_std = ""
        else:
            validation_selected_test_top1, validation_selected_test_top5 = evaluate_state_lora(
                torch, model, best_state, test_loader, device, len(classes), bool(train_cfg.get("amp", True))
            )
            final_test_top1, final_test_top5 = evaluate_state_lora(
                torch, model, final_state, test_loader, device, len(classes), bool(train_cfg.get("amp", True))
            )
            last5_test_top1 = np.array(
                [
                    evaluate_state_lora(torch, model, state, test_loader, device, len(classes), bool(train_cfg.get("amp", True)))[0]
                    for _, state in last5_states
                ],
                dtype=np.float32,
            )
            last5_test_mean = float(last5_test_top1.mean())
            last5_test_std = float(last5_test_top1.std())
        protocol_metrics = {
            "checkpoint_protocol": checkpoint_protocol,
            "official_test_evaluation": (
                "validation_selected_only"
                if official_test_selected_only
                else "posthoc_oracle_curve"
                if posthoc_oracle_test
                else "validation_selected_final_last5"
            ),
            "validation_samples": int(len(eval_idx)),
            "test_samples": int(len(test_idx)),
            "best_val_epoch": int(best_row["epoch"]),
            "best_val_top1": float(best_row["top1"]),
            "best_val_top5": float(best_row["top5"]),
            "validation_selected_test_top1": float(validation_selected_test_top1),
            "validation_selected_test_top5": float(validation_selected_test_top5),
            "final_test_top1": final_test_top1 if final_test_top1 == "" else float(final_test_top1),
            "final_test_top5": final_test_top5 if final_test_top5 == "" else float(final_test_top5),
            "last5_test_mean": last5_test_mean,
            "last5_test_std": last5_test_std,
        }
        if posthoc_oracle_test:
            oracle_rows = [(epoch, *metrics) for epoch, metrics in epoch_test_metrics.items()]
            oracle_epoch, oracle_top1, oracle_top5 = max(oracle_rows, key=lambda item: item[1])
            protocol_metrics.update(
                {
                    "oracle_best_test_epoch": int(oracle_epoch),
                    "oracle_best_test_top1": float(oracle_top1),
                    "oracle_best_test_top5": float(oracle_top5),
                    "oracle_best_to_final_drop": float(oracle_top1 - final_test_top1),
                }
            )
    if checkpoint_path is not None and best_state is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": method,
                "seed": int(seed),
                "classes": classes,
                "state_dict": best_state,
                "best_epoch": int(best_row["epoch"]) if best_row else None,
                "best_top1": float(best_row["top1"]) if best_row else None,
                "retention_ratio": "" if selection_strategy == "proto_only" else float(retention_ratio),
                "proto_keep_ratio": selected_proto_keep_ratio,
                "auto_proto_keep": auto_proto_keep is not None,
                "auto_proto_jaccard": auto_proto_jaccard,
                "warmup_epochs": int(warmup_epochs),
                "update_interval": int(update_interval),
                "budget_matched": class_budget_schedule is not None,
                "scheduler_retention_ratio": float(effective_scheduler_ratio),
                "selection_strategy": selection_strategy,
                "prototype_mode": prototype_mode,
                "geometry_mode": geometry_mode,
                "neighbor_margin_use_fallback": bool(neighbor_margin_use_fallback),
                "neighbor_margin_positive_only": bool(neighbor_margin_positive_only),
                "ambiguous_consistency": bool(ambiguous_consistency),
                "consistency_weight": float(consistency_weight),
                "consistency_backward_mode": consistency_backward_mode,
                "ambiguous_micro_batch_size": int(ambiguous_micro_batch_size),
                "checkpoint_protocol": checkpoint_protocol,
                **protocol_metrics,
            },
            checkpoint_path,
        )
    if final_checkpoint_path is not None:
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "method": method,
                "seed": int(seed),
                "classes": classes,
                "state_dict": final_state,
                "final_epoch": int(epochs),
                "retention_ratio": "" if selection_strategy == "proto_only" else float(retention_ratio),
                "proto_keep_ratio": selected_proto_keep_ratio,
                "auto_proto_keep": auto_proto_keep is not None,
                "auto_proto_jaccard": auto_proto_jaccard,
                "warmup_epochs": int(warmup_epochs),
                "update_interval": int(update_interval),
                "budget_matched": class_budget_schedule is not None,
                "scheduler_retention_ratio": float(effective_scheduler_ratio),
                "selection_strategy": selection_strategy,
                "prototype_mode": prototype_mode,
                "geometry_mode": geometry_mode,
                "neighbor_margin_use_fallback": bool(neighbor_margin_use_fallback),
                "neighbor_margin_positive_only": bool(neighbor_margin_positive_only),
                "ambiguous_consistency": bool(ambiguous_consistency),
                "consistency_weight": float(consistency_weight),
                "consistency_backward_mode": consistency_backward_mode,
                "ambiguous_micro_batch_size": int(ambiguous_micro_batch_size),
                "checkpoint_protocol": checkpoint_protocol,
                **protocol_metrics,
            },
            final_checkpoint_path,
        )
    if last5_checkpoint_dir is not None:
        last5_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch, state in last5_states:
            torch.save(
                {"method": method, "seed": int(seed), "classes": classes, "state_dict": state, "epoch": int(epoch), "checkpoint_protocol": checkpoint_protocol},
                last5_checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )

    summary = summarize_lora_logs(method, seed, logs)
    summary.update(protocol_metrics)
    summary.update(
        {
            "retention_ratio": "" if selection_strategy == "proto_only" else float(retention_ratio),
            "proto_keep_ratio": selected_proto_keep_ratio if selected_proto_keep_ratio is not None else "",
            "auto_proto_keep": "yes" if auto_proto_keep is not None else "no",
            "auto_proto_jaccard": auto_proto_jaccard if auto_proto_jaccard is not None else "",
            "warmup_epochs": int(warmup_epochs),
            "update_interval": int(update_interval),
            "budget_matched": "yes" if class_budget_schedule is not None else "no",
            "scheduler_retention_ratio": float(effective_scheduler_ratio),
            "candidate_samples": int(candidate_mask.sum()),
            "final_selected_samples": int(selected_mask.sum()),
            "selection_updates": len(update_rows),
            "selection_strategy": selection_strategy,
            "prototype_mode": prototype_mode,
            "geometry_mode": geometry_mode,
            "neighbor_margin_use_fallback": "yes" if neighbor_margin_use_fallback else "no",
            "neighbor_margin_positive_only": "yes" if neighbor_margin_positive_only else "no",
            "ambiguous_consistency": "yes" if ambiguous_consistency else "no",
            "consistency_weight": float(consistency_weight),
            "consistency_backward_mode": consistency_backward_mode,
            "ambiguous_micro_batch_size": int(ambiguous_micro_batch_size),
            "backbone_frozen": "yes",
            "lora_updated_before_selection": (
                "yes" if any(row.get("lora_updated_since_initial") == "yes" for row in update_rows) else "no"
            ),
        }
    )
    return DynamicLossRunResult(
        logs=logs,
        summary=summary,
        trainable_modules=trainable_modules,
        trainable_params=trainable_params,
        total_params=total_params,
        selection_rows=selection_rows,
        update_rows=update_rows,
        per_class_rows=per_class_rows,
    )


def validate_dynamic_args(
    retention_ratio: float,
    warmup_epochs: int,
    update_interval: int,
    proto_keep_ratio: float | None = None,
    auto_proto_keep: dict[str, float] | None = None,
    *,
    selection_strategy: str = "loss_only",
    prototype_mode: str = "fixed",
    geometry_mode: str = "prototype_similarity",
    neighbor_margin_use_fallback: bool = False,
    neighbor_margin_positive_only: bool = True,
    ambiguous_consistency: bool = False,
    consistency_weight: float = 0.5,
    consistency_backward_mode: str = "joint",
    ambiguous_micro_batch_size: int = 20,
) -> None:
    if not 0.0 < retention_ratio <= 1.0:
        raise ValueError("retention_ratio must satisfy 0 < ratio <= 1.")
    if proto_keep_ratio is not None and not 0.0 < proto_keep_ratio <= 1.0:
        raise ValueError("proto_keep_ratio must satisfy 0 < ratio <= 1.")
    if proto_keep_ratio is not None and auto_proto_keep is not None:
        raise ValueError("Use either proto_keep_ratio or auto_proto_keep, not both.")
    if auto_proto_keep is not None:
        validate_auto_proto_keep(auto_proto_keep)
    if selection_strategy not in {"loss_only", "proto_only", "loss_and_proto"}:
        raise ValueError("selection_strategy must be loss_only, proto_only, or loss_and_proto.")
    if prototype_mode not in {"fixed", "dynamic_lora"}:
        raise ValueError("prototype_mode must be fixed or dynamic_lora.")
    if geometry_mode not in {"prototype_similarity", "neighbor_margin"}:
        raise ValueError("geometry_mode must be prototype_similarity or neighbor_margin.")
    if not isinstance(neighbor_margin_use_fallback, bool):
        raise ValueError("neighbor_margin_use_fallback must be boolean.")
    if not isinstance(neighbor_margin_positive_only, bool):
        raise ValueError("neighbor_margin_positive_only must be boolean.")
    if not isinstance(ambiguous_consistency, bool):
        raise ValueError("ambiguous_consistency must be boolean.")
    if not math.isfinite(float(consistency_weight)) or float(consistency_weight) < 0.0:
        raise ValueError("consistency_weight must be finite and non-negative.")
    if consistency_backward_mode not in CONSISTENCY_BACKWARD_MODES:
        raise ValueError(
            "consistency_backward_mode must be one of "
            f"{', '.join(CONSISTENCY_BACKWARD_MODES)}."
        )
    if int(ambiguous_micro_batch_size) <= 0:
        raise ValueError("ambiguous_micro_batch_size must be positive.")
    if ambiguous_consistency and not (
        geometry_mode == "neighbor_margin" and not neighbor_margin_positive_only
    ):
        raise ValueError(
            "ambiguous_consistency requires Margin-Rank "
            "(geometry_mode='neighbor_margin', neighbor_margin_positive_only=False)."
        )
    if selection_strategy in {"proto_only", "loss_and_proto"} and proto_keep_ratio is None:
        raise ValueError("Prototype selection requires proto_keep_ratio.")
    if selection_strategy == "proto_only" and auto_proto_keep is not None:
        raise ValueError("proto_only selection cannot use auto_proto_keep.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if update_interval <= 0:
        raise ValueError("update_interval must be positive.")


def should_update_selection(epoch: int, warmup_epochs: int, update_interval: int) -> bool:
    return epoch >= warmup_epochs and (epoch - warmup_epochs) % update_interval == 0


def selection_update_epochs(epochs: int, warmup_epochs: int, update_interval: int) -> list[int]:
    """Return post-epoch selection updates; the final epoch has no successor."""
    return [
        epoch
        for epoch in range(1, max(0, int(epochs)))
        if should_update_selection(epoch, warmup_epochs, update_interval)
    ]


def estimate_dynamic_total_steps(candidate_count: int, batch_size: int, epochs: int, warmup_epochs: int, retention_ratio: float) -> int:
    """Estimate optimizer steps so LR schedules are comparable across retention ratios."""
    full_steps = max(1, math.ceil(candidate_count / max(batch_size, 1)))
    retained_count = max(1, int(math.floor(candidate_count * retention_ratio)))
    retained_steps = max(1, math.ceil(retained_count / max(batch_size, 1)))
    warmup_epoch_count = min(max(warmup_epochs, 0), max(epochs, 0))
    filtered_epoch_count = max(0, epochs - warmup_epoch_count)
    return max(1, warmup_epoch_count * full_steps + filtered_epoch_count * retained_steps)


def estimate_selection_retention_ratio(retention_ratio: float, proto_keep_ratio: float | None, auto_proto_keep: dict[str, float] | None = None) -> float:
    """Conservative LR-step estimate for optional loss/prototype intersection selection."""
    if proto_keep_ratio is None:
        if auto_proto_keep is not None:
            return min(
                retention_ratio,
                min(
                    float(auto_proto_keep["p_high"]),
                    float(auto_proto_keep["p_mid"]),
                    float(auto_proto_keep["p_low"]),
                    float(auto_proto_keep["p_very_low"]),
                ),
            )
        return retention_ratio
    return min(retention_ratio, proto_keep_ratio)


def format_optional_ratio(value: float | None) -> str:
    return "none" if value is None else f"{value:.3f}"


def validate_auto_proto_keep(rule: dict[str, float]) -> None:
    required = ["high_jaccard", "mid_jaccard", "low_jaccard", "p_high", "p_mid", "p_low", "p_very_low"]
    missing = [key for key in required if key not in rule]
    if missing:
        raise ValueError(f"auto_proto_keep is missing required keys: {missing}")
    if float(rule["high_jaccard"]) < float(rule["mid_jaccard"]) or float(rule["mid_jaccard"]) < float(rule["low_jaccard"]):
        raise ValueError("auto_proto_keep thresholds must satisfy high_jaccard >= mid_jaccard >= low_jaccard.")
    for key in ["high_jaccard", "mid_jaccard", "low_jaccard", "p_high", "p_mid", "p_low", "p_very_low"]:
        value = float(rule[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"auto_proto_keep {key} must be in [0, 1], got {value}.")
    for key in ["p_high", "p_mid", "p_low", "p_very_low"]:
        if float(rule[key]) <= 0.0:
            raise ValueError(f"auto_proto_keep {key} must be > 0.")


def choose_auto_proto_keep_ratio(jaccard: float, rule: dict[str, float]) -> float:
    if jaccard >= float(rule["high_jaccard"]):
        return float(rule["p_high"])
    if jaccard >= float(rule["mid_jaccard"]):
        return float(rule["p_mid"])
    if jaccard >= float(rule["low_jaccard"]):
        return float(rule["p_low"])
    return float(rule["p_very_low"])


def compute_train_losses(torch: Any, model: Any, loader: Any, device: str, total_train: int, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    was_training = bool(model.training)
    model.eval()
    losses = np.full(total_train, np.nan, dtype=np.float32)
    confidence = np.full(total_train, np.nan, dtype=np.float32)
    try:
        with torch.no_grad():
            for images, labels, indices in progress_iter(loader, total=len(loader), desc="Dynamic loss eval"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
                    logits = model(images)
                    ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
                    probs = torch.softmax(logits, dim=1)
                    conf = probs.gather(1, labels[:, None]).squeeze(1)
                idx = indices.cpu().numpy().astype(np.int64)
                losses[idx] = ce.detach().cpu().numpy().astype(np.float32)
                confidence[idx] = conf.detach().cpu().numpy().astype(np.float32)
    finally:
        model.train(was_training)
    return losses, confidence


def extract_current_lora_cls_features(
    torch: Any,
    model: Any,
    loader: Any,
    device: str,
    total_train: int,
    amp: bool,
) -> np.ndarray:
    """Extract current, deterministic CLS representations before the head.

    ``loader`` is built over the entire formal noisy training pool with the
    evaluation transform.  Calling ``extract_cls_features`` bypasses the head,
    while the backbone's injected LoRA modules remain active in the forward
    pass.  The original model train/eval mode is restored before returning.
    """
    if not hasattr(model, "extract_cls_features"):
        raise AttributeError("Dynamic prototype extraction requires model.extract_cls_features().")
    was_training = bool(model.training)
    features: np.ndarray | None = None
    try:
        model.eval()
        with torch.no_grad():
            for images, _, indices in progress_iter(loader, total=len(loader), desc="Dynamic prototype CLS eval"):
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
                    cls = model.extract_cls_features(images)
                if cls.ndim != 2:
                    raise ValueError(f"Expected [batch, dim] CLS features, got shape {tuple(cls.shape)}.")
                if features is None:
                    features = np.full((total_train, int(cls.shape[1])), np.nan, dtype=np.float32)
                idx = indices.cpu().numpy().astype(np.int64)
                features[idx] = cls.detach().float().cpu().numpy().astype(np.float32)
    finally:
        model.train(was_training)
    if features is None:
        raise RuntimeError("Dynamic prototype extraction produced no CLS features.")
    return features


def build_dynamic_prototype_snapshot(
    current_features: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    proto_keep_ratio: float,
    model_state_checksum: str,
    *,
    build_gate: bool = True,
) -> DynamicPrototypeSnapshot:
    """Rebuild observed-label prototypes and, optionally, the original top-p gate.

    ``build_gate=False`` is used by Neighbor-Margin: it retains the identical
    prototype construction but deliberately skips the legacy absolute-
    similarity candidate decision.
    """
    features = np.asarray(current_features, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if features.ndim != 2 or features.shape[0] != len(labels):
        raise ValueError("current_features must be a [num_samples, feature_dim] array aligned with labels.")
    if candidate_mask.shape != labels.shape:
        raise ValueError("candidate_mask must align with labels.")
    if not 0.0 < proto_keep_ratio <= 1.0:
        raise ValueError("proto_keep_ratio must satisfy 0 < p <= 1.")
    if np.any(~np.isfinite(features[candidate_mask])):
        raise ValueError("Dynamic CLS extraction has missing or non-finite training-pool features.")

    scores = np.full(len(labels), np.nan, dtype=np.float32)
    gate_mask = np.zeros(len(labels), dtype=bool)
    prototype_chunks: list[bytes] = []
    prototype_values: list[np.ndarray] = []
    prototype_vectors: list[np.ndarray] = []
    prototype_labels: list[str] = []
    class_retained_counts: dict[str, int] = {}
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        vectors = np.asarray(features[idx], dtype=np.float64)
        row_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(row_norms <= 0.0):
            raise ValueError(f"Dynamic CLS features contain a zero-norm vector for observed-label class {label}.")
        normalized_vectors = vectors / row_norms
        prototype = normalized_vectors.mean(axis=0)
        prototype_norm = float(np.linalg.norm(prototype))
        if prototype_norm <= 0.0:
            raise ValueError(f"Observed-label prototype has zero norm for class {label}.")
        prototype = prototype / prototype_norm
        class_scores = (normalized_vectors @ prototype).astype(np.float32)
        scores[idx] = class_scores
        if build_gate:
            keep = len(idx) if proto_keep_ratio >= 1.0 else max(1, int(math.floor(proto_keep_ratio * len(idx))))
            # Stable sort makes ties resolve by the original training index.
            order = np.argsort(-class_scores, kind="mergesort")
            gate_mask[idx[order[:keep]]] = True
            class_retained_counts[str(label)] = int(keep)
        else:
            class_retained_counts[str(label)] = 0
        prototype_values.append(prototype.astype(np.float32, copy=False))
        prototype_vectors.append(prototype)
        prototype_labels.append(str(label))
        prototype_chunks.extend((str(label).encode("utf-8"), prototype.astype(np.float32, copy=False).tobytes()))

    if np.any(np.isnan(scores[candidate_mask])):
        raise RuntimeError("Dynamic prototype scoring did not cover every training-pool sample.")
    prototype_digest = hashlib.sha256()
    for chunk in prototype_chunks:
        prototype_digest.update(chunk)
    gate_digest = hashlib.sha256(np.where(gate_mask)[0].astype(np.int64).tobytes()).hexdigest()
    stacked_prototypes = np.vstack(prototype_values)
    return DynamicPrototypeSnapshot(
        scores=scores,
        gate_mask=gate_mask,
        class_retained_counts=class_retained_counts,
        feature_dim=int(features.shape[1]),
        prototype_checksum=prototype_digest.hexdigest(),
        gate_membership_hash=gate_digest,
        prototype_mean=float(stacked_prototypes.mean()),
        prototype_std=float(stacked_prototypes.std()),
        model_state_checksum=model_state_checksum,
        prototype_labels=tuple(prototype_labels),
        prototypes=np.vstack(prototype_vectors),
    )


def compute_neighbor_margin(
    current_features: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    prototype_snapshot: DynamicPrototypeSnapshot,
) -> NeighborMarginSnapshot:
    """Compute observed-vs-nearest-other-class geometry without retaining N×C.

    Features and prototypes are normalized exactly as in
    :func:`build_dynamic_prototype_snapshot`.  The all-class matrix exists only
    as a local update-time array; only four per-sample diagnostics survive.
    """
    features = np.asarray(current_features, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if features.ndim != 2 or features.shape[0] != len(labels):
        raise ValueError("current_features must align with labels for Neighbor-Margin geometry.")
    if candidate_mask.shape != labels.shape:
        raise ValueError("candidate_mask must align with labels for Neighbor-Margin geometry.")
    prototype_labels = tuple(str(label) for label in prototype_snapshot.prototype_labels)
    prototypes = np.asarray(prototype_snapshot.prototypes, dtype=np.float64)
    if prototypes.ndim != 2 or prototypes.shape[0] != len(prototype_labels):
        raise ValueError("Dynamic prototype labels and vectors are misaligned.")
    if len(prototype_labels) < 2:
        raise ValueError("Neighbor-Margin requires at least two non-empty observed-label prototypes.")
    if not np.all(np.isfinite(prototypes)):
        raise ValueError("Neighbor-Margin prototypes contain non-finite values.")
    prototype_norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    if np.any(prototype_norms <= 0.0):
        raise ValueError("Neighbor-Margin prototypes contain a zero-norm vector.")
    prototypes = prototypes / prototype_norms

    candidate_idx = np.where(candidate_mask)[0]
    vectors = features[candidate_idx]
    if np.any(~np.isfinite(vectors)):
        raise ValueError("Neighbor-Margin features contain missing or non-finite training-pool values.")
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(vector_norms <= 0.0):
        raise ValueError("Neighbor-Margin features contain a zero-norm training-pool vector.")
    normalized_vectors = vectors / vector_norms
    label_to_column = {label: column for column, label in enumerate(prototype_labels)}
    try:
        observed_columns = np.asarray([label_to_column[str(labels[index])] for index in candidate_idx], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Candidate observed-label class has no prototype: {exc.args[0]!r}.") from exc

    # This N_pool x C temporary is intentionally discarded before return.
    similarities = normalized_vectors @ prototypes.T
    observed_values = similarities[np.arange(len(candidate_idx)), observed_columns].copy()
    similarities[np.arange(len(candidate_idx)), observed_columns] = -np.inf
    competitor_columns = np.argmax(similarities, axis=1)
    competitor_values = similarities[np.arange(len(candidate_idx)), competitor_columns]
    if np.any(~np.isfinite(observed_values)) or np.any(~np.isfinite(competitor_values)):
        raise RuntimeError("Neighbor-Margin could not determine a finite observed/competing similarity.")

    margin = np.full(len(labels), np.nan, dtype=np.float32)
    observed_similarity = np.full(len(labels), np.nan, dtype=np.float32)
    competitor_similarity = np.full(len(labels), np.nan, dtype=np.float32)
    max_label_width = max(1, max(len(label) for label in prototype_labels))
    competitor_class = np.full(len(labels), "", dtype=f"<U{max_label_width}")
    margin[candidate_idx] = (observed_values - competitor_values).astype(np.float32)
    observed_similarity[candidate_idx] = observed_values.astype(np.float32)
    competitor_similarity[candidate_idx] = competitor_values.astype(np.float32)
    competitor_class[candidate_idx] = np.asarray(
        [prototype_labels[column] for column in competitor_columns], dtype=competitor_class.dtype
    )
    if np.any(competitor_class[candidate_idx] == labels[candidate_idx]):
        raise RuntimeError("Neighbor-Margin competitor_class must differ from observed class.")
    if not np.allclose(
        margin[candidate_idx],
        observed_similarity[candidate_idx] - competitor_similarity[candidate_idx],
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise RuntimeError("Neighbor-Margin invariant failed: margin != observed_similarity - competitor_similarity.")
    return NeighborMarginSnapshot(
        margin=margin,
        observed_similarity=observed_similarity,
        competitor_similarity=competitor_similarity,
        competitor_class=competitor_class,
        candidate_mask=candidate_mask.copy(),
    )


def build_neighbor_margin_candidates(
    margin: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    proto_keep_ratio: float,
    *,
    positive_only: bool = True,
) -> np.ndarray:
    """Keep observed-class top-p Neighbor-Margin samples with an optional sign gate."""
    margin = np.asarray(margin, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if margin.shape != labels.shape or candidate_mask.shape != labels.shape:
        raise ValueError("Neighbor-Margin arrays must all align with labels.")
    if not 0.0 < proto_keep_ratio <= 1.0:
        raise ValueError("proto_keep_ratio must satisfy 0 < p <= 1 for Neighbor-Margin.")
    if np.any(~np.isfinite(margin[candidate_mask])):
        raise ValueError("Neighbor-Margin is missing or non-finite for a training-pool sample.")

    selected = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        keep = len(idx) if proto_keep_ratio >= 1.0 else max(1, int(math.floor(proto_keep_ratio * len(idx))))
        # mergesort preserves ``idx`` order, which is the stable original
        # training-array index order when margins tie.
        order = np.argsort(-margin[idx], kind="mergesort")
        top_idx = idx[order[:keep]]
        if positive_only:
            top_idx = top_idx[margin[top_idx] > 0.0]
        selected[top_idx] = True
    if np.any(selected & ~candidate_mask):
        raise RuntimeError("Neighbor-Margin candidate contains a non-training-pool sample.")
    if positive_only and np.any(margin[selected] <= 0.0):
        raise RuntimeError("Neighbor-Margin candidate must contain only strictly positive margins.")
    return selected


def combine_loss_and_neighbor_margin_classwise(
    loss_selected_mask: np.ndarray,
    neighbor_margin_candidate_mask: np.ndarray,
    margin: np.ndarray,
    losses: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    use_fallback: bool,
) -> tuple[np.ndarray, int]:
    """Return strict Reliable intersection, with an explicitly opt-in extension.

    The formal default never backfills a class.  If a later experiment opts in,
    a vacant class may receive only its best positive-margin small-loss sample;
    negative-margin (Suspicious) samples are never promoted.
    """
    loss_selected_mask = np.asarray(loss_selected_mask, dtype=bool)
    neighbor_margin_candidate_mask = np.asarray(neighbor_margin_candidate_mask, dtype=bool)
    margin = np.asarray(margin, dtype=np.float32)
    losses = np.asarray(losses, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    reliable = loss_selected_mask & neighbor_margin_candidate_mask
    if np.any(reliable & ~candidate_mask):
        raise RuntimeError("Reliable selection contains a non-training-pool sample.")
    if np.any(margin[reliable] <= 0.0):
        raise RuntimeError("Reliable samples must have strictly positive Neighbor-Margin.")
    fallback_count = 0
    if not use_fallback:
        if not np.array_equal(reliable, loss_selected_mask & neighbor_margin_candidate_mask):
            raise RuntimeError("Neighbor-Margin Reliable mask must be the strict gate intersection.")
        return reliable, fallback_count

    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        if np.any(reliable[idx]):
            continue
        eligible = idx[loss_selected_mask[idx] & (margin[idx] > 0.0)]
        if len(eligible) == 0:
            continue
        # Highest positive margin first; stable original-index order resolves
        # exact ties.  This extension remains explicitly isolated from the
        # baseline PGDF fallback and disabled for formal Neighbor-Margin runs.
        order = np.argsort(-margin[eligible], kind="mergesort")
        reliable[int(eligible[order[0]])] = True
        fallback_count += 1
    if np.any(margin[reliable] <= 0.0):
        raise RuntimeError("Neighbor-Margin fallback promoted a non-positive-margin sample.")
    return reliable, fallback_count


def stratify_neighbor_margin_samples(
    selected_mask: np.ndarray,
    margin: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    positive_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build Neighbor-Margin diagnostics without changing the active subset.

    Strict mode retains the historical partition: Reliable is exactly the
    active subset and necessarily has positive margin.  In Margin-Rank mode,
    ``selected_mask`` may contain a negative-margin sample because class-wise
    margin ranking and the original PGDF fallback intentionally allow it.
    Such a sample remains ``suspicious`` diagnostically while also remaining
    active for training; Reliable therefore records only positive-margin
    selected samples in that mode.
    """
    selected = np.asarray(selected_mask, dtype=bool)
    margin = np.asarray(margin, dtype=np.float32)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if selected.shape != candidate_mask.shape or margin.shape != candidate_mask.shape:
        raise ValueError("Neighbor-Margin stratification arrays must have matching shapes.")
    if np.any(~np.isfinite(margin[candidate_mask])):
        raise ValueError("Neighbor-Margin stratification requires finite candidate margins.")
    if np.any(selected & ~candidate_mask):
        raise RuntimeError("Neighbor-Margin active selection contains a non-training-pool sample.")
    if positive_only:
        reliable = selected
        if np.any(margin[reliable] <= 0.0):
            raise RuntimeError("Reliable partition contains a non-positive-margin sample.")
    else:
        reliable = selected & (margin > 0.0)
    suspicious = candidate_mask & (margin < 0.0)
    ambiguous = candidate_mask & ~reliable & ~suspicious
    if np.any(reliable & ambiguous) or np.any(reliable & suspicious) or np.any(ambiguous & suspicious):
        raise RuntimeError("Neighbor-Margin sample states must be mutually exclusive.")
    if not np.array_equal(reliable | ambiguous | suspicious, candidate_mask):
        raise RuntimeError("Neighbor-Margin sample states must cover the complete training pool.")
    if np.any(margin[suspicious] >= 0.0):
        raise RuntimeError("Suspicious partition contains a non-negative-margin sample.")
    return reliable, ambiguous, suspicious


def build_dual_evidence_strata(
    loss_selected_mask: np.ndarray,
    geometry_candidate_mask: np.ndarray,
    candidate_mask: np.ndarray,
    actual_active_mask: np.ndarray,
) -> DualEvidenceStrata:
    """Construct the Margin-Rank L/G diagnostic partition without reranking.

    The four diagnostic groups are defined by the already-computed class-wise
    L (small loss) and G (Neighbor-Margin) masks.  ``actual_active_mask`` is
    supplied by the unchanged original PGDF combine/fallback helper; it is not
    used to redefine strict Reliable.
    """
    loss_selected_mask = np.asarray(loss_selected_mask, dtype=bool)
    geometry_candidate_mask = np.asarray(geometry_candidate_mask, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    actual_active_mask = np.asarray(actual_active_mask, dtype=bool)
    if not (
        loss_selected_mask.shape
        == geometry_candidate_mask.shape
        == candidate_mask.shape
        == actual_active_mask.shape
    ):
        raise ValueError("Dual-evidence masks must have identical shapes.")
    if np.any((loss_selected_mask | geometry_candidate_mask | actual_active_mask) & ~candidate_mask):
        raise RuntimeError("Dual-evidence masks contain a non-training-pool sample.")

    strict_reliable = candidate_mask & loss_selected_mask & geometry_candidate_mask
    ambiguous_l = candidate_mask & loss_selected_mask & ~geometry_candidate_mask
    ambiguous_g = candidate_mask & geometry_candidate_mask & ~loss_selected_mask
    ambiguous = ambiguous_l | ambiguous_g
    suspicious = candidate_mask & ~loss_selected_mask & ~geometry_candidate_mask
    groups = (strict_reliable, ambiguous_l, ambiguous_g, suspicious)
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            if np.any(left & right):
                raise RuntimeError("Dual-evidence strata must be mutually exclusive.")
    if not np.array_equal(strict_reliable | ambiguous_l | ambiguous_g | suspicious, candidate_mask):
        raise RuntimeError("Dual-evidence strata must cover the full training pool.")
    if np.any(strict_reliable & ~actual_active_mask):
        raise RuntimeError("Actual active subset dropped a strict L/G intersection sample.")

    fallback = actual_active_mask & ~strict_reliable
    ambiguous_for_consistency = ambiguous & ~actual_active_mask
    if np.any(ambiguous_for_consistency & actual_active_mask):
        raise RuntimeError("An Ambiguous consistency sample also belongs to the active supervised subset.")
    return DualEvidenceStrata(
        strict_reliable=strict_reliable,
        ambiguous_l=ambiguous_l,
        ambiguous_g=ambiguous_g,
        ambiguous=ambiguous,
        suspicious=suspicious,
        actual_active=actual_active_mask,
        fallback=fallback,
        ambiguous_for_consistency=ambiguous_for_consistency,
    )


def compute_soft_consistency_loss(torch: Any, weak_logits: Any, strong_logits: Any) -> Any:
    """KL(softmax(stopgrad(weak)) || softmax(strong)) in stable float32."""
    if weak_logits.shape != strong_logits.shape:
        raise ValueError("Weak and strong logits must have identical shapes for consistency learning.")
    target = torch.softmax(weak_logits.float(), dim=-1).detach()
    if not bool(torch.isfinite(target).all().item()) or not bool(torch.isfinite(strong_logits).all().item()):
        raise FloatingPointError("Weak or strong logits are NaN or Inf.")
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(strong_logits.float(), dim=-1),
        target,
        reduction="batchmean",
    )
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Consistency KL is NaN or Inf.")
    return loss


def forward_ambiguous_consistency(
    torch: Any,
    model: Any,
    weak_images: Any,
    strong_images: Any,
    *,
    amp_enabled: bool,
) -> tuple[Any, Any, Any]:
    """Return label-free weak-target/strong-prediction consistency tensors.

    The weak branch temporarily enters eval mode and runs without autograd, so
    it cannot retain a gradient graph or update normalization state.  The
    caller remains responsible for backpropagating the returned KL term.
    """
    was_training = bool(model.training)
    try:
        model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                weak_logits = model(weak_images)
            weak_prob = torch.softmax(weak_logits.float(), dim=-1).detach()
    finally:
        model.train(was_training)
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        strong_logits = model(strong_images)
    consistency_loss = compute_soft_consistency_loss(torch, weak_logits, strong_logits)
    return consistency_loss, weak_prob, strong_logits


def count_empty_reliable_classes(
    reliable_mask: np.ndarray,
    candidate_mask: np.ndarray,
    labels: np.ndarray,
) -> int:
    """Count observed-label classes with no current Reliable training sample."""
    reliable_mask = np.asarray(reliable_mask, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    labels = np.asarray(labels).astype(str)
    if reliable_mask.shape != candidate_mask.shape or labels.shape != candidate_mask.shape:
        raise ValueError("Reliable-class count inputs must have matching shapes.")
    return sum(
        not np.any(reliable_mask[candidate_mask & (labels == label)])
        for label in sorted(set(labels[candidate_mask].tolist()))
    )


def count_empty_selected_classes(
    selected_mask: np.ndarray,
    candidate_mask: np.ndarray,
    labels: np.ndarray,
) -> int:
    """Count observed-label classes with no active training sample."""
    selected_mask = np.asarray(selected_mask, dtype=bool)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    labels = np.asarray(labels).astype(str)
    if selected_mask.shape != candidate_mask.shape or labels.shape != candidate_mask.shape:
        raise ValueError("Selected-class count inputs must have matching shapes.")
    return sum(
        not np.any(selected_mask[candidate_mask & (labels == label)])
        for label in sorted(set(labels[candidate_mask].tolist()))
    )


def trainable_model_checksum(model: Any) -> str:
    """Hash LoRA/head state only; frozen DINOv2 tensors need not be copied."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if "lora_" not in name and not name.startswith("head."):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_lora_backbone_invariants(model: Any) -> None:
    """Fail closed if a base DINOv2 tensor becomes trainable after LoRA injection."""
    lora_trainable = 0
    for name, parameter in model.backbone.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            if parameter.requires_grad:
                lora_trainable += parameter.numel()
            continue
        if parameter.requires_grad:
            raise RuntimeError(f"Frozen DINOv2 backbone parameter became trainable: backbone.{name}")
    if lora_trainable <= 0:
        raise RuntimeError("No trainable LoRA parameters were found in the DINOv2 backbone.")
    if not any(parameter.requires_grad for parameter in model.head.parameters()):
        raise RuntimeError("Classifier head must remain trainable.")


def select_small_loss_classwise(losses: np.ndarray, labels: np.ndarray, candidate_mask: np.ndarray, retention_ratio: float) -> np.ndarray:
    selected = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        if len(idx) == 0:
            continue
        if np.any(np.isnan(losses[idx])):
            raise ValueError(f"Missing loss values for class {label}.")
        keep = len(idx) if retention_ratio >= 1.0 else max(1, int(math.floor(len(idx) * retention_ratio)))
        order = np.argsort(losses[idx], kind="mergesort")
        selected[idx[order[:keep]]] = True
    return selected


def validate_class_budget_schedule(
    schedule: dict[int, dict[str, int]],
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    expected_update_epochs: list[int],
) -> None:
    found_epochs = sorted(int(epoch) for epoch in schedule)
    if found_epochs != list(expected_update_epochs):
        raise ValueError(
            f"class_budget_schedule epochs {found_epochs} do not match expected updates "
            f"{expected_update_epochs}."
        )
    for epoch in expected_update_epochs:
        validate_class_budgets(labels, candidate_mask, schedule[epoch], epoch=epoch)


def validate_class_budgets(
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    class_budgets: dict[str, int],
    *,
    epoch: int | None = None,
) -> None:
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    expected_classes = sorted(set(labels[candidate_mask].tolist()))
    normalized = {str(label): int(count) for label, count in class_budgets.items()}
    if set(normalized) != set(expected_classes):
        missing = sorted(set(expected_classes) - set(normalized))
        extra = sorted(set(normalized) - set(expected_classes))
        raise ValueError(
            f"Class budget keys do not match noisy-label classes at epoch {epoch}: "
            f"missing={missing}, extra={extra}."
        )
    for label in expected_classes:
        total = int(np.sum(candidate_mask & (labels == label)))
        keep = normalized[label]
        if not 1 <= keep <= total:
            raise ValueError(
                f"Invalid class budget at epoch {epoch}, class {label!r}: "
                f"keep={keep}, available={total}."
            )


def select_small_loss_classwise_by_budget(
    losses: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    class_budgets: dict[str, int],
) -> np.ndarray:
    """Select exactly the supplied count in each noisy-label class by loss only."""
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    validate_class_budgets(labels, candidate_mask, class_budgets)
    normalized = {str(label): int(count) for label, count in class_budgets.items()}
    selected = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        if np.any(np.isnan(losses[idx])):
            raise ValueError(f"Missing loss values for class {label}.")
        # Stable sorting makes equal-loss ties reproducible by original index.
        order = np.argsort(losses[idx], kind="mergesort")
        selected[idx[order[: normalized[label]]]] = True
    return selected


def select_top_proto_classwise(proto_scores: np.ndarray, labels: np.ndarray, candidate_mask: np.ndarray, proto_keep_ratio: float) -> np.ndarray:
    """Select class-wise high-prototype-score samples. Higher prototype score is safer."""
    selected = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        if len(idx) == 0:
            continue
        if np.any(np.isnan(proto_scores[idx])):
            raise ValueError(f"Missing prototype scores for class {label}.")
        keep = len(idx) if proto_keep_ratio >= 1.0 else max(1, int(math.floor(len(idx) * proto_keep_ratio)))
        order = np.argsort(-proto_scores[idx], kind="mergesort")
        selected[idx[order[:keep]]] = True
    return selected


def combine_loss_and_proto_classwise(
    loss_selected_mask: np.ndarray,
    proto_pass_mask: np.ndarray,
    losses: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """Intersect loss and prototype gates while preventing empty classes."""
    selected = loss_selected_mask & proto_pass_mask
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        if len(idx) == 0 or np.any(selected[idx]):
            continue
        fallback_idx = idx[proto_pass_mask[idx]]
        if len(fallback_idx) == 0:
            fallback_idx = idx
        best = fallback_idx[np.argmin(losses[fallback_idx])]
        selected[int(best)] = True
    return selected


def count_classwise_intersection_fallbacks(
    loss_selected_mask: np.ndarray,
    proto_pass_mask: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
) -> int:
    """Count classes that will require PGDF's existing prototype-constrained fallback."""
    count = 0
    intersection = np.asarray(loss_selected_mask, dtype=bool) & np.asarray(proto_pass_mask, dtype=bool)
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        if len(idx) and not np.any(intersection[idx]):
            count += 1
    return count


def build_update_row(
    method: str,
    dataset: str,
    seed: int,
    retention_ratio: float,
    proto_keep_ratio: float | None,
    epoch: int,
    candidate_mask: np.ndarray,
    labels: np.ndarray,
    selected_mask: np.ndarray,
    previous_mask: np.ndarray,
    centroid_mask: np.ndarray | None,
    losses: np.ndarray | None,
    loss_selected_mask: np.ndarray | None = None,
    proto_pass_mask: np.ndarray | None = None,
    proto_scores: np.ndarray | None = None,
    auto_proto_jaccard: float | None = None,
    selection_strategy: str = "loss_only",
    prototype_mode: str = "fixed",
    prototype_snapshot: DynamicPrototypeSnapshot | None = None,
    selection_model_checksum: str = "",
    lora_updated_since_initial: bool = False,
    fallback_count: int = 0,
    geometry_mode: str = "prototype_similarity",
    neighbor_margin_use_fallback: bool = False,
    neighbor_margin_positive_only: bool = True,
    neighbor_margin_snapshot: NeighborMarginSnapshot | None = None,
    reliable_mask: np.ndarray | None = None,
    ambiguous_mask: np.ndarray | None = None,
    suspicious_mask: np.ndarray | None = None,
    dual_evidence_strata: DualEvidenceStrata | None = None,
) -> dict[str, Any]:
    selected_losses = losses[selected_mask] if losses is not None else np.asarray([], dtype=np.float32)
    unselected_mask = candidate_mask & ~selected_mask
    unselected_losses = losses[unselected_mask] if losses is not None else np.asarray([], dtype=np.float32)
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    proto_rejected_mask = (
        loss_selected_mask & ~selected_mask
        if loss_selected_mask is not None
        else np.zeros(len(candidate_mask), dtype=bool)
    )
    if geometry_mode == "neighbor_margin":
        if (
            neighbor_margin_snapshot is None
            or reliable_mask is None
            or ambiguous_mask is None
            or suspicious_mask is None
        ):
            raise RuntimeError("Neighbor-Margin update logging requires all three sample-state masks.")
        reliable_mask = np.asarray(reliable_mask, dtype=bool)
        ambiguous_mask = np.asarray(ambiguous_mask, dtype=bool)
        suspicious_mask = np.asarray(suspicious_mask, dtype=bool)
        margin = neighbor_margin_snapshot.margin
        observed_similarity = neighbor_margin_snapshot.observed_similarity
        competitor_similarity = neighbor_margin_snapshot.competitor_similarity
    else:
        margin = None
        observed_similarity = None
        competitor_similarity = None
    reliable_zero_class_count: int | str = ""
    selected_zero_class_count: int | str = ""
    strict_intersection_count: int | str = ""
    selected_negative_margin_count: int | str = ""
    selected_negative_margin_ratio: float | str = ""
    if geometry_mode == "neighbor_margin":
        reliable_zero_class_count = count_empty_reliable_classes(
            reliable_mask,
            candidate_mask,
            labels,
        )
        selected_zero_class_count = count_empty_selected_classes(
            selected_mask,
            candidate_mask,
            labels,
        )
        if loss_selected_mask is None:
            raise RuntimeError("Neighbor-Margin update logging requires a small-loss mask.")
        strict_intersection_count = int(np.sum(loss_selected_mask & proto_pass_mask))
        selected_negative_margin_count = int(np.sum(selected_mask & (margin < 0.0)))
        selected_negative_margin_ratio = safe_ratio(
            int(selected_negative_margin_count),
            int(selected_mask.sum()),
        )
    return {
        "method": method,
        "dataset": dataset,
        "seed": int(seed),
        "retention_ratio": "" if selection_strategy == "proto_only" else float(retention_ratio),
        "proto_keep_ratio": proto_keep_ratio if proto_keep_ratio is not None else "",
        "auto_proto_jaccard": auto_proto_jaccard if auto_proto_jaccard is not None else "",
        "selection_strategy": selection_strategy,
        "prototype_mode": prototype_mode,
        "geometry_mode": geometry_mode,
        "neighbor_margin_use_fallback": "yes" if neighbor_margin_use_fallback else "no",
        "neighbor_margin_positive_only": "yes" if neighbor_margin_positive_only else "no",
        "epoch": int(epoch),
        "num_candidates": int(candidate_mask.sum()),
        "full_training_pool_size": int(candidate_mask.sum()),
        "num_loss_selected": int(loss_selected_mask.sum()) if loss_selected_mask is not None else "",
        "num_proto_pass": int(proto_pass_mask.sum()),
        "num_neighbor_margin_candidate": int(proto_pass_mask.sum()) if geometry_mode == "neighbor_margin" else "",
        "strict_intersection_count": strict_intersection_count,
        "num_selected": int(selected_mask.sum()),
        "fallback_count": int(fallback_count),
        "selected_negative_margin_count": selected_negative_margin_count,
        "selected_negative_margin_ratio": selected_negative_margin_ratio,
        "proto_reject_count": int(proto_rejected_mask.sum()),
        "selected_ratio": safe_ratio(int(selected_mask.sum()), int(candidate_mask.sum())),
        "mean_loss_selected": float(np.nanmean(selected_losses)) if selected_losses.size else "",
        "mean_loss_unselected": float(np.nanmean(unselected_losses)) if unselected_losses.size else "",
        "mean_loss_proto_rejected": (
            float(np.nanmean(losses[proto_rejected_mask])) if losses is not None and np.any(proto_rejected_mask) else ""
        ),
        "mean_proto_selected": float(np.nanmean(proto_scores[selected_mask])) if proto_scores is not None and np.any(selected_mask) else "",
        "mean_proto_unselected": float(np.nanmean(proto_scores[unselected_mask])) if proto_scores is not None and np.any(unselected_mask) else "",
        "overlap_with_previous_selection": mask_jaccard(selected_mask, previous_mask),
        "overlap_with_centroid": mask_jaccard(selected_mask, centroid_mask) if centroid_mask is not None else "",
        "prototype_feature_source": "current_lora_adapted_cls_before_head" if prototype_snapshot is not None else "frozen_cached_cls" if proto_scores is not None else "",
        "prototype_feature_dim": prototype_snapshot.feature_dim if prototype_snapshot is not None else "",
        "prototype_checksum": prototype_snapshot.prototype_checksum if prototype_snapshot is not None else "",
        "prototype_gate_membership_hash": prototype_snapshot.gate_membership_hash if prototype_snapshot is not None else "",
        "prototype_mean": prototype_snapshot.prototype_mean if prototype_snapshot is not None else "",
        "prototype_std": prototype_snapshot.prototype_std if prototype_snapshot is not None else "",
        "selection_model_state_checksum": selection_model_checksum,
        "prototype_model_state_checksum": prototype_snapshot.model_state_checksum if prototype_snapshot is not None else "",
        "same_model_state_for_loss_and_prototype": (
            "yes"
            if selection_strategy == "loss_and_proto"
            and prototype_snapshot is not None
            and prototype_snapshot.model_state_checksum == selection_model_checksum
            else ""
        ),
        "lora_updated_since_initial": "yes" if lora_updated_since_initial else "no",
        "reliable_count": int(reliable_mask.sum()) if reliable_mask is not None else "",
        "ambiguous_count": int(ambiguous_mask.sum()) if ambiguous_mask is not None else "",
        "suspicious_count": int(suspicious_mask.sum()) if suspicious_mask is not None else "",
        "reliable_ratio": safe_ratio(int(reliable_mask.sum()), int(candidate_mask.sum())) if reliable_mask is not None else "",
        "ambiguous_ratio": safe_ratio(int(ambiguous_mask.sum()), int(candidate_mask.sum())) if ambiguous_mask is not None else "",
        "suspicious_ratio": safe_ratio(int(suspicious_mask.sum()), int(candidate_mask.sum())) if suspicious_mask is not None else "",
        "reliable_zero_class_count": reliable_zero_class_count,
        "selected_zero_class_count": selected_zero_class_count,
        "margin_mean": float(np.mean(margin[candidate_mask])) if margin is not None else "",
        "margin_std": float(np.std(margin[candidate_mask])) if margin is not None else "",
        "margin_min": float(np.min(margin[candidate_mask])) if margin is not None else "",
        "margin_max": float(np.max(margin[candidate_mask])) if margin is not None else "",
        "margin_median": float(np.median(margin[candidate_mask])) if margin is not None else "",
        "mean_observed_similarity": float(np.mean(observed_similarity[candidate_mask])) if observed_similarity is not None else "",
        "mean_competitor_similarity": float(np.mean(competitor_similarity[candidate_mask])) if competitor_similarity is not None else "",
        # L/G dual-evidence diagnostics are intentionally additive.  The
        # historical Neighbor-Margin state fields above retain their original
        # meaning for strict-mode and pre-consistency comparisons.
        "dual_evidence_reliable_count": int(dual_evidence_strata.strict_reliable.sum()) if dual_evidence_strata is not None else "",
        "ambiguous_l_count": int(dual_evidence_strata.ambiguous_l.sum()) if dual_evidence_strata is not None else "",
        "ambiguous_g_count": int(dual_evidence_strata.ambiguous_g.sum()) if dual_evidence_strata is not None else "",
        "ambiguous_total_count": int(dual_evidence_strata.ambiguous.sum()) if dual_evidence_strata is not None else "",
        "dual_evidence_suspicious_count": int(dual_evidence_strata.suspicious.sum()) if dual_evidence_strata is not None else "",
        "actual_active_count": int(dual_evidence_strata.actual_active.sum()) if dual_evidence_strata is not None else "",
        "fallback_sample_count": int(dual_evidence_strata.fallback.sum()) if dual_evidence_strata is not None else "",
        "ambiguous_excluded_active_count": (
            int((dual_evidence_strata.ambiguous & dual_evidence_strata.actual_active).sum())
            if dual_evidence_strata is not None
            else ""
        ),
        "ambiguous_for_consistency_count": (
            int(dual_evidence_strata.ambiguous_for_consistency.sum())
            if dual_evidence_strata is not None
            else ""
        ),
    }


def build_selection_rows(
    method: str,
    dataset: str,
    seed: int,
    retention_ratio: float,
    proto_keep_ratio: float | None,
    epoch: int,
    paths: list[str],
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    losses: np.ndarray | None,
    confidence: np.ndarray | None,
    loss_selected_mask: np.ndarray | None = None,
    proto_pass_mask: np.ndarray | None = None,
    proto_scores: np.ndarray | None = None,
    selection_strategy: str = "loss_only",
    prototype_mode: str = "fixed",
    geometry_mode: str = "prototype_similarity",
    neighbor_margin_positive_only: bool = True,
    neighbor_margin_snapshot: NeighborMarginSnapshot | None = None,
    reliable_mask: np.ndarray | None = None,
    ambiguous_mask: np.ndarray | None = None,
    suspicious_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = []
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    if geometry_mode == "neighbor_margin" and (
        neighbor_margin_snapshot is None
        or reliable_mask is None
        or ambiguous_mask is None
        or suspicious_mask is None
    ):
        raise RuntimeError("Neighbor-Margin selection rows require margin diagnostics and all sample states.")
    for idx in np.where(candidate_mask)[0]:
        if geometry_mode == "neighbor_margin":
            if reliable_mask[int(idx)]:
                state = "reliable"
            elif ambiguous_mask[int(idx)]:
                state = "ambiguous"
            elif suspicious_mask[int(idx)]:
                state = "suspicious"
            else:
                raise RuntimeError(f"Candidate index {int(idx)} has no Neighbor-Margin state.")
            margin = float(neighbor_margin_snapshot.margin[int(idx)])
            observed_similarity = float(neighbor_margin_snapshot.observed_similarity[int(idx)])
            competitor_similarity = float(neighbor_margin_snapshot.competitor_similarity[int(idx)])
            competitor_class = str(neighbor_margin_snapshot.competitor_class[int(idx)])
            geometry_candidate = "yes" if proto_pass_mask[int(idx)] else "no"
            neighbor_margin_candidate = geometry_candidate
        else:
            state = "clean" if selected_mask[int(idx)] else "ignored"
            margin = ""
            observed_similarity = ""
            competitor_similarity = ""
            competitor_class = ""
            geometry_candidate = "yes" if proto_pass_mask[int(idx)] else "no"
            neighbor_margin_candidate = ""
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "seed": int(seed),
                "retention_ratio": "" if selection_strategy == "proto_only" else float(retention_ratio),
                "proto_keep_ratio": proto_keep_ratio if proto_keep_ratio is not None else "",
                "selection_strategy": selection_strategy,
                "prototype_mode": prototype_mode,
                "geometry_mode": geometry_mode,
                "neighbor_margin_positive_only": "yes" if neighbor_margin_positive_only else "no",
                "epoch": int(epoch),
                "index": int(idx),
                "path": paths[int(idx)],
                "web_label": str(labels[int(idx)]),
                "loss": float(losses[int(idx)]) if losses is not None else "",
                "confidence": float(confidence[int(idx)]) if confidence is not None else "",
                "proto_score": float(proto_scores[int(idx)]) if proto_scores is not None else "",
                "loss_selected": "yes" if loss_selected_mask is not None and loss_selected_mask[int(idx)] else "no" if loss_selected_mask is not None else "",
                "proto_pass": "yes" if proto_pass_mask[int(idx)] else "no",
                "geometry_candidate": geometry_candidate,
                "neighbor_margin_candidate": neighbor_margin_candidate,
                "neighbor_margin": margin,
                "observed_similarity": observed_similarity,
                "competitor_similarity": competitor_similarity,
                "competitor_class": competitor_class,
                "active_training": "yes" if selected_mask[int(idx)] else "no",
                "state": state,
            }
        )
    return rows


def build_per_class_rows(
    method: str,
    dataset: str,
    seed: int,
    retention_ratio: float,
    proto_keep_ratio: float | None,
    epoch: int,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    losses: np.ndarray | None,
    loss_selected_mask: np.ndarray | None = None,
    proto_pass_mask: np.ndarray | None = None,
    proto_scores: np.ndarray | None = None,
    selection_strategy: str = "loss_only",
    prototype_mode: str = "fixed",
    geometry_mode: str = "prototype_similarity",
    neighbor_margin_positive_only: bool = True,
    neighbor_margin_snapshot: NeighborMarginSnapshot | None = None,
    reliable_mask: np.ndarray | None = None,
    ambiguous_mask: np.ndarray | None = None,
    suspicious_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = []
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    if geometry_mode == "neighbor_margin" and (
        neighbor_margin_snapshot is None
        or reliable_mask is None
        or ambiguous_mask is None
        or suspicious_mask is None
    ):
        raise RuntimeError("Neighbor-Margin per-class rows require margin diagnostics and all sample states.")
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        selected_idx = idx[selected_mask[idx]]
        unselected_idx = idx[~selected_mask[idx]]
        proto_rejected_idx = (
            idx[loss_selected_mask[idx] & ~selected_mask[idx]]
            if loss_selected_mask is not None
            else np.asarray([], dtype=np.int64)
        )
        class_margin = neighbor_margin_snapshot.margin[idx] if neighbor_margin_snapshot is not None else None
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "seed": int(seed),
                "retention_ratio": "" if selection_strategy == "proto_only" else float(retention_ratio),
                "proto_keep_ratio": proto_keep_ratio if proto_keep_ratio is not None else "",
                "selection_strategy": selection_strategy,
                "prototype_mode": prototype_mode,
                "geometry_mode": geometry_mode,
                "neighbor_margin_positive_only": "yes" if neighbor_margin_positive_only else "no",
                "epoch": int(epoch),
                "web_label": str(label),
                "total_count": int(len(idx)),
                "loss_selected_count": int(np.sum(loss_selected_mask[idx])) if loss_selected_mask is not None else "",
                "proto_pass_count": int(np.sum(proto_pass_mask[idx])),
                "selected_count": int(len(selected_idx)),
                "proto_reject_count": int(len(proto_rejected_idx)),
                "selected_ratio": safe_ratio(len(selected_idx), len(idx)),
                "mean_loss_selected": float(np.nanmean(losses[selected_idx])) if losses is not None and len(selected_idx) else "",
                "mean_loss_unselected": float(np.nanmean(losses[unselected_idx])) if losses is not None and len(unselected_idx) else "",
                "mean_loss_proto_rejected": float(np.nanmean(losses[proto_rejected_idx])) if losses is not None and len(proto_rejected_idx) else "",
                "mean_proto_selected": float(np.nanmean(proto_scores[selected_idx])) if proto_scores is not None and len(selected_idx) else "",
                "mean_proto_unselected": float(np.nanmean(proto_scores[unselected_idx])) if proto_scores is not None and len(unselected_idx) else "",
                "neighbor_margin_candidate_count": int(np.sum(proto_pass_mask[idx])) if geometry_mode == "neighbor_margin" else "",
                "reliable_count": int(np.sum(reliable_mask[idx])) if reliable_mask is not None else "",
                "ambiguous_count": int(np.sum(ambiguous_mask[idx])) if ambiguous_mask is not None else "",
                "suspicious_count": int(np.sum(suspicious_mask[idx])) if suspicious_mask is not None else "",
                "reliable_zero": "yes" if reliable_mask is not None and not np.any(reliable_mask[idx]) else "no" if reliable_mask is not None else "",
                "mean_neighbor_margin": float(np.mean(class_margin)) if class_margin is not None else "",
                "mean_observed_similarity": (
                    float(np.mean(neighbor_margin_snapshot.observed_similarity[idx]))
                    if neighbor_margin_snapshot is not None
                    else ""
                ),
                "mean_competitor_similarity": (
                    float(np.mean(neighbor_margin_snapshot.competitor_similarity[idx]))
                    if neighbor_margin_snapshot is not None
                    else ""
                ),
            }
        )
    return rows


def mask_jaccard(left: np.ndarray, right: np.ndarray | None) -> float:
    if right is None:
        return 0.0
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = left | right
    if not np.any(union):
        return 1.0
    return safe_ratio(int(np.sum(left & right)), int(np.sum(union)))
