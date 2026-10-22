from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .fusion_ablation import FUSION_MODES, FusionSelection, select_budget_matched_fusion, soft_weighted_mean
from .features import resolve_device
from .lora_training import (
    DINOv2LoRAClassifier,
    ImageSplitDataset,
    build_scheduler,
    build_transforms,
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
from revision_experiments.profiling import RunProfiler
from revision_experiments.selection_logging import SelectionLogger
from revision_experiments.training_integration import (
    SelectionRunMetadata,
    profiling_enabled,
    profiling_memory_phase,
    profiling_stage,
    record_optimizer_and_scheduler_step,
    record_selection_update,
)


BUDGET_MATCHED_PROTO_STRATEGY = "proto_budget_matched"
BUDGET_MATCHED_RANDOM_STRATEGY = "random_budget_matched"
RANDOM_BUDGET_SELECTION_SCHEME = "sha256-v1(method, dataset, training_seed, update_epoch, observed_label, original_index)"


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
    fusion_update_rows: list[dict[str, Any]] = field(default_factory=list)
    fusion_sample_rows: list[dict[str, Any]] = field(default_factory=list)
    fusion_per_class_rows: list[dict[str, Any]] = field(default_factory=list)


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
    replay_selection_masks: dict[int, np.ndarray] | None = None,
    scheduler_retention_ratio: float | None = None,
    official_test_selected_only: bool = False,
    selection_strategy: str = "auto",
    prototype_mode: str = "fixed",
    fusion_mode: str | None = None,
    profiler: RunProfiler | None = None,
    selection_logger: SelectionLogger | None = None,
    selection_log_metadata: SelectionRunMetadata | None = None,
) -> DynamicLossRunResult:
    """Train DINOv2-LoRA with periodically updated class-wise small-loss selection.

    With an explicit test split, ``eval_paths`` is reserved for validation-only
    checkpoint selection and test metrics are evaluated after fitting.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    if selection_logger is not None:
        if selection_log_metadata is None:
            raise ValueError("selection_log_metadata is required when selection_logger is enabled")
        if selection_log_metadata.training_seed != int(seed):
            raise ValueError("selection_log_metadata.training_seed must equal the training seed")

    # Preserve the original public call pattern: callers that supplied static
    # prototype scores previously received PGDF intersection selection without
    # an extra strategy argument.
    if selection_strategy == "auto":
        selection_strategy = (
            "loss_and_proto"
            if proto_scores is not None or auto_proto_keep is not None or prototype_mode == "dynamic_lora"
            else "loss_only"
        )
    fusion_strategy_modes = {
        "fusion_weighted": "weighted_ranking",
        "fusion_union": "union_relaxed",
        "fusion_soft": "soft_weighting",
    }
    if selection_strategy in fusion_strategy_modes:
        expected_fusion_mode = fusion_strategy_modes[selection_strategy]
        if fusion_mode != expected_fusion_mode:
            raise ValueError(
                f"{selection_strategy} requires fusion_mode={expected_fusion_mode!r}, got {fusion_mode!r}."
            )
    elif fusion_mode is not None:
        raise ValueError("fusion_mode is only valid for an explicit fusion selection strategy.")
    if fusion_mode is not None and fusion_mode not in FUSION_MODES:
        raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}.")
    is_fusion_strategy = selection_strategy in fusion_strategy_modes
    is_budget_matched_proto = selection_strategy == BUDGET_MATCHED_PROTO_STRATEGY
    is_budget_matched_random = selection_strategy == BUDGET_MATCHED_RANDOM_STRATEGY
    is_fixed_replay = selection_strategy == "fixed_replay"

    validate_dynamic_args(
        retention_ratio,
        warmup_epochs,
        update_interval,
        proto_keep_ratio,
        auto_proto_keep,
        selection_strategy=selection_strategy,
        prototype_mode=prototype_mode,
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
    if is_fixed_replay:
        if replay_selection_masks is None:
            raise ValueError("fixed_replay requires replay_selection_masks.")
        if class_budget_schedule is not None:
            raise ValueError("fixed_replay cannot be combined with class_budget_schedule.")
        if centroid_mask is not None or proto_scores is not None or auto_proto_keep is not None:
            raise ValueError("fixed_replay cannot receive live selection inputs.")
    elif replay_selection_masks is not None:
        raise ValueError("replay_selection_masks is only valid with selection_strategy='fixed_replay'.")
    elif class_budget_schedule is not None:
        if selection_strategy in fusion_strategy_modes:
            if centroid_mask is not None or auto_proto_keep is not None:
                raise ValueError(
                    "Budget-matched fusion forbids centroid/auto-prototype routing."
                )
        elif is_budget_matched_proto:
            if centroid_mask is not None or auto_proto_keep is not None:
                raise ValueError(
                    "Budget-matched prototype-only selection forbids centroid/auto-prototype routing."
                )
        elif any(value is not None for value in (centroid_mask, proto_scores, proto_keep_ratio, auto_proto_keep)):
            raise ValueError("class_budget_schedule cannot be combined with prototype, centroid, or graph selection inputs.")
    elif is_fusion_strategy:
        raise ValueError("Budget-matched fusion requires a validated PGDF class_budget_schedule.")
    elif is_budget_matched_proto or is_budget_matched_random:
        raise ValueError(f"{selection_strategy} requires a validated PGDF class_budget_schedule.")
    if selection_strategy == "proto_only" and class_budget_schedule is not None:
        raise ValueError("proto_only selection cannot use a dynamic small-loss class budget.")
    if prototype_mode == "dynamic_lora":
        if selection_strategy not in {
            "proto_only",
            "loss_and_proto",
            BUDGET_MATCHED_PROTO_STRATEGY,
            *fusion_strategy_modes,
        }:
            raise ValueError("dynamic_lora prototypes require a prototype-based selection strategy.")
        if proto_scores is not None:
            raise ValueError("dynamic_lora prototypes must not receive frozen prototype scores.")
        if auto_proto_keep is not None:
            raise ValueError("dynamic_lora prototypes require a pre-declared fixed proto_keep_ratio.")
        if proto_keep_ratio is None:
            raise ValueError("dynamic_lora prototypes require proto_keep_ratio.")
    elif prototype_mode == "fixed" and selection_strategy in {
        "proto_only",
        "loss_and_proto",
        BUDGET_MATCHED_PROTO_STRATEGY,
        *fusion_strategy_modes,
    } and proto_scores is None:
        raise ValueError("fixed prototype selection requires proto_scores.")

    lora_cfg = cfg["lora"]
    train_cfg = cfg["lora_train"]
    feature_cfg = cfg["feature"]
    dataset_name = str(cfg.get("dataset", {}).get("name", ""))
    device = resolve_device(torch, feature_cfg.get("device", "auto"))
    if profiling_enabled(profiler):
        # The runner may receive device="auto"; record the actual resolved
        # device without changing the training choice or CUDA state.
        profiler.device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    total_training_token = (
        profiler.start_stage("total_training", phase="train_dynamic_loss_lora")
        if profiling_enabled(profiler)
        else None
    )
    set_torch_seed(torch, seed)

    classes = sorted(set(train_labels.tolist()))
    label_to_id = {label: i for i, label in enumerate(classes)}
    eval_known = np.array([label in label_to_id for label in eval_labels], dtype=bool)
    eval_idx = np.where(eval_known)[0]
    if len(eval_idx) == 0:
        raise ValueError("Eval split has no labels that appear in the train split.")

    input_size = int(feature_cfg["input_size"])
    train_transform, eval_transform = build_transforms(transforms, input_size)
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
    if is_fixed_replay:
        assert replay_selection_masks is not None
        validate_replay_selection_masks(
            replay_selection_masks,
            candidate_mask,
            expected_budget_epochs,
        )
        if scheduler_retention_ratio is None:
            raise ValueError(
                "fixed_replay requires scheduler_retention_ratio so its LR schedule is pre-declared."
            )
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
    use_soft_weighting = fusion_mode == "soft_weighting"
    criterion = torch.nn.CrossEntropyLoss(reduction="none" if use_soft_weighting else "mean")
    logs: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    fusion_update_rows: list[dict[str, Any]] = []
    fusion_sample_rows: list[dict[str, Any]] = []
    fusion_per_class_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    last5_states: list[tuple[int, dict[str, Any]]] = []
    oracle_states: list[tuple[int, dict[str, Any]]] = []
    selected_proto_keep_ratio = proto_keep_ratio
    auto_proto_jaccard: float | None = None
    optimizer_steps_since_selection = 0
    optimizer_steps_attempted_since_selection = 0
    optimizer_steps_attempted_total = 0
    optimizer_steps_actual_total = 0
    scheduler_steps_actual_total = 0

    selected_mask = candidate_mask.copy()
    soft_sample_weights = np.zeros(len(train_labels), dtype=np.float64)
    soft_sample_weights[candidate_mask] = 1.0
    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)
    initial_trainable_checksum = trainable_model_checksum(model)
    selection_split_name = "validation" if test_loader is not None else "eval"
    log_stage(
        f"[dynamic-loss] {method} seed={seed}: candidates={int(candidate_mask.sum())}, "
        f"{selection_split_name}_images={len(eval_idx)}, retention_ratio="
        f"{'not_used' if selection_strategy == 'proto_only' else f'{retention_ratio:.3f}'}, "
        f"proto_keep_ratio={format_optional_ratio(proto_keep_ratio)}, "
        f"selection_strategy={selection_strategy}, prototype_mode={prototype_mode}, fusion_mode={fusion_mode or 'none'}, "
        f"auto_proto_keep={'yes' if auto_proto_keep is not None else 'no'}, trainable_params={trainable_params}"
    )

    fit_time_token = (
        profiler.start_stage("fit_time", phase="epoch_loop") if profiling_enabled(profiler) else None
    )
    for epoch in range(1, epochs + 1):
        epoch_token = (
            profiler.start_stage("epoch", phase=f"epoch_{epoch}") if profiling_enabled(profiler) else None
        )
        epoch_train_idx = np.where(selected_mask)[0]
        train_dataset = ImageSplitDataset(train_paths, train_labels, epoch_train_idx, label_to_id, train_transform, path_maps)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            drop_last=False,
        )

        loss_sum = 0.0
        seen = 0.0
        with profiling_memory_phase(profiler, "training"):
            with profiling_stage(
                profiler,
                "optimizer_training",
                phase=f"epoch_{epoch}",
                data_loading_included=True,
            ):
                model.train()
                progress = progress_iter(train_loader, total=len(train_loader), desc=f"Dynamic LoRA {method} seed={seed} epoch {epoch}/{epochs}")
                for images, labels, sample_indices in progress:
                    if profiling_enabled(profiler):
                        profiler.record_training_batch()
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    batch_weights = None
                    if use_soft_weighting:
                        batch_weight_values = soft_sample_weights[sample_indices.detach().cpu().numpy()]
                        batch_weights = torch.as_tensor(batch_weight_values, device=device, dtype=torch.float32)
                    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                        logits = model(images)
                        per_sample_loss = criterion(logits, labels)
                        loss = (
                            soft_weighted_mean(per_sample_loss, batch_weights.to(dtype=per_sample_loss.dtype))
                            if batch_weights is not None
                            else per_sample_loss
                        )
                    scaler.scale(loss).backward()
                    # Exact-budget controls share PGDF's scheduler horizon but
                    # must also expose their realized update counts.  This is
                    # a record-only change: optimizer and scheduler calls are
                    # unchanged for every existing method.
                    track_step_counts = (
                        class_budget_schedule is not None
                        or selection_strategy in fusion_strategy_modes
                        or is_fixed_replay
                        or profiling_enabled(profiler)
                    )
                    scale_before = (
                        float(scaler.get_scale())
                        if track_step_counts and scaler.is_enabled()
                        else None
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    if track_step_counts:
                        scale_after = float(scaler.get_scale()) if scaler.is_enabled() else None
                        actual_update = record_optimizer_and_scheduler_step(
                            profiler,
                            epoch=epoch,
                            amp_enabled=bool(scaler.is_enabled()),
                            scale_before=scale_before,
                            scale_after=scale_after,
                        )
                        optimizer_steps_attempted_total += 1
                        optimizer_steps_actual_total += int(actual_update)
                        scheduler_steps_actual_total += 1
                        optimizer_steps_attempted_since_selection += 1
                        optimizer_steps_since_selection += int(actual_update)
                    current_batch = int(images.shape[0])
                    if batch_weights is None:
                        loss_sum += float(loss.detach().cpu()) * current_batch
                        seen += current_batch
                    else:
                        batch_weight_sum = float(batch_weights.detach().sum().cpu())
                        loss_sum += float(loss.detach().cpu()) * batch_weight_sum
                        seen += batch_weight_sum

        with profiling_stage(profiler, "validation", phase=f"epoch_{epoch}", data_loading_included=True):
            top1, top5 = evaluate_lora(torch, model, eval_loader, device, len(classes), bool(train_cfg.get("amp", True)))
        row = {
            "method": method,
            "seed": int(seed),
            "epoch": int(epoch),
            "lr_lora": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
            "loss": safe_ratio(loss_sum, seen),
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
            selection_update_token = (
                profiler.start_stage("selection_update_total", phase=f"epoch_{epoch}")
                if profiling_enabled(profiler)
                else None
            )
            previous_mask = selected_mask.copy()
            selection_model_checksum = trainable_model_checksum(model)
            losses: np.ndarray | None = None
            confidence: np.ndarray | None = None
            loss_selected_mask: np.ndarray | None = None
            proto_pass_mask = None
            current_proto_scores: np.ndarray | None = proto_scores
            prototype_snapshot: DynamicPrototypeSnapshot | None = None
            fallback_count = 0
            fusion_selection: FusionSelection | None = None

            # Dynamic Prototype Gate Only intentionally never runs this block:
            # no CE ranking is computed or used for its selection decision.
            if selection_strategy in {"loss_only", "loss_and_proto", *fusion_strategy_modes}:
                with profiling_memory_phase(profiler, "selection"):
                    with profiling_stage(
                        profiler,
                        "selection_loss_forward",
                        phase=f"epoch_{epoch}",
                        data_loading_included=True,
                    ):
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
                with profiling_memory_phase(profiler, "selection"):
                    with profiling_stage(profiler, "fusion_selection", phase=f"epoch_{epoch}"):
                        if class_budget_schedule is not None and not is_fusion_strategy:
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

            if selection_strategy in {
                "proto_only",
                "loss_and_proto",
                BUDGET_MATCHED_PROTO_STRATEGY,
                *fusion_strategy_modes,
            }:
                if prototype_mode == "dynamic_lora":
                    with profiling_memory_phase(profiler, "selection"):
                        with profiling_stage(
                            profiler,
                            "selection_cls_forward",
                            phase=f"epoch_{epoch}",
                            data_loading_included=True,
                        ):
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
                    with profiling_memory_phase(profiler, "selection"):
                        with profiling_stage(profiler, "prototype_construction", phase=f"epoch_{epoch}"):
                            prototype_snapshot = build_dynamic_prototype_snapshot(
                                dynamic_features,
                                train_labels,
                                candidate_mask,
                                float(selected_proto_keep_ratio),
                                selection_model_checksum,
                            )
                    current_proto_scores = prototype_snapshot.scores
                    proto_pass_mask = prototype_snapshot.gate_mask
                else:
                    with profiling_memory_phase(profiler, "selection"):
                        with profiling_stage(profiler, "fusion_selection", phase=f"epoch_{epoch}"):
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

            with profiling_memory_phase(profiler, "selection"):
                with profiling_stage(profiler, "fusion_selection", phase=f"epoch_{epoch}"):
                    if is_fixed_replay:
                        assert replay_selection_masks is not None
                        selected_mask = np.asarray(replay_selection_masks[epoch], dtype=bool).copy()
                    elif selection_strategy == "loss_only":
                        if loss_selected_mask is None:
                            raise RuntimeError("loss_only selection did not produce a small-loss mask.")
                        selected_mask = loss_selected_mask
                    elif selection_strategy == "proto_only":
                        if proto_pass_mask is None:
                            raise RuntimeError("proto_only selection did not produce a prototype gate.")
                        selected_mask = proto_pass_mask
                    elif selection_strategy == BUDGET_MATCHED_PROTO_STRATEGY:
                        if class_budget_schedule is None or current_proto_scores is None:
                            raise RuntimeError(
                                "Budget-matched prototype-only selection requires prototype scores and reference budgets."
                            )
                        selected_mask = select_top_proto_classwise_by_budget(
                            current_proto_scores,
                            train_labels,
                            candidate_mask,
                            class_budget_schedule[epoch],
                        )
                    elif selection_strategy == BUDGET_MATCHED_RANDOM_STRATEGY:
                        if class_budget_schedule is None:
                            raise RuntimeError("Budget-matched random selection requires reference budgets.")
                        selected_mask = select_random_classwise_by_budget(
                            train_labels,
                            candidate_mask,
                            class_budget_schedule[epoch],
                            dataset=dataset_name,
                            training_seed=seed,
                            epoch=epoch,
                        )
                    elif selection_strategy == "loss_and_proto":
                        if loss_selected_mask is None or proto_pass_mask is None or losses is None:
                            raise RuntimeError("loss_and_proto selection requires loss and prototype masks.")
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
                    elif is_fusion_strategy:
                        if (
                            fusion_mode is None
                            or class_budget_schedule is None
                            or losses is None
                            or current_proto_scores is None
                            or loss_selected_mask is None
                            or proto_pass_mask is None
                        ):
                            raise RuntimeError("Budget-matched fusion requires loss/prototype scores, gates, and reference budgets.")
                        fusion_selection = select_budget_matched_fusion(
                            mode=fusion_mode,
                            losses=losses,
                            prototype_scores=current_proto_scores,
                            labels=train_labels,
                            candidate_mask=candidate_mask,
                            loss_candidate_mask=loss_selected_mask,
                            prototype_candidate_mask=proto_pass_mask,
                            reference_budgets=class_budget_schedule[epoch],
                            epoch=epoch,
                        )
                        selected_mask = fusion_selection.final_selected_mask
                        if fusion_selection.sample_weights is not None:
                            soft_sample_weights = fusion_selection.sample_weights
                    else:
                        raise RuntimeError(f"Unsupported selection_strategy: {selection_strategy}")
            if selection_logger is not None:
                assert selection_log_metadata is not None
                record_selection_update(
                    selection_logger,
                    train_paths=train_paths,
                    observed_labels=train_labels,
                    candidate_mask=candidate_mask,
                    losses=losses,
                    prototype_scores=current_proto_scores,
                    loss_candidate_mask=loss_selected_mask,
                    prototype_candidate_mask=proto_pass_mask,
                    final_selected_mask=selected_mask,
                    epoch=epoch,
                    selection_strategy=selection_strategy,
                    model_state_identifier=selection_model_checksum,
                    metadata=selection_log_metadata,
                )
            # Keep the audit unambiguous: Random deliberately has no
            # prototype gate, so do not serialize the default all-candidate
            # placeholder as if it were a prototype decision.
            audit_proto_pass_mask = (
                np.zeros(len(candidate_mask), dtype=bool)
                if selection_strategy == BUDGET_MATCHED_RANDOM_STRATEGY
                else proto_pass_mask
            )
            if profiling_enabled(profiler):
                profiler.record_selection_interval_steps(epoch, optimizer_steps_since_selection)
            if fusion_selection is not None:
                fusion_update = {
                    "method": method,
                    "dataset": dataset_name,
                    "seed": int(seed),
                    **fusion_selection.update_row,
                    "optimizer_steps_attempted_since_previous_update": int(optimizer_steps_attempted_since_selection),
                    "optimizer_steps_actual_since_previous_update": int(optimizer_steps_since_selection),
                    "scheduler_steps_since_previous_update": int(optimizer_steps_attempted_since_selection),
                }
                fusion_update_rows.append(fusion_update)
                fusion_sample_rows.extend(
                    {
                        "method": method,
                        "dataset": dataset_name,
                        "seed": int(seed),
                        **sample_row,
                    }
                    for sample_row in fusion_selection.sample_rows
                )
                fusion_per_class_rows.extend(
                    {
                        "method": method,
                        "dataset": dataset_name,
                        "seed": int(seed),
                        **class_row,
                    }
                    for class_row in fusion_selection.per_class_rows
                )
            update_rows.append(
                build_update_row(
                    method,
                    dataset_name,
                    seed,
                    retention_ratio,
                    selected_proto_keep_ratio,
                    epoch,
                    candidate_mask,
                    selected_mask,
                    previous_mask,
                    centroid_mask,
                    losses,
                    loss_selected_mask=loss_selected_mask,
                    proto_pass_mask=audit_proto_pass_mask,
                    proto_scores=current_proto_scores,
                    auto_proto_jaccard=auto_proto_jaccard,
                    selection_strategy=selection_strategy,
                    prototype_mode=prototype_mode,
                    prototype_snapshot=prototype_snapshot,
                    selection_model_checksum=selection_model_checksum,
                    lora_updated_since_initial=(selection_model_checksum != initial_trainable_checksum),
                    fallback_count=fallback_count,
                    optimizer_steps_attempted_since_previous_update=optimizer_steps_attempted_since_selection,
                    optimizer_steps_actual_since_previous_update=optimizer_steps_since_selection,
                    scheduler_steps_since_previous_update=optimizer_steps_attempted_since_selection,
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
                    proto_pass_mask=audit_proto_pass_mask,
                    proto_scores=current_proto_scores,
                    selection_strategy=selection_strategy,
                    prototype_mode=prototype_mode,
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
                    proto_pass_mask=audit_proto_pass_mask,
                    proto_scores=current_proto_scores,
                    selection_strategy=selection_strategy,
                    prototype_mode=prototype_mode,
                )
            )
            log_stage(
                f"[dynamic-loss] selection update epoch={epoch}: "
                f"selected={int(selected_mask.sum())}, proto_mode={prototype_mode}, "
                f"prev_jaccard={update_rows[-1]['overlap_with_previous_selection']:.4f}"
            )
            optimizer_steps_since_selection = 0
            optimizer_steps_attempted_since_selection = 0
            if selection_update_token is not None:
                profiler.stop_stage(selection_update_token)

        if epoch_token is not None:
            profiler.stop_stage(epoch_token)

    if fit_time_token is not None:
        profiler.stop_stage(fit_time_token)
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
    checkpoint_save_token = (
        profiler.start_stage("checkpoint_save", phase="training_checkpoints")
        if profiling_enabled(profiler)
        and (checkpoint_path is not None or final_checkpoint_path is not None or last5_checkpoint_dir is not None)
        else None
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
    if checkpoint_save_token is not None:
        profiler.stop_stage(checkpoint_save_token)

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
            "scheduler_estimated_total_steps": int(total_steps),
            "scheduler_warmup_steps": int(warmup_steps),
            "candidate_samples": int(candidate_mask.sum()),
            "final_selected_samples": int(selected_mask.sum()),
            "selection_updates": len(update_rows),
            "selection_strategy": selection_strategy,
            "prototype_mode": prototype_mode,
            "fusion_mode": fusion_mode or "",
            "initial_trainable_checksum": initial_trainable_checksum,
            "optimizer_steps_attempted": (
                optimizer_steps_attempted_total
                if class_budget_schedule is not None or is_fusion_strategy or is_fixed_replay
                else ""
            ),
            "optimizer_steps_actual": (
                optimizer_steps_actual_total
                if class_budget_schedule is not None or is_fusion_strategy or is_fixed_replay
                else ""
            ),
            "scheduler_steps_actual": (
                scheduler_steps_actual_total
                if class_budget_schedule is not None or is_fusion_strategy or is_fixed_replay
                else ""
            ),
            "soft_weighting_effective_budget": (
                fusion_update_rows[-1].get("effective_weight_budget", "")
                if fusion_mode == "soft_weighting" and fusion_update_rows
                else ""
            ),
            "backbone_frozen": "yes",
            "lora_updated_before_selection": (
                "yes" if any(row.get("lora_updated_since_initial") == "yes" for row in update_rows) else "no"
            ),
        }
    )
    result = DynamicLossRunResult(
        logs=logs,
        summary=summary,
        trainable_modules=trainable_modules,
        trainable_params=trainable_params,
        total_params=total_params,
        selection_rows=selection_rows,
        update_rows=update_rows,
        per_class_rows=per_class_rows,
        fusion_update_rows=fusion_update_rows,
        fusion_sample_rows=fusion_sample_rows,
        fusion_per_class_rows=fusion_per_class_rows,
    )
    if total_training_token is not None:
        profiler.stop_stage(total_training_token)
    return result


def validate_dynamic_args(
    retention_ratio: float,
    warmup_epochs: int,
    update_interval: int,
    proto_keep_ratio: float | None = None,
    auto_proto_keep: dict[str, float] | None = None,
    *,
    selection_strategy: str = "loss_only",
    prototype_mode: str = "fixed",
) -> None:
    if not 0.0 < retention_ratio <= 1.0:
        raise ValueError("retention_ratio must satisfy 0 < ratio <= 1.")
    if proto_keep_ratio is not None and not 0.0 < proto_keep_ratio <= 1.0:
        raise ValueError("proto_keep_ratio must satisfy 0 < ratio <= 1.")
    if proto_keep_ratio is not None and auto_proto_keep is not None:
        raise ValueError("Use either proto_keep_ratio or auto_proto_keep, not both.")
    if auto_proto_keep is not None:
        validate_auto_proto_keep(auto_proto_keep)
    allowed_strategies = {
        "loss_only",
        "proto_only",
        "loss_and_proto",
        BUDGET_MATCHED_PROTO_STRATEGY,
        BUDGET_MATCHED_RANDOM_STRATEGY,
        "fixed_replay",
        "fusion_weighted",
        "fusion_union",
        "fusion_soft",
    }
    if selection_strategy not in allowed_strategies:
        raise ValueError(f"selection_strategy must be one of {sorted(allowed_strategies)}.")
    if prototype_mode not in {"fixed", "dynamic_lora"}:
        raise ValueError("prototype_mode must be fixed or dynamic_lora.")
    if selection_strategy in {
        "proto_only",
        "loss_and_proto",
        BUDGET_MATCHED_PROTO_STRATEGY,
        "fusion_weighted",
        "fusion_union",
        "fusion_soft",
    } and proto_keep_ratio is None:
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


def validate_replay_selection_masks(
    replay_selection_masks: dict[int, np.ndarray],
    candidate_mask: np.ndarray,
    expected_update_epochs: list[int],
) -> None:
    """Fail closed on a fixed replay trajectory before any model is created.

    The masks are full-original-training-pool boolean arrays.  They are applied
    after each historic update and therefore must cover precisely the update
    epochs that have a successor; no current-model selection signal is read.
    """
    expected = set(int(epoch) for epoch in expected_update_epochs)
    found = set(int(epoch) for epoch in replay_selection_masks)
    if found != expected:
        raise ValueError(
            f"Replay mask epochs {sorted(found)} do not match expected updates {sorted(expected)}."
        )
    candidate = np.asarray(candidate_mask, dtype=bool)
    for epoch in sorted(expected):
        mask = np.asarray(replay_selection_masks[epoch], dtype=bool)
        if mask.shape != candidate.shape:
            raise ValueError(
                f"Replay mask at epoch {epoch} has shape {mask.shape}; expected {candidate.shape}."
            )
        if np.any(mask & ~candidate):
            raise ValueError(f"Replay mask at epoch {epoch} selects samples outside the candidate pool.")
        if not np.any(mask):
            raise ValueError(f"Replay mask at epoch {epoch} is empty.")


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
) -> DynamicPrototypeSnapshot:
    """Rebuild observed-label prototypes and a top-p gate from current CLS features."""
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
        keep = len(idx) if proto_keep_ratio >= 1.0 else max(1, int(math.floor(proto_keep_ratio * len(idx))))
        order = np.argsort(-class_scores, kind="mergesort")
        gate_mask[idx[order[:keep]]] = True
        class_retained_counts[str(label)] = int(keep)
        prototype_values.append(prototype.astype(np.float32, copy=False))
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


def select_top_proto_classwise_by_budget(
    proto_scores: np.ndarray,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    class_budgets: dict[str, int],
) -> np.ndarray:
    """Select exactly the supplied count in each observed-label class by prototype score."""
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    proto_scores = np.asarray(proto_scores, dtype=np.float64)
    if proto_scores.shape != labels.shape:
        raise ValueError("proto_scores must align with labels.")
    validate_class_budgets(labels, candidate_mask, class_budgets)
    normalized = {str(label): int(count) for label, count in class_budgets.items()}
    selected = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        if np.any(~np.isfinite(proto_scores[idx])):
            raise ValueError(f"Missing prototype scores for class {label}.")
        # Stable sorting makes equal-score ties reproducible by original index.
        order = np.argsort(-proto_scores[idx], kind="mergesort")
        selected[idx[order[: normalized[label]]]] = True
    return selected


def select_random_classwise_by_budget(
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    class_budgets: dict[str, int],
    *,
    dataset: str,
    training_seed: int,
    epoch: int,
) -> np.ndarray:
    """Select an exact observed-class budget with a stateless SHA-256 random order.

    The order is a deterministic function of the fixed method identifier,
    dataset, training seed, selection update, observed label, and original
    pool index.  It is therefore stable across Python/NumPy versions without
    reusing the strict PGDF sample identities or accessing oracle labels.
    """
    labels = np.asarray(labels).astype(str)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    validate_class_budgets(labels, candidate_mask, class_budgets)
    normalized = {str(label): int(count) for label, count in class_budgets.items()}
    selected = np.zeros(len(labels), dtype=bool)
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        ordered = sorted(
            idx.tolist(),
            key=lambda index: (
                hashlib.sha256(
                    f"pgdf_exact_budget_random|{dataset}|{int(training_seed)}|{int(epoch)}|{label}|{int(index)}".encode(
                        "utf-8"
                    )
                ).digest(),
                int(index),
            ),
        )
        selected[np.asarray(ordered[: normalized[label]], dtype=np.int64)] = True
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
    optimizer_steps_attempted_since_previous_update: int | str = "",
    optimizer_steps_actual_since_previous_update: int | str = "",
    scheduler_steps_since_previous_update: int | str = "",
) -> dict[str, Any]:
    if np.any(np.asarray(selected_mask, dtype=bool) & ~np.asarray(candidate_mask, dtype=bool)):
        raise RuntimeError("Selection contains samples outside the training candidate pool.")
    selected_losses = losses[selected_mask] if losses is not None else np.asarray([], dtype=np.float32)
    unselected_mask = candidate_mask & ~selected_mask
    unselected_losses = losses[unselected_mask] if losses is not None else np.asarray([], dtype=np.float32)
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    proto_rejected_mask = (
        loss_selected_mask & ~selected_mask
        if loss_selected_mask is not None
        else np.zeros(len(candidate_mask), dtype=bool)
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
        "epoch": int(epoch),
        "num_candidates": int(candidate_mask.sum()),
        "full_training_pool_size": int(candidate_mask.sum()),
        "num_loss_selected": int(loss_selected_mask.sum()) if loss_selected_mask is not None else "",
        "num_proto_pass": int(proto_pass_mask.sum()),
        "num_selected": int(selected_mask.sum()),
        "selected_identity_sha256": selected_identity_sha256(selected_mask),
        "fallback_count": int(fallback_count),
        "optimizer_steps_attempted_since_previous_update": optimizer_steps_attempted_since_previous_update,
        "optimizer_steps_actual_since_previous_update": optimizer_steps_actual_since_previous_update,
        "scheduler_steps_since_previous_update": scheduler_steps_since_previous_update,
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
) -> list[dict[str, Any]]:
    rows = []
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    for idx in np.where(candidate_mask)[0]:
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "seed": int(seed),
                "retention_ratio": "" if selection_strategy == "proto_only" else float(retention_ratio),
                "proto_keep_ratio": proto_keep_ratio if proto_keep_ratio is not None else "",
                "selection_strategy": selection_strategy,
                "prototype_mode": prototype_mode,
                "epoch": int(epoch),
                "index": int(idx),
                "path": paths[int(idx)],
                "web_label": str(labels[int(idx)]),
                "loss": float(losses[int(idx)]) if losses is not None else "",
                "confidence": float(confidence[int(idx)]) if confidence is not None else "",
                "proto_score": float(proto_scores[int(idx)]) if proto_scores is not None else "",
                "loss_selected": "yes" if loss_selected_mask is not None and loss_selected_mask[int(idx)] else "no" if loss_selected_mask is not None else "",
                "proto_pass": "yes" if proto_pass_mask[int(idx)] else "no",
                "state": "clean" if selected_mask[int(idx)] else "ignored",
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
) -> list[dict[str, Any]]:
    rows = []
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        selected_idx = idx[selected_mask[idx]]
        unselected_idx = idx[~selected_mask[idx]]
        proto_rejected_idx = (
            idx[loss_selected_mask[idx] & ~selected_mask[idx]]
            if loss_selected_mask is not None
            else np.asarray([], dtype=np.int64)
        )
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "seed": int(seed),
                "retention_ratio": "" if selection_strategy == "proto_only" else float(retention_ratio),
                "proto_keep_ratio": proto_keep_ratio if proto_keep_ratio is not None else "",
                "selection_strategy": selection_strategy,
                "prototype_mode": prototype_mode,
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


def selected_identity_sha256(selected_mask: np.ndarray) -> str:
    """Hash selected original training-pool indices without exposing a source identity list."""
    indices = np.flatnonzero(np.asarray(selected_mask, dtype=bool)).astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(indices.tobytes())
    return digest.hexdigest()
