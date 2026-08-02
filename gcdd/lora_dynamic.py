from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
) -> DynamicLossRunResult:
    """Train DINOv2-LoRA with periodically updated class-wise small-loss selection.

    With an explicit test split, ``eval_paths`` is reserved for validation-only
    checkpoint selection and test metrics are evaluated after fitting.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    validate_dynamic_args(retention_ratio, warmup_epochs, update_interval, proto_keep_ratio, auto_proto_keep)
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

    lora_cfg = cfg["lora"]
    train_cfg = cfg["lora_train"]
    feature_cfg = cfg["feature"]
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
    effective_scheduler_ratio = (
        estimate_selection_retention_ratio(retention_ratio, proto_keep_ratio, auto_proto_keep)
        if scheduler_retention_ratio is None
        else float(scheduler_retention_ratio)
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
    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)
    selection_split_name = "validation" if test_loader is not None else "eval"
    log_stage(
        f"[dynamic-loss] {method} seed={seed}: candidates={int(candidate_mask.sum())}, "
        f"{selection_split_name}_images={len(eval_idx)}, retention_ratio={retention_ratio:.3f}, "
        f"proto_keep_ratio={format_optional_ratio(proto_keep_ratio)}, "
        f"auto_proto_keep={'yes' if auto_proto_keep is not None else 'no'}, trainable_params={trainable_params}"
    )

    for epoch in range(1, epochs + 1):
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

        model.train()
        loss_sum = 0.0
        seen = 0
        progress = progress_iter(train_loader, total=len(train_loader), desc=f"Dynamic LoRA {method} seed={seed} epoch {epoch}/{epochs}")
        for images, labels, _ in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            current_batch = int(images.shape[0])
            loss_sum += float(loss.detach().cpu()) * current_batch
            seen += current_batch

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
            losses, confidence = compute_train_losses(torch, model, loss_loader, device, len(train_labels), bool(train_cfg.get("amp", True)))
            previous_mask = selected_mask.copy()
            if class_budget_schedule is not None:
                loss_selected_mask = select_small_loss_classwise_by_budget(
                    losses,
                    train_labels,
                    candidate_mask,
                    class_budget_schedule[epoch],
                )
            else:
                loss_selected_mask = select_small_loss_classwise(losses, train_labels, candidate_mask, retention_ratio)
            proto_pass_mask = None
            if auto_proto_keep is not None and selected_proto_keep_ratio is None:
                auto_proto_jaccard = mask_jaccard(loss_selected_mask, centroid_mask)
                selected_proto_keep_ratio = choose_auto_proto_keep_ratio(auto_proto_jaccard, auto_proto_keep)
                log_stage(
                    f"[dynamic-loss] auto proto_keep_ratio selected: jaccard={auto_proto_jaccard:.4f}, "
                    f"p={selected_proto_keep_ratio:.3f}"
                )
            if proto_scores is not None and selected_proto_keep_ratio is not None:
                proto_pass_mask = select_top_proto_classwise(proto_scores, train_labels, candidate_mask, selected_proto_keep_ratio)
                selected_mask = combine_loss_and_proto_classwise(loss_selected_mask, proto_pass_mask, losses, train_labels, candidate_mask)
            else:
                selected_mask = loss_selected_mask
            update_rows.append(
                build_update_row(
                    method,
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
                    proto_pass_mask=proto_pass_mask,
                    proto_scores=proto_scores,
                    auto_proto_jaccard=auto_proto_jaccard,
                )
            )
            selection_rows.extend(
                build_selection_rows(
                    method,
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
                    proto_scores=proto_scores,
                )
            )
            per_class_rows.extend(
                build_per_class_rows(
                    method,
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
                    proto_scores=proto_scores,
                )
            )
            log_stage(
                f"[dynamic-loss] selection update epoch={epoch}: "
                f"selected={int(selected_mask.sum())}, prev_jaccard={update_rows[-1]['overlap_with_previous_selection']:.4f}"
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
                "retention_ratio": float(retention_ratio),
                "proto_keep_ratio": selected_proto_keep_ratio,
                "auto_proto_keep": auto_proto_keep is not None,
                "auto_proto_jaccard": auto_proto_jaccard,
                "warmup_epochs": int(warmup_epochs),
                "update_interval": int(update_interval),
                "budget_matched": class_budget_schedule is not None,
                "scheduler_retention_ratio": float(effective_scheduler_ratio),
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
                "retention_ratio": float(retention_ratio),
                "proto_keep_ratio": selected_proto_keep_ratio,
                "auto_proto_keep": auto_proto_keep is not None,
                "auto_proto_jaccard": auto_proto_jaccard,
                "warmup_epochs": int(warmup_epochs),
                "update_interval": int(update_interval),
                "budget_matched": class_budget_schedule is not None,
                "scheduler_retention_ratio": float(effective_scheduler_ratio),
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
            "retention_ratio": float(retention_ratio),
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
) -> None:
    if not 0.0 < retention_ratio <= 1.0:
        raise ValueError("retention_ratio must satisfy 0 < ratio <= 1.")
    if proto_keep_ratio is not None and not 0.0 < proto_keep_ratio <= 1.0:
        raise ValueError("proto_keep_ratio must satisfy 0 < ratio <= 1.")
    if proto_keep_ratio is not None and auto_proto_keep is not None:
        raise ValueError("Use either proto_keep_ratio or auto_proto_keep, not both.")
    if auto_proto_keep is not None:
        validate_auto_proto_keep(auto_proto_keep)
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
    model.eval()
    losses = np.full(total_train, np.nan, dtype=np.float32)
    confidence = np.full(total_train, np.nan, dtype=np.float32)
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
    return losses, confidence


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


def build_update_row(
    method: str,
    seed: int,
    retention_ratio: float,
    proto_keep_ratio: float | None,
    epoch: int,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    previous_mask: np.ndarray,
    centroid_mask: np.ndarray | None,
    losses: np.ndarray,
    loss_selected_mask: np.ndarray | None = None,
    proto_pass_mask: np.ndarray | None = None,
    proto_scores: np.ndarray | None = None,
    auto_proto_jaccard: float | None = None,
) -> dict[str, Any]:
    selected_losses = losses[selected_mask]
    unselected_mask = candidate_mask & ~selected_mask
    unselected_losses = losses[unselected_mask]
    loss_selected_mask = selected_mask if loss_selected_mask is None else loss_selected_mask
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    proto_rejected_mask = loss_selected_mask & ~selected_mask
    return {
        "method": method,
        "seed": int(seed),
        "retention_ratio": float(retention_ratio),
        "proto_keep_ratio": proto_keep_ratio if proto_keep_ratio is not None else "",
        "auto_proto_jaccard": auto_proto_jaccard if auto_proto_jaccard is not None else "",
        "epoch": int(epoch),
        "num_candidates": int(candidate_mask.sum()),
        "num_loss_selected": int(loss_selected_mask.sum()),
        "num_proto_pass": int(proto_pass_mask.sum()),
        "num_selected": int(selected_mask.sum()),
        "proto_reject_count": int(proto_rejected_mask.sum()),
        "selected_ratio": safe_ratio(int(selected_mask.sum()), int(candidate_mask.sum())),
        "mean_loss_selected": float(np.nanmean(selected_losses)) if selected_losses.size else 0.0,
        "mean_loss_unselected": float(np.nanmean(unselected_losses)) if unselected_losses.size else 0.0,
        "mean_loss_proto_rejected": float(np.nanmean(losses[proto_rejected_mask])) if np.any(proto_rejected_mask) else "",
        "mean_proto_selected": float(np.nanmean(proto_scores[selected_mask])) if proto_scores is not None and np.any(selected_mask) else "",
        "mean_proto_unselected": float(np.nanmean(proto_scores[unselected_mask])) if proto_scores is not None and np.any(unselected_mask) else "",
        "overlap_with_previous_selection": mask_jaccard(selected_mask, previous_mask),
        "overlap_with_centroid": mask_jaccard(selected_mask, centroid_mask) if centroid_mask is not None else "",
    }


def build_selection_rows(
    method: str,
    seed: int,
    retention_ratio: float,
    proto_keep_ratio: float | None,
    epoch: int,
    paths: list[str],
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    losses: np.ndarray,
    confidence: np.ndarray,
    loss_selected_mask: np.ndarray | None = None,
    proto_pass_mask: np.ndarray | None = None,
    proto_scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = []
    loss_selected_mask = selected_mask if loss_selected_mask is None else loss_selected_mask
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    for idx in np.where(candidate_mask)[0]:
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "retention_ratio": float(retention_ratio),
                "proto_keep_ratio": proto_keep_ratio if proto_keep_ratio is not None else "",
                "epoch": int(epoch),
                "index": int(idx),
                "path": paths[int(idx)],
                "web_label": str(labels[int(idx)]),
                "loss": float(losses[int(idx)]),
                "confidence": float(confidence[int(idx)]),
                "proto_score": float(proto_scores[int(idx)]) if proto_scores is not None else "",
                "loss_selected": "yes" if loss_selected_mask[int(idx)] else "no",
                "proto_pass": "yes" if proto_pass_mask[int(idx)] else "no",
                "state": "clean" if selected_mask[int(idx)] else "ignored",
            }
        )
    return rows


def build_per_class_rows(
    method: str,
    seed: int,
    retention_ratio: float,
    proto_keep_ratio: float | None,
    epoch: int,
    labels: np.ndarray,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    losses: np.ndarray,
    loss_selected_mask: np.ndarray | None = None,
    proto_pass_mask: np.ndarray | None = None,
    proto_scores: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows = []
    loss_selected_mask = selected_mask if loss_selected_mask is None else loss_selected_mask
    proto_pass_mask = candidate_mask if proto_pass_mask is None else proto_pass_mask
    for label in sorted(set(labels[candidate_mask].tolist())):
        idx = np.where(candidate_mask & (labels == label))[0]
        selected_idx = idx[selected_mask[idx]]
        unselected_idx = idx[~selected_mask[idx]]
        proto_rejected_idx = idx[loss_selected_mask[idx] & ~selected_mask[idx]]
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "retention_ratio": float(retention_ratio),
                "proto_keep_ratio": proto_keep_ratio if proto_keep_ratio is not None else "",
                "epoch": int(epoch),
                "web_label": str(label),
                "total_count": int(len(idx)),
                "loss_selected_count": int(np.sum(loss_selected_mask[idx])),
                "proto_pass_count": int(np.sum(proto_pass_mask[idx])),
                "selected_count": int(len(selected_idx)),
                "proto_reject_count": int(len(proto_rejected_idx)),
                "selected_ratio": safe_ratio(len(selected_idx), len(idx)),
                "mean_loss_selected": float(np.nanmean(losses[selected_idx])) if len(selected_idx) else 0.0,
                "mean_loss_unselected": float(np.nanmean(losses[unselected_idx])) if len(unselected_idx) else 0.0,
                "mean_loss_proto_rejected": float(np.nanmean(losses[proto_rejected_idx])) if len(proto_rejected_idx) else "",
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
