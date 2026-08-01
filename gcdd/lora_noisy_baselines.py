from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
    freeze_all,
    inject_lora,
    lora_parameters,
    parse_target_modules,
    safe_ratio,
    set_torch_seed,
    trainable_state_dict,
)
from .progress import log_stage, progress_iter


@dataclass
class NoisyBaselineRunResult:
    logs: list[dict[str, Any]]
    summary: dict[str, Any]
    trainable_modules: list[dict[str, Any]]
    trainable_params: int
    total_params: int


def get_remember_rate(
    epoch: int,
    warmup_epochs: int,
    total_epochs: int,
    *,
    mode: str = "fixed",
    fixed_rate: float = 0.8,
    final_rate: float = 0.6,
) -> float:
    if epoch <= warmup_epochs:
        return 1.0
    if mode == "fixed":
        return float(fixed_rate)
    if mode != "schedule":
        raise ValueError(f"Unsupported remember mode: {mode}")
    progress = (epoch - warmup_epochs - 1) / max(1, total_epochs - warmup_epochs - 1)
    return 1.0 - min(max(progress, 0.0), 1.0) * (1.0 - float(final_rate))


def select_small_loss_indices(torch: Any, loss_each: Any, remember_rate: float) -> Any:
    batch_size = int(loss_each.shape[0])
    keep = batch_size if remember_rate >= 1.0 else max(1, int(math.floor(batch_size * remember_rate)))
    return torch.argsort(loss_each.detach())[:keep]


def symmetric_kl_each(torch: Any, logits_a: Any, logits_b: Any) -> Any:
    log_p_a = torch.nn.functional.log_softmax(logits_a, dim=1)
    log_p_b = torch.nn.functional.log_softmax(logits_b, dim=1)
    p_a = log_p_a.exp()
    p_b = log_p_b.exp()
    kl_ab = (p_a * (log_p_a - log_p_b)).sum(dim=1)
    kl_ba = (p_b * (log_p_b - log_p_a)).sum(dim=1)
    return kl_ab + kl_ba


def train_coteaching_lora(
    train_paths: list[str],
    train_labels: np.ndarray,
    eval_paths: list[str],
    eval_labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: dict[str, Any],
    method: str,
    seed: int,
    remember_mode: str,
    remember_rate: float,
    final_remember_rate: float,
    warmup_epochs: int,
    grad_accum_steps: int = 1,
    path_maps: list[tuple[str, str]] | None = None,
    gt_clean_mask: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    test_paths: list[str] | None = None,
    test_labels: np.ndarray | None = None,
    final_checkpoint_path: Path | None = None,
    last5_checkpoint_dir: Path | None = None,
    checkpoint_protocol: str = "legacy_test_selected",
    posthoc_oracle_test: bool = False,
) -> NoisyBaselineRunResult:
    return train_dual_model_lora(
        train_paths,
        train_labels,
        eval_paths,
        eval_labels,
        train_mask,
        cfg,
        method=method,
        seed=seed,
        warmup_epochs=warmup_epochs,
        remember_mode=remember_mode,
        remember_rate=remember_rate,
        final_remember_rate=final_remember_rate,
        lambda_cor=0.0,
        grad_accum_steps=grad_accum_steps,
        path_maps=path_maps,
        gt_clean_mask=gt_clean_mask,
        checkpoint_path=checkpoint_path,
        test_paths=test_paths,
        test_labels=test_labels,
        final_checkpoint_path=final_checkpoint_path,
        last5_checkpoint_dir=last5_checkpoint_dir,
        checkpoint_protocol=checkpoint_protocol,
        posthoc_oracle_test=posthoc_oracle_test,
        batch_loss_fn=coteaching_batch_loss,
    )


def train_jocor_lora(
    train_paths: list[str],
    train_labels: np.ndarray,
    eval_paths: list[str],
    eval_labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: dict[str, Any],
    method: str,
    seed: int,
    remember_mode: str,
    remember_rate: float,
    final_remember_rate: float,
    warmup_epochs: int,
    lambda_cor: float,
    grad_accum_steps: int = 1,
    path_maps: list[tuple[str, str]] | None = None,
    gt_clean_mask: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    test_paths: list[str] | None = None,
    test_labels: np.ndarray | None = None,
    final_checkpoint_path: Path | None = None,
    last5_checkpoint_dir: Path | None = None,
    checkpoint_protocol: str = "legacy_test_selected",
    posthoc_oracle_test: bool = False,
) -> NoisyBaselineRunResult:
    return train_dual_model_lora(
        train_paths,
        train_labels,
        eval_paths,
        eval_labels,
        train_mask,
        cfg,
        method=method,
        seed=seed,
        warmup_epochs=warmup_epochs,
        remember_mode=remember_mode,
        remember_rate=remember_rate,
        final_remember_rate=final_remember_rate,
        lambda_cor=lambda_cor,
        grad_accum_steps=grad_accum_steps,
        path_maps=path_maps,
        gt_clean_mask=gt_clean_mask,
        checkpoint_path=checkpoint_path,
        test_paths=test_paths,
        test_labels=test_labels,
        final_checkpoint_path=final_checkpoint_path,
        last5_checkpoint_dir=last5_checkpoint_dir,
        checkpoint_protocol=checkpoint_protocol,
        posthoc_oracle_test=posthoc_oracle_test,
        batch_loss_fn=jocor_batch_loss,
    )


def train_dual_model_lora(
    train_paths: list[str],
    train_labels: np.ndarray,
    eval_paths: list[str],
    eval_labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: dict[str, Any],
    *,
    method: str,
    seed: int,
    warmup_epochs: int,
    remember_mode: str,
    remember_rate: float,
    final_remember_rate: float,
    lambda_cor: float,
    grad_accum_steps: int,
    path_maps: list[tuple[str, str]] | None,
    gt_clean_mask: np.ndarray | None,
    checkpoint_path: Path | None,
    test_paths: list[str] | None,
    test_labels: np.ndarray | None,
    final_checkpoint_path: Path | None,
    last5_checkpoint_dir: Path | None,
    checkpoint_protocol: str,
    posthoc_oracle_test: bool,
    batch_loss_fn: Callable[..., dict[str, Any]],
) -> NoisyBaselineRunResult:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    validate_dual_args(remember_rate, final_remember_rate, warmup_epochs, lambda_cor, grad_accum_steps)
    path_maps = path_maps or []
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape != (len(train_labels),):
        raise ValueError(f"train_mask must have shape ({len(train_labels)},), got {train_mask.shape}.")
    if gt_clean_mask is not None:
        gt_clean_mask = np.asarray(gt_clean_mask, dtype=bool)
        if gt_clean_mask.shape != train_mask.shape:
            raise ValueError("gt_clean_mask must match train_mask shape.")
    if (test_paths is None) != (test_labels is None):
        raise ValueError("test_paths and test_labels must be provided together.")

    lora_cfg = cfg["lora"]
    train_cfg = cfg["lora_train"]
    feature_cfg = cfg["feature"]
    device = resolve_device(torch, feature_cfg.get("device", "auto"))

    classes = sorted(set(train_labels.tolist()))
    label_to_id = {label: i for i, label in enumerate(classes)}
    eval_known = np.array([label in label_to_id for label in eval_labels], dtype=bool)
    train_idx = np.where(train_mask)[0]
    eval_idx = np.where(eval_known)[0]
    if len(train_idx) == 0:
        raise ValueError(f"{method} selected no training images.")
    if len(eval_idx) == 0:
        raise ValueError("Eval split has no labels that appear in the train split.")

    input_size = int(feature_cfg["input_size"])
    train_transform, eval_transform = build_transforms(transforms, input_size)
    train_dataset = ImageSplitDataset(train_paths, train_labels, train_idx, label_to_id, train_transform, path_maps)
    eval_dataset = ImageSplitDataset(eval_paths, eval_labels, eval_idx, label_to_id, eval_transform, path_maps)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        drop_last=False,
    )
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

    model_a, modules_a = build_lora_model(torch, cfg, len(classes), seed, device)
    model_b, modules_b = build_lora_model(torch, cfg, len(classes), seed + 1000, device)
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters(model_a), "lr": float(train_cfg["lora_lr"])},
            {"params": model_a.head.parameters(), "lr": float(train_cfg["head_lr"])},
            {"params": lora_parameters(model_b), "lr": float(train_cfg["lora_lr"])},
            {"params": model_b.head.parameters(), "lr": float(train_cfg["head_lr"])},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )

    epochs = int(train_cfg["epochs"])
    optimizer_steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_accum_steps)))
    total_steps = max(1, epochs * optimizer_steps_per_epoch)
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
    scheduler = build_scheduler(torch, optimizer, total_steps, warmup_steps, str(train_cfg.get("scheduler", "cosine")))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.startswith("cuda"))
    amp = bool(train_cfg.get("amp", True))

    trainable_params = count_trainable_params(model_a) + count_trainable_params(model_b)
    total_params = count_total_params(model_a) + count_total_params(model_b)
    logs: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    last5_states: list[tuple[int, dict[str, Any]]] = []
    oracle_states: list[tuple[int, dict[str, Any]]] = []
    log_stage(
        f"[{method}] seed={seed}: train_images={len(train_idx)}, eval_images={len(eval_idx)}, "
        f"remember_mode={remember_mode}, remember_rate={remember_rate:.3f}, trainable_params={trainable_params}"
    )

    for epoch in range(1, epochs + 1):
        model_a.train()
        model_b.train()
        optimizer.zero_grad(set_to_none=True)
        current_remember = get_remember_rate(
            epoch,
            warmup_epochs,
            epochs,
            mode=remember_mode,
            fixed_rate=remember_rate,
            final_rate=final_remember_rate,
        )
        loss_sum = 0.0
        seen = 0
        selected_sum = 0.0
        selected_clean_sum = 0.0
        optimizer_step_pending = 0
        progress = progress_iter(train_loader, total=len(train_loader), desc=f"{method} seed={seed} epoch {epoch}/{epochs}")
        for batch_pos, (images, labels, indices) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits_a = model_a(images)
                logits_b = model_b(images)
                out = batch_loss_fn(torch, logits_a, logits_b, labels, current_remember, lambda_cor)
                loss = out["loss"] / float(grad_accum_steps)
            scaler.scale(loss).backward()
            optimizer_step_pending += 1
            if optimizer_step_pending == grad_accum_steps or batch_pos == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_pending = 0

            batch_size = int(images.shape[0])
            loss_sum += float(out["loss"].detach().cpu()) * batch_size
            seen += batch_size
            selected_count, selected_clean = summarize_batch_selection(torch, out, indices, gt_clean_mask)
            selected_sum += selected_count
            selected_clean_sum += selected_clean

        top1_a, top5_a = evaluate_lora(torch, model_a, eval_loader, device, len(classes), amp)
        top1_b, top5_b = evaluate_lora(torch, model_b, eval_loader, device, len(classes), amp)
        mean_top1 = (float(top1_a) + float(top1_b)) / 2.0
        mean_top5 = (float(top5_a) + float(top5_b)) / 2.0
        row = {
            "method": method,
            "seed": int(seed),
            "epoch": int(epoch),
            "lr_lora": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
            "loss": safe_ratio(loss_sum, seen),
            "remember_rate": float(current_remember),
            "selected_count": float(selected_sum),
            "selected_ratio": safe_ratio(selected_sum, seen),
            "selected_clean": float(selected_clean_sum) if gt_clean_mask is not None else "",
            "selected_purity": safe_ratio(selected_clean_sum, selected_sum) if gt_clean_mask is not None else "",
            "clean_recall": safe_ratio(selected_clean_sum, int(gt_clean_mask[train_mask].sum())) if gt_clean_mask is not None else "",
            "top1_a": float(top1_a),
            "top5_a": float(top5_a),
            "top1_b": float(top1_b),
            "top5_b": float(top5_b),
            "mean_ab_top1": float(mean_top1),
            "mean_ab_top5": float(mean_top5),
            "top1": float(mean_top1),
            "top5": float(mean_top5),
            "val_top1": float(mean_top1),
            "val_top5": float(mean_top5),
            "train_samples": int(len(train_idx)),
            "eval_samples": int(len(eval_idx)),
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "selection_mode": "two_branch_mean_validation_top1",
        }
        logs.append(row)
        if select_best_dual_validation_row(logs) is row:
            best_row = row
            best_state = snapshot_dual_state(model_a, model_b)
        row["best_top1"] = float(best_row["mean_ab_top1"])
        row["best_epoch"] = int(best_row["epoch"])
        epoch_state = snapshot_dual_state(model_a, model_b)
        last5_states.append((epoch, epoch_state))
        if len(last5_states) > 5:
            last5_states.pop(0)
        if posthoc_oracle_test and test_loader is not None:
            oracle_states.append((epoch, epoch_state))
        log_stage(
            f"[{method}] seed={seed} epoch {epoch}/{epochs}: "
            f"loss={row['loss']:.4f}, mean_ab_top1={mean_top1:.4f}, selected_ratio={row['selected_ratio']:.4f}"
        )

    final_state = snapshot_dual_state(model_a, model_b)
    protocol_metrics: dict[str, Any] = {}
    if test_loader is not None and best_state is not None and best_row is not None:
        epoch_test_metrics: dict[int, dict[str, float]] = {}
        if posthoc_oracle_test:
            epoch_test_metrics = {
                epoch: evaluate_dual_state(
                    torch,
                    model_a,
                    model_b,
                    state,
                    test_loader,
                    device,
                    len(classes),
                    amp,
                )
                for epoch, state in oracle_states
            }
            selected_test = epoch_test_metrics[int(best_row["epoch"])]
            final_test = epoch_test_metrics[int(epochs)]
            last5_test = [epoch_test_metrics[epoch] for epoch, _ in last5_states]
        else:
            selected_test = evaluate_dual_state(
                torch, model_a, model_b, best_state, test_loader, device, len(classes), amp
            )
            final_test = evaluate_dual_state(
                torch, model_a, model_b, final_state, test_loader, device, len(classes), amp
            )
            last5_test = [
                evaluate_dual_state(torch, model_a, model_b, state, test_loader, device, len(classes), amp)
                for _, state in last5_states
            ]
        last5_top1 = np.asarray([row["mean_top1"] for row in last5_test], dtype=np.float32)
        last5_top5 = np.asarray([row["mean_top5"] for row in last5_test], dtype=np.float32)
        protocol_metrics = {
            "checkpoint_protocol": checkpoint_protocol,
            "validation_samples": int(len(eval_idx)),
            "test_samples": int(len(test_idx)),
            "best_val_epoch": int(best_row["epoch"]),
            "best_val_top1": float(best_row["mean_ab_top1"]),
            "best_val_top5": float(best_row["mean_ab_top5"]),
            "validation_selected_test_top1": float(selected_test["mean_top1"]),
            "validation_selected_test_top5": float(selected_test["mean_top5"]),
            "validation_selected_test_model_a_top1": float(selected_test["top1_a"]),
            "validation_selected_test_model_a_top5": float(selected_test["top5_a"]),
            "validation_selected_test_model_b_top1": float(selected_test["top1_b"]),
            "validation_selected_test_model_b_top5": float(selected_test["top5_b"]),
            "final_test_top1": float(final_test["mean_top1"]),
            "final_test_top5": float(final_test["mean_top5"]),
            "final_test_model_a_top1": float(final_test["top1_a"]),
            "final_test_model_a_top5": float(final_test["top5_a"]),
            "final_test_model_b_top1": float(final_test["top1_b"]),
            "final_test_model_b_top5": float(final_test["top5_b"]),
            "last5_test_mean": float(last5_top1.mean()),
            "last5_test_std": float(last5_top1.std()),
            "last5_test_top5_mean": float(last5_top5.mean()),
            "last5_test_top5_std": float(last5_top5.std()),
        }
        if posthoc_oracle_test:
            oracle_epoch, oracle_test = max(epoch_test_metrics.items(), key=lambda item: item[1]["mean_top1"])
            protocol_metrics.update(
                {
                    "oracle_best_test_epoch": int(oracle_epoch),
                    "oracle_best_test_top1": float(oracle_test["mean_top1"]),
                    "oracle_best_test_top5": float(oracle_test["mean_top5"]),
                    "oracle_best_to_final_drop": float(oracle_test["mean_top1"] - final_test["mean_top1"]),
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
                "best_mean_ab_top1": float(best_row["mean_ab_top1"]) if best_row else None,
                "remember_mode": remember_mode,
                "remember_rate": float(remember_rate),
                "final_remember_rate": float(final_remember_rate),
                "lambda_cor": float(lambda_cor),
                "selection_metric": "arithmetic mean of branch A/B validation Top-1; not ensemble prediction",
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
                "selection_metric": "arithmetic mean of branch A/B validation Top-1; not ensemble prediction",
                "checkpoint_protocol": checkpoint_protocol,
                **protocol_metrics,
            },
            final_checkpoint_path,
        )
    if last5_checkpoint_dir is not None:
        last5_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for epoch, state in last5_states:
            torch.save(
                {
                    "method": method,
                    "seed": int(seed),
                    "classes": classes,
                    "state_dict": state,
                    "epoch": int(epoch),
                    "checkpoint_protocol": checkpoint_protocol,
                },
                last5_checkpoint_dir / f"epoch_{epoch:03d}.pt",
            )

    summary = summarize_dual_logs(method, seed, logs, remember_mode, remember_rate, final_remember_rate, lambda_cor, warmup_epochs)
    summary.update(protocol_metrics)
    return NoisyBaselineRunResult(
        logs=logs,
        summary=summary,
        trainable_modules=build_module_rows(modules_a, modules_b),
        trainable_params=trainable_params,
        total_params=total_params,
    )


def build_lora_model(torch: Any, cfg: dict[str, Any], num_classes: int, seed: int, device: str) -> tuple[Any, list[str]]:
    set_torch_seed(torch, seed)
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
    for param in model.head.parameters():
        param.requires_grad_(True)
    return model, modules


def select_best_dual_validation_row(logs: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the first epoch maximizing branch-mean validation Top-1.

    Only ``mean_ab_top1`` is read.  In particular, official-test fields cannot
    influence checkpoint selection even if diagnostic rows contain them.
    """
    if not logs:
        raise ValueError("At least one validation log row is required.")
    return max(logs, key=lambda row: float(row["mean_ab_top1"]))


def snapshot_dual_state(model_a: Any, model_b: Any) -> dict[str, dict[str, Any]]:
    """Clone both trainable states so later CPU optimizer steps cannot mutate them."""
    return {
        "model_a": {name: value.clone() for name, value in trainable_state_dict(model_a).items()},
        "model_b": {name: value.clone() for name, value in trainable_state_dict(model_b).items()},
    }


def evaluate_dual_state(
    torch: Any,
    model_a: Any,
    model_b: Any,
    state: dict[str, dict[str, Any]],
    loader: Any,
    device: str,
    num_classes: int,
    amp: bool,
) -> dict[str, float]:
    """Evaluate branches separately and report arithmetic means, never an ensemble."""
    model_a.load_state_dict(state["model_a"], strict=False)
    model_b.load_state_dict(state["model_b"], strict=False)
    model_a.to(device)
    model_b.to(device)
    top1_a, top5_a = evaluate_lora(torch, model_a, loader, device, num_classes, amp)
    top1_b, top5_b = evaluate_lora(torch, model_b, loader, device, num_classes, amp)
    return {
        "top1_a": float(top1_a),
        "top5_a": float(top5_a),
        "top1_b": float(top1_b),
        "top5_b": float(top5_b),
        "mean_top1": (float(top1_a) + float(top1_b)) / 2.0,
        "mean_top5": (float(top5_a) + float(top5_b)) / 2.0,
    }


def coteaching_batch_loss(torch: Any, logits_a: Any, logits_b: Any, labels: Any, remember_rate: float, lambda_cor: float) -> dict[str, Any]:
    ce_a = torch.nn.functional.cross_entropy(logits_a, labels, reduction="none")
    ce_b = torch.nn.functional.cross_entropy(logits_b, labels, reduction="none")
    idx_a = select_small_loss_indices(torch, ce_a, remember_rate)
    idx_b = select_small_loss_indices(torch, ce_b, remember_rate)
    loss_a = ce_a[idx_b].mean()
    loss_b = ce_b[idx_a].mean()
    return {"loss": loss_a + loss_b, "idx_a": idx_a, "idx_b": idx_b}


def jocor_batch_loss(torch: Any, logits_a: Any, logits_b: Any, labels: Any, remember_rate: float, lambda_cor: float) -> dict[str, Any]:
    ce_a = torch.nn.functional.cross_entropy(logits_a, labels, reduction="none")
    ce_b = torch.nn.functional.cross_entropy(logits_b, labels, reduction="none")
    co_reg = symmetric_kl_each(torch, logits_a, logits_b)
    joint = ce_a + ce_b + float(lambda_cor) * co_reg
    idx = select_small_loss_indices(torch, joint, remember_rate)
    return {"loss": joint[idx].mean(), "idx": idx}


def summarize_batch_selection(torch: Any, out: dict[str, Any], indices: Any, gt_clean_mask: np.ndarray | None) -> tuple[float, float]:
    selected_keys = [key for key in ["idx", "idx_a", "idx_b"] if key in out]
    if not selected_keys:
        return 0.0, 0.0
    selected_count = float(sum(int(out[key].shape[0]) for key in selected_keys)) / float(len(selected_keys))
    if gt_clean_mask is None:
        return selected_count, 0.0
    idx_np = indices.detach().cpu().numpy().astype(np.int64)
    selected_clean = 0.0
    for key in selected_keys:
        local = out[key].detach().cpu().numpy().astype(np.int64)
        selected_clean += float(np.sum(gt_clean_mask[idx_np[local]]))
    selected_clean /= float(len(selected_keys))
    return selected_count, selected_clean


def summarize_dual_logs(
    method: str,
    seed: int,
    logs: list[dict[str, Any]],
    remember_mode: str,
    remember_rate: float,
    final_remember_rate: float,
    lambda_cor: float,
    warmup_epochs: int,
) -> dict[str, Any]:
    if not logs:
        raise ValueError(f"No logs available for {method}.")
    best = select_best_dual_validation_row(logs)
    final = logs[-1]
    last5 = logs[-5:]
    last5_mean = np.array([float(row["mean_ab_top1"]) for row in last5], dtype=np.float32)
    return {
        "method": method,
        "seed": int(seed),
        "remember_mode": remember_mode,
        "remember_rate": float(remember_rate),
        "final_remember_rate": float(final_remember_rate),
        "lambda_cor": float(lambda_cor),
        "warmup_epochs": int(warmup_epochs),
        "train_samples": int(final["train_samples"]),
        "eval_samples": int(final["eval_samples"]),
        "best_epoch": int(best["epoch"]),
        "best_mean_ab_top1": float(best["mean_ab_top1"]),
        "best_model_a_top1": float(best["top1_a"]),
        "best_model_b_top1": float(best["top1_b"]),
        "best_mean_ab_top5": float(best["mean_ab_top5"]),
        "final_mean_ab_top1": float(final["mean_ab_top1"]),
        "final_mean_ab_top5": float(final["mean_ab_top5"]),
        "final_model_a_top1": float(final["top1_a"]),
        "final_model_b_top1": float(final["top1_b"]),
        "last5_mean": float(last5_mean.mean()),
        "last5_std": float(last5_mean.std()),
        "final_selected_ratio": float(final["selected_ratio"]),
        "final_selected_count": float(final["selected_count"]),
        "final_selected_purity": final["selected_purity"],
        "final_clean_recall": final["clean_recall"],
        "trainable_params": int(final["trainable_params"]),
        "total_params": int(final["total_params"]),
        "selection_mode": "two_branch_mean_validation_top1",
    }


def build_module_rows(modules_a: list[str], modules_b: list[str]) -> list[dict[str, Any]]:
    return [{"model": "a", "module": name} for name in modules_a] + [{"model": "b", "module": name} for name in modules_b]


def validate_dual_args(remember_rate: float, final_remember_rate: float, warmup_epochs: int, lambda_cor: float, grad_accum_steps: int) -> None:
    for name, value in [("remember_rate", remember_rate), ("final_remember_rate", final_remember_rate)]:
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must satisfy 0 < value <= 1.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if lambda_cor < 0.0:
        raise ValueError("lambda_cor must be non-negative.")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1.")
