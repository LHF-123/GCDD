from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

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
class GMMReliabilityResult:
    gamma: np.ndarray
    success: bool
    reason: str
    component_means: tuple[float, float] | None


@dataclass
class SNSCLRunResult:
    logs: list[dict[str, Any]]
    summary: dict[str, Any]
    trainable_modules: list[str]
    trainable_params: int
    total_params: int
    reliability_rows: list[dict[str, Any]]
    reliability_summary_rows: list[dict[str, Any]]
    queue_rows: list[dict[str, Any]]


class SNSCLHealthError(RuntimeError):
    """Raised when a long-running SNSCL experiment is mechanically unhealthy."""


class ProjectionHead(torch.nn.Module):
    """Single-linear projection followed by L2 normalization."""

    def __init__(self, feat_dim: int, proj_dim: int = 512) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(feat_dim, proj_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(features), dim=1)


class StochasticHead(torch.nn.Module):
    """Predict a diagonal Gaussian and sample a normalized embedding."""

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int | None = None) -> None:
        super().__init__()
        output_dim = int(output_dim or input_dim)
        self.hidden = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(inplace=True))
        self.mu = torch.nn.Linear(hidden_dim, output_dim)
        self.logvar = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.hidden(features)
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(min=-20.0, max=10.0)
        sampled = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return F.normalize(sampled, dim=1), mu, logvar


class ClassWiseQueue(torch.nn.Module):
    """Class-wise FIFO embedding queue stored entirely as registered buffers."""

    def __init__(self, num_classes: int, queue_size: int, embedding_dim: int) -> None:
        super().__init__()
        if num_classes <= 0 or queue_size <= 0 or embedding_dim <= 0:
            raise ValueError("num_classes, queue_size, and embedding_dim must be positive.")
        self.num_classes = int(num_classes)
        self.queue_size = int(queue_size)
        self.embedding_dim = int(embedding_dim)
        self.register_buffer("features", torch.zeros(num_classes, queue_size, embedding_dim))
        self.register_buffer("valid", torch.zeros(num_classes, queue_size, dtype=torch.bool))
        self.register_buffer("pointers", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("counts", torch.zeros(num_classes, dtype=torch.long))

    @torch.no_grad()
    def enqueue(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        random_values: torch.Tensor | None = None,
    ) -> int:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"embeddings must have shape [N, {self.embedding_dim}].")
        if labels.shape != (embeddings.shape[0],) or weights.shape != (embeddings.shape[0],):
            raise ValueError("labels and weights must have shape [N].")
        random_values = torch.rand_like(weights) if random_values is None else random_values.to(weights.device)
        if random_values.shape != weights.shape:
            raise ValueError("random_values must match weights shape.")

        accepted = random_values < weights.clamp(0.0, 1.0)
        inserted = 0
        for embedding, label in zip(embeddings[accepted], labels[accepted]):
            class_id = int(label.item())
            if class_id < 0 or class_id >= self.num_classes:
                raise ValueError(f"Queue label {class_id} is outside [0, {self.num_classes}).")
            pointer = int(self.pointers[class_id].item())
            self.features[class_id, pointer].copy_(F.normalize(embedding.detach(), dim=0))
            self.valid[class_id, pointer] = True
            self.pointers[class_id] = (pointer + 1) % self.queue_size
            self.counts[class_id] = min(int(self.counts[class_id].item()) + 1, self.queue_size)
            inserted += 1
        return inserted

    def flattened(self) -> tuple[torch.Tensor, torch.Tensor]:
        class_ids = torch.arange(self.num_classes, device=self.features.device)[:, None].expand(-1, self.queue_size)
        return self.features[self.valid], class_ids[self.valid]

    def fill_ratio(self) -> float:
        return float(self.valid.float().mean().item())


def gaussian_kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence from a diagonal Gaussian to N(0, I)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()


def forward_stochastic_fp32(
    projection: ProjectionHead,
    stochastic: StochasticHead,
    cls_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the numerically sensitive SNSCL embedding branch in FP32."""
    with torch.autocast(device_type=cls_features.device.type, enabled=False):
        projected = projection(cls_features.float())
        sampled, mu, logvar = stochastic(projected)
    return projected, sampled, mu, logvar


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def paper_ntcl_loss(
    anchors: torch.Tensor,
    labels: torch.Tensor,
    queue: ClassWiseQueue,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute paper-style NTCL by averaging one log-ratio per positive."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    queue_features, queue_labels = queue.flattened()
    losses: list[torch.Tensor] = []
    positive_counts: list[int] = []
    negative_counts: list[int] = []
    if queue_features.numel():
        queue_features = F.normalize(queue_features, dim=1)
        anchors = F.normalize(anchors, dim=1)
        for anchor, label in zip(anchors, labels):
            positive_mask = queue_labels == label
            negative_mask = queue_labels != label
            positive_count = int(positive_mask.sum().item())
            negative_count = int(negative_mask.sum().item())
            if positive_count == 0 or negative_count == 0:
                continue
            logits = torch.matmul(queue_features, anchor) / float(temperature)
            denominator = torch.logsumexp(logits, dim=0)
            losses.append(-(logits[positive_mask] - denominator).mean())
            positive_counts.append(positive_count)
            negative_counts.append(negative_count)
    loss = torch.stack(losses).mean() if losses else anchors.sum() * 0.0
    valid = len(losses)
    return loss, {
        "num_valid_ntcl_anchors": float(valid),
        "valid_anchor_ratio": safe_ratio(valid, int(anchors.shape[0])),
        "mean_positive_count": float(np.mean(positive_counts)) if positive_counts else 0.0,
        "mean_negative_count": float(np.mean(negative_counts)) if negative_counts else 0.0,
    }


def fit_gmm_reliability(losses: np.ndarray, previous_gamma: np.ndarray | None = None, seed: int = 0) -> GMMReliabilityResult:
    """Fit a two-component GMM and return posterior probability of the low-loss component."""
    losses = np.asarray(losses, dtype=np.float64)
    fallback = np.asarray(previous_gamma, dtype=np.float32).copy() if previous_gamma is not None else np.ones(len(losses), dtype=np.float32)
    try:
        if losses.ndim != 1 or len(losses) < 2:
            raise ValueError("at least two one-dimensional losses are required")
        if not np.all(np.isfinite(losses)):
            raise ValueError("losses contain non-finite values")
        minimum = float(losses.min())
        span = float(losses.max() - minimum)
        if span <= 1.0e-12:
            raise ValueError("losses are constant")
        from sklearn.mixture import GaussianMixture

        normalized = ((losses - minimum) / span).reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=int(seed), reg_covar=1.0e-6)
        gmm.fit(normalized)
        means = gmm.means_.reshape(-1)
        clean_component = int(np.argmin(means))
        gamma = gmm.predict_proba(normalized)[:, clean_component].astype(np.float32)
        if not np.all(np.isfinite(gamma)):
            raise ValueError("GMM posterior contains non-finite values")
        return GMMReliabilityResult(gamma=gamma, success=True, reason="", component_means=(float(means[0]), float(means[1])))
    except Exception as exc:
        return GMMReliabilityResult(gamma=fallback, success=False, reason=str(exc), component_means=None)


def reliability_weights(gamma: np.ndarray, threshold: float) -> np.ndarray:
    gamma = np.asarray(gamma, dtype=np.float32)
    return np.where(gamma >= float(threshold), 1.0, gamma).astype(np.float32)


def update_soft_labels(
    previous: torch.Tensor,
    probabilities: torch.Tensor,
    noisy_one_hot: torch.Tensor,
    omega: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    omega = omega[:, None].clamp(0.0, 1.0)
    current = (1.0 - omega) * probabilities + omega * noisy_one_hot
    updated = float(alpha) * previous + (1.0 - float(alpha)) * current
    return updated.clamp_min(0.0) / updated.sum(dim=1, keepdim=True).clamp_min(1.0e-12)


def compute_noise_metrics(gamma: np.ndarray, omega: np.ndarray, is_clean: np.ndarray | None) -> dict[str, float | str]:
    """Compute optional GT-noise metrics without changing any training state."""
    if is_clean is None:
        return {
            "gamma_clean_auc": "",
            "clean_gamma_mean": "",
            "noisy_gamma_mean": "",
            "threshold_purity": "",
            "threshold_clean_recall": "",
        }
    from sklearn.metrics import roc_auc_score

    gamma = np.asarray(gamma, dtype=np.float32)
    omega = np.asarray(omega, dtype=np.float32)
    is_clean = np.asarray(is_clean, dtype=bool)
    selected = omega >= 1.0
    auc = float(roc_auc_score(is_clean.astype(np.int64), gamma)) if np.unique(is_clean).size == 2 else ""
    return {
        "gamma_clean_auc": auc,
        "clean_gamma_mean": float(gamma[is_clean].mean()) if np.any(is_clean) else "",
        "noisy_gamma_mean": float(gamma[~is_clean].mean()) if np.any(~is_clean) else "",
        "threshold_purity": safe_ratio(int(np.sum(is_clean & selected)), int(selected.sum())),
        "threshold_clean_recall": safe_ratio(int(np.sum(is_clean & selected)), int(is_clean.sum())),
    }


def train_snscl_lora(
    train_paths: list[str],
    train_labels: np.ndarray,
    eval_paths: list[str],
    eval_labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: dict[str, Any],
    method: str,
    seed: int,
    path_maps: list[tuple[str, str]] | None = None,
    gt_clean_mask: np.ndarray | None = None,
    checkpoint_path: Path | None = None,
    latest_checkpoint_path: Path | None = None,
    epoch_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SNSCLRunResult:
    """Train the standalone SNSCL-DINOv2+LoRA adapted baseline."""
    from torch.utils.data import DataLoader
    from torchvision import transforms

    path_maps = path_maps or []
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape != (len(train_labels),) or not np.any(train_mask):
        raise ValueError("train_mask must match train_labels and select at least one sample.")
    if gt_clean_mask is not None:
        gt_clean_mask = np.asarray(gt_clean_mask, dtype=bool)
        if gt_clean_mask.shape != train_mask.shape:
            raise ValueError("gt_clean_mask must match train_mask shape.")

    train_cfg = cfg["lora_train"]
    snscl_cfg = cfg["snscl"]
    feature_cfg = cfg["feature"]
    device = resolve_device(torch, feature_cfg.get("device", "auto"))
    set_torch_seed(torch, seed)
    validate_snscl_config(snscl_cfg)

    classes = sorted(set(train_labels.tolist()))
    label_to_id = {label: i for i, label in enumerate(classes)}
    train_idx = np.where(train_mask)[0]
    eval_known = np.array([label in label_to_id for label in eval_labels], dtype=bool)
    eval_idx = np.where(eval_known)[0]
    if len(eval_idx) == 0:
        raise ValueError("Eval split has no labels that appear in the train split.")
    noisy_ids = np.array([label_to_id[str(label)] for label in train_labels], dtype=np.int64)
    num_classes = len(classes)

    train_transform, eval_transform = build_transforms(transforms, int(feature_cfg["input_size"]))
    train_dataset = ImageSplitDataset(train_paths, train_labels, train_idx, label_to_id, train_transform, path_maps)
    reliability_dataset = ImageSplitDataset(train_paths, train_labels, train_idx, label_to_id, eval_transform, path_maps)
    eval_dataset = ImageSplitDataset(eval_paths, eval_labels, eval_idx, label_to_id, eval_transform, path_maps)
    loader_kwargs = {
        "num_workers": int(train_cfg.get("num_workers", 4)),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, batch_size=int(train_cfg["batch_size"]), shuffle=True, **loader_kwargs)
    reliability_loader = DataLoader(
        reliability_dataset, batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])), shuffle=False, **loader_kwargs
    )
    eval_loader = DataLoader(eval_dataset, batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])), shuffle=False, **loader_kwargs)

    model = DINOv2LoRAClassifier.make(torch, cfg, num_classes).to(device)
    freeze_all(model.backbone)
    trainable_modules = inject_lora(
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

    projection = ProjectionHead(model.embed_dim, int(snscl_cfg["proj_dim"])).to(device)
    stochastic = StochasticHead(
        int(snscl_cfg["proj_dim"]),
        int(snscl_cfg["stochastic_hidden_dim"]),
        int(snscl_cfg["proj_dim"]),
    ).to(device)
    queue = ClassWiseQueue(num_classes, int(snscl_cfg["queue_size"]), int(snscl_cfg["proj_dim"])).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"name": "lora", "params": lora_parameters(model), "lr": float(train_cfg["lora_lr"])},
            {"name": "classifier", "params": model.head.parameters(), "lr": float(train_cfg["head_lr"])},
            {"name": "projection", "params": projection.parameters(), "lr": float(snscl_cfg["projection_lr"])},
            {"name": "stochastic", "params": stochastic.parameters(), "lr": float(snscl_cfg["stochastic_lr"])},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(train_cfg["epochs"])
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
    scheduler = build_scheduler(torch, optimizer, total_steps, warmup_steps, str(train_cfg.get("scheduler", "cosine")))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.startswith("cuda"))
    amp = bool(train_cfg.get("amp", True))

    gamma = np.ones(len(train_labels), dtype=np.float32)
    omega = np.ones(len(train_labels), dtype=np.float32)
    soft_labels = F.one_hot(torch.from_numpy(noisy_ids), num_classes=num_classes).float()
    logs: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    reliability_summary_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    consecutive_gmm_failures = 0
    consecutive_amp_skips = 0
    trainable_params = count_trainable_params(model) + count_trainable_params(projection) + count_trainable_params(stochastic)
    total_params = count_total_params(model) + count_total_params(projection) + count_total_params(stochastic)
    log_stage(f"[snscl] seed={seed}: train_images={len(train_idx)}, eval_images={len(eval_idx)}, trainable_params={trainable_params}")

    for epoch_id in range(1, epochs + 1):
        model.train()
        projection.train()
        stochastic.train()
        sums = {
            "total": 0.0,
            "cls": 0.0,
            "ntcl": 0.0,
            "kl": 0.0,
            "valid": 0.0,
            "pos": 0.0,
            "neg": 0.0,
            "grad_norm": 0.0,
            "grad_seen": 0.0,
            "amp_skipped": 0.0,
            "max_consecutive_amp_skips": 0.0,
            "amp_scale_min": float(scaler.get_scale()),
            "amp_scale_max": float(scaler.get_scale()),
            "mu_abs": 0.0,
            "logvar_min": float("inf"),
            "logvar_max": float("-inf"),
        }
        amp_overflow_groups: set[str] = set()
        seen = 0
        for batch_id, (images, noisy_labels, indices) in enumerate(
            progress_iter(train_loader, total=len(train_loader), desc=f"SNSCL seed={seed} epoch {epoch_id}/{epochs}"),
            start=1,
        ):
            images = images.to(device, non_blocking=True)
            noisy_labels = noisy_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits, cls_features = model(images, return_features=True)
                if epoch_id <= int(snscl_cfg["warmup_epochs"]):
                    loss_cls = F.cross_entropy(logits, noisy_labels)
                else:
                    batch_soft = soft_labels[indices].to(device)
                    loss_cls = soft_cross_entropy(logits, batch_soft)
            projected, sampled, mu, logvar = forward_stochastic_fp32(projection, stochastic, cls_features)
            with torch.autocast(device_type=logits.device.type, enabled=False):
                if epoch_id <= int(snscl_cfg["warmup_epochs"]):
                    loss_ntcl = logits.float().sum() * 0.0
                    loss_kl = logits.float().sum() * 0.0
                    ntcl_stats = _empty_ntcl_stats()
                else:
                    corrected = batch_soft.argmax(dim=1)
                    loss_ntcl, ntcl_stats = paper_ntcl_loss(sampled, corrected, queue, float(snscl_cfg["temperature"]))
                    loss_kl = gaussian_kl_loss(mu, logvar)
                loss = loss_cls.float() + float(snscl_cfg["lambda_ntcl"]) * loss_ntcl + float(snscl_cfg["lambda_kl"]) * loss_kl
            assert_finite_tensors(
                epoch_id,
                batch_id,
                logits=logits,
                cls_features=cls_features,
                projected=projected,
                sampled=sampled,
                mu=mu,
                logvar=logvar,
                loss_total=loss,
                loss_cls=loss_cls,
                loss_ntcl=loss_ntcl,
                loss_kl=loss_kl,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            non_finite_groups = find_non_finite_gradient_groups(torch, optimizer)
            optimizer_step_applied = not non_finite_groups
            if non_finite_groups and scaler.is_enabled():
                amp_overflow_groups.update(non_finite_groups)
                consecutive_amp_skips += 1
                sums["amp_skipped"] += 1
                sums["max_consecutive_amp_skips"] = max(sums["max_consecutive_amp_skips"], consecutive_amp_skips)
                # GradScaler observes the recorded inf values and skips this optimizer step.
                scaler.step(optimizer)
                scaler.update()
                grad_norm = 0.0
                raise_if_amp_overflow_persistent(
                    non_finite_groups,
                    consecutive_amp_skips,
                    int(snscl_cfg["amp_overflow_patience"]),
                    epoch_id,
                    batch_id,
                )
            else:
                grad_norm = check_and_clip_gradients(
                    torch,
                    optimizer,
                    max_grad_norm=float(snscl_cfg["max_grad_norm"]),
                    epoch_id=epoch_id,
                    batch_id=batch_id,
                )
                scaler.step(optimizer)
                scaler.update()
                assert_finite_parameters(epoch_id, batch_id, model=model, projection=projection, stochastic=stochastic)
                scheduler.step()
                consecutive_amp_skips = 0
            current_amp_scale = float(scaler.get_scale())
            sums["amp_scale_min"] = min(sums["amp_scale_min"], current_amp_scale)
            sums["amp_scale_max"] = max(sums["amp_scale_max"], current_amp_scale)

            if optimizer_step_applied and epoch_id >= int(snscl_cfg["queue_start_epoch"]):
                corrected = soft_labels[indices].argmax(dim=1).to(device)
                batch_omega = torch.from_numpy(omega[indices.numpy()]).to(device)
                queue.enqueue(sampled.detach(), corrected, batch_omega)

            batch_size = int(images.shape[0])
            seen += batch_size
            sums["total"] += float(loss.detach().cpu()) * batch_size
            sums["cls"] += float(loss_cls.detach().cpu()) * batch_size
            sums["ntcl"] += float(loss_ntcl.detach().cpu()) * batch_size
            sums["kl"] += float(loss_kl.detach().cpu()) * batch_size
            sums["valid"] += ntcl_stats["num_valid_ntcl_anchors"]
            sums["pos"] += ntcl_stats["mean_positive_count"] * ntcl_stats["num_valid_ntcl_anchors"]
            sums["neg"] += ntcl_stats["mean_negative_count"] * ntcl_stats["num_valid_ntcl_anchors"]
            if optimizer_step_applied:
                sums["grad_norm"] += grad_norm * batch_size
                sums["grad_seen"] += batch_size
            sums["mu_abs"] = max(sums["mu_abs"], float(mu.detach().abs().max().cpu()))
            sums["logvar_min"] = min(sums["logvar_min"], float(logvar.detach().min().cpu()))
            sums["logvar_max"] = max(sums["logvar_max"], float(logvar.detach().max().cpu()))

        top1, top5 = evaluate_lora(torch, model, eval_loader, device, num_classes, amp)
        valid_anchors = sums["valid"]
        row = {
            "method": method,
            "seed": int(seed),
            "epoch": int(epoch_id),
            "lr_lora": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
            "lr_projection": float(optimizer.param_groups[2]["lr"]),
            "lr_stochastic": float(optimizer.param_groups[3]["lr"]),
            "loss_total": safe_ratio(sums["total"], seen),
            "loss_cls": safe_ratio(sums["cls"], seen),
            "loss_ntcl": safe_ratio(sums["ntcl"], seen),
            "loss_kl": safe_ratio(sums["kl"], seen),
            "mean_gamma": float(gamma[train_idx].mean()),
            "gamma_std": float(gamma[train_idx].std()),
            "mean_omega": float(omega[train_idx].mean()),
            "queue_fill_ratio": queue.fill_ratio(),
            "num_valid_ntcl_anchors": int(valid_anchors),
            "valid_anchor_ratio": safe_ratio(valid_anchors, seen),
            "mean_positive_count": safe_ratio(sums["pos"], valid_anchors),
            "mean_negative_count": safe_ratio(sums["neg"], valid_anchors),
            "mean_grad_norm": safe_ratio(sums["grad_norm"], sums["grad_seen"]),
            "amp_skipped_steps": int(sums["amp_skipped"]),
            "max_consecutive_amp_skips": int(sums["max_consecutive_amp_skips"]),
            "amp_scale_final": float(scaler.get_scale()),
            "amp_scale_min": float(sums["amp_scale_min"]),
            "amp_scale_max": float(sums["amp_scale_max"]),
            "amp_overflow_groups": ",".join(sorted(amp_overflow_groups)),
            "max_mu_abs": float(sums["mu_abs"]),
            "min_logvar": float(sums["logvar_min"]),
            "max_logvar": float(sums["logvar_max"]),
            "max_model_param_abs": max_floating_parameter_abs(model),
            "max_projection_param_abs": max_floating_parameter_abs(projection),
            "max_stochastic_param_abs": max_floating_parameter_abs(stochastic),
            "val_top1": float(top1),
            "val_top5": float(top5),
            "best_top1": max([float(item["val_top1"]) for item in logs] + [float(top1)]),
            "train_samples": int(len(train_idx)),
            "eval_samples": int(len(eval_idx)),
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "gmm_success": "",
            "consecutive_gmm_failures": int(consecutive_gmm_failures),
            "health_status": "ok",
            "health_reasons": "",
            "corrected_label_changes": int(np.sum(soft_labels.argmax(dim=1).numpy()[train_idx] != noisy_ids[train_idx])),
            "corrected_label_change_ratio": safe_ratio(
                int(np.sum(soft_labels.argmax(dim=1).numpy()[train_idx] != noisy_ids[train_idx])),
                len(train_idx),
            ),
        }
        queue_rows.append(build_queue_row(queue, method, seed, epoch_id))

        reliability_summary: dict[str, Any] | None = None
        if epoch_id >= int(snscl_cfg["warmup_epochs"]):
            losses, probabilities = evaluate_reliability(torch, model, reliability_loader, device, len(train_labels), num_classes, amp)
            previous = gamma[train_idx].copy()
            gmm = fit_gmm_reliability(losses[train_idx], previous_gamma=previous, seed=seed + epoch_id)
            if gmm.success:
                consecutive_gmm_failures = 0
                gamma[train_idx] = gmm.gamma
                omega[train_idx] = reliability_weights(gmm.gamma, float(snscl_cfg["reliability_threshold"]))
                selected = torch.from_numpy(train_idx)
                soft_labels[selected] = update_soft_labels(
                    soft_labels[selected],
                    torch.from_numpy(probabilities[train_idx]),
                    F.one_hot(torch.from_numpy(noisy_ids[train_idx]), num_classes=num_classes).float(),
                    torch.from_numpy(omega[train_idx]),
                    float(snscl_cfg["label_ma_alpha"]),
                )
            else:
                consecutive_gmm_failures += 1
            noise_metrics = compute_noise_metrics(gamma[train_idx], omega[train_idx], gt_clean_mask[train_idx] if gt_clean_mask is not None else None)
            reliability_summary = build_reliability_summary(method, seed, epoch_id, gamma[train_idx], omega[train_idx], gmm, noise_metrics)
            reliability_summary_rows.append(reliability_summary)
            if should_save_reliability(epoch_id, epochs, int(snscl_cfg["reliability_save_interval"])):
                reliability_rows.extend(
                    build_reliability_rows(
                        method,
                        seed,
                        epoch_id,
                        train_idx,
                        train_paths,
                        train_labels,
                        noisy_ids,
                        losses,
                        gamma,
                        omega,
                        soft_labels,
                        gt_clean_mask,
                    )
                )
            row["mean_gamma"] = float(gamma[train_idx].mean())
            row["gamma_std"] = float(gamma[train_idx].std())
            row["mean_omega"] = float(omega[train_idx].mean())
            row["gmm_success"] = "yes" if gmm.success else "no"
            row["consecutive_gmm_failures"] = int(consecutive_gmm_failures)
            corrected_ids = soft_labels.argmax(dim=1).numpy()
            row["corrected_label_changes"] = int(np.sum(corrected_ids[train_idx] != noisy_ids[train_idx]))
            row["corrected_label_change_ratio"] = safe_ratio(row["corrected_label_changes"], len(train_idx))

        health_reasons = evaluate_snscl_health(row, reliability_summary, snscl_cfg, consecutive_gmm_failures)
        if health_reasons:
            row["health_status"] = "failed"
            row["health_reasons"] = "; ".join(health_reasons)
        if is_better_healthy_checkpoint(row, best_row):
            best_row = row
            best_state = build_snscl_checkpoint_state(model, projection, stochastic, queue, gamma, omega, soft_labels)
            save_snscl_checkpoint(torch, checkpoint_path, method, seed, classes, best_row, best_state, checkpoint_kind="best")
        if row["health_status"] == "ok":
            latest_state = build_snscl_checkpoint_state(model, projection, stochastic, queue, gamma, omega, soft_labels)
            save_snscl_checkpoint(torch, latest_checkpoint_path, method, seed, classes, row, latest_state, checkpoint_kind="latest")
        if best_row is not None:
            row["best_top1"] = float(best_row["val_top1"])
        logs.append(row)
        if epoch_callback is not None:
            epoch_callback(
                {
                    "method": method,
                    "seed": int(seed),
                    "epoch": int(epoch_id),
                    "logs": logs,
                    "queue_rows": queue_rows,
                    "reliability_summary_rows": reliability_summary_rows,
                    "reliability_rows": reliability_rows,
                    "health_status": row["health_status"],
                    "health_reasons": row["health_reasons"],
                }
            )
        log_stage(
            f"[snscl] seed={seed} epoch={epoch_id}/{epochs}: loss={row['loss_total']:.4f}, "
            f"top1={top1:.4f}, queue={row['queue_fill_ratio']:.4f}, valid_ntcl={int(valid_anchors)}, "
            f"gamma={row['mean_gamma']:.4f}, health={row['health_status']}"
        )
        if health_reasons and bool(snscl_cfg.get("fail_on_health_check", True)):
            raise SNSCLHealthError(f"SNSCL health check failed at epoch {epoch_id}: {row['health_reasons']}")

    return SNSCLRunResult(
        logs=logs,
        summary=summarize_snscl_logs(method, seed, logs),
        trainable_modules=trainable_modules,
        trainable_params=trainable_params,
        total_params=total_params,
        reliability_rows=reliability_rows,
        reliability_summary_rows=reliability_summary_rows,
        queue_rows=queue_rows,
    )


def evaluate_reliability(
    torch_module: Any,
    model: Any,
    loader: Any,
    device: str,
    total_train: int,
    num_classes: int,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate original noisy-label CE and class probabilities on the full training split."""
    model.eval()
    losses = np.full(total_train, np.nan, dtype=np.float32)
    probabilities = np.full((total_train, num_classes), np.nan, dtype=np.float32)
    with torch_module.no_grad():
        for images, noisy_labels, indices in progress_iter(loader, total=len(loader), desc="SNSCL reliability eval"):
            images = images.to(device, non_blocking=True)
            noisy_labels = noisy_labels.to(device, non_blocking=True)
            with torch_module.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
                logits = model(images)
                ce = F.cross_entropy(logits, noisy_labels, reduction="none")
                probs = torch_module.softmax(logits, dim=1)
            idx = indices.numpy().astype(np.int64)
            losses[idx] = ce.detach().cpu().numpy().astype(np.float32)
            probabilities[idx] = probs.detach().cpu().numpy().astype(np.float32)
    return losses, probabilities


def build_snscl_checkpoint_state(
    model: Any,
    projection: ProjectionHead,
    stochastic: StochasticHead,
    queue: ClassWiseQueue,
    gamma: np.ndarray,
    omega: np.ndarray,
    soft_labels: torch.Tensor,
) -> dict[str, Any]:
    return {
        "model_state_dict": trainable_state_dict(model),
        "projection_state_dict": copy.deepcopy({key: value.detach().cpu() for key, value in projection.state_dict().items()}),
        "stochastic_state_dict": copy.deepcopy({key: value.detach().cpu() for key, value in stochastic.state_dict().items()}),
        "queue_state_dict": copy.deepcopy({key: value.detach().cpu() for key, value in queue.state_dict().items()}),
        "gamma": gamma.copy(),
        "omega": omega.copy(),
        "soft_labels": soft_labels.clone(),
    }


def save_snscl_checkpoint(
    torch_module: Any,
    checkpoint_path: Path | None,
    method: str,
    seed: int,
    classes: list[str],
    row: dict[str, Any],
    state: dict[str, Any],
    checkpoint_kind: str,
) -> None:
    """Persist a healthy checkpoint immediately so later failures do not erase progress."""
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "seed": int(seed),
        "classes": classes,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_epoch": int(row["epoch"]),
        "checkpoint_top1": float(row["val_top1"]),
        **state,
    }
    if checkpoint_kind == "best":
        payload.update({"best_epoch": int(row["epoch"]), "best_top1": float(row["val_top1"])})
    else:
        payload.update({"latest_epoch": int(row["epoch"]), "latest_top1": float(row["val_top1"])})
    torch_module.save(payload, checkpoint_path)


def assert_finite_tensors(epoch_id: int, batch_id: int, **tensors: torch.Tensor) -> None:
    non_finite = [name for name, tensor in tensors.items() if not bool(torch.isfinite(tensor).all().item())]
    if non_finite:
        raise SNSCLHealthError(
            f"Non-finite tensors at epoch {epoch_id}, batch {batch_id}: {', '.join(non_finite)}."
        )


def check_and_clip_gradients(
    torch_module: Any,
    optimizer: Any,
    max_grad_norm: float,
    epoch_id: int,
    batch_id: int,
) -> float:
    """Reject non-finite gradients, then clip the finite global norm."""
    parameters = []
    non_finite_groups = find_non_finite_gradient_groups(torch_module, optimizer)
    for group_id, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            parameters.append(parameter)
    if non_finite_groups:
        raise SNSCLHealthError(
            f"Non-finite gradients at epoch {epoch_id}, batch {batch_id}: {', '.join(non_finite_groups)}."
        )
    if not parameters:
        raise SNSCLHealthError(f"No gradients at epoch {epoch_id}, batch {batch_id}.")
    total_norm = torch_module.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=float(max_grad_norm),
        error_if_nonfinite=True,
    )
    return float(total_norm.detach().cpu())


def find_non_finite_gradient_groups(torch_module: Any, optimizer: Any) -> list[str]:
    non_finite_groups = []
    for group_id, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", group_id))
        if any(
            parameter.grad is not None and not bool(torch_module.isfinite(parameter.grad).all().item())
            for parameter in group["params"]
        ):
            non_finite_groups.append(group_name)
    return non_finite_groups


def raise_if_amp_overflow_persistent(
    non_finite_groups: list[str],
    consecutive_amp_skips: int,
    patience: int,
    epoch_id: int,
    batch_id: int,
) -> None:
    if non_finite_groups and consecutive_amp_skips >= patience:
        raise SNSCLHealthError(
            f"AMP gradient overflow for {consecutive_amp_skips} consecutive steps at "
            f"epoch {epoch_id}, batch {batch_id}: {', '.join(non_finite_groups)}."
        )


def assert_finite_parameters(epoch_id: int, batch_id: int, **modules: Any) -> None:
    non_finite = []
    for module_name, module in modules.items():
        trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if any(not bool(torch.isfinite(parameter).all().item()) for parameter in trainable):
            non_finite.append(module_name)
    if non_finite:
        raise SNSCLHealthError(
            f"Non-finite parameters after optimizer step at epoch {epoch_id}, batch {batch_id}: "
            f"{', '.join(non_finite)}."
        )


def max_floating_parameter_abs(module: Any) -> float:
    parameters = [parameter.detach() for parameter in module.parameters() if parameter.requires_grad and parameter.numel()]
    if any(not bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
        return float("nan")
    values = [float(parameter.abs().max().cpu()) for parameter in parameters]
    return max(values) if values else 0.0


def evaluate_snscl_health(
    row: dict[str, Any],
    reliability_summary: dict[str, Any] | None,
    cfg: dict[str, Any],
    consecutive_gmm_failures: int,
) -> list[str]:
    """Return mechanical health failures that should stop a long experiment."""
    reasons: list[str] = []
    finite_fields = [
        "loss_total",
        "loss_cls",
        "loss_ntcl",
        "loss_kl",
        "mean_gamma",
        "gamma_std",
        "mean_omega",
        "queue_fill_ratio",
        "mean_grad_norm",
        "max_mu_abs",
        "min_logvar",
        "max_logvar",
        "max_model_param_abs",
        "max_projection_param_abs",
        "max_stochastic_param_abs",
        "val_top1",
        "val_top5",
    ]
    non_finite = [name for name in finite_fields if not np.isfinite(float(row[name]))]
    if non_finite:
        reasons.append(f"non-finite epoch metrics: {', '.join(non_finite)}")

    patience = int(cfg.get("gmm_failure_patience", 2))
    if reliability_summary is not None and consecutive_gmm_failures >= patience:
        reasons.append(
            f"GMM fallback occurred for {consecutive_gmm_failures} consecutive reliability updates "
            f"(latest: {reliability_summary['gmm_reason']})"
        )

    if int(row["epoch"]) >= int(cfg.get("health_check_epoch", 7)):
        min_queue_fill = float(cfg.get("min_queue_fill_ratio", 1.0e-6))
        if float(row["queue_fill_ratio"]) < min_queue_fill:
            reasons.append(f"queue_fill_ratio={float(row['queue_fill_ratio']):.6g} < {min_queue_fill:.6g}")
        min_valid_anchors = int(cfg.get("min_valid_ntcl_anchors", 1))
        if int(row["num_valid_ntcl_anchors"]) < min_valid_anchors:
            reasons.append(f"num_valid_ntcl_anchors={int(row['num_valid_ntcl_anchors'])} < {min_valid_anchors}")
        min_gamma_std = float(cfg.get("min_gamma_std", 1.0e-6))
        if float(row["gamma_std"]) < min_gamma_std:
            reasons.append(f"gamma_std={float(row['gamma_std']):.6g} < {min_gamma_std:.6g}")
    return reasons


def is_better_healthy_checkpoint(row: dict[str, Any], best_row: dict[str, Any] | None) -> bool:
    if row.get("health_status") != "ok":
        return False
    return best_row is None or float(row["val_top1"]) > float(best_row["val_top1"])


def build_queue_row(queue: ClassWiseQueue, method: str, seed: int, epoch_id: int) -> dict[str, Any]:
    counts = queue.counts.detach().cpu().numpy()
    return {
        "method": method,
        "seed": int(seed),
        "epoch": int(epoch_id),
        "queue_fill_ratio": queue.fill_ratio(),
        "filled_entries": int(queue.valid.sum().item()),
        "total_capacity": int(queue.valid.numel()),
        "min_class_count": int(counts.min()) if len(counts) else 0,
        "max_class_count": int(counts.max()) if len(counts) else 0,
        "mean_class_count": float(counts.mean()) if len(counts) else 0.0,
    }


def build_reliability_summary(
    method: str,
    seed: int,
    epoch_id: int,
    gamma: np.ndarray,
    omega: np.ndarray,
    gmm: GMMReliabilityResult,
    noise_metrics: dict[str, float | str],
) -> dict[str, Any]:
    means = gmm.component_means or ("", "")
    return {
        "method": method,
        "seed": int(seed),
        "epoch": int(epoch_id),
        "gmm_success": "yes" if gmm.success else "no",
        "gmm_reason": gmm.reason,
        "gmm_mean_0": means[0],
        "gmm_mean_1": means[1],
        "mean_gamma": float(gamma.mean()),
        "mean_omega": float(omega.mean()),
        "threshold_selected": int(np.sum(omega >= 1.0)),
        **noise_metrics,
    }


def build_reliability_rows(
    method: str,
    seed: int,
    epoch_id: int,
    train_idx: np.ndarray,
    train_paths: list[str],
    train_labels: np.ndarray,
    noisy_ids: np.ndarray,
    losses: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    soft_labels: torch.Tensor,
    gt_clean_mask: np.ndarray | None,
) -> list[dict[str, Any]]:
    corrected = soft_labels.argmax(dim=1).numpy()
    rows = []
    for idx in train_idx:
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "epoch": int(epoch_id),
                "index": int(idx),
                "path": train_paths[int(idx)],
                "web_label": str(train_labels[int(idx)]),
                "noisy_label_id": int(noisy_ids[int(idx)]),
                "corrected_label_id": int(corrected[int(idx)]),
                "noisy_ce_loss": float(losses[int(idx)]),
                "gamma": float(gamma[int(idx)]),
                "omega": float(omega[int(idx)]),
                "is_clean": int(gt_clean_mask[int(idx)]) if gt_clean_mask is not None else "",
            }
        )
    return rows


def should_save_reliability(epoch_id: int, total_epochs: int, interval: int) -> bool:
    return epoch_id == total_epochs or (interval > 0 and epoch_id % interval == 0)


def summarize_snscl_logs(method: str, seed: int, logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not logs:
        raise ValueError("No SNSCL logs available.")
    healthy_logs = [row for row in logs if row.get("health_status", "ok") == "ok"]
    if not healthy_logs:
        raise ValueError("No healthy SNSCL epochs available for summary.")
    best = max(healthy_logs, key=lambda row: float(row["val_top1"]))
    final = logs[-1]
    last5 = np.array([float(row["val_top1"]) for row in logs[-5:]], dtype=np.float32)
    return {
        "method": method,
        "seed": int(seed),
        "train_samples": int(final["train_samples"]),
        "eval_samples": int(final["eval_samples"]),
        "best_epoch": int(best["epoch"]),
        "best_top1": float(best["val_top1"]),
        "best_top5": float(best["val_top5"]),
        "final_top1": float(final["val_top1"]),
        "final_top5": float(final["val_top5"]),
        "last5_mean": float(last5.mean()),
        "last5_std": float(last5.std()),
        "final_mean_gamma": float(final["mean_gamma"]),
        "final_mean_omega": float(final["mean_omega"]),
        "final_queue_fill_ratio": float(final["queue_fill_ratio"]),
        "trainable_params": int(final["trainable_params"]),
        "total_params": int(final["total_params"]),
    }


def validate_snscl_config(cfg: dict[str, Any]) -> None:
    for key in ["proj_dim", "stochastic_hidden_dim", "queue_size", "queue_start_epoch", "warmup_epochs", "reliability_save_interval"]:
        if int(cfg[key]) <= 0:
            raise ValueError(f"snscl.{key} must be positive.")
    for key in ["lambda_kl", "lambda_ntcl"]:
        if float(cfg[key]) < 0.0:
            raise ValueError(f"snscl.{key} must be non-negative.")
    for key in ["reliability_threshold", "label_ma_alpha"]:
        if not 0.0 <= float(cfg[key]) <= 1.0:
            raise ValueError(f"snscl.{key} must be in [0, 1].")
    if float(cfg["temperature"]) <= 0.0:
        raise ValueError("snscl.temperature must be positive.")
    for key in ["projection_lr", "stochastic_lr", "max_grad_norm"]:
        if float(cfg[key]) <= 0.0:
            raise ValueError(f"snscl.{key} must be positive.")
    if int(cfg.get("gmm_failure_patience", 2)) <= 0:
        raise ValueError("snscl.gmm_failure_patience must be positive.")
    if int(cfg.get("health_check_epoch", 7)) <= 0:
        raise ValueError("snscl.health_check_epoch must be positive.")
    for key in ["min_queue_fill_ratio", "min_gamma_std"]:
        if float(cfg.get(key, 0.0)) < 0.0:
            raise ValueError(f"snscl.{key} must be non-negative.")
    if int(cfg.get("min_valid_ntcl_anchors", 1)) < 0:
        raise ValueError("snscl.min_valid_ntcl_anchors must be non-negative.")
    if int(cfg.get("amp_overflow_patience", 5)) <= 0:
        raise ValueError("snscl.amp_overflow_patience must be positive.")


def _empty_ntcl_stats() -> dict[str, float]:
    return {
        "num_valid_ntcl_anchors": 0.0,
        "valid_anchor_ratio": 0.0,
        "mean_positive_count": 0.0,
        "mean_negative_count": 0.0,
    }
