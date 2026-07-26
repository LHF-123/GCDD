from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .features import load_dinov2_model, resolve_device
from .progress import log_stage, progress_iter


@dataclass
class LoRARunResult:
    logs: list[dict[str, Any]]
    summary: dict[str, Any]
    trainable_modules: list[str]
    trainable_params: int
    total_params: int


class LoRALinear:
    """Factory namespace for replacing Linear modules without importing torch at module import time."""

    @staticmethod
    def make(torch: Any, base: Any, rank: int, alpha: float, dropout: float) -> Any:
        class _LoRALinear(torch.nn.Module):
            def __init__(self, base_layer: Any):
                super().__init__()
                self.base = base_layer
                self.base.weight.requires_grad_(False)
                if self.base.bias is not None:
                    self.base.bias.requires_grad_(False)
                self.lora_a = torch.nn.Linear(base_layer.in_features, rank, bias=False)
                self.lora_b = torch.nn.Linear(rank, base_layer.out_features, bias=False)
                self.dropout = torch.nn.Dropout(dropout)
                self.scaling = float(alpha) / float(rank)
                torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
                torch.nn.init.zeros_(self.lora_b.weight)

            def forward(self, x: Any) -> Any:
                return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

        return _LoRALinear(base)


class ImageSplitDataset:
    def __init__(self, paths: list[str], labels: np.ndarray, indices: np.ndarray, label_to_id: dict[str, int], transform: Any, path_maps: list[tuple[str, str]]):
        self.paths = paths
        self.labels = labels
        self.indices = indices.astype(np.int64)
        self.label_to_id = label_to_id
        self.transform = transform
        self.path_maps = path_maps

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> tuple[Any, int, int]:
        idx = int(self.indices[position])
        path = resolve_image_path(self.paths[idx], self.path_maps)
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return image, int(self.label_to_id[str(self.labels[idx])]), idx


class DINOv2LoRAClassifier:
    @staticmethod
    def make(torch: Any, cfg: dict[str, Any], num_classes: int) -> Any:
        class _DINOv2LoRAClassifier(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = load_dinov2_model(torch, cfg).to(resolve_device(torch, cfg["feature"]["device"]))
                self.embed_dim = infer_embed_dim(self.backbone)
                self.head = torch.nn.Linear(self.embed_dim, num_classes)

            def forward(self, images: Any, return_features: bool = False) -> Any:
                features = self.backbone.forward_features(images)
                cls = features["x_norm_clstoken"]
                logits = self.head(cls)
                if return_features:
                    return logits, cls
                return logits

        return _DINOv2LoRAClassifier()


def train_dinov2_lora(
    train_paths: list[str],
    train_labels: np.ndarray,
    eval_paths: list[str],
    eval_labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: dict[str, Any],
    method: str,
    seed: int,
    path_maps: list[tuple[str, str]] | None = None,
    checkpoint_path: Path | None = None,
    test_paths: list[str] | None = None,
    test_labels: np.ndarray | None = None,
    final_checkpoint_path: Path | None = None,
    last5_checkpoint_dir: Path | None = None,
    full_noisy_candidate_mask: np.ndarray | None = None,
    checkpoint_protocol: str = "legacy_test_selected",
    posthoc_oracle_test: bool = False,
) -> LoRARunResult:
    """Train DINOv2 with LoRA on image data for one method and seed.

    With an explicit test split, ``eval_paths`` is used only for validation-based
    checkpoint selection; test metrics are evaluated after all epochs finish.
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    path_maps = path_maps or []
    train_mask = np.asarray(train_mask, dtype=bool)
    if train_mask.shape != (len(train_labels),):
        raise ValueError(f"train_mask must have shape ({len(train_labels)},), got {train_mask.shape}.")
    lora_cfg = cfg["lora"]
    train_cfg = cfg["lora_train"]
    feature_cfg = cfg["feature"]
    loss_cfg = resolve_loss_config(cfg)
    if (test_paths is None) != (test_labels is None):
        raise ValueError("test_paths and test_labels must be provided together.")
    expected_jal_mask = np.ones(len(train_labels), dtype=bool) if full_noisy_candidate_mask is None else np.asarray(full_noisy_candidate_mask, dtype=bool)
    if expected_jal_mask.shape != train_mask.shape:
        raise ValueError("full_noisy_candidate_mask must match train_mask shape.")
    if loss_cfg["loss_type"] == "jal_ce" and not np.array_equal(train_mask, expected_jal_mask):
        raise ValueError("JAL-CE must use every sample in its declared full-noisy training pool.")
    device = resolve_device(torch, feature_cfg.get("device", "auto"))
    set_torch_seed(torch, seed)

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
    # LoRA modules are injected after the backbone is moved, so move newly created layers too.
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
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
    scheduler = build_scheduler(torch, optimizer, total_steps, warmup_steps, str(train_cfg.get("scheduler", "cosine")))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.startswith("cuda"))
    criterion = build_criterion(torch, cfg)
    logs: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    last5_states: list[tuple[int, dict[str, Any]]] = []
    oracle_states: list[tuple[int, dict[str, Any]]] = []

    trainable_params = count_trainable_params(model)
    total_params = count_total_params(model)
    selection_split_name = "validation" if test_loader is not None else "eval"
    log_stage(
        f"[lora] {method} seed={seed}: train_images={len(train_idx)}, {selection_split_name}_images={len(eval_idx)}, "
        f"trainable_params={trainable_params}, loss_type={loss_cfg['loss_type']}, selection=full_noisy"
        if loss_cfg["loss_type"] == "jal_ce"
        else f"[lora] {method} seed={seed}: train_images={len(train_idx)}, {selection_split_name}_images={len(eval_idx)}, "
        f"trainable_params={trainable_params}, loss_type={loss_cfg['loss_type']}"
    )
    if loss_cfg["loss_type"] == "jal_ce":
        log_stage(
            f"[lora] JAL-CE parameters: alpha={loss_cfg['jal_alpha']}, beta={loss_cfg['jal_beta']}, "
            f"a={loss_cfg['jal_a']}, eps={loss_cfg['jal_eps']}, selection=full_noisy"
        )

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        progress = progress_iter(train_loader, total=len(train_loader), desc=f"LoRA {method} seed={seed} epoch {epoch}/{epochs}")
        for images, labels, _ in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(images)
                if loss_cfg["loss_type"] == "ce":
                    loss = criterion(logits, labels)
            if loss_cfg["loss_type"] == "jal_ce":
                loss = criterion(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = int(images.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            seen += batch_size

        top1, top5 = evaluate_lora(torch, model, eval_loader, device, len(classes), bool(train_cfg.get("amp", True)))
        row = {
            "method": method,
            "seed": int(seed),
            "epoch": int(epoch),
            "lr_lora": float(optimizer.param_groups[0]["lr"]),
            "lr_head": float(optimizer.param_groups[1]["lr"]),
            "loss": safe_ratio(loss_sum, seen),
            "train_loss": safe_ratio(loss_sum, seen),
            "top1": float(top1),
            "top5": float(top5),
            "val_top1": float(top1),
            "val_top5": float(top5),
            "train_samples": int(len(train_idx)),
            "eval_samples": int(len(eval_idx)),
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            **loss_log_fields(loss_cfg),
        }
        if best_row is None or float(row["top1"]) > float(best_row["top1"]):
            best_row = row
            best_state = trainable_state_dict(model)
        row["best_top1"] = float(best_row["top1"])
        row["best_epoch"] = int(best_row["epoch"])
        logs.append(row)
        epoch_state = trainable_state_dict(model)
        last5_states.append((epoch, epoch_state))
        if len(last5_states) > 5:
            last5_states.pop(0)
        if posthoc_oracle_test and test_loader is not None:
            oracle_states.append((epoch, epoch_state))
        log_stage(
            f"[lora] {method} seed={seed} epoch {epoch}/{epochs}: "
            f"loss={row['loss']:.4f}, top1={top1:.4f}, top5={top5:.4f}"
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
        protocol_metrics = {
            "checkpoint_protocol": checkpoint_protocol,
            "validation_samples": int(len(eval_idx)),
            "test_samples": int(len(test_idx)),
            "best_val_epoch": int(best_row["epoch"]),
            "best_val_top1": float(best_row["top1"]),
            "best_val_top5": float(best_row["top5"]),
            "validation_selected_test_top1": float(validation_selected_test_top1),
            "validation_selected_test_top5": float(validation_selected_test_top5),
            "final_test_top1": float(final_test_top1),
            "final_test_top5": float(final_test_top5),
            "last5_test_mean": float(last5_test_top1.mean()),
            "last5_test_std": float(last5_test_top1.std()),
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
                "loss": loss_cfg,
                "selection": "full_noisy" if loss_cfg["loss_type"] == "jal_ce" else "provided_train_mask",
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
                "loss": loss_cfg,
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
    return LoRARunResult(
        logs=logs,
        summary=summary,
        trainable_modules=trainable_modules,
        trainable_params=trainable_params,
        total_params=total_params,
    )


def build_transforms(transforms: Any, input_size: int) -> tuple[Any, Any]:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def inject_lora(torch: Any, module: Any, target_modules: list[str], rank: int, alpha: float, dropout: float) -> list[str]:
    replaced: list[str] = []
    for name, child in list(module.named_modules()):
        if not isinstance(child, torch.nn.Linear):
            continue
        if not any(target in name for target in target_modules):
            continue
        parent, child_name = get_parent_module(module, name)
        setattr(parent, child_name, LoRALinear.make(torch, child, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No Linear modules matched LoRA target_modules={target_modules}.")
    return replaced


def get_parent_module(root: Any, qualified_name: str) -> tuple[Any, str]:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def freeze_all(module: Any) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def lora_parameters(model: Any) -> list[Any]:
    return [param for name, param in model.named_parameters() if param.requires_grad and "head." not in name]


def trainable_state_dict(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if "lora_" in name or name.startswith("head.")}


def count_trainable_params(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def count_total_params(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def infer_embed_dim(backbone: Any) -> int:
    if hasattr(backbone, "embed_dim"):
        return int(backbone.embed_dim)
    if hasattr(backbone, "num_features"):
        return int(backbone.num_features)
    raise AttributeError("Cannot infer DINOv2 embedding dimension from backbone.")


def evaluate_lora(torch: Any, model: Any, loader: Any, device: str, num_classes: int, amp: bool) -> tuple[float, float]:
    model.eval()
    top1 = 0
    top5 = 0
    total = 0
    k5 = min(5, num_classes)
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.startswith("cuda")):
                logits = model(images)
            pred = torch.argsort(logits, dim=1, descending=True)
            top1 += int((pred[:, 0] == labels).sum().item())
            top5 += int((pred[:, :k5] == labels[:, None]).any(dim=1).sum().item())
            total += int(labels.shape[0])
    return safe_ratio(top1, total), safe_ratio(top5, total)


def evaluate_state_lora(
    torch: Any,
    model: Any,
    state_dict: dict[str, Any],
    loader: Any,
    device: str,
    num_classes: int,
    amp: bool,
) -> tuple[float, float]:
    """Evaluate a trainable-only LoRA/head state without rebuilding the frozen backbone."""
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return evaluate_lora(torch, model, loader, device, num_classes, amp)


def build_scheduler(torch: Any, optimizer: Any, total_steps: int, warmup_steps: int, scheduler: str) -> Any:
    if scheduler == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if scheduler == "linear":
            return 1.0 - progress
        if scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported LoRA scheduler: {scheduler}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def summarize_lora_logs(method: str, seed: int, logs: list[dict[str, Any]]) -> dict[str, Any]:
    if not logs:
        raise ValueError(f"No LoRA logs available for {method}.")
    best = max(logs, key=lambda row: float(row["top1"]))
    final = logs[-1]
    last5 = logs[-5:]
    last5_top1 = np.array([float(row["top1"]) for row in last5], dtype=np.float32)
    return {
        "method": method,
        "seed": int(seed),
        "train_samples": int(final["train_samples"]),
        "eval_samples": int(final["eval_samples"]),
        "best_epoch": int(best["epoch"]),
        "best_top1": float(best["top1"]),
        "best_top5": float(best["top5"]),
        "final_top1": float(final["top1"]),
        "final_top5": float(final["top5"]),
        "last5_mean": float(last5_top1.mean()),
        "last5_std": float(last5_top1.std()),
        "trainable_params": int(final["trainable_params"]),
        "total_params": int(final["total_params"]),
        "loss_type": str(final.get("loss_type", "ce")),
        "jal_alpha": final.get("jal_alpha", ""),
        "jal_beta": final.get("jal_beta", ""),
        "jal_a": final.get("jal_a", ""),
        "jal_eps": final.get("jal_eps", ""),
        "selection_mode": str(final.get("selection_mode", "dynamic_or_provided_mask")),
    }


def resolve_loss_config(cfg: dict[str, Any]) -> dict[str, Any]:
    loss_type = str(cfg.get("loss_type", "ce")).lower()
    if loss_type not in {"ce", "jal_ce"}:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    return {
        "loss_type": loss_type,
        "jal_alpha": float(cfg.get("jal_alpha", 1.0)),
        "jal_beta": float(cfg.get("jal_beta", 1.0)),
        "jal_a": float(cfg.get("jal_a", 30.0)),
        "jal_eps": float(cfg.get("jal_eps", 1.0e-8)),
    }


def build_criterion(torch: Any, cfg: dict[str, Any]) -> Any:
    loss_cfg = resolve_loss_config(cfg)
    if loss_cfg["loss_type"] == "ce":
        return torch.nn.CrossEntropyLoss()
    if loss_cfg["loss_type"] == "jal_ce":
        from losses.jal import JALCELoss

        return JALCELoss(
            alpha=loss_cfg["jal_alpha"],
            beta=loss_cfg["jal_beta"],
            a=loss_cfg["jal_a"],
            eps=loss_cfg["jal_eps"],
        )
    raise ValueError(f"Unknown loss_type: {loss_cfg['loss_type']}")


def loss_log_fields(loss_cfg: dict[str, Any]) -> dict[str, Any]:
    jal_enabled = loss_cfg["loss_type"] == "jal_ce"
    return {
        "loss_type": loss_cfg["loss_type"],
        "jal_alpha": loss_cfg["jal_alpha"] if jal_enabled else "",
        "jal_beta": loss_cfg["jal_beta"] if jal_enabled else "",
        "jal_a": loss_cfg["jal_a"] if jal_enabled else "",
        "jal_eps": loss_cfg["jal_eps"] if jal_enabled else "",
        "selection_mode": "full_noisy" if jal_enabled else "provided_train_mask",
    }


def parse_target_modules(raw: str) -> list[str]:
    targets = [item.strip() for item in raw.split(",") if item.strip()]
    if not targets:
        raise ValueError("At least one LoRA target module pattern is required.")
    return targets


def resolve_image_path(path_text: str, path_maps: list[tuple[str, str]]) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    raw = str(path).replace("\\", "/")
    for old, new in path_maps:
        old_norm = old.replace("\\", "/").rstrip("/")
        if raw == old_norm or raw.startswith(old_norm + "/"):
            suffix = raw[len(old_norm) :].lstrip("/")
            mapped = Path(new).expanduser() / Path(*suffix.split("/"))
            if mapped.exists():
                return mapped
    raise FileNotFoundError(f"Image path does not exist: {path_text}")


def set_torch_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_ratio(num: int | float, denom: int | float) -> float:
    return float(num / denom) if denom else 0.0
