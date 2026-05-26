from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .progress import log_stage


@dataclass
class LinearModel:
    weights: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    classes: list[str]


def train_linear_smoke(features: np.ndarray, labels: np.ndarray, clean_mask: np.ndarray, cfg: dict) -> list[dict[str, float | int]]:
    """Train a tiny numpy linear classifier for V0 pipeline validation."""
    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    encoded, classes = encode_labels(labels)
    num_classes = len(classes)
    if num_classes == 0:
        raise ValueError("No labels available for training.")
    if num_classes == 1:
        return [{"epoch": 1, "lr": float(cfg["train"]["lr"]), "loss": 0.0, "top1": 1.0, "top5": 1.0, "train_samples": int(clean_mask.sum())}]

    x = standardize(features.astype(np.float32))
    train_idx = np.where(clean_mask)[0]
    if len(train_idx) == 0:
        raise ValueError("Clean split is empty; cannot run V0 training smoke test.")

    weights = rng.normal(scale=0.01, size=(x.shape[1], num_classes)).astype(np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    logs: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        epoch_lr = learning_rate_for_epoch(cfg, epoch, epochs)
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            logits = x[idx] @ weights + bias
            probs = softmax(logits)
            y = encoded[idx]
            losses.append(float(-np.log(probs[np.arange(len(idx)), y] + 1.0e-12).mean()))

            grad = probs
            grad[np.arange(len(idx)), y] -= 1.0
            grad /= len(idx)
            weights -= epoch_lr * (x[idx].T @ grad)
            bias -= epoch_lr * grad.sum(axis=0)

        eval_logits = x @ weights + bias
        top1, top5 = topk_accuracy(eval_logits, encoded, k5=min(5, num_classes))
        log_stage(f"[train] smoke epoch {epoch}/{epochs}: lr={epoch_lr:.6g}, loss={float(np.mean(losses)) if losses else 0.0:.4f}, top1={top1:.4f}")
        logs.append(
            {
                "epoch": epoch,
                "lr": epoch_lr,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "top1": top1,
                "top5": top5,
                "train_samples": int(len(train_idx)),
            }
        )
    return logs


def train_linear_eval(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    train_mask: np.ndarray,
    cfg: dict,
    method: str,
    sample_weights: np.ndarray | None = None,
) -> tuple[list[dict[str, float | int | str]], LinearModel]:
    """Train a linear classifier on frozen features and evaluate each epoch."""
    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    encoded_train, classes = encode_labels(train_labels)
    encoded_eval = encode_with_classes(eval_labels, classes)
    eval_known = encoded_eval >= 0
    if not np.any(eval_known):
        raise ValueError("Eval split has no labels that appear in the train split.")

    x_train, mean, std = standardize_fit(train_features.astype(np.float32))
    x_eval = standardize_apply(eval_features.astype(np.float32), mean, std)
    weights_all = prepare_sample_weights(sample_weights, len(train_labels))
    train_idx = np.where(train_mask & (weights_all > 0))[0]
    if len(train_idx) == 0:
        raise ValueError(f"{method} selected no training samples.")

    num_classes = len(classes)
    weights = rng.normal(scale=0.01, size=(x_train.shape[1], num_classes)).astype(np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    logs: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        epoch_lr = learning_rate_for_epoch(cfg, epoch, epochs)
        rng.shuffle(train_idx)
        loss_num = 0.0
        loss_den = 0.0
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            logits = x_train[idx] @ weights + bias
            probs = softmax(logits)
            y = encoded_train[idx]
            batch_weights = weights_all[idx].astype(np.float32)
            weight_sum = float(batch_weights.sum())
            if weight_sum <= 0:
                continue
            ce = -np.log(probs[np.arange(len(idx)), y] + 1.0e-12)
            loss_num += float(np.sum(batch_weights * ce))
            loss_den += weight_sum

            grad = probs
            grad[np.arange(len(idx)), y] -= 1.0
            grad *= (batch_weights / (weight_sum + 1.0e-8))[:, None]
            weights -= epoch_lr * (x_train[idx].T @ grad)
            bias -= epoch_lr * grad.sum(axis=0)

        eval_logits = x_eval[eval_known] @ weights + bias
        top1, top5 = topk_accuracy(eval_logits, encoded_eval[eval_known], k5=min(5, num_classes))
        epoch_loss = loss_num / (loss_den + 1.0e-8)
        log_stage(
            f"[train] {method} epoch {epoch}/{epochs}: "
            f"lr={epoch_lr:.6g}, loss={epoch_loss:.4f}, "
            f"top1={top1:.4f}, top5={top5:.4f}"
        )
        logs.append(
            {
                "method": method,
                "epoch": epoch,
                "lr": epoch_lr,
                "loss": epoch_loss,
                "top1": top1,
                "top5": top5,
                "train_samples": int(len(train_idx)),
                "eval_samples": int(eval_known.sum()),
            }
        )

    return logs, LinearModel(weights=weights, bias=bias, mean=mean, std=std, classes=classes)


def prepare_sample_weights(sample_weights: np.ndarray | None, n: int) -> np.ndarray:
    if sample_weights is None:
        return np.ones(n, dtype=np.float32)
    weights = np.asarray(sample_weights, dtype=np.float32)
    if weights.shape != (n,):
        raise ValueError(f"sample_weights shape must be ({n},), got {weights.shape}.")
    if np.any(weights < 0):
        raise ValueError("sample_weights cannot contain negative values.")
    return weights


def train_linear_partial_label_eval(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    clean_mask: np.ndarray,
    recover_mask: np.ndarray,
    candidate_mask: np.ndarray,
    lambda_rec: float,
    cfg: dict,
    method: str,
) -> tuple[list[dict[str, float | int | str]], LinearModel]:
    """Train with clean CE plus partial-label loss for recoverable samples."""
    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    encoded_train, classes = encode_labels(train_labels)
    encoded_eval = encode_with_classes(eval_labels, classes)
    eval_known = encoded_eval >= 0
    if not np.any(eval_known):
        raise ValueError("Eval split has no labels that appear in the train split.")

    clean_mask = np.asarray(clean_mask, dtype=bool)
    recover_mask = np.asarray(recover_mask, dtype=bool)
    if clean_mask.shape != recover_mask.shape or clean_mask.shape[0] != len(train_labels):
        raise ValueError("clean_mask and recover_mask must match train label length.")
    if candidate_mask.shape != (len(train_labels), len(classes)):
        raise ValueError(f"candidate_mask must have shape ({len(train_labels)}, {len(classes)}), got {candidate_mask.shape}.")

    train_idx = np.where(clean_mask | recover_mask)[0]
    if len(train_idx) == 0:
        raise ValueError(f"{method} selected no training samples.")
    if np.any(recover_mask & (candidate_mask.sum(axis=1) == 0)):
        raise ValueError("Recoverable samples must have at least one candidate label.")

    x_train, mean, std = standardize_fit(train_features.astype(np.float32))
    x_eval = standardize_apply(eval_features.astype(np.float32), mean, std)
    num_classes = len(classes)
    weights = rng.normal(scale=0.01, size=(x_train.shape[1], num_classes)).astype(np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    logs: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        epoch_lr = learning_rate_for_epoch(cfg, epoch, epochs)
        rng.shuffle(train_idx)
        clean_loss_num = 0.0
        clean_loss_den = 0
        rec_loss_num = 0.0
        rec_loss_den = 0
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            logits = x_train[idx] @ weights + bias
            probs = softmax(logits)
            grad = np.zeros_like(probs, dtype=np.float32)

            batch_clean = clean_mask[idx]
            if np.any(batch_clean):
                clean_positions = np.where(batch_clean)[0]
                y = encoded_train[idx[clean_positions]]
                ce = -np.log(probs[clean_positions, y] + 1.0e-12)
                clean_loss_num += float(ce.sum())
                clean_loss_den += len(clean_positions)

                clean_grad = probs[clean_positions].copy()
                clean_grad[np.arange(len(clean_positions)), y] -= 1.0
                clean_grad /= len(clean_positions)
                grad[clean_positions] += clean_grad

            batch_recover = recover_mask[idx]
            if np.any(batch_recover):
                recover_positions = np.where(batch_recover)[0]
                cand = candidate_mask[idx[recover_positions]].astype(np.float32)
                candidate_prob = np.sum(probs[recover_positions] * cand, axis=1)
                candidate_prob = np.clip(candidate_prob, 1.0e-12, 1.0)
                rec_loss = -np.log(candidate_prob)
                rec_loss_num += float(rec_loss.sum())
                rec_loss_den += len(recover_positions)

                rec_grad = probs[recover_positions] - (probs[recover_positions] * cand / candidate_prob[:, None])
                rec_grad *= float(lambda_rec) / len(recover_positions)
                grad[recover_positions] += rec_grad.astype(np.float32)

            weights -= epoch_lr * (x_train[idx].T @ grad)
            bias -= epoch_lr * grad.sum(axis=0)

        clean_loss = clean_loss_num / max(1, clean_loss_den)
        rec_loss = rec_loss_num / max(1, rec_loss_den)
        epoch_loss = clean_loss + float(lambda_rec) * rec_loss
        eval_logits = x_eval[eval_known] @ weights + bias
        top1, top5 = topk_accuracy(eval_logits, encoded_eval[eval_known], k5=min(5, num_classes))
        log_stage(
            f"[train] {method} epoch {epoch}/{epochs}: "
            f"lr={epoch_lr:.6g}, loss={epoch_loss:.4f}, clean_loss={clean_loss:.4f}, "
            f"rec_loss={rec_loss:.4f}, top1={top1:.4f}, top5={top5:.4f}"
        )
        logs.append(
            {
                "method": method,
                "epoch": epoch,
                "lr": epoch_lr,
                "loss": epoch_loss,
                "clean_loss": clean_loss,
                "rec_loss": rec_loss,
                "top1": top1,
                "top5": top5,
                "train_samples": int(len(train_idx)),
                "clean_samples": int(clean_mask.sum()),
                "recover_samples": int(recover_mask.sum()),
                "eval_samples": int(eval_known.sum()),
            }
        )

    return logs, LinearModel(weights=weights, bias=bias, mean=mean, std=std, classes=classes)


def learning_rate_for_epoch(cfg: dict, epoch: int, total_epochs: int) -> float:
    base_lr = float(cfg["train"]["lr"])
    min_lr = float(cfg["train"].get("min_lr", 0.0))
    scheduler = str(cfg["train"].get("scheduler", "none")).lower()
    if total_epochs <= 1 or scheduler == "none":
        return base_lr
    progress = (epoch - 1) / max(1, total_epochs - 1)
    if scheduler == "linear":
        return min_lr + (base_lr - min_lr) * (1.0 - progress)
    if scheduler == "cosine":
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))
    raise ValueError(f"Unsupported train.scheduler: {scheduler}")


def predict_logits(model: LinearModel, features: np.ndarray) -> np.ndarray:
    x = standardize_apply(features.astype(np.float32), model.mean, model.std)
    return x @ model.weights + model.bias


def true_class_scores(logits: np.ndarray, labels: np.ndarray, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded = encode_with_classes(labels, classes)
    probs = softmax(logits)
    confidence = np.zeros(len(labels), dtype=np.float32)
    loss = np.full(len(labels), np.inf, dtype=np.float32)
    known = encoded >= 0
    confidence[known] = probs[np.where(known)[0], encoded[known]]
    loss[known] = -np.log(confidence[known] + 1.0e-12)
    return confidence, loss


def summarize_epoch_logs(method: str, logs: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    if not logs:
        raise ValueError(f"No logs available for {method}.")
    best = max(logs, key=lambda row: float(row["top1"]))
    final = logs[-1]
    last10 = logs[-10:]
    last10_top1 = np.array([float(row["top1"]) for row in last10], dtype=np.float32)
    return {
        "method": method,
        "train_samples": int(final["train_samples"]),
        "eval_samples": int(final["eval_samples"]),
        "best_epoch": int(best["epoch"]),
        "best_top1": float(best["top1"]),
        "best_top5": float(best["top5"]),
        "final_top1": float(final["top1"]),
        "final_top5": float(final["top5"]),
        "last10_mean": float(last10_top1.mean()),
        "last10_std": float(last10_top1.std()),
    }


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes = sorted(set(labels.tolist()))
    mapping = {label: i for i, label in enumerate(classes)}
    return np.array([mapping[label] for label in labels], dtype=np.int64), classes


def encode_with_classes(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(classes)}
    return np.array([mapping.get(label, -1) for label in labels], dtype=np.int64)


def standardize(features: np.ndarray) -> np.ndarray:
    standardized, _, _ = standardize_fit(features)
    return standardized


def standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1.0e-6] = 1.0
    return (features - mean) / std, mean, std


def standardize_apply(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (features - mean) / std


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, k5: int) -> tuple[float, float]:
    order = np.argsort(-logits, axis=1)
    top1 = float(np.mean(order[:, 0] == labels))
    top5 = float(np.mean([label in row[:k5] for label, row in zip(labels, order)]))
    return top1, top5
