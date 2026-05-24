from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
        return [{"epoch": 1, "loss": 0.0, "top1": 1.0, "top5": 1.0, "train_samples": int(clean_mask.sum())}]

    x = standardize(features.astype(np.float32))
    train_idx = np.where(clean_mask)[0]
    if len(train_idx) == 0:
        raise ValueError("Clean split is empty; cannot run V0 training smoke test.")

    weights = rng.normal(scale=0.01, size=(x.shape[1], num_classes)).astype(np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    lr = float(cfg["train"]["lr"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    logs: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
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
            weights -= lr * (x[idx].T @ grad)
            bias -= lr * grad.sum(axis=0)

        eval_logits = x @ weights + bias
        top1, top5 = topk_accuracy(eval_logits, encoded, k5=min(5, num_classes))
        logs.append(
            {
                "epoch": epoch,
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
    train_idx = np.where(train_mask)[0]
    if len(train_idx) == 0:
        raise ValueError(f"{method} selected no training samples.")

    num_classes = len(classes)
    weights = rng.normal(scale=0.01, size=(x_train.shape[1], num_classes)).astype(np.float32)
    bias = np.zeros(num_classes, dtype=np.float32)
    lr = float(cfg["train"]["lr"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    logs: list[dict[str, float | int | str]] = []

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            logits = x_train[idx] @ weights + bias
            probs = softmax(logits)
            y = encoded_train[idx]
            losses.append(float(-np.log(probs[np.arange(len(idx)), y] + 1.0e-12).mean()))

            grad = probs
            grad[np.arange(len(idx)), y] -= 1.0
            grad /= len(idx)
            weights -= lr * (x_train[idx].T @ grad)
            bias -= lr * grad.sum(axis=0)

        eval_logits = x_eval[eval_known] @ weights + bias
        top1, top5 = topk_accuracy(eval_logits, encoded_eval[eval_known], k5=min(5, num_classes))
        logs.append(
            {
                "method": method,
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "top1": top1,
                "top5": top5,
                "train_samples": int(len(train_idx)),
                "eval_samples": int(eval_known.sum()),
            }
        )

    return logs, LinearModel(weights=weights, bias=bias, mean=mean, std=std, classes=classes)


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
