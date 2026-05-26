from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json, write_yaml
from gcdd.progress import log_stage
from gcdd.training import summarize_epoch_logs, train_linear_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V1.7 qp-risk soft weighting on existing V1 outputs.")
    parser.add_argument("--input-dir", default="outputs/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--gate-dir", help="V1.6 gated split directory. Defaults to <input-dir>/v1_6_gated_splits.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/qp_soft_weighting.")
    parser.add_argument("--alphas", default="0.3,0.5,0.7", help="Comma-separated qp-risk weights.")
    parser.add_argument("--epochs", type=int, help="Override linear classifier epochs.")
    parser.add_argument("--train-batch-size", type=int, help="Override linear classifier batch size.")
    parser.add_argument("--lr", type=float, help="Override linear classifier learning rate.")
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], help="Override learning-rate scheduler.")
    parser.add_argument("--seed", type=int, help="Override train seed.")
    parser.add_argument("--feature", choices=["cls", "gap", "top"], help="Feature file used for linear training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    gate_dir = Path(args.gate_dir) if args.gate_dir else input_dir / "v1_6_gated_splits"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "qp_soft_weighting"
    ensure_dir(output_dir)

    alphas = parse_alphas(args.alphas)
    cfg = load_resolved_config(input_dir)
    apply_train_overrides(cfg, args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/4] Loading V1/V1.6 outputs and defining qp-risk samples.")
    data = load_data(input_dir, gate_dir)
    qp_risk = data["full_gcdd_clean"] & ~data["qp_gate_clean"]
    if int(qp_risk.sum()) == 0:
        raise ValueError("qp-risk set is empty. Check the qp_gate split.")
    write_qp_risk_indices(output_dir / "qp_risk_indices.csv", data, qp_risk)

    log_stage("[2/4] Writing sample-weight tables.")
    weights_by_alpha = {}
    for alpha in alphas:
        weights = build_weights(data["full_gcdd_clean"], qp_risk, alpha)
        weights_by_alpha[alpha] = weights
        write_sample_weights(output_dir / f"sample_weight_alpha_{alpha:g}.csv", data, data["full_gcdd_clean"], qp_risk, weights)

    log_stage("[3/4] Training qp-risk soft weighting variants.")
    train_logs, method_summaries = train_soft_variants(input_dir, cfg, data, qp_risk, weights_by_alpha)
    write_csv(
        output_dir / "train_log.csv",
        train_logs,
        ["method", "epoch", "lr", "loss", "top1", "top5", "train_samples", "eval_samples"],
    )
    write_csv(
        output_dir / "qp_soft_results.csv",
        method_summaries,
        [
            "method",
            "alpha",
            "train_samples",
            "effective_train_weight",
            "num_qp_risk",
            "qp_risk_ratio",
            "best_top1",
            "final_top1",
            "last10_mean",
            "last10_std",
            "best_epoch",
            "best_top5",
            "final_top5",
            "eval_samples",
        ],
    )
    write_combined_compare(input_dir, gate_dir, output_dir, method_summaries)

    log_stage("[4/4] Writing V1.7 summary.")
    write_summary(output_dir / "run_summary.md", input_dir, gate_dir, alphas, data, qp_risk, method_summaries)
    write_json(
        output_dir / "v1_7_summary.json",
        {
            "input_dir": str(input_dir),
            "gate_dir": str(gate_dir),
            "output_dir": str(output_dir),
            "alphas": alphas,
            "num_full_gcdd_clean": int(data["full_gcdd_clean"].sum()),
            "num_qp_risk": int(qp_risk.sum()),
            "qp_risk_ratio": safe_ratio(int(qp_risk.sum()), int(data["full_gcdd_clean"].sum())),
            "method_summaries": method_summaries,
        },
    )
    log_stage(f"V1.7 qp-risk soft weighting written to {output_dir}")


def parse_alphas(raw: str) -> list[float]:
    alphas = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError("Alpha values must be non-negative.")
        alphas.append(value)
    if not alphas:
        raise ValueError("At least one alpha is required.")
    return alphas


def load_resolved_config(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "resolved_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"V1 resolved config is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_train_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    train_cfg = cfg.setdefault("train", {})
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.train_batch_size is not None:
        train_cfg["batch_size"] = args.train_batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.scheduler is not None:
        train_cfg["scheduler"] = args.scheduler
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.feature is not None:
        train_cfg["feature"] = args.feature


def load_data(input_dir: Path, gate_dir: Path) -> dict[str, Any]:
    required = [
        input_dir / "full_gcdd_clean_split.csv",
        gate_dir / "gcdd_qp_gate_split.csv",
        input_dir / "gcdd_scores.csv",
        input_dir / "centroid_scores.csv",
        input_dir / "linear_all_train_scores.csv",
        input_dir / "labels.npy",
        input_dir / "eval_labels.npy",
        input_dir / "paths.txt",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file is missing: {path}")

    score_rows = sorted(read_csv(input_dir / "gcdd_scores.csv"), key=lambda row: int(row["index"]))
    centroid_rows = sorted(read_csv(input_dir / "centroid_scores.csv"), key=lambda row: int(row["index"]))
    train_score_rows = sorted(read_csv(input_dir / "linear_all_train_scores.csv"), key=lambda row: int(row["index"]))
    labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    n = len(labels)
    if len(score_rows) != n or len(centroid_rows) != n or len(train_score_rows) != n or len(paths) != n:
        raise ValueError("V1 score, label, path, centroid, and train-score files have inconsistent lengths.")
    return {
        "index": np.array([int(row["index"]) for row in score_rows], dtype=np.int64),
        "path": np.array([row["path"] for row in score_rows], dtype=object),
        "labels": labels,
        "eval_labels": np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str),
        "S_clean": float_array(score_rows, "S_clean"),
        "Q_same": float_array(score_rows, "Q_same"),
        "centroid_score": float_array(centroid_rows, "centroid_score"),
        "loss": float_array(train_score_rows, "loss"),
        "confidence": float_array(train_score_rows, "confidence"),
        "full_gcdd_clean": read_clean_mask(input_dir / "full_gcdd_clean_split.csv", n),
        "qp_gate_clean": read_clean_mask(gate_dir / "gcdd_qp_gate_split.csv", n),
    }


def float_array(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.array([float(row[field]) for row in rows], dtype=np.float32)


def read_clean_mask(path: Path, n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for row in read_csv(path):
        idx = int(row["index"])
        if idx >= n:
            raise ValueError(f"Index {idx} in {path} exceeds expected length {n}.")
        mask[idx] = row["state"] == "clean"
    return mask


def build_weights(clean_mask: np.ndarray, qp_risk: np.ndarray, alpha: float) -> np.ndarray:
    weights = np.zeros(len(clean_mask), dtype=np.float32)
    weights[clean_mask] = 1.0
    weights[qp_risk] = alpha
    return weights


def write_qp_risk_indices(path: Path, data: dict[str, Any], qp_risk: np.ndarray) -> None:
    rows = []
    for i in np.where(data["full_gcdd_clean"])[0]:
        rows.append(
            {
                "index": int(data["index"][i]),
                "path": data["path"][i],
                "web_label": data["labels"][i],
                "S_clean": float(data["S_clean"][i]),
                "Q_same": float(data["Q_same"][i]),
                "centroid_score": float(data["centroid_score"][i]),
                "loss": float(data["loss"][i]),
                "confidence": float(data["confidence"][i]),
                "is_qp_risk": int(qp_risk[i]),
            }
        )
    write_csv(path, rows, ["index", "path", "web_label", "S_clean", "Q_same", "centroid_score", "loss", "confidence", "is_qp_risk"])


def write_sample_weights(path: Path, data: dict[str, Any], clean_mask: np.ndarray, qp_risk: np.ndarray, weights: np.ndarray) -> None:
    rows = []
    for i in range(len(clean_mask)):
        rows.append(
            {
                "index": int(data["index"][i]),
                "path": data["path"][i],
                "web_label": data["labels"][i],
                "state": "clean" if clean_mask[i] else "ignored",
                "weight": float(weights[i]),
                "is_qp_risk": int(qp_risk[i]),
            }
        )
    write_csv(path, rows, ["index", "path", "web_label", "state", "weight", "is_qp_risk"])


def train_soft_variants(
    input_dir: Path,
    cfg: dict[str, Any],
    data: dict[str, Any],
    qp_risk: np.ndarray,
    weights_by_alpha: dict[float, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_name = cfg.get("train", {}).get("feature", "cls")
    train_features = load_feature(input_dir, "", feature_name)
    eval_features = load_feature(input_dir, "eval_", feature_name)
    train_logs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    num_qp_risk = int(qp_risk.sum())
    train_samples = int(data["full_gcdd_clean"].sum())
    normal_count = train_samples - num_qp_risk
    for alpha, weights in weights_by_alpha.items():
        method = f"qp_soft_{alpha:g}"
        logs, _ = train_linear_eval(
            train_features,
            data["labels"],
            eval_features,
            data["eval_labels"],
            data["full_gcdd_clean"],
            cfg,
            method,
            sample_weights=weights,
        )
        train_logs.extend(logs)
        row = summarize_epoch_logs(method, logs)
        row["alpha"] = float(alpha)
        row["effective_train_weight"] = float(normal_count + alpha * num_qp_risk)
        row["num_qp_risk"] = num_qp_risk
        row["qp_risk_ratio"] = safe_ratio(num_qp_risk, train_samples)
        summaries.append(row)
    return train_logs, summaries


def load_feature(input_dir: Path, prefix: str, feature_name: str) -> np.ndarray:
    path = input_dir / f"{prefix}features_{feature_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Feature file is missing: {path}")
    return np.load(path)


def write_combined_compare(input_dir: Path, gate_dir: Path, output_dir: Path, method_summaries: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    baseline_path = input_dir / "baseline_compare_web_bird.csv"
    if baseline_path.exists():
        for row in read_csv(baseline_path):
            if row["method"] in ["DINOv2 Linear all", "Centroid filtering", "Full GCDD-clean"]:
                out = dict(row)
                out["source"] = "existing_v1"
                out["alpha"] = ""
                out["effective_train_weight"] = ""
                out["num_qp_risk"] = ""
                out["qp_risk_ratio"] = ""
                rows.append(out)
    gate_path = gate_dir / "gated_compare_web_bird.csv"
    if gate_path.exists():
        for row in read_csv(gate_path):
            if row["method"] == "GCDD + Q_same AND centroid gate":
                out = dict(row)
                out["source"] = "existing_v1_6"
                out["alpha"] = "delete"
                out["effective_train_weight"] = row.get("train_samples", "")
                out["num_qp_risk"] = ""
                out["qp_risk_ratio"] = ""
                rows.append(out)
    for row in method_summaries:
        out = dict(row)
        out["source"] = "v1_7_soft"
        rows.append(out)
    write_csv(
        output_dir / "combined_compare_web_bird.csv",
        rows,
        [
            "source",
            "method",
            "alpha",
            "train_samples",
            "effective_train_weight",
            "num_qp_risk",
            "qp_risk_ratio",
            "eval_samples",
            "best_epoch",
            "best_top1",
            "best_top5",
            "final_top1",
            "final_top5",
            "last10_mean",
            "last10_std",
        ],
    )


def write_summary(
    path: Path,
    input_dir: Path,
    gate_dir: Path,
    alphas: list[float],
    data: dict[str, Any],
    qp_risk: np.ndarray,
    method_summaries: list[dict[str, Any]],
) -> None:
    train_samples = int(data["full_gcdd_clean"].sum())
    num_qp_risk = int(qp_risk.sum())
    normal_count = train_samples - num_qp_risk
    lines = [
        "# V1.7 QP-Risk Soft Weighting Summary",
        "",
        "V1.7 tests whether qp-risk samples should be deleted or softly down-weighted.",
        "The qp-risk set is defined as samples selected by Full GCDD-clean but removed by the conservative Q_same AND centroid gate.",
        "We keep the original Full GCDD-clean training set unchanged and assign lower CE weights to qp-risk samples.",
        "This isolates whether hard deletion is too aggressive while avoiding changes to the clean selection rule.",
        "",
        "中文总结：V1.7 用于验证 qp-risk 样本是否应被硬删除；实验保持原 GCDD-clean 训练集不变，只降低 qp-risk 样本的 CE 权重。",
        "",
        f"- Source V1 output: {input_dir}",
        f"- Source V1.6 gate output: {gate_dir}",
        f"- Alphas: {', '.join(str(alpha) for alpha in alphas)}",
        f"- Full GCDD-clean train samples: {train_samples}",
        f"- qp-risk samples: {num_qp_risk}",
        f"- qp-risk ratio: {safe_ratio(num_qp_risk, train_samples):.4f}",
        "",
        "## Results",
        "| method | alpha | train_samples | effective_train_weight | best_top1 | final_top1 | last10_mean | best_epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in method_summaries:
        lines.append(
            f"| {row['method']} | {float(row['alpha']):.2f} | {int(row['train_samples'])} | "
            f"{float(row['effective_train_weight']):.1f} | {float(row['best_top1']):.4f} | "
            f"{float(row['final_top1']):.4f} | {float(row['last10_mean']):.4f} | {int(row['best_epoch'])} |"
        )
    best = max(method_summaries, key=lambda row: float(row["best_top1"]))
    lines.extend(
        [
            "",
            "## Fixed Counts",
            f"- Normal clean samples: {normal_count}",
            f"- qp-risk samples: {num_qp_risk}",
            "",
            "## Output Files",
            "- `qp_risk_indices.csv`",
            "- `sample_weight_alpha_0.3.csv`, `sample_weight_alpha_0.5.csv`, `sample_weight_alpha_0.7.csv`",
            "- `qp_soft_results.csv`",
            "- `combined_compare_web_bird.csv`",
            "- `train_log.csv`",
            "",
            "## Immediate Read",
            f"- Best soft method: {best['method']} with best_top1={float(best['best_top1']):.4f}.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def safe_ratio(num: int | float, denom: int | float) -> float:
    return float(num / denom) if denom else 0.0


if __name__ == "__main__":
    main()
