from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json, write_yaml
from gcdd.progress import log_stage
from gcdd.training import summarize_epoch_logs, train_linear_partial_label_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V1.8 partial-label recovery from GCDD non-clean samples.")
    parser.add_argument("--input-dir", default="outputs/Web-Bird/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/recover_partial_label.")
    parser.add_argument("--lambda-rec", default="0.25,0.5", help="Comma-separated recovery loss weights.")
    parser.add_argument("--q-alt-min", type=float, default=0.30, help="Minimum top non-original global-neighbor label ratio.")
    parser.add_argument("--candidate-min-prop", type=float, default=0.10, help="Minimum global-neighbor label ratio for candidate labels.")
    parser.add_argument("--entropy-max", type=float, default=0.70, help="Maximum normalized neighbor-label entropy for safe recover.")
    parser.add_argument("--global-density-percentile-min", type=float, default=50.0, help="Minimum global-density percentile for safe recover.")
    parser.add_argument("--min-candidates", type=int, default=2, help="Minimum candidate label set size.")
    parser.add_argument("--max-candidates", type=int, default=6, help="Maximum candidate label set size.")
    parser.add_argument("--global-top-k", type=int, default=50, help="Number of global neighbors used to build candidate labels.")
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
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "recover_partial_label"
    ensure_dir(output_dir)

    lambdas = parse_float_list(args.lambda_rec, name="--lambda-rec")
    cfg = load_resolved_config(input_dir)
    apply_train_overrides(cfg, args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/5] Loading V1 outputs.")
    data = load_data(input_dir)
    validate_args(args, data)

    log_stage("[2/5] Building global-neighbor candidate-label statistics.")
    candidate_data = build_candidate_data(data, args)
    write_candidate_stats(output_dir / "recover_candidate_stats.csv", data, candidate_data)

    recover_masks = {
        "recover_top_qalt": candidate_data["base_recoverable"],
        "safe_recover": candidate_data["safe_recoverable"],
    }
    for name, mask in recover_masks.items():
        write_recoverable_samples(output_dir / f"{name}_samples.csv", data, candidate_data, mask)

    log_stage("[3/5] Training clean CE plus partial-label recovery variants.")
    train_logs, result_rows = train_recovery_variants(input_dir, cfg, data, candidate_data, recover_masks, lambdas)

    log_stage("[4/5] Writing logs and comparison tables.")
    write_csv(
        output_dir / "train_log.csv",
        train_logs,
        ["method", "epoch", "lr", "loss", "clean_loss", "rec_loss", "top1", "top5", "train_samples", "clean_samples", "recover_samples", "eval_samples"],
    )
    write_csv(
        output_dir / "recover_results.csv",
        result_rows,
        [
            "method",
            "recover_set",
            "lambda_rec",
            "clean_samples",
            "recover_samples",
            "train_samples",
            "recover_ratio_vs_clean",
            "mean_candidate_size",
            "mean_Q_alt",
            "mean_entropy",
            "mean_D_global_percentile",
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
    write_combined_compare(input_dir, output_dir, result_rows)

    log_stage("[5/5] Writing V1.8 summary.")
    write_summary(output_dir / "run_summary.md", input_dir, args, data, candidate_data, recover_masks, lambdas, result_rows)
    write_json(
        output_dir / "v1_8_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "lambda_rec": lambdas,
            "q_alt_min": args.q_alt_min,
            "candidate_min_prop": args.candidate_min_prop,
            "entropy_max": args.entropy_max,
            "global_density_percentile_min": args.global_density_percentile_min,
            "recover_counts": {name: int(mask.sum()) for name, mask in recover_masks.items()},
            "result_rows": result_rows,
        },
    )
    log_stage(f"V1.8 partial-label recovery written to {output_dir}")


def parse_float_list(raw: str, name: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError(f"{name} requires at least one value.")
    return values


def validate_args(args: argparse.Namespace, data: dict[str, Any]) -> None:
    if args.global_top_k <= 0:
        raise ValueError("--global-top-k must be positive.")
    if args.global_top_k > data["global_indices"].shape[1]:
        raise ValueError(f"--global-top-k={args.global_top_k} exceeds available global neighbors {data['global_indices'].shape[1]}.")
    if args.min_candidates < 1 or args.max_candidates < args.min_candidates:
        raise ValueError("Candidate size bounds are invalid.")


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


def load_data(input_dir: Path) -> dict[str, Any]:
    required = [
        "full_gcdd_clean_split.csv",
        "gcdd_scores.csv",
        "labels.npy",
        "eval_labels.npy",
        "paths.txt",
        "global_knn_indices.npy",
        "global_knn_weights.npy",
    ]
    for name in required:
        path = input_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required file is missing: {path}")

    score_rows = sorted(read_csv(input_dir / "gcdd_scores.csv"), key=lambda row: int(row["index"]))
    labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    n = len(labels)
    if len(score_rows) != n or len(paths) != n:
        raise ValueError("gcdd_scores.csv, labels.npy, and paths.txt have inconsistent lengths.")
    return {
        "index": np.array([int(row["index"]) for row in score_rows], dtype=np.int64),
        "path": np.array([row["path"] for row in score_rows], dtype=object),
        "labels": labels,
        "eval_labels": np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str),
        "S_clean": float_array(score_rows, "S_clean"),
        "Q_same": float_array(score_rows, "Q_same"),
        "full_gcdd_clean": read_clean_mask(input_dir / "full_gcdd_clean_split.csv", n),
        "global_indices": np.load(input_dir / "global_knn_indices.npy"),
        "global_weights": np.load(input_dir / "global_knn_weights.npy"),
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


def build_candidate_data(data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    labels = data["labels"]
    classes = sorted(set(labels.tolist()))
    class_to_col = {label: i for i, label in enumerate(classes)}
    n = len(labels)
    candidate_mask = np.zeros((n, len(classes)), dtype=bool)
    q_alt = np.zeros(n, dtype=np.float32)
    c_alt = np.array([""] * n, dtype=object)
    entropy = np.zeros(n, dtype=np.float32)
    candidate_size = np.zeros(n, dtype=np.int32)
    candidate_labels = np.array([""] * n, dtype=object)
    global_density = global_density_scores(data["global_indices"], data["global_weights"], args.global_top_k)
    density_percentile = percentile(global_density)

    for i in range(n):
        valid = data["global_indices"][i, : args.global_top_k]
        valid = valid[valid >= 0]
        if len(valid) == 0:
            continue
        counts = Counter(labels[valid].tolist())
        total = float(sum(counts.values()))
        proportions = {label: count / total for label, count in counts.items()}

        alternatives = [(label, prop) for label, prop in proportions.items() if label != labels[i]]
        if alternatives:
            best_label, best_prop = sorted(alternatives, key=lambda item: (-item[1], item[0]))[0]
            q_alt[i] = float(best_prop)
            c_alt[i] = best_label

        entropy[i] = normalized_entropy(np.array(list(proportions.values()), dtype=np.float32))
        candidates = {label for label, prop in proportions.items() if prop >= args.candidate_min_prop}
        candidates.add(labels[i])
        candidate_size[i] = len(candidates)
        candidate_labels[i] = "|".join(sorted(candidates))
        for label in candidates:
            if label in class_to_col:
                candidate_mask[i, class_to_col[label]] = True

    size_ok = (candidate_size >= args.min_candidates) & (candidate_size <= args.max_candidates)
    non_clean = ~data["full_gcdd_clean"]
    base_recoverable = non_clean & size_ok & (q_alt >= args.q_alt_min)
    safe_recoverable = base_recoverable & (entropy <= args.entropy_max) & (density_percentile >= args.global_density_percentile_min)
    return {
        "classes": classes,
        "candidate_mask": candidate_mask,
        "candidate_labels": candidate_labels,
        "candidate_size": candidate_size,
        "Q_alt": q_alt,
        "c_alt": c_alt,
        "neighbor_entropy": entropy,
        "D_global": global_density,
        "D_global_percentile": density_percentile,
        "base_recoverable": base_recoverable,
        "safe_recoverable": safe_recoverable,
    }


def global_density_scores(indices: np.ndarray, weights: np.ndarray, top_k: int) -> np.ndarray:
    valid = indices[:, :top_k] >= 0
    denom = valid.sum(axis=1)
    totals = (weights[:, :top_k] * valid).sum(axis=1)
    out = np.zeros(indices.shape[0], dtype=np.float32)
    np.divide(totals, denom, out=out, where=denom != 0)
    return out


def percentile(values: np.ndarray) -> np.ndarray:
    if len(values) == 1:
        return np.ones(1, dtype=np.float32) * 100.0
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32) * 100.0 / (len(values) - 1)
    return ranks


def normalized_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / float(np.log(len(probs)))


def write_candidate_stats(path: Path, data: dict[str, Any], cdata: dict[str, Any]) -> None:
    rows = []
    for i in range(len(data["labels"])):
        rows.append(candidate_row(data, cdata, i, extra={}))
    write_csv(path, rows, candidate_fieldnames(extra_fields=[]))


def write_recoverable_samples(path: Path, data: dict[str, Any], cdata: dict[str, Any], mask: np.ndarray) -> None:
    rows = []
    for i in np.where(mask)[0]:
        rows.append(candidate_row(data, cdata, i, extra={"is_recoverable": 1}))
    write_csv(path, rows, candidate_fieldnames(extra_fields=["is_recoverable"]))


def candidate_row(data: dict[str, Any], cdata: dict[str, Any], i: int, extra: dict[str, Any]) -> dict[str, Any]:
    row = {
        "index": int(data["index"][i]),
        "path": data["path"][i],
        "web_label": data["labels"][i],
        "state": "clean" if data["full_gcdd_clean"][i] else "ignored",
        "S_clean": float(data["S_clean"][i]),
        "Q_same": float(data["Q_same"][i]),
        "Q_alt": float(cdata["Q_alt"][i]),
        "c_alt": cdata["c_alt"][i],
        "neighbor_entropy": float(cdata["neighbor_entropy"][i]),
        "D_global": float(cdata["D_global"][i]),
        "D_global_percentile": float(cdata["D_global_percentile"][i]),
        "candidate_size": int(cdata["candidate_size"][i]),
        "candidate_labels": cdata["candidate_labels"][i],
    }
    row.update(extra)
    return row


def candidate_fieldnames(extra_fields: list[str]) -> list[str]:
    return [
        "index",
        "path",
        "web_label",
        "state",
        "S_clean",
        "Q_same",
        "Q_alt",
        "c_alt",
        "neighbor_entropy",
        "D_global",
        "D_global_percentile",
        "candidate_size",
        "candidate_labels",
        *extra_fields,
    ]


def train_recovery_variants(
    input_dir: Path,
    cfg: dict[str, Any],
    data: dict[str, Any],
    cdata: dict[str, Any],
    recover_masks: dict[str, np.ndarray],
    lambdas: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_name = cfg.get("train", {}).get("feature", "cls")
    train_features = load_feature(input_dir, "", feature_name)
    eval_features = load_feature(input_dir, "eval_", feature_name)
    train_logs: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for recover_name, recover_mask in recover_masks.items():
        for lambda_rec in lambdas:
            method = f"{recover_name}_lambda_{lambda_rec:g}"
            logs, _ = train_linear_partial_label_eval(
                train_features,
                data["labels"],
                eval_features,
                data["eval_labels"],
                data["full_gcdd_clean"],
                recover_mask,
                cdata["candidate_mask"],
                lambda_rec,
                cfg,
                method,
            )
            train_logs.extend(logs)
            row = summarize_epoch_logs(method, logs)
            recover_idx = np.where(recover_mask)[0]
            row["recover_set"] = recover_name
            row["lambda_rec"] = float(lambda_rec)
            row["clean_samples"] = int(data["full_gcdd_clean"].sum())
            row["recover_samples"] = int(recover_mask.sum())
            row["recover_ratio_vs_clean"] = safe_ratio(int(recover_mask.sum()), int(data["full_gcdd_clean"].sum()))
            row["mean_candidate_size"] = safe_mean(cdata["candidate_size"][recover_idx])
            row["mean_Q_alt"] = safe_mean(cdata["Q_alt"][recover_idx])
            row["mean_entropy"] = safe_mean(cdata["neighbor_entropy"][recover_idx])
            row["mean_D_global_percentile"] = safe_mean(cdata["D_global_percentile"][recover_idx])
            results.append(row)
    return train_logs, results


def load_feature(input_dir: Path, prefix: str, feature_name: str) -> np.ndarray:
    path = input_dir / f"{prefix}features_{feature_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Feature file is missing: {path}")
    return np.load(path)


def write_combined_compare(input_dir: Path, output_dir: Path, result_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    baseline_path = input_dir / "baseline_compare_web_bird.csv"
    if baseline_path.exists():
        for row in read_csv(baseline_path):
            if row["method"] in ["DINOv2 Linear all", "Centroid filtering", "Full GCDD-clean"]:
                out = dict(row)
                out["source"] = "existing_v1"
                out["recover_set"] = ""
                out["lambda_rec"] = ""
                out["recover_samples"] = ""
                rows.append(out)
    for row in result_rows:
        out = dict(row)
        out["source"] = "v1_8_recover"
        rows.append(out)
    write_csv(
        output_dir / "combined_compare_web_bird.csv",
        rows,
        [
            "source",
            "method",
            "recover_set",
            "lambda_rec",
            "train_samples",
            "clean_samples",
            "recover_samples",
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
    args: argparse.Namespace,
    data: dict[str, Any],
    cdata: dict[str, Any],
    recover_masks: dict[str, np.ndarray],
    lambdas: list[float],
    result_rows: list[dict[str, Any]],
) -> None:
    clean_count = int(data["full_gcdd_clean"].sum())
    lines = [
        "# V1.8 Partial-Label Recovery Summary",
        "",
        "V1.8 keeps the current Full GCDD-clean CE training set and adds partial-label learning for selected non-clean samples.",
        "Recoverable samples are selected by global-neighbor label concentration, then trained with -log sum p(c) over a conservative candidate label set.",
        "",
        f"- Source V1 output: {input_dir}",
        f"- Clean samples: {clean_count}",
        f"- lambda_rec: {', '.join(str(value) for value in lambdas)}",
        f"- Q_alt min: {args.q_alt_min}",
        f"- Candidate min prop: {args.candidate_min_prop}",
        f"- Candidate size: [{args.min_candidates}, {args.max_candidates}]",
        f"- Safe entropy max: {args.entropy_max}",
        f"- Safe D_global percentile min: {args.global_density_percentile_min}",
        "",
        "## Recoverable Counts",
        "| recover_set | count | ratio_vs_clean | mean_candidate_size | mean_Q_alt | mean_entropy | mean_D_global_percentile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, mask in recover_masks.items():
        idx = np.where(mask)[0]
        lines.append(
            f"| {name} | {int(mask.sum())} | {safe_ratio(int(mask.sum()), clean_count):.4f} | "
            f"{safe_mean(cdata['candidate_size'][idx]):.2f} | {safe_mean(cdata['Q_alt'][idx]):.4f} | "
            f"{safe_mean(cdata['neighbor_entropy'][idx]):.4f} | {safe_mean(cdata['D_global_percentile'][idx]):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Results",
            "| method | recover_samples | lambda_rec | best_top1 | final_top1 | last10_mean | best_epoch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result_rows:
        lines.append(
            f"| {row['method']} | {int(row['recover_samples'])} | {float(row['lambda_rec']):.2f} | "
            f"{float(row['best_top1']):.4f} | {float(row['final_top1']):.4f} | "
            f"{float(row['last10_mean']):.4f} | {int(row['best_epoch'])} |"
        )
    best = max(result_rows, key=lambda row: float(row["best_top1"])) if result_rows else None
    if best is not None:
        lines.extend(["", "## Immediate Read", f"- Best recovery method: {best['method']} with best_top1={float(best['best_top1']):.4f}."])
    path.write_text("\n".join(lines), encoding="utf-8")


def safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else 0.0


def safe_ratio(num: int | float, denom: int | float) -> float:
    return float(num / denom) if denom else 0.0


if __name__ == "__main__":
    main()
