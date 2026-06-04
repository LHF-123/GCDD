from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.io_utils import ensure_dir, read_csv, write_csv, write_json, write_yaml
from gcdd.lora_dynamic import should_update_selection, train_dynamic_loss_lora
from gcdd.progress import log_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prototype-guided Dynamic Filtering with DINOv2-LoRA.")
    parser.add_argument("--input-dir", default="outputs/Web-Car/v1_web_car_0.9_448", help="V1 output directory.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/proto_guided_dynamic.")
    parser.add_argument("--retention-ratios", default="0.8", help="Comma-separated dynamic loss retention ratios.")
    parser.add_argument("--proto-keep-ratios", default="0.9,0.8", help="Comma-separated class-wise prototype keep ratios.")
    parser.add_argument("--auto-proto-keep", action="store_true", help="Automatically choose prototype keep ratio from initial dynamic/prototype overlap.")
    parser.add_argument("--auto-high-jaccard", type=float, default=0.75, help="If Jaccard >= this value, use --auto-p-high.")
    parser.add_argument("--auto-mid-jaccard", type=float, default=0.60, help="If Jaccard >= this value, use --auto-p-mid.")
    parser.add_argument("--auto-low-jaccard", type=float, default=0.50, help="If Jaccard >= this value, use --auto-p-low; otherwise use --auto-p-very-low.")
    parser.add_argument("--auto-p-high", type=float, default=0.8, help="Auto-selected p for high dynamic/prototype agreement.")
    parser.add_argument("--auto-p-mid", type=float, default=0.6, help="Auto-selected p for medium dynamic/prototype agreement.")
    parser.add_argument("--auto-p-low", type=float, default=0.5, help="Auto-selected p for low dynamic/prototype agreement.")
    parser.add_argument("--auto-p-very-low", type=float, default=0.4, help="Auto-selected p for very low dynamic/prototype agreement.")
    parser.add_argument("--proto-scores", default="proto_gcdd/proto_gcdd_scores.csv", help="Prototype score CSV relative to input-dir.")
    parser.add_argument("--proto-score-col", default="centroid_score", help="Column used as prototype score. Higher is safer.")
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Epochs trained on all candidates before first selection update.")
    parser.add_argument("--update-interval", type=int, default=5, help="Selection update interval in epochs after warm-up.")
    parser.add_argument("--centroid-split", default="centroid_filtering_split.csv", help="Optional centroid split file relative to input-dir for overlap.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map stored image root to local root.")
    parser.add_argument("--no-save-checkpoints", action="store_true", help="Do not save best LoRA/head checkpoint files.")

    parser.add_argument("--epochs", type=int, help="Override LoRA epochs.")
    parser.add_argument("--batch-size", type=int, help="Override LoRA train batch size.")
    parser.add_argument("--eval-batch-size", type=int, help="Override LoRA eval batch size.")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers.")
    parser.add_argument("--lora-lr", type=float, help="Override LoRA parameter learning rate.")
    parser.add_argument("--head-lr", type=float, help="Override classifier head learning rate.")
    parser.add_argument("--weight-decay", type=float, help="Override AdamW weight decay.")
    parser.add_argument("--rank", type=int, help="Override LoRA rank.")
    parser.add_argument("--alpha", type=float, help="Override LoRA alpha.")
    parser.add_argument("--dropout", type=float, help="Override LoRA dropout.")
    parser.add_argument("--target-modules", help="Comma-separated target module patterns. Default qkv.")
    parser.add_argument("--scheduler", choices=["none", "linear", "cosine"], help="Override LoRA scheduler.")
    parser.add_argument("--warmup-ratio", type=float, help="Override LR warmup ratio.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override device.")
    parser.add_argument("--local-repo", help="Override DINOv2 local torch hub repo path.")
    parser.add_argument("--input-size", type=int, help="Override image input size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "proto_guided_dynamic"
    ensure_dir(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    if not args.no_save_checkpoints:
        ensure_dir(checkpoint_dir)

    ratios = parse_float_list(args.retention_ratios, "--retention-ratios")
    proto_ratios = [] if args.auto_proto_keep else parse_float_list(args.proto_keep_ratios, "--proto-keep-ratios")
    seeds = parse_int_list(args.seeds, "--seeds")
    path_maps = parse_path_maps(args.path_map)
    auto_proto_rule = build_auto_proto_rule(args) if args.auto_proto_keep else None

    cfg = load_config(input_dir)
    apply_lora_defaults(cfg)
    apply_overrides(cfg, args)
    validate_auto_proto_update_schedule(args, cfg)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/4] Loading V1 paths, labels, centroid split, and prototype scores.")
    data = load_data(input_dir, args.centroid_split)
    proto_scores = load_proto_scores(input_dir, args.proto_scores, args.proto_score_col, len(data["train_labels"]))

    log_stage("[2/4] Running Prototype-guided Dynamic Filtering LoRA.")
    train_logs: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for ratio in ratios:
        proto_items: list[float | None] = [None] if args.auto_proto_keep else list(proto_ratios)
        for proto_ratio in proto_items:
            for seed in seeds:
                ratio_text = ratio_to_text(ratio)
                proto_text = "auto" if proto_ratio is None else ratio_to_text(proto_ratio)
                method = f"DINOv2 LoRA PGDF-auto r={ratio:g}" if args.auto_proto_keep else f"DINOv2 LoRA PGDF r={ratio:g} p={proto_ratio:g}"
                checkpoint_path = None
                if not args.no_save_checkpoints:
                    checkpoint_path = checkpoint_dir / f"pgdf_r{ratio_text}_p{proto_text}_seed{seed}_best.pt"
                run_cfg = copy.deepcopy(cfg)
                run_cfg["lora_train"]["seed"] = int(seed)
                result = train_dynamic_loss_lora(
                    data["train_paths"],
                    data["train_labels"],
                    data["eval_paths"],
                    data["eval_labels"],
                    data["candidate_mask"],
                    run_cfg,
                    method=method,
                    seed=seed,
                    retention_ratio=ratio,
                    warmup_epochs=int(args.warmup_epochs),
                    update_interval=int(args.update_interval),
                    path_maps=path_maps,
                    centroid_mask=data["centroid_mask"],
                    proto_scores=proto_scores,
                    proto_keep_ratio=proto_ratio,
                    auto_proto_keep=auto_proto_rule,
                    checkpoint_path=checkpoint_path,
                )
                train_logs.extend(result.logs)
                result_rows.append(result.summary)
                update_rows.extend(result.update_rows)
                per_class_rows.extend(result.per_class_rows)
                selected_proto_text = ratio_to_text(float(result.summary["proto_keep_ratio"])) if result.summary["proto_keep_ratio"] != "" else proto_text
                write_selection_files(output_dir, ratio_text, selected_proto_text, seed, result.selection_rows)
                for name in result.trainable_modules:
                    module_rows.append(
                        {
                            "method": method,
                            "seed": seed,
                            "retention_ratio": ratio,
                            "proto_keep_ratio": result.summary["proto_keep_ratio"],
                            "auto_proto_keep": result.summary.get("auto_proto_keep", "no"),
                            "auto_proto_jaccard": result.summary.get("auto_proto_jaccard", ""),
                            "module": name,
                            "trainable_params": result.trainable_params,
                            "total_params": result.total_params,
                        }
                    )

    log_stage("[3/4] Writing PGDF result tables.")
    write_csv(output_dir / "pgdf_train_log.csv", train_logs, train_log_fields())
    write_csv(output_dir / "pgdf_results.csv", result_rows, result_fields())
    write_csv(output_dir / "pgdf_update_summary.csv", update_rows, update_fields())
    write_csv(output_dir / "pgdf_per_class_summary.csv", per_class_rows, per_class_fields())
    write_csv(output_dir / "pgdf_modules.csv", module_rows, ["method", "seed", "retention_ratio", "proto_keep_ratio", "auto_proto_keep", "auto_proto_jaccard", "module", "trainable_params", "total_params"])
    method_summary = build_method_summary(result_rows)
    write_csv(
        output_dir / "pgdf_summary.csv",
        method_summary,
        [
            "method",
            "retention_ratio",
            "proto_keep_ratio",
            "min_proto_keep_ratio",
            "max_proto_keep_ratio",
            "auto_proto_keep",
            "mean_auto_proto_jaccard",
            "num_seeds",
            "mean_best_top1",
            "std_best_top1",
            "max_best_top1",
            "mean_final_top1",
            "mean_final_selected_samples",
            "min_final_selected_samples",
            "max_final_selected_samples",
        ],
    )

    log_stage("[4/4] Writing PGDF run summary.")
    write_summary(output_dir / "run_summary.md", input_dir, args, cfg, method_summary, update_rows)
    write_json(
        output_dir / "pgdf_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "retention_ratios": ratios,
            "proto_keep_ratios": proto_ratios,
            "auto_proto_keep": args.auto_proto_keep,
            "auto_proto_rule": auto_proto_rule,
            "proto_scores": args.proto_scores,
            "proto_score_col": args.proto_score_col,
            "seeds": seeds,
            "warmup_epochs": int(args.warmup_epochs),
            "update_interval": int(args.update_interval),
            "results": result_rows,
            "summary": method_summary,
            "updates": update_rows,
        },
    )
    log_stage(f"PGDF results written to {output_dir}")


def load_config(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "resolved_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"V1 resolved config is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_auto_proto_rule(args: argparse.Namespace) -> dict[str, float]:
    rule = {
        "high_jaccard": float(args.auto_high_jaccard),
        "mid_jaccard": float(args.auto_mid_jaccard),
        "low_jaccard": float(args.auto_low_jaccard),
        "p_high": float(args.auto_p_high),
        "p_mid": float(args.auto_p_mid),
        "p_low": float(args.auto_p_low),
        "p_very_low": float(args.auto_p_very_low),
    }
    if rule["high_jaccard"] < rule["mid_jaccard"] or rule["mid_jaccard"] < rule["low_jaccard"]:
        raise ValueError("--auto-high-jaccard must be >= --auto-mid-jaccard >= --auto-low-jaccard.")
    for key, value in rule.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be in [0, 1], got {value}.")
    for key in ["p_high", "p_mid", "p_low", "p_very_low"]:
        if rule[key] <= 0.0:
            raise ValueError(f"{key} must be > 0.")
    return rule


def validate_auto_proto_update_schedule(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    if not args.auto_proto_keep:
        return
    epochs = int(cfg["lora_train"]["epochs"])
    warmup_epochs = int(args.warmup_epochs)
    update_interval = int(args.update_interval)
    has_update = any(
        epoch < epochs and should_update_selection(epoch, warmup_epochs, update_interval)
        for epoch in range(1, epochs + 1)
    )
    if not has_update:
        raise ValueError(
            "--auto-proto-keep requires at least one selection update because p is chosen from "
            "the first dynamic/prototype overlap. Increase --epochs above --warmup-epochs, "
            "or reduce --warmup-epochs."
        )


def apply_lora_defaults(cfg: dict[str, Any]) -> None:
    cfg.setdefault("feature", {})
    cfg["feature"].setdefault("backend", "dinov2_vitb14")
    cfg["feature"].setdefault("device", "auto")
    cfg["feature"].setdefault("input_size", 224)
    cfg["feature"].setdefault("local_repo", "")
    cfg.setdefault("lora", {"rank": 8, "alpha": 16.0, "dropout": 0.05, "target_modules": "qkv"})
    cfg.setdefault(
        "lora_train",
        {
            "epochs": 30,
            "batch_size": 32,
            "eval_batch_size": 64,
            "num_workers": 4,
            "pin_memory": True,
            "lora_lr": 1.0e-4,
            "head_lr": 1.0e-3,
            "weight_decay": 0.05,
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "seed": 1,
            "amp": True,
        },
    )


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    feature_cfg = cfg.setdefault("feature", {})
    lora_cfg = cfg.setdefault("lora", {})
    train_cfg = cfg.setdefault("lora_train", {})
    if args.device is not None:
        feature_cfg["device"] = args.device
    if args.local_repo is not None:
        feature_cfg["local_repo"] = args.local_repo
    if args.input_size is not None:
        feature_cfg["input_size"] = args.input_size
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        train_cfg["eval_batch_size"] = args.eval_batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.lora_lr is not None:
        train_cfg["lora_lr"] = args.lora_lr
    if args.head_lr is not None:
        train_cfg["head_lr"] = args.head_lr
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.scheduler is not None:
        train_cfg["scheduler"] = args.scheduler
    if args.warmup_ratio is not None:
        train_cfg["warmup_ratio"] = args.warmup_ratio
    if args.rank is not None:
        lora_cfg["rank"] = args.rank
    if args.alpha is not None:
        lora_cfg["alpha"] = args.alpha
    if args.dropout is not None:
        lora_cfg["dropout"] = args.dropout
    if args.target_modules is not None:
        lora_cfg["target_modules"] = args.target_modules


def load_data(input_dir: Path, centroid_split: str) -> dict[str, Any]:
    required = [input_dir / "paths.txt", input_dir / "labels.npy", input_dir / "eval_paths.txt", input_dir / "eval_labels.npy"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file is missing: {path}")
    train_paths = (input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    eval_paths = (input_dir / "eval_paths.txt").read_text(encoding="utf-8").splitlines()
    train_labels = np.load(input_dir / "labels.npy", allow_pickle=True).astype(str)
    eval_labels = np.load(input_dir / "eval_labels.npy", allow_pickle=True).astype(str)
    if len(train_paths) != len(train_labels):
        raise ValueError("paths.txt and labels.npy lengths do not match.")
    if len(eval_paths) != len(eval_labels):
        raise ValueError("eval_paths.txt and eval_labels.npy lengths do not match.")

    centroid_path = input_dir / centroid_split if centroid_split else None
    centroid_mask = read_clean_mask(centroid_path, len(train_labels)) if centroid_path and centroid_path.exists() else None
    return {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "eval_paths": eval_paths,
        "eval_labels": eval_labels,
        "candidate_mask": np.ones(len(train_labels), dtype=bool),
        "centroid_mask": centroid_mask,
    }


def load_proto_scores(input_dir: Path, proto_scores: str, score_col: str, n: int) -> np.ndarray:
    path = Path(proto_scores)
    if not path.is_absolute():
        path = input_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Prototype score file is missing: {path}")
    scores = np.full(n, np.nan, dtype=np.float32)
    rows = read_csv(path)
    for row in rows:
        if "index" not in row:
            raise ValueError(f"{path} is missing required column: index")
        if score_col not in row:
            raise ValueError(f"{path} is missing requested score column: {score_col}")
        idx = int(row["index"])
        if idx < 0 or idx >= n:
            raise ValueError(f"Index {idx} in {path} is outside [0, {n}).")
        scores[idx] = float(row[score_col])
    if np.any(np.isnan(scores)):
        missing = int(np.sum(np.isnan(scores)))
        raise ValueError(f"{path} did not provide {score_col} for {missing} samples.")
    return scores


def read_clean_mask(path: Path, n: int) -> np.ndarray:
    rows = read_csv(path)
    if len(rows) != n:
        raise ValueError(f"{path} length {len(rows)} does not match expected length {n}.")
    mask = np.zeros(n, dtype=bool)
    for row in rows:
        idx = int(row["index"])
        if idx < 0 or idx >= n:
            raise ValueError(f"Index {idx} in {path} is outside [0, {n}).")
        mask[idx] = row["state"] == "clean"
    return mask


def write_selection_files(output_dir: Path, ratio_text: str, proto_text: str, seed: int, rows: list[dict[str, Any]]) -> None:
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_epoch.setdefault(int(row["epoch"]), []).append(row)
    for epoch, epoch_rows in sorted(by_epoch.items()):
        path = output_dir / f"pgdf_selection_r{ratio_text}_p{proto_text}_seed{seed}_epoch_{epoch:03d}.csv"
        write_csv(path, epoch_rows, selection_fields())


def build_method_summary(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in result_rows:
        if row.get("proto_keep_ratio", "") == "":
            raise ValueError(
                "Missing proto_keep_ratio in PGDF result summary. In auto mode this usually means "
                "no selection update occurred; ensure --epochs is greater than --warmup-epochs."
            )
        by_key.setdefault((str(row["method"]), float(row["retention_ratio"]), str(row.get("auto_proto_keep", "no"))), []).append(row)
    out = []
    for (method, ratio, auto_mode), rows in by_key.items():
        best = np.array([float(row["best_top1"]) for row in rows], dtype=np.float64)
        final = np.array([float(row["final_top1"]) for row in rows], dtype=np.float64)
        final_selected = np.array([int(row["final_selected_samples"]) for row in rows], dtype=np.int64)
        proto_ratios = np.array([float(row["proto_keep_ratio"]) for row in rows], dtype=np.float64)
        auto_jaccards = np.array([float(row["auto_proto_jaccard"]) for row in rows if row.get("auto_proto_jaccard", "") != ""], dtype=np.float64)
        out.append(
            {
                "method": method,
                "retention_ratio": float(ratio),
                "proto_keep_ratio": float(proto_ratios.mean()),
                "min_proto_keep_ratio": float(proto_ratios.min()),
                "max_proto_keep_ratio": float(proto_ratios.max()),
                "auto_proto_keep": auto_mode,
                "mean_auto_proto_jaccard": float(auto_jaccards.mean()) if len(auto_jaccards) else "",
                "num_seeds": int(len(rows)),
                "mean_best_top1": float(best.mean()),
                "std_best_top1": float(best.std(ddof=1)) if len(best) > 1 else 0.0,
                "max_best_top1": float(best.max()),
                "mean_final_top1": float(final.mean()),
                "mean_final_selected_samples": float(final_selected.mean()),
                "min_final_selected_samples": int(final_selected.min()),
                "max_final_selected_samples": int(final_selected.max()),
            }
        )
    return out


def write_summary(
    path: Path,
    input_dir: Path,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    update_rows: list[dict[str, Any]],
) -> None:
    resolved_epochs = int(cfg["lora_train"]["epochs"])
    lines = [
        "# Prototype-guided Dynamic Filtering Summary",
        "",
        "This run intersects dynamic class-wise small-loss selection with class-wise high prototype score filtering.",
        "",
        f"- Source V1 output: {input_dir}",
        f"- Retention ratios: {args.retention_ratios}",
        f"- Prototype keep ratios: {'auto' if args.auto_proto_keep else args.proto_keep_ratios}",
        f"- Auto prototype keep: {args.auto_proto_keep}",
        f"- Auto rule: J>={args.auto_high_jaccard} -> p={args.auto_p_high}; "
        f"J>={args.auto_mid_jaccard} -> p={args.auto_p_mid}; "
        f"J>={args.auto_low_jaccard} -> p={args.auto_p_low}; else p={args.auto_p_very_low}",
        f"- Prototype scores: {args.proto_scores}",
        f"- Prototype score column: {args.proto_score_col}",
        f"- Seeds: {args.seeds}",
        f"- Warm-up epochs: {args.warmup_epochs}",
        f"- Update interval: {args.update_interval}",
        f"- Total epochs: {resolved_epochs}",
        "",
        "## Method Summary",
        "| method | r | p_mean | p_min | p_max | auto | J_auto | seeds | mean_best_top1 | std_best_top1 | mean_final_top1 | mean_selected | min_selected | max_selected |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        auto_j = row["mean_auto_proto_jaccard"]
        auto_j_text = f"{float(auto_j):.6f}" if auto_j != "" else ""
        lines.append(
            f"| {row['method']} | {float(row['retention_ratio']):.3f} | {float(row['proto_keep_ratio']):.3f} | "
            f"{float(row['min_proto_keep_ratio']):.3f} | {float(row['max_proto_keep_ratio']):.3f} | "
            f"{row['auto_proto_keep']} | {auto_j_text} | {int(row['num_seeds'])} | {float(row['mean_best_top1']):.6f} | "
            f"{float(row['std_best_top1']):.6f} | {float(row['mean_final_top1']):.6f} | "
            f"{float(row['mean_final_selected_samples']):.1f} | {int(row['min_final_selected_samples'])} | "
            f"{int(row['max_final_selected_samples'])} |"
        )

    if update_rows:
        lines.extend(["", "## Selection Updates", "| method | seed | r | p | J_auto | epoch | loss_selected | proto_pass | selected | proto_reject | prev_jaccard | centroid_jaccard |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in update_rows:
            centroid = row["overlap_with_centroid"]
            centroid_text = f"{float(centroid):.6f}" if centroid != "" else ""
            auto_j = row.get("auto_proto_jaccard", "")
            auto_j_text = f"{float(auto_j):.6f}" if auto_j != "" else ""
            lines.append(
                f"| {row['method']} | {int(row['seed'])} | {float(row['retention_ratio']):.3f} | "
                f"{float(row['proto_keep_ratio']):.3f} | {auto_j_text} | {int(row['epoch'])} | "
                f"{int(row['num_loss_selected'])} | {int(row['num_proto_pass'])} | {int(row['num_selected'])} | "
                f"{int(row['proto_reject_count'])} | {float(row['overlap_with_previous_selection']):.6f} | {centroid_text} |"
            )
    lines.extend(
        [
            "",
            "## Output Files",
            "- `pgdf_results.csv`: per-ratio, per-seed final metrics.",
            "- `pgdf_train_log.csv`: epoch-level training log.",
            "- `pgdf_update_summary.csv`: selection stability, prototype rejection, and overlap metrics per update.",
            "- `pgdf_per_class_summary.csv`: class-wise selected counts per update.",
            "- `pgdf_selection_r*_p*_seed*_epoch_*.csv`: per-sample loss, confidence, prototype score, and selected state at each update.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def parse_int_list(raw: str, name: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    return values


def parse_path_maps(items: list[str]) -> list[tuple[str, str]]:
    maps = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must use OLD=NEW format, got: {item}")
        old, new = item.split("=", 1)
        maps.append((old, new))
    return maps


def ratio_to_text(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def train_log_fields() -> list[str]:
    return [
        "method",
        "seed",
        "epoch",
        "lr_lora",
        "lr_head",
        "loss",
        "top1",
        "top5",
        "train_samples",
        "candidate_samples",
        "selected_ratio",
        "eval_samples",
        "trainable_params",
        "total_params",
    ]


def result_fields() -> list[str]:
    return [
        "method",
        "seed",
        "retention_ratio",
        "proto_keep_ratio",
        "auto_proto_keep",
        "auto_proto_jaccard",
        "warmup_epochs",
        "update_interval",
        "candidate_samples",
        "final_selected_samples",
        "selection_updates",
        "train_samples",
        "eval_samples",
        "best_epoch",
        "best_top1",
        "best_top5",
        "final_top1",
        "final_top5",
        "last5_mean",
        "last5_std",
        "trainable_params",
        "total_params",
    ]


def update_fields() -> list[str]:
    return [
        "method",
        "seed",
        "retention_ratio",
        "proto_keep_ratio",
        "auto_proto_jaccard",
        "epoch",
        "num_candidates",
        "num_loss_selected",
        "num_proto_pass",
        "num_selected",
        "proto_reject_count",
        "selected_ratio",
        "mean_loss_selected",
        "mean_loss_unselected",
        "mean_loss_proto_rejected",
        "mean_proto_selected",
        "mean_proto_unselected",
        "overlap_with_previous_selection",
        "overlap_with_centroid",
    ]


def per_class_fields() -> list[str]:
    return [
        "method",
        "seed",
        "retention_ratio",
        "proto_keep_ratio",
        "epoch",
        "web_label",
        "total_count",
        "loss_selected_count",
        "proto_pass_count",
        "selected_count",
        "proto_reject_count",
        "selected_ratio",
        "mean_loss_selected",
        "mean_loss_unselected",
        "mean_loss_proto_rejected",
        "mean_proto_selected",
        "mean_proto_unselected",
    ]


def selection_fields() -> list[str]:
    return [
        "method",
        "seed",
        "retention_ratio",
        "proto_keep_ratio",
        "epoch",
        "index",
        "path",
        "web_label",
        "loss",
        "confidence",
        "proto_score",
        "loss_selected",
        "proto_pass",
        "state",
    ]


if __name__ == "__main__":
    main()
