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
from gcdd.lora_training import train_dinov2_lora
from gcdd.progress import log_stage


METHODS = {
    "all": {
        "method": "DINOv2 LoRA all noisy samples",
        "split": "",
        "purpose": "full noisy web-train LoRA baseline; tests whether filtering is useful under backbone adaptation",
    },
    "gcdd_proto": {
        "method": "DINOv2 LoRA + GCDD+Proto-only added",
        "split": "proto_gcdd/splits/split_gcdd_proto.csv",
        "purpose": "both only plus GCDD+Proto-only samples; tests whether graph-retained hard clean helps after backbone adaptation",
    },
    "both_only": {
        "method": "DINOv2 LoRA + both only",
        "split": "",
        "purpose": "intersection of GCDD+Proto clean and centroid clean; easy-clean upper bound and conservative set",
    },
    "centroid": {
        "method": "DINOv2 LoRA + Centroid filtering",
        "split": "centroid_filtering_split.csv",
        "purpose": "resource-allowing comparison baseline for hard-clean contribution",
    },
    "full_gcdd": {
        "method": "DINOv2 LoRA + Full GCDD-clean",
        "split": "full_gcdd_clean_split.csv",
        "purpose": "optional original GCDD reference",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOv2 LoRA training on Web-Bird clean splits.")
    parser.add_argument("--input-dir", default="outputs/Web-Bird/v1_web_bird", help="V1 output directory.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <input-dir>/lora.")
    parser.add_argument("--methods", default="all,full_gcdd,both_only,gcdd_proto,centroid", help="Comma-separated methods: all, full_gcdd, both_only, gcdd_proto, centroid.")
    parser.add_argument("--seeds", default="1", help="Comma-separated seeds.")
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Map stored image root to local root, e.g. /root/autodl-tmp/web-bird=E:\\data\\web-bird.")
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
    parser.add_argument("--warmup-ratio", type=float, help="Override warmup ratio.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Override device.")
    parser.add_argument("--local-repo", help="Override DINOv2 local torch hub repo path.")
    parser.add_argument("--input-size", type=int, help="Override image input size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "lora"
    ensure_dir(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    if not args.no_save_checkpoints:
        ensure_dir(checkpoint_dir)

    methods = parse_methods(args.methods)
    seeds = parse_seeds(args.seeds)
    path_maps = parse_path_maps(args.path_map)

    cfg = load_config(input_dir)
    apply_lora_defaults(cfg)
    apply_overrides(cfg, args)
    write_yaml(output_dir / "resolved_config.yaml", cfg)

    log_stage("[1/4] Loading paths, labels, and split masks.")
    data = load_data(input_dir, methods)

    log_stage("[2/4] Running DINOv2 LoRA training.")
    train_logs: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    for key in methods:
        method_info = METHODS[key]
        train_mask = data["split_masks"][key]
        for seed in seeds:
            checkpoint_path = None
            if not args.no_save_checkpoints:
                checkpoint_path = checkpoint_dir / f"{key}_seed{seed}_best.pt"
            run_cfg = copy.deepcopy(cfg)
            run_cfg["lora_train"]["seed"] = int(seed)
            result = train_dinov2_lora(
                data["train_paths"],
                data["train_labels"],
                data["eval_paths"],
                data["eval_labels"],
                train_mask,
                run_cfg,
                method=method_info["method"],
                seed=seed,
                path_maps=path_maps,
                checkpoint_path=checkpoint_path,
            )
            train_logs.extend(result.logs)
            result_rows.append(result.summary)
            for name in result.trainable_modules:
                module_rows.append(
                    {
                        "method": method_info["method"],
                        "seed": seed,
                        "module": name,
                        "trainable_params": result.trainable_params,
                        "total_params": result.total_params,
                    }
                )

    log_stage("[3/4] Writing LoRA result tables.")
    write_csv(
        output_dir / "train_log.csv",
        train_logs,
        [
            "method",
            "seed",
            "epoch",
            "lr_lora",
            "lr_head",
            "loss",
            "top1",
            "top5",
            "train_samples",
            "eval_samples",
            "trainable_params",
            "total_params",
        ],
    )
    write_csv(
        output_dir / "lora_results.csv",
        result_rows,
        [
            "method",
            "seed",
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
        ],
    )
    write_csv(output_dir / "lora_modules.csv", module_rows, ["method", "seed", "module", "trainable_params", "total_params"])
    summary_rows = build_method_summary(result_rows)
    write_csv(output_dir / "lora_summary.csv", summary_rows, ["method", "num_seeds", "mean_best_top1", "std_best_top1", "max_best_top1", "mean_final_top1", "train_samples"])

    log_stage("[4/4] Writing LoRA summary.")
    write_summary(output_dir / "run_summary.md", input_dir, methods, seeds, summary_rows, result_rows)
    write_json(
        output_dir / "lora_summary.json",
        {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "methods": methods,
            "seeds": seeds,
            "results": result_rows,
            "summary": summary_rows,
        },
    )
    log_stage(f"LoRA results written to {output_dir}")


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in methods if item not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(METHODS)}")
    if not methods:
        raise ValueError("At least one method is required.")
    return methods


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_path_maps(items: list[str]) -> list[tuple[str, str]]:
    maps = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--path-map must use OLD=NEW format, got: {item}")
        old, new = item.split("=", 1)
        maps.append((old, new))
    return maps


def load_config(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "resolved_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"V1 resolved config is missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_lora_defaults(cfg: dict[str, Any]) -> None:
    cfg.setdefault("feature", {})
    cfg["feature"].setdefault("backend", "dinov2_vitb14")
    cfg["feature"].setdefault("device", "auto")
    cfg["feature"].setdefault("input_size", 224)
    cfg["feature"].setdefault("local_repo", "")
    cfg.setdefault(
        "lora",
        {
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.05,
            "target_modules": "qkv",
        },
    )
    cfg.setdefault(
        "lora_train",
        {
            "epochs": 10,
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


def load_data(input_dir: Path, methods: list[str]) -> dict[str, Any]:
    required = [input_dir / "paths.txt", input_dir / "labels.npy", input_dir / "eval_paths.txt", input_dir / "eval_labels.npy"]
    for key in methods:
        split = METHODS[key]["split"]
        if key == "both_only":
            required.append(input_dir / METHODS["gcdd_proto"]["split"])
            required.append(input_dir / METHODS["centroid"]["split"])
        elif split:
            required.append(input_dir / split)
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

    split_masks = {}
    for key in methods:
        split = METHODS[key]["split"]
        if key == "both_only":
            proto_mask = read_clean_mask(input_dir / METHODS["gcdd_proto"]["split"], len(train_labels))
            centroid_mask = read_clean_mask(input_dir / METHODS["centroid"]["split"], len(train_labels))
            split_masks[key] = proto_mask & centroid_mask
        elif split:
            split_masks[key] = read_clean_mask(input_dir / split, len(train_labels))
        else:
            split_masks[key] = np.ones(len(train_labels), dtype=bool)
    return {
        "train_paths": train_paths,
        "train_labels": train_labels,
        "eval_paths": eval_paths,
        "eval_labels": eval_labels,
        "split_masks": split_masks,
    }


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


def build_method_summary(result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in result_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
    for method, rows in by_method.items():
        best = np.array([float(row["best_top1"]) for row in rows], dtype=np.float64)
        final = np.array([float(row["final_top1"]) for row in rows], dtype=np.float64)
        out.append(
            {
                "method": method,
                "num_seeds": len(rows),
                "mean_best_top1": float(best.mean()),
                "std_best_top1": float(best.std(ddof=1)) if len(best) > 1 else 0.0,
                "max_best_top1": float(best.max()),
                "mean_final_top1": float(final.mean()),
                "train_samples": int(rows[0]["train_samples"]),
            }
        )
    return out


def write_summary(path: Path, input_dir: Path, methods: list[str], seeds: list[int], summary_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> None:
    summary_map = {row["method"]: row for row in summary_rows}
    lines = [
        "# LoRA Web-Bird Summary",
        "",
        "This route trains DINOv2 with LoRA on image data instead of using frozen-feature linear CE.",
        "The main purpose is to test whether GCDD+Proto-only hard-clean samples become useful when the backbone can adapt.",
        "",
        f"- Source V1 output: {input_dir}",
        f"- Methods: {', '.join(METHODS[key]['method'] for key in methods)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        "",
        "## Method Summary",
        "| method | seeds | mean_best_top1 | std_best_top1 | max_best_top1 | mean_final_top1 | train_samples |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['num_seeds']} | {float(row['mean_best_top1']):.6f} | "
            f"{float(row['std_best_top1']):.6f} | {float(row['max_best_top1']):.6f} | "
            f"{float(row['mean_final_top1']):.6f} | {int(row['train_samples'])} |"
        )
    if "DINOv2 LoRA + GCDD+Proto-only added" in summary_map and "DINOv2 LoRA + Centroid filtering" in summary_map:
        diff = float(summary_map["DINOv2 LoRA + GCDD+Proto-only added"]["mean_best_top1"]) - float(summary_map["DINOv2 LoRA + Centroid filtering"]["mean_best_top1"])
        lines.extend(
            [
                "",
                "## Hard-Clean Contribution Check",
                f"- mean_best_top1 difference: DINOv2 LoRA + GCDD+Proto-only added - DINOv2 LoRA + Centroid filtering = {diff:.6f}",
                "- If this difference reaches +0.003 to +0.005, it is strong evidence that GCDD+Proto hard-clean samples help under LoRA.",
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "- `lora_results.csv`: per-method, per-seed metrics.",
            "- `lora_summary.csv`: method-level mean/std.",
            "- `train_log.csv`: epoch-level training log.",
            "- `lora_modules.csv`: LoRA-injected module names.",
            "- `checkpoints/`: best LoRA/head checkpoints unless `--no-save-checkpoints` is used.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
