from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.data import VALID_EXTENSIONS
from gcdd.io_utils import ensure_dir, write_csv, write_yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "name": "FolderDataset",
        "root": "",
        "train_split": "train",
        "test_split": "test",
    },
    "noise": {
        "ratio": 0.4,
        "seed": 42,
        "target_strategy": "adjacent_cyclic",
    },
    "output": {
        "dir": "outputs/folder_noise_indices",
        "prefix": "",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a folder-format CSV index with class-asymmetric training label noise.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--data-root", help="Dataset root containing train/ and test/.")
    parser.add_argument("--noise-ratio", type=float, help="Per-class train noise ratio. Overrides noise.ratio.")
    parser.add_argument("--seed", type=int, help="Noise sampling seed. Overrides noise.seed.")
    parser.add_argument("--output-dir", help="Output directory. Overrides output.dir.")
    parser.add_argument("--prefix", help="Output file prefix. Overrides output.prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config), args)
    root = Path(cfg["dataset"]["root"]).expanduser().resolve()
    dataset_name = str(cfg["dataset"]["name"])
    train_split = str(cfg["dataset"].get("train_split", "train"))
    test_split = str(cfg["dataset"].get("test_split", "test"))
    ratio = float(cfg["noise"]["ratio"])
    seed = int(cfg["noise"]["seed"])
    strategy = str(cfg["noise"]["target_strategy"])
    output_dir = Path(cfg["output"]["dir"]).expanduser()
    prefix = str(cfg["output"].get("prefix") or default_prefix(dataset_name, ratio, seed))

    validate_config(root, train_split, test_split, ratio, strategy)
    ensure_dir(output_dir)

    records = discover_folder_records(root, train_split, test_split)
    class_names = sorted({record["clean_label"] for record in records})
    target_map = build_target_map(class_names, strategy)
    noisy_keys = sample_noisy_train_keys(records, ratio, seed, train_split)
    rows = build_index_rows(records, target_map, noisy_keys, ratio, seed, strategy)

    index_path = output_dir / f"{prefix}_index.csv"
    mapping_path = output_dir / f"{prefix}_mapping.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    resolved_config_path = output_dir / f"{prefix}_resolved_config.yaml"

    write_csv(index_path, rows, index_fieldnames())
    write_csv(mapping_path, build_mapping_rows(target_map), ["class_name", "target_class_name"])
    write_csv(summary_path, build_summary_rows(rows, train_split), summary_fieldnames())
    write_yaml(resolved_config_path, cfg)

    train_count = sum(1 for row in rows if row["split"] == "train")
    noisy_count = sum(1 for row in rows if row["is_noisy"] == "1")
    print(f"{dataset_name} asymmetric-noise index written to {index_path}", flush=True)
    print(f"Train noisy samples: {noisy_count}/{train_count} ({noisy_count / max(train_count, 1):.4f})", flush=True)


def load_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = deep_copy(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        deep_update(cfg, yaml_cfg)
    if args.data_root:
        cfg["dataset"]["root"] = args.data_root
    if args.noise_ratio is not None:
        cfg["noise"]["ratio"] = args.noise_ratio
    if args.seed is not None:
        cfg["noise"]["seed"] = args.seed
    if args.output_dir:
        cfg["output"]["dir"] = args.output_dir
    if args.prefix:
        cfg["output"]["prefix"] = args.prefix
    return cfg


def deep_copy(value: dict[str, Any]) -> dict[str, Any]:
    return {key: deep_copy(item) if isinstance(item, dict) else item for key, item in value.items()}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value


def validate_config(root: Path, train_split: str, test_split: str, ratio: float, strategy: str) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    if not (root / train_split).exists():
        raise FileNotFoundError(f"Train split directory not found: {root / train_split}")
    if not (root / test_split).exists():
        raise FileNotFoundError(f"Test split directory not found: {root / test_split}")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("noise.ratio must satisfy 0 <= ratio <= 1.")
    if strategy != "adjacent_cyclic":
        raise ValueError("Only noise.target_strategy=adjacent_cyclic is supported now.")


def discover_folder_records(root: Path, train_split: str, test_split: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for split_dir_name, normalized_split in [(train_split, "train"), (test_split, "test")]:
        split_root = root / split_dir_name
        class_dirs = [path for path in sorted(split_root.iterdir()) if path.is_dir()]
        for class_dir in class_dirs:
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                    continue
                records.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "split": normalized_split,
                        "clean_label": class_dir.name,
                    }
                )
    if not records:
        raise ValueError(f"No images found under {root / train_split} or {root / test_split}.")
    return records


def build_target_map(class_names: list[str], strategy: str) -> dict[str, str]:
    if strategy != "adjacent_cyclic":
        raise ValueError(f"Unsupported target strategy: {strategy}")
    return {label: class_names[(i + 1) % len(class_names)] for i, label in enumerate(class_names)}


def sample_noisy_train_keys(records: list[dict[str, str]], ratio: float, seed: int, train_split: str) -> set[str]:
    grouped: dict[str, list[str]] = {}
    for i, record in enumerate(records):
        if record["split"] != "train":
            continue
        grouped.setdefault(record["clean_label"], []).append(str(i))

    rng = random.Random(seed)
    noisy_keys: set[str] = set()
    for label in sorted(grouped):
        keys = grouped[label][:]
        rng.shuffle(keys)
        take = int(round(len(keys) * ratio))
        noisy_keys.update(keys[:take])
    return noisy_keys


def build_index_rows(
    records: list[dict[str, str]],
    target_map: dict[str, str],
    noisy_keys: set[str],
    ratio: float,
    seed: int,
    strategy: str,
) -> list[dict[str, str]]:
    rows = []
    for i, record in enumerate(records):
        is_noisy = record["split"] == "train" and str(i) in noisy_keys
        clean_label = record["clean_label"]
        web_label = target_map[clean_label] if is_noisy else clean_label
        rows.append(
            {
                "index": str(i),
                "path": record["path"],
                "split": record["split"],
                "clean_label": clean_label,
                "web_label": web_label,
                "is_noisy": "1" if is_noisy else "0",
                "noise_target_label": target_map[clean_label],
                "noise_type": "asymmetric",
                "noise_strategy": strategy,
                "noise_ratio": f"{ratio:.6f}",
                "noise_seed": str(seed),
            }
        )
    return rows


def build_mapping_rows(target_map: dict[str, str]) -> list[dict[str, str]]:
    return [{"class_name": label, "target_class_name": target_map[label]} for label in sorted(target_map)]


def build_summary_rows(rows: list[dict[str, str]], train_split: str) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        label = row["clean_label"]
        stats = grouped.setdefault(label, {"train_count": 0, "noisy_count": 0})
        stats["train_count"] += 1
        stats["noisy_count"] += int(row["is_noisy"])

    out = []
    for label in sorted(grouped):
        train_count = grouped[label]["train_count"]
        noisy_count = grouped[label]["noisy_count"]
        out.append(
            {
                "clean_label": label,
                "train_count": str(train_count),
                "noisy_count": str(noisy_count),
                "actual_noise_ratio": f"{noisy_count / max(train_count, 1):.6f}",
            }
        )
    total_train = sum(int(row["train_count"]) for row in out)
    total_noisy = sum(int(row["noisy_count"]) for row in out)
    out.append(
        {
            "clean_label": "ALL",
            "train_count": str(total_train),
            "noisy_count": str(total_noisy),
            "actual_noise_ratio": f"{total_noisy / max(total_train, 1):.6f}",
        }
    )
    return out


def default_prefix(dataset_name: str, ratio: float, seed: int) -> str:
    dataset = dataset_name.lower().replace("-", "_").replace(" ", "_")
    ratio_text = f"{ratio:g}".replace(".", "p")
    return f"{dataset}_asym_r{ratio_text}_s{seed}"


def index_fieldnames() -> list[str]:
    return [
        "index",
        "path",
        "split",
        "clean_label",
        "web_label",
        "is_noisy",
        "noise_target_label",
        "noise_type",
        "noise_strategy",
        "noise_ratio",
        "noise_seed",
    ]


def summary_fieldnames() -> list[str]:
    return ["clean_label", "train_count", "noisy_count", "actual_noise_ratio"]


if __name__ == "__main__":
    main()
