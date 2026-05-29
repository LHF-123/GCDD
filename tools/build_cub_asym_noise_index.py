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

from gcdd.io_utils import ensure_dir, write_csv, write_yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "cub_root": "dataset/CUB_200_2011/CUB_200_2011",
    },
    "noise": {
        "ratio": 0.4,
        "seed": 42,
        "target_strategy": "adjacent_cyclic",
    },
    "output": {
        "dir": "outputs/CUB_200_2011/noise_indices",
        "prefix": "",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CUB-200-2011 CSV index with class-asymmetric training label noise.")
    parser.add_argument("--config", default="configs/cub_asym_noise.yaml", help="YAML config path.")
    parser.add_argument("--cub-root", help="CUB_200_2011 root containing images.txt and images/.")
    parser.add_argument("--noise-ratio", type=float, help="Per-class train noise ratio. Overrides noise.ratio.")
    parser.add_argument("--seed", type=int, help="Noise sampling seed. Overrides noise.seed.")
    parser.add_argument("--output-dir", help="Output directory. Overrides output.dir.")
    parser.add_argument("--prefix", help="Output file prefix. Overrides output.prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config), args)
    cub_root = Path(cfg["dataset"]["cub_root"]).expanduser().resolve()
    ratio = float(cfg["noise"]["ratio"])
    seed = int(cfg["noise"]["seed"])
    strategy = str(cfg["noise"]["target_strategy"])
    output_dir = Path(cfg["output"]["dir"]).expanduser()
    prefix = str(cfg["output"].get("prefix") or default_prefix(ratio, seed))

    validate_config(cub_root, ratio, strategy)
    ensure_dir(output_dir)

    metadata = load_cub_metadata(cub_root)
    target_map = build_target_map(metadata["classes"], strategy)
    noisy_ids = sample_noisy_train_ids(metadata["rows"], ratio, seed)
    rows = build_index_rows(cub_root, metadata["rows"], target_map, noisy_ids, ratio, seed, strategy)

    index_path = output_dir / f"{prefix}_index.csv"
    mapping_path = output_dir / f"{prefix}_mapping.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    resolved_config_path = output_dir / f"{prefix}_resolved_config.yaml"

    write_csv(index_path, rows, index_fieldnames())
    write_csv(mapping_path, build_mapping_rows(metadata["classes"], target_map), ["class_id", "class_name", "target_label", "target_class_name"])
    write_csv(summary_path, build_summary_rows(rows), summary_fieldnames())
    write_yaml(resolved_config_path, cfg)

    train_count = sum(1 for row in rows if row["split"] == "train")
    noisy_count = sum(1 for row in rows if row["is_noisy"] == "1")
    print(f"CUB asymmetric-noise index written to {index_path}", flush=True)
    print(f"Train noisy samples: {noisy_count}/{train_count} ({noisy_count / max(train_count, 1):.4f})", flush=True)


def load_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = deep_copy(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        deep_update(cfg, yaml_cfg)
    if args.cub_root:
        cfg["dataset"]["cub_root"] = args.cub_root
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


def validate_config(cub_root: Path, ratio: float, strategy: str) -> None:
    if not cub_root.exists():
        raise FileNotFoundError(f"CUB root not found: {cub_root}")
    for name in ["images.txt", "image_class_labels.txt", "train_test_split.txt", "classes.txt"]:
        if not (cub_root / name).exists():
            raise FileNotFoundError(f"Required CUB file not found: {cub_root / name}")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("noise.ratio must satisfy 0 <= ratio <= 1.")
    if strategy != "adjacent_cyclic":
        raise ValueError("Only noise.target_strategy=adjacent_cyclic is supported now.")


def load_cub_metadata(cub_root: Path) -> dict[str, Any]:
    images = read_two_column_file(cub_root / "images.txt")
    labels = read_two_column_file(cub_root / "image_class_labels.txt")
    splits = read_two_column_file(cub_root / "train_test_split.txt")
    classes = dict(read_two_column_file(cub_root / "classes.txt"))

    ids = sorted(images, key=int)
    if ids != sorted(labels, key=int) or ids != sorted(splits, key=int):
        raise ValueError("CUB metadata files contain different image ids.")

    rows = []
    for image_id in ids:
        rows.append(
            {
                "image_id": image_id,
                "relative_path": images[image_id],
                "clean_label": labels[image_id],
                "split": "train" if splits[image_id] == "1" else "test",
            }
        )
    return {"rows": rows, "classes": classes}


def read_two_column_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            data[key] = value
    return data


def build_target_map(classes: dict[str, str], strategy: str) -> dict[str, str]:
    class_ids = sorted(classes, key=lambda value: int(value))
    if strategy != "adjacent_cyclic":
        raise ValueError(f"Unsupported target strategy: {strategy}")
    return {class_id: class_ids[(i + 1) % len(class_ids)] for i, class_id in enumerate(class_ids)}


def sample_noisy_train_ids(rows: list[dict[str, str]], ratio: float, seed: int) -> set[str]:
    grouped: dict[str, list[str]] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        grouped.setdefault(row["clean_label"], []).append(row["image_id"])

    rng = random.Random(seed)
    noisy_ids: set[str] = set()
    for label in sorted(grouped, key=int):
        image_ids = grouped[label][:]
        rng.shuffle(image_ids)
        # Per-class rounding keeps the requested ratio stable without moving samples across classes.
        take = int(round(len(image_ids) * ratio))
        noisy_ids.update(image_ids[:take])
    return noisy_ids


def build_index_rows(
    cub_root: Path,
    metadata_rows: list[dict[str, str]],
    target_map: dict[str, str],
    noisy_ids: set[str],
    ratio: float,
    seed: int,
    strategy: str,
) -> list[dict[str, str]]:
    rows = []
    for i, item in enumerate(metadata_rows):
        image_id = item["image_id"]
        clean_label = item["clean_label"]
        is_noisy = item["split"] == "train" and image_id in noisy_ids
        noisy_label = target_map[clean_label] if is_noisy else clean_label
        path = Path("images") / Path(item["relative_path"])
        rows.append(
            {
                "index": str(i),
                "image_id": image_id,
                "path": path.as_posix(),
                "abs_path": str(cub_root / path),
                "split": item["split"],
                "clean_label": clean_label,
                "web_label": noisy_label,
                "is_noisy": "1" if is_noisy else "0",
                "noise_target_label": target_map[clean_label],
                "noise_type": "asymmetric",
                "noise_strategy": strategy,
                "noise_ratio": f"{ratio:.6f}",
                "noise_seed": str(seed),
            }
        )
    return rows


def build_mapping_rows(classes: dict[str, str], target_map: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for class_id in sorted(classes, key=int):
        target = target_map[class_id]
        rows.append(
            {
                "class_id": class_id,
                "class_name": classes[class_id],
                "target_label": target,
                "target_class_name": classes[target],
            }
        )
    return rows


def build_summary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        label = row["clean_label"]
        stats = grouped.setdefault(label, {"train_count": 0, "noisy_count": 0})
        stats["train_count"] += 1
        stats["noisy_count"] += int(row["is_noisy"])

    summary_rows = []
    for label in sorted(grouped, key=int):
        train_count = grouped[label]["train_count"]
        noisy_count = grouped[label]["noisy_count"]
        summary_rows.append(
            {
                "clean_label": label,
                "train_count": str(train_count),
                "noisy_count": str(noisy_count),
                "actual_noise_ratio": f"{noisy_count / max(train_count, 1):.6f}",
            }
        )
    total_train = sum(int(row["train_count"]) for row in summary_rows)
    total_noisy = sum(int(row["noisy_count"]) for row in summary_rows)
    summary_rows.append(
        {
            "clean_label": "ALL",
            "train_count": str(total_train),
            "noisy_count": str(total_noisy),
            "actual_noise_ratio": f"{total_noisy / max(total_train, 1):.6f}",
        }
    )
    return summary_rows


def default_prefix(ratio: float, seed: int) -> str:
    ratio_text = f"{ratio:g}".replace(".", "p")
    return f"cub_asym_r{ratio_text}_s{seed}"


def index_fieldnames() -> list[str]:
    return [
        "index",
        "image_id",
        "path",
        "abs_path",
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
