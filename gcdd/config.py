from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "name": "Web-Bird",
        "root": "",
        "index_file": "",
        "split": "train",
        "samples_per_class": 10,
        "max_classes": None,
        "verify_images": True,
    },
    "output": {
        "root": "outputs",
        "version": "v0_smoke",
    },
    "feature": {
        "backend": "dinov2_vitb14",
        "device": "auto",
        "batch_size": 8,
        "input_size": 224,
        "top_patch_ratio": 0.2,
        "random_dim": 128,
    },
    "graph": {
        "knn_backend": "auto",
        "k_pool_class": 20,
        "k_pool_global": 50,
        "k_class": 5,
        "k_global": 10,
        "rrf_k0": 20,
    },
    "selection": {
        "otsu_bins": 256,
        "clean_ratio_clip": [0.3, 0.9],
        "epsilon": 1.0e-8,
    },
    "train": {
        "epochs": 1,
        "batch_size": 32,
        "lr": 0.05,
        "min_lr": 0.0,
        "scheduler": "none",
        "seed": 42,
    },
}


def load_config(config_path: Path, cli_overrides: dict[str, Any] | None = None, set_overrides: list[str] | None = None) -> dict[str, Any]:
    """Resolve config using CLI > YAML > defaults."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        deep_update(cfg, yaml_cfg)

    for key, value in (cli_overrides or {}).items():
        set_by_path(cfg, key, value)
    for item in set_overrides or []:
        key, value = parse_set_override(item)
        set_by_path(cfg, key, value)
    return cfg


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_by_path(cfg: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = cfg
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def parse_set_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"--set must use KEY=VALUE format, got: {item}")
    key, raw_value = item.split("=", 1)
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value
