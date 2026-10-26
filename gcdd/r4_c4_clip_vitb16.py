"""R4-C4 CLIP ViT-B/16 cyclic-asym40 orchestration helpers.

This module owns only the new cross-backbone experiment.  It copies no
historical result and never derives a new label realization: the three
existing cyclic-asymmetric 40% manifests and clean-validation manifests are
verified then referenced from an isolated output root.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from gcdd.checkpoint_validation import load_clean_labels_from_noise_index
from gcdd.clip_vit import (
    CLIP_VITB16_BACKEND,
    clip_backbone_protocol_record,
    load_clip_vitb16_visual,
    sha256_file,
    validate_clip_vitb16_config,
)
from gcdd.lora_dynamic import train_dynamic_loss_lora
from gcdd.lora_training import train_dinov2_lora
from revision_experiments.profiling import ProfilingMode, RunProfiler
from revision_experiments.selection_logging import SelectionLogError, read_selection_bundle
from revision_experiments.training_integration import build_experiment_id


METHODS: tuple[tuple[str, str, bool], ...] = (
    ("all_noisy_ce", "all_noisy", False),
    ("dynamic_small_loss", "dynamic_r08", True),
    ("jal_ce", "jal_ce", False),
    ("pgdf", "pgdf_dynamic_proto", True),
)
SELECTION_METHODS = frozenset({"dynamic_small_loss", "pgdf"})
FROZEN = {
    "noise_rate": 0.4,
    "noise_seed": 42,
    "validation_ratio": 0.1,
    "validation_seed": 20250726,
    "training_seeds": (1, 42, 88),
    "epochs": 30,
    "dynamic_ratio": 0.8,
    "fixed_p": 0.4,
    "warmup_epochs": 5,
    "update_interval": 5,
    "effective_batch_size": 80,
    "microbatch_size": 16,
}
FORMAL_DATASETS = ("cub", "cars", "aircraft")
FORMAL_RUNS = len(FORMAL_DATASETS) * len(METHODS) * len(FROZEN["training_seeds"])
NOMINAL_24GB_CLASS_MINIMUM_MIB = 22000


@dataclass(frozen=True)
class Task:
    dataset_key: str
    dataset_name: str
    method_key: str
    runner_method: str
    selection_logging: bool
    seed: int
    output_root: Path
    input_dir: Path
    manifest_path: Path
    validation_dir: Path
    instrumentation_root: Path
    path_maps: tuple[tuple[str, str], ...] = ()
    validation_path_maps: tuple[tuple[str, str], ...] = ()

    @property
    def trainer_root(self) -> Path:
        return self.output_root / "runs" / self.dataset_key / self.method_key / f"seed{self.seed}" / "trainer"

    @property
    def run_dir(self) -> Path:
        return self.trainer_root / self.runner_method / f"seed{self.seed}"

    @property
    def task_dir(self) -> Path:
        return self.output_root / "runs" / self.dataset_key / self.method_key / f"seed{self.seed}"

    @property
    def experiment_id(self) -> str:
        """Match the common runner's immutable profiling/selection directory key.

        The common checkpoint-validation entry point derives this key before
        it sees the R4-C4 wrapper.  The wrapper must therefore consume that
        exact key rather than inventing a CLIP-specific directory name.
        """
        return build_experiment_id(
            dataset=self.dataset_name,
            noise_protocol="adjacent_cyclic",
            noise_seed=42,
            method=self.runner_method,
            training_seed=self.seed,
        )


def safe_component(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def current_code_fingerprints() -> dict[str, str]:
    """Fingerprint every implementation file that can change R4-C4 behavior."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "gcdd/clip_vit.py", "gcdd/r4_c4_clip_vitb16.py", "gcdd/lora_training.py", "gcdd/lora_dynamic.py",
        "scripts/run_r4_c4_clip_vitb16.py", "scripts/run_lora_checkpoint_validation.py", "revision_experiments/profiling.py",
    )
    return {item: sha256_file(root / item) for item in paths}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def semantic_sha256(path: Path) -> str:
    if path.suffix.lower() == ".json":
        return canonical_sha256(json.loads(path.read_text(encoding="utf-8-sig")))
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            return canonical_sha256({"fieldnames": reader.fieldnames or [], "rows": list(reader)})
    return sha256_file(path)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"Existing R4-C4 CLIP asset differs from source: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        import shutil

        shutil.copy2(source, destination)


def expand_environment(value: str) -> str:
    return os.path.expandvars(value)


def resolve_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    copied = copy.deepcopy(config)
    base = config_path.parent.resolve()
    for section, keys in {
        "output": ("root", "instrumentation_root"),
        "historical": ("repository_root",),
    }.items():
        for key in keys:
            if copied.get(section, {}).get(key):
                copied[section][key] = str((base / copied[section][key]).resolve())
    for dataset in copied.get("datasets", {}).values():
        for key in ("base_input_dir", "source_noise_index", "source_resolved_config", "source_validation_dir"):
            if dataset.get(key):
                dataset[key] = str((base / dataset[key]).resolve())
        dataset["path_maps"] = [
            [expand_environment(str(item[0])), expand_environment(str(item[1]))]
            for item in dataset.get("path_maps", [])
        ]
    backbone = copied.setdefault("backbone", {})
    if backbone.get("weights_path"):
        backbone["weights_path"] = expand_environment(str(backbone["weights_path"]))
    return copied


def load_plan(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("R4-C4 CLIP configuration must be a mapping.")
    return payload


def validate_frozen_config(config: dict[str, Any]) -> None:
    formal = config.get("formal", {})
    for key, expected in FROZEN.items():
        actual = tuple(formal.get(key, ())) if key == "training_seeds" else formal.get(key)
        if actual != expected:
            raise ValueError(f"Frozen R4-C4 CLIP protocol mismatch for {key}: {actual!r} != {expected!r}")
    if tuple(formal.get("methods", ())) != tuple(item[0] for item in METHODS):
        raise ValueError("R4-C4 CLIP requires exactly All-noisy, Dynamic, JAL-CE, and PGDF.")
    if tuple(config.get("datasets", {}).keys()) != FORMAL_DATASETS:
        raise ValueError("R4-C4 CLIP requires exactly CUB, Cars, and Aircraft.")
    backbone = config.get("backbone", {})
    if backbone.get("architecture") != "ViT-B-16" or backbone.get("pretrained") != "openai":
        raise ValueError("R4-C4 CLIP cannot substitute a different CLIP architecture or pretrained source.")
    if int(backbone.get("input_size", 0)) != 224:
        raise ValueError("R4-C4 CLIP input resolution is frozen at 224.")
    if backbone.get("cls_feature_space") != "final_ln_pre_projection":
        raise ValueError("R4-C4 CLIP prototype feature space is frozen to final-LN pre-projection CLS.")
    if not str(backbone.get("weights_sha256", "")).strip():
        raise ValueError("R4-C4 CLIP requires an explicit checkpoint SHA-256.")
    if int(config.get("runtime", {}).get("nominal_24gb_minimum_mib", 0)) != NOMINAL_24GB_CLASS_MINIMUM_MIB:
        raise ValueError(
            "R4-C4 CLIP requires the documented 22000 MiB nominal-24GB-class admission threshold."
        )


def mapped_path(path_text: str, path_maps: tuple[tuple[str, str], ...]) -> Path:
    normalized = path_text.replace("\\", "/")
    for old, new in path_maps:
        source = old.replace("\\", "/").rstrip("/")
        if normalized == source or normalized.startswith(source + "/"):
            suffix = normalized[len(source):].lstrip("/")
            return Path(new) / Path(*suffix.split("/"))
    return Path(path_text)


def prepared_cfg(config: dict[str, Any], dataset_key: str) -> dict[str, Any]:
    source = yaml.safe_load(Path(config["datasets"][dataset_key]["source_resolved_config"]).read_text(encoding="utf-8")) or {}
    generated = copy.deepcopy(source)
    backbone = config["backbone"]
    generated["feature"] = {
        **generated.get("feature", {}),
        "backend": CLIP_VITB16_BACKEND,
        "device": "cuda",
        "input_size": 224,
        "clip_architecture": "ViT-B-16",
        "clip_pretrained": "openai",
        "clip_weights_path": str(backbone["weights_path"]),
        "clip_weights_sha256": str(backbone["weights_sha256"]),
        "cls_feature_space": "final_ln_pre_projection",
    }
    generated["lora"] = {
        **generated.get("lora", {}),
        "rank": 8,
        "alpha": 16.0,
        "dropout": 0.05,
        "target_modules": "qkv",
    }
    generated["lora_train"] = {
        **generated.get("lora_train", {}),
        "epochs": 30,
        "batch_size": 80,
        "microbatch_size": 16,
        "eval_batch_size": int(config["runtime"]["eval_microbatch_size"]),
        "amp": True,
    }
    return generated


def _expected_observed_labels(base_paths: list[str], source_rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    by_path: dict[str, tuple[str, str]] = {}
    for row in source_rows:
        if row.get("split", "").lower() != "train":
            continue
        for candidate in (row.get("abs_path", ""), row.get("path", "")):
            if candidate:
                by_path[candidate.replace("\\", "/")] = (str(row["web_label"]), str(row["clean_label"]))
    observed: list[str] = []
    clean: list[str] = []
    for original in base_paths:
        normalized = original.replace("\\", "/")
        candidates = [normalized]
        for marker in ("/images/", "/train/"):
            if marker in normalized:
                candidates.append(marker[1:] + normalized.split(marker, 1)[1])
        match = next((by_path[item] for item in candidates if item in by_path), None)
        if match is None:
            raise ValueError(f"Cyclic-40 manifest does not identify training path: {original}")
        observed.append(match[0]); clean.append(match[1])
    return np.asarray(observed, dtype=str), np.asarray(clean, dtype=str)


def prepare_dataset(config: dict[str, Any], key: str) -> dict[str, Any]:
    data = config["datasets"][key]
    root = Path(config["output"]["root"])
    base = Path(data["base_input_dir"])
    source_manifest = Path(data["source_noise_index"])
    source_validation = Path(data["source_validation_dir"])
    source_cfg_path = Path(data["source_resolved_config"])
    required = (
        base / "paths.txt", base / "labels.npy", base / "eval_paths.txt", base / "eval_labels.npy",
        source_manifest, source_validation / "validation_manifest.csv", source_validation / "validation_manifest.json", source_cfg_path,
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError("R4-C4 CLIP source protocol asset missing: " + "; ".join(missing))
    source_cfg = yaml.safe_load(source_cfg_path.read_text(encoding="utf-8")) or {}
    noise = source_cfg.get("checkpoint_validation", {}).get("noise_realization", {})
    expected_noise = {"noise_strategy": "adjacent_cyclic", "noise_ratio": 0.4, "noise_seed": 42, "mapping_type": "cyclic"}
    for field, value in expected_noise.items():
        actual = noise.get(field)
        if field == "noise_ratio":
            if float(actual) != float(value):
                raise ValueError(f"{key} source resolved config has mismatched {field}: {actual!r}")
        elif actual != value:
            raise ValueError(f"{key} source resolved config has mismatched {field}: {actual!r}")
    if source_cfg.get("checkpoint_validation", {}).get("validation_seed") != FROZEN["validation_seed"]:
        raise ValueError(f"{key} source validation seed is not the formal cyclic-40 seed.")
    base_paths = (base / "paths.txt").read_text(encoding="utf-8").splitlines()
    source_rows = read_csv(source_manifest)
    observed, clean = _expected_observed_labels(base_paths, source_rows)
    base_observed = np.load(base / "labels.npy", allow_pickle=True).astype(str)
    if not np.array_equal(observed, base_observed):
        raise ValueError(f"{key} labels.npy differs from the historical cyclic-40 observed-label manifest.")
    manifest = root / "manifests" / key / source_manifest.name
    hardlink_or_copy(source_manifest, manifest)
    prepared = root / "prepared_inputs" / key
    for name in ("paths.txt", "labels.npy", "eval_paths.txt", "eval_labels.npy"):
        hardlink_or_copy(base / name, prepared / name)
    validation = root / "manifests" / key / "fixed_validation"
    for name in ("validation_manifest.csv", "validation_manifest.json"):
        source = source_validation / name
        destination = validation / name
        hardlink_or_copy(source, destination)
        if semantic_sha256(source) != semantic_sha256(destination):
            raise RuntimeError(f"Validation manifest semantic copy verification failed: {destination}")
    validation_rows = read_csv(validation / "validation_manifest.csv")
    pool = np.zeros(len(base_paths), dtype=bool)
    for row in validation_rows:
        index = int(row["index"])
        if row["partition"] == "training_pool":
            pool[index] = True
        elif row["partition"] != "validation":
            raise ValueError(f"Unknown validation partition in {key}: {row['partition']!r}")
        if clean[index] != str(row["clean_label"]):
            raise ValueError(f"{key} clean-label mismatch against reused validation manifest at index {index}.")
    if not pool.any() or int(pool.sum()) + sum(row["partition"] == "validation" for row in validation_rows) != len(base_paths):
        raise ValueError(f"{key} reused validation manifest is incomplete.")
    generated = prepared_cfg(config, key)
    generated["dataset"] = {**generated.get("dataset", {}), "name": f"{data['display_name']}-asym40-s42-clip-vitb16", "index_file": str(manifest)}
    generated["output"] = {**generated.get("output", {}), "root": str(root), "version": f"r4_c4_clip_vitb16_{key}_cyclic40"}
    generated["checkpoint_validation"] = {
        **generated.get("checkpoint_validation", {}),
        "protocol": "fixed_clean_validation_v1",
        "validation_ratio": 0.1,
        "validation_seed": 20250726,
        "dynamic_ratio": 0.8,
        "fixed_p": 0.4,
        "warmup_epochs": 5,
        "update_interval": 5,
        "posthoc_oracle_test": False,
        "official_test_selected_only": True,
        "noise_realization": expected_noise,
        "noise_index_sha256": sha256_file(manifest),
    }
    validate_clip_vitb16_config(generated)
    yaml_text = yaml.safe_dump(generated, allow_unicode=True, sort_keys=False)
    config_path = prepared / "resolved_config.yaml"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != yaml_text:
        raise FileExistsError(f"Refusing to overwrite a different frozen CLIP config: {config_path}")
    if not config_path.exists():
        config_path.write_text(yaml_text, encoding="utf-8")
    return {
        "key": key, "dataset_name": generated["dataset"]["name"], "manifest_path": manifest,
        "prepared_input": prepared, "validation_dir": validation, "source_manifest": source_manifest,
        "source_validation": source_validation, "source_config": source_cfg_path, "base_paths": base_paths,
        "clean_labels": clean, "observed_labels": observed, "train_mask": pool, "config": generated,
    }


def tasks_for(config: dict[str, Any], prepared: dict[str, dict[str, Any]]) -> list[Task]:
    root = Path(config["output"]["root"])
    instrumentation = Path(config["output"]["instrumentation_root"])
    tasks: list[Task] = []
    for key in FORMAL_DATASETS:
        info = prepared[key]
        maps = tuple((str(item[0]), str(item[1])) for item in config["datasets"][key].get("path_maps", []))
        for method_key, runner_method, selection_logging in METHODS:
            for seed in FROZEN["training_seeds"]:
                tasks.append(Task(key, info["dataset_name"], method_key, runner_method, selection_logging, seed, root,
                                  info["prepared_input"], info["manifest_path"], info["validation_dir"], instrumentation, maps, maps))
    return tasks


def task_command(task: Task) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, str(root / "scripts" / "run_lora_checkpoint_validation.py"),
        "--input-dir", str(task.input_dir), "--noise-index", str(task.manifest_path),
        "--output-dir", str(task.trainer_root), "--methods", task.runner_method, "--seeds", str(task.seed),
        "--validation-ratio", "0.1", "--validation-seed", "20250726", "--dynamic-ratio", "0.8", "--fixed-p", "0.4",
        "--warmup-epochs", "5", "--update-interval", "5", "--official-test-selected-only", "--no-posthoc-oracle-test",
        "--revision-profiling-mode", "detailed", "--revision-output-root", str(task.instrumentation_root),
    ]
    for old, new in task.path_maps:
        command.extend(("--path-map", f"{old}={new}"))
    if task.selection_logging:
        command.append("--revision-selection-logging")
    return command


def _runtime_gpu_state(minimum_total_mib: int = NOMINAL_24GB_CLASS_MINIMUM_MIB) -> dict[str, Any]:
    """Report capacity using a tolerant nominal-24GB-class admission gate."""
    if minimum_total_mib < NOMINAL_24GB_CLASS_MINIMUM_MIB:
        raise ValueError("The nominal-24GB-class admission threshold cannot be below 22000 MiB.")
    try:
        import torch
    except ImportError as exc:
        return {"status": "CUDA_UNAVAILABLE", "reason": str(exc)}
    if not torch.cuda.is_available():
        return {"status": "CUDA_UNAVAILABLE", "reason": "torch.cuda.is_available() is false"}
    properties = torch.cuda.get_device_properties(0)
    free, total_runtime = torch.cuda.mem_get_info(0)
    total = int(properties.total_memory)
    safe_limit = min(22 * 1024**3, total - 2 * 1024**3)
    nominal_24gb = total >= int(minimum_total_mib) * 1024**2
    free_ok = free >= safe_limit
    return {
        "status": "READY_FOR_GPU_PREFLIGHT" if nominal_24gb and free_ok else "READY_WAIT_GPU",
        "name": properties.name,
        "total_bytes": total,
        "total_gib": total / 1024**3,
        "free_bytes": int(free),
        "free_gib": free / 1024**3,
        "safe_reserved_limit_bytes": int(safe_limit),
        "safe_reserved_limit_gib": safe_limit / 1024**3,
        "nominal_24gb_class_minimum_mib": int(minimum_total_mib),
        "nominal_24gb_capacity": nominal_24gb,
        "free_memory_satisfies_limit": free_ok,
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for key in FORMAL_DATASETS:
        data = config["datasets"][key]
        for field in ("base_input_dir", "source_noise_index", "source_resolved_config", "source_validation_dir"):
            path = Path(data[field])
            findings.append({"dataset": key, "check": field, "status": "PASS" if path.exists() else "FAIL", "path": str(path)})
        source_cfg = yaml.safe_load(Path(data["source_resolved_config"]).read_text(encoding="utf-8")) or {} if Path(data["source_resolved_config"]).is_file() else {}
        root_path = str(source_cfg.get("dataset", {}).get("root", ""))
        mappings = tuple((str(item[0]), str(item[1])) for item in data.get("path_maps", []))
        mapped_root = mapped_path(root_path, mappings)
        sample = ""
        paths_file = Path(data["base_input_dir"]) / "paths.txt"
        if paths_file.is_file():
            items = paths_file.read_text(encoding="utf-8").splitlines()
            sample = items[0] if items else ""
        mapped_sample = mapped_path(sample, mappings)
        findings.extend([
            {"dataset": key, "check": "mapped_dataset_root", "status": "PASS" if mapped_root.is_dir() else "REMOTE_CONFIGURATION_REQUIRED", "path": str(mapped_root)},
            {"dataset": key, "check": "mapped_first_training_image", "status": "PASS" if mapped_sample.is_file() else "REMOTE_CONFIGURATION_REQUIRED", "path": str(mapped_sample)},
        ])
    backbone_status = "PASS"
    backbone_detail = ""
    try:
        clip_cfg = {"feature": {
            "backend": CLIP_VITB16_BACKEND, "clip_architecture": "ViT-B-16", "clip_pretrained": "openai", "input_size": 224,
            "clip_weights_path": config["backbone"]["weights_path"], "clip_weights_sha256": config["backbone"]["weights_sha256"],
            "cls_feature_space": "final_ln_pre_projection",
        }}
        protocol = clip_backbone_protocol_record(clip_cfg)
        # CPU model construction is deliberate: a matching file hash alone is
        # insufficient evidence that this OpenCLIP version can load the stated
        # checkpoint as the audited visual Transformer.
        visual = load_clip_vitb16_visual(None, clip_cfg)
        attention_modules = [name for name, module in visual.named_modules() if module.__class__.__name__ == "MultiheadAttention"]
        if len(attention_modules) != 12:
            raise RuntimeError(f"Expected 12 CLIP visual attention modules, found {len(attention_modules)}.")
        protocol["cpu_load_verified"] = True
        protocol["fused_qkv_attention_modules"] = attention_modules
        del visual
        backbone_detail = json.dumps(protocol, ensure_ascii=False)
    except Exception as exc:  # record a precise pre-launch block, never substitute a model
        backbone_status = "FAIL"
        backbone_detail = f"{type(exc).__name__}: {exc}"
    findings.append({"dataset": "backbone", "check": "local_openai_clip_vit_b16_weights_and_identity", "status": backbone_status, "path": backbone_detail})
    core_paths = ("gcdd/clip_vit.py", "gcdd/lora_training.py", "gcdd/lora_dynamic.py", "scripts/run_lora_checkpoint_validation.py")
    repo = Path(__file__).resolve().parents[1]
    findings.extend({"dataset": "code", "check": item, "status": "PASS" if (repo / item).is_file() else "FAIL", "path": str(repo / item)} for item in core_paths)
    required_ok = all(item["status"] == "PASS" for item in findings if item["status"] != "REMOTE_CONFIGURATION_REQUIRED")
    data_ok = all(item["status"] == "PASS" for item in findings if "mapped_" in item["check"])
    gpu = _runtime_gpu_state(int(config["runtime"]["nominal_24gb_minimum_mib"]))
    status = "BLOCKED_PROTOCOL" if not required_ok else "BLOCKED_DATASET_LOCATION" if not data_ok else "READY_WAIT_GPU" if gpu["status"] != "READY_FOR_GPU_PREFLIGHT" else "READY_FOR_GPU_PREFLIGHT"
    return {"status": status, "planned_runs": FORMAL_RUNS, "checks": findings, "gpu": gpu}


def write_prepared_assets(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[Task]]:
    prepared = {key: prepare_dataset(config, key) for key in FORMAL_DATASETS}
    root = Path(config["output"]["root"])
    protocol = clip_backbone_protocol_record(prepared["cub"]["config"])
    protocol.update({"formal_protocol": FROZEN, "formal_run_count": FORMAL_RUNS})
    write_json(root / "CLIP_BACKBONE_PROTOCOL_AUDIT.json", protocol)
    write_text(root / "CLIP_BACKBONE_PROTOCOL_AUDIT.md", "\n".join([
        "# R4-C4 CLIP ViT-B/16 backbone protocol audit", "",
        "- Architecture: OpenAI CLIP `ViT-B-16`; only the visual tower is retained after loading.",
        f"- Weights SHA-256: `{protocol['weights_sha256']}`.",
        "- Prototype feature: final layer-normalized visual CLS token before the 768→512 CLIP projection; no feature normalization is applied before PGDF's existing L2 normalization.",
        "- LoRA mapping: all 12 fused PyTorch MultiheadAttention Q/K/V projections; base visual parameters remain frozen and the classifier head remains trainable.",
        "- Text encoder, prompts, zero-shot logits, and class-name text features are not used.",
        "- CLIP 224 preprocessing is deliberately distinct from historical DINOv2 448 preprocessing; only within-CLIP method deltas support fair method comparisons.",
        "",
    ]))
    sources = {
        key: {
            "source_noise_manifest": str(info["source_manifest"]), "source_noise_manifest_sha256": sha256_file(info["source_manifest"]),
            "reused_noise_manifest": str(info["manifest_path"]), "reused_noise_manifest_sha256": sha256_file(info["manifest_path"]),
            "source_validation_manifest": str(info["source_validation"] / "validation_manifest.json"),
            "source_validation_manifest_sha256": semantic_sha256(info["source_validation"] / "validation_manifest.json"),
            "reused_validation_manifest": str(info["validation_dir"] / "validation_manifest.json"),
            "reused_validation_manifest_sha256": semantic_sha256(info["validation_dir"] / "validation_manifest.json"),
            "source_resolved_config": str(info["source_config"]), "source_resolved_config_sha256": sha256_file(info["source_config"]),
            "training_pool_samples": int(info["train_mask"].sum()),
        }
        for key, info in prepared.items()
    }
    write_json(root / "manifests" / "source_protocol_provenance.json", {"schema_version": "r4_c4_clip_cyclic40_v1", "sources": sources})
    tasks = tasks_for(config, prepared)
    code_hashes = current_code_fingerprints()
    planned: list[dict[str, Any]] = []
    for task in tasks:
        item = {
            "schema_version": "r4_c4_clip_vitb16_planned_run_v1", "dataset": task.dataset_key, "dataset_name": task.dataset_name,
            "method": task.method_key, "runner_method": task.runner_method, "training_seed": task.seed, "frozen_protocol": FROZEN,
            "backbone_protocol": protocol, "noise_manifest": str(task.manifest_path), "noise_manifest_sha256": sha256_file(task.manifest_path),
            "validation_manifest": str(task.validation_dir / "validation_manifest.json"), "validation_manifest_sha256": semantic_sha256(task.validation_dir / "validation_manifest.json"),
            "command": task_command(task), "selection_logging": task.selection_logging, "profiling_mode": "detailed", "source_code_sha256": code_hashes,
        }
        write_json(root / "orchestration" / "resolved_configs" / f"{task.dataset_key}_{task.method_key}_seed{task.seed}.json", item)
        planned.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed, "status": task_status(task)[0], "command": " ".join(task_command(task))})
    write_csv(root / "orchestration" / "formal_36_run_matrix.csv", planned, list(planned[0]))
    return prepared, tasks


def stage_shared_validation(task: Task) -> None:
    for name in ("validation_manifest.csv", "validation_manifest.json"):
        hardlink_or_copy(task.validation_dir / name, task.trainer_root / name)


def task_status(task: Task) -> tuple[str, list[str]]:
    issues: list[str] = []
    # The orchestration parent creates this marker before spawning the
    # checkpoint-validation child.  Treat it as authoritative while present:
    # epoch/checkpoint/profile artifacts are necessarily incomplete mid-run.
    # This also prevents a second --resume process from overwriting the same
    # formal task directory.
    if (task.task_dir / "RUNNING.json").is_file():
        return "RUNNING", []
    result = task.run_dir / "result.json"
    if not result.is_file():
        if (task.task_dir / "FAILED.json").is_file():
            return "FAILED", ["prior_attempt_failed_preserved"]
        if task.trainer_root.exists() and any(task.trainer_root.iterdir()):
            return "PARTIAL", ["partial_output_not_eligible_for_automatic_resume"]
        return "PENDING", []
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
        required = {"seed": task.seed, "method_key": task.runner_method, "noise_strategy": "adjacent_cyclic", "noise_seed": 42, "validation_seed": 20250726, "checkpoint_protocol": "fixed_clean_validation_v1", "official_test_evaluation": "validation_selected_only"}
        for field, expected in required.items():
            if payload.get(field) != expected:
                issues.append(f"result_{field}_mismatch")
        if float(payload.get("noise_ratio", -1.0)) != 0.4:
            issues.append("result_noise_ratio_mismatch")
        if payload.get("validation_selected_test_top1", "") == "" or int(payload.get("best_epoch", 0)) < 1:
            issues.append("missing_validation_selected_official_test")
    except (OSError, ValueError, json.JSONDecodeError):
        issues.append("invalid_result_json")
    for path in (task.run_dir / "checkpoints" / "best_val.pt", task.run_dir / "checkpoints" / "last.pt", task.run_dir / "train_log.csv"):
        if not path.is_file():
            issues.append(f"missing:{path.name}")
    audit_path = task.run_dir / "model_adaptation_audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("backbone") != "CLIP ViT-B/16 visual encoder only" or len(audit.get("lora_module_names", [])) != 12:
            issues.append("clip_lora_adaptation_audit_mismatch")
        if int(audit.get("trainable_parameter_count", 0)) <= 0 or int(audit.get("frozen_parameter_count", 0)) <= 0:
            issues.append("clip_lora_parameter_counts_invalid")
    except (OSError, ValueError, json.JSONDecodeError):
        issues.append("missing_or_invalid_clip_lora_adaptation_audit")
    profile_path = task.instrumentation_root / task.experiment_id / "profiling.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if profile.get("recording_status") != "complete" or profile.get("profiling_mode") != "detailed":
            issues.append("profiling_incomplete")
    except (OSError, ValueError, json.JSONDecodeError):
        issues.append("missing_or_invalid_profiling")
    logs = read_csv(task.run_dir / "train_log.csv") if (task.run_dir / "train_log.csv").is_file() else []
    if sorted({int(row.get("epoch", 0)) for row in logs}) != list(range(1, 31)):
        issues.append("train_log_not_exactly_30_epochs")
    for name in ("validation_manifest.csv", "validation_manifest.json"):
        copy_path = task.trainer_root / name
        if not copy_path.is_file() or semantic_sha256(task.validation_dir / name) != semantic_sha256(copy_path):
            issues.append(f"validation_manifest_content_mismatch:{name}")
    if task.selection_logging:
        records = task.instrumentation_root / task.experiment_id / "selection_records"
        try:
            bundle = read_selection_bundle(records)
            updates = {(sample.training_seed, sample.epoch) for sample in bundle.samples}
            if updates != {(task.seed, epoch) for epoch in (5, 10, 15, 20, 25)}:
                issues.append("selection_instrumentation_not_expected_five_stages")
        except (OSError, ValueError, SelectionLogError):
            issues.append("selection_instrumentation_invalid")
        updates_path = task.run_dir / "selection_updates.csv"
        if not updates_path.is_file() or sorted({int(row["epoch"]) for row in read_csv(updates_path)}) != [5, 10, 15, 20, 25]:
            issues.append("selection_updates_not_expected_five_stages")
    return ("COMPLETE_VERIFIED" if not issues else "PARTIAL"), issues


def _runtime_environment() -> dict[str, Any]:
    result: dict[str, Any] = {"python_executable": sys.executable, "gpu": _runtime_gpu_state()}
    try:
        import torch
        import open_clip
        result.update({"torch_version": torch.__version__, "cuda_version": torch.version.cuda, "open_clip_version": getattr(open_clip, "__version__", "unknown")})
    except ImportError:
        pass
    return result


def run_tasks(tasks: list[Task], *, resume: bool) -> int:
    for task in tasks:
        state, issues = task_status(task)
        if state == "RUNNING":
            print(f"ALREADY_RUNNING {task.dataset_key}/{task.method_key}/seed{task.seed}", flush=True)
            return 5
        if state == "COMPLETE_VERIFIED":
            print(f"SKIP COMPLETE {task.dataset_key}/{task.method_key}/seed{task.seed}", flush=True)
            continue
        if state in {"FAILED", "PARTIAL"}:
            print(f"NOT_RESTARTED {task.dataset_key}/{task.method_key}/seed{task.seed}: {';'.join(issues)}", flush=True)
            if not resume:
                return 3
            continue
        stage_shared_validation(task)
        task.task_dir.mkdir(parents=True, exist_ok=True)
        write_json(task.task_dir / "RUNNING.json", {"command": task_command(task), "pid": os.getpid(), "started_unix": time.time()})
        write_json(task.task_dir / "runtime_environment.json", _runtime_environment())
        log = task.output_root / "logs" / task.dataset_key / task.method_key / f"seed{task.seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(task_command(task), stdout=handle, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1], check=False)
        (task.task_dir / "RUNNING.json").unlink(missing_ok=True)
        if completed.returncode != 0:
            write_json(task.task_dir / "FAILED.json", {"returncode": completed.returncode, "log": str(log), "command": task_command(task)})
            return completed.returncode
        state, issues = task_status(task)
        if state != "COMPLETE_VERIFIED":
            write_json(task.task_dir / "FAILED.json", {"returncode": 0, "completion_issues": issues, "log": str(log)})
            return 4
    return 0


def _preflight_subset(info: dict[str, Any], count: int = 96) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    pool_indices = np.flatnonzero(info["train_mask"])
    chosen = pool_indices[: min(count, len(pool_indices))]
    labels = info["observed_labels"][chosen]
    labels_seen = set(labels.tolist())
    validation_rows = read_csv(info["validation_dir"] / "validation_manifest.csv")
    validation_indices = [int(row["index"]) for row in validation_rows if row["partition"] == "validation" and str(row["clean_label"]) in labels_seen]
    if len(chosen) < 2 or not validation_indices:
        raise RuntimeError("GPU preflight cannot construct a non-empty local train/validation subset.")
    return ([info["base_paths"][int(item)] for item in chosen], labels, [info["base_paths"][item] for item in validation_indices], info["clean_labels"][validation_indices])


def gpu_preflight(config: dict[str, Any], prepared: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run actual CUDA smoke paths without creating any formal run artifact."""
    gpu = _runtime_gpu_state(int(config["runtime"]["nominal_24gb_minimum_mib"]))
    root = Path(config["output"]["root"])
    report_path = root / "GPU_24GB_MEMORY_PREFLIGHT.json"
    if gpu["status"] != "READY_FOR_GPU_PREFLIGHT":
        payload = {"status": "READY_WAIT_GPU", "config_sha256": canonical_sha256(config), "source_code_sha256": current_code_fingerprints(), "reason": "A nominal 24GB GPU with at least the 2GiB safety reserve is required before CUDA smoke execution.", "gpu": gpu}
        write_json(report_path, payload)
        write_text(root / "GPU_24GB_MEMORY_PREFLIGHT.md", "# R4-C4 GPU preflight\n\n`READY_WAIT_GPU`: no eligible 24GB GPU was available; no CUDA model or formal run was started.\n")
        return payload
    import torch
    from gcdd.lora_training import DINOv2LoRAClassifier, freeze_all, inject_lora, parse_target_modules

    info = prepared["cub"]
    train_paths, train_labels, validation_paths, validation_labels = _preflight_subset(info)
    base_cfg = copy.deepcopy(info["config"])
    base_cfg["lora_train"]["epochs"] = 2
    base_cfg["lora_train"]["num_workers"] = 0
    base_cfg["lora_train"]["pin_memory"] = False
    profiles: list[dict[str, Any]] = []
    gradient_check: dict[str, Any] = {}
    failure: str | None = None
    try:
        model = DINOv2LoRAClassifier.make(torch, base_cfg, 3).cuda()
        freeze_all(model.backbone)
        modules = inject_lora(torch, model.backbone, parse_target_modules("qkv"), 8, 16.0, 0.05)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
        output = model(torch.randn(2, 3, 224, 224, device="cuda"))
        output.mean().backward()
        gradient_check = {
            "lora_module_names": modules,
            "lora_module_count": len(modules),
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "frozen_parameter_count": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
            "nonzero_lora_gradient_parameter_tensors": sum(int(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))) for name, parameter in model.backbone.named_parameters() if "lora_" in name),
            "nonzero_head_gradient_parameter_tensors": sum(int(parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))) for parameter in model.head.parameters()),
        }
        del model, output
        torch.cuda.empty_cache()
        for method_key, label in (("all_noisy_ce", "all-noisy CE"), ("jal_ce", "JAL-CE"), ("dynamic_small_loss", "dynamic small-loss"), ("pgdf", "PGDF DynamicProto")):
            cfg = copy.deepcopy(base_cfg)
            profiler = RunProfiler(f"r4-c4-clip-preflight-{method_key}", ProfilingMode.DETAILED, device_type="cuda", torch_module=torch)
            checkpoint_root = root / "gpu_preflight" / method_key / "checkpoints"
            if method_key in {"all_noisy_ce", "jal_ce"}:
                cfg["loss_type"] = "jal_ce" if method_key == "jal_ce" else "ce"
                result = train_dinov2_lora(train_paths, train_labels, validation_paths, validation_labels, np.ones(len(train_labels), dtype=bool), cfg, label, 1,
                                           path_maps=[], checkpoint_path=checkpoint_root / "best_val.pt", final_checkpoint_path=checkpoint_root / "last.pt", profiler=profiler)
            else:
                result = train_dynamic_loss_lora(train_paths, train_labels, validation_paths, validation_labels, np.ones(len(train_labels), dtype=bool), cfg, label, 1,
                                                 retention_ratio=0.8, warmup_epochs=0, update_interval=1, path_maps=[], proto_keep_ratio=0.4 if method_key == "pgdf" else None,
                                                 selection_strategy="loss_and_proto" if method_key == "pgdf" else "loss_only", prototype_mode="dynamic_lora" if method_key == "pgdf" else "fixed",
                                                 checkpoint_path=checkpoint_root / "best_val.pt", final_checkpoint_path=checkpoint_root / "last.pt", profiler=profiler)
            profile_path = profiler.write_json(root / "gpu_preflight" / "profiling")
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            checkpoint_readable = False
            if (checkpoint_root / "best_val.pt").is_file() and (checkpoint_root / "last.pt").is_file():
                loaded = torch.load(checkpoint_root / "best_val.pt", map_location="cpu")
                checkpoint_readable = isinstance(loaded, dict) and isinstance(loaded.get("state_dict"), dict)
            profiles.append({"method": method_key, "checkpoint_readable": checkpoint_readable, "profile": profile, "summary": result.summary})
            torch.cuda.empty_cache()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    max_reserved = max((item["profile"]["memory"]["overall_run_peak"]["reserved_bytes"] or 0 for item in profiles), default=0)
    selection_peak = max((item["profile"]["memory"]["phases"].get("selection", {}).get("reserved_bytes") or 0 for item in profiles), default=0)
    passed = failure is None and len(profiles) == 4 and max_reserved <= gpu["safe_reserved_limit_bytes"] and all(item["checkpoint_readable"] for item in profiles) and gradient_check.get("nonzero_lora_gradient_parameter_tensors", 0) > 0 and gradient_check.get("nonzero_head_gradient_parameter_tensors", 0) > 0
    payload = {"status": "PASS" if passed else "BLOCKED_GPU_MEMORY" if failure and "out of memory" in failure.lower() else "FAIL", "config_sha256": canonical_sha256(config), "source_code_sha256": current_code_fingerprints(), "gpu": gpu, "formal_batching": {"effective_batch": 80, "microbatch": 16, "gradient_accumulation_steps": 5, "amp_dtype": "float16 autocast"}, "gradient_check": gradient_check, "methods": profiles, "overall_peak_reserved_bytes": max_reserved, "pgdf_selection_peak_reserved_bytes": selection_peak, "safety_limit_bytes": gpu["safe_reserved_limit_bytes"], "failure": failure}
    write_json(report_path, payload)
    write_text(root / "GPU_24GB_MEMORY_PREFLIGHT.md", "# R4-C4 GPU preflight\n\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def _selection_quality(task: Task, info: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if task.method_key not in SELECTION_METHODS:
        return [], []
    bundle = read_selection_bundle(task.instrumentation_root / task.experiment_id / "selection_records")
    records_by_epoch: dict[int, list[Any]] = defaultdict(list)
    for record in bundle.samples:
        records_by_epoch[record.epoch].append(record)
    quality: list[dict[str, Any]] = []; auc_rows: list[dict[str, Any]] = []
    pool = info["train_mask"]; clean = info["clean_labels"]; observed = info["observed_labels"]
    pool_clean = int(np.sum(pool & (clean == observed))); pool_noisy = int(pool.sum()) - pool_clean
    for epoch, records in sorted(records_by_epoch.items()):
        selected = [record for record in records if record.final_selected]
        index_of = lambda record: int(str(record.stable_sample_id).removeprefix("train_index:"))
        selected_clean = sum(clean[index_of(record)] == observed[index_of(record)] for record in selected)
        selected_noisy = len(selected) - selected_clean
        quality.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed, "epoch": epoch, "selected": len(selected), "selected_clean": selected_clean, "selected_noisy": selected_noisy, "pool_samples": int(pool.sum()), "purity": selected_clean / len(selected) if selected else "", "clean_recall": selected_clean / pool_clean if pool_clean else "", "noisy_retention": selected_noisy / pool_noisy if pool_noisy else "", "active_ratio": len(selected) / int(pool.sum())})
        if task.method_key == "pgdf":
            by_class: dict[str, list[Any]] = defaultdict(list)
            for record in records:
                if record.prototype_score is not None:
                    by_class[record.observed_label].append(record)
            aucs = [_rank_auc([float(item.prototype_score) for item in rows], [clean[index_of(item)] == observed[index_of(item)] for item in rows]) for rows in by_class.values()]
            aucs = [item for item in aucs if item is not None]
            auc_rows.append({"dataset": task.dataset_key, "seed": task.seed, "epoch": epoch, "macro_auc": statistics.mean(aucs) if aucs else "", "valid_observed_classes": len(aucs), "score_direction": "raw_higher_is_clean;_not_auto_reversed"})
    return quality, auc_rows


def _rank_auc(scores: list[float], positives: list[bool]) -> float | None:
    positives_count = sum(positives); negatives_count = len(positives) - positives_count
    if not positives_count or not negatives_count:
        return None
    order = sorted(range(len(scores)), key=lambda item: scores[item]); ranks = [0.0] * len(scores); position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        for ordinal in range(position, end):
            ranks[order[ordinal]] = (position + 1 + end) / 2
        position = end
    return (sum(ranks[index] for index, positive in enumerate(positives) if positive) - positives_count * (positives_count + 1) / 2) / (positives_count * negatives_count)


def analyze(tasks: list[Task], prepared: dict[str, dict[str, Any]]) -> dict[str, int]:
    states = {task: task_status(task) for task in tasks}
    counts: dict[str, int] = defaultdict(int)
    for state, _ in states.values(): counts[state] += 1
    if not any(state == "COMPLETE_VERIFIED" for state, _ in states.values()):
        raise RuntimeError("No verified formal run exists; refusing to generate placeholder result summaries.")
    root = tasks[0].output_root
    results: list[dict[str, Any]] = []; profiles: list[dict[str, Any]] = []; quality: list[dict[str, Any]] = []; auc: list[dict[str, Any]] = []
    for task, (state, _) in states.items():
        if state != "COMPLETE_VERIFIED":
            continue
        result = json.loads((task.run_dir / "result.json").read_text(encoding="utf-8"))
        results.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed, "official_test_top1_percent": 100 * float(result["validation_selected_test_top1"]), "best_val_epoch": result["best_epoch"], "best_val_top1_percent": 100 * float(result["best_val_top1"])})
        profile = json.loads((task.instrumentation_root / task.experiment_id / "profiling.json").read_text(encoding="utf-8"))
        stage = profile["stage_totals"]; memory = profile["memory"]["overall_run_peak"]
        logs = read_csv(task.run_dir / "train_log.csv")
        total_seconds = stage["total_training"]["duration_seconds"]
        selection_seconds = stage["selection_update_total"]["duration_seconds"]
        profiles.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed, "training_function_wall_clock_seconds": total_seconds, "fit_wall_clock_seconds": stage["fit_time"]["duration_seconds"], "validation_wall_clock_seconds": stage["validation"]["duration_seconds"], "official_test_wall_clock_seconds": stage["official_test"]["duration_seconds"], "checkpoint_io_wall_clock_seconds": stage["checkpoint_save"]["duration_seconds"], "selection_update_wall_clock_seconds": selection_seconds, "selection_overhead_fraction_of_training_function": float(selection_seconds or 0.0) / float(total_seconds) if total_seconds else "", "peak_allocated_bytes": memory["allocated_bytes"], "peak_reserved_bytes": memory["reserved_bytes"], "optimizer_steps_actual": profile["optimizer_steps_actual"], "scheduler_steps_actual": profile["scheduler_steps_actual"], "sample_exposure": sum(int(float(item.get("train_samples", 0))) for item in logs), "effective_batch": logs[-1].get("effective_batch_size", ""), "microbatch": logs[-1].get("microbatch_size", ""), "gradient_accumulation_steps": logs[-1].get("gradient_accumulation_steps", "")})
        q, a = _selection_quality(task, prepared[task.dataset_key]); quality.extend(q); auc.extend(a)
    analysis = root / "analysis"
    write_csv(analysis / "seed_level_official_test.csv", results, list(results[0]))
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for dataset in FORMAL_DATASETS:
        by_method = {method: [row for row in results if row["dataset"] == dataset and row["method"] == method] for method, _, _ in METHODS}
        for method, rows in by_method.items():
            values = [float(row["official_test_top1_percent"]) for row in rows]
            if values: summary.append({"dataset": dataset, "method": method, "n": len(values), "mean_top1_percent": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else ""})
        pgdf = {row["seed"]: row["official_test_top1_percent"] for row in by_method["pgdf"]}
        for baseline in ("all_noisy_ce", "dynamic_small_loss", "jal_ce"):
            base = {row["seed"]: row["official_test_top1_percent"] for row in by_method[baseline]}
            deltas = [pgdf[seed] - base[seed] for seed in sorted(set(pgdf) & set(base))]
            if deltas: paired.append({"dataset": dataset, "comparison": f"PGDF - {baseline}", "paired_seeds": len(deltas), "mean_difference_pp": statistics.mean(deltas), "sample_sd_pp": statistics.stdev(deltas) if len(deltas) > 1 else "", "wins": sum(value > 0 for value in deltas), "ties": sum(value == 0 for value in deltas), "losses": sum(value < 0 for value in deltas)})
    write_csv(analysis / "accuracy_summary.csv", summary, list(summary[0]))
    if paired: write_csv(analysis / "paired_accuracy_summary.csv", paired, list(paired[0]))
    write_csv(analysis / "seed_level_profiling.csv", profiles, list(profiles[0]))
    cost_summary: list[dict[str, Any]] = []
    cost_metrics = (
        "training_function_wall_clock_seconds", "fit_wall_clock_seconds", "validation_wall_clock_seconds",
        "official_test_wall_clock_seconds", "checkpoint_io_wall_clock_seconds", "selection_update_wall_clock_seconds",
        "selection_overhead_fraction_of_training_function", "peak_allocated_bytes", "peak_reserved_bytes",
        "optimizer_steps_actual", "scheduler_steps_actual", "sample_exposure",
    )
    for dataset in FORMAL_DATASETS:
        for method, _, _ in METHODS:
            matched = [row for row in profiles if row["dataset"] == dataset and row["method"] == method]
            if not matched:
                continue
            row: dict[str, Any] = {
                "dataset": dataset,
                "method": method,
                "n": len(matched),
                "effective_batch": matched[0]["effective_batch"],
                "microbatch": matched[0]["microbatch"],
                "gradient_accumulation_steps": matched[0]["gradient_accumulation_steps"],
                "timing_boundary": "training_function includes setup, fitting, validation, selection where applicable, official test, and checkpoint I/O; nested stage durations are reported separately and must not be summed.",
            }
            for metric in cost_metrics:
                values = [float(item[metric]) for item in matched if item.get(metric) not in ("", None)]
                row[f"mean_{metric}"] = statistics.mean(values) if values else ""
                row[f"sample_sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else ""
            cost_summary.append(row)
    if cost_summary:
        write_csv(analysis / "training_cost_summary.csv", cost_summary, list(cost_summary[0]))
    if quality: write_csv(analysis / "selection_quality_by_update.csv", quality, list(quality[0]))
    if auc: write_csv(analysis / "prototype_reliability_summary.csv", auc, list(auc[0]))
    write_json(analysis / "completion_validation.json", {"status_counts": dict(counts), "formal_runs": FORMAL_RUNS, "complete_only_results_written": len(results)})
    write_text(root / "CROSS_BACKBONE_FINAL_REPORT.md", "# R4-C4 cross-backbone result report\n\nThis report is generated only from verified formal CLIP runs. See `analysis/` CSV artifacts for measured values.\n")
    return dict(counts)
