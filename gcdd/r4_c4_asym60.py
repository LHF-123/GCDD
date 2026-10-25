"""R4-C4 cyclic-asym60 preparation, execution accounting, and offline analysis.

This module intentionally orchestrates the established checkpoint-validation
entry point.  It does not implement a second PGDF, small-loss, or JAL-CE
algorithm.  Its only generated data are a deterministic 60% label manifest,
an input adapter containing the corresponding observed labels, and auditable
reports under the isolated R4-C4 output root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from gcdd.checkpoint_validation import build_or_load_fixed_validation_split, load_clean_labels_from_noise_index


METHODS: tuple[tuple[str, str, bool], ...] = (
    ("all_noisy_ce", "all_noisy", False),
    ("dynamic_small_loss", "dynamic_r08", True),
    ("jal_ce", "jal_ce", False),
    ("pgdf", "pgdf_dynamic_proto", True),
)
SELECTION_METHODS = {"dynamic_small_loss", "pgdf"}
FROZEN = {
    "noise_rate": 0.6,
    "noise_seed": 42,
    "validation_ratio": 0.1,
    "validation_seed": 20250726,
    "training_seeds": (1, 42, 88),
    "epochs": 30,
    "dynamic_ratio": 0.8,
    "fixed_p": 0.4,
    "warmup_epochs": 5,
    "update_interval": 5,
}


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
        return safe_component(self.dataset_name) + f"_adjacent_cyclic_noise42_{self.runner_method}_seed{self.seed}"


def safe_component(value: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_plan(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("R4-C4 configuration must be a mapping")
    return config


def resolve_config_paths(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    base = config_path.parent.resolve()
    copied = json.loads(json.dumps(config))
    for section, keys in {
        "output": ("root", "instrumentation_root"),
        "historical": ("repository_root",),
    }.items():
        for key in keys:
            if copied.get(section, {}).get(key):
                copied[section][key] = str((base / copied[section][key]).resolve())
    for dataset in copied.get("datasets", {}).values():
        for key in ("base_input_dir", "source_noise_index", "source_resolved_config"):
            if dataset.get(key):
                dataset[key] = str((base / dataset[key]).resolve())
    return copied


def validate_frozen_config(config: dict[str, Any]) -> None:
    formal = config.get("formal", {})
    for key, expected in FROZEN.items():
        actual = tuple(formal.get(key, ())) if key == "training_seeds" else formal.get(key)
        if actual != expected:
            raise ValueError(f"Frozen R4-C4 protocol mismatch for {key}: {actual!r} != {expected!r}")
    if tuple(formal.get("methods", ())) != tuple(key for key, _, _ in METHODS):
        raise ValueError("R4-C4 method matrix must be the frozen four-method protocol")
    if set(config.get("datasets", {})) != {"cub", "cars"}:
        raise ValueError("R4-C4 formal extension must include exactly CUB and Cars")


def numeric_label_key(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def normalized_path(path: str) -> str:
    return path.replace("\\", "/").rstrip("/")


def source_identity(row: dict[str, str]) -> str:
    return row.get("image_id") or row.get("index", "")


def generate_cyclic_manifest(source_rows: list[dict[str, str]], *, ratio: float, seed: int) -> list[dict[str, str]]:
    """Reproduce the historical per-class ``random.Random`` label corruption.

    CUB used image ids as sample identities; folder datasets used their source
    row indices.  Both builders advance one shared RNG while visiting sorted
    clean-label classes and use ``round(class_count * ratio)``.
    """
    ordered = sorted(source_rows, key=lambda row: int(row["index"]))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ordered:
        if row.get("split", "").lower() == "train":
            grouped[str(row["clean_label"])].append(row)
    classes = sorted(grouped, key=numeric_label_key)
    if not classes:
        raise ValueError("Source noise index has no training rows")
    target = {label: classes[(position + 1) % len(classes)] for position, label in enumerate(classes)}
    rng = random.Random(seed)
    noisy_ids: set[str] = set()
    for label in classes:
        values = grouped[label][:]
        rng.shuffle(values)
        noisy_ids.update(source_identity(row) for row in values[: int(round(len(values) * ratio))])
    output: list[dict[str, str]] = []
    for row in ordered:
        item = dict(row)
        clean = str(item["clean_label"])
        changed = item.get("split", "").lower() == "train" and source_identity(item) in noisy_ids
        item["web_label"] = target[clean] if changed else clean
        item["is_noisy"] = "1" if changed else "0"
        item["noise_target_label"] = target[clean]
        item["noise_type"] = "asymmetric"
        item["noise_strategy"] = "adjacent_cyclic"
        item["noise_ratio"] = f"{ratio:.6f}"
        item["noise_seed"] = str(seed)
        output.append(item)
    return output


def manifest_columns(rows: list[dict[str, str]]) -> list[str]:
    required = [
        "index", "path", "split", "clean_label", "web_label", "is_noisy", "noise_target_label",
        "noise_type", "noise_strategy", "noise_ratio", "noise_seed",
    ]
    extras = [field for field in rows[0] if field not in required] if rows else []
    return ["index", *extras, *[field for field in required if field != "index"]]


def row_path_keys(row: dict[str, str], dataset_root: str | None) -> set[str]:
    keys = {normalized_path(row.get("path", "")), normalized_path(row.get("abs_path", ""))}
    if dataset_root and row.get("path"):
        keys.add(normalized_path(str(Path(dataset_root) / row["path"])))
    return {key for key in keys if key}


def observed_labels_for_base(
    base_paths: list[str], manifest_rows: list[dict[str, str]], *, dataset_root: str | None
) -> tuple[np.ndarray, np.ndarray]:
    by_path: dict[str, dict[str, str]] = {}
    for row in manifest_rows:
        if row.get("split", "").lower() != "train":
            continue
        for key in row_path_keys(row, dataset_root):
            existing = by_path.get(key)
            if existing is not None and existing["index"] != row["index"]:
                raise ValueError(f"Ambiguous source path in manifest: {key}")
            by_path[key] = row
    labels: list[str] = []
    clean: list[str] = []
    missing: list[str] = []
    for path in base_paths:
        match = next((by_path[key] for key in row_path_keys({"path": path}, None) if key in by_path), None)
        if match is None:
            missing.append(path)
        else:
            labels.append(str(match["web_label"]))
            clean.append(str(match["clean_label"]))
    if missing:
        raise ValueError(f"60% manifest cannot be aligned to {len(missing)} base training paths; first={missing[0]}")
    return np.asarray(labels, dtype=str), np.asarray(clean, dtype=str)


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"Refusing to replace non-identical prepared input file: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def require_nested_values(payload: dict[str, Any], expected: dict[tuple[str, ...], Any], *, context: str) -> None:
    """Fail closed when a reference resolved config is not Aircraft-compatible."""
    for path, required in expected.items():
        current: Any = payload
        for component in path:
            if not isinstance(current, dict) or component not in current:
                raise ValueError(f"{context} is missing required historical setting {'.'.join(path)}")
            current = current[component]
        if current != required:
            raise ValueError(f"{context} has protocol difference {'.'.join(path)}={current!r}; expected {required!r}")


def prepare_dataset(config: dict[str, Any], key: str) -> dict[str, Any]:
    dataset = config["datasets"][key]
    output_root = Path(config["output"]["root"])
    source_index = Path(dataset["source_noise_index"])
    base_input = Path(dataset["base_input_dir"])
    source_config_path = Path(dataset["source_resolved_config"])
    for required in (source_index, base_input / "paths.txt", base_input / "eval_paths.txt", base_input / "eval_labels.npy", source_config_path):
        if not required.exists():
            raise FileNotFoundError(f"R4-C4 preparation prerequisite is missing: {required}")
    source_cfg = yaml.safe_load(source_config_path.read_text(encoding="utf-8")) or {}
    require_nested_values(
        source_cfg,
        {
            ("feature", "backend"): "dinov2_vitb14",
            ("feature", "input_size"): 448,
            ("lora", "rank"): 8,
            ("lora", "alpha"): 16.0,
            ("lora", "dropout"): 0.05,
            ("lora", "target_modules"): "qkv",
            ("lora_train", "epochs"): FROZEN["epochs"],
            ("lora_train", "batch_size"): 80,
            ("lora_train", "eval_batch_size"): 160,
            ("lora_train", "lora_lr"): 0.0001,
            ("lora_train", "head_lr"): 0.001,
            ("lora_train", "weight_decay"): 0.05,
            ("lora_train", "scheduler"): "cosine",
            ("lora_train", "warmup_ratio"): 0.1,
            ("checkpoint_validation", "jal", "alpha"): 1.0,
            ("checkpoint_validation", "jal", "beta"): 1.0,
            ("checkpoint_validation", "jal", "a"): 30.0,
            ("checkpoint_validation", "jal", "eps"): 1.0e-8,
        },
        context=f"{key} source resolved config",
    )
    source_rows = read_csv(source_index)
    # Establish equivalence to the already materialized 40% source index
    # before deriving its 60% sibling. This is stronger than assuming the
    # helper happens to resemble the historical generator.
    reproduced_source = generate_cyclic_manifest(source_rows, ratio=0.4, seed=FROZEN["noise_seed"])
    reproduction_mismatches = sum(
        any(
            generated.get(field, "") != original.get(field, "")
            for field in ("web_label", "is_noisy", "noise_target_label", "noise_strategy", "noise_ratio", "noise_seed")
        )
        for original, generated in zip(source_rows, reproduced_source, strict=True)
    )
    if reproduction_mismatches:
        raise ValueError(
            f"Historical-generator reproduction disagrees with {reproduction_mismatches} source rows for {key}; refusing 60% derivation."
        )
    manifest = generate_cyclic_manifest(source_rows, ratio=FROZEN["noise_rate"], seed=FROZEN["noise_seed"])
    manifest_path = output_root / "noise_manifests" / key / f"{key}_cyclic_asym60_s42_index.csv"
    write_csv(manifest_path, manifest, manifest_columns(manifest))
    base_paths = (base_input / "paths.txt").read_text(encoding="utf-8").splitlines()
    observed, clean = observed_labels_for_base(
        base_paths,
        manifest,
        dataset_root=str(source_cfg.get("dataset", {}).get("root", "")),
    )
    prepared = output_root / "prepared_inputs" / key
    for name in ("paths.txt", "eval_paths.txt", "eval_labels.npy"):
        hardlink_or_copy(base_input / name, prepared / name)
    labels_path = prepared / "labels.npy"
    if labels_path.exists():
        existing = np.load(labels_path, allow_pickle=True).astype(str)
        if not np.array_equal(existing, observed):
            raise FileExistsError(f"Prepared labels disagree with frozen manifest: {labels_path}")
    else:
        prepared.mkdir(parents=True, exist_ok=True)
        np.save(labels_path, observed)
    resolved = json.loads(json.dumps(source_cfg))
    resolved.setdefault("dataset", {})["name"] = f"{dataset['display_name']}-asym60-s42"
    resolved["dataset"]["index_file"] = str(manifest_path)
    resolved.setdefault("output", {})["root"] = str(output_root)
    resolved["output"]["version"] = f"r4_c4_{key}_asym60_s42"
    resolved.setdefault("checkpoint_validation", {})
    resolved["checkpoint_validation"].update(
        {
            "protocol": "fixed_clean_validation_v1",
            "validation_ratio": FROZEN["validation_ratio"],
            "validation_seed": FROZEN["validation_seed"],
            "dynamic_ratio": FROZEN["dynamic_ratio"],
            "fixed_p": FROZEN["fixed_p"],
            "warmup_epochs": FROZEN["warmup_epochs"],
            "update_interval": FROZEN["update_interval"],
            "posthoc_oracle_test": False,
            "official_test_selected_only": True,
            "noise_realization": {
                "noise_strategy": "adjacent_cyclic",
                "noise_ratio": FROZEN["noise_rate"],
                "noise_seed": FROZEN["noise_seed"],
                "mapping_type": "cyclic",
                "mapping_seed": "",
            },
            "noise_index_sha256": sha256_file(manifest_path),
        }
    )
    prepared.mkdir(parents=True, exist_ok=True)
    prepared_config = prepared / "resolved_config.yaml"
    yaml_text = yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False)
    if prepared_config.exists() and prepared_config.read_text(encoding="utf-8") != yaml_text:
        raise FileExistsError(f"Prepared resolved config already exists with different content: {prepared_config}")
    if not prepared_config.exists():
        prepared_config.write_text(yaml_text, encoding="utf-8")
    validation_dir = output_root / "fixed_validation" / key
    # The established helper writes its two manifest files atomically at this
    # caller-owned location but intentionally does not create a new parent.
    validation_dir.mkdir(parents=True, exist_ok=True)
    split = build_or_load_fixed_validation_split(
        validation_dir,
        base_paths,
        clean,
        validation_ratio=FROZEN["validation_ratio"],
        validation_seed=FROZEN["validation_seed"],
    )
    return {
        "key": key,
        "dataset_name": resolved["dataset"]["name"],
        "manifest_path": manifest_path,
        "prepared_input": prepared,
        "validation_dir": validation_dir,
        "manifest_rows": manifest,
        "base_paths": base_paths,
        "clean_labels": clean,
        "observed_labels": observed,
        "split": split,
        "source_config": source_cfg,
        "source_index": source_index,
        "source_generator_reproduction_mismatches": reproduction_mismatches,
    }


def tasks_for(config: dict[str, Any], prepared: dict[str, dict[str, Any]]) -> list[Task]:
    output_root = Path(config["output"]["root"])
    instrumentation = Path(config["output"]["instrumentation_root"])
    tasks: list[Task] = []
    for dataset_key in ("cub", "cars"):
        info = prepared[dataset_key]
        for method_key, runner_method, selection_logging in METHODS:
            for seed in FROZEN["training_seeds"]:
                tasks.append(Task(dataset_key, info["dataset_name"], method_key, runner_method, selection_logging, seed,
                                  output_root, info["prepared_input"], info["manifest_path"], info["validation_dir"], instrumentation,
                                  tuple((str(item[0]), str(item[1])) for item in config["datasets"][dataset_key].get("path_maps", []))))
    return tasks


def task_command(task: Task) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable, str(root / "scripts" / "run_lora_checkpoint_validation.py"),
        "--input-dir", str(task.input_dir), "--noise-index", str(task.manifest_path),
        "--output-dir", str(task.trainer_root), "--methods", task.runner_method, "--seeds", str(task.seed),
        "--validation-ratio", str(FROZEN["validation_ratio"]), "--validation-seed", str(FROZEN["validation_seed"]),
        "--dynamic-ratio", str(FROZEN["dynamic_ratio"]), "--fixed-p", str(FROZEN["fixed_p"]),
        "--warmup-epochs", str(FROZEN["warmup_epochs"]), "--update-interval", str(FROZEN["update_interval"]),
        "--official-test-selected-only", "--no-posthoc-oracle-test",
        "--revision-profiling-mode", "detailed", "--revision-output-root", str(task.instrumentation_root),
    ]
    for old_root, new_root in task.path_maps:
        command.extend(("--path-map", f"{old_root}={new_root}"))
    if task.selection_logging:
        command.append("--revision-selection-logging")
    return command


def stage_shared_validation(task: Task) -> None:
    for name in ("validation_manifest.csv", "validation_manifest.json"):
        hardlink_or_copy(task.validation_dir / name, task.trainer_root / name)


def runtime_environment() -> dict[str, Any]:
    """Record runtime identity without constructing a model or touching a dataset."""
    environment: dict[str, Any] = {"python_executable": sys.executable}
    try:
        import torch

        environment.update({"torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda})
    except ImportError:
        environment.update({"torch_version": "unavailable", "torch_cuda_version": "unavailable"})
    environment["gpu_query"] = gpu_status(0)
    return environment


def task_status(task: Task) -> tuple[str, list[str]]:
    issues: list[str] = []
    result = task.run_dir / "result.json"
    if not result.exists():
        if (task.task_dir / "FAILED.json").exists():
            return "FAILED", ["prior_attempt_failed_preserved"]
        if task.trainer_root.exists() and any(task.trainer_root.iterdir()):
            return "PARTIAL", ["partial_output_not_eligible_for_automatic_resume"]
        return "PENDING", []
    for path in (task.run_dir / "checkpoints" / "best_val.pt", task.run_dir / "checkpoints" / "last.pt", task.run_dir / "train_log.csv"):
        if not path.is_file():
            issues.append(f"missing:{path.name}")
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
        if int(payload.get("seed", -1)) != task.seed or payload.get("method_key") != task.runner_method:
            issues.append("result_identity_mismatch")
        expected_result = {
            "noise_strategy": "adjacent_cyclic",
            "noise_ratio": FROZEN["noise_rate"],
            "noise_seed": FROZEN["noise_seed"],
            "validation_seed": FROZEN["validation_seed"],
            "checkpoint_protocol": "fixed_clean_validation_v1",
        }
        for field, expected in expected_result.items():
            observed = payload.get(field)
            if field == "noise_ratio":
                try:
                    mismatch = float(observed) != float(expected)
                except (TypeError, ValueError):
                    mismatch = True
            else:
                mismatch = observed != expected
            if mismatch:
                issues.append(f"result_{field}_mismatch")
        if int(payload.get("best_epoch", 0)) < 1 or payload.get("official_test_evaluation") != "validation_selected_only":
            issues.append("checkpoint_or_official_test_protocol_mismatch")
        if payload.get("validation_selected_test_top1", "") == "":
            issues.append("missing_validation_selected_official_test")
    except (json.JSONDecodeError, OSError, ValueError):
        issues.append("invalid_result_json")
    profile = task.instrumentation_root / task.experiment_id / "profiling.json"
    if not profile.is_file():
        issues.append("missing_profiling")
    elif json.loads(profile.read_text(encoding="utf-8")).get("recording_status") != "complete":
        issues.append("profiling_incomplete")
    else:
        profiling = json.loads(profile.read_text(encoding="utf-8"))
        if profiling.get("profiling_mode") != "detailed" or profiling.get("optimizer_steps_actual") is None:
            issues.append("profiling_fields_incomplete")
    log_rows = read_csv(task.run_dir / "train_log.csv") if (task.run_dir / "train_log.csv").is_file() else []
    if sorted({int(row.get("epoch", 0)) for row in log_rows}) != list(range(1, FROZEN["epochs"] + 1)):
        issues.append("train_log_not_exactly_30_epochs")
    for name in ("validation_manifest.csv", "validation_manifest.json"):
        shared = task.validation_dir / name
        copied = task.trainer_root / name
        if not copied.is_file() or not shared.is_file() or sha256_file(shared) != sha256_file(copied):
            issues.append(f"validation_manifest_not_shared:{name}")
    if not (task.task_dir / "runtime_environment.json").is_file():
        issues.append("missing_runtime_environment")
    if task.selection_logging:
        records = task.instrumentation_root / task.experiment_id / "selection_records"
        if not records.is_dir():
            issues.append("missing_selection_instrumentation")
        for name in ("selection_rows.csv", "selection_updates.csv", "selection_per_class.csv"):
            if not (task.run_dir / name).is_file():
                issues.append(f"missing:{name}")
        updates = read_csv(task.run_dir / "selection_updates.csv") if (task.run_dir / "selection_updates.csv").is_file() else []
        if sorted({int(row.get("epoch", 0)) for row in updates}) != [5, 10, 15, 20, 25]:
            issues.append("selection_updates_not_expected_five_stages")
    return ("COMPLETE_VERIFIED" if not issues else "PARTIAL"), issues


def gpu_status(required_gib: int) -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "GPU_QUERY_UNAVAILABLE", "reason": str(exc)}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"status": "GPU_QUERY_UNAVAILABLE", "reason": completed.stderr.strip() or "nvidia-smi failed"}
    devices = []
    for line in completed.stdout.splitlines():
        name, total, free, utilisation = [value.strip() for value in line.split(",")]
        devices.append({"name": name, "total_mib": int(total), "free_mib": int(free), "utilization_percent": int(utilisation)})
    eligible = [device for device in devices if device["total_mib"] >= required_gib * 1024]
    return {
        "status": "READY" if eligible else "READY_WAIT_GPU",
        "minimum_total_gib": required_gib,
        "devices": devices,
        "reason": "No GPU meets the frozen 24 GiB minimum for formal DINOv2-B/14+LoRA runs." if not eligible else "eligible_gpu_present",
    }


def build_noise_audit(info: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = info["manifest_rows"]
    split = info["split"]
    clean = info["clean_labels"]
    observed = info["observed_labels"]
    source_train = [row for row in rows if row.get("split", "").lower() == "train"]
    source_flips = sum(row["is_noisy"] == "1" for row in source_train)
    pool = np.asarray(split.train_mask, dtype=bool)
    pool_flips = int(np.sum(clean[pool] != observed[pool]))
    composition: list[dict[str, Any]] = []
    transition: list[dict[str, Any]] = []
    for label in sorted(set(observed[pool].tolist()), key=numeric_label_key):
        indices = np.where(pool & (observed == label))[0]
        clean_count = int(np.sum(clean[indices] == observed[indices]))
        noisy_count = len(indices) - clean_count
        origins: dict[str, int] = defaultdict(int)
        for index in indices:
            origins[str(clean[index])] += 1
        wrong = [(source, count) for source, count in origins.items() if source != label]
        major = max(wrong, key=lambda item: item[1]) if wrong else ("", 0)
        composition.append({
            "dataset": info["key"], "observed_label": label, "total": len(indices), "clean_count": clean_count,
            "noisy_count": noisy_count, "clean_ratio": clean_count / len(indices), "noise_ratio": noisy_count / len(indices),
            "noisy_majority": noisy_count > clean_count, "major_error_source": major[0],
            "major_error_source_count": major[1], "major_error_source_ratio": major[1] / len(indices),
        })
        for source, count in sorted(origins.items(), key=lambda pair: numeric_label_key(pair[0])):
            transition.append({"dataset": info["key"], "true_label": source, "observed_label": label, "count": count})
    audit = {
        "dataset": info["key"], "source_train_samples": len(source_train), "source_train_flips": source_flips,
        "source_train_realized_noise_rate": source_flips / len(source_train),
        "training_pool_samples": int(pool.sum()), "training_pool_flips": pool_flips,
        "training_pool_realized_noise_rate": pool_flips / int(pool.sum()), "observed_classes": len(composition),
        "noisy_majority_observed_classes": sum(bool(row["noisy_majority"]) for row in composition),
    }
    return audit, composition, transition


def historical_audit(config: dict[str, Any]) -> str:
    root = Path(config["historical"]["repository_root"])
    current = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_pgdf_dynamic_proto_aircraft_noise_rates.sh"
    pgdf = root / "outputs" / "FGVC_Aircraft" / "FGVC-Aircraft-asym60-s42" / "v1_aircraft_asym_r0p6_s42" / "checkpoint_validation_pgdf_dynamic_proto_r08_p04_s20250726"
    controls = root / "outputs" / "FGVC_Aircraft" / "FGVC-Aircraft-asym60-s42" / "v1_aircraft_asym_r0p6_s42" / "checkpoint_validation_s20250726"
    return "\n".join([
        "# 历史 FGVC-Aircraft cyclic-asym60 协议审计", "",
        "## 已核验的历史入口与资产", "",
        f"- 历史启动脚本：`{script}`（{'FOUND' if script.is_file() else 'NOT_FOUND'}）。",
        f"- PGDF 验证根目录：`{pgdf}`（{'FOUND' if pgdf.is_dir() else 'NOT_FOUND'}）。",
        f"- All-noisy / Dynamic small-loss / JAL-CE 验证根目录：`{controls}`（{'FOUND' if controls.is_dir() else 'NOT_FOUND'}）。",
        "- 共同入口为 `scripts/run_lora_checkpoint_validation.py`；静态 All-noisy/JAL-CE 经 `gcdd/lora_training.py`，Dynamic small-loss/PGDF 经 `gcdd/lora_dynamic.py`。",
        "", "## 冻结的历史配置", "",
        "| 项目 | 实际值 |", "|---|---|", "| Backbone | DINOv2 ViT-B/14 |",
        "| LoRA | rank 8，alpha 16，dropout 0.05，目标 `qkv` |", "| 训练 | 30 epochs；batch 80；eval batch 160；LoRA LR 1e-4；head LR 1e-3；AdamW WD 0.05；cosine warmup 0.1 |",
        "| 噪声 | adjacent cyclic；名义 0.6；noise seed 42；cyclic 映射 |", "| 验证/测试 | fixed clean validation 0.1、seed 20250726；official test 只使用 validation-selected checkpoint |",
        "| Dynamic small-loss | r=0.8；warmup 5；每 5 epoch 更新 |", "| PGDF | strict loss/prototype intersection；r=0.8，p=0.4；当前 LoRA 特征动态原型；warmup 5；每 5 epoch 更新 |",
        "| JAL-CE | alpha=1，beta=1，a=30，eps=1e-8 |", "", "## 向本扩展的映射", "",
        "| 组件 | 状态 | 证据 |", "|---|---|---|", "| CLI 协议与冻结参数 | BEHAVIORALLY_EQUIVALENT | 复用相同 checkpoint-validation 入口/选项，并对 CUB/Cars resolved config 逐字段 fail-closed 核验。 |",
        "| 方法源码身份 | UNVERIFIED | 历史与 major-revision worktree 的源码哈希不同；本扩展记录来源/配置指纹，不声称字节完全相同。 |",
        "| 静态 CE/JAL 的 profiler | HARMLESS_INFRASTRUCTURE_DIFFERENCE | 仅 opt-in 观测既有步骤，不改变 mask、损失、optimizer、scheduler 或 checkpoint 选择。 |",
        "| 数据集噪声实现 | BEHAVIORALLY_EQUIVALENT | 重现已保存 40% index 为零差异后，复用每类 `random.Random(seed)`、`round(n_c*r)`、排序类别和 next-class cyclic 映射。 |",
        "", "## Source identity fingerprints", "",
        "| File | historical SHA-256 | major-revision SHA-256 |", "|---|---|---|",
        *[
            f"| `{relative}` | `{sha256_file(root / relative) if (root / relative).is_file() else 'NOT_FOUND'}` | `{sha256_file(current / relative) if (current / relative).is_file() else 'NOT_FOUND'}` |"
            for relative in ("scripts/run_lora_checkpoint_validation.py", "gcdd/lora_training.py", "gcdd/lora_dynamic.py")
        ],
    ]) + "\n"


def write_prepared_assets(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[Task]]:
    validate_frozen_config(config)
    prepared = {key: prepare_dataset(config, key) for key in ("cub", "cars")}
    root = Path(config["output"]["root"])
    write_text(root / "HISTORICAL_AIRCRAFT_PROTOCOL_AUDIT.md", historical_audit(config))
    write_text(
        root / "historical_to_current_code_mapping.md",
        "# 历史到当前实现映射\n\n"
        "| 历史方法角色 | 当前扩展路径 | 状态 |\n|---|---|---|\n"
        "| All-noisy CE | `run_lora_checkpoint_validation.py --methods all_noisy` → `train_dinov2_lora` | BEHAVIORALLY_EQUIVALENT |\n"
        "| Dynamic small-loss r=0.8 | `--methods dynamic_r08` → `train_dynamic_loss_lora` | BEHAVIORALLY_EQUIVALENT |\n"
        "| JAL-CE | `--methods jal_ce` → 带历史 JAL 参数的 `train_dinov2_lora` | BEHAVIORALLY_EQUIVALENT |\n"
        "| PGDF DynamicProto | `--methods pgdf_dynamic_proto` → `train_dynamic_loss_lora` | BEHAVIORALLY_EQUIVALENT |\n"
        "\n源码树身份标记为 `UNVERIFIED`：历史与当前 worktree 并非字节完全相同。\n",
    )
    audit_rows: list[dict[str, Any]] = []
    compositions: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for info in prepared.values():
        audit, composition, transition = build_noise_audit(info)
        audit_rows.append(audit); compositions.extend(composition); transitions.extend(transition)
    write_csv(root / "noise_protocol_audit.csv", audit_rows, list(audit_rows[0]))
    write_csv(root / "observed_class_composition.csv", compositions, list(compositions[0]))
    write_csv(root / "noise_transition_matrix.csv", transitions, ["dataset", "true_label", "observed_label", "count"])
    provenance = {
        "schema_version": "r4_c4_asym60_v1", "formal_protocol": FROZEN,
        "source_files": {key: {"source_noise_index": str(info["source_index"]), "source_noise_index_sha256": sha256_file(info["source_index"]),
                               "source_resolved_config": str(config["datasets"][key]["source_resolved_config"]),
                               "source_resolved_config_sha256": sha256_file(Path(config["datasets"][key]["source_resolved_config"])),
                               "source_40pct_generator_reproduction_mismatches": info["source_generator_reproduction_mismatches"],
                               "asym60_manifest": str(info["manifest_path"]), "asym60_manifest_sha256": sha256_file(info["manifest_path"]),
                               "prepared_input": str(info["prepared_input"]), "validation_manifest": str(info["validation_dir"] / "validation_manifest.json")}
                         for key, info in prepared.items()},
    }
    write_json(root / "source_and_config_provenance.json", provenance)
    tasks = tasks_for(config, prepared)
    current_root = Path(__file__).resolve().parents[1]
    code_fingerprints = {
        relative: sha256_file(current_root / relative)
        for relative in ("scripts/run_lora_checkpoint_validation.py", "gcdd/lora_training.py", "gcdd/lora_dynamic.py", "revision_experiments/profiling.py", "revision_experiments/training_integration.py")
    }
    for task in tasks:
        write_json(
            root / "resolved_configs" / f"{task.dataset_key}_{task.method_key}_seed{task.seed}.json",
            {
                "schema_version": "r4_c4_planned_resolved_run_v1",
                "dataset": task.dataset_key,
                "dataset_name": task.dataset_name,
                "method": task.method_key,
                "runner_method": task.runner_method,
                "training_seed": task.seed,
                "frozen_protocol": FROZEN,
                "input_dir": str(task.input_dir),
                "noise_manifest": str(task.manifest_path),
                "noise_manifest_sha256": sha256_file(task.manifest_path),
                "validation_manifest": str(task.validation_dir / "validation_manifest.json"),
                "validation_manifest_sha256": sha256_file(task.validation_dir / "validation_manifest.json"),
                "command": task_command(task),
                "selection_logging": task.selection_logging,
                "profiling_mode": "detailed",
                "source_code_sha256": code_fingerprints,
                "note": "The trainer writes its actual merged resolved_config.yaml under trainer/ when this planned run starts; this file is the frozen pre-launch specification.",
            },
        )
    inventory = []
    for task in tasks:
        state, issues = task_status(task)
        inventory.append({"dataset": task.dataset_key, "method": task.method_key, "runner_method": task.runner_method, "seed": task.seed,
                          "status": state, "issues": ";".join(issues), "trainer_root": str(task.trainer_root), "command": " ".join(task_command(task))})
    write_csv(root / "experiment_manifest.csv", inventory, list(inventory[0]))
    return prepared, tasks


def run_tasks(tasks: list[Task], *, resume: bool) -> int:
    for task in tasks:
        state, issues = task_status(task)
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
        write_json(task.task_dir / "RUNNING.json", {"command": task_command(task), "pid": os.getpid()})
        write_json(task.task_dir / "runtime_environment.json", runtime_environment())
        log = task.output_root / "logs" / task.dataset_key / task.method_key / f"seed{task.seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(task_command(task), stdout=handle, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1], check=False)
        (task.task_dir / "RUNNING.json").unlink(missing_ok=True)
        if completed.returncode != 0:
            write_json(task.task_dir / "FAILED.json", {"returncode": completed.returncode, "log": str(log), "command": task_command(task)})
            print(f"FAILED {task.dataset_key}/{task.method_key}/seed{task.seed}; preserved {log}", flush=True)
            return completed.returncode
        state, issues = task_status(task)
        if state != "COMPLETE_VERIFIED":
            write_json(task.task_dir / "FAILED.json", {"returncode": 0, "completion_validation_issues": issues, "log": str(log)})
            print(f"INCOMPLETE {task.dataset_key}/{task.method_key}/seed{task.seed}: {';'.join(issues)}", flush=True)
            return 4
    return 0


def rank_auc(scores: list[float], positives: list[bool]) -> float | None:
    n_pos = sum(positives); n_neg = len(positives) - n_pos
    if not n_pos or not n_neg:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average = (position + 1 + end) / 2.0
        for ordinal in range(position, end): ranks[order[ordinal]] = average
        position = end
    return (sum(ranks[index] for index, positive in enumerate(positives) if positive) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def completed_result(task: Task) -> dict[str, Any] | None:
    state, _ = task_status(task)
    if state != "COMPLETE_VERIFIED": return None
    return json.loads((task.run_dir / "result.json").read_text(encoding="utf-8"))


def selection_diagnostics(task: Task, info: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if task.method_key not in SELECTION_METHODS or completed_result(task) is None: return [], [], []
    rows = read_csv(task.run_dir / "selection_rows.csv")
    pool = np.asarray(info["split"].train_mask, dtype=bool); clean = info["clean_labels"]; observed = info["observed_labels"]
    pool_clean = int(np.sum(pool & (clean == observed))); pool_noisy = int(pool.sum()) - pool_clean
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        index = int(row["index"])
        if pool[index]: grouped[int(row["epoch"])].append(row)
    summary: list[dict[str, Any]] = []; class_rows: list[dict[str, Any]] = []; auc_rows: list[dict[str, Any]] = []
    for epoch, records in sorted(grouped.items()):
        selected = [row for row in records if row.get("state") == "selected"]
        selected_clean = sum(clean[int(row["index"])] == observed[int(row["index"])] for row in selected)
        selected_noisy = len(selected) - selected_clean
        row = {"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed, "epoch": epoch,
               "selected": len(selected), "pool_samples": int(pool.sum()), "purity": selected_clean / len(selected) if selected else "",
               "clean_recall": selected_clean / pool_clean if pool_clean else "", "noisy_retention": selected_noisy / pool_noisy if pool_noisy else "",
               "active_ratio": len(selected) / int(pool.sum()), "loss_candidates": sum(item.get("loss_selected") == "yes" for item in records),
               "prototype_candidates": sum(item.get("proto_pass") == "yes" for item in records),
               "intersection_candidates": sum(item.get("loss_selected") == "yes" and item.get("proto_pass") == "yes" for item in records)}
        summary.append(row)
        by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
        for record in records: by_class[record["web_label"]].append(record)
        auc_values: list[float] = []
        for label, items in by_class.items():
            chosen = [item for item in items if item.get("state") == "selected"]
            class_clean = sum(clean[int(item["index"])] == observed[int(item["index"])] for item in items)
            chosen_clean = sum(clean[int(item["index"])] == observed[int(item["index"])] for item in chosen)
            class_rows.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed, "epoch": epoch, "observed_label": label,
                               "pool_count": len(items), "selected": len(chosen), "purity": chosen_clean / len(chosen) if chosen else "",
                               "clean_recall": chosen_clean / class_clean if class_clean else ""})
            scored = [item for item in items if item.get("proto_score", "") not in {"", "nan", "NaN"}]
            auc = rank_auc([float(item["proto_score"]) for item in scored], [clean[int(item["index"])] == observed[int(item["index"])] for item in scored]) if task.method_key == "pgdf" else None
            if auc is not None: auc_values.append(auc)
        if task.method_key == "pgdf":
            scored_all = [item for item in records if item.get("proto_score", "") not in {"", "nan", "NaN"}]
            auc_rows.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed, "epoch": epoch,
                             "macro_auc": statistics.mean(auc_values) if auc_values else "", "valid_observed_classes": len(auc_values),
                             "micro_auc": rank_auc([float(item["proto_score"]) for item in scored_all], [clean[int(item["index"])] == observed[int(item["index"])] for item in scored_all]) if scored_all else "",
                             "score_direction": "raw_higher_is_clean;_not_auto_reversed"})
    return summary, class_rows, auc_rows


def analyze(config: dict[str, Any], prepared: dict[str, dict[str, Any]], tasks: list[Task]) -> dict[str, int]:
    root = Path(config["output"]["root"]); inventory = []; results = []; profiles = []; epoch_profiles = []; quality = []; class_quality = []; auc = []
    count: dict[str, int] = defaultdict(int)
    for task in tasks:
        state, issues = task_status(task); count[state] += 1
        inventory.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed, "status": state, "issues": ";".join(issues),
                          "run_dir": str(task.run_dir), "instrumentation_id": task.experiment_id})
        result = completed_result(task)
        if result is not None:
            results.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed,
                            "official_test_top1_percent": 100 * float(result["validation_selected_test_top1"]), "best_val_epoch": result["best_val_epoch"],
                            "best_val_top1_percent": 100 * float(result["best_val_top1"]), "run_dir": str(task.run_dir)})
            profile = json.loads((task.instrumentation_root / task.experiment_id / "profiling.json").read_text(encoding="utf-8"))
            memory = profile["memory"]["overall_run_peak"]
            stage_totals = profile["stage_totals"]
            train_log = read_csv(task.run_dir / "train_log.csv")
            profiles.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed,
                             "wall_clock_seconds": stage_totals["total_training"]["duration_seconds"],
                             "fit_seconds": stage_totals["fit_time"]["duration_seconds"], "validation_seconds": stage_totals["validation"]["duration_seconds"],
                             "official_test_seconds": stage_totals["official_test"]["duration_seconds"], "selection_seconds": stage_totals["selection_update_total"]["duration_seconds"],
                             "full_pool_reevaluation_seconds": sum(float(stage_totals[name]["duration_seconds"] or 0.0) for name in ("selection_loss_forward", "selection_cls_forward")),
                             "peak_allocated_bytes": memory["allocated_bytes"], "peak_reserved_bytes": memory["reserved_bytes"],
                             "optimizer_steps_actual": profile["optimizer_steps_actual"], "scheduler_steps_actual": profile["scheduler_steps_actual"],
                             "training_batches_seen": profile["training_batches_seen"], "training_samples_processed": sum(int(float(row.get("train_samples", 0))) for row in train_log),
                             "selection_updates": len(profile["selection_interval_steps"])})
            for event in profile.get("detailed_events", []):
                if event.get("name") == "epoch" and event.get("status") == "recorded":
                    phase = str(event.get("phase", ""))
                    epoch_profiles.append({"dataset": task.dataset_key, "method": task.method_key, "seed": task.seed,
                                           "epoch": int(phase.removeprefix("epoch_")), "epoch_wall_clock_seconds": event["duration_seconds"],
                                           "data_loading_included": event.get("data_loading_included", False)})
            q, cq, ar = selection_diagnostics(task, prepared[task.dataset_key]); quality.extend(q); class_quality.extend(cq); auc.extend(ar)
    write_csv(root / "training_run_inventory.csv", inventory, list(inventory[0]))
    write_json(root / "completion_validation.json", {"status_counts": dict(count), "tasks": inventory})
    if results:
        write_csv(root / "seed_level_official_test.csv", results, list(results[0]))
        summary = []
        for dataset in ("cub", "cars"):
            for method, _, _ in METHODS:
                values = [row["official_test_top1_percent"] for row in results if row["dataset"] == dataset and row["method"] == method]
                if values:
                    summary.append({"dataset": dataset, "method": method, "n": len(values), "mean_top1_percent": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else ""})
        write_csv(root / "accuracy_summary.csv", summary, list(summary[0]))
        paired = []
        for dataset in ("cub", "cars"):
            pgdf_by_seed = {row["seed"]: row["official_test_top1_percent"] for row in results if row["dataset"] == dataset and row["method"] == "pgdf"}
            for baseline in ("all_noisy_ce", "dynamic_small_loss", "jal_ce"):
                base_by_seed = {row["seed"]: row["official_test_top1_percent"] for row in results if row["dataset"] == dataset and row["method"] == baseline}
                differences = [pgdf_by_seed[seed] - base_by_seed[seed] for seed in sorted(set(pgdf_by_seed) & set(base_by_seed))]
                if differences:
                    paired.append({"dataset": dataset, "comparison": f"PGDF - {baseline}", "paired_seeds": len(differences),
                                   "mean_difference_pp": statistics.mean(differences), "sample_sd_pp": statistics.stdev(differences) if len(differences) > 1 else "",
                                   "wins": sum(value > 0 for value in differences), "ties": sum(value == 0 for value in differences), "losses": sum(value < 0 for value in differences),
                                   "interpretation": "descriptive_only_three_paired_seeds"})
        if paired: write_csv(root / "paired_accuracy_summary.csv", paired, list(paired[0]))
    if profiles:
        write_csv(root / "seed_level_profiling.csv", profiles, list(profiles[0]))
        if epoch_profiles: write_csv(root / "epoch_level_profiling.csv", epoch_profiles, list(epoch_profiles[0]))
        cost_summary = []
        metrics = ("wall_clock_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "optimizer_steps_actual", "scheduler_steps_actual", "training_batches_seen", "training_samples_processed", "selection_seconds", "full_pool_reevaluation_seconds")
        for dataset in ("cub", "cars"):
            for method, _, _ in METHODS:
                matching = [row for row in profiles if row["dataset"] == dataset and row["method"] == method]
                if not matching: continue
                row: dict[str, Any] = {"dataset": dataset, "method": method, "n": len(matching)}
                for metric in metrics:
                    values = [float(item[metric]) for item in matching if item.get(metric) is not None]
                    row[f"mean_{metric}"] = statistics.mean(values) if values else ""
                    row[f"sample_sd_{metric}"] = statistics.stdev(values) if len(values) > 1 else ""
                row["timing_boundary"] = "total_training includes model setup, fit/validation, selection when applicable, official test, checkpoint I/O; data loading only inside explicitly profiled loops"
                cost_summary.append(row)
        if cost_summary: write_csv(root / "compute_cost_summary.csv", cost_summary, list(cost_summary[0]))
    if quality: write_csv(root / "selection_quality_by_update.csv", quality, list(quality[0]))
    if class_quality: write_csv(root / "selection_quality_by_observed_class.csv", class_quality, list(class_quality[0]))
    if auc: write_csv(root / "prototype_reliability.csv", auc, list(auc[0]))
    noise_summary = []
    for key, info in prepared.items():
        audit, _, _ = build_noise_audit(info)
        noise_summary.append(
            f"- {key}: source-train 实际噪声率 {audit['source_train_realized_noise_rate']:.4%}；"
            f"training-pool 实际噪声率 {audit['training_pool_realized_noise_rate']:.4%}；"
            f"noisy-majority observed classes {audit['noisy_majority_observed_classes']}/{audit['observed_classes']}。"
        )
    report = [
        "# R4-C4 CUB/Cars cyclic-asym60 正式实验状态", "",
        "## 冻结协议", "",
        "- 24 个正式 run：CUB/Cars × All-noisy CE、Dynamic small-loss r=0.8、JAL-CE、PGDF DynamicProto × seeds 1/42/88。",
        "- 使用 adjacent-cyclic 60%、noise seed 42、clean validation 10%（seed 20250726）、30 epochs、validation-selected official-test。",
        "- CUB/Cars 已将历史 40% source index 以同一 generator 重现为零差异，再生成 60% manifest；此过程不使用 clean/noisy oracle 参与训练。",
        "", "## 已核验的噪声组成", "", *noise_summary, "", "## 当前训练状态", "",
        f"- 完成且核验通过：{count['COMPLETE_VERIFIED']}；失败：{count['FAILED']}；部分产物：{count['PARTIAL']}；未启动：{count['PENDING']}。",
        "- 完整性要求：30 个 epoch、共享 validation manifest、validation-selected official test、完整 checkpoint、详细 profiling；动态方法另要求 epoch 5/10/15/20/25 的原始选择记录。",
        "", "## 后续汇总规则", "",
        "正式 Top-1、三 seed 均值/SD、PGDF 的逐 seed 配对差、成本均值/SD，只有在对应 run 全部完成并通过上述核验后才写入正式 CSV。",
    ]
    write_text(root / "R4_C4_CUB_CARS_ASYM60_FINAL_REPORT.md", "\n".join(report) + "\n")
    return dict(count)
