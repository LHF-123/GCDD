from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .checkpoint_validation import PROTOCOL_NAME, build_or_load_fixed_validation_split, hash_paths
from .io_utils import read_csv, write_csv, write_json, write_yaml
from .selection_utils import path_key_candidates


AUDIT_FIELDS = [
    "dataset",
    "noise_rate",
    "noise_seed",
    "validation_seed",
    "v1_index",
    "source_sample_index",
    "stable_sample_id",
    "image_path",
    "clean_label",
    "noisy_label",
    "is_flipped",
    "cyclic_target_label",
    "is_validation",
    "is_training_pool",
]


def prepare_cub_noise_realization(
    *,
    base_input_dir: Path,
    noise_index_path: Path,
    validation_dir: Path,
    output_dir: Path,
    noise_seed: int,
    validation_seed: int,
    peer_noise_indices: list[Path] | None = None,
    create_validation_if_missing: bool = False,
) -> dict[str, Any]:
    """Create a label-only CUB realization while reusing label-independent features."""

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite realization input directory: {output_dir}")
    required_base = [
        base_input_dir / "paths.txt",
        base_input_dir / "eval_paths.txt",
        base_input_dir / "eval_labels.npy",
        base_input_dir / "features_cls.npy",
        base_input_dir / "features_gap.npy",
        base_input_dir / "features_top.npy",
        base_input_dir / "resolved_config.yaml",
    ]
    missing = [str(path) for path in required_base if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Base V1 input is incomplete: {missing}")
    if not noise_index_path.is_file():
        raise FileNotFoundError(f"Noise index does not exist: {noise_index_path}")

    train_paths = (base_input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    noise_rows = align_noise_rows(train_paths, read_csv(noise_index_path), source_name=str(noise_index_path))
    noise_summary = validate_cyclic_noise_rows(noise_rows, noise_seed=noise_seed, noise_rate=0.4)
    manifest_path = validation_dir / "validation_manifest.csv"
    manifest_json_path = validation_dir / "validation_manifest.json"
    manifest_exists = manifest_path.is_file()
    manifest_json_exists = manifest_json_path.is_file()
    if manifest_exists != manifest_json_exists:
        raise FileNotFoundError(
            f"Fixed validation manifest CSV/JSON is only partially present under {validation_dir}."
        )
    manifest_origin = "reused"
    if not manifest_exists:
        if not create_validation_if_missing:
            raise FileNotFoundError(f"Fixed validation manifest is missing under {validation_dir}.")
        validation_dir.mkdir(parents=True, exist_ok=False)
        clean_labels = np.asarray([str(row["clean_label"]) for row in noise_rows], dtype=str)
        build_or_load_fixed_validation_split(
            validation_dir,
            train_paths,
            clean_labels,
            validation_ratio=0.10,
            validation_seed=int(validation_seed),
        )
        manifest_origin = "deterministically_created"
    manifest_rows, manifest_metadata = load_and_validate_manifest(
        manifest_path,
        manifest_json_path,
        train_paths,
        noise_rows,
        validation_seed,
        expected_validation_ratio=0.10,
    )
    peer_comparisons = compare_peer_realizations(
        train_paths,
        noise_rows,
        peer_noise_indices or [],
    )

    audit_rows: list[dict[str, Any]] = []
    noisy_labels: list[str] = []
    for v1_index, (sample, manifest) in enumerate(zip(noise_rows, manifest_rows, strict=True)):
        partition = str(manifest["partition"])
        noisy_label = str(sample["web_label"])
        noisy_labels.append(noisy_label)
        audit_rows.append(
            {
                "dataset": "CUB-200-2011",
                "noise_rate": 0.4,
                "noise_seed": int(noise_seed),
                "validation_seed": int(validation_seed),
                "v1_index": int(v1_index),
                "source_sample_index": str(sample.get("index", "")),
                "stable_sample_id": str(sample.get("image_id", sample.get("index", ""))),
                "image_path": train_paths[v1_index],
                "clean_label": str(sample["clean_label"]),
                "noisy_label": noisy_label,
                "is_flipped": int(parse_bool(sample["is_noisy"])),
                "cyclic_target_label": str(sample["noise_target_label"]),
                "is_validation": int(partition == "validation"),
                "is_training_pool": int(partition == "training_pool"),
            }
        )

    pool_rows = [row for row in audit_rows if row["is_training_pool"] == 1]
    pool_flipped = sum(int(row["is_flipped"]) for row in pool_rows)
    pool_clean = len(pool_rows) - pool_flipped
    validation_rows = [row for row in audit_rows if row["is_validation"] == 1]

    output_dir.mkdir(parents=True)
    link_modes: dict[str, str] = {}
    shared_names = [
        "paths.txt",
        "eval_paths.txt",
        "eval_labels.npy",
        "features_cls.npy",
        "features_gap.npy",
        "features_top.npy",
    ]
    for optional_name in ("eval_features_cls.npy", "eval_features_gap.npy", "eval_features_top.npy"):
        if (base_input_dir / optional_name).is_file():
            shared_names.append(optional_name)
    for name in shared_names:
        link_modes[name] = hardlink_or_copy(base_input_dir / name, output_dir / name)

    np.save(output_dir / "labels.npy", np.asarray(noisy_labels, dtype=str))
    resolved_config = load_resolved_config(base_input_dir / "resolved_config.yaml")
    resolved_config.setdefault("dataset", {})
    resolved_config["dataset"]["name"] = f"CUB-200-2011-asym40-s{noise_seed}"
    resolved_config["dataset"]["index_file"] = str(noise_index_path)
    resolved_config.setdefault("output", {})
    resolved_config["output"]["version"] = f"v1_cub_asym_r0p4_s{noise_seed}"
    resolved_config["noise_realization"] = {
        "noise_type": "asymmetric",
        "noise_strategy": "adjacent_cyclic",
        "noise_rate": 0.4,
        "noise_seed": int(noise_seed),
        "validation_seed": int(validation_seed),
        "noise_index": str(noise_index_path),
        "noise_index_sha256": file_sha256(noise_index_path),
    }
    write_yaml(output_dir / "resolved_config.yaml", resolved_config)
    shutil.copy2(manifest_path, output_dir / "validation_manifest.csv")
    shutil.copy2(manifest_json_path, output_dir / "validation_manifest.json")
    write_csv(output_dir / "noise_realization_audit.csv", audit_rows, AUDIT_FIELDS)

    metadata = {
        "dataset": "CUB-200-2011",
        "noise_type": "asymmetric",
        "noise_strategy": "adjacent_cyclic",
        "noise_rate": 0.4,
        "noise_seed": int(noise_seed),
        "validation_seed": int(validation_seed),
        "checkpoint_protocol": PROTOCOL_NAME,
        "source_samples": len(audit_rows),
        "source_flipped_samples": int(noise_summary["flipped_samples"]),
        "source_noise_ratio": noise_summary["flipped_samples"] / max(1, len(audit_rows)),
        "training_pool_samples": len(pool_rows),
        "training_pool_clean_samples": pool_clean,
        "training_pool_noisy_samples": pool_flipped,
        "training_pool_clean_ratio": pool_clean / max(1, len(pool_rows)),
        "validation_samples": len(validation_rows),
        "validation_source_flipped_samples": sum(int(row["is_flipped"]) for row in validation_rows),
        "class_order": noise_summary["class_order"],
        "cyclic_target_mapping": noise_summary["target_mapping"],
        "per_class": noise_summary["per_class"],
        "noise_index": str(noise_index_path),
        "noise_index_sha256": file_sha256(noise_index_path),
        "labels_sha256": file_sha256(output_dir / "labels.npy"),
        "train_paths_sha256": hash_paths(train_paths),
        "validation_manifest_sha256": file_sha256(manifest_path),
        "validation_manifest_origin": manifest_origin,
        "validation_manifest_metadata": manifest_metadata,
        "peer_realization_comparisons": peer_comparisons,
        "shared_artifact_modes": link_modes,
        "generator": "tools/build_cub_asym_noise_index.py",
        "generator_sha256": repository_file_sha256("tools/build_cub_asym_noise_index.py"),
        "git_commit": git_commit(),
        "git_worktree_dirty": git_worktree_dirty(),
    }
    write_json(output_dir / "noise_realization_audit.json", metadata)
    return metadata


def align_noise_rows(
    train_paths: list[str],
    rows: list[dict[str, str]],
    *,
    source_name: str,
) -> list[dict[str, str]]:
    train_rows = [row for row in rows if str(row.get("split", "train")).lower() == "train"]
    key_to_row: dict[str, int] = {}
    for row_index, row in enumerate(train_rows):
        for raw_path in (row.get("path", ""), row.get("abs_path", "")):
            for key in path_key_candidates(raw_path):
                existing = key_to_row.get(key)
                if existing is not None and existing != row_index:
                    raise ValueError(f"Ambiguous path key {key!r} in {source_name}.")
                key_to_row[key] = row_index

    aligned: list[dict[str, str]] = []
    used_rows: set[int] = set()
    for path in train_paths:
        matches = {key_to_row[key] for key in path_key_candidates(path) if key in key_to_row}
        if len(matches) != 1:
            raise ValueError(f"Expected one noise-index match for {path!r}, found {len(matches)}.")
        row_index = matches.pop()
        if row_index in used_rows:
            raise ValueError(f"Noise-index row {row_index} matched more than one V1 path.")
        used_rows.add(row_index)
        aligned.append(train_rows[row_index])
    if len(used_rows) != len(train_rows):
        raise ValueError(
            f"Noise index has {len(train_rows) - len(used_rows)} unmatched training rows in {source_name}."
        )
    return aligned


def validate_cyclic_noise_rows(
    rows: list[dict[str, str]],
    *,
    noise_seed: int,
    noise_rate: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Noise realization contains no training rows.")
    required = {
        "clean_label",
        "web_label",
        "is_noisy",
        "noise_target_label",
        "noise_strategy",
        "noise_ratio",
        "noise_seed",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Noise index is missing required columns: {missing}.")

    class_order = sorted({str(row["clean_label"]) for row in rows}, key=numeric_label_key)
    target_mapping = {
        label: class_order[(index + 1) % len(class_order)]
        for index, label in enumerate(class_order)
    }
    per_class: list[dict[str, Any]] = []
    flipped_samples = 0
    for label in class_order:
        class_rows = [row for row in rows if str(row["clean_label"]) == label]
        expected_flips = int(round(noise_rate * len(class_rows)))
        actual_flips = 0
        for row in class_rows:
            if int(row["noise_seed"]) != int(noise_seed):
                raise ValueError(f"Noise seed mismatch in class {label}.")
            if not np.isclose(float(row["noise_ratio"]), noise_rate):
                raise ValueError(f"Noise ratio mismatch in class {label}.")
            if row["noise_strategy"] != "adjacent_cyclic":
                raise ValueError(f"Noise strategy mismatch in class {label}.")
            if str(row["noise_target_label"]) != target_mapping[label]:
                raise ValueError(f"Cyclic target mismatch in class {label}.")
            flipped = parse_bool(row["is_noisy"])
            expected_label = target_mapping[label] if flipped else label
            if str(row["web_label"]) != expected_label:
                raise ValueError(
                    f"Noisy-label invariant failed for source sample {row.get('index', '')}."
                )
            actual_flips += int(flipped)
        if actual_flips != expected_flips:
            raise ValueError(
                f"Class {label} has {actual_flips} flips; expected round({noise_rate} * "
                f"{len(class_rows)}) = {expected_flips}."
            )
        flipped_samples += actual_flips
        per_class.append(
            {
                "clean_label": label,
                "source_samples": len(class_rows),
                "flipped_samples": actual_flips,
                "actual_noise_ratio": actual_flips / max(1, len(class_rows)),
                "cyclic_target_label": target_mapping[label],
            }
        )
    return {
        "class_order": class_order,
        "target_mapping": target_mapping,
        "per_class": per_class,
        "flipped_samples": flipped_samples,
    }


def load_and_validate_manifest(
    csv_path: Path,
    json_path: Path,
    train_paths: list[str],
    noise_rows: list[dict[str, str]],
    validation_seed: int,
    *,
    expected_validation_ratio: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = read_csv(csv_path)
    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("protocol") != PROTOCOL_NAME:
        raise ValueError(f"Validation protocol must be {PROTOCOL_NAME}.")
    if int(metadata.get("validation_seed", -1)) != int(validation_seed):
        raise ValueError("Validation seed does not match the requested realization protocol.")
    if not np.isclose(float(metadata.get("validation_ratio", -1.0)), expected_validation_ratio):
        raise ValueError(
            "Validation ratio does not match the requested realization protocol: "
            f"{metadata.get('validation_ratio')!r} != {expected_validation_ratio!r}."
        )
    if metadata.get("train_paths_sha256") != hash_paths(train_paths):
        raise ValueError("Validation manifest train-path hash does not match the base V1 input.")
    if len(rows) != len(train_paths):
        raise ValueError("Validation manifest does not cover every source-train sample.")
    for index, (manifest, path, sample) in enumerate(zip(rows, train_paths, noise_rows, strict=True)):
        if int(manifest.get("index", -1)) != index or manifest.get("path") != path:
            raise ValueError(f"Validation manifest path/index mismatch at V1 index {index}.")
        if str(manifest.get("clean_label", "")) != str(sample["clean_label"]):
            raise ValueError(f"Validation manifest clean-label mismatch at V1 index {index}.")
        if manifest.get("partition") not in {"training_pool", "validation"}:
            raise ValueError(f"Invalid validation partition at V1 index {index}.")
    training_pool_samples = sum(row["partition"] == "training_pool" for row in rows)
    validation_samples = sum(row["partition"] == "validation" for row in rows)
    expected_counts = {
        "source_train_samples": len(rows),
        "training_pool_samples": training_pool_samples,
        "validation_samples": validation_samples,
    }
    for key, expected in expected_counts.items():
        if int(metadata.get(key, -1)) != expected:
            raise ValueError(
                f"Validation manifest metadata has incompatible {key}: "
                f"{metadata.get(key)!r} != {expected!r}."
            )
    return rows, metadata


def compare_peer_realizations(
    train_paths: list[str],
    current_rows: list[dict[str, str]],
    peer_paths: list[Path],
) -> list[dict[str, Any]]:
    current_seeds = {int(row["noise_seed"]) for row in current_rows}
    if len(current_seeds) != 1:
        raise ValueError("Current noise index contains multiple noise seeds.")
    current_seed = current_seeds.pop()
    current_flipped = {
        index for index, row in enumerate(current_rows) if parse_bool(row["is_noisy"])
    }
    comparisons: list[dict[str, Any]] = []
    for peer_path in peer_paths:
        if not peer_path.is_file():
            raise FileNotFoundError(f"Peer noise index does not exist: {peer_path}")
        peer_rows = align_noise_rows(train_paths, read_csv(peer_path), source_name=str(peer_path))
        peer_seeds = {int(row["noise_seed"]) for row in peer_rows}
        if len(peer_seeds) != 1:
            raise ValueError(f"Peer noise index contains multiple noise seeds: {peer_path}")
        peer_seed = peer_seeds.pop()
        if peer_seed == current_seed:
            raise ValueError(f"Peer noise index reuses current noise seed {current_seed}: {peer_path}")
        validate_cyclic_noise_rows(peer_rows, noise_seed=peer_seed, noise_rate=0.4)
        for index, (current, peer) in enumerate(zip(current_rows, peer_rows, strict=True)):
            if str(current["clean_label"]) != str(peer["clean_label"]):
                raise ValueError(f"Peer clean-label mismatch at V1 index {index}: {peer_path}")
        peer_flipped = {
            index for index, row in enumerate(peer_rows) if parse_bool(row["is_noisy"])
        }
        if current_flipped == peer_flipped:
            raise ValueError(f"Noise realizations have identical flipped-index sets: {peer_path}")
        union = current_flipped | peer_flipped
        comparisons.append(
            {
                "peer_noise_index": str(peer_path),
                "peer_noise_seed": peer_seed,
                "peer_noise_index_sha256": file_sha256(peer_path),
                "flipped_set_equal": False,
                "flipped_set_jaccard": len(current_flipped & peer_flipped) / max(1, len(union)),
                "symmetric_difference_samples": len(current_flipped ^ peer_flipped),
            }
        )
    return comparisons


def hardlink_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def load_resolved_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def numeric_label_key(value: str) -> tuple[int, int | str]:
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_file_sha256(relative_path: str) -> str:
    path = Path(__file__).resolve().parents[1] / relative_path
    return file_sha256(path) if path.is_file() else ""


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_worktree_dirty() -> bool | str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
