"""Fixed class-derangement noise artifacts for controlled asym40 experiments.

This module changes only the target class assigned to samples that were
already flipped by a saved cyclic-noise realization.  It never samples a new
flip mask and has no dependency on any training implementation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .noise_realization import (
    align_noise_rows,
    load_and_validate_manifest,
    parse_bool,
    validate_cyclic_noise_rows,
)


MAPPING_TYPE = "fixed_random_derangement"
MAPPING_ALGORITHM = "numpy_pcg64_permutation_rejection_v1"
NOISE_STRATEGY = "fixed_random_derangement"

MANIFEST_EXTRA_FIELDS = [
    "source_train_index",
    "original_clean_label",
    "flipped",
    "original_cyclic_target",
    "random_derangement_target",
    "final_noisy_observed_label",
    "mapping_type",
    "mapping_seed",
    "mapping_sha256",
    "validation_seed",
    "original_noise_index_sha256",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_file_sha256(path: Path) -> str:
    """Hash a mapping JSON with platform-independent CRLF normalization.

    The mappings were first frozen on Windows, so their published hashes are
    hashes of the CRLF bytes.  Git may check the same JSON out with LF on
    Linux.  Normalizing only line endings preserves the original published
    hashes while all JSON content remains covered by the digest.
    """

    data = path.read_bytes()
    lf_data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    canonical_data = lf_data.replace(b"\n", b"\r\n")
    return hashlib.sha256(canonical_data).hexdigest()


def generate_derangement(class_order: Iterable[str], seed: int) -> dict[str, str]:
    """Generate one deterministic bijection with no fixed points."""

    labels = [str(label) for label in class_order]
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("A derangement requires at least two unique canonical classes.")
    rng = np.random.default_rng(int(seed))
    positions = np.arange(len(labels), dtype=np.int64)
    for _ in range(100_000):
        permutation = rng.permutation(len(labels))
        if np.all(permutation != positions):
            return {source: labels[int(permutation[index])] for index, source in enumerate(labels)}
    raise RuntimeError("Failed to generate a derangement after 100000 deterministic attempts.")


def validate_derangement(mapping: dict[str, str], class_order: Iterable[str]) -> None:
    labels = [str(label) for label in class_order]
    if list(mapping) != labels:
        raise ValueError("Mapping keys do not preserve the canonical class order.")
    targets = [str(mapping[label]) for label in labels]
    if set(targets) != set(labels) or len(set(targets)) != len(labels):
        raise ValueError("Random class-transition mapping is not a bijection on the class universe.")
    fixed = [label for label in labels if str(mapping[label]) == label]
    if fixed:
        raise ValueError(f"Random class-transition mapping contains self-loops: {fixed[:5]}")


def prepare_mapping(
    path: Path,
    *,
    dataset: str,
    class_order: Iterable[str],
    mapping_seed: int,
) -> tuple[dict[str, str], str]:
    """Create a mapping once, or strictly validate and reuse the frozen file."""

    labels = [str(label) for label in class_order]
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() != sidecar.exists():
        raise FileNotFoundError(
            f"Mapping JSON and SHA-256 sidecar must either both exist or both be absent: {path}"
        )
    if path.exists():
        payload = read_json(path)
        _validate_mapping_payload(payload, dataset=dataset, class_order=labels, mapping_seed=mapping_seed)
        digest = mapping_file_sha256(path)
        recorded = sidecar.read_text(encoding="ascii").strip().split()[0]
        if recorded != digest:
            raise ValueError(f"Frozen mapping SHA-256 mismatch for {path}: {recorded} != {digest}")
        mapping = {str(key): str(value) for key, value in payload["mapping"].items()}
        return mapping, digest

    mapping = generate_derangement(labels, mapping_seed)
    validate_derangement(mapping, labels)
    payload = {
        "dataset": dataset,
        "mapping_type": MAPPING_TYPE,
        "mapping_seed": int(mapping_seed),
        "mapping_algorithm": MAPPING_ALGORITHM,
        "num_classes": len(labels),
        "canonical_class_order": labels,
        "mapping": mapping,
        "no_self_loop": True,
        "bijective": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_exclusive(path, payload)
    digest = mapping_file_sha256(path)
    _write_text_exclusive(sidecar, f"{digest}  {path.name}\n", encoding="ascii")
    return mapping, digest


def prepare_noise_manifest(
    path: Path,
    *,
    dataset: str,
    original_noise_index: Path,
    train_paths: list[str],
    mapping: dict[str, str],
    mapping_file: Path,
    mapping_sha256: str,
    noise_seed: int,
    mapping_seed: int,
    validation_seed: int,
    noise_rate: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Persist source-level labels using the original saved flip identities."""

    original_rows, original_fields = read_csv_with_fields(original_noise_index)
    aligned_original = align_noise_rows(
        train_paths,
        original_rows,
        source_name=str(original_noise_index),
    )
    cyclic_summary = validate_cyclic_noise_rows(
        aligned_original,
        noise_seed=int(noise_seed),
        noise_rate=float(noise_rate),
    )
    validate_derangement(mapping, cyclic_summary["class_order"])
    original_sha256 = file_sha256(original_noise_index)
    expected_rows = remap_aligned_noise_rows(
        aligned_original,
        mapping=mapping,
        mapping_sha256=mapping_sha256,
        mapping_seed=mapping_seed,
        validation_seed=validation_seed,
        original_noise_index_sha256=original_sha256,
    )
    fields = list(original_fields)
    for field in MANIFEST_EXTRA_FIELDS:
        if field not in fields:
            fields.append(field)
    metadata_path = path.with_suffix(".json")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    present = [path.exists(), metadata_path.exists(), sidecar.exists()]
    if any(present) and not all(present):
        raise FileNotFoundError(
            f"Noise manifest CSV, metadata JSON, and SHA-256 sidecar are only partially present: {path}"
        )

    if path.exists():
        actual_rows, actual_fields = read_csv_with_fields(path)
        if actual_fields != fields or actual_rows != expected_rows:
            raise ValueError(f"Frozen alternative noisy-label manifest differs from the expected remapping: {path}")
        digest = file_sha256(path)
        recorded = sidecar.read_text(encoding="ascii").strip().split()[0]
        if recorded != digest:
            raise ValueError(f"Frozen noisy-label manifest SHA-256 mismatch for {path}.")
        metadata = read_json(metadata_path)
        _validate_manifest_metadata(
            metadata,
            dataset=dataset,
            mapping_file=mapping_file,
            mapping_sha256=mapping_sha256,
            manifest_sha256=digest,
            original_noise_index=original_noise_index,
            original_noise_index_sha256=original_sha256,
            source_samples=len(expected_rows),
            flipped_samples=int(cyclic_summary["flipped_samples"]),
            noise_seed=noise_seed,
            mapping_seed=mapping_seed,
            validation_seed=validation_seed,
            noise_rate=noise_rate,
        )
        return actual_rows, metadata

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv_exclusive(path, expected_rows, fields)
    digest = file_sha256(path)
    _write_text_exclusive(sidecar, f"{digest}  {path.name}\n", encoding="ascii")
    metadata = {
        "dataset": dataset,
        "mapping_type": MAPPING_TYPE,
        "mapping_seed": int(mapping_seed),
        "mapping_file": str(mapping_file),
        "mapping_sha256": mapping_sha256,
        "noise_rate": float(noise_rate),
        "noise_seed": int(noise_seed),
        "validation_seed": int(validation_seed),
        "source_scope": "source_train_before_validation_exclusion",
        "source_samples": len(expected_rows),
        "flipped_samples": int(cyclic_summary["flipped_samples"]),
        "flip_identity_source": str(original_noise_index),
        "original_noise_index": str(original_noise_index),
        "original_noise_index_sha256": original_sha256,
        "manifest_sha256": digest,
        "canonical_class_order": cyclic_summary["class_order"],
        "original_cyclic_mapping": cyclic_summary["target_mapping"],
        "flipped_sample_identity_policy": "exact_reuse_from_original_is_noisy",
    }
    _write_json_exclusive(metadata_path, metadata)
    return expected_rows, metadata


def remap_aligned_noise_rows(
    original_rows: list[dict[str, str]],
    *,
    mapping: dict[str, str],
    mapping_sha256: str,
    mapping_seed: int,
    validation_seed: int,
    original_noise_index_sha256: str,
) -> list[dict[str, str]]:
    """Apply a class mapping only where the saved cyclic row says flipped."""

    remapped: list[dict[str, str]] = []
    for source_index, original in enumerate(original_rows):
        clean_label = str(original["clean_label"])
        flipped = parse_bool(original["is_noisy"])
        random_target = str(mapping[clean_label])
        final_label = random_target if flipped else clean_label
        row = {str(key): str(value) for key, value in original.items()}
        row.update(
            {
                "web_label": final_label,
                "noise_target_label": random_target,
                "noise_strategy": NOISE_STRATEGY,
                "source_train_index": str(source_index),
                "original_clean_label": clean_label,
                "flipped": "true" if flipped else "false",
                "original_cyclic_target": str(original["noise_target_label"]),
                "random_derangement_target": random_target,
                "final_noisy_observed_label": final_label,
                "mapping_type": MAPPING_TYPE,
                "mapping_seed": str(int(mapping_seed)),
                "mapping_sha256": mapping_sha256,
                "validation_seed": str(int(validation_seed)),
                "original_noise_index_sha256": original_noise_index_sha256,
            }
        )
        remapped.append(row)
    return remapped


def audit_noise_control(
    *,
    dataset: str,
    train_paths: list[str],
    original_noise_index: Path,
    alternative_manifest: Path,
    validation_source_dir: Path,
    validation_destination_dir: Path,
    mapping: dict[str, str],
    mapping_file: Path,
    mapping_sha256: str,
    noise_seed: int,
    mapping_seed: int,
    validation_seed: int,
    noise_rate: float,
    expected_pool: tuple[int, int, int],
    peer_validation_dirs: Iterable[Path] = (),
) -> dict[str, Any]:
    """Run mandatory source-mask, validation, pool, and mapping parity checks."""

    original_rows = align_noise_rows(
        train_paths,
        read_csv_with_fields(original_noise_index)[0],
        source_name=str(original_noise_index),
    )
    alternative_rows = align_noise_rows(
        train_paths,
        read_csv_with_fields(alternative_manifest)[0],
        source_name=str(alternative_manifest),
    )
    cyclic_summary = validate_cyclic_noise_rows(
        original_rows,
        noise_seed=int(noise_seed),
        noise_rate=float(noise_rate),
    )
    validate_derangement(mapping, cyclic_summary["class_order"])
    if mapping_file_sha256(mapping_file) != mapping_sha256:
        raise ValueError(f"Mapping JSON changed after it was loaded: {mapping_file}")

    original_mask = np.asarray([parse_bool(row["is_noisy"]) for row in original_rows], dtype=bool)
    recovered_mask = np.asarray(
        [str(row["web_label"]) != str(row["clean_label"]) for row in original_rows],
        dtype=bool,
    )
    new_mask = np.asarray(
        [str(row["web_label"]) != str(row["clean_label"]) for row in alternative_rows],
        dtype=bool,
    )
    if not np.array_equal(original_mask, recovered_mask):
        raise ValueError(f"{dataset}: original is_noisy does not match labels recovered from cyclic artifact.")
    flip_mismatch_count = int(np.sum(original_mask != new_mask))
    if flip_mismatch_count:
        raise ValueError(f"{dataset}: alternative mapping changed {flip_mismatch_count} flip identities.")
    for index, (original, alternative) in enumerate(zip(original_rows, alternative_rows, strict=True)):
        if str(original["clean_label"]) != str(alternative["clean_label"]):
            raise ValueError(f"{dataset}: clean-label mismatch at source index {index}.")
        expected = mapping[str(original["clean_label"])] if original_mask[index] else str(original["clean_label"])
        if str(alternative["web_label"]) != expected:
            raise ValueError(f"{dataset}: alternative observed-label mismatch at source index {index}.")

    source_csv = validation_source_dir / "validation_manifest.csv"
    source_json = validation_source_dir / "validation_manifest.json"
    destination_csv = validation_destination_dir / "validation_manifest.csv"
    destination_json = validation_destination_dir / "validation_manifest.json"
    _copy_or_validate_identical(source_csv, destination_csv)
    _copy_or_validate_identical(source_json, destination_json)
    source_hash = file_sha256(source_csv)
    source_json_hash = file_sha256(source_json)
    if file_sha256(destination_csv) != source_hash or file_sha256(destination_json) != source_json_hash:
        raise ValueError(f"{dataset}: copied validation manifest is not byte-identical to the formal manifest.")
    for peer_dir in peer_validation_dirs:
        for name, expected_hash in (
            ("validation_manifest.csv", source_hash),
            ("validation_manifest.json", source_json_hash),
        ):
            peer_path = peer_dir / name
            if not peer_path.is_file() or file_sha256(peer_path) != expected_hash:
                raise ValueError(f"{dataset}: formal validation manifest differs at {peer_path}.")

    manifest_rows, manifest_metadata = load_and_validate_manifest(
        source_csv,
        source_json,
        train_paths,
        original_rows,
        validation_seed=int(validation_seed),
        expected_validation_ratio=0.10,
    )
    destination_rows, _ = load_and_validate_manifest(
        destination_csv,
        destination_json,
        train_paths,
        alternative_rows,
        validation_seed=int(validation_seed),
        expected_validation_ratio=0.10,
    )
    source_validation_indices = [
        index for index, row in enumerate(manifest_rows) if row["partition"] == "validation"
    ]
    destination_validation_indices = [
        index for index, row in enumerate(destination_rows) if row["partition"] == "validation"
    ]
    validation_mismatch_count = len(
        set(source_validation_indices).symmetric_difference(destination_validation_indices)
    )
    if validation_mismatch_count:
        raise ValueError(f"{dataset}: validation sample identities changed.")

    pool_indices = [index for index, row in enumerate(manifest_rows) if row["partition"] == "training_pool"]
    pool_noisy = int(original_mask[pool_indices].sum())
    pool_clean = len(pool_indices) - pool_noisy
    expected_total, expected_clean, expected_noisy = expected_pool
    if (len(pool_indices), pool_clean, pool_noisy) != expected_pool:
        raise ValueError(
            f"{dataset}: training-pool composition {(len(pool_indices), pool_clean, pool_noisy)} "
            f"!= expected {expected_pool}."
        )

    return {
        "dataset": dataset,
        "status": "PASS",
        "noise_rate": float(noise_rate),
        "noise_seed": int(noise_seed),
        "mapping_seed": int(mapping_seed),
        "mapping_type": MAPPING_TYPE,
        "mapping_file": str(mapping_file),
        "mapping_sha256": mapping_sha256,
        "num_classes": len(cyclic_summary["class_order"]),
        "num_source_classes": len(mapping),
        "num_target_classes": len(set(mapping.values())),
        "mapping_no_self_loop": all(source != target for source, target in mapping.items()),
        "mapping_bijective": len(set(mapping.values())) == len(mapping),
        "source_train_size": len(train_paths),
        "flipped_count": int(original_mask.sum()),
        "flip_mask_mismatch_count": flip_mismatch_count,
        "validation_seed": int(validation_seed),
        "validation_count": len(source_validation_indices),
        "validation_manifest": str(source_csv),
        "validation_manifest_sha256": source_hash,
        "validation_manifest_json_sha256": source_json_hash,
        "validation_manifest_mismatch_count": validation_mismatch_count,
        "training_pool": expected_total,
        "training_pool_clean": expected_clean,
        "training_pool_noisy": expected_noisy,
        "training_pool_clean_ratio": expected_clean / expected_total,
        "validation_manifest_metadata": manifest_metadata,
    }


def prepare_input_adapter(
    *,
    base_input_dir: Path,
    adapter_dir: Path,
    alternative_manifest: Path,
    aligned_alternative_rows: list[dict[str, str]],
    dataset: str,
    experiment_tag: str,
    mapping_file: Path,
    mapping_sha256: str,
    mapping_seed: int,
    noise_seed: int,
    validation_seed: int,
) -> dict[str, Any]:
    """Build a label-only V1 adapter while sharing immutable feature files."""

    required_shared = [
        "paths.txt",
        "eval_paths.txt",
        "eval_labels.npy",
        "features_cls.npy",
        "features_gap.npy",
        "features_top.npy",
    ]
    adapter_dir.mkdir(parents=True, exist_ok=True)
    link_modes: dict[str, str] = {}
    for name in required_shared:
        source = base_input_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"Base formal V1 input is missing: {source}")
        link_modes[name] = _ensure_shared_file(source, adapter_dir / name)
    for name in ("eval_features_cls.npy", "eval_features_gap.npy", "eval_features_top.npy"):
        source = base_input_dir / name
        if source.is_file():
            link_modes[name] = _ensure_shared_file(source, adapter_dir / name)

    labels = np.asarray([str(row["web_label"]) for row in aligned_alternative_rows], dtype=str)
    labels_path = adapter_dir / "labels.npy"
    if labels_path.exists():
        existing = np.load(labels_path, allow_pickle=True).astype(str)
        if not np.array_equal(existing, labels):
            raise ValueError(f"Prepared alternative labels changed: {labels_path}")
    else:
        np.save(labels_path, labels)

    base_config_path = base_input_dir / "resolved_config.yaml"
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("dataset", {})
    config["dataset"]["name"] = f"{dataset}-{experiment_tag}"
    config["dataset"]["index_file"] = str(alternative_manifest)
    config.setdefault("output", {})
    config["output"]["version"] = experiment_tag
    config["noise_realization"] = {
        "noise_type": "asymmetric",
        "noise_strategy": NOISE_STRATEGY,
        "noise_rate": 0.4,
        "noise_seed": int(noise_seed),
        "mapping_type": MAPPING_TYPE,
        "mapping_seed": int(mapping_seed),
        "mapping_file": str(mapping_file),
        "mapping_sha256": mapping_sha256,
        "validation_seed": int(validation_seed),
        "flip_identity_policy": "exact_reuse_from_formal_cyclic_asym40_noise_seed42",
    }
    config_path = adapter_dir / "resolved_config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            existing_config = yaml.safe_load(handle) or {}
        if existing_config != config:
            raise ValueError(f"Prepared input config changed: {config_path}")
    else:
        with config_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    metadata = {
        "dataset": dataset,
        "experiment_tag": experiment_tag,
        "base_input_dir": str(base_input_dir),
        "alternative_manifest": str(alternative_manifest),
        "alternative_manifest_sha256": file_sha256(alternative_manifest),
        "mapping_file": str(mapping_file),
        "mapping_sha256": mapping_sha256,
        "mapping_seed": int(mapping_seed),
        "noise_seed": int(noise_seed),
        "validation_seed": int(validation_seed),
        "source_samples": len(labels),
        "labels_sha256": file_sha256(labels_path),
        "shared_artifact_modes": link_modes,
    }
    metadata_path = adapter_dir / "adapter_metadata.json"
    if metadata_path.exists():
        if read_json(metadata_path) != metadata:
            raise ValueError(f"Prepared input metadata changed: {metadata_path}")
    else:
        _write_json_exclusive(metadata_path, metadata)
    return metadata


def read_csv_with_fields(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return [dict(row) for row in reader], fields


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_mapping_payload(
    payload: dict[str, Any],
    *,
    dataset: str,
    class_order: list[str],
    mapping_seed: int,
) -> None:
    expected = {
        "dataset": dataset,
        "mapping_type": MAPPING_TYPE,
        "mapping_seed": int(mapping_seed),
        "mapping_algorithm": MAPPING_ALGORITHM,
        "num_classes": len(class_order),
        "canonical_class_order": class_order,
        "no_self_loop": True,
        "bijective": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Frozen mapping has incompatible {key}: {payload.get(key)!r} != {value!r}")
    raw_mapping = payload.get("mapping")
    if not isinstance(raw_mapping, dict):
        raise ValueError("Frozen mapping JSON does not contain an object-valued mapping.")
    mapping = {str(key): str(value) for key, value in raw_mapping.items()}
    validate_derangement(mapping, class_order)


def _validate_manifest_metadata(
    metadata: dict[str, Any],
    *,
    dataset: str,
    mapping_file: Path,
    mapping_sha256: str,
    manifest_sha256: str,
    original_noise_index: Path,
    original_noise_index_sha256: str,
    source_samples: int,
    flipped_samples: int,
    noise_seed: int,
    mapping_seed: int,
    validation_seed: int,
    noise_rate: float,
) -> None:
    expected = {
        "dataset": dataset,
        "mapping_type": MAPPING_TYPE,
        "mapping_seed": int(mapping_seed),
        "mapping_file": str(mapping_file),
        "mapping_sha256": mapping_sha256,
        "noise_rate": float(noise_rate),
        "noise_seed": int(noise_seed),
        "validation_seed": int(validation_seed),
        "source_samples": int(source_samples),
        "flipped_samples": int(flipped_samples),
        "original_noise_index": str(original_noise_index),
        "original_noise_index_sha256": original_noise_index_sha256,
        "manifest_sha256": manifest_sha256,
        "source_scope": "source_train_before_validation_exclusion",
        "flipped_sample_identity_policy": "exact_reuse_from_original_is_noisy",
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError(f"Frozen noisy-label manifest metadata mismatches: {mismatches}")


def _copy_or_validate_identical(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required formal artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(destination) != file_sha256(source):
            raise ValueError(f"Existing destination is not identical to formal artifact: {destination}")
        return
    shutil.copy2(source, destination)


def _ensure_shared_file(source: Path, destination: Path) -> str:
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return "hardlink"
        except OSError:
            pass
        if source.stat().st_size == destination.stat().st_size and file_sha256(source) == file_sha256(destination):
            return "copy"
        raise ValueError(f"Prepared shared artifact differs from its formal source: {destination}")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")


def _write_csv_exclusive(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text_exclusive(path: Path, text: str, *, encoding: str) -> None:
    with path.open("x", encoding=encoding) as handle:
        handle.write(text)
