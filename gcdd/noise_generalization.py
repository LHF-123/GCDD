"""Frozen IDN and GCCN realization helpers for PGDF noise-generalization runs.

The module intentionally operates before model training.  It reads clean source-train
labels only to construct a saved synthetic-noise artifact, then writes an isolated
label-only V1 adapter.  PGDF and every training baseline receive only the adapter's
observed ``labels.npy`` values; clean labels are used solely by the fixed validation
protocol already used by the existing synthetic-noise experiments.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from .io_utils import read_csv, write_csv, write_json, write_yaml
from .noise_realization import (
    align_noise_rows,
    file_sha256,
    hardlink_or_copy,
    load_and_validate_manifest,
    numeric_label_key,
    repository_file_sha256,
)


ProtocolName = Literal["idn", "gccn"]

GENERATOR_VERSION = "noise_generalization_v1"
IDN_STRATEGY = "instance_dependent_dinov2_cls_v1"
IDN_MAPPING_TYPE = "feature_conditioned_nearest_prototype_v1"
GCCN_STRATEGY = "general_class_conditional_multi_offset_v1"
GCCN_MAPPING_TYPE = "cyclic_offsets_1_2_3_weights_5_3_2_v1"
GCCN_OFFSETS = (1, 2, 3)
GCCN_TARGET_PROBABILITIES = (0.5, 0.3, 0.2)


def prepare_noise_generalization(
    *,
    protocol: ProtocolName,
    dataset: str,
    base_input_dir: Path,
    source_noise_index: Path,
    validation_dir: Path,
    output_dir: Path,
    noise_rate: float = 0.4,
    noise_seed: int = 42,
    validation_seed: int = 20250726,
    idn_probability_modulation: float = 0.5,
) -> dict[str, Any]:
    """Create one immutable IDN/GCCN realization and a compatible V1 adapter.

    The saved realization covers the original source train set before the fixed
    validation exclusion, matching the established cyclic-noise protocol.  The
    existing clean validation manifest is copied byte-for-byte into ``protocol``;
    its samples are never candidates in the later training/selection runs.
    """

    if protocol not in {"idn", "gccn"}:
        raise ValueError(f"Unsupported noise-generalization protocol: {protocol!r}.")
    _validate_noise_parameters(
        noise_rate=noise_rate,
        noise_seed=noise_seed,
        validation_seed=validation_seed,
        idn_probability_modulation=idn_probability_modulation,
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing noise-generalization output: {output_dir}")

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
    if not source_noise_index.is_file():
        raise FileNotFoundError(f"Source synthetic-noise index is missing: {source_noise_index}")

    train_paths = (base_input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
    source_rows = align_noise_rows(
        train_paths,
        read_csv(source_noise_index),
        source_name=str(source_noise_index),
    )
    _validate_source_rows(source_rows, expected_count=len(train_paths))
    validation_csv = validation_dir / "validation_manifest.csv"
    validation_json = validation_dir / "validation_manifest.json"
    if validation_csv.is_file() != validation_json.is_file():
        raise FileNotFoundError(
            f"Fixed validation manifest CSV/JSON is only partially present under {validation_dir}."
        )
    if not validation_csv.is_file():
        raise FileNotFoundError(
            f"The established fixed validation manifest is required under {validation_dir}; "
            "this generator will not create a new split."
        )
    _, validation_metadata = load_and_validate_manifest(
        validation_csv,
        validation_json,
        train_paths,
        source_rows,
        validation_seed=int(validation_seed),
        expected_validation_ratio=0.10,
    )

    if protocol == "idn":
        features = np.load(base_input_dir / "features_cls.npy", mmap_mode="r")
        generated_rows, protocol_details = generate_instance_dependent_noise_rows(
            source_rows,
            np.asarray(features, dtype=np.float32),
            noise_rate=float(noise_rate),
            noise_seed=int(noise_seed),
            probability_modulation=float(idn_probability_modulation),
        )
        matrix_rows: list[dict[str, Any]] = []
        matrix_payload: dict[str, Any] | None = None
    else:
        generated_rows, matrix_rows, matrix_payload = generate_general_class_conditional_noise_rows(
            source_rows,
            noise_rate=float(noise_rate),
            noise_seed=int(noise_seed),
        )
        protocol_details = matrix_payload

    _validate_generated_rows(
        generated_rows,
        expected_paths=train_paths,
        protocol=protocol,
        noise_rate=float(noise_rate),
        noise_seed=int(noise_seed),
    )

    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "noise_manifest.csv"
    manifest_fields = manifest_fieldnames(generated_rows)
    write_csv(manifest_path, generated_rows, manifest_fields)
    manifest_sha256 = file_sha256(manifest_path)
    (output_dir / "noise_manifest.csv.sha256").write_text(
        f"{manifest_sha256}  {manifest_path.name}\n", encoding="ascii"
    )

    if matrix_payload is not None:
        matrix_csv = output_dir / "transition_matrix.csv"
        write_csv(matrix_csv, matrix_rows, transition_matrix_fieldnames())
        matrix_payload = dict(matrix_payload)
        matrix_payload["matrix_csv"] = str(matrix_csv)
        matrix_payload["matrix_csv_sha256"] = file_sha256(matrix_csv)
        write_json(output_dir / "transition_matrix.json", matrix_payload)

    protocol_dir = output_dir / "protocol"
    protocol_dir.mkdir()
    _copy_identical(validation_csv, protocol_dir / validation_csv.name)
    _copy_identical(validation_json, protocol_dir / validation_json.name)
    adapter_metadata = prepare_label_only_v1_adapter(
        base_input_dir=base_input_dir,
        adapter_dir=output_dir / "prepared_input",
        manifest_path=manifest_path,
        rows=generated_rows,
        dataset=dataset,
        protocol=protocol,
        noise_rate=float(noise_rate),
        noise_seed=int(noise_seed),
        validation_seed=int(validation_seed),
        protocol_details=protocol_details,
    )
    statistics = build_noise_statistics(
        rows=generated_rows,
        dataset=dataset,
        protocol=protocol,
        noise_rate=float(noise_rate),
        noise_seed=int(noise_seed),
        validation_seed=int(validation_seed),
        manifest_path=manifest_path,
        source_noise_index=source_noise_index,
        validation_metadata=validation_metadata,
        protocol_details=protocol_details,
    )
    statistics["prepared_input"] = str(output_dir / "prepared_input")
    statistics["adapter_metadata"] = adapter_metadata
    statistics["validation_protocol_dir"] = str(protocol_dir)
    statistics["generator"] = "tools/prepare_noise_generalization.py"
    statistics["generator_sha256"] = repository_file_sha256("tools/prepare_noise_generalization.py")
    statistics["generator_version"] = GENERATOR_VERSION
    write_json(output_dir / "noise_statistics.json", statistics)
    write_csv(
        output_dir / "noise_statistics.csv",
        statistics["per_class"],
        per_class_statistics_fieldnames(),
    )
    write_csv(
        output_dir / "per_class_transition_counts.csv",
        statistics["per_class_transition_counts"],
        transition_count_fieldnames(),
    )
    return statistics


def generate_instance_dependent_noise_rows(
    source_rows: list[dict[str, str]],
    features_cls: np.ndarray,
    *,
    noise_rate: float,
    noise_seed: int,
    probability_modulation: float = 0.5,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Generate Bernoulli IDN from frozen CLS geometry without test information.

    For each clean source class, a sample's difficulty is one minus the cosine
    similarity to its clean-class prototype.  Its deterministic within-class
    difficulty percentile is converted to a pre-declared probability
    ``rate * (1 + modulation * (2 * percentile - 1))``.  This preserves the
    requested rate in expectation within every class while making label flips
    sample-dependent.  A PCG64 generator seeded by ``noise_seed`` performs the
    independent Bernoulli draw.  A flipped sample receives the closest *other*
    clean-class prototype label, also determined only from frozen source-train
    features and source-train clean labels.
    """

    _validate_noise_parameters(
        noise_rate=noise_rate,
        noise_seed=noise_seed,
        validation_seed=0,
        idn_probability_modulation=probability_modulation,
    )
    _validate_source_rows(source_rows, expected_count=len(source_rows))
    features = np.asarray(features_cls, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] != len(source_rows):
        raise ValueError(
            "features_cls must be a two-dimensional array aligned one-to-one with source rows."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("features_cls contains non-finite values.")
    feature_norm = _row_normalize(features)
    labels = np.asarray([str(row["clean_label"]) for row in source_rows], dtype=str)
    class_order = canonical_class_order(labels.tolist())
    if len(class_order) < 2:
        raise ValueError("IDN requires at least two clean classes.")
    class_to_position = {label: position for position, label in enumerate(class_order)}
    prototypes = np.empty((len(class_order), feature_norm.shape[1]), dtype=np.float64)
    for position, label in enumerate(class_order):
        class_indices = np.flatnonzero(labels == label)
        prototypes[position] = _row_normalize(feature_norm[class_indices].mean(axis=0, keepdims=True))[0]

    source_positions = np.asarray([class_to_position[label] for label in labels], dtype=np.int64)
    source_similarity = np.einsum("ij,ij->i", feature_norm, prototypes[source_positions])
    distance = 1.0 - np.clip(source_similarity, -1.0, 1.0)
    percentile = np.empty(len(source_rows), dtype=np.float64)
    for label in class_order:
        indices = np.flatnonzero(labels == label)
        percentile[indices] = stable_percentile(distance[indices], indices)
    probability = noise_rate * (1.0 + probability_modulation * (2.0 * percentile - 1.0))
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("IDN probability calibration left the [0, 1] interval.")

    similarity_to_all = feature_norm @ prototypes.T
    similarity_to_all[np.arange(len(source_rows)), source_positions] = -np.inf
    target_positions = np.argmax(similarity_to_all, axis=1)
    target_similarity = similarity_to_all[np.arange(len(source_rows)), target_positions]
    target_labels = np.asarray([class_order[position] for position in target_positions], dtype=str)
    rng = np.random.default_rng(int(noise_seed))
    flipped = rng.random(len(source_rows)) < probability
    rows = []
    for source_index, original in enumerate(source_rows):
        clean_label = str(original["clean_label"])
        target_label = str(target_labels[source_index])
        row = _base_generated_row(
            original,
            source_train_index=source_index,
            noisy_label=target_label if flipped[source_index] else clean_label,
            target_label=target_label,
            flipped=bool(flipped[source_index]),
            noise_type="instance_dependent",
            noise_strategy=IDN_STRATEGY,
            noise_rate=noise_rate,
            noise_seed=noise_seed,
            mapping_type=IDN_MAPPING_TYPE,
            mapping_seed=noise_seed,
        )
        row.update(
            {
                "idn_feature_distance": f"{distance[source_index]:.12f}",
                "idn_difficulty_percentile": f"{percentile[source_index]:.12f}",
                "idn_flip_probability": f"{probability[source_index]:.12f}",
                "idn_target_prototype_cosine": f"{target_similarity[source_index]:.12f}",
            }
        )
        rows.append(row)
    details = {
        "protocol_definition": "per-class frozen-DINOv2-CLS prototype distance rank with calibrated Bernoulli flip probability",
        "feature_source": "base_input_dir/features_cls.npy",
        "feature_geometry": "L2-normalized CLS cosine similarity",
        "clean_label_usage": "source-train only during pre-training noise construction; never passed to PGDF/training selection",
        "probability_formula": "noise_rate * (1 + probability_modulation * (2 * within_class_difficulty_percentile - 1))",
        "probability_modulation": float(probability_modulation),
        "target_rule": "closest non-source clean-class prototype by cosine similarity; canonical-label tie break",
        "class_order": class_order,
        "feature_dimension": int(features.shape[1]),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
        "distance_probability_correlation": _safe_correlation(distance, probability),
    }
    return rows, details


def generate_general_class_conditional_noise_rows(
    source_rows: list[dict[str, str]],
    *,
    noise_rate: float,
    noise_seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, Any]]:
    """Generate GCCN with a frozen three-target transition distribution.

    Given the canonical clean-class order, a source class transitions to the
    classes at circular offsets +1, +2, and +3 with conditional probabilities
    0.5, 0.3, and 0.2 once selected for corruption.  The requested noise rate
    is applied per source class with the same round(n * rate) convention as the
    established cyclic implementation.  Largest-remainder allocation fixes the
    per-source target counts, and ``noise_seed`` only shuffles affected samples
    and their pre-allocated targets.  Thus every formal class with enough flips
    realizes multiple target labels rather than a one-to-one remapping.
    """

    _validate_noise_parameters(
        noise_rate=noise_rate,
        noise_seed=noise_seed,
        validation_seed=0,
        idn_probability_modulation=0.5,
    )
    _validate_source_rows(source_rows, expected_count=len(source_rows))
    labels = np.asarray([str(row["clean_label"]) for row in source_rows], dtype=str)
    class_order = canonical_class_order(labels.tolist())
    if len(class_order) < len(GCCN_OFFSETS) + 1:
        raise ValueError("GCCN requires at least four classes for three non-self targets per source.")
    class_to_position = {label: position for position, label in enumerate(class_order)}
    targets_by_label = {
        label: [class_order[(position + offset) % len(class_order)] for offset in GCCN_OFFSETS]
        for label, position in class_to_position.items()
    }
    rng = np.random.default_rng(int(noise_seed))
    noisy_labels = labels.copy()
    selected_target = np.asarray([targets_by_label[label][0] for label in labels], dtype=str)
    flipped = np.zeros(len(source_rows), dtype=bool)
    for label in class_order:
        indices = np.flatnonzero(labels == label)
        selected = rng.permutation(indices)[: int(round(len(indices) * noise_rate))]
        target_counts = largest_remainder_counts(len(selected), GCCN_TARGET_PROBABILITIES)
        target_values = np.concatenate(
            [np.repeat(target, count) for target, count in zip(targets_by_label[label], target_counts, strict=True)]
        )
        if len(target_values):
            target_values = target_values[rng.permutation(len(target_values))]
        flipped[selected] = True
        noisy_labels[selected] = target_values
        selected_target[indices] = targets_by_label[label][0]
        selected_target[selected] = target_values

    rows = []
    for source_index, original in enumerate(source_rows):
        clean_label = str(original["clean_label"])
        row = _base_generated_row(
            original,
            source_train_index=source_index,
            noisy_label=str(noisy_labels[source_index]),
            target_label=str(selected_target[source_index]),
            flipped=bool(flipped[source_index]),
            noise_type="general_class_conditional",
            noise_strategy=GCCN_STRATEGY,
            noise_rate=noise_rate,
            noise_seed=noise_seed,
            mapping_type=GCCN_MAPPING_TYPE,
            mapping_seed=noise_seed,
        )
        rows.append(row)

    matrix_rows: list[dict[str, Any]] = []
    for source_label in class_order:
        matrix_rows.append(
            {
                "source_label": source_label,
                "target_label": source_label,
                "transition_probability": f"{1.0 - noise_rate:.12f}",
                "probability_given_flip": "0.000000000000",
                "transition_kind": "clean_diagonal",
            }
        )
        for target_label, conditional_probability in zip(
            targets_by_label[source_label], GCCN_TARGET_PROBABILITIES, strict=True
        ):
            matrix_rows.append(
                {
                    "source_label": source_label,
                    "target_label": target_label,
                    "transition_probability": f"{noise_rate * conditional_probability:.12f}",
                    "probability_given_flip": f"{conditional_probability:.12f}",
                    "transition_kind": "noisy_multi_target",
                }
            )
    details = {
        "protocol_definition": "fixed three-target general class-conditional transition matrix",
        "mapping_type": GCCN_MAPPING_TYPE,
        "canonical_class_order": class_order,
        "target_offsets": list(GCCN_OFFSETS),
        "probability_given_flip": list(GCCN_TARGET_PROBABILITIES),
        "matrix_row_sum": 1.0,
        "per_class_flip_count_rule": "round(source_class_count * requested_noise_rate)",
        "target_count_rule": "largest_remainder(requested_flips * [0.5, 0.3, 0.2]), then PCG64 seeded shuffle",
        "uses_test_labels": False,
    }
    return rows, matrix_rows, details


def prepare_label_only_v1_adapter(
    *,
    base_input_dir: Path,
    adapter_dir: Path,
    manifest_path: Path,
    rows: list[dict[str, str]],
    dataset: str,
    protocol: ProtocolName,
    noise_rate: float,
    noise_seed: int,
    validation_seed: int,
    protocol_details: dict[str, Any],
) -> dict[str, Any]:
    """Share immutable V1 files and materialize only observed noisy labels."""

    adapter_dir.mkdir(parents=True)
    shared_names = [
        "paths.txt",
        "eval_paths.txt",
        "eval_labels.npy",
        "features_cls.npy",
        "features_gap.npy",
        "features_top.npy",
    ]
    for optional in ("eval_features_cls.npy", "eval_features_gap.npy", "eval_features_top.npy"):
        if (base_input_dir / optional).is_file():
            shared_names.append(optional)
    link_modes = {
        name: hardlink_or_copy(base_input_dir / name, adapter_dir / name)
        for name in shared_names
    }
    labels = np.asarray([str(row["web_label"]) for row in rows], dtype=str)
    np.save(adapter_dir / "labels.npy", labels)
    with (base_input_dir / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("dataset", {})
    config["dataset"]["name"] = f"{dataset}-{protocol}-r{noise_rate:g}-s{noise_seed}"
    config["dataset"]["index_file"] = str(manifest_path)
    config.setdefault("output", {})
    config["output"]["version"] = f"{protocol}_r{noise_rate:g}_s{noise_seed}"
    config["noise_realization"] = {
        "generator_version": GENERATOR_VERSION,
        "noise_type": "instance_dependent" if protocol == "idn" else "general_class_conditional",
        "noise_strategy": IDN_STRATEGY if protocol == "idn" else GCCN_STRATEGY,
        "noise_rate": float(noise_rate),
        "noise_seed": int(noise_seed),
        "mapping_type": IDN_MAPPING_TYPE if protocol == "idn" else GCCN_MAPPING_TYPE,
        "mapping_seed": int(noise_seed),
        "noise_manifest": str(manifest_path),
        "noise_manifest_sha256": file_sha256(manifest_path),
        "validation_seed": int(validation_seed),
        "clean_label_usage": "source-train construction and fixed validation only; not supplied to PGDF selection/training",
        "protocol_details": protocol_details,
    }
    write_yaml(adapter_dir / "resolved_config.yaml", config)
    metadata = {
        "base_input_dir": str(base_input_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "source_samples": len(rows),
        "labels_sha256": file_sha256(adapter_dir / "labels.npy"),
        "shared_artifact_modes": link_modes,
        "official_test_label_source": str(base_input_dir / "eval_labels.npy"),
        "official_test_label_source_sha256": file_sha256(base_input_dir / "eval_labels.npy"),
    }
    write_json(adapter_dir / "adapter_metadata.json", metadata)
    return metadata


def validate_noise_generalization_training_input(
    *,
    manifest_path: Path,
    train_paths: list[str],
    train_labels: np.ndarray,
    labels_path: Path,
) -> dict[str, Any]:
    """Fail closed unless an IDN/GCCN manifest matches its V1 observed labels.

    This guard is deliberately limited to the new noise-generalization
    protocols.  Legacy cyclic and random-derangement inputs retain their
    historical loading behavior.  Matching by path prevents an accidental
    manifest/input-directory mix-up, while ``source_train_index`` proves that
    the saved V1 row order is still the one used during noise construction.
    """

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Noise-generalization manifest is missing: {manifest_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Prepared V1 labels are missing: {labels_path}")
    labels = np.asarray(train_labels, dtype=str)
    if labels.ndim != 1 or labels.shape != (len(train_paths),):
        raise ValueError("Prepared V1 train labels must be one-dimensional and align with paths.txt.")
    stored_labels = np.load(labels_path, allow_pickle=True).astype(str)
    if stored_labels.shape != labels.shape or not np.array_equal(stored_labels, labels):
        raise ValueError("Prepared V1 labels.npy changed after input loading; refuse an inconsistent training input.")

    rows = read_csv(manifest_path)
    if not rows:
        raise ValueError("Noise-generalization manifest contains no rows.")
    non_train = [index for index, row in enumerate(rows) if str(row.get("split", "")).lower() != "train"]
    if non_train:
        raise ValueError(
            "Noise-generalization manifest must contain only source-train rows; "
            f"found non-training rows at {non_train[:5]}."
        )
    required = {"path", "source_train_index", "web_label", "noisy_label", "noise_strategy"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Noise-generalization manifest is missing required columns: {missing}.")

    aligned_rows = align_noise_rows(train_paths, rows, source_name=str(manifest_path))
    strategies = {str(row["noise_strategy"]) for row in aligned_rows}
    allowed_strategies = {IDN_STRATEGY, GCCN_STRATEGY}
    if len(strategies) != 1 or not strategies <= allowed_strategies:
        raise ValueError(
            "Prepared-input consistency validation only accepts one IDN/GCCN strategy; "
            f"found {sorted(strategies)}."
        )

    for index, row in enumerate(aligned_rows):
        try:
            source_index = int(row["source_train_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source_train_index at manifest row {index}.") from exc
        if source_index != index:
            raise ValueError(
                "Noise-generalization manifest source_train_index does not match the aligned V1 order: "
                f"row {index} declares {source_index}."
            )
        observed = str(row["web_label"])
        if observed != str(row["noisy_label"]):
            raise ValueError(f"Manifest observed-label aliases disagree at source_train_index {index}.")
        if labels[index] != observed:
            raise ValueError(
                "Prepared V1 labels.npy disagrees with the noise manifest observed label at "
                f"source_train_index {index} ({train_paths[index]!r}): "
                f"labels.npy={labels[index]!r}, manifest={observed!r}."
            )

    return {
        "status": "verified_path_and_source_index_aligned_observed_labels",
        "noise_strategy": next(iter(strategies)),
        "source_train_sample_count": len(aligned_rows),
        "manifest_sha256": file_sha256(manifest_path),
        "labels_sha256": file_sha256(labels_path),
        "alignment": "path_keys_and_source_train_index",
    }


def build_noise_statistics(
    *,
    rows: list[dict[str, str]],
    dataset: str,
    protocol: ProtocolName,
    noise_rate: float,
    noise_seed: int,
    validation_seed: int,
    manifest_path: Path,
    source_noise_index: Path,
    validation_metadata: dict[str, Any],
    protocol_details: dict[str, Any],
) -> dict[str, Any]:
    """Summarize source labels, observed labels, and transition counts."""

    class_order = canonical_class_order([str(row["clean_label"]) for row in rows])
    observed_distribution = Counter(str(row["web_label"]) for row in rows)
    per_class: list[dict[str, Any]] = []
    transition_counts: Counter[tuple[str, str]] = Counter()
    for label in class_order:
        class_rows = [row for row in rows if str(row["clean_label"]) == label]
        flipped_rows = [row for row in class_rows if row["is_noisy"] == "1"]
        for row in flipped_rows:
            transition_counts[(label, str(row["web_label"]))] += 1
        probabilities = [float(row["idn_flip_probability"]) for row in class_rows if row.get("idn_flip_probability")]
        per_class.append(
            {
                "clean_label": label,
                "source_samples": len(class_rows),
                "flipped_samples": len(flipped_rows),
                "realized_noise_ratio": len(flipped_rows) / max(1, len(class_rows)),
                "idn_probability_mean": float(np.mean(probabilities)) if probabilities else "",
                "idn_probability_min": float(np.min(probabilities)) if probabilities else "",
                "idn_probability_max": float(np.max(probabilities)) if probabilities else "",
                "observed_label_count": observed_distribution[label],
            }
        )
    transition_rows = [
        {
            "source_clean_label": source,
            "target_noisy_label": target,
            "transition_count": count,
            "source_flipped_total": sum(value for (candidate, _), value in transition_counts.items() if candidate == source),
        }
        for (source, target), count in sorted(
            transition_counts.items(), key=lambda item: (numeric_label_key(item[0][0]), numeric_label_key(item[0][1]))
        )
    ]
    flipped_count = sum(row["is_noisy"] == "1" for row in rows)
    return {
        "dataset": dataset,
        "noise_protocol": protocol,
        "noise_strategy": IDN_STRATEGY if protocol == "idn" else GCCN_STRATEGY,
        "requested_noise_rate": float(noise_rate),
        "realized_noise_rate": flipped_count / max(1, len(rows)),
        "noise_seed": int(noise_seed),
        "mapping_type": IDN_MAPPING_TYPE if protocol == "idn" else GCCN_MAPPING_TYPE,
        "mapping_seed": int(noise_seed),
        "validation_seed": int(validation_seed),
        "source_scope": "source_train_before_fixed_validation_exclusion",
        "source_train_sample_count": len(rows),
        "source_train_flipped_count": int(flipped_count),
        "clean_class_count": len(class_order),
        "noisy_observed_class_count": len(observed_distribution),
        "noisy_label_distribution": dict(sorted(observed_distribution.items(), key=lambda item: numeric_label_key(item[0]))),
        "per_class": per_class,
        "per_class_transition_counts": transition_rows,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "source_noise_index": str(source_noise_index),
        "source_noise_index_sha256": file_sha256(source_noise_index),
        "validation_manifest_metadata": validation_metadata,
        "protocol_details": protocol_details,
        "official_test_policy": "unchanged base V1 eval_paths.txt/eval_labels.npy; test labels are neither read nor modified by this generator",
    }


def stable_percentile(values: np.ndarray, tie_break_indices: np.ndarray) -> np.ndarray:
    """Return deterministic [0, 1] ranks, using original source order for ties."""

    values = np.asarray(values, dtype=np.float64)
    tie_break_indices = np.asarray(tie_break_indices, dtype=np.int64)
    if values.ndim != 1 or tie_break_indices.shape != values.shape:
        raise ValueError("stable_percentile inputs must be aligned one-dimensional arrays.")
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    if len(values) == 1:
        return np.asarray([0.5], dtype=np.float64)
    order = np.lexsort((tie_break_indices, values))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64) / float(len(values) - 1)
    return ranks


def largest_remainder_counts(total: int, probabilities: tuple[float, ...]) -> tuple[int, ...]:
    """Allocate an integer target count with deterministic largest remainders."""

    if total < 0 or not probabilities or not np.isclose(sum(probabilities), 1.0):
        raise ValueError("largest_remainder_counts requires non-negative total and probabilities summing to one.")
    expected = np.asarray(probabilities, dtype=np.float64) * int(total)
    counts = np.floor(expected).astype(np.int64)
    remaining = int(total - counts.sum())
    order = np.lexsort((np.arange(len(probabilities), dtype=np.int64), -(expected - counts)))
    counts[order[:remaining]] += 1
    return tuple(int(value) for value in counts)


def canonical_class_order(labels: list[str]) -> list[str]:
    return sorted(set(str(label) for label in labels), key=numeric_label_key)


def manifest_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    preferred = [
        "index", "source_train_index", "image_id", "path", "abs_path", "split", "clean_label",
        "web_label", "noisy_label", "is_noisy", "noise_target_label", "noise_type", "noise_strategy",
        "noise_ratio", "noise_seed", "mapping_type", "mapping_seed", "idn_feature_distance",
        "idn_difficulty_percentile", "idn_flip_probability", "idn_target_prototype_cosine",
    ]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    return [field for field in preferred if any(field in row for row in rows)] + extras


def transition_matrix_fieldnames() -> list[str]:
    return ["source_label", "target_label", "transition_probability", "probability_given_flip", "transition_kind"]


def per_class_statistics_fieldnames() -> list[str]:
    return [
        "clean_label", "source_samples", "flipped_samples", "realized_noise_ratio",
        "idn_probability_mean", "idn_probability_min", "idn_probability_max", "observed_label_count",
    ]


def transition_count_fieldnames() -> list[str]:
    return ["source_clean_label", "target_noisy_label", "transition_count", "source_flipped_total"]


def _base_generated_row(
    original: dict[str, str],
    *,
    source_train_index: int,
    noisy_label: str,
    target_label: str,
    flipped: bool,
    noise_type: str,
    noise_strategy: str,
    noise_rate: float,
    noise_seed: int,
    mapping_type: str,
    mapping_seed: int,
) -> dict[str, str]:
    row = {str(key): str(value) for key, value in original.items()}
    row.update(
        {
            "source_train_index": str(source_train_index),
            "split": "train",
            "web_label": noisy_label,
            "noisy_label": noisy_label,
            "is_noisy": "1" if flipped else "0",
            "noise_target_label": target_label,
            "noise_type": noise_type,
            "noise_strategy": noise_strategy,
            "noise_ratio": f"{noise_rate:.12f}",
            "noise_seed": str(int(noise_seed)),
            "mapping_type": mapping_type,
            "mapping_seed": str(int(mapping_seed)),
        }
    )
    return row


def _validate_source_rows(rows: list[dict[str, str]], *, expected_count: int) -> None:
    if len(rows) != expected_count or not rows:
        raise ValueError("Source noise rows must be a non-empty, path-aligned source-train collection.")
    required = {"path", "split", "clean_label"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Source noise index is missing required columns: {sorted(missing)}")
    if any(str(row.get("split", "")).lower() != "train" for row in rows):
        raise ValueError("Only source train rows may be used to generate new noise.")
    if any(not str(row.get("clean_label", "")) for row in rows):
        raise ValueError("Every source row must contain a clean_label.")


def _validate_generated_rows(
    rows: list[dict[str, str]],
    *,
    expected_paths: list[str],
    protocol: ProtocolName,
    noise_rate: float,
    noise_seed: int,
) -> None:
    if len(rows) != len(expected_paths):
        raise ValueError("Generated manifest does not cover every source-train sample.")
    expected_strategy = IDN_STRATEGY if protocol == "idn" else GCCN_STRATEGY
    expected_type = "instance_dependent" if protocol == "idn" else "general_class_conditional"
    for source_index, row in enumerate(rows):
        if str(row.get("split")) != "train":
            raise ValueError(f"Generated row {source_index} is not a training row.")
        if int(row.get("source_train_index", -1)) != source_index:
            raise ValueError(f"Generated source_train_index mismatch at row {source_index}.")
        if not np.isclose(float(row.get("noise_ratio", -1.0)), noise_rate):
            raise ValueError(f"Generated noise rate mismatch at row {source_index}.")
        if int(row.get("noise_seed", -1)) != noise_seed:
            raise ValueError(f"Generated noise seed mismatch at row {source_index}.")
        if row.get("noise_strategy") != expected_strategy or row.get("noise_type") != expected_type:
            raise ValueError(f"Generated protocol metadata mismatch at row {source_index}.")
        if row.get("web_label") != row.get("noisy_label"):
            raise ValueError(f"Generated noisy-label aliases differ at row {source_index}.")
        flipped = row.get("is_noisy") == "1"
        if flipped != (row.get("web_label") != row.get("clean_label")):
            raise ValueError(f"Generated clean/noisy label invariant failed at row {source_index}.")
        if flipped and row.get("noise_target_label") != row.get("web_label"):
            raise ValueError(f"Generated target-label invariant failed at row {source_index}.")


def _validate_noise_parameters(
    *,
    noise_rate: float,
    noise_seed: int,
    validation_seed: int,
    idn_probability_modulation: float,
) -> None:
    if not 0.0 < float(noise_rate) < 1.0:
        raise ValueError("noise_rate must satisfy 0 < noise_rate < 1.")
    if int(noise_seed) < 0 or int(validation_seed) < 0:
        raise ValueError("noise_seed and validation_seed must be non-negative integers.")
    if not 0.0 <= float(idn_probability_modulation) <= 1.0:
        raise ValueError("idn_probability_modulation must satisfy 0 <= value <= 1.")
    if float(noise_rate) * (1.0 + float(idn_probability_modulation)) > 1.0:
        raise ValueError("IDN probability calibration exceeds one; reduce rate or modulation.")


def _row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Frozen CLS features contain a zero-norm vector.")
    return values / norms


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.isclose(np.std(left), 0.0) or np.isclose(np.std(right), 0.0):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _copy_identical(source: Path, destination: Path) -> None:
    import shutil

    if destination.exists():
        raise FileExistsError(f"Destination unexpectedly exists while freezing validation protocol: {destination}")
    shutil.copy2(source, destination)
