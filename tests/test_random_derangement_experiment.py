from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from gcdd.checkpoint_validation import PROTOCOL_NAME, hash_paths
from gcdd.checkpoint_validation import build_or_load_fixed_validation_split, load_clean_labels_from_noise_index
from gcdd.noise_generalization import (
    GCCN_STRATEGY,
    IDN_STRATEGY,
    generate_general_class_conditional_noise_rows,
    generate_instance_dependent_noise_rows,
    prepare_noise_generalization,
)
from gcdd.random_derangement import (
    audit_noise_control,
    file_sha256,
    mapping_file_sha256,
    prepare_mapping,
    prepare_noise_manifest,
    read_csv_with_fields,
)
from scripts.run_lora_checkpoint_validation import load_data, load_noise_metadata
from scripts.run_random_derangement_asym40_45runs import (
    DATASETS,
    DEFAULT_EXPERIMENT_ROOT,
    METHODS,
    build_run_plan,
)


class RandomDerangementExperimentTests(unittest.TestCase):
    def test_noise_generalization_protocols_are_reproducible_and_v1_compatible(self) -> None:
        """Exercise the IDN/GCCN generators without data downloads or GPU work."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base_v1"
            validation_dir = root / "formal_validation"
            source_index = root / "source_cyclic_index.csv"
            base.mkdir()
            validation_dir.mkdir()
            source_rows, paths, features = _synthetic_noise_generalization_source()
            fields = [
                "index", "path", "split", "clean_label", "web_label", "is_noisy",
                "noise_target_label", "noise_type", "noise_strategy", "noise_ratio", "noise_seed",
            ]
            _write_csv(source_index, source_rows, fields)
            (base / "paths.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
            (base / "eval_paths.txt").write_text("official/test.jpg\n", encoding="utf-8")
            np.save(base / "eval_labels.npy", np.asarray(["0"], dtype=str))
            np.save(base / "features_cls.npy", features)
            np.save(base / "features_gap.npy", features)
            np.save(base / "features_top.npy", features)
            (base / "resolved_config.yaml").write_text(
                yaml.safe_dump({"dataset": {"name": "synthetic"}, "output": {"version": "base"}}),
                encoding="utf-8",
            )
            clean_labels = np.asarray([row["clean_label"] for row in source_rows], dtype=str)
            build_or_load_fixed_validation_split(
                validation_dir,
                paths,
                clean_labels,
                validation_ratio=0.10,
                validation_seed=20250726,
            )

            original_rows = [dict(row) for row in source_rows]
            idn_rows_a, idn_details_a = generate_instance_dependent_noise_rows(
                source_rows,
                features,
                noise_rate=0.4,
                noise_seed=42,
                probability_modulation=0.5,
            )
            idn_rows_b, idn_details_b = generate_instance_dependent_noise_rows(
                source_rows,
                features,
                noise_rate=0.4,
                noise_seed=42,
                probability_modulation=0.5,
            )
            self.assertEqual(idn_rows_a, idn_rows_b)
            self.assertEqual(idn_details_a, idn_details_b)
            self.assertEqual(source_rows, original_rows)
            probabilities = {row["idn_flip_probability"] for row in idn_rows_a}
            self.assertGreater(len(probabilities), 1)
            self.assertGreater(float(idn_details_a["distance_probability_correlation"]), 0.0)
            self.assertTrue(all(row["noise_strategy"] == IDN_STRATEGY for row in idn_rows_a))

            gccn_rows, matrix_rows, gccn_details = generate_general_class_conditional_noise_rows(
                source_rows,
                noise_rate=0.4,
                noise_seed=42,
            )
            self.assertTrue(all(row["noise_strategy"] == GCCN_STRATEGY for row in gccn_rows))
            self.assertTrue(all(float(row["transition_probability"]) >= 0.0 for row in matrix_rows))
            for source_label in {row["clean_label"] for row in source_rows}:
                noisy_targets = {
                    row["target_label"]
                    for row in matrix_rows
                    if row["source_label"] == source_label and row["transition_kind"] == "noisy_multi_target"
                }
                self.assertEqual(len(noisy_targets), 3)
                self.assertEqual(
                    sum(
                        float(row["transition_probability"])
                        for row in matrix_rows
                        if row["source_label"] == source_label
                    ),
                    1.0,
                )
            self.assertEqual(gccn_details["probability_given_flip"], [0.5, 0.3, 0.2])

            idn_output = root / "idn_output"
            idn_stats = prepare_noise_generalization(
                protocol="idn",
                dataset="Synthetic",
                base_input_dir=base,
                source_noise_index=source_index,
                validation_dir=validation_dir,
                output_dir=idn_output,
                noise_rate=0.4,
                noise_seed=42,
                validation_seed=20250726,
            )
            self.assertEqual(idn_stats["source_train_sample_count"], len(paths))
            self.assertAlmostEqual(idn_stats["requested_noise_rate"], 0.4)
            self.assertTrue((idn_output / "noise_manifest.csv.sha256").is_file())
            self.assertEqual(
                (validation_dir / "validation_manifest.csv").read_bytes(),
                (idn_output / "protocol" / "validation_manifest.csv").read_bytes(),
            )
            self.assertEqual(
                load_noise_metadata(idn_output / "noise_manifest.csv")["noise_strategy"],
                IDN_STRATEGY,
            )
            prepared = load_data(idn_output / "prepared_input")
            self.assertEqual(prepared["train_paths"], paths)
            self.assertTrue(np.array_equal(prepared["test_labels"], np.asarray(["0"], dtype=str)))
            self.assertTrue(
                np.array_equal(
                    load_clean_labels_from_noise_index(idn_output / "noise_manifest.csv", paths),
                    clean_labels,
                )
            )

            gccn_output = root / "gccn_output"
            gccn_stats = prepare_noise_generalization(
                protocol="gccn",
                dataset="Synthetic",
                base_input_dir=base,
                source_noise_index=source_index,
                validation_dir=validation_dir,
                output_dir=gccn_output,
                noise_rate=0.4,
                noise_seed=42,
                validation_seed=20250726,
            )
            self.assertTrue((gccn_output / "transition_matrix.csv").is_file())
            self.assertEqual(gccn_stats["noise_strategy"], GCCN_STRATEGY)
            self.assertEqual(gccn_stats["realized_noise_rate"], 5 / 12)
            transitions = gccn_stats["per_class_transition_counts"]
            self.assertGreaterEqual(len({row["target_noisy_label"] for row in transitions}), 3)

    def test_mapping_is_deranged_bijective_and_reload_hash_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.json"
            classes = [str(index) for index in range(20)]
            first, first_hash = prepare_mapping(
                path,
                dataset="synthetic",
                class_order=classes,
                mapping_seed=20260815,
            )
            second, second_hash = prepare_mapping(
                path,
                dataset="synthetic",
                class_order=classes,
                mapping_seed=20260815,
            )
            self.assertEqual(first, second)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first_hash, mapping_file_sha256(path))
            self.assertEqual(set(first), set(classes))
            self.assertEqual(set(first.values()), set(classes))
            self.assertTrue(all(source != target for source, target in first.items()))

            # Simulate Git checking a Windows-created JSON out on Linux.  The
            # semantic mapping and its frozen published hash must not change.
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            linux_mapping, linux_hash = prepare_mapping(
                path,
                dataset="synthetic",
                class_order=classes,
                mapping_seed=20260815,
            )
            self.assertEqual(first, linux_mapping)
            self.assertEqual(first_hash, linux_hash)
            self.assertEqual(first_hash, mapping_file_sha256(path))

    def test_source_flip_mask_validation_and_pool_counts_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.csv"
            mapping_file = root / "mapping.json"
            manifest = root / "alternative.csv"
            validation_source = root / "formal_validation"
            validation_peer = root / "peer_validation"
            validation_missing_peer = root / "missing_peer_validation"
            validation_partial_peer = root / "partial_peer_validation"
            validation_destination = root / "new_validation"
            validation_source.mkdir()
            validation_peer.mkdir()

            paths = [f"train/{label}/sample_{index:02d}.jpg" for label in ("0", "1") for index in range(10)]
            fields = [
                "index", "path", "split", "clean_label", "web_label", "is_noisy",
                "noise_target_label", "noise_type", "noise_strategy", "noise_ratio", "noise_seed",
            ]
            rows = []
            for source_index, path in enumerate(paths):
                label = "0" if source_index < 10 else "1"
                target = "1" if label == "0" else "0"
                class_index = source_index % 10
                flipped = class_index < 4
                rows.append(
                    {
                        "index": source_index,
                        "path": path,
                        "split": "train",
                        "clean_label": label,
                        "web_label": target if flipped else label,
                        "is_noisy": "true" if flipped else "false",
                        "noise_target_label": target,
                        "noise_type": "asymmetric",
                        "noise_strategy": "adjacent_cyclic",
                        "noise_ratio": 0.4,
                        "noise_seed": 42,
                    }
                )
            _write_csv(original, rows, fields)

            mapping, mapping_hash = prepare_mapping(
                mapping_file,
                dataset="synthetic",
                class_order=["0", "1"],
                mapping_seed=20260815,
            )
            prepare_noise_manifest(
                manifest,
                dataset="synthetic",
                original_noise_index=original,
                train_paths=paths,
                mapping=mapping,
                mapping_file=mapping_file,
                mapping_sha256=mapping_hash,
                noise_seed=42,
                mapping_seed=20260815,
                validation_seed=20250726,
                noise_rate=0.4,
            )

            validation_indices = {0, 10}
            validation_rows = [
                {
                    "index": index,
                    "path": path,
                    "clean_label": "0" if index < 10 else "1",
                    "partition": "validation" if index in validation_indices else "training_pool",
                }
                for index, path in enumerate(paths)
            ]
            _write_csv(
                validation_source / "validation_manifest.csv",
                validation_rows,
                ["index", "path", "clean_label", "partition"],
            )
            validation_metadata = {
                "protocol": PROTOCOL_NAME,
                "validation_ratio": 0.10,
                "validation_seed": 20250726,
                "train_paths_sha256": hash_paths(paths),
                "source_train_samples": 20,
                "training_pool_samples": 18,
                "validation_samples": 2,
                "stratification_label": "clean_label",
                "validation_label": "clean_label",
            }
            (validation_source / "validation_manifest.json").write_text(
                json.dumps(validation_metadata, indent=2),
                encoding="utf-8",
            )
            source_csv = validation_source / "validation_manifest.csv"
            # Same rows with LF-only bytes and differently formatted JSON must
            # remain the exact same validation experiment semantically.
            (validation_peer / "validation_manifest.csv").write_bytes(
                source_csv.read_bytes().replace(b"\r\n", b"\n")
            )
            (validation_peer / "validation_manifest.json").write_bytes(
                (json.dumps(validation_metadata, separators=(",", ":")) + "\n").encode("utf-8")
            )
            self.assertNotEqual(
                file_sha256(source_csv),
                file_sha256(validation_peer / "validation_manifest.csv"),
            )

            audit = audit_noise_control(
                dataset="synthetic",
                train_paths=paths,
                original_noise_index=original,
                alternative_manifest=manifest,
                validation_source_dir=validation_source,
                validation_destination_dir=validation_destination,
                mapping=mapping,
                mapping_file=mapping_file,
                mapping_sha256=mapping_hash,
                noise_seed=42,
                mapping_seed=20260815,
                validation_seed=20250726,
                noise_rate=0.4,
                expected_pool=(18, 12, 6),
                peer_validation_dirs=(validation_peer, validation_missing_peer),
            )
            alternative_rows = read_csv_with_fields(manifest)[0]
            original_mask = [row["is_noisy"] == "true" for row in rows]
            alternative_mask = [row["web_label"] != row["clean_label"] for row in alternative_rows]
            self.assertEqual(original_mask, alternative_mask)
            self.assertEqual(audit["flip_mask_mismatch_count"], 0)
            self.assertEqual(audit["validation_manifest_mismatch_count"], 0)
            self.assertEqual(file_sha256(validation_source / "validation_manifest.csv"), file_sha256(validation_destination / "validation_manifest.csv"))
            self.assertEqual((audit["training_pool"], audit["training_pool_clean"], audit["training_pool_noisy"]), (18, 12, 6))
            self.assertTrue(audit["peer_validation_manifest_comparisons"][0]["semantic_equal"])
            self.assertEqual(audit["peer_validation_manifest_comparisons"][0]["status"], "PASS")
            self.assertEqual(audit["peer_validation_manifest_comparisons"][1]["status"], "OPTIONAL_MISSING")

            validation_partial_peer.mkdir()
            (validation_partial_peer / "validation_manifest.csv").write_bytes(source_csv.read_bytes())
            with self.assertRaises(FileNotFoundError):
                audit_noise_control(
                    dataset="synthetic",
                    train_paths=paths,
                    original_noise_index=original,
                    alternative_manifest=manifest,
                    validation_source_dir=validation_source,
                    validation_destination_dir=validation_destination,
                    mapping=mapping,
                    mapping_file=mapping_file,
                    mapping_sha256=mapping_hash,
                    noise_seed=42,
                    mapping_seed=20260815,
                    validation_seed=20250726,
                    noise_rate=0.4,
                    expected_pool=(18, 12, 6),
                    peer_validation_dirs=(validation_partial_peer,),
                )

            changed_rows = [dict(row) for row in validation_rows]
            changed_rows[1]["partition"] = "validation"
            _write_csv(
                validation_peer / "validation_manifest.csv",
                changed_rows,
                ["index", "path", "clean_label", "partition"],
            )
            with self.assertRaises(ValueError):
                audit_noise_control(
                    dataset="synthetic",
                    train_paths=paths,
                    original_noise_index=original,
                    alternative_manifest=manifest,
                    validation_source_dir=validation_source,
                    validation_destination_dir=validation_destination,
                    mapping=mapping,
                    mapping_file=mapping_file,
                    mapping_sha256=mapping_hash,
                    noise_seed=42,
                    mapping_seed=20260815,
                    validation_seed=20250726,
                    noise_rate=0.4,
                    expected_pool=(18, 12, 6),
                    peer_validation_dirs=(validation_peer,),
                )

    def test_master_plan_has_45_unique_runs_and_exact_dependencies(self) -> None:
        mapping_hashes = {dataset.key: dataset.key * 64 for dataset in DATASETS}
        plan = build_run_plan(DEFAULT_EXPERIMENT_ROOT, mapping_hashes)
        self.assertEqual(len(plan), 45)
        self.assertEqual(len({row["run_id"] for row in plan}), 45)
        self.assertEqual(
            {method.key: sum(row["method"] == method.key for row in plan) for method in METHODS},
            {method.key: 9 for method in METHODS},
        )
        by_id = {row["run_id"]: row for row in plan}
        bmd = [row for row in plan if row["method"] == "budget_matched_dynamic"]
        self.assertEqual(len(bmd), 9)
        for row in bmd:
            dependency = by_id[row["dependency_run_id"]]
            self.assertEqual(dependency["method"], "pgdf")
            self.assertEqual(dependency["dataset"], row["dataset"])
            self.assertEqual(dependency["training_seed"], row["training_seed"])
            self.assertEqual(dependency["mapping_sha256"], row["mapping_sha256"])


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_noise_generalization_source() -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    """Four classes with per-instance prototype distance variation for CPU tests."""
    rows: list[dict[str, str]] = []
    paths: list[str] = []
    features: list[np.ndarray] = []
    identity = np.eye(4, dtype=np.float32)
    for class_index in range(4):
        label = str(class_index)
        target = str((class_index + 1) % 4)
        for sample_index in range(12):
            path = f"train/{label}/sample_{sample_index:02d}.jpg"
            paths.append(path)
            # The last item is visibly less typical of its clean class, which
            # must receive a higher IDN flip probability than the prototypes.
            feature = identity[class_index].copy()
            if sample_index == 11:
                feature = 0.6 * identity[class_index] + 0.8 * identity[(class_index + 1) % 4]
            features.append(feature / np.linalg.norm(feature))
            rows.append(
                {
                    "index": str(len(rows)),
                    "path": path,
                    "split": "train",
                    "clean_label": label,
                    "web_label": label,
                    "is_noisy": "0",
                    "noise_target_label": target,
                    "noise_type": "asymmetric",
                    "noise_strategy": "adjacent_cyclic",
                    "noise_ratio": "0.400000",
                    "noise_seed": "42",
                }
            )
    return rows, paths, np.asarray(features, dtype=np.float32)


if __name__ == "__main__":
    unittest.main()
