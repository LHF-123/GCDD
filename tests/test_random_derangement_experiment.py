from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from gcdd.checkpoint_validation import PROTOCOL_NAME, hash_paths
from gcdd.random_derangement import (
    audit_noise_control,
    file_sha256,
    prepare_mapping,
    prepare_noise_manifest,
    read_csv_with_fields,
)
from scripts.run_random_derangement_asym40_45runs import (
    DATASETS,
    DEFAULT_EXPERIMENT_ROOT,
    METHODS,
    build_run_plan,
)


class RandomDerangementExperimentTests(unittest.TestCase):
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
            self.assertEqual(first_hash, file_sha256(path))
            self.assertEqual(set(first), set(classes))
            self.assertEqual(set(first.values()), set(classes))
            self.assertTrue(all(source != target for source, target in first.items()))

    def test_source_flip_mask_validation_and_pool_counts_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.csv"
            mapping_file = root / "mapping.json"
            manifest = root / "alternative.csv"
            validation_source = root / "formal_validation"
            validation_destination = root / "new_validation"
            validation_source.mkdir()

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
            (validation_source / "validation_manifest.json").write_text(
                json.dumps(
                    {
                        "protocol": PROTOCOL_NAME,
                        "validation_ratio": 0.10,
                        "validation_seed": 20250726,
                        "train_paths_sha256": hash_paths(paths),
                        "source_train_samples": 20,
                        "training_pool_samples": 18,
                        "validation_samples": 2,
                        "stratification_label": "clean_label",
                        "validation_label": "clean_label",
                    },
                    indent=2,
                ),
                encoding="utf-8",
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
            )
            alternative_rows = read_csv_with_fields(manifest)[0]
            original_mask = [row["is_noisy"] == "true" for row in rows]
            alternative_mask = [row["web_label"] != row["clean_label"] for row in alternative_rows]
            self.assertEqual(original_mask, alternative_mask)
            self.assertEqual(audit["flip_mask_mismatch_count"], 0)
            self.assertEqual(audit["validation_manifest_mismatch_count"], 0)
            self.assertEqual(file_sha256(validation_source / "validation_manifest.csv"), file_sha256(validation_destination / "validation_manifest.csv"))
            self.assertEqual((audit["training_pool"], audit["training_pool_clean"], audit["training_pool_noisy"]), (18, 12, 6))

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


if __name__ == "__main__":
    unittest.main()
