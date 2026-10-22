from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcdd.clean_replacement import ReplacementInputError, build_oracle_clean_replacement
from gcdd.lora_dynamic import validate_replay_selection_masks


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_historical_assets(tmp_path):
    paths = [f"/images/{index}.jpg" for index in range(16)]
    observed = np.asarray(["a"] * 8 + ["b"] * 8, dtype=str)
    clean = observed.copy()
    clean[6] = "not_a"
    clean[14] = "not_b"
    samples = []
    classes = []
    updates = []
    for epoch in (5, 10):
        for index, path in enumerate(paths):
            local = index % 8
            loss = local <= 6
            proto = local <= 4 or local == 6
            samples.append(
                {
                    "training_seed": 1,
                    "epoch": epoch,
                    "stable_sample_id": f"train_index:{index}",
                    "sample_path": path,
                    "observed_label": observed[index],
                    "loss_candidate": "yes" if loss else "no",
                    "prototype_candidate": "yes" if proto else "no",
                    "final_selected": "yes" if loss and proto else "no",
                    "selection_strategy": "loss_and_proto",
                }
            )
        for label in ("a", "b"):
            classes.append(
                {
                    "training_seed": 1,
                    "epoch": epoch,
                    "observed_label": label,
                    "total_count": 8,
                    "loss_candidate_count": 7,
                    "prototype_candidate_count": 6,
                    "intersection_count": 6,
                    "final_selected_count": 6,
                    "fallback_triggered": "no",
                    "selection_strategy": "loss_and_proto",
                }
            )
        updates.append(
            {
                "training_seed": 1,
                "update_epoch": epoch,
                "full_training_pool_size": 16,
                "selected_total": 12,
                "fallback_count": 0,
                "selection_strategy": "loss_and_proto",
            }
        )
    sample_path = tmp_path / "selection_samples.csv"
    class_path = tmp_path / "selection_per_class.csv"
    update_path = tmp_path / "selection_updates.csv"
    write_csv(sample_path, samples)
    write_csv(class_path, classes)
    write_csv(update_path, updates)
    return paths, observed, clean, sample_path, class_path, update_path


class CleanReplacementTests(unittest.TestCase):
    def test_oracle_clean_replacement_is_deterministic_and_budget_matched(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as raw_tmp:
            from pathlib import Path

            paths, observed, clean, sample_path, class_path, update_path = make_historical_assets(Path(raw_tmp))
            kwargs = {
                "dataset": "synthetic",
                "training_seed": 1,
                "train_paths": paths,
                "observed_labels": observed,
                "clean_labels": clean,
                "training_pool_mask": np.ones(len(paths), dtype=bool),
                "selection_samples_path": sample_path,
                "selection_per_class_path": class_path,
                "selection_updates_path": update_path,
                "expected_update_epochs": (5, 10),
            }
            first = build_oracle_clean_replacement(**kwargs)
            second = build_oracle_clean_replacement(**kwargs)
            for epoch in (5, 10):
                self.assertTrue(np.array_equal(first.replay_masks[epoch], second.replay_masks[epoch]))
                self.assertTrue(np.array_equal(first.oracle_masks[epoch], second.oracle_masks[epoch]))
                self.assertEqual(int(first.replay_masks[epoch].sum()), 12)
                self.assertEqual(int(first.oracle_masks[epoch].sum()), 12)
                r_clean = int(np.sum(first.replay_masks[epoch] & (clean == observed)))
                o_clean = int(np.sum(first.oracle_masks[epoch] & (clean == observed)))
                self.assertEqual(r_clean, o_clean)
                self.assertEqual(int(np.sum(first.replay_masks[epoch] != first.oracle_masks[epoch])), 4)
            self.assertEqual(sum(row["replacement_count"] for row in first.classwise_rows), 4)
            self.assertTrue(all(row["status"] == "PASS" for row in first.classwise_rows))

    def test_replacement_rejects_saved_fallback(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path

        with TemporaryDirectory() as raw_tmp:
            paths, observed, clean, sample_path, class_path, update_path = make_historical_assets(Path(raw_tmp))
            with class_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["fallback_triggered"] = "yes"
            write_csv(class_path, rows)
            with self.assertRaisesRegex(ReplacementInputError, "fallback"):
                build_oracle_clean_replacement(
                    dataset="synthetic",
                    training_seed=1,
                    train_paths=paths,
                    observed_labels=observed,
                    clean_labels=clean,
                    training_pool_mask=np.ones(len(paths), dtype=bool),
                    selection_samples_path=sample_path,
                    selection_per_class_path=class_path,
                    selection_updates_path=update_path,
                    expected_update_epochs=(5, 10),
                )

    def test_replay_mask_validation_rejects_out_of_pool_member(self):
        candidate = np.asarray([True, True, False], dtype=bool)
        with self.assertRaisesRegex(ValueError, "outside the candidate pool"):
            validate_replay_selection_masks(
                {5: np.asarray([True, False, True], dtype=bool)},
                candidate,
                [5],
            )


if __name__ == "__main__":
    unittest.main()
