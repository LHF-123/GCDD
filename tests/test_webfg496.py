from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.webfg496.dataset import inventory_webfg, load_webfg_splits, parse_list_file
from experiments.webfg496.runner import build_matrix, dynamic_profile, load_config, static_profile


class WebFG496DatasetTests(unittest.TestCase):
    def test_list_adapter_and_mapping_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset in ("web-bird", "web-car", "web-aircraft"):
                for split in ("train", "val"):
                    image = root / dataset / split / "class_zero" / "sample.jpg"
                    image.parent.mkdir(parents=True)
                    image.write_bytes(b"not-opened-in-this-structural-test")
                    (root / dataset / f"{split}-list.txt").write_text(
                        f"{split}/class_zero/sample.jpg: 0\n", encoding="utf-8"
                    )
            records = parse_list_file(root / "web-bird" / "train-list.txt")
            self.assertEqual(records[0].label, 0)
            train, evaluation = load_webfg_splits(root, "web-bird")
            self.assertEqual(len(train.paths), 1)
            self.assertEqual(len(evaluation.labels), 1)
            rows, mapping = inventory_webfg(
                root,
                {"web-bird": 1, "web-car": 1, "web-aircraft": 1},
                verify_images=False,
            )
            self.assertTrue(all(row["dataset_status"] == "PASS" for row in rows))
            self.assertTrue(mapping["web-bird"]["classes"][0]["mapping_consistent"])

    def test_rejects_unsafe_list_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-list.txt"
            path.write_text("../outside.jpg: 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_list_file(path)


class WebFG496SchedulerTests(unittest.TestCase):
    def test_formal_matrix_has_36_coordinates(self) -> None:
        config = load_config(Path("configs/webfg496/formal.yaml"))
        matrix = build_matrix(config)
        self.assertEqual(len(matrix), 36)
        self.assertEqual({item["seed"] for item in matrix}, {1, 42, 88})
        self.assertEqual({item["method"] for item in matrix}, {"pgdf_dynamic_proto", "all_noisy_ce", "jal_ce", "coteaching"})

    def test_profiling_schemas_keep_static_gap_and_pgdf_details_separate(self) -> None:
        base = {
            "sample_exposure": 30 * 96,
            "epoch_count": 30,
            "effective_batch": {"batch_size": 48, "gradient_accumulation_steps": 1},
            "training_wall_seconds": 10.0,
        }
        static = static_profile(dict(base), "all_noisy_ce", grad_accum_steps=1)
        self.assertEqual(static["optimizer_steps_attempted"], 60)
        self.assertIsNone(static["optimizer_steps_actual"])
        profiler = {
            "optimizer_steps_attempted": 60,
            "optimizer_steps_actual": 60,
            "scheduler_steps_actual": 60,
            "training_batches_seen": 60,
            "stage_totals": {
                "selection_update_total": {"duration_seconds": 2.0},
                "selection_loss_forward": {"duration_seconds": 0.6},
                "prototype_construction": {"duration_seconds": 0.8},
                "fusion_selection": {"duration_seconds": 0.2},
            },
            "detailed_events": [],
        }
        pgdf = dynamic_profile(dict(base), profiler, [{"epoch": 5, "num_selected": 80, "fallback_count": 1}])
        self.assertEqual(pgdf["profiling_completeness"], "COMPLETE")
        self.assertEqual(pgdf["selection"]["selection_update_count"], 1)


if __name__ == "__main__":
    unittest.main()
