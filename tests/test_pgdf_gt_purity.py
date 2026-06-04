from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analyze_pgdf_gt_purity import (
    binary_auc,
    build_gt_index,
    compute_selection_metrics,
    path_key_candidates,
    resolve_selection_source,
)


class PgdfGtPurityTest(unittest.TestCase):
    def test_path_key_candidates_keep_class_and_file(self) -> None:
        keys = path_key_candidates("/remote/root/CUB_200_2011/images/001.Class/image_001.jpg")

        self.assertIn("001.Class/image_001.jpg", keys)
        self.assertIn("images/001.Class/image_001.jpg", keys)
        self.assertNotIn("image_001.jpg", keys)

    def test_compute_selection_metrics_matches_different_roots(self) -> None:
        gt_rows = [
            {
                "index": "0",
                "path": "train/Class A/0001.jpg",
                "split": "train",
                "clean_label": "Class A",
                "web_label": "Class A",
                "is_noisy": "0",
            },
            {
                "index": "1",
                "path": "train/Class A/0002.jpg",
                "split": "train",
                "clean_label": "Class A",
                "web_label": "Class B",
                "is_noisy": "1",
            },
            {
                "index": "2",
                "path": "train/Class B/0003.jpg",
                "split": "train",
                "clean_label": "Class B",
                "web_label": "Class B",
                "is_noisy": "0",
            },
        ]
        gt_map, gt_samples = build_gt_index(gt_rows)
        base_clean_total = sum(1 for sample in gt_samples if sample.is_clean)
        selection_rows = [
            {"path": "/mnt/other_root/train/Class A/0001.jpg", "state": "clean"},
            {"path": "/mnt/other_root/train/Class A/0002.jpg", "state": "clean"},
            {"path": "/mnt/other_root/train/Class B/0003.jpg", "state": "ignored"},
        ]

        metrics = compute_selection_metrics(selection_rows, gt_map, base_clean_total)

        self.assertEqual(metrics.selected, 2)
        self.assertEqual(metrics.clean_selected, 1)
        self.assertEqual(metrics.purity, 0.5)
        self.assertEqual(metrics.clean_recall, 0.5)

    def test_ambiguous_path_key_raises(self) -> None:
        rows = [
            {
                "index": "0",
                "path": "train/Class A/shared.jpg",
                "split": "train",
                "clean_label": "Class A",
                "web_label": "Class A",
                "is_noisy": "0",
            },
            {
                "index": "1",
                "path": "other/Class A/shared.jpg",
                "split": "train",
                "clean_label": "Class A",
                "web_label": "Class B",
                "is_noisy": "1",
            },
        ]

        with self.assertRaisesRegex(ValueError, "Ambiguous path key"):
            build_gt_index(rows)

    def test_resolve_selection_source_uses_largest_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            old = tmp_path / "dynamic_loss_selection_r0p8_seed42_epoch_005.csv"
            latest = tmp_path / "dynamic_loss_selection_r0p8_seed42_epoch_025.csv"
            wrong_ratio = tmp_path / "dynamic_loss_selection_r0p9_seed42_epoch_025.csv"
            _write_csv(old, [{"path": "a.jpg", "state": "clean"}])
            _write_csv(latest, [{"path": "b.jpg", "state": "clean"}])
            _write_csv(wrong_ratio, [{"path": "c.jpg", "state": "clean"}])

            selected = resolve_selection_source(tmp_path, "dynamic", "last", retention_ratio=0.8)

        self.assertEqual(selected, latest)

    def test_binary_auc_handles_ties(self) -> None:
        labels = [1, 1, 0, 0]
        scores = [0.9, 0.5, 0.5, 0.1]

        self.assertEqual(binary_auc(labels, scores), 0.875)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
