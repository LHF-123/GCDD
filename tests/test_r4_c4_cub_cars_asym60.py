from __future__ import annotations

import csv
import json
import tempfile
import unittest
from subprocess import CompletedProcess
from pathlib import Path
from unittest.mock import patch

from gcdd.r4_c4_asym60 import (
    FROZEN,
    METHODS,
    Task,
    generate_cyclic_manifest,
    gpu_status,
    task_command,
    task_status,
)
from revision_experiments.profiling import RunProfiler


class R4C4Asym60Tests(unittest.TestCase):
    def test_cyclic_manifest_uses_per_class_rounding_and_numeric_next_class(self) -> None:
        rows = []
        for label in ("1", "2"):
            for offset in range(5):
                rows.append(
                    {
                        "index": str(len(rows)), "image_id": str(100 + len(rows)), "path": f"images/{label}/{offset}.jpg",
                        "split": "train", "clean_label": label, "web_label": label, "is_noisy": "0",
                    }
                )
        actual = generate_cyclic_manifest(rows, ratio=0.6, seed=42)
        repeated = generate_cyclic_manifest(rows, ratio=0.6, seed=42)
        self.assertEqual(actual, repeated)
        self.assertEqual(6, sum(row["is_noisy"] == "1" for row in actual))
        for row in actual:
            self.assertEqual("2" if row["clean_label"] == "1" else "1", row["noise_target_label"])
            if row["is_noisy"] == "1":
                self.assertEqual(row["noise_target_label"], row["web_label"])

    def test_command_is_a_real_single_method_seed_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = Task("cub", "CUB-200-2011-asym60-s42", "pgdf", "pgdf_dynamic_proto", True, 42, root,
                        root / "input", root / "manifest.csv", root / "validation", root / "instrumentation")
            command = task_command(task)
            self.assertIn("run_lora_checkpoint_validation.py", " ".join(command))
            self.assertIn("--methods", command)
            self.assertIn("pgdf_dynamic_proto", command)
            self.assertIn("--revision-selection-logging", command)
            self.assertIn("--official-test-selected-only", command)
            self.assertIn("--no-posthoc-oracle-test", command)
            self.assertEqual(4, len(METHODS))
            self.assertEqual((1, 42, 88), FROZEN["training_seeds"])

    def test_completion_validator_requires_full_protocol_and_five_selection_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = Task("cub", "CUB-200-2011-asym60-s42", "dynamic_small_loss", "dynamic_r08", True, 1, root,
                        root / "input", root / "manifest.csv", root / "validation", root / "instrumentation")
            task.validation_dir.mkdir(parents=True)
            for name in ("validation_manifest.csv", "validation_manifest.json"):
                (task.validation_dir / name).write_text(name, encoding="utf-8")
                task.trainer_root.mkdir(parents=True, exist_ok=True)
                (task.trainer_root / name).write_text(name, encoding="utf-8")
            (task.run_dir / "checkpoints").mkdir(parents=True)
            for name in ("best_val.pt", "last.pt"):
                (task.run_dir / "checkpoints" / name).write_bytes(b"checkpoint")
            result = {
                "method_key": "dynamic_r08", "seed": 1, "noise_strategy": "adjacent_cyclic", "noise_ratio": 0.6,
                "noise_seed": 42, "validation_seed": 20250726, "checkpoint_protocol": "fixed_clean_validation_v1",
                "best_epoch": 3, "official_test_evaluation": "validation_selected_only", "validation_selected_test_top1": 0.5,
            }
            (task.run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
            with (task.run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch"]); writer.writeheader()
                writer.writerows({"epoch": epoch} for epoch in range(1, 31))
            with (task.run_dir / "selection_updates.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch"]); writer.writeheader()
                writer.writerows({"epoch": epoch} for epoch in (5, 10, 15, 20, 25))
            for name in ("selection_rows.csv", "selection_per_class.csv"):
                (task.run_dir / name).write_text("epoch\n", encoding="utf-8")
            records = task.instrumentation_root / task.experiment_id / "selection_records"
            records.mkdir(parents=True)
            profile = {
                "recording_status": "complete", "profiling_mode": "detailed", "optimizer_steps_actual": 10,
                "stage_totals": {}, "memory": {"overall_run_peak": {}}, "selection_interval_steps": {},
            }
            profile_path = task.instrumentation_root / task.experiment_id / "profiling.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            task.task_dir.mkdir(parents=True, exist_ok=True)
            (task.task_dir / "runtime_environment.json").write_text("{}", encoding="utf-8")
            self.assertEqual(("COMPLETE_VERIFIED", []), task_status(task))
            (task.run_dir / "selection_updates.csv").write_text("epoch\n5\n10\n", encoding="utf-8")
            status, issues = task_status(task)
            self.assertEqual("PARTIAL", status)
            self.assertIn("selection_updates_not_expected_five_stages", issues)

    def test_shared_profiler_has_cpu_safe_step_and_memory_fields(self) -> None:
        profiler = RunProfiler("r4-c4-static-cpu", mode="detailed", device_type="cpu")
        with profiler.stage("total_training", phase="static_ce"):
            with profiler.memory_phase("training"):
                profiler.record_training_batch()
                profiler.record_optimizer_step(actual_update=True, epoch=1)
                profiler.record_scheduler_step()
        payload = profiler.to_dict()
        self.assertEqual("complete", payload["recording_status"])
        self.assertEqual(1, payload["optimizer_steps_actual"])
        self.assertEqual(1, payload["scheduler_steps_actual"])
        self.assertEqual(1, payload["training_batches_seen"])
        self.assertEqual("not_applicable_cpu_or_cuda_unavailable", payload["memory"]["overall_run_peak"]["status"])

    @patch("gcdd.r4_c4_asym60.subprocess.run")
    def test_nominal_24gb_gpu_gate_accepts_4090d_nvidia_smi_capacity(self, mocked_run) -> None:
        mocked_run.return_value = CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090 D, 24564, 24082, 0\n",
            stderr="",
        )
        accepted = gpu_status(24000)
        rejected = gpu_status(24576)
        self.assertEqual("READY", accepted["status"])
        self.assertEqual(24000, accepted["minimum_total_mib"])
        self.assertEqual("nominal_24gb_24000_mib", accepted["capacity_policy"])
        self.assertEqual("READY_WAIT_GPU", rejected["status"])


if __name__ == "__main__":
    unittest.main()
