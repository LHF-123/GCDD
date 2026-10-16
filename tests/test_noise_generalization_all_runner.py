from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from tools.run_noise_generalization_all import (
    DYNAMIC_RATIO,
    EPOCHS,
    FIXED_P,
    METHODS,
    NOISE_RATE,
    NOISE_SEED,
    PROTOCOLS,
    TRAINING_SEEDS,
    UPDATE_INTERVAL,
    VALIDATION_SEED,
    WARMUP_EPOCHS,
    build_plan,
    build_training_command,
    ensure_fixed_validation_manifest,
    is_task_complete,
    load_config,
    materialize_training_validation_manifest,
    run_orchestration,
    validate_all_dataset_inputs,
)


class NoiseGeneralizationAllRunnerTests(unittest.TestCase):
    def test_plan_has_all_126_tasks_and_correct_method_protocol_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(_write_config(Path(tmp)))
            plan = build_plan(config)

            self.assertEqual(126, len(plan))
            self.assertEqual(set(METHODS), {task.method for task in plan})
            self.assertEqual(set(PROTOCOLS), {task.protocol for task in plan})
            pgdf = next(task for task in plan if task.method == "pgdf_dynamic_proto")
            self.assertIn("--official-test-selected-only", build_training_command(config, pgdf))
            self.assertNotIn("pgdf_fixed", build_training_command(config, pgdf))
            coteaching = next(task for task in plan if task.method == "coteaching")
            self.assertNotIn("--official-test-selected-only", build_training_command(config, coteaching))

    def test_dry_run_validates_inputs_and_does_not_launch_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(_write_config(root))
            with contextlib.redirect_stdout(io.StringIO()):
                run_orchestration(config, dry_run=True, summarize_only=False, resume=True)

            self.assertTrue((config.training_outputs_root / "orchestration" / "run_plan.csv").is_file())
            self.assertFalse(any(config.noise_assets_root.rglob("noise_manifest.csv")) if config.noise_assets_root.exists() else False)
            for dataset in config.datasets:
                self.assertFalse((dataset.validation_dir / "validation_manifest.csv").exists())

    def test_missing_base_input_fails_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(_write_config(root))
            missing = config.datasets[0].base_input_dir / "features_cls.npy"
            missing.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "base V1 input is incomplete"):
                validate_all_dataset_inputs(config.datasets)

    def test_completion_audit_rejects_mismatched_or_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(_write_config(root))
            task = build_plan(config)[0]
            ensure_fixed_validation_manifest(task.dataset, dry_run=False)
            materialize_training_validation_manifest(task)
            task.run_dir.mkdir(parents=True)
            _write_complete_task(task)
            complete, _, _ = is_task_complete(task)
            self.assertTrue(complete)

            result_path = task.run_dir / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["noise_seed"] = 7
            result_path.write_text(json.dumps(result), encoding="utf-8")
            complete, detail, _ = is_task_complete(task)
            self.assertFalse(complete)
            self.assertIn("noise_seed", detail)

    def test_independent_validation_manifest_is_created_and_copied_to_training_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = load_config(_write_config(root))
            dataset = config.datasets[0]
            self.assertEqual("CREATED", ensure_fixed_validation_manifest(dataset, dry_run=False))
            task = build_plan(config)[0]
            materialize_training_validation_manifest(task)
            for name in ("validation_manifest.csv", "validation_manifest.json"):
                self.assertEqual(
                    (dataset.validation_dir / name).read_bytes(),
                    (task.run_root / name).read_bytes(),
                )

    def test_repository_config_does_not_depend_on_dynamic_fusion_outputs(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "noise_generalization_all.yaml"
        config = load_config(config_path)
        for dataset in config.datasets:
            self.assertIn("noise_generalization/fixed_validation", dataset.validation_dir.as_posix())
            self.assertNotIn("pgdf_dynamic_ref", dataset.validation_dir.as_posix())


def _write_config(root: Path) -> Path:
    datasets: dict[str, dict[str, object]] = {}
    for key, display_name in (("cub", "CUB-200-2011"), ("cars", "Stanford Cars"), ("aircraft", "FGVC-Aircraft")):
        base = root / key / "base"
        validation = root / key / "validation"
        base.mkdir(parents=True)
        paths = [f"/images/{label}/{index}.jpg" for label in ("A", "B") for index in range(2)]
        labels = ["A", "A", "B", "B"]
        (base / "paths.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        for name in ("eval_paths.txt", "eval_labels.npy", "features_cls.npy", "features_gap.npy", "features_top.npy", "resolved_config.yaml"):
            (base / name).write_bytes(b"test")
        source_rows = ["path,split,clean_label", *[f"{path},train,{label}" for path, label in zip(paths, labels, strict=True)]]
        (root / key / "source.csv").write_text("\n".join(source_rows) + "\n", encoding="utf-8")
        datasets[key] = {
            "display_name": display_name,
            "base_input_dir": str(base),
            "source_noise_index": str(root / key / "source.csv"),
            "validation_dir": str(validation),
            "path_maps": [],
            "training_config": None,
        }
    payload = {
        "schema_version": 1,
        "runtime": {"python_executable": sys.executable},
        "output": {"noise_assets_root": str(root / "assets"), "training_outputs_root": str(root / "training")},
        "instrumentation": {"profiling_mode": "off", "selection_logging": False, "output_root": None},
        "formal": {
            "protocols": list(PROTOCOLS), "noise_rate": NOISE_RATE, "noise_seed": NOISE_SEED,
            "validation_seed": VALIDATION_SEED, "training_seeds": list(TRAINING_SEEDS), "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS, "update_interval": UPDATE_INTERVAL,
            "dynamic_ratio": DYNAMIC_RATIO, "fixed_p": FIXED_P,
        },
        "datasets": datasets,
    }
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_complete_task(task: object) -> None:
    run_dir = task.run_dir
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "best_val.pt").write_bytes(b"checkpoint")
    result = {
        "method_key": task.method,
        "seed": task.seed,
        "noise_strategy": task.expected_noise_strategy,
        "noise_ratio": NOISE_RATE,
        "noise_seed": NOISE_SEED,
        "validation_seed": VALIDATION_SEED,
        "checkpoint_protocol": "fixed_clean_validation_v1",
        "validation_selected_test_top1": 0.5,
        "official_test_evaluation": "validation_selected_only" if task.single_branch else "",
    }
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    policy = {
        "official_test_used_for_checkpoint_selection": False,
        "validation_samples_are_training_candidates": False,
    }
    (run_dir / "selection_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerows({"epoch": epoch} for epoch in range(1, EPOCHS + 1))


if __name__ == "__main__":
    unittest.main()
