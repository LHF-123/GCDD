from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from PIL import Image

from gcdd.ido_batch import (
    IDO_RUNTIME_CODE_PATHS,
    build_plan,
    build_training_command,
    build_task_contract,
    find_reusable_completed_task,
    is_complete,
    load_config,
    materialize_task_contract,
    materialize_validation_manifest,
    main_for_protocols,
    preflight,
    summarize_all,
    validate_ido_trainable_state_dict,
)
from gcdd.lora_training import (
    build_dinov2_vitb14_lora_structure_contract,
    canonical_structure_sha256,
)
from scripts.run_lora_checkpoint_validation import load_ido_task_contract


class IDOBatchTests(unittest.TestCase):
    @staticmethod
    def _valid_controlled_state_dict() -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {
            "head.weight": torch.zeros(2, 768),
            "head.bias": torch.zeros(2),
        }
        for block_index in range(12):
            prefix = f"backbone.blocks.{block_index}.attn.qkv"
            state[f"{prefix}.lora_a.weight"] = torch.zeros(2, 768)
            state[f"{prefix}.lora_b.weight"] = torch.zeros(2304, 2)
        return state

    @classmethod
    def _valid_lora_structure_metadata(cls) -> dict[str, object]:
        contract = build_dinov2_vitb14_lora_structure_contract(
            target_modules=("qkv",),
            rank=2,
            num_classes=2,
        )
        state = cls._valid_controlled_state_dict()
        runtime = {
            "schema_version": 1,
            "record_source": "live_model_after_inject_lora",
            "injected_backbone_modules": [item["backbone_module"] for item in contract["lora_modules"]],
            "target_modules": ["qkv"],
            "rank": 2,
            "trainable_parameters": [
                {
                    "key": key,
                    "shape": [int(size) for size in state[key].shape],
                    "dtype": str(state[key].dtype),
                }
                for key in sorted(state)
            ],
        }
        return {
            "lora_structure_contract": contract,
            "lora_structure_contract_sha256": canonical_structure_sha256(contract),
            "lora_runtime_structure": runtime,
            "lora_runtime_structure_sha256": canonical_structure_sha256(runtime),
            "dinov2_pretrained_source": {"loader": "synthetic-test"},
            "runtime_environment": {"python": "synthetic-test"},
        }

    def _write_fixture_config(
        self,
        root: Path,
        *,
        protocol_frozen: bool = False,
        bmm_mode: str = "classwise",
    ) -> Path:
        datasets: dict[str, object] = {}
        strategies = {
            "cyclic": "adjacent_cyclic",
            "idn": "instance_dependent_dinov2_cls_v1",
            "gccn": "general_class_conditional_multi_offset_v1",
        }
        for dataset in ("cub", "cars", "aircraft"):
            protocol_data: dict[str, object] = {}
            for protocol, strategy in strategies.items():
                v1 = root / "inputs" / dataset / protocol
                v1.mkdir(parents=True)
                image_paths = []
                for index in range(6):
                    image_path = v1 / f"image_{index}.jpg"
                    Image.new("RGB", (8, 8), (index * 10, 0, 0)).save(image_path)
                    image_paths.append(str(image_path))
                (v1 / "paths.txt").write_text("\n".join(image_paths) + "\n", encoding="utf-8")
                np.save(v1 / "labels.npy", np.asarray(["a", "a", "a", "b", "b", "b"]))
                (v1 / "eval_paths.txt").write_text("\n".join(image_paths[:2]) + "\n", encoding="utf-8")
                np.save(v1 / "eval_labels.npy", np.asarray(["a", "b"]))
                (v1 / "resolved_config.yaml").write_text(
                    "feature:\n  backend: dinov2_vitb14\nlora:\n  rank: 2\n  target_modules: qkv\nlora_train: {}\n",
                    encoding="utf-8",
                )
                manifest = v1 / "noise_manifest.csv"
                with manifest.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "index",
                            "source_train_index",
                            "split",
                            "path",
                            "web_label",
                            "noisy_label",
                            "noise_strategy",
                            "noise_ratio",
                            "noise_seed",
                        ],
                    )
                    writer.writeheader()
                    for index, image_path in enumerate(image_paths):
                        observed = "a" if index < 3 else "b"
                        writer.writerow(
                            {
                                "index": str(index),
                                "source_train_index": "" if protocol == "cyclic" else str(index),
                                "split": "train",
                                "path": image_path,
                                "web_label": observed,
                                "noisy_label": "" if protocol == "cyclic" else observed,
                                "noise_strategy": strategy,
                                "noise_ratio": "0.4",
                                "noise_seed": "42",
                            }
                        )
                validation = root / "validation" / dataset / protocol
                validation.mkdir(parents=True, exist_ok=True)
                with (validation / "validation_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["index", "path", "clean_label", "partition"])
                    writer.writeheader()
                    for index, image_path in enumerate(image_paths):
                        writer.writerow({"index": index, "path": image_path, "clean_label": "a" if index < 3 else "b", "partition": "validation" if index in {0, 3} else "training_pool"})
                (validation / "validation_manifest.json").write_text("{}\n", encoding="utf-8")
                reference = None
                if protocol in {"idn", "gccn"}:
                    reference = root / "reference_126" / dataset / protocol
                    reference.mkdir(parents=True)
                    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
                    (reference / "resolved_config.yaml").write_text(yaml.safe_dump({"checkpoint_validation": {"noise_index_sha256": digest}}), encoding="utf-8")
                    for name in ("validation_manifest.csv", "validation_manifest.json"):
                        (reference / name).write_bytes((validation / name).read_bytes())
                protocol_data[protocol] = {
                    "input_dir": str(v1),
                    "noise_index": str(manifest),
                    "validation_dir": str(validation),
                    "reference_126_training_root": None if reference is None else str(reference),
                }
            datasets[dataset] = {"display_name": dataset, "expected_classes": 2, "path_maps": [], "protocols": protocol_data}
        config_path = root / "ido_batch.yaml"
        payload = {
            "schema_version": 1,
            "runtime": {"python_executable": sys.executable},
            "output": {"root": str(root / "outputs"), "instrumentation_root": str(root / "instrumentation")},
            "protocol": {"protocol_frozen": protocol_frozen},
            "formal": {"epochs": 30, "stage1_epochs": 10, "validation_seed": 20250726, "bmm_mode": bmm_mode},
            "datasets": datasets,
        }
        config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return config_path

    def _materialize_complete_task(self, task, config, score: float) -> None:
        from gcdd.ido import ido_adaptation_name, ido_bmm_scope

        task.run_dir.mkdir(parents=True, exist_ok=True)
        task.run_root.mkdir(parents=True, exist_ok=True)
        (task.run_root / "resolved_config.yaml").write_text(
            "feature:\n  backend: dinov2_vitb14\nlora:\n  rank: 2\n  target_modules: qkv\n",
            encoding="utf-8",
        )
        for name in ("validation_manifest.csv", "validation_manifest.json"):
            (task.run_root / name).write_bytes((task.input.validation_dir / name).read_bytes())
        contract = materialize_task_contract(config, task)
        contract_sha256 = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        (task.run_dir / "checkpoints").mkdir(exist_ok=True)
        common = {
            "method": ido_adaptation_name(config.bmm_mode),
            "ido_adaptation": ido_adaptation_name(config.bmm_mode),
            "seed": task.seed,
            "classes": ["a", "b"],
            "state_dict": self._valid_controlled_state_dict(),
            "checkpoint_protocol": "fixed_clean_validation_v1",
            "stage1_epochs": int(config.stage1_epochs or config.epochs // 3),
            "stage2_epochs": int(config.epochs - int(config.stage1_epochs or config.epochs // 3)),
            "bmm_mode": config.bmm_mode,
            "bmm_scope": ido_bmm_scope(config.bmm_mode),
            "task_contract_sha256": contract_sha256,
            "task_id": task.task_id,
            "experiment_signature": contract["experiment_signature"],
            **self._valid_lora_structure_metadata(),
        }
        stage1_wrong_event = np.asarray([0, 0, 1, 0, 0, 1], dtype=np.int64)
        torch.save({**common, "wrong_event": stage1_wrong_event, "base_stage_epoch": 1}, task.run_dir / "checkpoints" / "stage1_base.pt")
        torch.save({**common, "best_epoch": 20, "best_top1": score}, task.run_dir / "checkpoints" / "best_val.pt")
        torch.save({**common, "final_epoch": config.epochs}, task.run_dir / "checkpoints" / "last.pt")
        with (task.run_dir / "stage1_wrong_event.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "path", "observed_label", "wrong_event"])
            writer.writeheader()
            paths = (task.input.input_dir / "paths.txt").read_text(encoding="utf-8").splitlines()
            labels = np.load(task.input.input_dir / "labels.npy", allow_pickle=True).astype(str)
            for index in (1, 2, 4, 5):
                writer.writerow(
                    {
                        "index": index,
                        "path": paths[index],
                        "observed_label": labels[index],
                        "wrong_event": int(stage1_wrong_event[index]),
                    }
                )
        with (task.run_dir / "bmm_class_fits.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = [
                "bmm_mode", "bmm_scope", "fit_round", "stage2_epoch", "observed_label_id", "observed_label", "sample_count",
                "unique_wrong_event_count", "wrong_event_min", "wrong_event_max", "bmm_status", "bmm_reason",
                "bmm_fit_iterations", "alpha_1", "beta_1", "weight_1", "alpha_2", "beta_2", "weight_2",
                "component_1_normalized_mean", "component_2_normalized_mean",
                "component_1_soft_assignment", "component_1_kish_ess", "component_1_weight_times_n",
                "component_1_max_responsibility", "component_1_responsibility_ge_half_count",
                "component_2_soft_assignment", "component_2_kish_ess", "component_2_weight_times_n",
                "component_2_max_responsibility", "component_2_responsibility_ge_half_count",
                "posterior_1_min", "posterior_1_max", "posterior_2_min", "posterior_2_max",
                "cdf_1_min", "cdf_1_max", "cdf_2_min", "cdf_2_max",
                "posterior_mass_sum", "posterior_mass_deficit", "posterior_mass_min", "posterior_mass_max",
                "bmm_prediction_clipped_count",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for stage2_epoch in range(20):
                fit_specs = [("", "", 4)] if config.bmm_mode == "global" else [(0, "a", 2), (1, "b", 2)]
                for label_id, label, sample_count in fit_specs:
                    writer.writerow(
                        {
                            "bmm_mode": config.bmm_mode,
                            "bmm_scope": ido_bmm_scope(config.bmm_mode),
                            "fit_round": "stage2_initial" if stage2_epoch == 0 else "stage2_epoch_update",
                            "stage2_epoch": stage2_epoch,
                            "observed_label_id": label_id,
                            "observed_label": label,
                            "sample_count": sample_count,
                            "unique_wrong_event_count": 2,
                            "wrong_event_min": 0.0,
                            "wrong_event_max": 1.0,
                            "bmm_status": "fitted",
                            "bmm_reason": "",
                            "bmm_fit_iterations": 20 if stage2_epoch == 0 else 2,
                            "alpha_1": 2.0,
                            "beta_1": 4.0,
                            "weight_1": 0.5,
                            "alpha_2": 4.0,
                            "beta_2": 2.0,
                            "weight_2": 0.5,
                            "component_1_normalized_mean": 1.0 / 3.0,
                            "component_2_normalized_mean": 2.0 / 3.0,
                            "component_1_soft_assignment": sample_count / 2.0,
                            "component_1_kish_ess": sample_count / 2.0,
                            "component_1_weight_times_n": sample_count / 2.0,
                            "component_1_max_responsibility": 0.5,
                            "component_1_responsibility_ge_half_count": sample_count,
                            "component_2_soft_assignment": sample_count / 2.0,
                            "component_2_kish_ess": sample_count / 2.0,
                            "component_2_weight_times_n": sample_count / 2.0,
                            "component_2_max_responsibility": 0.5,
                            "component_2_responsibility_ge_half_count": sample_count,
                            "posterior_1_min": 0.5,
                            "posterior_1_max": 0.5,
                            "posterior_2_min": 0.5,
                            "posterior_2_max": 0.5,
                            "cdf_1_min": 0.2,
                            "cdf_1_max": 0.8,
                            "cdf_2_min": 0.2,
                            "cdf_2_max": 0.8,
                            "posterior_mass_sum": sample_count,
                            "posterior_mass_deficit": 0.0,
                            "posterior_mass_min": 1.0,
                            "posterior_mass_max": 1.0,
                            "bmm_prediction_clipped_count": 0,
                        }
                    )
        with (task.run_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch"])
            writer.writeheader()
            writer.writerows({"epoch": epoch} for epoch in range(1, 31))
        (task.run_dir / "selection_policy.json").write_text(
            json.dumps(
                {
                    "method_key": "ido",
                    "ido_bmm_mode": config.bmm_mode,
                    "ido_bmm_scope": ido_bmm_scope(config.bmm_mode),
                    "ido_task_contract_sha256": contract_sha256,
                    "ido_experiment_signature": contract["experiment_signature"],
                }
            ),
            encoding="utf-8",
        )
        (task.run_dir / "result.json").write_text(
            json.dumps(
                {
                    "method_key": "ido",
                    "method": ido_adaptation_name(config.bmm_mode),
                    "seed": task.seed,
                    "bmm_mode": config.bmm_mode,
                    "bmm_scope": ido_bmm_scope(config.bmm_mode),
                    "bmm_degenerate_policy": "fail_closed_record_and_abort",
                    "d1_recording_status": "complete",
                    "stage1_epochs": config.stage1_epochs,
                    "stage2_epochs": config.epochs - int(config.stage1_epochs or 0),
                    "checkpoint_protocol": "fixed_clean_validation_v1",
                    "official_test_evaluation": "validation_selected_only",
                    "validation_selected_test_top5": score,
                    "best_top1": score,
                    "best_top5": score,
                    "best_epoch": 20,
                    "train_samples": 4,
                    "validation_selected_test_top1": score,
                    "ido_task_id": task.task_id,
                    "ido_task_contract_sha256": contract_sha256,
                    "ido_experiment_signature": contract["experiment_signature"],
                }
            ),
            encoding="utf-8",
        )

    def test_plan_has_exact_cyclic_idn_gccn_and_complete_27_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            self.assertEqual(9, len(build_plan(config, ("cyclic",), attempt=1)))
            self.assertEqual(18, len(build_plan(config, ("idn", "gccn"), attempt=1)))
            all_tasks = build_plan(config, ("cyclic", "idn", "gccn"), attempt=1)
            self.assertEqual(27, len(all_tasks))
            self.assertEqual(27, len({task.task_id for task in all_tasks}))
            findings = preflight(config, all_tasks)
            self.assertTrue(all(item["ok"] for item in findings), findings)

    def test_global_mode_has_a_distinct_contract_signature_and_completion_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classwise = load_config(self._write_fixture_config(root / "classwise", bmm_mode="classwise"))
            global_mode = load_config(self._write_fixture_config(root / "global", bmm_mode="global"))
            classwise_task = build_plan(classwise, ("cyclic",), attempt=1)[0]
            global_task = build_plan(global_mode, ("cyclic",), attempt=1)[0]
            self.assertNotEqual(
                build_task_contract(classwise, classwise_task)["experiment_signature"],
                build_task_contract(global_mode, global_task)["experiment_signature"],
            )
            command = build_training_command(global_mode, global_task)
            self.assertIn("--ido-bmm-mode", command)
            self.assertEqual("global", command[command.index("--ido-bmm-mode") + 1])
            self._materialize_complete_task(global_task, global_mode, score=0.2)
            complete, detail = is_complete(global_task, global_mode)
            self.assertTrue(complete, detail)
            with (global_task.run_dir / "bmm_class_fits.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(20, len(rows))
            self.assertTrue(all(row["observed_label_id"] == "" for row in rows))

    def test_seed_contract_location_matches_the_single_run_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            task = build_plan(config, ("cyclic",), attempt=1)[0]
            materialize_validation_manifest(task)
            expected = materialize_task_contract(config, task)
            loaded = load_ido_task_contract(
                task.contract_path,
                input_dir=task.input.input_dir,
                noise_index=task.input.noise_index,
                output_dir=task.run_root,
                methods=["ido"],
                seeds=[task.seed],
                validation_seed=config.validation_seed,
                stage_budget=(10, 20),
                path_maps=[],
                official_test_selected_only=True,
                posthoc_oracle_test=False,
            )
            self.assertEqual(expected["task_signature"], loaded["task_signature"])

    def test_dry_run_never_launches_subprocess_and_execute_requires_frozen_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_fixture_config(Path(tmp), protocol_frozen=False)
            with mock.patch.object(sys, "argv", ["run_ido_all.py", "--config", str(config_path), "--dry-run"]), mock.patch(
                "gcdd.ido_batch.subprocess.call"
            ) as call:
                main_for_protocols(protocols=("cyclic", "idn", "gccn"), description="test")
            call.assert_not_called()
            with mock.patch.object(sys, "argv", ["run_ido_all.py", "--config", str(config_path), "--execute"]):
                with self.assertRaises(PermissionError):
                    main_for_protocols(protocols=("cyclic", "idn", "gccn"), description="test")

    def test_incomplete_is_not_complete_and_3seed_summary_uses_sample_sd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            tasks = build_plan(config, ("cyclic",), attempt=1)
            for index, task in enumerate(tasks):
                self._materialize_complete_task(task, config, score=(index % 3 + 1) / 10.0)
            self.assertTrue(is_complete(tasks[0], config)[0])
            (tasks[0].run_dir / "checkpoints" / "last.pt").unlink()
            self.assertFalse(is_complete(tasks[0], config)[0])
            self._materialize_complete_task(tasks[0], config, score=0.1)
            (tasks[0].run_root / "validation_manifest.json").write_text('{"different": true}\n', encoding="utf-8")
            self.assertFalse(is_complete(tasks[0], config)[0])
            (tasks[0].run_root / "validation_manifest.json").write_bytes(
                (tasks[0].input.validation_dir / "validation_manifest.json").read_bytes()
            )
            self.assertTrue(is_complete(tasks[0], config)[0])
            summarize_all(config, ("cyclic",))
            summary_path = config.output_root / "orchestration" / "ido_3seed_summary.csv"
            with summary_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(3, len(rows))
            self.assertTrue(all(row["status"] == "COMPLETE_3_SEEDS" for row in rows))
            self.assertEqual("0.2", rows[0]["mean_validation_selected_official_test_top1"])
            self.assertEqual("0.1", rows[0]["sample_sd_validation_selected_official_test_top1"])

    def test_completion_rejects_corrupt_checkpoint_missing_score_and_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            task = build_plan(config, ("cyclic",), attempt=1)[0]
            self._materialize_complete_task(task, config, score=0.2)
            (task.run_dir / "checkpoints" / "best_val.pt").write_bytes(b"not a torch checkpoint")
            self.assertFalse(is_complete(task, config)[0])
            self._materialize_complete_task(task, config, score=0.2)
            result_path = task.run_dir / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result.pop("validation_selected_test_top1")
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertFalse(is_complete(task, config)[0])
            self._materialize_complete_task(task, config, score=0.2)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["ido_experiment_signature"] = "wrong"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            self.assertFalse(is_complete(task, config)[0])

    def test_completion_rejects_semantically_invalid_trainable_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            task = build_plan(config, ("cyclic",), attempt=1)[0]
            self._materialize_complete_task(task, config, score=0.2)
            checkpoint_path = task.run_dir / "checkpoints" / "best_val.pt"
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            payload["state_dict"] = {"random_key_unrelated_to_dinov2": torch.ones(1)}
            torch.save(payload, checkpoint_path)
            complete, detail = is_complete(task, config)
            self.assertFalse(complete)
            self.assertIn("structural audit", detail)

    def test_trainable_state_audit_rejects_unpaired_or_rank_incompatible_lora(self) -> None:
        unpaired = self._valid_controlled_state_dict()
        del unpaired["backbone.blocks.0.attn.qkv.lora_b.weight"]
        error, _ = validate_ido_trainable_state_dict(
            torch=torch,
            state=unpaired,
            expected_classes=2,
            expected_lora_rank=2,
        )
        self.assertIn("key set does not equal", error)
        incompatible_rank = self._valid_controlled_state_dict()
        incompatible_rank["backbone.blocks.0.attn.qkv.lora_a.weight"] = torch.ones(3, 768)
        error, _ = validate_ido_trainable_state_dict(
            torch=torch,
            state=incompatible_rank,
            expected_classes=2,
            expected_lora_rank=2,
        )
        self.assertIn("shape incompatible", error)

    def test_f5_checkpoint_structure_contract_rejects_every_listed_invalid_lora_layout(self) -> None:
        valid = self._valid_controlled_state_dict()
        error, _ = validate_ido_trainable_state_dict(
            torch=torch,
            state=valid,
            expected_classes=2,
            expected_lora_rank=2,
        )
        self.assertEqual("", error)

        def replace_prefix(state: dict[str, torch.Tensor], old: str, new: str) -> dict[str, torch.Tensor]:
            return {key.replace(old, new): value for key, value in state.items()}

        only_block_zero = {
            key: value
            for key, value in valid.items()
            if key.startswith("head.") or key.startswith("backbone.blocks.0.")
        }
        missing_last_block = dict(valid)
        del missing_last_block["backbone.blocks.11.attn.qkv.lora_a.weight"]
        del missing_last_block["backbone.blocks.11.attn.qkv.lora_b.weight"]
        extra_block = dict(valid)
        extra_block["backbone.blocks.999.attn.qkv.lora_a.weight"] = torch.zeros(2, 768)
        extra_block["backbone.blocks.999.attn.qkv.lora_b.weight"] = torch.zeros(2304, 2)
        wrong_target = replace_prefix(valid, ".blocks.0.attn.qkv.", ".blocks.0.attn.proj.")
        unpaired = dict(valid)
        del unpaired["backbone.blocks.0.attn.qkv.lora_b.weight"]
        cases = {
            "nonexistent_block_999": replace_prefix(valid, ".blocks.0.", ".blocks.999."),
            "only_block_zero": only_block_zero,
            "missing_block_eleven": missing_last_block,
            "extra_block": extra_block,
            "wrong_target_module": wrong_target,
            "unpaired_lora": unpaired,
        }
        for name, state in cases.items():
            with self.subTest(name=name):
                error, _ = validate_ido_trainable_state_dict(
                    torch=torch,
                    state=state,
                    expected_classes=2,
                    expected_lora_rank=2,
                )
                self.assertIn("key set does not equal", error)

    def test_f5_is_complete_fail_closed_for_every_invalid_lora_layout(self) -> None:
        def replace_prefix(state: dict[str, torch.Tensor], old: str, new: str) -> dict[str, torch.Tensor]:
            return {key.replace(old, new): value for key, value in state.items()}

        def only_block_zero(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {
                key: value
                for key, value in state.items()
                if key.startswith("head.") or key.startswith("backbone.blocks.0.")
            }

        def missing_block_eleven(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            result = dict(state)
            del result["backbone.blocks.11.attn.qkv.lora_a.weight"]
            del result["backbone.blocks.11.attn.qkv.lora_b.weight"]
            return result

        def extra_block(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            result = dict(state)
            result["backbone.blocks.999.attn.qkv.lora_a.weight"] = torch.zeros(2, 768)
            result["backbone.blocks.999.attn.qkv.lora_b.weight"] = torch.zeros(2304, 2)
            return result

        def unpaired(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            result = dict(state)
            del result["backbone.blocks.0.attn.qkv.lora_b.weight"]
            return result

        transformations = {
            "nonexistent_block_999": lambda state: replace_prefix(state, ".blocks.0.", ".blocks.999."),
            "only_block_zero": only_block_zero,
            "missing_block_eleven": missing_block_eleven,
            "extra_block": extra_block,
            "wrong_target_module": lambda state: replace_prefix(
                state, ".blocks.0.attn.qkv.", ".blocks.0.attn.proj."
            ),
            "unpaired_lora": unpaired,
        }
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            task = build_plan(config, ("cyclic",), attempt=1)[0]
            for name, transform in transformations.items():
                with self.subTest(name=name):
                    self._materialize_complete_task(task, config, score=0.2)
                    for checkpoint_name in ("stage1_base.pt", "best_val.pt", "last.pt"):
                        checkpoint_path = task.run_dir / "checkpoints" / checkpoint_name
                        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                        payload["state_dict"] = transform(dict(payload["state_dict"]))
                        torch.save(payload, checkpoint_path)
                    complete, detail = is_complete(task, config)
                    self.assertFalse(complete)
                    self.assertIn("structural audit", detail)

    def test_completion_rejects_truncated_or_incoherent_stage1_and_bmm_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            task = build_plan(config, ("cyclic",), attempt=1)[0]
            self._materialize_complete_task(task, config, score=0.2)
            stage1 = task.run_dir / "stage1_wrong_event.csv"
            stage1.write_text("index,path,observed_label,wrong_event\n", encoding="utf-8")
            complete, detail = is_complete(task, config)
            self.assertFalse(complete)
            self.assertIn("row count", detail)

            self._materialize_complete_task(task, config, score=0.2)
            with stage1.open(encoding="utf-8", newline="") as handle:
                stage1_rows = list(csv.DictReader(handle))
            stage1_rows[1]["index"] = stage1_rows[0]["index"]
            with stage1.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(stage1_rows[0]))
                writer.writeheader()
                writer.writerows(stage1_rows)
            complete, detail = is_complete(task, config)
            self.assertFalse(complete)
            self.assertIn("stable-index", detail)

            self._materialize_complete_task(task, config, score=0.2)
            bmm_path = task.run_dir / "bmm_class_fits.csv"
            with bmm_path.open(encoding="utf-8", newline="") as handle:
                bmm_rows = list(csv.DictReader(handle))
            with bmm_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(bmm_rows[0]))
                writer.writeheader()
                writer.writerows(bmm_rows[:-1])
            complete, detail = is_complete(task, config)
            self.assertFalse(complete)
            self.assertIn("required class-wise", detail)

    def test_contract_covers_runtime_dependency_closure_and_changes_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            task = build_plan(config, ("cyclic",), attempt=1)[0]
            baseline = build_task_contract(config, task)
            code_sha = baseline["experiment"]["code_sha256"]
            self.assertTrue(set(IDO_RUNTIME_CODE_PATHS) <= set(code_sha))
            self.assertIn("gcdd/lora_dynamic.py", code_sha)
            self.assertIn("gcdd/lora_noisy_baselines.py", code_sha)
            self.assertIn("revision_experiments/selection_logging.py", code_sha)
            from gcdd import ido_batch

            original_sha256_file = ido_batch.sha256_file
            changed_module = (ido_batch.ROOT / "gcdd" / "lora_dynamic.py").resolve()

            def altered_sha256(path: Path) -> str:
                if Path(path).resolve() == changed_module:
                    return "f" * 64
                return original_sha256_file(path)

            with mock.patch("gcdd.ido_batch.sha256_file", side_effect=altered_sha256):
                changed = build_task_contract(config, task)
            self.assertNotEqual(baseline["experiment_signature"], changed["experiment_signature"])

    def test_preflight_rejects_idn_reference_validation_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            task = build_plan(config, ("idn",), attempt=1)[0]
            assert task.input.reference_126_training_root is not None
            (task.input.reference_126_training_root / "validation_manifest.csv").write_text(
                "incompatible reference validation manifest\n", encoding="utf-8"
            )
            findings = preflight(config, [task])
            self.assertFalse(findings[0]["ok"])
            self.assertIn("validation_manifest.csv SHA256 differs", findings[0]["detail"])

    def test_preflight_rejects_noise_index_path_label_and_source_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            cyclic_task = build_plan(config, ("cyclic",), attempt=1)[0]
            with cyclic_task.input.noise_index.open(encoding="utf-8", newline="") as handle:
                cyclic_rows = list(csv.DictReader(handle))
            for row in cyclic_rows:
                row["web_label"] = "wrong-observed-label"
            with cyclic_task.input.noise_index.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(cyclic_rows[0]))
                writer.writeheader()
                writer.writerows(reversed(cyclic_rows))
            cyclic_findings = preflight(config, [cyclic_task])
            self.assertFalse(cyclic_findings[0]["ok"])
            self.assertIn("labels.npy disagrees", cyclic_findings[0]["detail"])

            idn_task = build_plan(config, ("idn",), attempt=1)[0]
            with idn_task.input.noise_index.open(encoding="utf-8", newline="") as handle:
                idn_rows = list(csv.DictReader(handle))
            idn_rows[1]["source_train_index"] = "0"
            with idn_task.input.noise_index.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(idn_rows[0]))
                writer.writeheader()
                writer.writerows(idn_rows)
            idn_findings = preflight(config, [idn_task])
            self.assertFalse(idn_findings[0]["ok"])
            self.assertIn("source_train_index", idn_findings[0]["detail"])

    def test_new_attempt_reuses_only_one_prior_completed_seed_with_matching_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            first = {
                task.seed: task
                for task in build_plan(config, ("cyclic",), attempt=1)
                if task.dataset.key == "cub"
            }
            second = {
                task.seed: task
                for task in build_plan(config, ("cyclic",), attempt=2)
                if task.dataset.key == "cub"
            }
            self._materialize_complete_task(first[1], config, score=0.1)
            self.assertEqual(first[1], find_reusable_completed_task(config, second[1]))
            self.assertIsNone(find_reusable_completed_task(config, second[42]))
            self._materialize_complete_task(second[42], config, score=0.2)
            self._materialize_complete_task(second[88], config, score=0.3)
            for completed_task in (first[1], second[42], second[88]):
                complete, detail = is_complete(completed_task, config)
                self.assertTrue(complete, detail)
            summarize_all(config, ("cyclic",))
            with (config.output_root / "orchestration" / "ido_3seed_summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            cub = next(row for row in rows if row["dataset_key"] == "cub")
            self.assertEqual("COMPLETE_3_SEEDS", cub["status"], rows)
            self.assertEqual("attempt-001", cub["seed1_attempt"])
            self.assertEqual("attempt-002", cub["seed42_attempt"])
            self.assertEqual("attempt-002", cub["seed88_attempt"])

    def test_summary_never_merges_a_seed_with_a_mismatched_experiment_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(self._write_fixture_config(Path(tmp)))
            first = {task.seed: task for task in build_plan(config, ("cyclic",), attempt=1)}
            second = {task.seed: task for task in build_plan(config, ("cyclic",), attempt=2)}
            self._materialize_complete_task(first[1], config, score=0.1)
            self._materialize_complete_task(second[42], config, score=0.2)
            self._materialize_complete_task(second[88], config, score=0.3)
            result_path = second[42].run_dir / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["ido_experiment_signature"] = "mismatched"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            summarize_all(config, ("cyclic",))
            with (config.output_root / "orchestration" / "ido_3seed_summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            cyclic_rows = [row for row in rows if row["dataset_key"] == "cub"]
            self.assertTrue(cyclic_rows)
            self.assertTrue(all(row["status"] != "COMPLETE_3_SEEDS" for row in cyclic_rows))


if __name__ == "__main__":
    unittest.main()
