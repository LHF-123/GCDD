"""CPU integration tests for the minimal PGDF D1/D2 call boundary.

These tests exercise adapters and observer contracts with synthetic arrays.
They intentionally do not instantiate DINOv2 or claim a GPU training run.
"""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

import numpy as np

from gcdd.lora_dynamic import train_dynamic_loss_lora
from revision_experiments.offline_diagnostics import DiagnosticError
from revision_experiments.profiling import ProfilingError, ProfilingMode, RunProfiler
from revision_experiments.selection_logging import SAMPLE_FIELDS, SelectionLogError, SelectionLogger, read_selection_bundle
from revision_experiments.training_integration import (
    PGDF_ACTUAL_FALLBACK_RULE,
    SelectionRunMetadata,
    actual_pgdf_fallback_labels,
    build_code_provenance,
    build_experiment_id,
    build_selection_sample_records,
    protect_monitored_training_output_dir,
    profiling_memory_phase,
    profiling_stage,
    record_optimizer_and_scheduler_step,
    record_selection_update,
    write_instrumentation_outputs,
)


class TrainingIntegrationTests(unittest.TestCase):
    """Synthetic CPU tests; no model forward, optimizer, or CUDA invocation."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = ["train/a0.jpg", "train/a1.jpg", "train/b0.jpg", "train/b1.jpg"]
        self.labels = np.asarray(["a", "a", "b", "b"])
        self.metadata = SelectionRunMetadata("synthetic", "cyclic-asym40", 42, 1)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _strict_arrays(self) -> dict[str, np.ndarray]:
        return {
            "candidate": np.asarray([True, False, True, True]),
            "losses": np.asarray([0.1, np.nan, 0.2, 0.3], dtype=np.float32),
            "prototype_scores": np.asarray([0.6, np.nan, 0.9, 0.4], dtype=np.float32),
            "loss_mask": np.asarray([True, False, True, False]),
            "prototype_mask": np.asarray([True, False, True, False]),
            "final": np.asarray([True, False, True, False]),
        }

    def _record_strict_update(self, logger: SelectionLogger, *, epoch: int = 5) -> None:
        data = self._strict_arrays()
        record_selection_update(
            logger,
            train_paths=self.paths,
            observed_labels=self.labels,
            candidate_mask=data["candidate"],
            losses=data["losses"],
            prototype_scores=data["prototype_scores"],
            loss_candidate_mask=data["loss_mask"],
            prototype_candidate_mask=data["prototype_mask"],
            final_selected_mask=data["final"],
            epoch=epoch,
            selection_strategy="loss_and_proto",
            model_state_identifier=f"state-{epoch}",
            metadata=self.metadata,
        )

    def test_01_train_function_exposes_optional_observers_with_default_none(self) -> None:
        signature = inspect.signature(train_dynamic_loss_lora)
        self.assertIsNone(signature.parameters["profiler"].default)
        self.assertIsNone(signature.parameters["selection_logger"].default)
        self.assertIsNone(signature.parameters["selection_log_metadata"].default)

    def test_02_disabled_observer_contexts_are_noops_and_write_nothing(self) -> None:
        with profiling_memory_phase(None, "training"):
            with profiling_stage(None, "optimizer_training"):
                pass
        self.assertEqual(list(self.root.iterdir()), [])

    def test_02b_experiment_id_contains_non_metric_run_metadata(self) -> None:
        identifier = build_experiment_id(
            dataset="CUB-200-2011",
            noise_protocol="cyclic asym40",
            noise_seed=42,
            method="pgdf_dynamic_proto",
            training_seed=88,
        )
        self.assertEqual(identifier, "CUB-200-2011_cyclic-asym40_noise42_pgdf_dynamic_proto_seed88")

    def test_03_full_pool_stable_ids_and_scores_are_index_aligned(self) -> None:
        data = self._strict_arrays()
        records = build_selection_sample_records(
            train_paths=self.paths,
            observed_labels=self.labels,
            candidate_mask=data["candidate"],
            losses=data["losses"],
            prototype_scores=data["prototype_scores"],
            loss_candidate_mask=data["loss_mask"],
            prototype_candidate_mask=data["prototype_mask"],
            final_selected_mask=data["final"],
            epoch=5,
            selection_strategy="loss_and_proto",
            model_state_identifier="state-5",
            metadata=self.metadata,
            experiment_id="synthetic-run",
        )
        self.assertEqual([record.stable_sample_id for record in records], ["train_index:0", "train_index:2", "train_index:3"])
        self.assertEqual([record.sample_path for record in records], [self.paths[0], self.paths[2], self.paths[3]])
        np.testing.assert_allclose([record.loss_score for record in records], [0.1, 0.2, 0.3])
        np.testing.assert_allclose([record.prototype_score for record in records], [0.6, 0.9, 0.4])

    def test_04_invalid_full_pool_alignment_is_rejected(self) -> None:
        data = self._strict_arrays()
        with self.assertRaises(ValueError):
            build_selection_sample_records(
                train_paths=self.paths,
                observed_labels=self.labels,
                candidate_mask=data["candidate"],
                losses=data["losses"][:3],
                prototype_scores=data["prototype_scores"],
                loss_candidate_mask=data["loss_mask"],
                prototype_candidate_mask=data["prototype_mask"],
                final_selected_mask=data["final"],
                epoch=5,
                selection_strategy="loss_and_proto",
                model_state_identifier="state-5",
                metadata=self.metadata,
                experiment_id="synthetic-run",
            )

    def test_05_strict_intersection_update_round_trips_with_full_pool_count(self) -> None:
        logger = SelectionLogger("synthetic-run")
        self._record_strict_update(logger)
        bundle = read_selection_bundle(logger.finalize(self.root / "outputs_major_revision"))
        self.assertEqual(bundle.update_rows[0]["full_training_pool_size"], 3)
        self.assertEqual(bundle.update_rows[0]["selected_total"], 2)
        self.assertEqual(bundle.update_rows[0]["fallback_count"], 0)
        self.assertEqual(len(bundle.samples), 3)

    def test_06_actual_pgdf_fallback_is_declared_without_changing_selection(self) -> None:
        candidate = np.asarray([True, True, True, True])
        loss_mask = np.asarray([True, False, True, False])
        prototype_mask = np.asarray([False, True, True, False])
        final = np.asarray([False, True, True, False])
        self.assertEqual(actual_pgdf_fallback_labels(self.labels, candidate, loss_mask, prototype_mask), {"a"})
        logger = SelectionLogger("fallback-run")
        record_selection_update(
            logger,
            train_paths=self.paths,
            observed_labels=self.labels,
            candidate_mask=candidate,
            losses=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            prototype_scores=np.asarray([0.1, 0.8, 0.9, 0.2], dtype=np.float32),
            loss_candidate_mask=loss_mask,
            prototype_candidate_mask=prototype_mask,
            final_selected_mask=final,
            epoch=5,
            selection_strategy="loss_and_proto",
            model_state_identifier="state-5",
            metadata=self.metadata,
        )
        bundle = read_selection_bundle(logger.finalize(self.root / "outputs_major_revision"))
        class_a = next(row for row in bundle.class_rows if row["observed_label"] == "a")
        self.assertTrue(class_a["fallback_triggered"])
        self.assertEqual(bundle.update_rows[0]["fallback_rule"], PGDF_ACTUAL_FALLBACK_RULE)

    def test_07_two_updates_remain_distinct_and_schema_contains_no_oracle(self) -> None:
        logger = SelectionLogger("synthetic-run")
        self._record_strict_update(logger, epoch=5)
        self._record_strict_update(logger, epoch=10)
        bundle = read_selection_bundle(logger.finalize(self.root / "outputs_major_revision"))
        self.assertEqual([row["update_epoch"] for row in bundle.update_rows], [5, 10])
        self.assertNotIn("clean_label", SAMPLE_FIELDS)
        self.assertNotIn("is_noisy", SAMPLE_FIELDS)

    def test_08_profiling_records_existing_step_calls_and_amp_skip_semantics(self) -> None:
        profiler = RunProfiler("profile", ProfilingMode.BASIC, device_type="cpu")
        with profiling_stage(profiler, "total_training"):
            with profiling_memory_phase(profiler, "training"):
                with profiling_stage(profiler, "epoch", phase="epoch_1"):
                    with profiling_stage(profiler, "optimizer_training", phase="epoch_1"):
                        profiler.record_training_batch()
                        profiler.record_training_batch()
                        self.assertTrue(
                            record_optimizer_and_scheduler_step(
                                profiler, epoch=1, amp_enabled=False, scale_before=None, scale_after=None
                            )
                        )
                        self.assertFalse(
                            record_optimizer_and_scheduler_step(
                                profiler, epoch=1, amp_enabled=True, scale_before=1024.0, scale_after=512.0
                            )
                        )
            with profiling_memory_phase(profiler, "selection"):
                with profiling_stage(profiler, "selection_update_total", phase="epoch_1"):
                    with profiling_stage(profiler, "selection_loss_forward", phase="epoch_1"):
                        pass
                    with profiling_stage(profiler, "selection_cls_forward", phase="epoch_1"):
                        pass
                profiler.record_selection_interval_steps(1, 1)
        payload = profiler.to_dict()
        self.assertEqual(payload["optimizer_steps_attempted"], 2)
        self.assertEqual(payload["optimizer_steps_actual"], 1)
        self.assertEqual(payload["scheduler_steps_actual"], 2)
        self.assertEqual(payload["training_batches_seen"], 2)
        self.assertEqual(payload["selection_interval_steps"], {1: 1})
        self.assertEqual(payload["stage_totals"]["selection_loss_forward"]["count"], 1)
        self.assertEqual(payload["stage_totals"]["selection_cls_forward"]["count"], 1)
        self.assertIsNone(payload["memory"]["phases"]["training"]["allocated_bytes"])

    def test_09_multiple_selection_updates_accumulate_without_pooling_stages(self) -> None:
        profiler = RunProfiler("profile", ProfilingMode.BASIC)
        for epoch in (5, 10):
            with profiling_stage(profiler, "selection_update_total", phase=f"epoch_{epoch}"):
                with profiling_stage(profiler, "selection_loss_forward", phase=f"epoch_{epoch}"):
                    pass
                with profiling_stage(profiler, "selection_cls_forward", phase=f"epoch_{epoch}"):
                    pass
        payload = profiler.to_dict()
        self.assertEqual(payload["stage_totals"]["selection_update_total"]["count"], 2)
        self.assertEqual(payload["stage_totals"]["selection_loss_forward"]["count"], 2)
        self.assertEqual(payload["stage_totals"]["selection_cls_forward"]["count"], 2)

    def test_10_observers_share_one_experiment_directory_and_support_independent_modes(self) -> None:
        root = self.root / "outputs_major_revision"
        profiler = RunProfiler("shared-run")
        logger = SelectionLogger("shared-run")
        self._record_strict_update(logger)
        written = write_instrumentation_outputs(
            output_root=root,
            profiler=profiler,
            selection_logger=logger,
            run_metadata={"dataset": "synthetic", "code_commit": "test-commit"},
        )
        self.assertEqual(written["profiling"].parent, written["selection_records"].parent)
        self.assertTrue((root / "shared-run" / "profiling.json").is_file())
        self.assertTrue((root / "shared-run" / "selection_records" / "run_status.json").is_file())
        metadata = json.loads((root / "shared-run" / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["experiment_id"], "shared-run")
        self.assertEqual(metadata["code_commit"], "test-commit")
        profiler_only = RunProfiler("profile-only")
        paths = write_instrumentation_outputs(output_root=root, profiler=profiler_only, selection_logger=None)
        self.assertEqual(set(paths), {"profiling"})
        logger_only = SelectionLogger("selection-only")
        self._record_strict_update(logger_only)
        paths = write_instrumentation_outputs(output_root=root, profiler=None, selection_logger=logger_only)
        self.assertEqual(set(paths), {"selection_records"})

    def test_11_incomplete_training_cannot_write_complete_observer_outputs(self) -> None:
        root = self.root / "outputs_major_revision"
        profiler = RunProfiler("failed-run")
        logger = SelectionLogger("failed-run")
        self._record_strict_update(logger)
        token = profiler.start_stage("epoch")
        with self.assertRaises(ProfilingError):
            write_instrumentation_outputs(output_root=root, profiler=profiler, selection_logger=logger)
        self.assertFalse((root / "failed-run" / "profiling.json").exists())
        self.assertFalse((root / "failed-run" / "selection_records" / "run_status.json").exists())
        profiler.stop_stage(token)

    def test_12_protected_history_is_rejected_without_test_writes(self) -> None:
        profiler = RunProfiler("protected")
        logger = SelectionLogger("protected")
        self._record_strict_update(logger)
        with self.assertRaises(ValueError):
            write_instrumentation_outputs(output_root=self.root / "outputs", profiler=profiler, selection_logger=logger)
        self.assertFalse((self.root / "outputs").exists())

    def test_13_fit_time_is_a_distinct_nested_metric_with_static_training_boundary(self) -> None:
        profiler = RunProfiler("fit-boundary")
        with profiling_stage(profiler, "total_training", phase="train_dynamic_loss_lora"):
            with profiling_stage(profiler, "fit_time", phase="epoch_loop"):
                pass
        payload = profiler.to_dict()
        self.assertEqual(payload["stage_totals"]["total_training"]["count"], 1)
        self.assertEqual(payload["stage_totals"]["fit_time"]["count"], 1)
        self.assertIn("not summed", payload["measurement_notes"]["stage_boundaries"])
        source = inspect.getsource(train_dynamic_loss_lora)
        self.assertLess(source.index('profiler.start_stage("fit_time"'), source.index("for epoch in range"))
        self.assertLess(source.index("profiler.stop_stage(fit_time_token)"), source.index("final_state ="))

    def test_14_monitored_primary_output_protection_is_opt_in_and_write_free(self) -> None:
        protected_candidate = self.root / "outputs"
        with self.assertRaises(ValueError):
            protect_monitored_training_output_dir(protected_candidate, instrumentation_requested=True)
        self.assertFalse(protected_candidate.exists())
        self.assertEqual(
            protect_monitored_training_output_dir(protected_candidate, instrumentation_requested=False),
            protected_candidate,
        )
        runner_source = (Path(__file__).resolve().parents[2] / "scripts" / "run_lora_checkpoint_validation.py").read_text(
            encoding="utf-8"
        )
        self.assertLess(
            runner_source.index("output_dir = protect_monitored_training_output_dir("),
            runner_source.index("ensure_dir(output_dir)"),
        )

    def test_15_empty_logger_preflight_writes_no_complete_profile(self) -> None:
        root = self.root / "outputs_major_revision"
        profiler = RunProfiler("empty-logger")
        logger = SelectionLogger("empty-logger")
        with self.assertRaises(SelectionLogError):
            write_instrumentation_outputs(output_root=root, profiler=profiler, selection_logger=logger)
        self.assertFalse((root / "empty-logger" / "profiling.json").exists())
        self.assertFalse((root / "empty-logger" / "selection_records" / "run_status.json").exists())

    def test_16_incomplete_profiler_preflight_writes_nothing(self) -> None:
        root = self.root / "outputs_major_revision"
        profiler = RunProfiler("active-profiler")
        logger = SelectionLogger("active-profiler")
        self._record_strict_update(logger)
        token = profiler.start_stage("epoch")
        with self.assertRaises(ProfilingError):
            write_instrumentation_outputs(output_root=root, profiler=profiler, selection_logger=logger)
        self.assertFalse((root / "active-profiler" / "profiling.json").exists())
        self.assertFalse((root / "active-profiler" / "selection_records" / "run_status.json").exists())
        profiler.stop_stage(token)

    def test_17_code_provenance_records_git_state_and_exact_source_hashes(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        provenance = build_code_provenance(repository_root)
        self.assertRegex(provenance["git_head"], r"^[0-9a-f]{40}$|^unavailable$")
        self.assertIn(provenance["git_dirty"], (True, False, None))
        self.assertEqual(
            provenance["training_function_sha256"],
            sha256((repository_root / "gcdd" / "lora_dynamic.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            provenance["training_entry_sha256"],
            sha256((repository_root / "scripts" / "run_lora_checkpoint_validation.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            set(provenance["d1_d2_module_sha256"]),
            {
                "revision_experiments/_paths.py",
                "revision_experiments/profiling.py",
                "revision_experiments/selection_logging.py",
                "revision_experiments/offline_diagnostics.py",
                "revision_experiments/training_integration.py",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
