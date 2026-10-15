"""CPU-only contract tests for the revision profiling and diagnostics modules."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from revision_experiments._paths import historical_outputs_root
from revision_experiments.offline_diagnostics import (
    DiagnosticError,
    OracleLabelRecord,
    RunDiagnostics,
    aggregate_across_seeds,
    diagnose_bundle_update,
    diagnose_selection_update,
    tie_aware_auc,
    write_diagnostics_json,
)
from revision_experiments.profiling import ProfilingError, ProfilingMode, RunProfiler
from revision_experiments.selection_logging import (
    SAMPLE_FIELDS,
    SelectionLogError,
    SelectionLogger,
    SelectionSampleRecord,
    read_selection_bundle,
)


class RevisionFoundationTests(unittest.TestCase):
    """Thirty numbered requirements from the CPU test plan, without model calls."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def sample(
        sample_id: str,
        label: str,
        *,
        seed: int = 1,
        epoch: int = 5,
        loss: float | None = 0.2,
        proto: float | None = 0.8,
        loss_candidate: bool | None = True,
        prototype_candidate: bool | None = True,
        final_selected: bool = True,
        selection_strategy: str = "loss_and_proto",
    ) -> SelectionSampleRecord:
        return SelectionSampleRecord(
            experiment_id="synthetic-run",
            dataset="synthetic",
            noise_protocol="cyclic-asym40",
            noise_seed=42,
            training_seed=seed,
            epoch=epoch,
            stable_sample_id=sample_id,
            sample_path=f"train/{sample_id}.jpg",
            observed_label=label,
            loss_score=loss,
            prototype_score=proto,
            loss_candidate=loss_candidate,
            prototype_candidate=prototype_candidate,
            final_selected=final_selected,
            selection_strategy=selection_strategy,
            model_state_identifier="state-epoch-5",
        )

    @staticmethod
    def oracle(samples: list[SelectionSampleRecord], clean_flags: list[bool]) -> list[OracleLabelRecord]:
        return [
            OracleLabelRecord(
                stable_sample_id=sample.stable_sample_id,
                sample_path=sample.sample_path,
                observed_label=sample.observed_label,
                clean_label=sample.observed_label if clean else f"other-{sample.observed_label}",
            )
            for sample, clean in zip(samples, clean_flags)
        ]

    # Profiling requirements 1–10.
    def test_01_normal_start_stop_timing(self) -> None:
        profiler = RunProfiler("profile-01", ProfilingMode.BASIC)
        token = profiler.start_stage("epoch", phase="training")
        duration = profiler.stop_stage(token)
        self.assertIsNotNone(duration)
        self.assertGreaterEqual(duration or -1, 0.0)
        self.assertEqual(profiler.to_dict()["stage_totals"]["epoch"]["count"], 1)

    def test_02_nested_stage_timing_is_recorded_without_sum_claim(self) -> None:
        profiler = RunProfiler("profile-02", ProfilingMode.DETAILED)
        with profiler.stage("total_training", phase="run"):
            with profiler.stage("validation", phase="validation"):
                pass
        payload = profiler.to_dict()
        self.assertEqual(payload["stage_totals"]["total_training"]["count"], 1)
        self.assertEqual(payload["stage_totals"]["validation"]["count"], 1)
        self.assertEqual(len(payload["detailed_events"]), 2)
        self.assertIn("not summed", payload["measurement_notes"]["stage_boundaries"])

    def test_03_illegal_duplicate_start_and_stop_fail(self) -> None:
        profiler = RunProfiler("profile-03")
        token = profiler.start_stage("epoch")
        with self.assertRaises(ProfilingError):
            profiler.start_stage("epoch")
        profiler.stop_stage(token)
        with self.assertRaises(ProfilingError):
            profiler.stop_stage(token)

    def test_04_cpu_memory_is_na_not_zero(self) -> None:
        profiler = RunProfiler("profile-04", device_type="cpu")
        with profiler.memory_phase("training"):
            pass
        payload = profiler.to_dict()
        phase = payload["memory"]["phases"]["training"]
        self.assertIsNone(phase["allocated_bytes"])
        self.assertIsNone(phase["reserved_bytes"])
        self.assertEqual(phase["status"], "not_applicable_cpu_or_cuda_unavailable")

    def test_05_off_mode_has_no_records_or_counters(self) -> None:
        profiler = RunProfiler("profile-05", ProfilingMode.OFF)
        with profiler.stage("epoch"):
            pass
        profiler.record_optimizer_step()
        payload = profiler.to_dict()
        self.assertEqual(payload["recording_status"], "off")
        self.assertEqual(payload["stage_totals"], {})
        self.assertIsNone(payload["optimizer_steps_actual"])

    def test_06_basic_and_detailed_modes_differ_only_in_events(self) -> None:
        basic = RunProfiler("profile-06-basic", ProfilingMode.BASIC)
        detailed = RunProfiler("profile-06-detailed", ProfilingMode.DETAILED)
        with basic.stage("validation"):
            pass
        with detailed.stage("validation"):
            pass
        self.assertEqual(basic.to_dict()["detailed_events"], [])
        self.assertEqual(len(detailed.to_dict()["detailed_events"]), 1)

    def test_07_actual_optimizer_and_scheduler_steps_count(self) -> None:
        profiler = RunProfiler("profile-07")
        profiler.record_optimizer_step(actual_update=True, epoch=1)
        profiler.record_optimizer_step(actual_update=True, epoch=1)
        profiler.record_scheduler_step()
        profiler.record_scheduler_step()
        payload = profiler.to_dict()
        self.assertEqual(payload["optimizer_steps_actual"], 2)
        self.assertEqual(payload["optimizer_steps_by_epoch"], {1: 2})
        self.assertEqual(payload["scheduler_steps_actual"], 2)

    def test_08_batches_do_not_impersonate_gradient_accumulation_steps(self) -> None:
        profiler = RunProfiler("profile-08")
        for _ in range(4):
            profiler.record_training_batch()
        profiler.record_optimizer_step(actual_update=True)
        payload = profiler.to_dict()
        self.assertEqual(payload["training_batches_seen"], 4)
        self.assertEqual(payload["optimizer_steps_actual"], 1)

    def test_09_amp_skipped_step_is_separate_from_attempt(self) -> None:
        profiler = RunProfiler("profile-09")
        profiler.record_optimizer_step(actual_update=False)
        payload = profiler.to_dict()
        self.assertEqual(payload["optimizer_steps_attempted"], 1)
        self.assertEqual(payload["optimizer_steps_actual"], 0)

    def test_10_profile_write_isolated_and_non_overwriting(self) -> None:
        profiler = RunProfiler("profile-10")
        with profiler.stage("inference_only"):
            pass
        path = profiler.write_json(self.root / "outputs_major_revision")
        self.assertTrue(path.is_file())
        with self.assertRaises(FileExistsError):
            profiler.write_json(self.root / "outputs_major_revision")

    # Selection logging requirements 11–18.
    def test_11_valid_sample_records_write_and_read(self) -> None:
        records = [
            self.sample("a", "0"),
            self.sample("b", "0", loss_candidate=False, prototype_candidate=False, final_selected=False),
        ]
        logger = SelectionLogger("synthetic-run")
        logger.record_update(records, full_training_pool_size=2)
        bundle = read_selection_bundle(logger.finalize(self.root))
        self.assertEqual(len(bundle.samples), 2)
        self.assertEqual(len(bundle.update_rows), 1)

    def test_12_duplicate_stable_sample_key_is_rejected(self) -> None:
        logger = SelectionLogger("synthetic-run")
        with self.assertRaises(SelectionLogError):
            logger.record_update([self.sample("a", "0"), self.sample("a", "0")], full_training_pool_size=2)

    def test_13_candidate_final_counts_and_fallback_are_checked(self) -> None:
        records = [
            self.sample("a0", "a", loss_candidate=True, prototype_candidate=False, final_selected=False),
            self.sample("a1", "a", loss_candidate=False, prototype_candidate=True, final_selected=True),
            self.sample("b0", "b", loss_candidate=True, prototype_candidate=True, final_selected=True),
        ]
        logger = SelectionLogger("synthetic-run")
        logger.record_update(
            records,
            full_training_pool_size=3,
            fallback_observed_labels={"a"},
            fallback_rule="prototype_candidate_lowest_loss",
        )
        bundle = read_selection_bundle(logger.finalize(self.root))
        self.assertEqual(bundle.update_rows[0]["fallback_count"], 1)
        self.assertEqual(bundle.update_rows[0]["selected_total"], 2)

    def test_14_invalid_boolean_and_missing_required_score_fail(self) -> None:
        with self.assertRaises(SelectionLogError):
            self.sample("a", "0", final_selected="yes")  # type: ignore[arg-type]
        with self.assertRaises(SelectionLogError):
            self.sample("b", "0", loss=None, loss_candidate=True)

    def test_15_not_applicable_prototype_is_explicit_na(self) -> None:
        record = self.sample("a", "0", proto=None, prototype_candidate=None)
        self.assertIsNone(record.prototype_score)
        self.assertIsNone(record.prototype_candidate)
        logger = SelectionLogger("synthetic-run")
        logger.record_update([record], full_training_pool_size=1)
        bundle_dir = logger.finalize(self.root)
        contents = (bundle_dir / "selection_samples.csv").read_text(encoding="utf-8")
        self.assertIn("N/A", contents)

    def test_16_bundle_detects_sample_class_update_inconsistency(self) -> None:
        logger = SelectionLogger("synthetic-run")
        logger.record_update([self.sample("a", "0")], full_training_pool_size=1)
        bundle_dir = logger.finalize(self.root)
        update_path = bundle_dir / "selection_updates.csv"
        with update_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["selected_total"] = "0"
        with update_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(SelectionLogError):
            read_selection_bundle(bundle_dir)

    def test_17_training_schema_has_no_oracle_clean_or_noisy_fields(self) -> None:
        self.assertNotIn("clean_label", SAMPLE_FIELDS)
        self.assertNotIn("is_noisy", SAMPLE_FIELDS)
        self.assertNotIn("clean", " ".join(SAMPLE_FIELDS))

    def test_18_partial_bundle_never_reports_false_success(self) -> None:
        run_dir = self.root / "partial"
        run_dir.mkdir()
        (run_dir / "selection_samples.csv").write_text("broken", encoding="utf-8")
        with self.assertRaises(SelectionLogError):
            read_selection_bundle(run_dir)

    # Offline diagnostic requirements 19–30.
    def test_19_auc_perfect_ranking(self) -> None:
        self.assertEqual(tie_aware_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0)

    def test_20_auc_reverse_ranking(self) -> None:
        self.assertEqual(tie_aware_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]), 0.0)

    def test_21_auc_all_ties(self) -> None:
        self.assertEqual(tie_aware_auc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]), 0.5)

    def test_22_auc_single_outcome_is_na(self) -> None:
        self.assertIsNone(tie_aware_auc([0.2, 0.5], [True, True]))

    def test_23_purity_lift_uses_class_clean_ratio_definition(self) -> None:
        records = [
            self.sample("a", "0", proto=0.9, prototype_candidate=True, final_selected=True),
            self.sample("b", "0", proto=0.8, prototype_candidate=True, final_selected=True),
            self.sample("c", "0", proto=0.1, prototype_candidate=False, final_selected=False),
            self.sample("d", "0", proto=0.2, prototype_candidate=False, final_selected=False),
        ]
        result = diagnose_selection_update(records, oracle_records=self.oracle(records, [True, True, False, False]))
        row = result.class_rows[0]
        self.assertAlmostEqual(row["class_clean_ratio"], 0.5)
        self.assertAlmostEqual(row["prototype_purity"], 1.0)
        self.assertAlmostEqual(row["prototype_purity_lift"], 0.5)
        self.assertAlmostEqual(row["final_purity_lift"], 0.5)

    def test_24_micro_and_macro_are_not_mixed(self) -> None:
        records = [
            self.sample("a0", "a", final_selected=True),
            self.sample("a1", "a", final_selected=True),
            self.sample("a2", "a", final_selected=True),
            self.sample("b0", "b", loss_candidate=False, prototype_candidate=False, final_selected=False),
        ]
        result = diagnose_selection_update(records, oracle_records=self.oracle(records, [True, False, True, False]))
        self.assertAlmostEqual(result.micro_metrics["final_selected"]["selected_ratio"], 0.75)
        self.assertAlmostEqual(result.macro_metrics["actual_retention_ratio"]["mean"], 0.5)

    def test_25_across_seed_aggregation_uses_sample_sd(self) -> None:
        base = RunDiagnostics(
            "synthetic-run", "synthetic", 1, 5, "complete_synthetic_oracle", {}, (),
            {"prototype_auc": {"mean": 0.2}}, 0, "recorded",
        )
        results = [
            base,
            replace(base, training_seed=42, macro_metrics={"prototype_auc": {"mean": 0.4}}),
            replace(base, training_seed=88, macro_metrics={"prototype_auc": {"mean": 0.6}}),
        ]
        aggregate = aggregate_across_seeds(results)
        self.assertAlmostEqual(aggregate["macro"]["prototype_auc"]["mean"], 0.4)
        self.assertAlmostEqual(aggregate["macro"]["prototype_auc"]["sample_std"], 0.2)

    def test_26_missing_oracle_leaves_truth_metrics_not_computable(self) -> None:
        result = diagnose_selection_update([self.sample("a", "0")])
        self.assertEqual(result.oracle_status, "not_provided_truth_dependent_metrics_not_computable")
        self.assertIsNone(result.micro_metrics["final_selected"]["purity"])
        self.assertIsNone(result.class_rows[0]["prototype_auc"])

    def test_27_oracle_key_mismatch_raises(self) -> None:
        records = [self.sample("a", "0")]
        mismatched = [OracleLabelRecord("other", "train/other.jpg", "0", "0")]
        with self.assertRaises(DiagnosticError):
            diagnose_selection_update(records, oracle_records=mismatched)

    def test_28_validation_or_test_oracle_cannot_enter_training_diagnostics(self) -> None:
        records = [self.sample("a", "0")]
        oracle = self.oracle(records, [True])
        invalid = [replace(oracle[0], partition="validation")]
        with self.assertRaises(DiagnosticError):
            diagnose_selection_update(records, oracle_records=invalid)

    def test_29_bundle_adapter_rechecks_logged_counts(self) -> None:
        records = [
            self.sample("a", "0"),
            self.sample("b", "0", loss_candidate=False, prototype_candidate=False, final_selected=False),
        ]
        logger = SelectionLogger("synthetic-run")
        logger.record_update(records, full_training_pool_size=2)
        bundle = read_selection_bundle(logger.finalize(self.root))
        result = diagnose_bundle_update(bundle, training_seed=1, epoch=5, oracle_records=self.oracle(records, [True, False]))
        self.assertEqual(result.micro_metrics["final_selected"]["selected_count"], 1)

    def test_30_test_outputs_stay_in_temporary_root_and_legacy_root_is_rejected(self) -> None:
        records = [self.sample("a", "0")]
        logger = SelectionLogger("synthetic-run")
        logger.record_update(records, full_training_pool_size=1)
        output = logger.finalize(self.root / "outputs_major_revision")
        self.assertTrue(output.is_relative_to(self.root))
        second_logger = SelectionLogger("other-run")
        second_logger.record_update([replace(records[0], experiment_id="other-run")], full_training_pool_size=1)
        with self.assertRaises(SelectionLogError):
            second_logger.finalize(self.root / "outputs")
        profiler = RunProfiler("profile-30")
        with self.assertRaises(ValueError):
            profiler.write_json(self.root / "outputs")

    # Counterexamples found during the independent acceptance review.
    def test_31_cuda_stage_syncs_at_both_requested_boundaries(self) -> None:
        class FakeCuda:
            def __init__(self) -> None:
                self.synchronize_calls = 0

            def is_available(self) -> bool:
                return True

            def synchronize(self) -> None:
                self.synchronize_calls += 1

        class FakeTorch:
            def __init__(self) -> None:
                self.cuda = FakeCuda()

        fake_torch = FakeTorch()
        profiler = RunProfiler("fake-cuda", device_type="cuda", torch_module=fake_torch)
        token = profiler.start_stage("selection_loss_forward", synchronize_cuda=True)
        profiler.stop_stage(token)
        self.assertEqual(fake_torch.cuda.synchronize_calls, 2)

    def test_32_cuda_sync_disabled_never_syncs_at_stop(self) -> None:
        class FakeCuda:
            def __init__(self) -> None:
                self.synchronize_calls = 0

            def is_available(self) -> bool:
                return True

            def synchronize(self) -> None:
                self.synchronize_calls += 1

        class FakeTorch:
            def __init__(self) -> None:
                self.cuda = FakeCuda()

        fake_torch = FakeTorch()
        profiler = RunProfiler("fake-cuda-off", device_type="cuda", torch_module=fake_torch)
        token = profiler.start_stage("selection_cls_forward", synchronize_cuda=False)
        profiler.stop_stage(token)
        self.assertEqual(fake_torch.cuda.synchronize_calls, 0)

    def test_33_profile_and_selection_can_share_experiment_directory_in_both_orders(self) -> None:
        first_root = self.root / "outputs_major_revision"
        profiler_first = RunProfiler("synthetic-run")
        profile_path = profiler_first.write_json(first_root)
        logger_first = SelectionLogger("synthetic-run")
        logger_first.record_update([self.sample("a", "0")], full_training_pool_size=1)
        records_path = logger_first.finalize(first_root)
        self.assertEqual(profile_path.parent, records_path.parent)

        second_root = self.root / "outputs_major_revision_second"
        logger_second = SelectionLogger("synthetic-run-2")
        logger_second.record_update(
            [replace(self.sample("b", "0"), experiment_id="synthetic-run-2")],
            full_training_pool_size=1,
        )
        records_path = logger_second.finalize(second_root)
        profiler_second = RunProfiler("synthetic-run-2")
        profile_path = profiler_second.write_json(second_root)
        self.assertEqual(profile_path.parent, records_path.parent)

    def test_34_historical_outputs_children_are_protected(self) -> None:
        protected_child = historical_outputs_root() / "must-not-write"
        profiler = RunProfiler("protected-profile")
        with self.assertRaises(ValueError):
            profiler.write_json(protected_child)
        logger = SelectionLogger("protected-log")
        logger.record_update(
            [replace(self.sample("a", "0"), experiment_id="protected-log")], full_training_pool_size=1
        )
        with self.assertRaises(SelectionLogError):
            logger.finalize(protected_child)

    def test_35_strict_intersection_final_membership_mismatch_is_rejected(self) -> None:
        logger = SelectionLogger("synthetic-run")
        with self.assertRaises(SelectionLogError):
            logger.record_update(
                [self.sample("a", "0", loss_candidate=True, prototype_candidate=True, final_selected=False)],
                full_training_pool_size=1,
            )

    def test_36_non_strict_union_never_infers_fallback_from_empty_intersection(self) -> None:
        records = [
            self.sample(
                "a", "0", loss_candidate=True, prototype_candidate=False, final_selected=True, selection_strategy="union"
            )
        ]
        logger = SelectionLogger("synthetic-run")
        logger.record_update(records, full_training_pool_size=1)
        bundle = read_selection_bundle(logger.finalize(self.root))
        self.assertIsNone(bundle.class_rows[0]["fallback_triggered"])
        self.assertEqual(bundle.class_rows[0]["fallback_status"], "not_applicable_non_strict_strategy")
        self.assertIsNone(bundle.update_rows[0]["fallback_count"])

    def test_37_tampered_class_candidate_count_is_rejected(self) -> None:
        logger = SelectionLogger("synthetic-run")
        logger.record_update([self.sample("a", "0")], full_training_pool_size=1)
        bundle_dir = logger.finalize(self.root)
        path = bundle_dir / "selection_per_class.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["loss_candidate_count"] = "0"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(SelectionLogError):
            read_selection_bundle(bundle_dir)

    def test_38_tampered_actual_retention_ratio_is_rejected(self) -> None:
        logger = SelectionLogger("synthetic-run")
        logger.record_update([self.sample("a", "0")], full_training_pool_size=1)
        bundle_dir = logger.finalize(self.root)
        path = bundle_dir / "selection_per_class.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["actual_retention_ratio"] = "0.999"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(SelectionLogError):
            read_selection_bundle(bundle_dir)

    def test_39_extra_oracle_record_is_rejected(self) -> None:
        records = [self.sample("a", "0")]
        oracle = self.oracle(records, [True])
        oracle.append(OracleLabelRecord("extra", "train/extra.jpg", "0", "0"))
        with self.assertRaises(DiagnosticError):
            diagnose_selection_update(records, oracle_records=oracle)

    def test_40_declared_pgdf_fallback_must_select_minimum_loss_prototype(self) -> None:
        records = [
            self.sample(
                "low", "0", loss=0.1, loss_candidate=False, prototype_candidate=True, final_selected=False
            ),
            self.sample(
                "high", "0", loss=0.2, loss_candidate=False, prototype_candidate=True, final_selected=True
            ),
        ]
        logger = SelectionLogger("synthetic-run")
        with self.assertRaises(SelectionLogError):
            logger.record_update(
                records,
                full_training_pool_size=2,
                fallback_observed_labels={"0"},
                fallback_rule="prototype_candidate_lowest_loss",
            )

    def test_41_repeated_memory_phase_retains_cross_call_maximum(self) -> None:
        class FakeCuda:
            def __init__(self, peaks: list[tuple[int, int]]) -> None:
                self.peaks = peaks
                self.index = -1

            def is_available(self) -> bool:
                return True

            def synchronize(self) -> None:
                pass

            def reset_peak_memory_stats(self) -> None:
                self.index += 1

            def max_memory_allocated(self) -> int:
                return self.peaks[max(self.index, 0)][0]

            def max_memory_reserved(self) -> int:
                return self.peaks[max(self.index, 0)][1]

        class FakeTorch:
            def __init__(self, peaks: list[tuple[int, int]]) -> None:
                self.cuda = FakeCuda(peaks)

        for peaks, expected in (
            ([(100, 200), (20, 40)], (100, 200)),  # high then low
            ([(20, 40), (100, 200)], (100, 200)),  # low then high
            ([(5, 10), (80, 90), (30, 70), (120, 150), (60, 100)], (120, 150)),  # five calls
        ):
            with self.subTest(peaks=peaks):
                profiler = RunProfiler("memory-repeat", device_type="cuda", torch_module=FakeTorch(peaks))
                for _ in peaks:
                    with profiler.memory_phase("selection"):
                        pass
                observed = profiler.to_dict()["memory"]["phases"]["selection"]
                self.assertEqual((observed["allocated_bytes"], observed["reserved_bytes"]), expected)

    def test_42_diagnostics_writer_protects_simulated_history_and_writes_isolated_output(self) -> None:
        isolated = write_diagnostics_json({"status": "synthetic"}, self.root / "isolated" / "diagnostics.json")
        self.assertTrue(isolated.is_file())
        simulated_history = self.root / "simulated_history"
        simulated_history.mkdir()
        with patch(
            "revision_experiments._paths.historical_outputs_root", return_value=simulated_history.resolve()
        ):
            with self.assertRaises(DiagnosticError):
                write_diagnostics_json({"status": "must_not_write"}, simulated_history / "child" / "diagnostics.json")
        self.assertFalse((simulated_history / "child").exists())

    def test_43_cross_instance_stage_token_is_rejected_without_losing_active_stage(self) -> None:
        first = RunProfiler("first")
        second = RunProfiler("second")
        first_token = first.start_stage("epoch")
        second_token = second.start_stage("epoch")
        self.assertEqual(first_token.token_id, second_token.token_id)
        with self.assertRaises(ProfilingError):
            second.stop_stage(first_token)
        self.assertIsNotNone(second.stop_stage(second_token))
        self.assertIsNotNone(first.stop_stage(first_token))

    def test_44_active_timing_or_memory_phase_blocks_export_and_write(self) -> None:
        profiler = RunProfiler("active-export")
        token = profiler.start_stage("epoch")
        with self.assertRaises(ProfilingError):
            profiler.to_dict()
        with self.assertRaises(ProfilingError):
            profiler.write_json(self.root / "outputs_major_revision")
        self.assertFalse((self.root / "outputs_major_revision" / "active-export").exists())
        profiler.stop_stage(token)
        with profiler.memory_phase("selection"):
            with self.assertRaises(ProfilingError):
                profiler.to_dict()
            with self.assertRaises(ProfilingError):
                profiler.write_json(self.root / "outputs_major_revision")
        self.assertFalse((self.root / "outputs_major_revision" / "active-export").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
