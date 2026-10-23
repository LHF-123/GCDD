from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from gcdd.ido import (
    BMMDegenerateError,
    ClassConditionalBetaMixture,
    ClasswiseBMM,
    GlobalBMM,
    IDO_AMP_OVERFLOW_POLICY,
    IDO_BMM_MODE_GLOBAL,
    IDO_GLOBAL_BMM_SCOPE,
    _run_stage2_backward_with_amp_retry,
    ido_adaptation_name,
    official_we_loss,
    official_we_loss_per_sample,
    resolve_stage_budget,
    resolve_stage2_microbatch_size,
    snapshot_trainable_state,
    stage2_microbatch_slices,
    train_ido_lora,
    update_label_wave_state,
)
from revision_experiments.profiling import RunProfiler
from scripts.run_lora_checkpoint_validation import (
    CONTROLLED_ADAPTATION_METHODS,
    METHODS,
    parse_methods,
    preflight_run_dirs,
    require_fresh_run_dir,
    validate_ido_pilot_no_official_test_request,
)


class IDOTests(unittest.TestCase):
    def test_formal_task_contract_is_the_only_permitted_preexisting_run_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "ido" / "seed1"
            run_dir.mkdir(parents=True)
            contract = run_dir / "ido_task_contract.json"
            contract.write_text("{}", encoding="utf-8")

            # The formal batch writes this immutable input before invoking the
            # single-run entry point.  It must not be mistaken for a partial
            # training result.
            require_fresh_run_dir(run_dir, permitted_existing_file=contract)
            preflight_run_dirs(root, ["ido"], [1], permitted_existing_file=contract)

            (run_dir / "unexpected.txt").write_text("partial run", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                require_fresh_run_dir(run_dir, permitted_existing_file=contract)

            (run_dir / "unexpected.txt").unlink()
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                require_fresh_run_dir(
                    run_dir,
                    permitted_existing_file=root / "other" / "ido_task_contract.json",
                )

    def test_ido_pilot_no_official_test_is_strictly_isolated_from_formal_runs(self) -> None:
        pilot = SimpleNamespace(
            ido_pilot_no_official_test=True,
            official_test_selected_only=False,
            posthoc_oracle_test=False,
            ido_task_contract=None,
        )
        validate_ido_pilot_no_official_test_request(pilot, ["ido"], [1])
        invalid = SimpleNamespace(**{**pilot.__dict__, "official_test_selected_only": True})
        with self.assertRaisesRegex(ValueError, "cannot request"):
            validate_ido_pilot_no_official_test_request(invalid, ["ido"], [1])
        with self.assertRaisesRegex(ValueError, "one seed"):
            validate_ido_pilot_no_official_test_request(pilot, ["ido"], [1, 42])

    def test_stage_budget_scales_the_official_ratio_without_freezing_protocol(self) -> None:
        self.assertEqual((10, 20), resolve_stage_budget(30))
        self.assertEqual((5, 10), resolve_stage_budget(15))
        self.assertEqual((8, 17), resolve_stage_budget(25, stage1_epochs=8))
        with self.assertRaisesRegex(ValueError, "divisible by 3"):
            resolve_stage_budget(31)

    def test_bmm_degenerate_inputs_fail_closed_with_statistics(self) -> None:
        cases = {
            "all_zero": (np.zeros(24), "constant_wrong_event"),
            "same_nonzero": (np.full(24, 5.0), "constant_wrong_event"),
            "extreme_skew": (np.r_[np.zeros(23), [4.0]], "component_parameter_at_numeric_boundary"),
            "two_values": (np.asarray([0.0, 1.0] * 12), "component_parameter_at_numeric_boundary"),
            "nan": (np.asarray([0.0, np.nan]), "nonfinite_wrong_event"),
            "infinity": (np.asarray([0.0, np.inf]), "nonfinite_wrong_event"),
        }
        for name, (values, expected_reason) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(BMMDegenerateError) as raised:
                    ClassConditionalBetaMixture().fit(values, max_iter=20)
                self.assertEqual(expected_reason, raised.exception.record["bmm_reason"])
                self.assertEqual(len(values), raised.exception.record["sample_count"])
                self.assertEqual("degenerate_fail_closed", raised.exception.record["bmm_status"])

    def test_bmm_component_weight_collapse_is_fail_closed_and_records_parameters(self) -> None:
        values = np.tile(np.arange(6, dtype=np.float64), 4)
        collapsed = np.vstack((np.ones(len(values)), np.zeros(len(values))))
        with mock.patch.object(ClassConditionalBetaMixture, "_responsibilities", return_value=collapsed), mock.patch.object(
            ClassConditionalBetaMixture, "_fit_beta_weighted", return_value=(2.0, 2.0)
        ):
            with self.assertRaises(BMMDegenerateError) as raised:
                ClassConditionalBetaMixture().fit(values, max_iter=1)
        record = raised.exception.record
        self.assertEqual("component_weight_collapse", record["bmm_reason"])
        self.assertEqual(1.0, record["weight_1"])
        self.assertEqual(0.0, record["weight_2"])

    def test_bmm_component_label_swap_is_fail_closed(self) -> None:
        values = np.tile(np.arange(6, dtype=np.float64), 4)
        with mock.patch.object(
            ClassConditionalBetaMixture,
            "_fit_beta_weighted",
            side_effect=[(2.0, 1.0), (1.0, 2.0)],
        ):
            with self.assertRaises(BMMDegenerateError) as raised:
                ClassConditionalBetaMixture().fit(values, max_iter=1)
        record = raised.exception.record
        self.assertEqual("component_order_violation", record["bmm_reason"])
        self.assertGreater(
            record["component_1_normalized_mean"],
            record["component_2_normalized_mean"],
        )

    def test_bmm_normal_24_sample_support_fits_and_classwise_routes(self) -> None:
        values = np.tile(np.arange(6, dtype=np.int64), 8)
        labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 24)
        bmm = ClasswiseBMM(["a", "b"])
        rows = bmm.fit(values, labels, stage2_epoch=0, fit_round="stage2_initial", max_iter=20)
        terms = bmm.predict(np.asarray([0, 2, 4, 5], dtype=np.int64), np.asarray([0, 0, 1, 1], dtype=np.int64), stage2_epoch=1)

        self.assertTrue(all(row["bmm_status"] == "fitted" for row in rows))
        self.assertTrue(np.all(np.isfinite(terms["prob1"])))
        self.assertTrue(np.all(np.isfinite(terms["cdf1"])))
        self.assertTrue(np.all((terms["prob1"] >= 0.0) & (terms["prob1"] <= 1.0)))
        self.assertTrue(np.all((terms["cdf1"] >= 0.0) & (terms["cdf1"] <= 1.0)))
        np.testing.assert_allclose(terms["prob1"] + terms["prob2"], np.ones(4), atol=1.0e-5)

    def test_global_bmm_uses_one_pool_model_and_preserves_epsilon_posterior_semantics(self) -> None:
        values = np.tile(np.arange(11, dtype=np.int64), 24)
        labels = np.arange(len(values), dtype=np.int64) % 3
        bmm = GlobalBMM(["a", "b", "c"])
        rows = bmm.fit(values, labels, stage2_epoch=0, fit_round="stage2_initial", max_iter=20)
        terms = bmm.predict(values, labels, stage2_epoch=1)

        self.assertEqual(1, len(rows))
        self.assertEqual(IDO_BMM_MODE_GLOBAL, rows[0]["bmm_mode"])
        self.assertEqual(IDO_GLOBAL_BMM_SCOPE, rows[0]["bmm_scope"])
        self.assertEqual(len(values), rows[0]["sample_count"])
        self.assertEqual("", rows[0]["observed_label_id"])
        self.assertGreater(rows[0]["posterior_mass_deficit"], 0.0)
        self.assertTrue(np.all(np.isfinite(terms["prob1"])))
        self.assertTrue(np.all(np.isfinite(terms["cdf1"])))
        self.assertTrue(np.all(terms["prob1"] + terms["prob2"] < 1.0))
        self.assertFalse(np.allclose(1.0 - terms["prob1"], terms["prob2"], atol=0.0))
        import torch

        batch = 3
        output_s = torch.tensor([[2.0, 0.1], [0.4, 0.8], [1.0, -0.5]], requires_grad=True)
        output_w = torch.tensor([[1.5, 0.2], [0.1, 1.0], [0.9, -0.2]], requires_grad=True)
        labels_tensor = torch.tensor([0, 1, 0], dtype=torch.int64)
        weloss = official_we_loss_per_sample(
            torch,
            output_s,
            output_w,
            terms["prob1"][:batch],
            terms["prob2"][:batch],
            terms["cdf1"][:batch],
            terms["cdf2"][:batch],
            labels_tensor,
        )
        p1 = torch.as_tensor(terms["prob1"][:batch], dtype=output_s.dtype)
        p2 = torch.as_tensor(terms["prob2"][:batch], dtype=output_s.dtype)
        c1 = torch.as_tensor(terms["cdf1"][:batch], dtype=output_s.dtype)
        c2 = torch.as_tensor(terms["cdf2"][:batch], dtype=output_s.dtype)
        mse = torch.nn.functional.mse_loss(output_s, output_w, reduction="none").mean(dim=1)
        expected_weight = p1 * c1 + (1.0 - p2) * (1.0 - c2)
        self.assertTrue(torch.allclose(weloss["consistency_weight"], expected_weight, atol=1.0e-6))
        self.assertFalse(torch.allclose(1.0 - p1, p2, atol=0.0))

    def test_global_bmm_identity_is_distinct_from_classwise(self) -> None:
        self.assertNotEqual(ido_adaptation_name("classwise"), ido_adaptation_name("global"))

    def test_classwise_failure_includes_observed_class_epoch_and_reason(self) -> None:
        bmm = ClasswiseBMM(["a", "b"])
        with self.assertRaises(BMMDegenerateError) as raised:
            bmm.fit(
                np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64),
                np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
                stage2_epoch=3,
                fit_round="stage2_epoch_update",
                max_iter=2,
            )
        record = raised.exception.record
        self.assertEqual("a", record["observed_label"])
        self.assertEqual(0, record["observed_label_id"])
        self.assertEqual(3, record["stage2_epoch"])
        self.assertEqual("stage2_epoch_update", record["fit_round"])
        self.assertEqual("constant_wrong_event", record["bmm_reason"])

    def test_weloss_is_strictly_per_sample_and_matches_manual_reference(self) -> None:
        import torch

        generator = torch.Generator().manual_seed(9)
        for batch_size in (1, 2, 8):
            with self.subTest(batch_size=batch_size):
                output_s = torch.randn(batch_size, 5, generator=generator, requires_grad=True)
                output_w = torch.randn(batch_size, 5, generator=generator, requires_grad=True)
                labels = torch.arange(batch_size, dtype=torch.int64) % 5
                prob1 = np.linspace(0.15, 0.85, batch_size, dtype=np.float64)
                prob2 = 1.0 - prob1
                cdf1 = np.linspace(0.25, 0.75, batch_size, dtype=np.float64)
                cdf2 = np.linspace(0.8, 0.2, batch_size, dtype=np.float64)

                terms = official_we_loss_per_sample(torch, output_s, output_w, prob1, prob2, cdf1, cdf2, labels)
                for key, value in terms.items():
                    self.assertEqual((batch_size,), tuple(value.shape), key)
                    self.assertTrue(torch.isfinite(value).all(), key)
                loss, parts = official_we_loss(torch, output_s, output_w, prob1, prob2, cdf1, cdf2, labels)

                y_s = torch.softmax(output_s, dim=1).clamp(1.0e-4, 1.0 - 1.0e-4)
                y_w = torch.softmax(output_w, dim=1).clamp(1.0e-4, 1.0 - 1.0e-4)
                p1 = torch.as_tensor(prob1, dtype=output_s.dtype)
                p2 = torch.as_tensor(prob2, dtype=output_s.dtype)
                c1 = torch.as_tensor(cdf1, dtype=output_s.dtype)
                c2 = torch.as_tensor(cdf2, dtype=output_s.dtype)
                ce = p1 * (
                    torch.nn.functional.cross_entropy(output_s, labels, reduction="none")
                    + torch.nn.functional.cross_entropy(output_w, labels, reduction="none")
                )
                mse = torch.nn.functional.mse_loss(output_s, output_w, reduction="none").mean(dim=1)
                mse_term = (p1 * c1 + (1.0 - p2) * (1.0 - c2)) * mse
                average = (y_s + y_w) / 2.0
                entropy = -torch.max(average, dim=1)[0] * torch.sum(average * torch.log(average), dim=1)
                entropy_term = (1.0 - p1) * entropy
                reference = (ce + mse_term + entropy_term).mean()
                self.assertTrue(torch.allclose(loss, reference, atol=1.0e-6))
                self.assertAlmostEqual(float(ce.mean()), parts["ce_term"], places=6)
                self.assertAlmostEqual(float(mse_term.mean()), parts["mse_term"], places=6)
                self.assertAlmostEqual(float(entropy_term.mean()), parts["entropy_term"], places=6)
                loss.backward()
                self.assertTrue(torch.isfinite(output_s.grad).all())
                self.assertTrue(torch.isfinite(output_w.grad).all())

    def test_weloss_permutation_invariant_and_has_no_cross_sample_entropy_weighting(self) -> None:
        import torch

        output_s = torch.tensor([[3.0, 0.1, -1.0], [0.2, 0.3, 0.4]], requires_grad=True)
        output_w = torch.tensor([[2.0, 0.4, -0.3], [0.0, 0.1, 1.5]], requires_grad=True)
        labels = torch.tensor([0, 2], dtype=torch.int64)
        p1 = np.asarray([0.2, 0.9])
        p2 = np.asarray([0.8, 0.1])
        c1 = np.asarray([0.3, 0.7])
        c2 = np.asarray([0.6, 0.2])
        loss, _ = official_we_loss(torch, output_s, output_w, p1, p2, c1, c2, labels)
        order = torch.tensor([1, 0])
        permuted, _ = official_we_loss(
            torch, output_s[order], output_w[order], p1[[1, 0]], p2[[1, 0]], c1[[1, 0]], c2[[1, 0]], labels[order]
        )
        self.assertTrue(torch.allclose(loss, permuted, atol=1.0e-6))

        baseline = official_we_loss_per_sample(torch, output_s, output_w, p1, p2, c1, c2, labels)["entropy_per_sample"]
        changed_other = output_s.detach().clone()
        changed_other[1] = torch.tensor([20.0, -20.0, -20.0])
        isolated = official_we_loss_per_sample(torch, changed_other, output_w.detach(), p1, p2, c1, c2, labels)["entropy_per_sample"]
        self.assertTrue(torch.allclose(baseline[0], isolated[0], atol=1.0e-6))

    def test_weloss_rejects_non_batch_aligned_terms(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "prob1 must have shape"):
            official_we_loss(
                torch,
                torch.zeros(2, 3, requires_grad=True),
                torch.zeros(2, 3, requires_grad=True),
                np.asarray([0.5]),
                np.asarray([0.5, 0.5]),
                np.asarray([0.5, 0.5]),
                np.asarray([0.5, 0.5]),
                torch.tensor([0, 1]),
            )

    def test_stage2_microbatch_weloss_accumulation_matches_full_effective_batch(self) -> None:
        """Exercise the exact b_m / B scaling used by Stage 2 on CPU.

        The inputs use non-complementary posteriors to cover the global-BMM
        denominator semantics.  No dropout or AMP is used here so meaningful
        gradient/parameter equality is expected rather than masked by ordinary
        floating-point stochasticity.
        """
        import torch

        class CountingSGD(torch.optim.SGD):
            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.step_calls = 0

            def step(self, closure: object | None = None) -> object:
                self.step_calls += 1
                return super().step(closure=closure)

        class CountingScheduler:
            def __init__(self) -> None:
                self.step_calls = 0

            def step(self) -> None:
                self.step_calls += 1

        def run_once(batch_size: int, microbatch_size: int | None) -> dict[str, object]:
            torch.manual_seed(20261019)
            weak_images = torch.randn(batch_size, 7)
            strong_images = torch.randn(batch_size, 7)
            labels = torch.arange(batch_size, dtype=torch.int64) % 3
            prob1 = np.linspace(0.15, 0.85, batch_size, dtype=np.float64)
            # Deliberately do not force prob1 + prob2 == 1: global BMM uses a
            # density-sum-plus-epsilon denominator and WELoss consumes both.
            prob2 = np.linspace(0.78, 0.06, batch_size, dtype=np.float64)
            cdf1 = np.linspace(0.11, 0.91, batch_size, dtype=np.float64)
            cdf2 = np.linspace(0.88, 0.18, batch_size, dtype=np.float64)
            model = torch.nn.Linear(7, 3, bias=True)
            optimizer = CountingSGD(model.parameters(), lr=0.05)
            scheduler = CountingScheduler()
            optimizer.zero_grad(set_to_none=True)
            total_loss_sum = 0.0
            ce_sum = 0.0
            mse_sum = 0.0
            entropy_sum = 0.0
            for batch_slice in stage2_microbatch_slices(batch_size, microbatch_size):
                size = int(batch_slice.stop - batch_slice.start)
                weak_logits = model(weak_images[batch_slice])
                strong_logits = model(strong_images[batch_slice])
                loss, parts = official_we_loss(
                    torch,
                    strong_logits,
                    weak_logits,
                    prob1[batch_slice],
                    prob2[batch_slice],
                    cdf1[batch_slice],
                    cdf2[batch_slice],
                    labels[batch_slice],
                )
                (loss * (float(size) / float(batch_size))).backward()
                total_loss_sum += float(loss.detach()) * size
                ce_sum += parts["ce_term"] * size
                mse_sum += parts["mse_term"] * size
                entropy_sum += parts["entropy_term"] * size
            gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
            optimizer.step()
            scheduler.step()
            return {
                "loss": total_loss_sum / batch_size,
                "ce": ce_sum / batch_size,
                "mse": mse_sum / batch_size,
                "entropy": entropy_sum / batch_size,
                "gradients": gradients,
                "state": {name: value.detach().clone() for name, value in model.state_dict().items()},
                "optimizer_steps": optimizer.step_calls,
                "scheduler_steps": scheduler.step_calls,
            }

        for batch_size, microbatch_size in ((80, 20), (53, 20), (17, 20), (80, 80)):
            with self.subTest(batch_size=batch_size, microbatch_size=microbatch_size):
                full = run_once(batch_size, None)
                accumulated = run_once(batch_size, microbatch_size)
                for key in ("loss", "ce", "mse", "entropy"):
                    self.assertAlmostEqual(float(full[key]), float(accumulated[key]), places=6, msg=key)
                for name, gradient in full["gradients"].items():
                    torch.testing.assert_close(gradient, accumulated["gradients"][name], rtol=1.0e-6, atol=1.0e-6)
                for name, value in full["state"].items():
                    torch.testing.assert_close(value, accumulated["state"][name], rtol=1.0e-6, atol=1.0e-6)
                self.assertEqual(1, full["optimizer_steps"])
                self.assertEqual(1, accumulated["optimizer_steps"])
                self.assertEqual(1, full["scheduler_steps"])
                self.assertEqual(1, accumulated["scheduler_steps"])

        self.assertEqual((slice(0, 20), slice(20, 40), slice(40, 53)), stage2_microbatch_slices(53, 20))
        self.assertEqual((slice(0, 17),), stage2_microbatch_slices(17, 20))
        self.assertIsNone(resolve_stage2_microbatch_size(None))
        with self.assertRaisesRegex(ValueError, "positive"):
            resolve_stage2_microbatch_size(0)

    def test_stage2_amp_overflow_retries_same_batch_once_then_remains_fail_closed(self) -> None:
        """A recovered AMP overflow must not consume an optimizer/scheduler step.

        This CPU test uses a minimal scaler double to exercise the control
        flow without requiring CUDA.  The first backward deliberately injects
        an infinite gradient; the replay computes the same finite toy loss.
        """
        import torch

        class FakeGradScaler:
            def __init__(self) -> None:
                self.scale_value = 1024.0
                self.unscale_calls = 0
                self.update_calls = 0

            def is_enabled(self) -> bool:
                return True

            def get_scale(self) -> float:
                return self.scale_value

            def scale(self, value: torch.Tensor) -> torch.Tensor:
                return value

            def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
                self.unscale_calls += 1

            def update(self) -> None:
                self.update_calls += 1
                self.scale_value /= 2.0

        model = torch.nn.Linear(3, 2, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = FakeGradScaler()
        events: list[dict[str, object]] = []
        recovered: list[int] = []
        inputs = torch.tensor([[1.0, -2.0, 0.5]])
        targets = torch.zeros(1, 2)

        def backward_once(retry_index: int) -> dict[str, object]:
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            scaler.scale(loss).backward()
            if retry_index == 0:
                with torch.no_grad():
                    model.weight.grad.fill_(float("inf"))
            return {"loss_terms": {"total": float(loss.detach())}}

        def on_overflow(
            retry_index: int,
            scale_before: float | None,
            scale_after: float | None,
            failures: list[dict[str, object]],
            _payload: dict[str, object],
            terminal: bool,
        ) -> None:
            events.append(
                {
                    "retry_index": retry_index,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                    "failures": failures,
                    "terminal": terminal,
                }
            )

        payload, retry_count = _run_stage2_backward_with_amp_retry(
            torch,
            model,
            optimizer,
            scaler,
            backward_once,
            on_overflow,
            recovered.append,
        )
        self.assertEqual({"loss_terms"}, set(payload))
        self.assertEqual(1, retry_count)
        self.assertEqual(1, scaler.update_calls)
        self.assertEqual(2, scaler.unscale_calls)
        self.assertEqual([1], recovered)
        self.assertEqual(1, len(events))
        self.assertFalse(bool(events[0]["terminal"]))
        self.assertEqual(1024.0, events[0]["scale_before"])
        self.assertEqual(512.0, events[0]["scale_after"])
        self.assertTrue(torch.isfinite(model.weight.grad).all())

        terminal_model = torch.nn.Linear(3, 2, bias=False)
        terminal_optimizer = torch.optim.SGD(terminal_model.parameters(), lr=0.1)
        terminal_scaler = FakeGradScaler()
        terminal_events: list[dict[str, object]] = []

        def always_nonfinite(retry_index: int) -> dict[str, object]:
            loss = torch.nn.functional.mse_loss(terminal_model(inputs), targets)
            terminal_scaler.scale(loss).backward()
            with torch.no_grad():
                terminal_model.weight.grad.fill_(float("nan"))
            return {"loss_terms": {"total": float(loss.detach()), "retry_index": retry_index}}

        with self.assertRaisesRegex(FloatingPointError, "non-finite gradient"):
            _run_stage2_backward_with_amp_retry(
                torch,
                terminal_model,
                terminal_optimizer,
                terminal_scaler,
                always_nonfinite,
                lambda retry, before, after, failures, payload, terminal: terminal_events.append(
                    {
                        "retry_index": retry,
                        "scale_before": before,
                        "scale_after": after,
                        "failure_count": len(failures),
                        "terminal": terminal,
                        "payload": payload,
                    }
                ),
                lambda _retries: self.fail("a terminal AMP overflow must not report recovery"),
            )
        self.assertEqual(2, len(terminal_events))
        self.assertFalse(bool(terminal_events[0]["terminal"]))
        self.assertTrue(bool(terminal_events[1]["terminal"]))
        self.assertEqual(1, terminal_scaler.update_calls)
        self.assertEqual(
            "stage2_retry_same_effective_batch_once_after_gradscaler_backoff_then_fail_closed",
            IDO_AMP_OVERFLOW_POLICY,
        )

    def test_snapshot_deep_clones_mutable_trainable_state(self) -> None:
        import torch

        model = torch.nn.Module()
        model.head = torch.nn.Linear(2, 2, bias=False)
        state = snapshot_trainable_state(model)
        before = state["head.weight"].clone()
        with torch.no_grad():
            model.head.weight.add_(3.0)

        self.assertTrue(torch.equal(before, state["head.weight"]))
        self.assertFalse(torch.equal(model.head.weight.detach().cpu(), state["head.weight"]))

    def test_label_wave_selects_first_local_minimum_and_fixed_fallbacks(self) -> None:
        indices = np.arange(4, dtype=np.int64)
        epoch1 = np.asarray([0, 0, 0, 0])
        epoch2 = np.asarray([1, 1, 1, 0])  # 3 changes: improve from initial threshold 4
        epoch3 = np.asarray([1, 1, 0, 0])  # 1 change: improve again
        epoch4 = np.asarray([0, 1, 0, 0])  # 1 change: first non-decrease locks epoch 3
        previous: np.ndarray | None = None
        previous_change = len(indices)
        saved = False
        selected_epochs: list[int] = []
        for epoch, prediction in enumerate((epoch1, epoch2, epoch3, epoch4), start=1):
            changes, selected, previous, previous_change, saved = update_label_wave_state(
                previous_prediction=previous,
                previous_change=previous_change,
                label_wave_saved=saved,
                current_prediction=prediction,
                train_idx=indices,
                stage_epoch=epoch,
                stage1_epochs=4,
            )
            if selected:
                selected_epochs.append(epoch)
        self.assertEqual([1, 2, 3], selected_epochs)
        self.assertEqual(1, changes)
        self.assertTrue(saved)
        np.testing.assert_array_equal(epoch3, previous)

        # No strict improvement after epoch 1 leaves the documented epoch-1 fallback.
        changes, selected, previous, previous_change, saved = update_label_wave_state(
            previous_prediction=epoch1,
            previous_change=4,
            label_wave_saved=False,
            current_prediction=np.asarray([1, 1, 1, 1]),
            train_idx=indices,
            stage_epoch=2,
            stage1_epochs=3,
        )
        self.assertEqual(4, changes)
        self.assertFalse(selected)
        self.assertTrue(saved)
        self.assertEqual(4, previous_change)
        np.testing.assert_array_equal(epoch1, previous)

    def test_controlled_adaptation_parses_without_changing_primary_all(self) -> None:
        self.assertEqual(("ido",), CONTROLLED_ADAPTATION_METHODS)
        self.assertEqual(["ido"], parse_methods("ido"))
        self.assertEqual(list(METHODS), parse_methods("all"))
        self.assertNotIn("ido", parse_methods("all"))

    def test_cpu_smoke_writes_stage_boundary_state_validation_checkpoint_and_d1(self) -> None:
        import torch

        class FakeBackbone(torch.nn.Module):
            embed_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.stem = torch.nn.Linear(3, 4)
                self.qkv = torch.nn.Linear(4, 4)

            def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
                features = self.stem(images.mean(dim=(2, 3)))
                return {"x_norm_clstoken": self.qkv(features)}

        class StableTestBMM:
            def __init__(self, labels: list[str]) -> None:
                self.labels = labels

            def fit(self, wrong_event: np.ndarray, label_ids: np.ndarray, *, stage2_epoch: int, fit_round: str, max_iter: int) -> list[dict[str, object]]:
                return [
                    {
                        "fit_round": fit_round,
                        "stage2_epoch": stage2_epoch,
                        "observed_label_id": label_id,
                        "observed_label": label,
                        "sample_count": int((label_ids == label_id).sum()),
                        "unique_wrong_event_count": 2,
                        "wrong_event_min": 0.0,
                        "wrong_event_max": 1.0,
                        "bmm_status": "fitted",
                        "bmm_reason": "",
                        "bmm_fit_iterations": max_iter,
                        "alpha_1": 1.0,
                        "beta_1": 2.0,
                        "weight_1": 0.5,
                        "alpha_2": 2.0,
                        "beta_2": 1.0,
                        "weight_2": 0.5,
                    }
                    for label_id, label in enumerate(self.labels)
                ]

            def predict(self, wrong_event: np.ndarray, label_ids: np.ndarray, *, stage2_epoch: int) -> dict[str, np.ndarray | int]:
                length = len(wrong_event)
                return {
                    "prob1": np.full(length, 0.6), "prob2": np.full(length, 0.4),
                    "cdf1": np.full(length, 0.4), "cdf2": np.full(length, 0.6),
                    "fallback_predictions": 0, "clipped_predictions": 0,
                    "posterior_mass_deficit_sum": 0.0, "posterior_mass_deficit_max": 0.0,
                }

            def fallback_class_count(self) -> int:
                return 0

        class StableGlobalTestBMM(StableTestBMM):
            def fit(self, wrong_event: np.ndarray, label_ids: np.ndarray, *, stage2_epoch: int, fit_round: str, max_iter: int) -> list[dict[str, object]]:
                return [
                    {
                        "bmm_mode": "global",
                        "bmm_scope": "all_observed_training_pool",
                        "fit_round": fit_round,
                        "stage2_epoch": stage2_epoch,
                        "observed_label_id": "",
                        "observed_label": "",
                        "sample_count": int(len(wrong_event)),
                        "unique_wrong_event_count": 2,
                        "wrong_event_min": 0.0,
                        "wrong_event_max": 1.0,
                        "bmm_status": "fitted",
                        "bmm_reason": "",
                        "bmm_fit_iterations": max_iter,
                        "alpha_1": 1.0,
                        "beta_1": 2.0,
                        "weight_1": 0.5,
                        "alpha_2": 2.0,
                        "beta_2": 1.0,
                        "weight_2": 0.5,
                    }
                ]

        cfg = {
            "feature": {"device": "cpu", "input_size": 16},
            "lora": {"rank": 2, "alpha": 2.0, "dropout": 0.0, "target_modules": "qkv"},
            "lora_train": {
                "epochs": 3, "batch_size": 4, "eval_batch_size": 4, "num_workers": 0, "pin_memory": False,
                "lora_lr": 1.0e-3, "head_lr": 1.0e-3, "weight_decay": 0.0, "scheduler": "none",
                "warmup_ratio": 0.0, "amp": False,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths: list[str] = []
            for index, color in enumerate(((255, 0, 0), (240, 0, 0), (0, 255, 0), (0, 240, 0))):
                image_path = root / f"sample_{index}.png"
                Image.new("RGB", (24, 24), color).save(image_path)
                paths.append(str(image_path))
            labels = np.asarray(["a", "a", "b", "b"])
            profiler = RunProfiler("ido-cpu-smoke", mode="basic", device_type="cpu")
            with mock.patch("gcdd.lora_training.load_dinov2_model", side_effect=lambda *args, **kwargs: FakeBackbone()), mock.patch(
                "gcdd.ido.ClasswiseBMM", StableTestBMM
            ), mock.patch("gcdd.ido.GlobalBMM", StableGlobalTestBMM):
                result = train_ido_lora(
                    paths, labels, paths, labels, np.ones(len(labels), dtype=bool), cfg,
                    method="ido-smoke", seed=1, checkpoint_path=root / "best_val.pt",
                    test_paths=paths, test_labels=labels, final_checkpoint_path=root / "last.pt",
                    last5_checkpoint_dir=root / "last5", official_test_selected_only=True, profiler=profiler,
                    stage2_microbatch_size=2,
                )
                global_result = train_ido_lora(
                    paths, labels, paths, labels, np.ones(len(labels), dtype=bool), cfg,
                    method="ido-global-smoke", seed=2, bmm_mode="global", checkpoint_path=root / "global" / "best_val.pt",
                    final_checkpoint_path=root / "global" / "last.pt", last5_checkpoint_dir=root / "global" / "last5",
                    profiler=RunProfiler("ido-global-cpu-smoke", mode="basic", device_type="cpu"),
                )

            self.assertEqual(3, len(result.logs))
            self.assertEqual(1, result.summary["stage1_epochs"])
            self.assertEqual(2, result.summary["stage2_epochs"])
            self.assertIn("validation_selected_test_top1", result.summary)
            self.assertEqual("", result.summary["final_test_top1"])
            self.assertTrue((root / "best_val.pt").is_file())
            self.assertTrue((root / "last.pt").is_file())
            self.assertTrue((root / "stage1_base.pt").is_file())
            stage1_state = torch.load(root / "stage1_base.pt", map_location="cpu", weights_only=False)
            self.assertEqual("not_implemented_fail_closed_new_attempt_required", stage1_state["resume_support"])
            self.assertEqual(4, stage1_state["stage2_effective_batch_size"])
            self.assertEqual(2, stage1_state["stage2_microbatch_size"])
            self.assertTrue(stage1_state["stage2_gradient_accumulation_enabled"])
            self.assertEqual(
                stage1_state["lora_structure_contract_sha256"],
                result.summary["lora_structure_contract_sha256"],
            )
            self.assertEqual(
                stage1_state["lora_runtime_structure_sha256"],
                result.summary["lora_runtime_structure_sha256"],
            )
            self.assertEqual("live_model_after_inject_lora", stage1_state["lora_runtime_structure"]["record_source"])
            self.assertEqual(["qkv"], stage1_state["lora_runtime_structure"]["injected_backbone_modules"])
            self.assertEqual("dinov2_vitb14", stage1_state["lora_structure_contract"]["architecture"])
            self.assertIn("loader", stage1_state["dinov2_pretrained_source"])
            self.assertIn("torch", stage1_state["runtime_environment"])
            self.assertEqual(4, len(stage1_state["wrong_event"]))
            np.testing.assert_array_equal(result.stage1_wrong_event, stage1_state["wrong_event"])
            self.assertTrue(np.all(result.wrong_event >= result.stage1_wrong_event))
            self.assertGreaterEqual(result.logs[1]["wrong_event_total"], result.logs[0]["wrong_event_total"])
            self.assertEqual(2, len(list((root / "last5").glob("*.pt"))))
            # Initial fit plus the refresh after Stage-2 epoch 1.  The final
            # epoch deliberately skips its otherwise-unused refit.
            self.assertEqual(2 * 2, len(result.bmm_fit_rows))
            self.assertEqual("complete", result.summary["d1_recording_status"])
            self.assertEqual(4, result.summary["d1_stage1_processed_samples"])
            self.assertEqual(8, result.summary["d1_stage2_processed_samples"])
            self.assertEqual(8, result.summary["d1_stage2_extra_view_forwards"])
            self.assertEqual(16, result.summary["d1_stage2_prediction_prepass_view_forwards"])
            self.assertEqual(4, result.summary["stage2_effective_batch_size"])
            self.assertEqual(2, result.summary["stage2_microbatch_size"])
            self.assertTrue(result.summary["stage2_gradient_accumulation_enabled"])
            self.assertEqual(4, result.summary["stage2_microbatch_count_total"])
            self.assertEqual(3, result.summary["d1_optimizer_steps_attempted"])
            self.assertEqual(3, result.summary["d1_optimizer_steps_actual"])
            self.assertGreaterEqual(result.summary["d1_stage1_wall_seconds"], 0.0)
            self.assertGreaterEqual(result.summary["d1_stage2_wall_seconds"], 0.0)
            self.assertGreaterEqual(result.summary["d1_optimizer_loop_wall_seconds"], 0.0)
            self.assertGreaterEqual(result.summary["d1_bmm_fit_seconds"], 0.0)
            self.assertGreaterEqual(result.summary["d1_checkpoint_seconds"], 0.0)
            self.assertGreaterEqual(result.summary["d1_official_test_seconds"], 0.0)
            self.assertGreaterEqual(result.summary["d1_end_to_end_wall_seconds"], 0.0)
            self.assertGreaterEqual(result.summary["d1_validation_seconds"], 0.0)
            self.assertEqual(0, result.summary["final_bmm_fallback_classes"])
            self.assertEqual("global", global_result.summary["bmm_mode"])
            self.assertEqual(2, len(global_result.bmm_fit_rows))
            self.assertTrue(all(row["bmm_mode"] == "global" for row in global_result.bmm_fit_rows))


if __name__ == "__main__":
    unittest.main()
