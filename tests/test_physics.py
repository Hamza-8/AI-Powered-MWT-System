"""Bounded mechanism fixtures, not independent headset scientific validation."""
import unittest
import torch
from microscan.physics import (HeadsetForwardOperator, PhysicsConsistency,
    qualification_report, require_qualified_operator, masked_complex_loss,
    freeze_operator, forward_training_loss, field_equation_loss,
    physics_status, validation_signal_metrics, weights_hash)


def fixture_context():
    return {"tx_positions_mm": [[0., 10., 0.], [10., 0., 0.]],
            "rx_positions_mm": [[0., 10., 0.], [10., 0., 0.]],
            "source_excitation": [[1., 0.], [1., 0.]],
            "layout_id": "synthetic_test_only", "calibration_id": "fixture",
            "constant_spectrum": True}


def fixture_report(phi, context):
    # Artificially positive metrics exercise gate wiring; explicitly not science.
    metrics = {name: 0.0 for name in ("full_relative_rmse", "differential_relative_rmse",
        "noise_relative_rmse", "worst_channel_relative_rmse", "roi_relative_rmse")}
    return qualification_report(phi, metrics, context, [.6, .85],
        training_groups=["synthetic_train"], validation_groups=["synthetic_heldout"],
        coverage=["healthy", "lesion", "positive_change", "negative_change", "sources", "regions"],
        scientific=False, property_bounds=[[-100, 100], [-10, 10]])


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        self.phi = HeadsetForwardOperator(width=2, latent_dim=8)
        self.context = fixture_context()
        self.freq = torch.tensor([.6, .85])
        self.x = torch.rand(1, 2, 8, 8, 8)
        self.mask = torch.ones(1, 1, 2, 2, 2, dtype=torch.bool)

    def test_forward_spectral_and_context(self):
        y = self.phi(self.x, self.freq, self.context)
        self.assertEqual(tuple(y.shape), (1, 2, 2, 2, 2))
        spectral = self.x[:, :, None].expand(-1, -1, 2, -1, -1, -1)
        self.assertTrue(torch.allclose(y, self.phi(spectral, self.freq, self.context)))
        with self.assertRaises(ValueError):
            self.phi(self.x, self.freq, {**self.context, "observations": y})
        with self.assertRaises(ValueError):
            self.phi(self.x, self.freq, {**self.context, "constant_spectrum": False})
        changed = self.phi(self.x, self.freq+.1, self.context)
        self.assertFalse(torch.equal(y, changed))

    def test_fixture_cannot_qualify_scientifically_and_binding(self):
        report = fixture_report(self.phi, self.context)
        self.assertFalse(report["qualified"])
        with self.assertRaises(ValueError):
            require_qualified_operator(self.phi, report)
        require_qualified_operator(self.phi, report, self.context, self.phi.domain, allow_fixture=True)
        with self.assertRaises(ValueError):
            require_qualified_operator(self.phi, report, {**self.context, "calibration_id": "other"}, allow_fixture=True)
        with torch.no_grad():
            next(self.phi.parameters()).add_(.01)
        with self.assertRaises(ValueError):
            require_qualified_operator(self.phi, report, allow_fixture=True)

    def test_frozen_b_optimizer_input_gradients_and_provenance(self):
        report = fixture_report(self.phi, self.context)
        consistency = PhysicsConsistency(self.phi, report, allow_fixture=True)
        inverse = torch.nn.Conv3d(2, 2, 1)
        optimizer = torch.optim.Adam(inverse.parameters(), lr=.01)
        before = inverse.weight.detach().clone()
        state_before = weights_hash(self.phi)
        prediction = inverse(self.x)
        obs = torch.zeros(1, 2, 2, 2, 2)
        loss = consistency(prediction, obs, self.mask, self.freq, self.context)
        optimizer.zero_grad()
        loss.backward()
        self.assertGreater(float(inverse.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.phi.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, inverse.weight))
        self.assertEqual(state_before, weights_hash(self.phi))
        consistency.train()
        self.assertFalse(self.phi.training)
        self.assertEqual(report["weights_sha256"], weights_hash(self.phi))

    def test_mask_noise_covariance_hidden_nans_isolated(self):
        prediction = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
        observation = torch.randn_like(prediction)
        mask = self.mask.clone()
        mask[..., 0, 0] = False
        expanded = mask.expand_as(prediction)
        nan_p = torch.where(expanded, prediction, torch.full_like(prediction, float("nan")))
        nan_o = torch.where(expanded, observation, torch.full_like(observation, float("nan")))
        covariance = torch.eye(prediction.numel())
        hidden = ~expanded.reshape(-1)
        covariance[hidden] = float("nan")
        covariance[:, hidden] = float("nan")
        expected = masked_complex_loss(prediction, observation, mask)
        actual = masked_complex_loss(nan_p, nan_o, mask, covariance=covariance)
        self.assertTrue(torch.allclose(expected, actual))
        actual.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        empty = masked_complex_loss(nan_p, nan_o, torch.zeros_like(mask))
        self.assertEqual(float(empty), 0)

    def test_domains_differential_signed_and_original_observation(self):
        report = fixture_report(self.phi, self.context)
        consistency = PhysicsConsistency(self.phi, report, allow_fixture=True)
        change = torch.randn_like(self.x)*.01
        baseline = self.x.detach()
        measured = self.phi(baseline+change, self.freq, self.context)-self.phi(baseline, self.freq, self.context)
        loss = consistency(change, measured, self.mask, self.freq, self.context,
            baseline_properties=baseline, baseline_valid=torch.tensor([True]), observation_domain="current_minus_baseline")
        self.assertLess(float(loss), 1e-12)
        with self.assertRaises(ValueError):
            consistency(change, measured, self.mask, self.freq, self.context,
                        baseline_properties=baseline, baseline_valid=True)
        with self.assertRaises(ValueError):
            consistency(self.x, measured, self.mask, self.freq, self.context, observation_domain="decluttered_target")
        with self.assertRaises(ValueError):
            consistency(self.x, measured, self.mask, self.freq+.1, self.context)

    def test_independent_training_and_metrics_missing_evidence(self):
        optimizer = torch.optim.Adam(self.phi.parameters(), lr=.001)
        obs = torch.zeros(1, 2, 2, 2, 2)
        loss = forward_training_loss(self.phi, self.x, obs, self.mask, self.freq, self.context,
            baseline_properties=self.x*.9, baseline_observations=obs, baseline_mask=self.mask)
        optimizer.zero_grad()
        loss["total"].backward()
        optimizer.step()
        metrics = validation_signal_metrics(self.phi(self.x, self.freq, self.context), obs, self.mask)
        self.assertTrue(torch.isnan(torch.tensor(metrics["noise_relative_rmse"])))
        status = physics_status("B")
        self.assertEqual(status["physics_active"], "A")
        with self.assertRaises(ValueError):
            physics_status("B", required=True)

    def test_supplied_discrete_field_fixture_and_prerequisites(self):
        # Known two-node discrete linear field equation A E = J including boundary.
        matrix = torch.tensor([[2., -1.], [-1., 2.]])
        field = torch.tensor([1., 2.], requires_grad=True)
        source = matrix@field.detach()
        context = {"equation_name": "synthetic_discrete_field_fixture", "material_convention": "unit",
                   "boundary_condition": "supplied_Dirichlet", "operator_validated": True}
        loss = field_equation_loss(field, source, lambda e, c: matrix@e,
                                  torch.tensor([True, True]), context=context)
        self.assertEqual(float(loss), 0)
        with self.assertRaises(ValueError):
            field_equation_loss(None, source, None, None, context={})


if __name__ == "__main__":
    unittest.main()
