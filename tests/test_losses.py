import unittest
import torch
from microscan.losses import (balanced_property_loss, reconstruction_loss,
    stable_temporal_loss, segmentation_loss, repeated_acquisition_loss)


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(1)

    def test_foreground_balanced_and_hidden_label_isolation(self):
        target = torch.zeros(1, 2, 4, 4, 4)
        predicted = target.clone()
        predicted[:, :, 0, 0, 0] = 1
        foreground = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
        foreground[:, :, 0, 0, 0] = True
        balanced = balanced_property_loss(predicted, target, foreground, property_scales=(1, 1))
        ordinary = balanced_property_loss(predicted, target, property_scales=(1, 1))
        self.assertGreater(float(balanced), float(ordinary)*10)
        valid = ~foreground
        nan_target = torch.where(valid, target, torch.full_like(target, float("nan")))
        self.assertEqual(float(balanced_property_loss(predicted, nan_target, validity=valid)), 0)

    def test_signed_and_stable_temporal_terms(self):
        delta = torch.full((1, 2, 2, 2, 2), -.2, requires_grad=True)
        self.assertEqual(float(stable_temporal_loss(delta)), 0)
        self.assertGreater(float(stable_temporal_loss(delta, torch.ones_like(delta, dtype=torch.bool))), 0)
        baseline = torch.ones_like(delta)
        target = baseline+delta.detach()
        losses = reconstruction_loss({"coarse": target, "refined": target, "change": delta},
            target, baseline_target=baseline)
        self.assertAlmostEqual(float(losses["change"]), 0, places=7)
        self.assertEqual(float(losses["temporal"]), 0)
        with self.assertRaises(ValueError):
            reconstruction_loss({"refined": target}, target, weights={"equation": 1})

    def test_segmentation_binary_multiclass_overlap_ignore_and_absent(self):
        for schema, channels in (("binary", 1), ("multiclass", 5), ("multilabel", 4)):
            logits = torch.zeros(1, channels, 3, 3, 3, requires_grad=True)
            labels = torch.zeros(1, 3, 3, 3, dtype=torch.long) if schema == "multiclass" else torch.zeros_like(logits)
            if schema == "multilabel":
                labels[:, 0, 0, 0, 0] = 1
                labels[:, 1, 0, 0, 0] = 1  # explicitly overlapping voxel
            loss = segmentation_loss(logits, labels, schema=schema)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            ignored = torch.full_like(labels, -1)
            self.assertEqual(float(segmentation_loss(logits, ignored, schema=schema)), 0)
        with self.assertRaises(ValueError):
            segmentation_loss(torch.zeros(1, 1, 2, 2, 2), None)
        logits = torch.full((1, 1, 2, 2, 2), float("nan"), requires_grad=True)
        self.assertEqual(float(segmentation_loss(logits, torch.full_like(logits, -1))), 0)

    def test_noise2noise_assumptions_explicit(self):
        x = torch.zeros(1, 2, 2, 2, 2)
        with self.assertRaises(ValueError):
            repeated_acquisition_loss(x, x, torch.ones_like(x), shared_drift=True)


if __name__ == "__main__":
    unittest.main()
