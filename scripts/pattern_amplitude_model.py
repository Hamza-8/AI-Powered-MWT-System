"""Experimental MSRNN measurement representation; legacy package stays intact.

One positive, observed-support complex RMS separates pattern from magnitude.
Raw measurements and log magnitude condition the pattern stream through FiLM.
This is an engineering hypothesis, not a scale-equivariant EM forward model.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from microscan.models import MSRNN, MeasurementEncoder, sanitize_measurements


REPRESENTATION = "pattern_amplitude_v1"


def factor_measurements(features, mask, rms_floor=1e-8):
    """Return six pattern features, amplitude descriptors and complex RMS.

    Inputs are the legacy [real, imag, mask, GHz, Rx, Tx] features. RMS uses
    complex-entry count, including observed zeros, across ALL selected slots.
    Hidden values are removed before arithmetic. Statistics stay float32 under
    AMP; max scaling avoids squaring very large/small raw observations directly.
    The floor is in training-global-normalized input units, not physical units.
    """
    if features.ndim != 5 or features.shape[1] != 6:
        raise ValueError("Expected six fixed-slot measurement features")
    if not math.isfinite(rms_floor) or rms_floor <= 0:
        raise ValueError("RMS floor must be finite and positive")
    with torch.autocast(device_type=features.device.type, enabled=False):
        raw = sanitize_measurements(features[:, :2].float(), mask)
        count = mask.bool().flatten(1).sum(1).float()
        peak = raw.abs().flatten(1).amax(1)
        safe_peak = torch.where(peak > 0, peak, torch.ones_like(peak))
        scaled = raw / safe_peak[:, None, None, None, None]
        mean_square = scaled.square().flatten(1).sum(1) / count.clamp_min(1)
        rms = peak * mean_square.clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        denominator = rms.clamp_min(rms_floor)
        pattern = raw / denominator[:, None, None, None, None]
        descriptors = torch.stack((denominator.log(), count / mask[0].numel(),
                                   (rms > rms_floor).float()), dim=1)
        pattern_features = torch.cat((pattern, features[:, 2:].float()), dim=1)
    return pattern_features, descriptors, rms


class AmplitudeConditioner(nn.Module):
    """Compact parallel raw stream plus log-RMS FiLM, initially near identity."""
    def __init__(self, channels):
        super().__init__()
        self.raw_encoder = MeasurementEncoder(2 * channels)
        self.amplitude = nn.Sequential(nn.Linear(3, 32), nn.SiLU(),
                                       nn.Linear(32, 2 * channels))
        # Small nonzero weights preserve initial pattern features while allowing
        # gradients through BOTH context branches on the first optimizer step.
        for layer in (self.raw_encoder.readout, self.amplitude[-1]):
            nn.init.normal_(layer.weight, std=1e-4)
            nn.init.zeros_(layer.bias)

    def forward(self, pattern, raw_features, mask, descriptors):
        gamma, beta = (self.raw_encoder(raw_features, mask) + self.amplitude(descriptors)).chunk(2, 1)
        shape = (pattern.shape[0], pattern.shape[1], *([1] * (pattern.ndim - 2)))
        return pattern * (1 + gamma.reshape(shape)) + beta.reshape(shape)


class PatternCoarse(nn.Module):
    def __init__(self, original, rms_floor):
        super().__init__()
        self.core = original
        self.rms_floor = rms_floor
        self.feature_channels = original.feature_channels
        self.output_size = original.output_size
        self.conditioner = AmplitudeConditioner(original.adapter.in_channels)

    def forward(self, features):
        mask = features[:, 2:3].bool()
        pattern, descriptors, _ = factor_measurements(features, mask, self.rms_floor)
        encoded = self.core.encoder(pattern.flatten(2))
        pooled = F.adaptive_avg_pool1d(encoded, self.core.pooled_length)
        pooled = self.conditioner(pooled, features, mask, descriptors)
        latent = self.core.adapter(pooled).reshape(features.shape[0], self.core.latent_channels,
            self.core.latent_size, self.core.latent_size, self.core.latent_size)
        decoded = self.core.decoder(latent)
        return self.core.output(decoded), decoded


class PatternMeasurementEncoder(nn.Module):
    def __init__(self, original, rms_floor):
        super().__init__()
        self.pattern_encoder, self.rms_floor = original, rms_floor
        self.conditioner = AmplitudeConditioner(original.readout.out_features)

    def forward(self, features, mask):
        pattern, descriptors, _ = factor_measurements(features, mask, self.rms_floor)
        return self.conditioner(self.pattern_encoder(pattern, mask), features, mask, descriptors)


class PatternAmplitudeMSRNN(MSRNN):
    """Same volume decoder/loss/API; factored coarse and enhancement inputs.

    The pattern stream retains the baseline's seeded initialization. Additional
    raw/amplitude branches have new parameters. No target/geometry enters them.
    A/B retain the inherited deployment graph; C qualification is unchanged.
    """
    def __init__(self, config=None):
        config = dict(config or {})
        if config.get("measurement_representation", REPRESENTATION) != REPRESENTATION:
            raise ValueError("Wrong measurement representation for this model")
        floor = float(config.get("pattern_rms_floor", 1e-8))
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("Invalid pattern RMS floor")
        super().__init__(config)
        self.coarse = PatternCoarse(self.coarse, floor)
        self.enhancement.measurement_encoder = PatternMeasurementEncoder(
            self.enhancement.measurement_encoder, floor)
