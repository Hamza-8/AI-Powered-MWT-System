"""Targeted hypothesis: expose log magnitude after coarse decoder normalization.

Adds FiLM after each existing GroupNorm and before its SiLU. The original
pattern encoder, earlier raw/RMS branches, volume heads and refiner are retained.
No sphere radius rule, target-derived input, new loss, or spatial warp is used.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from pattern_amplitude_model import PatternAmplitudeMSRNN, factor_measurements


ADJUSTMENT = "postnorm_amplitude_v1"


class PostNormFiLM(nn.Module):
    def __init__(self, channels, hidden=32, init_std=.02):
        super().__init__()
        self.affine = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 2*channels))
        nn.init.normal_(self.affine[-1].weight, std=init_std)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, normalized, descriptors):
        gamma, beta = self.affine(descriptors).chunk(2, 1)
        shape = (normalized.shape[0], normalized.shape[1], 1, 1, 1)
        return normalized * (1 + gamma.reshape(shape)) + beta.reshape(shape)


class PostNormCoarse(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.core, self.conditioner = original.core, original.conditioner
        self.rms_floor = original.rms_floor
        self.output_size, self.feature_channels = original.output_size, original.feature_channels
        norms = list(self.core.decoder)[1::3]
        if len(self.core.decoder) != 3 * len(norms) or not all(isinstance(n, nn.GroupNorm) for n in norms):
            raise ValueError("Expected ConvTranspose/GroupNorm/SiLU decoder stages")
        hidden, std = int(config.get("hidden", 32)), float(config.get("init_std", .02))
        if hidden < 1 or not math.isfinite(std) or std <= 0:
            raise ValueError("Invalid post-normalization FiLM initialization")
        self.postnorm = nn.ModuleList(PostNormFiLM(n.num_channels, hidden, std) for n in norms)

    def forward(self, features):
        mask = features[:, 2:3].bool()
        pattern, descriptors, _ = factor_measurements(features, mask, self.rms_floor)
        encoded = self.core.encoder(pattern.flatten(2))
        pooled = F.adaptive_avg_pool1d(encoded, self.core.pooled_length)
        pooled = self.conditioner(pooled, features, mask, descriptors)
        decoded = self.core.adapter(pooled).reshape(features.shape[0], self.core.latent_channels,
            self.core.latent_size, self.core.latent_size, self.core.latent_size)
        for stage, modulation in enumerate(self.postnorm):
            offset = stage * 3
            decoded = self.core.decoder[offset + 1](self.core.decoder[offset](decoded))
            decoded = modulation(decoded, descriptors)
            decoded = self.core.decoder[offset + 2](decoded)
        return self.core.output(decoded), decoded


class PostNormAmplitudeMSRNN(PatternAmplitudeMSRNN):
    def __init__(self, config=None):
        config = dict(config or {})
        adjustment = config.get("postnorm_amplitude")
        if not isinstance(adjustment, dict) or adjustment.get("name") != ADJUSTMENT:
            raise ValueError("Explicit postnorm_amplitude_v1 architecture required")
        super().__init__(config)
        # All shared modules retain the baseline's seeded initialization. New
        # affine layers initialize last, after the existing architecture.
        # Added initialization must not change the paired training sample order,
        # which the unchanged trainer draws from the global CPU torch RNG.
        with torch.random.fork_rng(devices=[]):
            self.coarse = PostNormCoarse(self.coarse, adjustment)
