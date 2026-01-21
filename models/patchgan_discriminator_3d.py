"""3D PatchGAN discriminator for microwave decluttering."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        feature_channels: Sequence[int] = (32, 64, 128, 256),
        spectral_norm_on: bool = True,
        downsample_f: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        prev_channels = in_channels
        for idx, ch in enumerate(feature_channels):
            stride = (2, 2, 2) if downsample_f else (1, 2, 2)
            conv = nn.Conv3d(prev_channels, ch, kernel_size=4, stride=stride, padding=1)
            if spectral_norm_on:
                conv = spectral_norm(conv)
            layers.append(conv)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channels = ch
        final_conv = nn.Conv3d(prev_channels, 1, kernel_size=3, stride=1, padding=1)
        if spectral_norm_on:
            final_conv = spectral_norm(final_conv)
        layers.append(final_conv)
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("Expected input shape [B, 2, F, A, A]")
        return self.model(x)
