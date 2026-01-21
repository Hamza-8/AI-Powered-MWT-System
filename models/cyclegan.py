"""CycleGAN wrapper holding generators and discriminators."""
from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from models.patchgan_discriminator_3d import PatchGANDiscriminator3D
from models.unet_attention_generator_3d import AttentionUNetGenerator3D


@dataclass
class CycleGAN3D(nn.Module):
    """Bundle of generators and discriminators for clutter removal."""

    G: AttentionUNetGenerator3D
    F: AttentionUNetGenerator3D
    Dx: PatchGANDiscriminator3D
    Dy: PatchGANDiscriminator3D

    def __post_init__(self) -> None:
        super().__init__()

    def forward(self, x):  # type: ignore[override]
        raise NotImplementedError("Use individual generators for inference.")
