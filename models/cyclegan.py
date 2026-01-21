"""CycleGAN wrapper holding generators and discriminators."""

from __future__ import annotations

from torch import nn

from models.patchgan_discriminator_3d import PatchGANDiscriminator3D
from models.unet_attention_generator_3d import AttentionUNetGenerator3D


class CycleGAN3D(nn.Module):
    """Bundle of generators and discriminators for clutter removal.

    This wrapper exists mainly for checkpointing convenience.
    Training code can also use the individual modules directly.
    """

    def __init__(
        self,
        G: AttentionUNetGenerator3D,
        F: AttentionUNetGenerator3D,
        Dx: PatchGANDiscriminator3D,
        Dy: PatchGANDiscriminator3D,
    ) -> None:
        super().__init__()
        self.G = G
        self.F = F
        self.Dx = Dx
        self.Dy = Dy

    def forward(self, x, y=None):
        """Optional convenience forward.
        If y is provided, returns a dict of common CycleGAN tensors.
        """
        out = {}
        out["x"] = x
        out["fake_y"] = self.G(x)
        out["rec_x"] = self.F(out["fake_y"])

        if y is not None:
            out["y"] = y
            out["fake_x"] = self.F(y)
            out["rec_y"] = self.G(out["fake_x"])
        return out
