"""Attention gate for 3D U-Net skip connections."""
from __future__ import annotations

import torch
from torch import nn


def _conv1x1(in_channels: int, out_channels: int) -> nn.Conv3d:
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)


class AttentionGate3D(nn.Module):
    """Attention gate that filters skip features based on decoder context."""

    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.theta = _conv1x1(in_channels, inter_channels)
        self.phi = _conv1x1(gating_channels, inter_channels)
        self.psi = _conv1x1(inter_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        act = self.relu(theta_x + phi_g)
        psi = self.sigmoid(self.psi(act))
        return x * psi
