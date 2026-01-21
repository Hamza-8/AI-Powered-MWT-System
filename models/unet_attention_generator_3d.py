"""Attention-gated 3D U-Net generator for decluttering."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils import spectral_norm

from models.attention_gate_3d import AttentionGate3D


class DoubleConv3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        spectral_norm_on: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = tuple(((k - 1) // 2) * dilation for k in kernel_size)
        conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        if spectral_norm_on:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)
        self.block = nn.Sequential(
            conv1,
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            conv2,
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock3D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        spectral_norm_on: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = tuple(((k - 1) // 2) * dilation for k in kernel_size)
        conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        if spectral_norm_on:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)
        self.block = nn.Sequential(
            conv1,
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            conv2,
            nn.InstanceNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class AttentionUNetGenerator3D(nn.Module):
    """3D attention-gated U-Net with residual bottleneck blocks."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        feature_channels: Sequence[int] = (32, 64, 128, 256),
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        downsample_f: bool = False,
        res_blocks: int = 4,
        bottleneck_dilation: int = 1,
        attention: bool = True,
        spectral_norm_on: bool = False,
    ) -> None:
        super().__init__()
        self.downsample_f = downsample_f
        self.attention = attention

        self.encoder: nn.ModuleList = nn.ModuleList()
        self.downsamplers: nn.ModuleList = nn.ModuleList()

        prev_channels = in_channels
        for ch in feature_channels:
            self.encoder.append(DoubleConv3D(prev_channels, ch, kernel_size, spectral_norm_on))
            stride = (2, 2, 2) if downsample_f else (1, 2, 2)
            self.downsamplers.append(nn.Conv3d(ch, ch, kernel_size=3, stride=stride, padding=1))
            prev_channels = ch

        self.bottleneck = nn.Sequential(
            *[
                ResBlock3D(feature_channels[-1], kernel_size, spectral_norm_on, bottleneck_dilation)
                for _ in range(res_blocks)
            ]
        )

        self.upsamplers: nn.ModuleList = nn.ModuleList()
        self.decoder: nn.ModuleList = nn.ModuleList()
        self.attention_gates: nn.ModuleList = nn.ModuleList()

        for ch in reversed(feature_channels):
            stride = (2, 2, 2) if downsample_f else (1, 2, 2)
            self.upsamplers.append(
                nn.ConvTranspose3d(ch, ch, kernel_size=stride, stride=stride)
            )
            if attention:
                self.attention_gates.append(AttentionGate3D(ch, ch, ch // 2))
            else:
                self.attention_gates.append(nn.Identity())
            self.decoder.append(DoubleConv3D(ch * 2, ch, kernel_size, spectral_norm_on))

        self.output_conv = nn.Sequential(
            nn.Conv3d(feature_channels[0], out_channels, kernel_size=1),
            nn.Tanh(),
        )

    @staticmethod
    def _match_size(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Pad/crop src to match ref spatially."""
        diff_f = ref.shape[2] - src.shape[2]
        diff_h = ref.shape[3] - src.shape[3]
        diff_w = ref.shape[4] - src.shape[4]
        pad = [
            diff_w // 2,
            diff_w - diff_w // 2,
            diff_h // 2,
            diff_h - diff_h // 2,
            diff_f // 2,
            diff_f - diff_f // 2,
        ]
        if any(p > 0 for p in pad):
            src = nn.functional.pad(src, pad)
        if any(p < 0 for p in pad):
            f_start = (-pad[4])
            h_start = (-pad[2])
            w_start = (-pad[0])
            src = src[:, :, f_start : f_start + ref.shape[2], h_start : h_start + ref.shape[3], w_start : w_start + ref.shape[4]]
        return src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("Expected input shape [B, 2, F, A, A]")

        skips: List[torch.Tensor] = []
        out = x
        for block, down in zip(self.encoder, self.downsamplers):
            out = block(out)
            skips.append(out)
            out = down(out)

        out = self.bottleneck(out)

        for up, attn, dec, skip in zip(self.upsamplers, self.attention_gates, self.decoder, reversed(skips)):
            out = up(out)
            out = self._match_size(out, skip)
            gated = attn(skip, out) if self.attention else skip
            out = dec(torch.cat([out, gated], dim=1))

        return self.output_conv(out)
