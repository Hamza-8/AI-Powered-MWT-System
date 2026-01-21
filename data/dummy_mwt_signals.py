"""Dummy microwave head imaging signal generator.

Creates unpaired domains:
- Domain X: mixed (target + clutter)
- Domain Y: target-only
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _normalize_to_unit(x: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(x))
    if max_val < 1e-8:
        return x
    return x / max_val


def _smooth_random_curve(length: int, scale: float = 1.0) -> np.ndarray:
    points = np.random.randn(length) * scale
    kernel = np.exp(-0.5 * (np.linspace(-2, 2, 9) ** 2))
    kernel = kernel / kernel.sum()
    return np.convolve(points, kernel, mode="same")


def _low_rank_matrix(a: int, rank: int = 3) -> np.ndarray:
    u = np.random.randn(a, rank)
    v = np.random.randn(a, rank)
    return u @ v.T


def _make_targets(f: int, a: int) -> np.ndarray:
    base = np.zeros((f, a, a), dtype=np.complex64)
    freq_profile = _smooth_random_curve(f, scale=1.0)
    freq_profile = freq_profile / (np.max(np.abs(freq_profile)) + 1e-6)
    low_rank = _low_rank_matrix(a, rank=3)
    low_rank = low_rank / (np.max(np.abs(low_rank)) + 1e-6)
    for idx in range(f):
        phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        blob = low_rank * (0.4 + 0.6 * np.random.rand())
        base[idx] = freq_profile[idx] * blob * phase
    return base


def _make_clutter(f: int, a: int) -> np.ndarray:
    clutter = np.zeros((f, a, a), dtype=np.complex64)
    boundary_profile = _smooth_random_curve(f, scale=2.0)
    boundary_profile = boundary_profile / (np.max(np.abs(boundary_profile)) + 1e-6)
    common = np.ones((a, a), dtype=np.float32)
    for idx in range(f):
        phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        clutter[idx] += 2.0 * boundary_profile[idx] * common * phase
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, a), np.linspace(-1, 1, a), indexing="ij")
    spatial = np.exp(-2.5 * (grid_x**2 + grid_y**2))
    ripple = np.sin(2 * np.pi * grid_x) + np.cos(2 * np.pi * grid_y)
    spatial_pattern = spatial + 0.5 * ripple
    for idx in range(f):
        freq_ripple = 1.0 + 0.2 * np.sin(2 * np.pi * idx / max(f - 1, 1))
        phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
        clutter[idx] += freq_ripple * spatial_pattern * phase
    return clutter


def _add_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(np.abs(x) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / (snr_linear + 1e-6)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    drift = _smooth_random_curve(x.shape[0], scale=np.sqrt(noise_power))
    drift = drift[:, None, None]
    noise += drift
    return x + noise


@dataclass
class DummyMWTConfig:
    f: int = 50
    a: int = 24
    snr_db: float = 20.0
    length: int = 200


class DummyMWTSignalsDataset(Dataset):
    """Dataset producing either mixed or target-only volumes."""

    def __init__(self, config: DummyMWTConfig, domain: str = "mixed") -> None:
        if domain not in {"mixed", "target"}:
            raise ValueError("domain must be 'mixed' or 'target'")
        self.config = config
        self.domain = domain

    def __len__(self) -> int:
        return self.config.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = idx
        f = self.config.f
        a = self.config.a
        target = _make_targets(f, a)
        if self.domain == "mixed":
            clutter = _make_clutter(f, a)
            mixed = target + clutter
            mixed = _add_noise(mixed, self.config.snr_db)
            mixed = _normalize_to_unit(mixed)
            real = np.stack([mixed.real, mixed.imag], axis=0)
        else:
            target = _add_noise(target, self.config.snr_db + 5.0)
            target = _normalize_to_unit(target)
            real = np.stack([target.real, target.imag], axis=0)

        assert real.shape == (2, f, a, a)
        return torch.from_numpy(real.astype(np.float32)), torch.tensor(0)
