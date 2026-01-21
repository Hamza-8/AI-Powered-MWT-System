"""Metrics for monitoring decluttering quality."""
from __future__ import annotations

import torch


def _flatten_volume(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def psnr_volume(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Compute PSNR for complex-valued volumes treated as two-channel tensors."""
    if pred.shape != target.shape:
        raise ValueError("Pred/target shapes must match for PSNR")
    pred_flat = _flatten_volume(pred)
    target_flat = _flatten_volume(target)
    mse = torch.mean((pred_flat - target_flat) ** 2, dim=1)
    psnr = 20 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10 * torch.log10(mse + 1e-8)
    return psnr.mean()


def ssim_volume(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Compute a simple global SSIM over 3D volumes (per channel, averaged)."""
    if pred.shape != target.shape:
        raise ValueError("Pred/target shapes must match for SSIM")
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    pred_flat = _flatten_volume(pred)
    target_flat = _flatten_volume(target)

    mu_x = pred_flat.mean(dim=1)
    mu_y = target_flat.mean(dim=1)
    sigma_x = pred_flat.var(dim=1, unbiased=False)
    sigma_y = target_flat.var(dim=1, unbiased=False)
    sigma_xy = ((pred_flat - mu_x[:, None]) * (target_flat - mu_y[:, None])).mean(dim=1)

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    ssim = numerator / (denominator + 1e-8)
    return ssim.mean()
