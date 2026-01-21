import math

import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def nrmse(pred, target):
    return rmse(pred, target) / (torch.mean(torch.abs(target)) + 1e-8)


def psnr(pred, target, max_val=1.0):
    mse_val = torch.mean((pred - target) ** 2)
    if mse_val <= 0:
        return torch.tensor(float("inf"), device=pred.device)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse_val)


def compute_metrics(pred, target, use_psnr=False):
    metrics = {
        "mae": mae(pred, target).item(),
        "rmse": rmse(pred, target).item(),
        "nrmse": nrmse(pred, target).item(),
    }
    if use_psnr:
        metrics["psnr"] = psnr(pred, target).item()
    return metrics


def gradient_loss(pred, target):
    def gradient(x):
        dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dx, dy, dz

    pred_grads = gradient(pred)
    target_grads = gradient(target)
    loss = 0.0
    for pg, tg in zip(pred_grads, target_grads):
        loss += torch.mean(torch.abs(pg - tg))
    return loss
