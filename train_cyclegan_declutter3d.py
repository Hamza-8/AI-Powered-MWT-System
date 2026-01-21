"""Train a 3D CycleGAN for microwave clutter removal."""
from __future__ import annotations

import argparse
import json
import math
from itertools import cycle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import yaml

from data.dummy_mwt_signals import DummyMWTConfig, DummyMWTSignalsDataset
from models.patchgan_discriminator_3d import PatchGANDiscriminator3D
from models.unet_attention_generator_3d import AttentionUNetGenerator3D
from utils.metrics import psnr_volume, ssim_volume
from utils.replay_buffer import ReplayBuffer


def build_generators(cfg: Dict) -> Tuple[nn.Module, nn.Module]:
    gen_cfg = cfg["generator"]
    G = AttentionUNetGenerator3D(
        in_channels=2,
        out_channels=2,
        feature_channels=gen_cfg["channels"],
        kernel_size=tuple(gen_cfg["kernel_size"]),
        downsample_f=gen_cfg["downsample_f"],
        res_blocks=gen_cfg["res_blocks"],
        bottleneck_dilation=gen_cfg["bottleneck_dilation"],
        attention=gen_cfg["attention"],
        spectral_norm_on=gen_cfg["spectral_norm"],
    )
    F = AttentionUNetGenerator3D(
        in_channels=2,
        out_channels=2,
        feature_channels=gen_cfg["channels"],
        kernel_size=tuple(gen_cfg["kernel_size"]),
        downsample_f=gen_cfg["downsample_f"],
        res_blocks=gen_cfg["res_blocks"],
        bottleneck_dilation=gen_cfg["bottleneck_dilation"],
        attention=gen_cfg["attention"],
        spectral_norm_on=gen_cfg["spectral_norm"],
    )
    return G, F


def build_discriminators(cfg: Dict) -> Tuple[nn.Module, nn.Module]:
    disc_cfg = cfg["discriminator"]
    Dx = PatchGANDiscriminator3D(
        in_channels=2,
        feature_channels=disc_cfg["channels"],
        spectral_norm_on=disc_cfg["spectral_norm"],
        downsample_f=disc_cfg["downsample_f"],
    )
    Dy = PatchGANDiscriminator3D(
        in_channels=2,
        feature_channels=disc_cfg["channels"],
        spectral_norm_on=disc_cfg["spectral_norm"],
        downsample_f=disc_cfg["downsample_f"],
    )
    return Dx, Dy


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def save_sample(output_dir: Path, tag: str, data: torch.Tensor) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_np = data.detach().cpu().numpy()
    np.save(output_dir / f"{tag}.npy", data_np)
    magnitude = np.sqrt(data_np[:, 0] ** 2 + data_np[:, 1] ** 2)
    projection = magnitude.max(axis=1)
    for idx, proj in enumerate(projection):
        proj_norm = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
        from matplotlib import pyplot as plt

        plt.imsave(output_dir / f"{tag}_{idx}.png", proj_norm, cmap="magma")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 3D CycleGAN for decluttering")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--demo", action="store_true", help="Run a tiny CPU-friendly demo")
    parser.add_argument("--overfit_n", type=int, default=0, help="Overfit on N samples if > 0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    length = data_cfg["length"]
    if args.demo:
        length = min(length, 40)
    if args.overfit_n > 0:
        length = args.overfit_n

    dummy_cfg = DummyMWTConfig(
        f=data_cfg["f"],
        a=data_cfg["a"],
        snr_db=data_cfg["snr_db"],
        length=length,
    )

    x_dataset = DummyMWTSignalsDataset(dummy_cfg, domain="mixed")
    y_dataset = DummyMWTSignalsDataset(dummy_cfg, domain="target")

    batch_size = cfg["training"]["batch_size"]
    if args.demo:
        batch_size = min(batch_size, 2)

    x_loader = DataLoader(x_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    y_loader = DataLoader(y_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G, F = build_generators(cfg)
    Dx, Dy = build_discriminators(cfg)

    G.apply(init_weights)
    F.apply(init_weights)
    Dx.apply(init_weights)
    Dy.apply(init_weights)

    G.to(device)
    F.to(device)
    Dx.to(device)
    Dy.to(device)

    adv_loss_fn = nn.MSELoss()
    cycle_loss_fn = nn.L1Loss()
    identity_loss_fn = nn.L1Loss()

    optimizer_g = torch.optim.Adam(
        list(G.parameters()) + list(F.parameters()),
        lr=cfg["training"]["lr"],
        betas=tuple(cfg["training"]["betas"]),
    )
    optimizer_dx = torch.optim.Adam(
        Dx.parameters(), lr=cfg["training"]["lr"], betas=tuple(cfg["training"]["betas"])
    )
    optimizer_dy = torch.optim.Adam(
        Dy.parameters(), lr=cfg["training"]["lr"], betas=tuple(cfg["training"]["betas"])
    )

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lambda e: 1.0)
    scheduler_dx = torch.optim.lr_scheduler.LambdaLR(optimizer_dx, lr_lambda=lambda e: 1.0)
    scheduler_dy = torch.optim.lr_scheduler.LambdaLR(optimizer_dy, lr_lambda=lambda e: 1.0)

    buffer_x = ReplayBuffer(cfg["training"]["buffer_size"])
    buffer_y = ReplayBuffer(cfg["training"]["buffer_size"])

    lambda_cyc = cfg["loss"]["lambda_cyc"]
    lambda_id = cfg["loss"]["lambda_id"]

    scaler = GradScaler(enabled=cfg["training"]["amp"] and device.type == "cuda")

    output_root = Path(cfg["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = cfg["training"]["steps_per_epoch"]
    if args.demo:
        steps_per_epoch = min(steps_per_epoch, 5)

    log = []

    for epoch in range(cfg["training"]["epochs"]):
        for step, ((x, _), (y, _)) in enumerate(zip(x_loader, cycle(y_loader))):
            if step >= steps_per_epoch:
                break
            x = x.to(device)
            y = y.to(device)

            with autocast(enabled=scaler.is_enabled()):
                fake_y = G(x)
                fake_x = F(y)
                rec_x = F(fake_y)
                rec_y = G(fake_x)

                pred_fake_y = Dy(fake_y)
                pred_fake_x = Dx(fake_x)
                valid_y = torch.ones_like(pred_fake_y)
                valid_x = torch.ones_like(pred_fake_x)
                fake_y_label = torch.zeros_like(pred_fake_y)
                fake_x_label = torch.zeros_like(pred_fake_x)

                loss_g_adv = adv_loss_fn(pred_fake_y, valid_y)
                loss_f_adv = adv_loss_fn(pred_fake_x, valid_x)
                loss_cycle = cycle_loss_fn(rec_x, x) + cycle_loss_fn(rec_y, y)

                loss_identity = identity_loss_fn(G(y), y) + identity_loss_fn(F(x), x)
                loss_g_total = loss_g_adv + loss_f_adv + lambda_cyc * loss_cycle + lambda_id * loss_identity

            optimizer_g.zero_grad()
            scaler.scale(loss_g_total).backward()
            scaler.step(optimizer_g)
            scaler.update()

            with torch.no_grad():
                fake_x_buffered = buffer_x.push_and_pop(fake_x.detach())
                fake_y_buffered = buffer_y.push_and_pop(fake_y.detach())

            with autocast(enabled=scaler.is_enabled()):
                loss_dx = 0.5 * (
                    adv_loss_fn(Dx(x), valid_x) + adv_loss_fn(Dx(fake_x_buffered), fake_x_label)
                )
                loss_dy = 0.5 * (
                    adv_loss_fn(Dy(y), valid_y) + adv_loss_fn(Dy(fake_y_buffered), fake_y_label)
                )

            optimizer_dx.zero_grad()
            scaler.scale(loss_dx).backward()
            scaler.step(optimizer_dx)

            optimizer_dy.zero_grad()
            scaler.scale(loss_dy).backward()
            scaler.step(optimizer_dy)

            scaler.update()

            psnr = psnr_volume(fake_y, y).item()
            ssim = ssim_volume(fake_y, y).item()
            log.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "loss_g": loss_g_total.item(),
                    "loss_dx": loss_dx.item(),
                    "loss_dy": loss_dy.item(),
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

            if step % cfg["training"]["sample_every"] == 0:
                sample_dir = output_root / "samples" / f"epoch_{epoch}_step_{step}"
                save_sample(sample_dir, "x_mixed", x)
                save_sample(sample_dir, "y_target", y)
                save_sample(sample_dir, "g_x", fake_y)
                save_sample(sample_dir, "f_y", fake_x)
                save_sample(sample_dir, "rec_x", rec_x)
                save_sample(sample_dir, "rec_y", rec_y)

        scheduler_g.step()
        scheduler_dx.step()
        scheduler_dy.step()

        ckpt_dir = output_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"G": G.state_dict(), "F": F.state_dict()}, ckpt_dir / f"epoch_{epoch}.pt")

    with open(output_root / "training_log.json", "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)


if __name__ == "__main__":
    main()
