"""Run inference with a trained decluttering generator."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models.unet_attention_generator_3d import AttentionUNetGenerator3D
from utils.io import load_array, save_array
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer decluttered signals")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    data = load_array(args.input)
    if data.shape[0] != 2:
        raise ValueError("Input must have shape [2, F, A, A]")

    tensor = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)

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

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "G" in checkpoint:
        G.load_state_dict(checkpoint["G"])
    else:
        G.load_state_dict(checkpoint)

    G.eval()
    with torch.no_grad():
        output = G(tensor).squeeze(0).cpu().numpy()

    save_array(args.output, output)
    print(f"Saved decluttered output to {Path(args.output)}")


if __name__ == "__main__":
    main()
