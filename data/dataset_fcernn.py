import random

import torch
from torch.utils.data import Dataset

from utils.subsample import apply_budget_subsampling
from utils.io import complex_to_channels, tensor_to_flattened
from models.fcernn import select_block


class DummyFCERNNData(Dataset):
    def __init__(self, config, split="train"):
        self.config = config
        self.split = split
        dataset_cfg = config["dataset"]
        self.num_samples = dataset_cfg[f"{split}_size"]
        self.input_cfg = config["input"]
        self.output_cfg = config["output"]
        self.training_cfg = config["training"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        num_freq = self.input_cfg["num_freq"]
        tx = self.input_cfg["tx"]
        rx = self.input_cfg["rx"]

        real = torch.randn(num_freq, tx, rx)
        imag = torch.randn(num_freq, tx, rx)
        measurement = torch.complex(real, imag)

        mask = None
        if self.training_cfg.get("budget_augmentation", False) and self.split == "train":
            budgets = self.training_cfg.get("budgets", [1.0])
            budget = random.choice(budgets)
            measurement, mask = apply_budget_subsampling(measurement, budget)

        channels = complex_to_channels(measurement)
        if self.input_cfg.get("use_mask_channel", False):
            if mask is None:
                mask = torch.ones_like(measurement.real)
            mask = mask.unsqueeze(0)
            channels = torch.cat([channels, mask], dim=0)

        if self.input_cfg.get("input_format", "tensor") == "flattened":
            input_tensor = tensor_to_flattened(channels)
        else:
            input_tensor = channels

        depth = self.output_cfg["depth"]
        height = self.output_cfg["height"]
        width = self.output_cfg["width"]
        target = torch.randn(2, depth, height, width)

        if self.output_cfg.get("block_mode", False):
            block_splits = self.output_cfg["block_splits"]
            block_index = self.output_cfg["block_index"]
            target = select_block(target.unsqueeze(0), block_splits, block_index).squeeze(0)

        return input_tensor, target
