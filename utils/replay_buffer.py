"""Replay buffer to stabilize CycleGAN discriminators."""
from __future__ import annotations

import random
from typing import List

import torch


class ReplayBuffer:
    def __init__(self, max_size: int = 50) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self.data: List[torch.Tensor] = []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        output = []
        for item in batch:
            item = item.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(item)
                output.append(item)
            else:
                if random.random() > 0.5:
                    idx = random.randrange(len(self.data))
                    stored = self.data[idx]
                    self.data[idx] = item
                    output.append(stored)
                else:
                    output.append(item)
        return torch.cat(output, dim=0)
