import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def complex_to_channels(x):
    return torch.stack([x.real, x.imag], dim=0)


def channels_to_complex(x):
    return torch.complex(x[0], x[1])


def tensor_to_flattened(x):
    channels, freq, tx, rx = x.shape
    flattened = x.reshape(channels, freq * tx * rx)
    return flattened


def flattened_to_tensor(x, freq, tx, rx):
    channels = x.shape[0]
    return x.view(channels, freq, tx, rx)
