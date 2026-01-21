import torch

from models.fcernn import FCERNN
from utils.io import load_config, complex_to_channels, tensor_to_flattened


def run_shape_test(config_path="configs/fcernn.yaml"):
    config = load_config(config_path)
    model = FCERNN(config)
    num_freq = config["input"]["num_freq"]
    tx = config["input"]["tx"]
    rx = config["input"]["rx"]
    real = torch.randn(num_freq, tx, rx)
    imag = torch.randn(num_freq, tx, rx)
    measurement = torch.complex(real, imag)
    channels = complex_to_channels(measurement)
    if config["input"].get("use_mask_channel", False):
        mask = torch.ones_like(real).unsqueeze(0)
        channels = torch.cat([channels, mask], dim=0)
    if config["input"].get("input_format", "tensor") == "flattened":
        inputs = tensor_to_flattened(channels)
    else:
        inputs = channels
    if inputs.ndim == 3:
        inputs = inputs.view(1, inputs.size(0), -1)
    outputs = model(inputs)
    expected = (
        1,
        2,
        config["output"]["depth"],
        config["output"]["height"],
        config["output"]["width"],
    )
    assert outputs.shape == expected, f"Expected {expected}, got {outputs.shape}"
    print("Shape test passed")


if __name__ == "__main__":
    run_shape_test()
