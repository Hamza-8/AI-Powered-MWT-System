import torch
from torch import nn


class FCERNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        input_cfg = config["input"]
        output_cfg = config["output"]

        in_channels = 2 + int(input_cfg.get("use_mask_channel", False))
        encoder_channels = model_cfg["encoder_channels"]
        encoder_kernels = model_cfg["encoder_kernels"]
        encoder_strides = model_cfg["encoder_strides"]
        encoder_padding = model_cfg.get("encoder_padding")

        if len(encoder_channels) != 8:
            raise ValueError("Encoder must have 8 layers per paper.")

        self.encoder = nn.ModuleList()
        prev_channels = in_channels
        for idx, out_channels in enumerate(encoder_channels):
            kernel = encoder_kernels[idx]
            stride = encoder_strides[idx]
            padding = kernel // 2 if encoder_padding is None else encoder_padding[idx]
            self.encoder.append(
                nn.Sequential(
                    nn.Conv1d(prev_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = out_channels

        reshape_cfg = model_cfg["reshape"]
        self.reshape_channels = reshape_cfg["channels"]
        self.reshape_depth = reshape_cfg["depth"]
        self.reshape_height = reshape_cfg["height"]
        self.reshape_width = reshape_cfg["width"]

        decoder_channels = model_cfg["decoder_channels"]
        decoder_kernels = model_cfg["decoder_kernels"]
        decoder_strides = model_cfg["decoder_strides"]
        decoder_padding = model_cfg["decoder_padding"]
        decoder_output_padding = model_cfg["decoder_output_padding"]

        decoder_layers = []
        in_ch = self.reshape_channels
        for idx, out_ch in enumerate(decoder_channels):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch,
                        out_ch,
                        kernel_size=decoder_kernels[idx],
                        stride=decoder_strides[idx],
                        padding=decoder_padding[idx],
                        output_padding=decoder_output_padding[idx],
                    ),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        decoder_layers.append(
            nn.ConvTranspose3d(
                in_ch,
                2,
                kernel_size=decoder_kernels[-1],
                stride=decoder_strides[-1],
                padding=decoder_padding[-1],
                output_padding=decoder_output_padding[-1],
            )
        )
        self.decoder = nn.Sequential(*decoder_layers)

        self.output_depth = output_cfg["depth"]
        self.output_height = output_cfg["height"]
        self.output_width = output_cfg["width"]

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        batch, channels, length = x.shape
        expected_length = self.reshape_depth * self.reshape_height * self.reshape_width
        if channels != self.reshape_channels or length != expected_length:
            raise ValueError(
                "Encoder output shape mismatch. "
                f"Expected channels {self.reshape_channels} and length {expected_length}, "
                f"got channels {channels} and length {length}."
            )
        x = x.view(batch, self.reshape_channels, self.reshape_depth, self.reshape_height, self.reshape_width)
        x = self.decoder(x)
        if x.shape[2:] != (self.output_depth, self.output_height, self.output_width):
            raise ValueError(
                "Decoder output shape mismatch. "
                f"Expected {(self.output_depth, self.output_height, self.output_width)}, got {x.shape[2:]}"
            )
        return x


def split_blocks(volume, block_splits):
    splits_d, splits_h, splits_w = block_splits
    _, _, depth, height, width = volume.shape
    d_step = depth // splits_d
    h_step = height // splits_h
    w_step = width // splits_w
    blocks = []
    for d in range(splits_d):
        for h in range(splits_h):
            for w in range(splits_w):
                block = volume[
                    :,
                    :,
                    d * d_step : (d + 1) * d_step,
                    h * h_step : (h + 1) * h_step,
                    w * w_step : (w + 1) * w_step,
                ]
                blocks.append(block)
    return blocks


def select_block(volume, block_splits, block_index):
    blocks = split_blocks(volume, block_splits)
    if block_index >= len(blocks):
        raise IndexError("Block index out of range.")
    return blocks[block_index]


def stitch_blocks(blocks, block_splits):
    splits_d, splits_h, splits_w = block_splits
    batch, channels, d_block, h_block, w_block = blocks[0].shape
    depth = d_block * splits_d
    height = h_block * splits_h
    width = w_block * splits_w
    volume = torch.zeros((batch, channels, depth, height, width), device=blocks[0].device)
    idx = 0
    for d in range(splits_d):
        for h in range(splits_h):
            for w in range(splits_w):
                volume[
                    :,
                    :,
                    d * d_block : (d + 1) * d_block,
                    h * h_block : (h + 1) * h_block,
                    w * w_block : (w + 1) * w_block,
                ] = blocks[idx]
                idx += 1
    return volume
