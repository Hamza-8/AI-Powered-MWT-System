# 3-D MSRNN (PyTorch 2.x)

This repo provides a complete runnable PyTorch 2.x implementation of the 3-D FCERNN reconstruction network from the Diagnostics 2022 paper **"A 3-D Full Convolution Electromagnetic Reconstruction Neural Network (3-D FCERNN) for Fast Super-Resolution Electromagnetic Inversion of Human Brain"**. It runs end-to-end on dummy data with correct shapes and includes training, evaluation, checkpointing, TensorBoard logging, and block-based output stitching.

## Features

- Fully convolutional (no fully connected layers).
- Supports tensor and flattened input formats for complex scattered-field measurements.
- Optional mask channel for scan-budget experiments.
- Strided Conv1D encoder (8 layers) and ConvTranspose3D decoder.
- Block output mode with stitching utilities.
- Mixed precision training and cosine LR scheduler.
- Metrics: MAE, RMSE, nRMSE, optional PSNR.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio tensorboard pyyaml
```

## Training

```bash
python train_fcernn.py --config configs/fcernn.yaml
```

TensorBoard logs are written to `outputs/tensorboard`.

## Evaluation

```bash
python eval_fcernn.py --config configs/fcernn.yaml --checkpoint outputs/checkpoints/best.pt
```

## Shape Test

```bash
python tests/test_shapes.py
```

## Input Formats

- **Tensor format**: complex tensor `[F, 24, 24]`.
- **Flattened format**: `x ∈ R^(2×M)` where `M = F*24*24`.

The utilities in `utils/io.py` handle:
- complex → 2-channel real tensor
- tensor → flattened representation

## Block Output Mode

Set in `configs/fcernn.yaml`:

```yaml
output:
  block_mode: true
  block_splits: [2, 2, 2]
  block_index: 0
```

Training uses the selected `block_index` as the target. During evaluation, use `models.fcernn.stitch_blocks` to reassemble the full volume.

## Notes on Paper Fidelity

- Encoder kernel sizes and decoder kernel/stride settings are exposed in `configs/fcernn.yaml`. The paper's figure should be used to finalize these values. TODO comments are included in the config to highlight ambiguous details.
