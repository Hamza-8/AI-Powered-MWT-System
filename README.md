# AI-Powered-MWT-System: CycleGAN Decluttering Module

This repository implements a **self-supervised, unpaired CycleGAN** for microwave head imaging clutter removal inspired by Lai et al. The goal is to suppress dominant clutter from head/skull boundary reflections and tissue heterogeneity so that downstream anomaly detection and FCERNN reconstruction see a cleaner measurement volume. The model learns two mappings in **measurement space**:

- **Domain X (mixed):** clutter + target signals
- **Domain Y (decluttered):** target-only signals

A pair of generators (**G: X→Y**, **F: Y→X**) and discriminators (**Dy**, **Dx**) train with adversarial, cycle-consistency, and identity losses, enabling **unpaired** training on large datasets.

## Data representation

We operate on complex S-parameters in the 0.5–2 GHz band using a configurable antenna count **A** (default 24) and frequency samples **F** (default 50).

- Complex numbers are represented as **2 channels** (real, imag).
- Frequency is treated as a 3D depth axis.
- Tensor shape: **[B, 2, F, A, A]**.

The output of the decluttering generator matches the input shape exactly, preserving the spectral and antenna-pair structure required by downstream reconstruction models.

## Model architecture

**Generators:** Attention-Gated 3D U-Net with residual bottleneck blocks.
- 4 spatial scales (A×A): e.g., 24→12→6→3
- Frequency dimension preserved by default (configurable)
- Attention gates on skip connections
- Residual blocks at bottleneck (default 4), optional dilation
- Output: 1×1×1 conv to 2 channels + Tanh

**Discriminators:** 3D PatchGAN operating on [2, F, A, A]
- Stride-2 in spatial dimensions, stride-1 in frequency
- Spectral normalization enabled

## Repository structure

```
models/
  attention_gate_3d.py
  unet_attention_generator_3d.py
  patchgan_discriminator_3d.py
  cyclegan.py

data/
  dummy_mwt_signals.py

utils/
  replay_buffer.py
  metrics.py
  io.py

configs/
  cyclegan_declutter3d.yaml

train_cyclegan_declutter3d.py
infer_declutter3d.py
```

## Quick start (CPU demo)

The repository ships with a dummy signal generator so you can run without external data.

```bash
python train_cyclegan_declutter3d.py --config configs/cyclegan_declutter3d.yaml --demo
```

### Expected outputs

```
outputs/
  checkpoints/
    epoch_0.pt
    epoch_1.pt
  samples/
    epoch_0_step_0/
      x_mixed.npy
      y_target.npy
      g_x.npy
      f_y.npy
      rec_x.npy
      rec_y.npy
      x_mixed_0.png
      ...
  training_log.json
```

## Inference

```bash
python infer_declutter3d.py \
  --config configs/cyclegan_declutter3d.yaml \
  --checkpoint outputs/checkpoints/epoch_1.pt \
  --input path/to/mixed.npy \
  --output path/to/decluttered.npy
```

## Notes

- Training is **unpaired**; mixed and target-only samples are not aligned.
- **TODO** markers in configs indicate where exact paper hyperparameters can be inserted when known.
- The dummy data generator creates target-only signatures (smooth blobs + low-rank antenna structure) and clutter components (boundary reflections + multipath-like ripples + noise).
