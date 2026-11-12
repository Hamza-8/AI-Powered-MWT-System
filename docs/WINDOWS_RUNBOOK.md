# Windows / RTX 5070 Ti runbook

Target: Windows, Ryzen 9 9900X, 32 GB RAM, RTX 5070 Ti 16 GB. GPU validation has not run in the rebuild's CPU environment. Structural tests do not establish reconstruction accuracy, clinical skill or a latency target.

## Installation

Use PowerShell from the repository root. Create a fresh environment with a Python version supported by the current official PyTorch Windows/CUDA package. Install the current stable CUDA-enabled PyTorch command selected at <https://pytorch.org/get-started/locally/> for your driver/GPU; do **not** install `requirements-cpu.txt` on this machine. RTX 50-series support must be verified using a build that includes its CUDA architecture.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Run the CUDA-enabled torch installation command from pytorch.org here.
python -m pip install -e '.[test,hdf5]'
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
python -m pytest -q
```

CPU-only rebuild pins are recorded separately in `requirements-cpu.txt`. Reusing them on an RTX machine would deliberately install a CPU build. No GPU driver or CUDA compatibility is inferred merely from package installation.

## Bounded GPU forward/backward memory and AMP check

The profiler uses generated zero-valued structural inputs; the bounded optimizer command uses the real supplied pilot/exported inputs. Neither is scientific training. Verify the output records native 128³ shape, selected device, AMP mode and peak memory.

```powershell
python -m microscan convert --mat data/pilot/frequency_pilot.mat --output work/pilot_export --grid-size 128
python -m microscan validate --manifest work/pilot_export/manifest.json
python -m microscan profile --config configs/gpu_smoke.json --device cuda --steps 2
python -m microscan profile --config configs/gpu_smoke.json --device cuda --steps 2 --backward
python -m microscan train --config configs/gpu_smoke.json --max-steps 1
```

Use batch size1 and train stages sequentially. Full 3D activations/optimizer state can exceed a 16 GB budget depending on stage, AMP and correction depth; measured profile output governs changes. C must count Phi time and memory. The two-second target remains a measurement goal.

## Start/resume the agreed frequency experiment

The implementation task prepares these commands without launching the full experiment. Preserve the supplied `data/pilot/frequency_pilot.mat` and `frequency_subsets.csv`. Each K is a new independently initialized candidate; paired seeds and fixed ten-slot architecture remain identical. A bounded interrupted run is incomplete and cannot qualify for selection.

```powershell
python -m microscan.frequency prepare --config configs/frequency_pilot.json --mat data/pilot/frequency_pilot.mat --csv data/pilot/frequency_subsets.csv --output outputs/frequency
python -m microscan.frequency run --config configs/frequency_pilot.json --mat data/pilot/frequency_pilot.mat --csv data/pilot/frequency_subsets.csv --output outputs/frequency --device cuda --resume
```

The same `run ... --resume` command resumes identity-matched checkpoints and skips matching completed work. Editing data, source, preprocessing or protocol invalidates reuse. The full experiment uses K3–10 and three paired seeds, with reconstruction plus enhancement, A only, GAN bypassed and segmentation excluded.

After all candidates complete, explicitly request selection:

```powershell
python -m microscan.frequency select --config configs/frequency_pilot.json --mat data/pilot/frequency_pilot.mat --csv data/pilot/frequency_subsets.csv --output outputs/frequency
```

Review the selection report. An inconclusive ten-frequency baseline leaves selection pending. Only an established selection opens the explicit sealed-test command:

```powershell
python -m microscan.frequency test --config configs/frequency_pilot.json --mat data/pilot/frequency_pilot.mat --csv data/pilot/frequency_subsets.csv --output outputs/frequency --device cuda
```

Keep the serialized selected grid/subset with generated datasets and checkpoints. Copy a successful `outputs/frequency/frozen_frequencies.json` to the configured `configs/selected_frequencies.json` path. Do not hand-edit `pending_selection` into a selected result.

## Main staged training after selected frequencies and real data export

Prepare `data/main/manifest.json` following [the schema](DATA_EXPORT_SCHEMA.md); replace placeholder paths in configs with your actual exports. No main COMSOL solve or training campaign is launched automatically. The supplied pilot cannot qualify clinical segmentation or a scientific forward solver.

```powershell
python -m microscan validate --manifest data/main/manifest.json
python -m microscan train --config configs/train_gan.json
python -m microscan train --config configs/train_phi.json
python -m microscan train --config configs/train_coarse.json
python -m microscan train --config configs/main_A.json
python -m microscan train --config configs/main_A.json --resume
```

Train Phi independently and produce an independent held-out qualification report before setting the checkpoint/report paths in B or C. The gate requires genuine differential/noise/ROI evidence, disjoint groups, coverage and validated physical property bounds; edit `configs/qualify_phi.json` with those verified bounds first.

```powershell
python -m microscan qualify --checkpoint outputs/phi/best.pt --manifest data/main/manifest.json --config configs/qualify_phi.json --output outputs/phi/qualification.json --device cuda
```

A requires neither. Set `initialize_from` when moving from trained coarse inverse to enhancement, and from reconstruction to segmentation; those paths must refer to compatible v2 checkpoints. Run segmentation on detached reconstructed distributions, not only pristine targets. Joint fine-tuning is explicit and uses its configured freeze schedule.

```powershell
python -m microscan train --config configs/main_B.json
python -m microscan train --config configs/main_C.json
python -m microscan train --config configs/train_segmentation.json
python -m microscan train --config configs/train_joint.json
python -m microscan infer --checkpoint outputs/main_A/best.pt --manifest data/main/manifest.json --output outputs/predictions --split validation --device cuda
python -m microscan evaluate --checkpoint outputs/main_A/best.pt --manifest data/main/manifest.json --split validation --device cuda
```

B inference loads its inverse/enhancement checkpoint without Phi. C requires the compatible qualified operator explicitly. Legacy branch checkpoints with dummy-data architecture are not silently migrated; retain them as historical artifacts and retrain the new architecture.

## Optional offline preprocessing and scan export

These tools do not generate COMSOL cases or select frequencies. A subspace fit requires multiple independent training calibration geometries, each explicitly marked `subspace_cohort_eligible=true`, with matching hardware/calibration/frequency context. Noise repeats do not count as independent geometries. Lesion-preservation validation is still required before scientific use. Do not fit a population basis from the sphere pilot.

```powershell
python -m microscan.preprocessing fit-subspace --manifest data/calibration/manifest.json --context data/calibration/context.json --output outputs/calibration/subspace.npz --rank 1 --strength 0.25
python -m microscan.preprocessing export-scan --mask work/requested_mask.npy --frequencies work/frequencies_ghz.json --output outputs/requested_scan.json
```

`requested_mask.npy` is an explicit `[F,Rx,Tx]` acquisition request; the export enumerates physical Tx/Rx/frequency lists and budgets. Fewer requested transmitters can reduce excitations; discarding receiver channels after a full solve does not save already performed solves.
