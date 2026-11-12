# Rebuild handoff

This is a new implementation from the two historical source branches and supplied
specifications, not byte-for-byte recovery of the earlier unavailable workspace.

## Published implementation

Use `codex/rebuild-microscan-pipeline`. The complete chain includes real MAT and
manifest loading/calibration, acquisition/ROI validity, 3D complex CycleGAN,
FCERNN-derived coarse inversion and independent 3D enhancement, spectral property
queries, separate clinical segmentation, monitoring, optional subspace processing,
independent forward-operator training/qualification, A/B/C physics routing, staged
training, deployment/evaluation and atomic complete interruption recovery.

[Architecture](FINAL_MICROSCAN_ARCHITECTURE.md), [exact layer tables](ARCHITECTURE_LAYERS.md),
[export schema](DATA_EXPORT_SCHEMA.md), [source/evidence ledger](RESEARCH_EVIDENCE.md),
[Windows runbook](WINDOWS_RUNBOOK.md), and [fresh verification](verification/README.md)
provide the contracts and limitations. The supplied sphere pilot and CSV are in
`data/pilot/`. Main acquisition-mode presets are in `configs/` and can be applied
to a copied stage configuration before training a matching deployment route.

## Executed verification

Fresh full bounded suite: **86 passed**, no warnings. Production CPU structural
forwards: native128³ coarse/refined maps and five-class segmentation, 18-port GAN
at K3/K10. Editable installation, compilation, real128³ pilot export/validation,
and frequency protocol preparation passed. Full outputs are under
`docs/verification/`. A bounded update is a mechanism check, not a completed
research run. The verification README records the early structural test-access
exception transparently; no scientific sealed-test score or selection was used.

## Next command on the CUDA target

Install a current RTX50-series-capable CUDA PyTorch build using the Windows
runbook; do not install the separate CPU-only verification pins on that machine.
After its bounded GPU checks, start/resume the actual experiment:

```powershell
python -m microscan.frequency run --config configs/frequency_pilot.json --mat data/pilot/frequency_pilot.mat --csv data/pilot/frequency_subsets.csv --output outputs/frequency --device cuda --resume
```

This launches the agreed full experiment only when the operator runs it. It has
not run during implementation. Select explicitly with the corresponding
`frequency select` command, and keep `pending_selection` until useful validation
results establish a frequency count. Copy the resulting frozen configuration to
`configs/selected_frequencies.json`; main stages reject pending selection.

A is usable without Phi. B requires independently qualified frozen Phi while
training, and deploys without it. C requires the explicitly supplied compatible
operator at inference. No scientific Phi weights currently exist. Synthetic
qualification fixtures never qualify a headset model.

Still required: main whole-head property/material/clinical labels, calibrated
paired domains and source/layout/reference exports, registered baselines, actual
head bounds, independent scientific Phi qualification data, and target-GPU
validation. No COMSOL solve, dielectric-blob pretraining, full frequency campaign,
or main training was launched. The sphere ROI is not a whole-head dataset; two
coincident normalized property labels do not prove separate property recovery.

## Branch cleanup

Retain `main` and `codex/rebuild-microscan-pipeline`. Historical CycleGAN and FCERNN
heads are preserved as `archive/legacy-cyclegan` and `archive/legacy-fcernn` tags;
the redundant pre-rebuild `work` head is preserved by `archive/pre-rebuild-main`.
The old branches can therefore be removed without losing their committed source.
The review branch is not merged into main automatically.
