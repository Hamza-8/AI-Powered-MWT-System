# Controlled amplitude-routing test: K10, seed17

This is one targeted architecture experiment following frozen-checkpoint
ablations. It does not establish that the size error is caused solely by data,
normalization, conditioning, or the loss. Success requires a completed comparison
against the existing pattern-v1 final-epoch checkpoint.

## Diagnostic and decision

Run `scripts/diagnose_amplitude_conditioning.py ablate` against the existing
`outputs/frequency_pattern_v1/K10/seed_17/final.pt`. It compares the real B pair
(small validation5, large training6) and real C pair (small training7, large
validation8). Both raw measurements and log-RMS are available to the original
network. The diagnostic tests intact, raw-branch-off, RMS-branch-off, both-off,
coarse-conditioning-off, enhancement-conditioning-off, and matched-magnitude
controls. It reports coarse and refined maps separately.

The matched-magnitude intervention applies one positive scalar to the complex
measurements, preserving normalized pattern and phase. Its magnitude is taken
from the real other-size sphere at the same position. This is a sensitivity
diagnostic, not a physical substitute for that sphere's measurements. Zeroing a
branch also shifts its internal distribution: it is not a retrained ablation.
The raw branch contains pattern as well as magnitude, so raw-off does not isolate
only absolute scale. Taken together with the gain controls, these interventions
can establish whether final size is sensitive to the available magnitude paths.

The current targeted choice is **stronger amplitude access after coarse decoder
normalization**, because the frozen model's extent changes little under branch
knockouts and large gain interventions. Slight property-map changes mean the
branches are not literally unused. No extent-sensitive loss is added to this
experiment; testing such a loss would require a separately identified run.

The diagnostic validates checkpoint/data/source identities, requires completed
300-epoch/1500-update native K10 seed17 weights, checks both geometry pairs,
records physical per-case errors and verifies weights remain unchanged. Only
train/validation samples are constructed. It cannot construct or evaluate case3.
Private diagnostics, model weights and simulation data stay local.

## What changes

At each of the four coarse decoder stages:

```text
ConvTranspose3D -> existing GroupNorm -> new amplitude FiLM -> existing SiLU
```

Each new MLP receives the existing descriptors
`[log(complex_RMS), observed_fraction, amplitude_valid]`, with shared real/imaginary
RMS over observed support. It predicts a channel-wise scale and offset:
`h = h_normalized * (1 + delta_gamma) + beta`. Hidden width is32; new final
readout weights initialize with standard deviation0.02. This makes amplitude
available later than the original small conditioning readouts. It introduces
16,352 parameters, yielding **4,003,289** total versus3,986,937 in pattern-v1.

This follows the general conditioning construction in
[Perez et al., FiLM](https://arxiv.org/abs/1709.07871), equations1–2 and the
normalization/conditioning arrangement in section2.2. That paper studies visual
reasoning, not this microwave experiment. Placement and initialization strength
change together here; any improvement would not prove GroupNorm alone was the
cause. We retain the existing GroupNorm affine parameters and do not claim an
exact replication of the paper's network.

The normalized encoder, earlier raw/RMS branches, decoder convolutions, property
heads and residual refiner retain their designs. Shared initialization is
identical to baseline seed17. New-layer initialization preserves the CPU RNG
state so training sample order also remains paired. No radius rule, target-based
feature, spatial dilation or output warp is introduced.

## What stays fixed

The configuration is `configs/frequency_pilot_postnorm_v1.json`. Training starts
from scratch, not from the completed baseline weights. It uses only **K10,
seed17**, native128 volumes,300 epochs, five updates per epoch,1500 updates total.
The same training-global normalization, AdamW, cosine schedule, learning rate,
weight decay, gradient clipping, CUDA AMP and original weighted image/gradient
objective are reused. Foreground/background loss weighting stays unchanged.
Checkpoint policy is fixed final epoch; no balanced validation score is reported
or used for selection.

Split remains train1,2,4,6,7; validation5,8; reference0; sealed test3. No COMSOL
solve or frequency sweep is part of this experiment. The new runner rejects
other K/seeds and protocol changes. Original package files and three baseline
scripts remain unchanged, preserving ongoing pattern-v1 resume identities.

New run identities also bind the new source files, baseline checksum and completed
ablation report. Default output is `outputs/frequency_postnorm_v1`; writing into
the baseline candidate directory is rejected. As with the previous runner,
matching interrupted work resumes optimizer, scheduler, scaler, RNG and order.

## Windows launch

Run from the existing repo root. A ZIP-based checkout only needs four new files:

```powershell
$base = "https://raw.githubusercontent.com/Hamza-8/AI-Powered-MWT-System/codex/rebuild-microscan-pipeline"
$files = @(
    "scripts/diagnose_amplitude_conditioning.py"
    "scripts/postnorm_amplitude_model.py"
    "scripts/train_postnorm_pilot.py"
    "configs/frequency_pilot_postnorm_v1.json"
)
foreach ($file in $files) { Invoke-WebRequest "$base/$file" -OutFile $file }
```

Keep your existing pattern-v1 final.pt and the nine-case MAT in their usual local
folders. Generate the frozen-weight evidence, check native CUDA backward, then
launch the one controlled run:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_amplitude_conditioning.py ablate --device cuda
.\.venv\Scripts\python.exe scripts/train_postnorm_pilot.py profile --device cuda --backward
.\.venv\Scripts\python.exe scripts/train_postnorm_pilot.py run --device cuda --resume
```

The last command trains only K10 seed17, then automatically compares both final
checkpoints on cases5,6,7,8 and writes `outputs/frequency_postnorm_v1/comparison.json`.
Its terminal summary gives each validation case's baseline/candidate centre,
volume, epsilon RMSE and conductivity RMSE. The file also includes true volume,
relative volume error, soft volume, equivalent radius, foreground/background
physical property errors, coarse/refined results and both execution identities.

To repeat inference without retraining:

```powershell
.\.venv\Scripts\python.exe scripts/train_postnorm_pilot.py compare --device cuda
```

The standard new candidate folder contains `initial_metrics.json`, automatic
`training_history.json`, full `resume.pt`, final model-only `final.pt`, and
`result.json`. After completion, export the report and model:

```powershell
$run = "outputs/frequency_postnorm_v1/K10/seed_17"
$files = @(
    "outputs/frequency_postnorm_v1/comparison.json"
    "outputs/amplitude_diagnostic_v1/report.json"
    "$run/initial_metrics.json"
    "$run/training_history.json"
    "$run/result.json"
    "$run/final.pt"
)
Compress-Archive -Path $files -DestinationPath "postnorm_seed17_diagnostics.zip" -Force
```

Evaluate both held-out combinations, not an average volume where opposite size
errors cancel. Report unchanged/worse results as well as improvements. A more
amplitude-sensitive network is not automatically more physically accurate.

Cloud here has CPU-only PyTorch. Frozen-weight ablations and implementation
verification can run locally, but the matched CUDA training result must come from
the RTX run. Reduced-grid verification histories are explicitly marked mechanism
fixtures and cannot pass the native final-checkpoint comparison guard.
