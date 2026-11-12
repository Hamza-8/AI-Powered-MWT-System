# Pattern/amplitude reconstruction experiment

This experiment addresses a possible amplitude-driven position shortcut in the
crossed-sphere pilot. It changes measurement representation and conditioning,
while preserving the reconstruction objective, physical labels, split and update
budget. Its benefit must be measured by a new controlled training run. Existing
weights cannot establish that benefit by changing their inputs at inference.

## Research checked before implementation

| Primary source | Relevant finding | What it does not establish |
|---|---|---|
| [Ito, Jin and Zou, direct electromagnetic sampling](https://arxiv.org/abs/1212.5085), section 3.1, equation 16 | A normalized complex field/probe inner product provides a support indicator. This motivates giving a learned encoder access to signal direction independently of overall magnitude. | It is a model-based sampling method with Green-function probes, not this neural architecture or an identifiability result for headset S-parameters. |
| [Perez et al., FiLM](https://arxiv.org/abs/1709.07871), section 2.1, equations 1–2; appendix 7.2 | Conditioning can apply learned channel-wise scale and offset to intermediate features; an identity-centred scale retains the main feature path. | Visual-reasoning experiments do not prove an electromagnetic reconstruction improvement. |
| [Kim et al., RevIN: official implementation and paper links](https://github.com/ts-kim/RevIN) | Instance normalization need not discard statistics: its time-series method restores instance statistics at the output. | We do not use RevIN output denormalization: scattering amplitude is not a direct multiplier of permittivity or conductivity. |
| [Trabelsi et al., Deep Complex Networks](https://arxiv.org/abs/1705.09792), section 3.5 | Complex batch whitening uses a real/imaginary covariance matrix. Complex normalization choices have distinct meanings. | Whitening is not the operation implemented here. A shared positive real scalar is chosen specifically to preserve measured complex phase. |

The engineering inference is to expose a normalized pattern and retain amplitude
as separate conditioning, rather than discard amplitude or independently
standardize real and imaginary channels. No cited paper proves that this removes
the observed shortcut. Nonlinear scattering couples size, position and material;
normalization does not make their physical effects exactly separable. We neither
assume a Born approximation nor convert signal amplitude directly into radius.

## Implemented representation

Let `y` be complex reference-subtracted measurements divided by the existing
training-only global real/imaginary RMS `s`, and `M` the observed-support mask.
For each sample over all selected frequencies, receivers and transmitters:

```text
N = sum(M)                         # complex entries; observed zeros count
a = sqrt(sum(M * (Re(y)^2 + Im(y)^2)) / max(N, 1))
d = max(a, 1e-8)
u = y / d                          # one scalar shared by real and imaginary
c = [log(d), N / total_slots, a > 1e-8]
```

Hidden values are removed with `where` before arithmetic, including hidden NaNs.
Statistics use float32 outside autocast and max-scaled accumulation. Empty or
zero support produces zero pattern, a finite log-floor and an explicit invalid
amplitude flag. Below the floor, exact gain invariance is intentionally relaxed.
The unnormalized branch still retains those small observations. Above the floor,
`u` is invariant to a positive global gain on the same support, while `a` changes.
Complex phase, signed real/imaginary ratios, relative frequency response and port
order are preserved. No mean subtraction, phase alignment, per-frequency scaling,
label-derived features, or target-dependent inference operation is introduced.
`s * a` recovers the physical complex RMS; raw input and `s` remain available.

The existing eight-stage Conv1D encoder now receives six features:
`Re(u), Im(u), M, GHz, Rx, Tx`. At its pooled output, a compact parallel raw-signal
encoder and an amplitude MLP generate FiLM coefficients:
`h_conditioned = h_pattern * (1 + delta_gamma) + beta`. The enhancement U-Net's
measurement conditioner uses the same construction. New FiLM readout weights
start small; shared encoder/decoder weights retain their baseline seeded
initialization. The volume decoder, enhancement U-Net and spectral head are the
same designs as the baseline. Raw conditioning can still learn a shortcut; this
architecture makes pattern accessible, not compulsory.

The full-size model has **3,986,937 parameters**, versus 3,903,065 in the baseline
(an increase of 83,872, about 2.15%). This is an architecture/representation
comparison, not a parameter-matched proof of disentanglement. The fixed ten-slot
layout keeps its parameter count equal across candidate frequency counts.

## Controlled training and artifacts

Use `configs/frequency_pilot_pattern_v1.json` with
`scripts/train_pattern_pilot.py`. The crossed-v3 split stays:
training **1,2,4,6,7**, validation **5,8**, sealed test **3**, reference **0**.
The MAT checksum and original-prefix integrity checks remain enforced. All cases
are spheres. Case3 is never used to form a sample for training, preparation or profiling.

The runner directly calls the original `microscan.frequency._objective`:
foreground/background weighting is unchanged, coarse weight 0.5, refined weight
1, gradient weight 0.01. AdamW remains learning rate 0.0002, weight decay 0.00001,
cosine scheduling over 300 epochs, batch size one, gradient clip 1 and CUDA AMP.
Five training samples give **1,500 optimizer updates per seed**. Configured seeds
remain **17,29,43**; `--seeds 17` runs only that member without changing run identity.

**Final epoch is the checkpoint policy.** There is no validation-based best
checkpoint, composite ranking, early stopping or validation-driven LR change.
The balanced validation score is omitted from reports and histories. Report
epsilon RMSE, conductivity RMSE, centre error, volume error and each validation
case separately. The original weighted evaluation objective is also recorded
as a diagnostic, not a checkpoint selector. Compare to the baseline's final
`resume.pt` weights, rather than its legacy `best.pt`.

Each `outputs/frequency_pattern_v1/K10/seed_17` directory contains:

- `initial_metrics.json`: untrained train/validation metrics before any update.
- `resume.pt`: complete latest model, optimizer, scheduler, scaler, RNG and order.
- `training_history.json`: automatically exported per-epoch train/validation and
  per-case metrics, updated after every completed epoch.
- `final.pt`: model-only final-epoch weights and inference metadata, on completion.
- `result.json`: final-epoch metrics, checkpoint hash and experiment identity.

Old experiment files, package sources and weights remain intact. New identities
bind the new scripts, old data adapter, package sources, data, configuration and
execution settings. Changing any of those requires a fresh output directory.
An interrupted matching run resumes exactly, including order and RNG. Checkpoint
replacement retries transient Windows locks without deleting the previous file.
Do not run two writers against the same candidate directory.

## Windows commands

Run from the repository root containing `.venv`, `scripts` and `configs`. Keep the
nine-case MAT at `data/pilot_expanded/frequency_pilot.mat`.

If you use a downloaded ZIP without Git, download the three new files into that
existing repository folder first:

```powershell
$base = "https://raw.githubusercontent.com/Hamza-8/AI-Powered-MWT-System/codex/rebuild-microscan-pipeline"
$files = @("scripts/pattern_amplitude_model.py", "scripts/train_pattern_pilot.py", "configs/frequency_pilot_pattern_v1.json")
foreach ($file in $files) { Invoke-WebRequest "$base/$file" -OutFile $file }
```

For a Git clone, update with `git pull --ff-only` on the rebuild branch. Then:

```powershell
.\.venv\Scripts\python.exe scripts/train_pattern_pilot.py profile --device cuda --backward
.\.venv\Scripts\python.exe scripts/train_pattern_pilot.py prepare
.\.venv\Scripts\python.exe scripts/train_pattern_pilot.py run --device cuda --seeds 17 --resume
```

Start seed17 at K10. This is a new training run for the adjusted architecture;
`--resume` continues its own new output if interrupted. It does not warm-start
from the old amplitude-sensitive weights. Keep K10 and inspect both validation
cases before spending time on the other seeds or smaller frequency counts.
Once seed17 finishes, export the small reports for interpretation:

```powershell
$run = "outputs/frequency_pattern_v1/K10/seed_17"
Compress-Archive -Path "$run/initial_metrics.json", "$run/training_history.json", "$run/result.json" -DestinationPath "pattern_seed17_reports.zip" -Force
```

Success requires improved validation localization alongside property and volume
behavior while retaining training fit. An epsilon improvement alone, a training
fit alone, or a gain-controlled diagnostic with the wrong size is insufficient.
These validation combinations have informed model development; they are no longer
an untouched estimate of generalization. Case3 stays sealed. Repeating matched
seeds and later evaluating independently generated geometries is necessary before
choosing a production frequency count or making broader claims.

## Verification scope

Automated checks cover common-scalar phase preservation, positive-gain pattern
invariance, retained amplitude, measured zeros, hidden NaN/Inf isolation, empty and
extreme signals, AMP-safe statistics, branch gradients and weights round trips.
Local expanded-MAT integration checks verify exact interrupted/uninterrupted
training equality, final/resume weight equality, per-case histories, identity
rejection, source preservation and exclusion of case3. Those integration checks
skip if the private MAT is absent. Synthetic native128 backward verifies shape
and finite gradients. These checks establish implementation mechanics, not a
trained reconstruction improvement. CUDA performance must be measured on the
user's GPU; Cloud verification here uses CPU PyTorch.
