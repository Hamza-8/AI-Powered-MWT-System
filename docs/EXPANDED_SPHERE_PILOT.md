# Expanded sphere pilot v2

The new supplied MAT contains complex128 `[18,18,10,9]`: eight sphere geometries
and the empty reference. Its original five S-parameter blocks are bitwise equal
to the original file. All 90 frequency/case jobs report complete and finite.
The ten frequencies, physical port order, COMSOL mapping and cell-centred128³
object grid remain unchanged. The MAT's provisional split labels are preserved
as generation provenance; the following explicit pre-training allocation replaces
them for this revised experiment.

| Case | Radius mm | Centre XYZ mm | v2 role |
|---|---:|---|---|
| 0 | empty | reference | calibration only |
| 1 | 15 | 0,0,0 | training |
| 2 | 15 | 25,-20,15 | training; previously inspected v1 validation |
| 3 | 25 | 0,0,0 | original sealed test |
| 4 | 25 | 25,-20,15 | training |
| 5 | 15 | -25,15,10 | training |
| 6 | 25 | -25,15,10 | training |
| 7 | 15 | 10,25,-20 | validation |
| 8 | 25 | 10,25,-20 | validation |

There are five training geometries, both sizes at two translated positions,
and a two-size validation pair at a position absent from training. Case3 is never
materialized by preparation/training. New validation covers one spatial position,
not two independent subjects or head anatomies. Original case2 was repeatedly
examined during v1 diagnosis, so it is explicitly moved into training and is no
longer presented as unseen evaluation. v1 and v2 validation averages measure
different cases and must not be interpreted as a controlled before/after accuracy
comparison.

## First run: ten-frequency comparator

Keep the original package, MAT and `outputs/frequency` intact. This runner lives
outside `microscan/`, so it does not invalidate original package-source identities.
Its own SHA256, expanded data hash and exact geometry/split records enter every
new run identity. New output is `outputs/frequency_expanded_v2`.

The expanded MAT is user-provided local data and is not published in Git. Copy
your new nine-case MAT to `data/pilot_expanded/frequency_pilot.mat`; keep the
original five-case file at `data/pilot/frequency_pilot.mat` unchanged. The recorded
SHA256 validates the correct file before any training.

From the repository root on the Windows RTX machine:

```powershell
.\.venv\Scripts\python.exe scripts/train_expanded_pilot.py prepare
.\.venv\Scripts\python.exe scripts/train_expanded_pilot.py run --device cuda --resume
```

Default counts are **10 only**, with paired seeds17/29/43. Each is initialized
from scratch, runs300 epochs with five training updates per epoch (1500 total),
and retains AdamW, LR0.0002, weight decay0.00001, cosine decay, AMP and clipping1.
There is no warm start from the two-example experiment. The process retries
transient Windows checkpoint replacement locks for up to30 seconds.

Recorded evidence per candidate:

- `initial_metrics.json`: actual seeded untrained training/validation metrics,
  before any optimizer step, bound to the complete experiment identity.
- `resume.pt`: last saved weights, all optimizer/scheduler/scaler/RNG states and
  progress. Each history entry has validation coarse/refined and per-case metrics,
  plus `validation.training_fit` with training metrics evaluated in float32.
- `best.pt`: checkpoint selected by the unchanged original balanced-contrast rule.
- `result.json`: selected-checkpoint metrics, with `last_epoch_metrics` explicitly
  separate. Physical epsilon/sigma, localization, volume, foreground/background,
  and evaluation objective are retained. The float32 evaluation objective is not
  mislabeled as the exact AMP minibatch training loss.

The terminal prints training and validation physical errors and localization at
each epoch evaluation; the final selected-checkpoint evaluation prints once more.
The counter labels evaluations rather than implying an extra training epoch.

Before comparing smaller K, inspect whether ten-frequency training fits all five
examples and generalizes to both C spheres at their true position. If that fails,
report it honestly and diagnose the maps instead of choosing a frequency count.
When justified, the same runner supports `--counts 3,4,5,6,7,8,9,10`; matching
completed K10 work is reused with `--resume`. `select` requires all eight counts
and paired seeds; `test` requires explicit successful selection and locks it.
Frequency configuration remains pending until then.

## Verification and execution boundary

Fresh local checks use the real expanded MAT and reduced native16³ models for
bounded optimizer/resume mechanics. They verify training-only scales, original
array integrity, translated C targets, test exclusion, initialization history,
bitwise interruption equivalence and restoration of legacy process globals.
They are not a scientific training result. This Cloud environment has CPU-only
PyTorch; the full production128³ CUDA run must be launched on the user's RTX
machine. No new EM solves are needed for this revised experiment.
