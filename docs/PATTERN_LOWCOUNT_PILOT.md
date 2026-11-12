# K1/K2 extension of the existing pattern-v1 benchmark

The completed K3..10/seed17 runs already provide the matched reference sweep.
This runner adds only two explicit candidates, without changing package/model
sources or the supplied frequency CSV:

| Count | Frequencies, GHz | Zero-based slots |
| --- | --- | --- |
| K1 | 1.05 | 9 |
| K2 | .60,1.05 | 0,9 |

K1 was identified by a training-only scalar-size screen on A/B. K2 is the
previously declared band-endpoint pair. Neither has a proven neural-model result.
The original ten input slots, mask handling, network parameter count, weighted
objective, AdamW, AMP behavior, and training budget are unchanged. Missing
frequencies remain masked, and normalization is refitted using the selected
training observations only. The new candidate definitions and runner hash are
bound into checkpoint identities.

This uses the **original pattern-v1 split**, train1,2,4,6,7 / validation5,8,
reference0 / sealed3. It is deliberately matched to the completed K3..10 sweep;
it is not the fully crossed A/B -> C experiment. It tests held-out size/location
combinations at known positions. Keep the known radius-transfer limitation
visible in per-case results.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/train_pattern_lowcount.py run --counts 1 --device cuda --resume
```

Default is seed17, 300 epochs x5 cases =1500 updates, from scratch. Output is
`outputs/frequency_pattern_lowcount_v1/K01/seed_17/`. For K2, change `--counts 1`
to `--counts 2`. Additional configured seeds can be explicitly selected with
`--seeds`. Use another paired K10 reference if judging seed variability.

Each run exports initial metrics, full training history, resumable state,
fixed-final-epoch weights, and result JSON. Console output includes each
validation case's centre, volume, epsilon and sigma errors. Completed existing
experiments are not rerun or overwritten. The context adapter restores the
original subset reader, including on exceptions, and changes no source files.

Compare the same final-update, per-case metrics against existing K4 and K10.
There is no balanced score, early stopping, or automatic best-state selection.
The user requested comparison of actual property errors without a fixed10%
improvement margin. Do not conceal a per-case increase with an average decrease.
No candidate is automatically declared a minimum by this runner.

Any resulting frequency choice remains a provisional sphere-pilot choice before
head validation. Solver-tolerance tests (baseline1e-5, proposed1e-4/1e-3) are
separate paired experiments on existing geometry, with a matching empty
reference. This script launches no COMSOL solves and never evaluates case3.
