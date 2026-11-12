# Fully crossed A/B training, C validation

This is a split-only K10/seed17 experiment with the original pattern-v1 model
and foreground/background weighted property plus gradient objective. There is
no Dice addition, postnorm change, balanced validation score, early stopping,
or selection of the best validation checkpoint.

| Allocation | Cases |
| --- | --- |
| Training | 2,4,5,6: both radii at A and B |
| Validation | 7,8: both radii at C |
| Unused | 1: centre-small |
| Sealed test | 3: centre-large |
| Empty reference | 0 |

Every retained training location contains both radii. C is absent from training;
this tests transfer to an unseen location, rather than the old split's new
size/location combinations at familiar locations. These cases have already been
inspected, so this is development validation, not a new independent test.
Four training geometries remain a very small dataset; improvement is not assured.

The script derives its configuration from the unchanged
`configs/frequency_pilot_pattern_v1.json`. It does not need a new config file or
any existing trained checkpoint. It starts from seed17 initialization. The
input RMS is refitted using only cases 2,4,5,6. Previously fitted normalization
would otherwise include cases outside the new training allocation.

The run uses **1,500 optimizer updates**, or **375 complete four-case passes**.
Each case receives exactly 375 updates. The original AdamW settings are retained
(learning rate 0.0002, weight decay 0.00001, clip 1). CosineAnnealingLR has
T_max=300 and advances after every five updates, exactly matching the old
five-case/300-epoch learning-rate sequence. Evaluation also runs every five
updates; an evaluation block is not a four-case data pass. Final weights are
saved after update1500. AMP is enabled on CUDA as before.

From PowerShell **in the repository root**:

```powershell
.\.venv\Scripts\python.exe scripts/train_crossed_ab_pilot.py prepare --device cpu
.\.venv\Scripts\python.exe scripts/train_crossed_ab_pilot.py run --device cuda
```

To continue an interrupted run, use the same command with `--resume`. Do not use
an old run folder or checkpoint: the allocation, input scale and identity differ.
Default output is `outputs/frequency_crossed_ab_v1/K10/seed_17/`.

The script exports `initial_metrics.json`, `training_history.json`, `resume.pt`,
`final.pt`, and `result.json`. Histories include per-case train/validation centre,
volume, epsilon and sigma errors plus learning rate and optimizer update count.
Each console evaluation prints both validation cases separately. Existing
experiments are preserved. The runner has no test-evaluation command and rejects
requests to construct either excluded centre case. Full-container MAT integrity
checks still read/hash the original container.

Match comparisons to the new split. Case7 was previously a training case and
case5 previously validation, so the old aggregate validation score is not a
matched comparison. K10 is a controlled diagnostic, not a selected frequency
count. No frequency sweep or new COMSOL data is launched.
