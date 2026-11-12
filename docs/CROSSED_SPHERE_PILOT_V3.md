# Final cross-combination sphere split (v3)

Use this configuration for the next training run. The earlier v2 whole-position
holdout proposal was revised before the user launched training. The expanded MAT
is unchanged; only the explicitly versioned training/validation allocation changes.

| Position XYZ mm | Radius15 mm | Radius25 mm |
|---|---|---|
| 0,0,0 | Train1 | Sealed test3 |
| 25,-20,15 (A) | Train2 | Train4 |
| -25,15,10 (B) | Validation5 | Train6 |
| 10,25,-20 (C) | Train7 | Validation8 |

Training cases are **1,2,4,6,7**, validation cases **5,8**, and sealed test **3**.
Case0 is empty-reference calibration only. Both validation positions and both
sizes occur in training, but their exact geometry/measurement pairs do not.
This tests unseen combinations rather than requiring entirely new spatial regions.
Noise/mask variants of each geometry remain with that geometry. Original case2
was previously inspected as v1 validation and is explicitly training now.

The physical port mapping, ten frequencies, labels, architecture, optimization
budget and objective remain unchanged. The old `microscan/` modules and original
experiment data/output paths remain intact. Default new output is
`outputs/frequency_crossed_v3`, with the split and runner source hash bound into
every resume identity. Do not edit a run's configuration once it has started.

## Windows launch

From the repository root, place the local nine-case MAT at
`data/pilot_expanded/frequency_pilot.mat`. Its checksum is verified before training.
The MATLAB file is user-provided and is not uploaded to the public repository.

```powershell
.\.venv\Scripts\python.exe scripts/train_expanded_pilot.py prepare
.\.venv\Scripts\python.exe scripts/train_expanded_pilot.py run --device cuda --resume
```

The runner now defaults to `configs/frequency_pilot_crossed_v3.json` and K10 only,
seeds17/29/43,300 epochs and1500 optimizer updates per seed. It uses the original
AdamW/cosine/AMP/clipping setup and starts each candidate from scratch. The same
command resumes matching interrupted work. Terminal evaluation lines report
training and validation property errors and validation centre error.

`initial_metrics.json` records untrained errors; `resume.pt` stores complete
states and per-epoch train/validation/per-case metrics; `best.pt` and `result.json`
retain the original selected-checkpoint policy, with last-epoch metrics clearly
separate in `result.json`. Do not interpret selected and last weights as the same
snapshot. First inspect whether K10 generalizes to both held-out combinations
before deciding to run the remaining frequency counts. The sealed test remains
unopened until explicit frequency selection.

Only bounded reduced-grid optimizer/resume verification runs in Cloud, which has
CPU-only PyTorch. Full native128³ training must be launched on the user's RTX
machine. Existing verification of the prior full-size architecture still applies
to its unchanged network; no new scientific training result is claimed here.
