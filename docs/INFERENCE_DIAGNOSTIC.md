# Inference-only frequency-pilot diagnostic

Use existing weights to distinguish fitting the two training examples from
responding to measurements. No optimizer is created, checkpoint rewritten,
preprocessing refitted, epoch resumed, or frequency selected. Case3 stays sealed.

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_frequency.py --device cuda
```

The default checkpoint is `outputs/frequency/K10/seed_17/resume.pt`. Its `model`
holds the last saved weights (epoch300 after a completed experiment), whereas
`best.pt` holds the earlier balanced-score-selected weights. This distinction is
recorded explicitly. The script checks the saved source identity, pilot hash and
weight architecture, restores the saved input scale/subset/masks, and performs
FP32 eval/inference-mode calls matching the original validation precision.

Four distinct forwards are required:

1. Training case1, small centred sphere.
2. Training case4, larger translated sphere.
3. Validation case2, small translated sphere.
4. Zero measurement values with the same observed acquisition mask.

Zeroing values does not mark them missing. It is an out-of-distribution sensitivity
probe, not a new simulated empty headset. In this protocol all cases have identical
masks/frequencies, so swapped-input predictions are exactly the already computed
donor predictions: targets are only used in subsequent metrics. These comparisons
need no extra forward calls and never update the model.

The small report contains coarse/refined physical errors, object centroids in
XYZ millimetres, thresholded volumes, normal/zero/swapped comparisons and output
differences between cases. Upload
`outputs/inference_diagnostics/K10_seed17_final/diagnostic_report.json` for analysis.
It contains no weights. Add `--save-volumes` to retain full NPZ maps locally.

For another existing checkpoint, specify both its path and a distinct output:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_frequency.py --checkpoint outputs/frequency/K10/seed_17/best.pt --output outputs/inference_diagnostics/K10_seed17_selected --device cuda
```

The mechanism tests in `tests/test_inference_diagnostic.py` use untrained tiny
fixtures for code verification only. They establish no reconstruction performance.
The actual trained weights remain on the user's Windows machine; the script must
run there to obtain the scientific diagnostic results. Existing training modules
under `microscan/` remain unchanged, preserving experiment resume identities.
