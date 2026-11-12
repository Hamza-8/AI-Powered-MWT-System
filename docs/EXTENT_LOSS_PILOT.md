# Loss-only radius experiment

This tests whether explicit extent supervision improves the held-out sphere-size
combinations. It uses the original **pattern-v1 architecture**, not the postnorm
variant. The result is unknown until the controlled training run finishes.

## One objective change

```text
training_loss = original_weighted_property_and_gradient_loss
              + 0.1 * refined_soft_Dice_loss
```

For each refined physical-property channel independently:

```text
p_epsilon = (predicted_epsilon - 1) / 44
p_sigma   = predicted_conductivity / 0.8
t         = existing binary sphere foreground mask

loss_channel = 1 - (2 * sum(p*t) + 1e-6) / (sum(p*p) + sum(t*t) + 1e-6)
refined_soft_Dice_loss = mean(loss_epsilon, loss_sigma)
```

The squared-denominator form follows
[Milletari et al., V-Net](https://arxiv.org/abs/1606.04797), section2, which
introduces a differentiable overlap objective for volumetric segmentation.
Applying it to normalized physical contrast is an adaptation for this uniform,
fixed-material sphere pilot, not a result established by that paper. It penalizes
contrast errors as well as spatial overlap: it is not a pure radius estimator or
a loss specification for heterogeneous head tissues.

No hard threshold, sigmoid, clipping, segmentation head, or new model parameter
is introduced. Values above unit contrast retain a gradient. Separate channel
losses prevent epsilon and conductivity from cancelling through their average.
Spatial reductions use float32 outside autocast so native volumes do not overflow
float16 sums. Empty matching volumes produce zero Dice loss. The coefficient0.1
is a predeclared engineering choice for one experiment, not a tuned optimum.

All existing foreground/background property-loss weighting, coarse/refined
property supervision and gradient-loss weight remain unchanged. Dice is added
only at the refined output; the coarse map keeps its original objective. Masks
are used for loss/evaluation only and never become reconstruction inputs.

## Fixed controls

- K10, seed17 only;300 epochs, five updates per epoch,1500 total.
- Train1,2,4,6,7; validation5,8; reference0; case3 remains sealed.
- Original pattern/amplitude inputs, model,3,986,937 parameters and paired seed17
  initialization/sample-order policy.
- Same AdamW, learning rate, weight decay, cosine schedule, AMP and clipping.
- Fixed final-epoch checkpoint policy. No balanced validation score or early stop.
- A fresh initialization and output folder: `outputs/frequency_extent_v1`.

Existing model/package/runner files are untouched, preserving old frequency-run
resume identities. New identities bind this script, the loss specification and
the original baseline checkpoint checksum. The runner rejects the postnorm
architecture, another frequency count/seed, changed optimizer/budget controls,
or writing into the baseline candidate directory.

This K10 test isolates a **radius-training hypothesis**. It does not select10 as
the head-data frequency count. The minimum-frequency comparison remains separate.
No COMSOL solves or full frequency sweep are launched by this runner.

## Windows launch

Run from the existing repo root. You need only two new files:

```powershell
$base = "https://raw.githubusercontent.com/Hamza-8/AI-Powered-MWT-System/codex/rebuild-microscan-pipeline"
$files = @("scripts/train_extent_pilot.py", "configs/frequency_pilot_extent_v1.json")
foreach ($file in $files) { Invoke-WebRequest "$base/$file" -OutFile $file }

.\.venv\Scripts\python.exe scripts/train_extent_pilot.py profile --device cuda --backward
.\.venv\Scripts\python.exe scripts/train_extent_pilot.py run --device cuda --resume
```

Defaults expect the existing baseline at
`outputs/frequency_pattern_v1/K10/seed_17/final.pt` and the expanded nine-case MAT
at `data/pilot_expanded/frequency_pilot.mat`. Use `--baseline`/`--mat` for other
local paths. Baseline weights are inspected and compared; they do not initialize
the new training run.

Each epoch prints the physical errors plus validation case5/case8 predicted and
true volumes and centre errors. `initial_metrics.json`, `training_history.json`,
`resume.pt`, `final.pt` and `result.json` follow the existing format. Histories
separately record the original evaluation objective, Dice per property, and the
augmented training objective.

After completion, the runner automatically evaluates both frozen final models on
training/validation cases and writes `outputs/frequency_extent_v1/comparison.json`.
Its terminal summary reports each validation case's centre, volume and physical
epsilon/conductivity errors before and after. The comparison explicitly records
the objective change and both execution identities. Loss diagnostics evaluate the
same candidate formula on both models; the baseline was trained without Dice.

Repeat the comparison without training:

```powershell
.\.venv\Scripts\python.exe scripts/train_extent_pilot.py compare --device cuda
```

Send the completed reports and model for analysis:

```powershell
$run = "outputs/frequency_extent_v1/K10/seed_17"
$files = @("outputs/frequency_extent_v1/comparison.json", "$run/initial_metrics.json", "$run/training_history.json", "$run/result.json", "$run/final.pt")
Compress-Archive -Path $files -DestinationPath "extent_seed17_diagnostics.zip" -Force
```

Assess both validation cases individually. Better Dice loss or centre error
alone does not establish correct size or properties. The existing property and
volume metrics remain the basis for interpreting the result. Sparse geometry
coverage can still limit generalization even if the new loss is implemented
correctly. CPU implementation checks and reduced-grid fixtures do not establish
a trained improvement; full training requires the user's CUDA environment here.
