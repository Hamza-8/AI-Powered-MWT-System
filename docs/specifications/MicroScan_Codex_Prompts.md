# MicroScan: research-backed complete AI implementation prompt

Revision: 28 October 2026, integrating the architecture audit, earlier research
notes, supplied papers/figures and a primary-source search for 28 September to
28 October 2026.

**Use Prompt 2 below. It replaces every earlier Prompt 2.** Prompt 1 is retained
for reference; the audit is already complete. This revision includes 3D CycleGAN,
mode-aware MSRNN, residual-conditioned 3D enhancement, separate segmentation,
focused monitoring, independently validated forward consistency and staged training.
Frequency-count performance testing remains reconstruction-only.

Supply Codex the audit, earlier research notes, FCERNN paper, thesis, latent-space
and forward-solver PDFs, inverse-solver screenshots/abstract, and completed pilot
files. The evidence/decision ledger after Prompt 2 distinguishes verified findings
from proposed engineering adaptations and unavailable details.

## Prompt 1 — report the exact current architecture

```text
Work in the currently opened MicroScan AI repository. First read AGENTS.md and
inspect the actual code, configurations, entry points, tests and checkpoints.
This task is an architecture audit, not a model modification or training task.

We need a precise description of what is implemented TODAY before changing it.
Do not describe the intended paper architecture as though it were our code.
Do not rename the existing model MSRNN: our known starting point is an older
FCERNN implementation. Discover and name what is actually present.

Produce docs/CURRENT_ARCHITECTURE.md and a machine-readable companion,
docs/current_architecture.json. Change only those audit deliverables. Include:

1. Repository revision/dirty state; relevant files, classes, configurations,
   training/evaluation entry points, dependency versions, and available trained
   weights. Identify the configuration used for each reported result. Distinguish
   defaults, dormant alternatives and the path the actual training command uses.

2. Inventory every relevant model: FCERNN, CycleGAN's G/F/D_X/D_Y, enhancement
   U-Net and segmentation U-Net. If one is absent, say ABSENT. Do not invent or
   implement it. Distinguish a U-Net inside a CycleGAN generator from a separate
   reconstruction enhancement U-Net and a lesion segmentation network.

3. For each implemented model, report an exact layer-by-layer table with source
   file/line references: input/output shapes including axis meanings; operation;
   channels; kernel/stride/padding/dilation/output padding; normalization;
   activation/order; bias; trainable parameter counts. Include flatten/reshape,
   interpolation, pooling, skip concatenation/addition, attention and residual
   paths. Show the complete route from stored measurements to physical outputs.
   State which dimensions are fixed and which are configurable.

4. Trace the data/training contract: real data versus random dummy data; complex
   representation; frequency and physical-port ordering; calibration/background
   subtraction; normalization and inverse transform; target construction; splits;
   augmentation; actual losses and weights; optimizers/schedulers; GAN update
   rules if present; AMP; train/eval modes; checkpoint/resume contents and metric
   definitions. Identify leakage and numerical/shape failure points concretely.

5. Audit the real pilot MAT without altering it. It contains MATLAB structs
   data/report/cfg, with complex data.S axes [physical Rx, physical Tx,
   frequency, case] and expected shape [18,18,10,5]. List completion/finite-value
   checks, frequencies, physical-to-COMSOL port mapping and geometry metadata.
   Derive the adapter needed for this format rather than guessing from shapes.

6. Run inexpensive deterministic shape probes using the existing implementation
   and actual configs: current default input, and FCERNN inputs representing
   18x18 channels at K=3 through 10 frequencies. Report success or the exact
   exception and the preceding tensor shape. Try a single forward pass on a
   real pilot sample when possible. Use eval/no_grad for shape inspection.
   Do not repair failures during the audit. A missing dependency/GPU must be
   reported accurately; static inspection is not a runtime pass. Do not train.

7. Compare code with the two supplied references, separating VERIFIED SOURCE
   FACT, CURRENT CODE, and PROPOSED ADAPTATION. Read the relevant pages/figures:
   FCERNN paper PDF pp. 5-7, Figure 3 and section 2.2.2; section 3.2 and Eq. 8;
   thesis chapter 4, printed pp. 52-58, especially Figures 4.3/4.4 on printed
   p.55 (PDF p.73). Record ambiguities instead of silently filling them in.

8. End with a concrete discrepancy list and the smallest justified changes
   needed to support our tensors. Include exact candidate tensor shapes and
   arithmetic at every encoder-to-3D transition. Flag wrong/duplicated block
   extraction, incorrect axis permutations, random targets, incompatible latent
   sizes, and unsafe or incomplete checkpoint reuse if found.

The older archive had dummy random input/targets, nominal 16-frequency/24-port
defaults, a fixed 256x4x4x4 latent and a 64-cubed output. These are leads to
verify, not facts to overwrite contrary evidence in the current repository.

Do not start model development, new COMSOL solves, GAN training or frequency
experiments. Finish by linking the two audit files and stating which runtime
probes actually ran, the major incompatibilities, and any genuinely missing
source material.
```

## Prompt 2 — implement the complete physics-informed MicroScan pipeline

```text
Implement the complete MicroScan AI pipeline in this repository, using the
supplied architecture audit, earlier research notes, reference papers, inverse-
solver abstract/figures and simulation files. Integrate their useful mechanisms
into MicroScan; the research synthesis in this document explains source limits.
This is an implementation task. Finish the models, integration, data contracts,
focused-monitoring modes, training/evaluation code and meaningful verification.
Do not stop at a plan, a tensor adapter or a reconstruction-only scaffold.

The intended sequence is:
complete the pipeline architecture and training infrastructure -> test the
reconstruction frequency count -> freeze the chosen frequency configuration ->
generate the main dataset and train the pipeline.
Frequency selection is the remaining experiment-dependent architecture setting.
Make concrete, documented defaults for other engineering decisions instead of
leaving unfinished modules or broad design questions. Accuracy and optimality
must subsequently be established by training and evaluation.

1. INTEGRATE THE REAL STARTING CODE

Read AGENTS.md and inspect the actual working tree and supplied audit.
The audit identified two separate branches in Hamza-8/AI-Powered-MWT-System:
- CycleGAN: codex/build-pytorch-cyclegan-for-clutter-removal,
  commit 6c14ab3c376f9fee0d14bc9283bdd8e7f768f44f.
- FCERNN: codex/implement-3-d-fcernn-in-pytorch-2.x,
  commit ce342caa663c24e6c344a0de056ad7c324386ef2.
They contain conflicting models/data/utils paths. Integrate their useful code
into one package without replacing one branch's functionality with the other.
Use local branch/commit objects when available; inspect current changes first.
Preserve unrelated work. The pasted audit ends at section 3.2, so inspect actual
source for details not present in the paste rather than inventing the remainder.

Confirmed problems to resolve: production training uses dummy data; FCERNN's
encoder default ends at [B,256,144] but its reshape requires [B,256,64]; its
output is nominally 64 cubed; enhancement and segmentation U-Nets are absent;
there is no executable calibrated end-to-end chain. The 3D GAN already exists
and must be developed, not replaced with a 2D per-frequency network. Its current
discriminator's even frequency kernels shrink the frequency axis at each layer.
Remove assumptions inherited from 50 frequencies and 24 antennas.

2. IMPLEMENT THE COMPLETE CHAIN

calibration + acquisition masks + optional baseline context
-> 3D CycleGAN signal decluttering
-> mode-aware MSRNN coarse reconstruction + residual-conditioned 3D enhancement
-> separate 3D segmentation U-Net
-> physical property maps, lesion probabilities/masks and monitoring outputs.

Implement named, independently trainable modules and an integrated pipeline
wrapper. MSRNN is the project reconstruction stack: an improved FCERNN-style
measurement-to-volume core followed by the missing enhancement network. State
the exact definition in documentation. The audited FCERNN has no recurrence;
do not claim an RNN exists or add a costly recurrent unit just for the name.

Return intermediate calibrated/decluttered signals, coarse/refined property
volumes, segmentation logits/probabilities, baseline-relative changes and
measurement/ROI validity metadata. Expose real stage controls and bypasses.
CycleGAN F and its discriminators are training components. The electromagnetic
forward operator Phi is separately named; include it in deployment dependencies
when measurement-feedback correction is enabled, and include its runtime cost. The full pipeline must be implemented even though
some stages will be bypassed for the small frequency experiment.
Default mode routing: baseline monitoring uses calibrated differences and
baseline context with CycleGAN/subspace bypassed initially; snapshot mode can
use the trained 3D declutterer and optional subspace context. Missing trained
weights must never enable a random processor. Make fallback paths explicit and
consistent with how the reconstruction checkpoints were trained.

3. FREEZE EXPLICIT DATA AND PHYSICAL CONTRACTS

The completed frequency_pilot.mat contains data/report/cfg structs.
data.S is complex128 [physical Rx,physical Tx,frequency,case]=[18,18,10,5].
GHz=[0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05].
Physical labels already run 1..18 on both axes. COMSOL IDs in that order are
[7,8,9,10,11,1,2,3,4,6,12,13,14,15,16,17,18,19]. Do not reorder them again.
Use canonical network signal axes [B,C,F,Rx,Tx]; real/imag are the first two
channels. Keep all entries, including reflection channels, and preserve phase.
Perform complex subtraction in complex128 before converting channels to float32.

Support distinct, recorded calibration modes:
- Snapshot reconstruction: current S and an appropriate empty-headset/system
  reference. Healthy head structure is useful signal for full reconstruction;
  do not indiscriminately remove skull/brain anatomy as clutter.
- Monitoring: current and registered same-subject baseline measurements,
  their difference, and baseline reconstruction when available. Baselines can
  already contain lesions. Changes can be positive or negative.
Keep current/reference masks and calibration conventions explicit. A difference
is observed only where both operands are available. An absolute current volume
cannot be recovered from a difference alone without a baseline/absolute input;
implement the context needed for every output you claim. Distinguish true tissue
change from pose/calibration nuisance. Carry grid/pose registration and its
validity; registering reconstructed volumes alone does not correct misregistered
measurement subtraction. Gate/recalibrate a mismatched baseline explicitly.

Targets use [B,property,Z,Y,X] with an explicit affine/axis/units convention.
Primary property outputs are epsilon_r and effective conductivity in S/m at a
fixed reference frequency, default 0.85 GHz, independent of the selected input
frequencies. Respect a verified dataset reference definition if different.
For dispersive head data, construct labels from the actual material model at
that reference frequency and at observed frequencies for the auxiliary spectral
readout described in section 6A. Do not treat dispersion as constant or infer
unlabelled Cole-Cole coefficients from a few frequencies.

The pilot has 128-cubed voxels over [-60,60] mm, with centres
-60+0.9375/2+arange(128)*0.9375. Sphere epsilon_r=45 and sigma=0.8; outside
epsilon_r=1 and sigma=0. Use voxel_targets.mat or reconstruct these labels from
data.cases/cfg. Cast MATLAB unsigned coordinates before arithmetic. Test the
off-centre sphere to catch invisible axis swaps in cubic tensors.
The pilot's 120 mm cube is an object ROI, not a whole-head imaging extent.
Main-dataset targets must cover the verified head/brain bounds. Keep physical
extent and grid shape in the manifest; use 128-cubed as the initial full-volume
network grid and derive appropriate full-head extents from the actual anatomy.
Do not quietly crop the brain to reuse the sphere ROI.

Fit data-derived input preprocessing on training geometries/observations only.
Make transformations invertible to physical units and checkpoint them. Avoid
per-sample min/max normalization that removes lesion amplitude information.
Do not clip away signed complex signals or signed monitoring changes.

4. THREE DISTINCT MASK SYSTEMS AND FOCUSED MONITORING

Implement and propagate these as separate fields:
A. observation/acquisition mask M_obs:[B,1,F,Rx,Tx], optionally time-indexed;
B. spatial focus ROI M_roi:[B,1,Z,Y,X], in physical coordinates;
C. supervised lesion labels and label-validity masks, used for losses only.
Also preserve calibration/baseline validity and padding masks where applicable.
A measured zero is not missing data. A prior focus ROI is not ground truth.
Never derive inference inputs from the current target lesion mask/coordinates.

Preserve existing verified monitoring mode names if present. Otherwise implement
configurable full-scan, ROI-focused acquisition and baseline-relative ROI tracking
presets. Do not invent supposedly agreed fixed budgets. Support arbitrary
valid masks plus structured Tx selection, Rx selection, available frequency
selection and failed-channel masks. Train with representative structured modes;
independent random scalar dropout alone does not teach focused monitoring.

Generate focus requests from a clinician-specified ROI, an earlier baseline
prediction or a recorded previous-frame prediction. Include margin, displacement,
growth/shrinkage and imperfect-ROI training cases. Acquisition selection can
use a supplied offline sensitivity map; provide a clearly labelled geometric
fallback, not a claim that nearest antennas guarantee maximum sensitivity.
Selectors must not inspect current ground truth or hidden held-out measurements.

Export the actual physical Tx/Rx/frequency scan list and budget accounting.
Selecting fewer Tx can save forward excitations; discarding Rx values after
a full-Tx COMSOL solve does not retroactively save those solves. Mark any use
of reciprocity to infer a channel as inferred, not independently acquired.

Apply M_obs before normalization/model processing, preserve it through G and
MSRNN, and normalize masked losses by valid support. Hidden entries must not
affect input statistics or predictions. Define all-missing behavior explicitly.
Never mark GAN-filled or predicted channels as measured. Match real/fake GAN
mask distributions and prevent masks/padding becoming discriminator shortcuts.
For the default path, remask G's output before MSRNN; any future imputation must
be a separately flagged feature, never silently substituted for observations.
Focused outputs must identify their assessed region; unassessed outside-ROI
space is not a confirmed negative. Provide configurable global-refresh behavior
and an escalation/request-full-scan flag for unsupported or drifting ROIs.

5. FINISH THE 3D CYCLEGAN

Retain two independent 3D residual-attention U-Net generators G:X->Y and F:Y->X
and two 3D PatchGAN discriminators. These operate over measurement axes
[frequency,Rx,Tx], not anatomical XYZ. Preserve phase with real/imag channels,
and condition on acquisition masks/frequency coordinates where needed.

Use the thesis chapter 4, printed pp.52-58, Figures 4.3/4.4, as the source for
the residual/attention design; its original 2D time-domain magnitude formulation
must be distinguished from our 3D complex frequency-domain adaptation. Keep
the existing 32/64/128/256 feature schedule as a documented starting default,
with measured memory/performance deciding justified changes.

Use anisotropic strides that preserve the short frequency axis, odd padded
frequency kernels, and verified small-array discriminator geometry. Handle
18x18 matrices and the eventual selected frequencies without fake image resizing
or an implicit minimum of 50 frequency samples. Prefer a linear complex residual
output with near-identity initialization where appropriate; remove unqualified
Tanh saturation unless a documented reversible scaling makes it appropriate.
Use small-batch-safe normalization and record actual receptive fields.

Implement correct separate LSGAN G/D updates, cycle/identity losses, replay
buffers, freezing/detaching and complete resume. Add masked paired complex
reconstruction supervision when paired corrupted/clean simulations exist.
Use healthy/lesion or baseline/current pairs, when actually available, to test
and penalize loss of the small differential signal. Add amplitude/phase metrics
with a sensible low-amplitude floor; cycle consistency alone is not a proof
of lesion-signal preservation. Only apply reciprocity penalties where the clean
calibrated domain and observed reciprocal pairs justify them; passivity checks
belong to appropriately normalized raw S, not arbitrary DeltaS.

Define X and Y precisely for each training manifest. Additive hardware/noise
corruption, reference subtraction and phantom-background removal are not
interchangeable target definitions. The reconstruction target must remain
consistent with the decluttered signal domain. Fix any code that compares
G(x) with an unrelated unpaired y using PSNR/SSIM as if it were a ground-truth
pair. Report paired metrics only for verified aligned examples.

6. IMPLEMENT MSRNN AND THE MISSING ENHANCEMENT U-NET

Use the FCERNN paper's Conv1D encoder -> 3D latent -> learned 3D decoder as
the core, fixing the audited shape failure. Preserve measurement masks and
fixed physical port ordering. Use a fixed candidate-frequency grid with observed
values in their correct slots and an explicit availability channel, so all
frequency candidates can share architecture and parameter count. Store actual
GHz; zero-filled missing slots are not interpolated observations.

A concrete compact default to implement and verify is:
- eight Conv1D encoder stages, widths 32,32,64,64,128,128,256,256;
- explicit length handling, e.g. for the ten-slot pilot grid the audited stride
  schedule yields length 51; adaptive pooling to length 32 and a 1x1 Conv1D
  from 256 to 1024 channels gives 32768 elements per sample;
- explicit reshape to [B,64,8,8,8], which also contains 32768 elements;
- learned decoder stages to 16/32/64/128 cubed, configurable widths such as
  128/64/32/16, then two property-output channels with native 128-cubed output.
This latent adapter is our engineering choice, not a claimed paper detail.
Verify actual arithmetic, feature ordering and memory. A clearly justified
better compact alternative is acceptable, but implement and document one default
and keep it identical across frequency-count candidates. Do not use a giant
dense volume-output layer or interpolation disguised as native resolution.

Add the separate full-3D enhancement U-Net inside MSRNN. Use a compact residual
encoder-decoder, small-batch-safe normalization, skip connections and a
two-channel residual property correction. A 16/32/64/128 channel schedule is
a starting default. Initialize the correction near zero. Condition on coarse
maps plus optional baseline maps, ROI and context-validity channels; all optional
context must have an explicit missing-context path. Add the residual-embedding
conditioning and physics interfaces specified below. Keep the coarse output and
use direct coarse and refined property supervision plus a modest edge/gradient
term. Do not turn enhancement into a lesion classifier or suppress weak changes
to make images smoother. Distinguish this 3D adaptation from the paper's
slice-based enhancement U-Net. Learn signed monitoring residuals; apply physical
constraints to absolute property maps, not to signed changes.

6A. IMPLEMENT REAL PHYSICS-INFORMED MSRNN, NOT JUST PROPERTY BOUNDS

Transfer the mechanisms in the supplied papers and earlier research notes;
do not require literal reproduction of their 2D architectures. Implement a
differentiable HeadsetForwardOperator Phi, its training and independent validation,
and a physics-conditioned correction path inside MSRNN. Phi is distinct from
CycleGAN's reverse generator F; use unambiguous class/checkpoint names.

Default Phi implementation: a shared compact 3D volume encoder with a
frequency/Tx/Rx/geometry-conditioned complex port decoder. Cache volume features
and decode all requested antenna pairs without 18 unrelated forward networks.
Include actual source excitation, polarization/orientation, calibration/reference
and hardware-layout context when available. Phi may use fixed calibration
references but must not consume current measured responses as a shortcut to
copying its own target. Specify exactly which signal representation it predicts.
Export predicted signals in the same units, axes, calibration convention and
real/imag representation as the observed data.

Frequency-dependent materials must remain physically coherent: two maps at
0.85 GHz alone do not uniquely specify unknown dispersion at all frequencies.
Use a shared frequency-conditioned two-property readout from the reconstruction
features to supply x_hat(f) at observed frequencies, while displaying the primary
maps at the fixed reference frequency. Phi consumes properties at the queried
frequency plus frequency/source context. Build supervised spectral property labels
from the actual material library; store material-ID volumes and spectral tables
when that avoids duplicating large label arrays. Ground-truth material IDs are
labels, never inference inputs. Do not require new EM solves just to rasterize
known property labels. The pilot uses constant material spectra.

Train Phi independently using paired clean head-property/S-parameter data and
matching sources/layouts. Include healthy/lesion and baseline/current pairs to
supervise small signal differences, not merely dominant total-field response.
Validate on held-out configurations, spatial regions, contrast signs and source
conditions. Record full-signal error, differential error, error relative to the
lesion signal/noise floor, and worst-channel/ROI behavior. Set predeclared gate
thresholds in configuration and save a versioned validation report. Paper RMSE
values or speedups cannot substitute for our headset validation.

After validation, freeze Phi parameters and use eval mode during inverse
training, while preserving gradients through Phi with respect to its inputs.
Do not put the entire forward call under no_grad and break inverse training.
Test the expected gradient flow explicitly. Inverse outputs can exploit surrogate
errors even when Phi is frozen; use independent validation anchors from the
existing full-wave data and flag out-of-domain states. Do not launch extra COMSOL
runs automatically. An absent/unqualified Phi checkpoint disables physics feedback
with a clear recorded reason, rather than substituting random weights or claiming
physics consistency. Finish this implementation even though trained weights arrive
after main data generation.

Implement masked, noise-scaled complex measurement consistency:
- Snapshot: compare predicted and original calibrated measured responses.
- Monitoring: compare Phi(x_baseline(f)+Delta_x_hat(f)) - Phi(x_baseline(f))
  with the corresponding measured baseline-current difference.
Use measured/calibrated noise scales with stable floors when available. Restrict
covariance/whitening to observed channels before mixing values; missing channels
cannot affect normalization. Freeze the baseline context for that comparison.
Default the consistency reference to the original calibrated observations, not
solely the GAN output, so GAN/reconstruction agreement cannot hide erased changes.

Implement and log the reconstruction objective as distinct terms:
L = lambda_property*L_property + lambda_change*L_change
  + lambda_data*L_complex_data + lambda_gradient*L_gradient
  + lambda_temporal*L_stable_temporal + lambda_equation*L_equation.
Use normalized MSE or Huber/Charbonnier with explicit physical channel scales,
foreground/background balancing and healthy-case false-positive supervision.
Do not claim Huber alone solves small-lesion loss imbalance. Supervise coarse and
refined property maps and signed change/contrast outputs separately. Current
ground-truth lesion masks may weight training losses but never model inputs.
Enable temporal consistency only on labelled stable/no-change intervals, with
per-region validity; do not smooth every real progression event toward zero.
Ramp the data term after the forward gate passes and log term magnitudes.

Keep state/PDE consistency separate from port-data consistency. A true Maxwell
or Lippmann-Schwinger residual requires fields/currents, excitation, material
conventions and validated boundary/operators. Include an explicit operator-based
loss interface and deterministic tests on an analytically or numerically known
small fixture. It is disabled in production unless those prerequisites exist;
an image gradient/TV penalty must never be reported as Maxwell consistency.
Implement the complete default port-surrogate consistency path; the field/PDE
extension is not a prerequisite for the frequency pilot.

6B. USE MEASUREMENT RESIDUALS TO DRIVE THE ENHANCEMENT U-NET

Implement one compact correction stage by default, with configurable depths
0/1/2/3 and fixed depth across frequency candidates:
x_0 = CoarseInverse(y_processed, M_obs, context)
r_k = M_obs * (y_calibrated - Calibrate(Phi(x_k, context)))
z_k = ResidualEncoder(r_k, M_obs, frequency_and_port_context)
x_(k+1) = x_k + Enhance3D(x_k, z_k, ROI, baseline, context_validity).
Use the differential residual above in monitoring mode. Encode measurement
residuals into a compact 3D latent at the U-Net bottleneck; do not allocate a
full-resolution image volume for every Tx/Rx channel. Keep frequency/port
coordinates explicit. Use bounded learned update scaling and near-zero residual
initialization. Make weight sharing versus separate correction blocks explicit.

The paper's between-iteration correction is different from longitudinal patient
change; distinguish both names, targets and outputs. Transfer the target-sensitive
plus whole-volume supervision idea using actual paired property labels, rather
than inventing clinical labels from intermediate corrections.

Support an explicit image-only enhancement route with residual-context validity
set to false when Phi is unavailable. Train the required missing-context behavior;
do not silently supply random residual features. This keeps the completed
architecture usable for the current pilot without pretending four spheres can
validate a general head forward solver. Phi can be training-only when used just
for a loss; when actual inference correction uses Phi, count its time/memory in
the deployment budget. Do not silently omit a deployed dependency.

6C. MAKE SUBSPACE PROCESSING AND FOCUSED ACQUISITION INFORMATIVE

Implement an optional low-rank clutter module with explicit fit/transform/save/
load and a bypass. Use a calibration/healthy training cohort where available;
one healthy simulation is not a basis for population/repositioning variability.
The supplied paper fits temporal directions separately per Tx/Rx pair across
scans; a complex frequency-domain or joint-channel adaptation must be documented
and independently assessed. Do not blindly copy its two-direction choice onto
3-10 observed frequencies. Zero-padding/IFT does not create extra observations.

Default to a soft subtraction strength with preserved original input, estimated
clutter/residual and source-validity metadata. A hard filter must not discard
the only input carrying whole-head anatomy. Fit missing-data coefficients using
observed entries, e.g. weighted regularized projection, instead of treating
missing zero-filled values as measurements. Scope fitted bases to hardware,
calibration state and selected frequency layout; retain lesion-response tests.

For focused monitoring, implement training-time structured Tx holdout and
reprojection losses on genuinely withheld measured channels where labels exist.
Draw these only from training geometries and the candidate's selected frequencies.
Retain whole-configuration split isolation. A reconstruction that explains held-out
measurements provides a useful check beyond visual sharpness. Do not supervise
a K-frequency candidate with hidden observations from the other 10-K frequencies.

Use signal diversity/sensitivity relative to noise for optional acquisition
ranking. Implement offline correlation/singular-value/effective-rank diagnostics
on training-only responses or verified sensitivity operators. Raw healthy S may
be dominated by coupling; use lesion/change-sensitive responses where available.
Two training object cases cannot establish general operator rank or an optimum
frequency subset. Diagnostics supplement, never replace, held-out reconstruction.
Keep the supplied CSV subsets as the primary frequency-count protocol. Any
exploratory subset selection uses training cases only, is separately labelled,
and cannot silently change the primary test or consume its validation/test cases.

7. IMPLEMENT THE SEPARATE 3D SEGMENTATION U-NET

Use an independent compact residual 3D U-Net, e.g. widths 16/32/64/128/256,
with small-batch-safe normalization and optional training-only deep supervision.
Input refined dielectric maps and, for monitoring, optional baseline/difference
maps plus ROI/context validity. Output full-resolution logits with a documented
label schema, separate from dielectric regression.

Support binary lesion segmentation and the main dataset's healthy/background,
ischemia, hemorrhage, tumor and edema labels. Use Dice plus BCE for binary;
Dice plus CE for mutually exclusive multiclass voxels; use multilabel sigmoid
heads only when the actual label schema permits overlapping voxel classes.
Multiple lesion types in one volume do not by themselves require multilabel
voxels. Handle healthy/empty examples and absent classes correctly. The sphere
pilot has object masks, not clinical lesion-type labels; never invent stroke
or edema labels for it. Missing clinical labels must not become fake background.

Train on reconstructed/refined distributions, with detached or appropriately
cross-fitted training predictions, rather than only perfect target volumes.
Keep segmentation gradients detached from MSRNN initially. Support later
controlled joint fine-tuning as a separately evaluated stage. Never use current ground-truth
segmentation as a focus/baseline conditioning input. Emit connected components,
centres in mm, volumes in mm^3/mL, class probabilities and baseline changes
with explicit ROI validity. Validate probability calibration before claiming it.

8. COMPLETE REAL DATA LOADERS AND TRAINING INFRASTRUCTURE

Implement MAT-v5 and a scalable manifest-based format, with validation and
conversion tools. Support optional MATLAB-v7.3/HDF5 through an explicit tested
reader rather than assuming every MAT uses SciPy's v5 layout. Include subject,
head, geometry, lesion configuration, sequence/baseline, source domain, material
reference frequency, clean/corrupted pairing, axes, physical grid, port layout,
frequencies, acquisition mask and label-validity metadata. Validate healthy,
single-lesion and clinically planned multi-lesion samples without expanding
the agreed COMSOL case budget. Old 24-port/50-frequency plans are superseded.

Split by the underlying subject/head/geometry/sequence as appropriate; keep
noise variants, focus masks, paired measurements and baseline frames together.
Input corruption may vary hardware gain, phase, additive noise and reference
drift under explicit models; do not call arbitrary Gaussian noise validated SDR
noise. Do not rotate/translate reconstruction targets without the corresponding
physically consistent microwave measurements. Segmentation-only spatial
augmentations must transform all its volume inputs/labels together.

Provide executable stages for:
1) paired/unpaired mask-conditioned 3D CycleGAN training;
2) independent headset-forward-operator training, validation and gate report;
3) clean/calibrated coarse reconstruction training;
4) coarse+enhancement training and validated-physics fine-tuning with Phi frozen;
5) segmentation training on realistic reconstructions;
6) controlled joint fine-tuning with explicit freeze schedules;
7) snapshot and focused/baseline monitoring inference.
Include real objectives, optimizers, schedules, early stopping, reproducibility,
AMP, memory controls and atomic complete checkpoints: model/optimizer/scheduler/
scaler/RNG states, splits, preprocessing, frequency grid, masks and architecture.
Recover after interruption without silently restarting completed stages.
No production path may silently fall back to random inputs/targets.

Target Windows, Ryzen 9 9900X, 32 GB RAM and RTX 5070 Ti 16 GB. Use practical
batch sizes, sequential stage training and optional gradient checkpointing.
Profile latency and peak memory on the real target shapes when hardware exists;
the sub-two-second goal is a target to measure, not a claimed result.

9. PREPARE THE RECONSTRUCTION FREQUENCY-COUNT TEST

This is the only frequency performance ablation. Compare the implemented
MSRNN reconstruction stack, including enhancement, at K=3..10; retain coarse-only
metrics as a control. Use the same physics-feedback setting for every candidate.
If no eligible Phi exists, explicitly use the image-only enhancement path for
all candidates and record that scope. Bypass CycleGAN and do not use segmentation performance
to select K. Structural GAN compatibility tests are not a GAN frequency study.

Use frequency_subsets.csv, with correct conversion from MATLAB 1-based indices.
Use all 18 transmitters and receivers for each candidate. The pilot split is:
case 0 empty reference; train cases 1 and 4; validation case 2; test case 3.
Sphere cases are r15 at (0,0,0), r15 at (25,-20,15), r25 at (0,0,0), and r25
at (25,-20,15), in case-ID order 1..4. There are only four object geometries;
Tx/frequency samples are not extra independent heads.

Keep architecture, parameter count, loss, training budget, preprocessing
algorithm, augmentation policy and paired seeds consistent across K. Train
candidate weights from scratch; reuse completed ten-frequency comparator runs.
Compare to zero-contrast and training-only mean-volume baselines. Report both
property errors, foreground/background-balanced contrast error, localization,
volume error and seed variation; avoid mostly-empty-background accuracy.
The pilot's two normalized property maps coincide, so it cannot establish
independent permittivity/conductivity recovery or clinical segmentation skill.

After a useful ten-frequency baseline, prepare selection of the smallest K
within 5% of its mean validation reconstruction error, subject to predeclared
localization checks. Keep the test geometry sealed until selection. Report an
inconclusive pilot honestly instead of forcing a frequency count. Serialize
the chosen grid/subset into a versioned deployment/data-generation config used
by every stage; choosing K must not require an architecture rewrite.

10. VERIFY AND DELIVER THE WORKING IMPLEMENTATION

Run bounded meaningful tests, not a full research training campaign:
- real MAT loading, complex/axis/port round trips and translated voxel geometry;
- snapshot, signed no-change/growth/shrinkage monitoring and missing-baseline paths;
- full/focused/failure masks, measured zeros, reference-mask intersection,
  padding and all-missing behavior; hidden entries cannot change predictions;
- G/F/D forward/backward and optimizer separation; pair-aware metrics;
- coarse+enhancement+segmentation end-to-end shapes and finite gradients;
- FCERNN/MSRNN frequency-candidate shape and parameter-count consistency;
- empty/healthy, multiclass, ignored-label and overlapping-label loss behavior;
- Phi forward/validation reports and frozen-parameter versus nonzero input-gradient behavior;
- absolute/differential data losses, dispersion/query-frequency consistency,
  no-change temporal gating and independent observation-versus-GAN consistency;
- masked subspace fit/load/bypass and structured held-out-channel loss isolation;
- checkpoint reload/resume and bypass identity; one tiny pipeline overfit/debug
  fixture may check trainability but is not evidence of generalization.
CPU tests may use reduced widths/grids, but additionally verify production
configuration forward shapes and full-size memory where feasible. State what
was not run; do not equate tiny smoke tests with target-GPU validation.

Deliver working code, train/infer/evaluate/convert/validate CLIs, configs for
every stage and acquisition mode, and docs/FINAL_MICROSCAN_ARCHITECTURE.md.
Include exact layer tables/parameter counts, tensor and physical contracts,
mask flow, data-generation/export schema, label definitions, source-versus-
adaptation decisions, test evidence and short Windows-friendly next commands.
Choose defaults and finish integration rather than returning architectural
TODOs. Leave frequency_config.status=pending_selection until the pilot result.
Do not launch new COMSOL solves or full training automatically: finish bounded
verification, then hand over commands in the order frequency pilot -> freeze
frequencies -> main data generation/export -> staged pipeline training. Preserve the decision/evidence matrix and date-qualified
research findings below in the implementation handoff. Do not introduce hardware
redesigns, the discarded dielectric-blob pretraining stage or automatic extra
COMSOL case generation.

Read the supplied FCERNN paper and thesis directly. Useful additional primary
references for design and implementation are the original CycleGAN paper
(https://arxiv.org/abs/1703.10593), the authors' nnU-Net residual-encoder guidance
(https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md),
and the installed-version PyTorch documentation. Cite exact source sections and
distinguish their findings from MicroScan-specific engineering choices.
```

## Research synthesis and implementation decisions

Reviewed 28 October 2026. The recent-publication search window was **28 September
through 28 October 2026**. Publication/online-release dates were distinguished
from crawl dates, issue dates, conference dates and later project announcements.
This was a focused primary-source search, not an exhaustive systematic review.

The supplied audit, earlier research notes, FCERNN PDF, full thesis, full latent-
space PDF, full forward-solver PDF, and inverse-solver abstract/figure screenshots
were used. The unavailable inverse-solver body was not reconstructed from guesses.
The discarded dielectric-blob pretraining phase is excluded. The completed sphere
pilot remains available solely for the already agreed reconstruction-frequency
experiment, not as validation of a trained clinical pipeline.

### What becomes code

| Evidence / supplied idea | MicroScan adaptation | Implementation decision |
|---|---|---|
| FCERNN + separate enhancement | Measurement-to-volume coarse network plus native full-3D residual enhancement | Required; repair latent arithmetic and retain two-property regression |
| Thesis residual-attention CycleGAN | Phase-preserving 3D measurement-space G/F/Ds with masks and paired differential supervision | Required module; default bypass for clean frequency pilot and baseline mode |
| Earlier notes: baseline versus no baseline | Absolute and signed-change heads/context, with aligned measurement and image baselines | Required; do not force full-head inference from DeltaS alone |
| Earlier notes: complex forward-data consistency | Shared source/frequency/layout-conditioned port forward operator; masked original-measurement loss | Required implemented path; trained weights pass an independent validation gate |
| Earlier notes: frozen surrogate | Independently train Phi, freeze its parameters, retain input gradients | Required; frozen weights alone do not protect against surrogate exploitation |
| Earlier notes: Maxwell/state residual | Separate optional operator-based field loss, tested with known fields/sources/boundaries | Implement interface and small fixture; no fabricated PDE residual from property gradients |
| Earlier notes: gradient and temporal losses | Edge supervision and stable-period-only temporal regularization | Required; foreground balancing and signed progression remain explicit |
| Inverse-solver abstract/figures | Measurement residual -> compact latent -> conditional enhancement corrections | Required adaptable path; one correction default, measured depth expansion |
| Target and whole-domain losses in figures | Auxiliary property-change supervision plus whole-volume loss | Required; loss masks never become current-lesion inference inputs |
| Latent-space cohort projection | Optional per-path/mask-aware soft projection with original signal retained | Required optional module; fit only an eligible calibration cohort |
| Multi-frequency NeuralBIM | Explicit state/data objectives and controlled frequency-task weighting | Transfer loss structure; learned weights are not calibrated hardware reliability |
| Noise2Noise | Optional repeated-acquisition training under verified conditional-noise assumptions | Data adapter/loss support; do not treat shared drift as independent noise |
| Experimental-training evidence | Manifest source labels, calibrated phantom fine-tuning, acquisition-day splits | Required infrastructure; no claim that synthetic noise alone closes sim-to-real |
| Residual U-Net segmentation evidence | Compact separate 3D segmentation model with realistic reconstructed inputs | Required; detach segmentation gradients from inverse stage initially |
| October cascaded CNN | Residual staged refinement and stage-depth/latency reporting | Supports the chosen compact correction path, not mandatory deep unrolling |
| October frequency-diverse sensing | Correlation/SVD diagnostics for information redundancy | Optional training-only diagnostics; reconstruction remains frequency-selection criterion |
| Registration/positioning evidence | Pose/affine metadata, baseline validity and nuisance checks | Required; image registration alone does not repair shifted measurement subtraction |

### Newly published within the requested month

**N1. Kang et al., Application of Cascaded Convolutional Neural Network in
Electromagnetic Imaging.** Engineering Proceedings 159(1), 2;
DOI: https://doi.org/10.3390/engproc2026159002.
Publisher release: **9 October 2026**; originally presented at ICEIB, 24-26 April
2026. The publisher text describes 2D TM reconstruction, physics-based initial
backprojection and residual encoder-decoder cascades, with comparisons under
5% Gaussian noise. Transfer: a coarse inverse followed by residual refinement,
intermediate supervision and depth/latency checks. This does not establish an
optimal stage count or performance for the 3D MicroScan headset.

**N2. Arif et al., Broadband Near-Field Computational Microwave Imaging Enabled
by Nonperiodic Dispersive Unit-Cell Loading in a Passive Metasurface.**
Advanced Electronic Materials, e70606;
DOI: https://doi.org/10.1002/aelm.70606.
First published: **8 October 2026**. Its 18-28 GHz passive-aperture work assesses
frequency-dependent field diversity using correlations and singular values.
Transfer: inspect redundancy/information diversity when comparing acquisition
subsets. MicroScan should use its own calibrated responses or sensitivity operator,
not transfer that aperture's sampling interval, resolution or hardware.

**Dated carefully:** A 3BAtwin project announcement on **7 October 2026** describes
a breast/axillary pilot and positioning/repeatability challenges. The announcement
date alone does not prove the paper first appeared that month. It informs the
pose/calibration metadata requirement, not a new neural architecture claim:
https://3batwin.campus.ciencias.ulisboa.pt/2026/10/07/new-paper-pilot-study-of-an-axillary-and-breast-microwave-imaging-prototype-in-healthy-volunteers-and-patients/

Recent search results also included old papers crawled in October, hardware
inverse-design papers, and an inverse-scattering-transform FNO for time-dependent
PDEs. These were not treated as new evidence for electromagnetic brain tomography.
A September 3D level-set preprint candidate could not be verified at its primary
record and was not used to drive the implementation.

### Earlier primary sources retained in the synthesis

- **Cheng et al., 2022, FCERNN:** https://doi.org/10.3390/diagnostics12112786.
  Supplied full PDF; Conv1D-to-3D reconstruction and separate slice enhancement.
  The text's incomplete kernel list remains an ambiguity, not a source for invented
  exact layer specifications.
- **Lai, PhD thesis, 2025:** supplied full PDF, chapter 4. Original model uses
  time-domain magnitudes and 2D U-Net generators. The project explicitly adapts
  these mechanisms to complex 3D measurement tensors.
- **Lai et al., 2024:** https://doi.org/10.1109/JERM.2024.3409846.
  Correct head-clutter-paper identity; the earlier notes' breast-artifact link was
  not the source of CycleGAN architecture details.
- **Guo et al., latent-space removal, 2025:**
  https://doi.org/10.1109/TAP.2025.3606294. Supplied full author manuscript;
  per-Tx/Rx cohort temporal bases and subsequent confocal imaging. Its two-direction
  rank choice is not automatically suitable for a sparse 3-10-frequency grid.
  Clinical 81.6% denotes correct-quadrant localization in 31/38 stroke patients;
  the separate healthy false-positive illustration contains four volunteers.
- **Guo et al., forward solver:** https://doi.org/10.1109/TMI.2025.3648756.
  Supplied full published PDF; online 26 December 2025, issue May 2026. Uses a 2D
  TM formulation, incident-field inputs, local/global field constraints and 1 GHz
  testing. Transfer source-aware/local/global consistency; implement a distinct
  3D port-calibrated forward model for our headset.
- **Guo et al., inverse solver:** https://doi.org/10.1109/TMI.2026.3653974.
  May 2026 issue; user-supplied abstract and screenshots only. The figures support
  studying residual correction blocks, target/whole supervision and stage depth.
  They do not supply the missing full definitions, hyperparameters or code.
- **Liu et al., Multi-frequency NeuralBIM, 2024:**
  https://arxiv.org/abs/2409.01315. Primary full HTML reviewed; 2D inverse scattering
  and homoscedastic task-loss weighting. Transfer concepts rather than its physical
  operator or dimensional assumptions.
- **Dong et al., Physics-Informed Neural Operator for Electromagnetic Inverse
  Scattering Problems:** https://arxiv.org/abs/2603.25404; submitted 26 March 2026,
  preprint, outside the recent-month window. Full HTML reviewed. Separates induced-
  current state/data consistency and TV while optimizing the inverse problem;
  useful loss separation, not evidence of sub-two-second 3D amortized inference.
- **Lehtinen et al., Noise2Noise, 2018:**
  https://proceedings.mlr.press/v80/lehtinen18a.html. Noisy-target training depends
  on conditional target statistics; repeated correlated drift is not automatically
  an unbiased training target.
- **Cathers et al., experimental training, 2025:**
  https://doi.org/10.1109/OJAP.2025.3533373. Author/publisher manuscript reviewed;
  experimental or mixed training improved that study's experimental inversions.
  Transfer source-aware training and day/geometry splits, not its target accuracy.
- **Isensee et al., residual segmentation baselines, 2024:**
  https://arxiv.org/abs/2404.09556 and
  https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md.
  Transfer compact residual U-Net blocks, validation and hardware-aware planning.
- **Chu et al., CycleGAN information hiding, 2017:**
  https://arxiv.org/abs/1712.02950. Supports directly testing input-specific lesion
  preservation rather than relying on cycle loss or visual plausibility alone.

### What has and has not been validated here

The real pilot MAT has the expected finite complex [18,18,10,5] tensor and all
50 frequency-case jobs complete. The proposed latent arithmetic is dimensionally
consistent. Those facts do not establish reconstruction skill, headset surrogate
accuracy, clinical performance, 3D adaptation performance or target-GPU latency.
Only this research-backed implementation prompt is produced in this task.
No new COMSOL solves, model implementation or training was performed here.
