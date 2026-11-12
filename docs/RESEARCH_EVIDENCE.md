# Research evidence and engineering decisions

Rebuild review: 1 November 2026. The supplied prompt's research search was dated 28 October 2026 and covered 28 September–28 October 2026. This rebuild directly inspected the four supplied PDFs below. The additional publications in the inherited ledger have not been independently fetched during this implementation. Their dates and findings are retained as supplied claims rather than represented as a new literature search.

## Directly inspected sources

| Source | Exact source fact | MicroScan adaptation and limit |
|---|---|---|
| Cheng et al., Diagnostics 2022, 12, 2786, DOI 10.3390/diagnostics12112786; section 2.2.2, Figure 3, PDF pp. 5–7 | Eight Conv1D stages with widths 32/32/64/64/128/128/256/256; strided convolution, reshape, learned 3D decoding; separate U-Net enhancement. The text lists only seven kernel values for eight layers. | Retain the mechanism, choose explicit odd kernels and small-batch GroupNorm, pool/adapter to a known latent shape, and native 128³ decoding. The incomplete kernel list is not silently reconstructed as an exact source architecture. No recurrent unit is justified by the name. |
| Same paper, section 3.2, equations 6–8 | Study uses a 0.24 m cube, 256³/512³ voxel experiments, block-wise models, model/data misfits, and an MSE training objective. Labels at 300 MHz are tied to that study. | MicroScan reference frequency is 0.85 GHz and uses a single full-volume 128³ initial grid. Whole-head bounds must come from its anatomy. Sphere-pilot extent cannot stand in for the head. The paper's performance is not a benchmark result for this rebuild. |
| Lai PhD thesis, chapter 4, printed pp. 52–58, Figures 4.2–4.4 (Figure 4.3/4.4 on printed p.55, PDF p.73) | Modulated IFT, absolute time-domain magnitude and [0,1] scaling; two independent generators and discriminators; 2D 3×7 convolutions, 32/64/128/256 schedule, residual paths and attention gates. | Retain residual/attention and adversarial cycle mechanisms while using complex real/imaginary channels, full 3D frequency/Rx/Tx axes, odd padded frequency kernels, linear residual outputs and acquisition masks. This is a 3D complex-domain adaptation, not literal reproduction of the thesis network. |
| Guo et al., supplied author manuscript UQ3540db7_OA.pdf, DOI 10.1109/TAP.2025.3606294, section II-A, equations 1–6 and Table I | Cohort time-domain bases are estimated separately for each Tx/Rx pair; two clutter directions selected for that study; subtraction followed by confocal imaging. 43 healthy volunteers support the clinical basis and 38 stroke patients the pilot. | Optional observed-support complex frequency projection is an independent adaptation. Rank and cohort eligibility require validation; 3–10 frequencies do not justify copying a two-direction temporal basis. Preserve original measurements and use soft subtraction. |
| Same manuscript, section III-C, Table V and Figure 16 | 81.6% is the correct-quadrant stroke-detection/localization result (31/38 patients); four healthy volunteers are a separate raw-image illustration. | No claim of clinical segmentation accuracy, 81.6% voxel accuracy or an independently validated healthy false-positive rate is transferred to MicroScan. |
| Guo et al., Medical Microwave Imaging Using Physics-Guided Deep Learning—Part 1: The Forward Solver, DOI 10.1109/TMI.2025.3648756, sections II/III, Figures 2–4, equations 18–31 | 2D TM fields, source/incident-field-conditioned prediction, exterior/interior constraints and local/global checks; simulated fields evaluated at 1 GHz. Online 26 December 2025, current version 4 May 2026. | Implement a separate headset port operator Phi with explicit source/layout/frequency/domain bindings. Its learned port loss is not a Maxwell residual. Real fields, operators, sources and boundary support are mandatory before enabling the field-equation extension. The paper's speed and error numbers do not qualify our surrogate. |

The inverse-solver full text, earlier architecture audit, inverse-solver screenshots, material library, main whole-head paired dataset and qualified headset Phi weights were not supplied to this rebuild. The inverse-solver mechanism descriptions in the prompt are treated as design directions with unavailable details. No hidden full-text reconstruction or extra COMSOL case generation was performed.

## A/B/C decision override

A uses supervised inverse plus ordinary measurement-conditioned enhancement. B trains the identical deployment graph with an independent qualified frozen Phi loss. Phi is not a B deployment dependency and supplies no B inference feature. C alone uses explicit forward residuals during refinement and includes Phi runtime/deployment costs. No qualified scientific Phi currently exists. Synthetic positive qualification fixtures exercise machinery only.

## Preserved supplied research ledger

The following is retained from the user's implementation specification. Its original date qualifiers and source limitations remain attached; inclusion does not assert independent verification of every record in this rebuild.

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
