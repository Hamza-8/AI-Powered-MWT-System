Reimplement the complete MicroScan AI pipeline described in the attached instructions. The previous implementation became inaccessible when its execution environment went offline before the changes were committed or published. Rebuild from the existing repository and attached specifications; do not attempt to access the old environment.

This is a complete implementation task. Do not stop at an audit, plan, tensor adapter, or partial scaffold.

AUTHORITATIVE INSTRUCTIONS

Read both attached documents completely:

1. MicroScan_Codex_Prompts.md:
   Implement Prompt 2 and preserve its research/evidence distinctions. Prompt 1 is historical context; a separate architecture-audit task is unnecessary.

2. MicroScan_Physics_Followup_Prompt.md:
   Apply this as part of the initial implementation. It overrides conflicting physics defaults in the earlier prompt.

The latest decisions are:
• A: supervised reconstruction plus enhancement, without Phi.
• B: the same deployment architecture as A, with an independently trained, qualified, frozen electromagnetic forward operator Phi used only during training for measurement consistency.
• C: an optional experimental inference-time physics-feedback path, enabled explicitly.
• The reconstruction-frequency pilot uses A for every candidate. It does not require a surrogate.
• Phi is distinct from CycleGAN’s reverse generator F.

SOURCE REPOSITORY

Hamza-8/AI-Powered-MWT-System

Known starting implementations:
• CycleGAN branch: codex/build-pytorch-cyclegan-for-clutter-removal
  Commit: 6c14ab3c376f9fee0d14bc9283bdd8e7f768f44f
• FCERNN branch: codex/implement-3-d-fcernn-in-pytorch-2.x
  Commit: ce342caa663c24e6c344a0de056ad7c324386ef2

Inspect current repository state and AGENTS.md before editing. Integrate useful code from both implementations into one package. Preserve unrelated work. Resolve their conflicting models/data/utils paths deliberately.

The original implementations have dummy-data training and incompatible assumptions about frequency/port counts and latent shapes. Do not preserve these as the production path.

COMPLETE IMPLEMENTATION

Implement and connect:
1. Real calibrated complex-measurement loading, acquisition masks and optional baseline context.
2. Phase-preserving 3D CycleGAN decluttering over frequency/Rx/Tx, with independent G/F/discriminators and correct training updates.
3. MSRNN: an improved FCERNN-style measurement-to-volume coarse inverse followed by the separate full-3D residual enhancement U-Net.
4. A separate 3D segmentation U-Net supporting the actual clinical label schema.
5. Full-scan, focused acquisition and baseline-relative monitoring, including signed changes and imperfect/missing baseline or ROI context.
6. Independent forward-operator training, qualification, frozen-operator consistency losses and optional C feedback.
7. Real staged training, inference, evaluation, conversion, profiling and complete checkpoint/resume commands.
8. The reconstruction-frequency experiment and explicit frequency-selection/freeze workflow.

Use the exact tensor contracts, architecture defaults, masks, dispersion handling, loss definitions, source limitations and training requirements in the attached documents. Do not invent recurrence because the reconstruction stack is called MSRNN.

CRITICAL DATA CONTRACTS

The supplied frequency pilot contains:
• Complex128 S with MATLAB axes [Rx, Tx, frequency, case], shape [18,18,10,5].
• Frequencies 0.60–1.05 GHz in 0.05 GHz increments.
• Physical antenna ordering already 1–18.
• COMSOL IDs in that physical order:
  [7,8,9,10,11,1,2,3,4,6,12,13,14,15,16,17,18,19].

Do not reorder the physical axes again based on COMSOL IDs.

Canonical network measurements are [B,2,F,Rx,Tx], with real/imaginary channels. Perform reference subtraction in complex128 before conversion to training precision.

The sphere target grid is cell-centred 128³ over [-60,60] mm. Preserve explicit XYZ coordinates versus ZYX array indexing and test the translated sphere. This object grid is not the main dataset’s whole-head extent.

Pilot split:
• Case 0: empty reference only.
• Cases 1 and 4: training.
• Case 2: validation.
• Case 3: sealed test.

There are four object geometries, not 900 independent training examples. Keep every derived noise/mask variant with its underlying geometry.

A measured zero is observed data. Observation masks, spatial focus masks and supervised labels are separate. Current ground-truth labels must never become inference conditioning. Hidden entries, including hidden NaNs, must not influence statistics or predictions.

PHYSICS AND DEPLOYMENT

A and B must use the same reconstruction/enhancement inference graph, parameter count and correction depth. B deployment must work without loading Phi or its checkpoint. Do not train B using a Phi-residual input and replace that input with zeros at deployment.

Keep Phi-dependent residual correction in C. Preserve ordinary measurement-conditioned enhancement in A/B.

For B, freeze Phi parameters and use evaluation mode while retaining input gradients into reconstruction. Keep raw S, calibrated differences and decluttered signals explicitly distinguished.

No qualified scientific Phi currently exists. Implement its independent training and qualification machinery, but do not substitute random weights or treat synthetic qualification fixtures as a qualified headset model. Explicitly report requested versus active physics behavior.

No COMSOL solves inside training updates. Maxwell field losses require actual fields, sources, operators and boundary support; image gradients are not Maxwell residuals.

FREQUENCY EXPERIMENT

Prepare K=3–10 reconstruction-plus-enhancement candidates from the supplied CSV. Bypass CycleGAN and exclude segmentation from K selection.

Use the fixed ten-slot input representation, matching architecture, budgets, preprocessing policy and paired seeds across K. Use image-only A throughout. Train each candidate from scratch; reuse only identity-matched completed work.

Keep the test geometry sealed until explicit selection. Compare validation results against zero-contrast and training-only mean-volume baselines. Preserve the documented selection rules and report an inconclusive pilot if the ten-frequency reconstruction is not useful.

Serialize the selected frequencies so every subsequent stage consumes the same frozen configuration. Leave status pending_selection until the experiment establishes a result.

Do not launch the full frequency experiment, main training, new COMSOL solves, or dielectric-blob pretraining during this implementation task.

PRESERVE THE WORK

Commit and publish coherent milestones as work progresses. Do not wait until the final response to preserve the implementation remotely.

Verify publication actually succeeded. If remote publication is unavailable, state that clearly and export a downloadable source archive and patch at milestones. A local commit alone must not be described as published.

VERIFICATION AND DELIVERY

Re-run verification from scratch. The previous report’s test counts and performance profiles are historical claims, not evidence for this rebuild.

Use meaningful bounded tests for real MAT loading and axes, translated targets, mask isolation, signed monitoring, all model training paths, GAN optimizer separation, spectral queries, physics gradients and domains, A/B/C behavior, segmentation labels, subspace processing, checkpoint/resume and the frequency protocol.

Include bounded optimizer checks and native 128³ production-configuration structural forwards where feasible. Clearly distinguish these from scientific training and reconstruction accuracy.

Target Windows with an RTX 5070 Ti 16 GB. If CUDA is unavailable, do not claim GPU validation. Provide bounded GPU memory/AMP checks and the exact command to start/resume the frequency experiment. Keep CPU dependency pins separate from target-GPU installation guidance.

Deliver the complete code, configs, data-export schema, architecture documentation, fresh verification artifacts and published commit/PR link. Explain any difference from the specification and any genuinely unavailable inputs. Never claim the inaccessible implementation was recovered byte-for-byte.