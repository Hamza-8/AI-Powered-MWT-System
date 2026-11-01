Continue from your completed MicroScan implementation and verification report. This is a targeted follow-up, not a rebuild. Inspect the actual code, configs and checkpoint schemas first; retain correct existing work and implement only missing behavior.

Our decision: training-only forward consistency is the default physics-informed approach. Inference-time forward feedback is a separate optional experiment. Phi is the electromagnetic forward operator; it is distinct from CycleGAN's reverse generator F.

1. Make these three configurations explicit:
A — supervised: coarse inverse plus 3D enhancement, with no Phi loss or inference calls.
B — training consistency: the same inverse/enhancement inference graph as A, trained additionally through an independently trained, qualified, frozen Phi. This is the preferred physics-informed configuration when eligible weights exist.
C — inference feedback: an explicitly selected experimental residual-conditioned correction path that evaluates Phi during inference.

A and B must have matching deployment architecture, parameter count and correction depth. Keep measurement-conditioned enhancement available in both; it must not secretly require a Phi residual. B's inference path must not load or call Phi, require its checkpoint, or replace a training-time Phi input with zeros at deployment. If the current enhancement consumes Phi residuals, separate that optional feedback path from the shared A/B path. Preserve useful existing C functionality without enabling it automatically.

2. Verify the physics loss, rather than adding another forward model.
Compute complex measurement consistency between Phi(predicted physical properties, frequency, source and physical context) and the corresponding calibrated observations on valid observed support. Preserve real/imaginary information, saved scales and spectral-property semantics. Keep the measurement domain explicit: raw S, background-subtracted DeltaS and decluttered target signals are not interchangeable. Use original calibrated observations for an independent physical check where the operator models that domain; do not equate them with CycleGAN outputs.

For signed longitudinal consistency, compare predicted current-minus-baseline responses with measured current-minus-baseline responses using compatible acquisitions and valid baselines. Do not confuse patient change with successive reconstruction corrections. Retain existing mask and covariance handling; hidden measurements must not influence losses, normalization or residual features.

Freeze Phi parameters, set evaluation mode, but preserve gradients through its inputs into the inverse/enhancement network. Reuse existing qualification checks and context bindings. Synthetic positive fixtures are mechanism tests only. No eligible Phi currently exists: keep the ordinary no-Phi path usable, report physics_requested versus physics_active, and fail clearly if a run explicitly requires qualified physics. Never silently label a no-Phi run as B.

COMSOL supplies training/reference data and independent validation. Do not add COMSOL solves inside training updates. Keep field-equation loss disabled without its actual field/source/operator/boundary prerequisites.

3. Keep the agreed frequency experiment unchanged.
Run reconstruction plus enhancement only, with CycleGAN bypassed and segmentation excluded from K selection. Use image-only A for every K because the pilot cannot qualify Phi. Preserve the existing CSV, grouped splits, sealed test geometry, paired seeds, fixed ten-slot architecture, selected-frequency isolation and declared selection rules. Do not add cases, frequencies, surrogate training or new COMSOL solves. Do not turn this into a physics-mode comparison.

Prepare a reproducible command or script that starts the existing protocol and resumes matching completed work, but do not launch the full experiment in this task.

4. Keep main training and experimental feedback ready.
Expose A/B/C consistently in configs, training, checkpoint metadata and inference. Save training physics provenance separately from inference requirements, so a B checkpoint deploys without Phi. Preserve compatibility with existing checkpoints or document an explicit migration.

Prepare matched A/B/C experiment configs for later use after frequency selection and suitable main data/Phi qualification. Record reconstruction and lesion metrics plus measured latency and memory when those experiments actually run. Do not run them now or claim improvement in advance. Preserve the existing CycleGAN, focused-monitoring masks and segmentation pipeline.

5. Verify only the changed behavior.
Add targeted checks that B's physics loss produces inverse-network gradients with no Phi parameter gradients; B inference succeeds without Phi and makes zero forward-operator calls; C requires an explicit compatible operator when feedback is enabled; masks and domains remain correct; and physics provenance survives checkpoint/resume. Include a bounded optimizer check exercising the actual B training path with a clearly labelled synthetic operator fixture. Reuse existing tests instead of duplicating them.

Run relevant regressions. If the target CUDA environment is unavailable, provide the exact bounded Windows/RTX5070Ti smoke/profile commands and state that GPU validation remains unrun. Do not pin the CPU-only PyTorch build for that machine. Do not launch full training or solves.

Return the commit/branch, files changed, what was already correct versus changed, executed verification, the command to begin the existing frequency experiment, and remaining data/hardware requirements. Keep scientific performance claims separate from structural verification.

