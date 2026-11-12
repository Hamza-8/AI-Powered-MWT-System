# MicroScan: AI-Powered MWT System

MicroScan

MicroScan is a research-stage, non-invasive brain-monitoring system designed to detect, localize, and continuously track intracranial abnormalities such as cerebral edema, stroke, tumors, and traumatic brain injury. The project combines microwave tomography, ultrasound-assisted anatomical mapping, and deep learning to reconstruct changes in the brain’s electromagnetic properties from measurements collected by a portable headset.

Conventional imaging methods such as MRI and CT provide detailed clinical images, but they are expensive, infrastructure-intensive, and poorly suited to continuous bedside monitoring. As a result, important changes may occur between imaging sessions without being detected immediately. MicroScan is intended to address this monitoring gap by providing rapid, repeatable measurements that can identify meaningful changes from a patient-specific baseline.

MicroScan is not intended to replace MRI or CT. Instead, its proposed role is to function as an adjunctive monitoring and early-warning system. When the system detects a significant change, clinicians could use established imaging methods to confirm and characterize the finding.

How it works

The proposed system uses a wearable multichannel antenna array operating across approximately 0.6–1.5 GHz. Microwave signals are transmitted through the head, and the resulting complex scattering parameters—particularly the amplitude and phase of inter-antenna transmission measurements—are recorded.

Because healthy tissue, blood, edema, and tumors have different frequency-dependent dielectric properties, intracranial changes affect how these signals propagate. MicroScan processes these measurements to estimate the spatial distribution of electromagnetic tissue properties and identify abnormalities.

A-mode ultrasound measurements are incorporated to estimate patient-specific head geometry and improve anatomical registration. This helps the reconstruction pipeline account for variations in head shape, tissue boundaries, and antenna positioning.

AI reconstruction pipeline

The current computational pipeline is organized into four main stages:

Clutter and artifact removal – A 3D CycleGAN-based model suppresses background clutter and system artifacts while preserving lesion-related information.
Electromagnetic reconstruction – A 3D fully convolutional recurrent neural network reconstructs volumetric dielectric-property distributions from microwave measurements.
Tissue-property estimation – Frequency-dependent measurements are fitted to the Cole–Cole model to estimate physically meaningful tissue parameters.
Lesion segmentation and monitoring – A 3D U-Net localizes suspected abnormalities and measures changes in their position, shape, and volume over time.

The system is being designed for rapid inference, with a long-term target of producing a reconstructed volume within approximately two seconds and a fallback target of ten seconds.

Development and validation

MicroScan is being developed through a staged simulation-to-hardware workflow. Initial experiments use parametric multilayer head phantoms to optimize individual antennas, array geometry, frequency selection, signal acquisition, and reconstruction performance. Later stages introduce MRI-derived voxel models with more realistic anatomy and heterogeneous tissue properties.

The dataset design includes:

Healthy baseline cases
Single-lesion cases with varied locations, dimensions, and dielectric properties
Clinically relevant multiple-lesion cases
Longitudinal scenarios representing lesion appearance, growth, movement, or resolution
Patient-to-patient variation in anatomy and tissue properties
Clean and noisy measurement pairs incorporating realistic hardware effects
Variations in antenna placement, RF clock stability, receiver noise, and calibration error

Following simulation, the system will be evaluated using physical tissue-mimicking head phantoms and RF measurement equipment. The primary validation objective is longitudinal monitoring: determining whether MicroScan can reliably recognize clinically meaningful changes relative to an earlier measurement of the same subject.

Repository contents

This repository documents the engineering and research development of MicroScan, including:

COMSOL electromagnetic models
Broadband antenna and array optimization
Parametric and MRI-derived head models
Synthetic dataset-generation workflows
Microwave signal and hardware-noise simulation
Clutter-removal and reconstruction networks
Cole–Cole parameter-estimation methods
Lesion segmentation and change-detection algorithms
Evaluation metrics and ablation studies
Phantom-design and experimental-validation protocols
Hardware architecture and data-acquisition planning
Project objective

The long-term objective is to develop an affordable and portable platform for repeated brain monitoring in intensive-care units, emergency settings, ambulances, and healthcare facilities with limited access to advanced imaging infrastructure.

By prioritizing change detection rather than attempting to immediately reproduce the full diagnostic detail of MRI or CT, MicroScan aims to establish a technically achievable and clinically relevant first application for microwave brain imaging.

Important: MicroScan is an experimental research project and is not a certified medical device. It must not be used for diagnosis, patient monitoring, or treatment decisions outside appropriately supervised and approved research.

## Rebuilt implementation

The executable package is now `microscan`. Its precise implemented contracts are in
[the rebuilt architecture](docs/FINAL_MICROSCAN_ARCHITECTURE.md),
[data export schema](docs/DATA_EXPORT_SCHEMA.md),
[research evidence ledger](docs/RESEARCH_EVIDENCE.md), and
[Windows runbook](docs/WINDOWS_RUNBOOK.md).

The current controlled sphere experiment is documented in the
[pattern/amplitude pilot](docs/PATTERN_AMPLITUDE_PILOT.md), including research
sources, the unchanged weighted training loss, final-epoch checkpoint policy and
Windows launch commands. Start with K10 seed17 before a frequency-count campaign.

The subsequent [controlled amplitude-routing test](docs/POSTNORM_AMPLITUDE_PILOT.md)
ablates the saved model and tests post-normalization conditioning at K10 seed17
with an unchanged loss, split and budget. Its training benefit remains unproven.

The separate [extent-loss experiment](docs/EXTENT_LOSS_PILOT.md) retains the
pattern-v1 architecture and tests a refined soft-Dice training term at K10 seed17.
It does not set the frequency count for head data generation.

In this implementation MSRNN is a feed-forward coarse inverse plus a separate full-3D
residual enhancement U-Net; it has no recurrent unit. The reference-frequency maps
and supervised spectral readout do not infer unlabelled Cole–Cole coefficients.
Frequency selection is pending; no scientific training or target-GPU performance
is established by the structural implementation checks.
