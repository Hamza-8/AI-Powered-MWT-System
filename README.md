# MicroScan: AI-Powered MWT System

MicroScan is a portable, non-invasive brain-monitoring system designed to detect and track changes associated with stroke, cerebral edema, tumors, and traumatic brain injury. It combines microwave tomography with ultrasound-assisted anatomical mapping and an AI reconstruction pipeline to produce rapid 3D estimates of intracranial tissue properties.

Unlike MRI and CT, MicroScan is being developed for repeated bedside measurements and continuous monitoring between conventional imaging sessions. Its primary goal is not to replace clinical imaging, but to provide change-from-baseline alerts that help clinicians identify deterioration earlier and determine when confirmatory imaging may be necessary.

The proposed system uses a multichannel antenna array operating across approximately 0.6–1.5 GHz to acquire complex microwave scattering measurements. These measurements are processed through a pipeline for clutter removal, electromagnetic-property reconstruction, Cole–Cole parameter estimation, and lesion segmentation. A-mode ultrasound is incorporated to improve anatomical registration and account for patient-specific head geometry.

This repository documents the development of MicroScan, including:

COMSOL electromagnetic simulations and antenna-array optimization
Parametric head phantoms and MRI-derived voxel models
Synthetic snapshot and longitudinal monitoring datasets
Realistic hardware, calibration, and measurement-noise simulation
Microwave clutter-removal and 3D reconstruction models
Lesion localization, segmentation, and change-detection algorithms
Phantom-validation protocols and system-performance evaluation

MicroScan is currently a research-stage engineering prototype. It is not a certified medical device and must not be used for clinical diagnosis or treatment decisions.
