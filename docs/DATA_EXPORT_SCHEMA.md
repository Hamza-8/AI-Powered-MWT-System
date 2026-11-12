# Calibrated measurement and property export contract

## Stored arrays and axes

Canonical network measurements are `[B,2,F,Rx,Tx]` (real, imaginary). Store original responses as **complex128 `[F,Rx,Tx]`**; convert supplied pilot `data.S[Rx,Tx,F,case]` with `S[:,:,:,case].transpose(2,0,1)`. Subtract compatible references in complex128 before float32 channel conversion. Physical ports already run 1–18. The COMSOL identifier mapping `[7,8,9,10,11,1,2,3,4,6,12,13,14,15,16,17,18,19]` is metadata, not another permutation. Reflection channels remain present.

Separate these three supports:

| Array | Shape without batch | Meaning |
|---|---|---|
| Observation mask | `[1,F,Rx,Tx]` in network; stored `[F,Rx,Tx]` | Independently observed entries. A measured zero is valid. Difference support is the intersection of operands. |
| Prior spatial ROI | `[1,Z,Y,X]` | Clinician/previous-frame assessment request; never current labels. |
| Labels and validity | `[Z,Y,X]` or explicit multilabel channels | Supervision only. Missing clinical labels are absent, not background. |

Properties are float `[2,Z,Y,X]`: relative permittivity and effective conductivity S/m at the recorded reference frequency (default 0.85 GHz). Coordinates are XYZ in mm; array indexing is ZYX. Every grid records physical bounds, shape and axis convention. Cell centres are `min+(i+0.5)*(max-min)/N`. A main whole-head dataset must declare verified anatomy bounds. Its extent is not inferred from the sphere ROI.

Domains must be recorded explicitly: `raw_s`, `calibrated_delta_s`, `decluttered_target`, `current_minus_baseline`. Never compare a raw-domain Phi prediction with a decluttered observation. Source and calibration context must establish domain equivalence before physics can activate.

## Manifest version 1

`microscan.data.ManifestDataset` accepts one JSON manifest plus per-sample NPY/NPZ/MAT arrays. NPY permits memory mapping; NPZ contains exactly one named numeric array. A MAT array file exposes `array`. Object/pickle arrays are prohibited. The MATLAB-v7.3 reader reverses MATLAB's HDF5 numeric axis storage and explicitly decodes references and compound complex arrays. Run the format tests when exporting a new writer variation.

Required per-sample metadata:

```json
{
  "schema_version": 1,
  "samples": [{
    "id": "head001_frame003",
    "split": "train",
    "subject_id": "subject001",
    "head_id": "head001",
    "geometry_id": "head001_poseA",
    "sequence_id": "subject001_sequenceA",
    "source_domain": "COMSOL_or_measured_phantom",
    "domain": "calibrated_delta_s",
    "mode": "snapshot",
    "signal_path": "head001_frame003_signal.npy",
    "reference_path": "system_reference.npy",
    "mask_path": "head001_frame003_mask.npy",
    "reference_mask_path": "system_reference_mask.npy",
    "target_path": "head001_frame003_properties.npy",
    "grid": {
      "shape_zyx": [128,128,128],
      "bounds_xyz_mm": [[-110,110],[-130,130],[-100,140]],
      "array_axes": "ZYX",
      "coordinate_axes": "XYZ",
      "units": "mm"
    },
    "frequencies_ghz": [0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05],
    "physical_ports": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
    "reference_frequency_ghz": 0.85
  }]
}
```

The illustrative head bounds above are **not verified anatomy**; replace them with the export's measured bounds. The grid is an input to export validation, not a requested new COMSOL solve. Current target volume, lesion coordinates and material IDs are labels and must not be passed as model conditioning.

Optional sample fields include `target_validity_path`, `object_mask_path`, `physics_context`, `physics_context_paths`, `context_validity`, `stable_mask_path`, `noise_scale_path`, `noise_covariance_path`, `roi_channel_mask_path`, `baseline_spectral_path`, `pair_signal_path`, `pair_clean_signal_path`, `spectral_target_path`, `roi_path`, `baseline_target_path`, `baseline_signal_path`, `baseline_mask_path`, `registration.measurement_compatible`, `clean_signal_path`, `repeated_signal_path`, `label_path`, `label_validity_path`, `label_schema`, `material_ids_path` and `spectral_table_path`. `physics_context_paths` maps context names to numeric arrays (antenna positions, source excitations/orientations, geometry descriptors); inline context stores compatible IDs/conventions. These are acquisition metadata, not current measurement targets. A `roi_path` requires a declared `roi_source` of `clinician`, `prior_baseline_prediction` or `previous_frame_prediction`; current supervised labels are not an accepted focus source. `baseline_spectral_path` supplies `[2,F,Z,Y,X]`. The spectral table is `[material,F,2]` and material IDs index it to build spectral property supervision. It must derive from the actual material model; the network does not infer unknown Cole–Cole coefficients from two reference maps. Record calibration state, layout ID, source excitations/orientations, pose, source domain, acquisition day, clean-pair identity and lesion configuration in metadata.

All subject/head/geometry/sequence identities must stay in one split. Derived hardware-noise, focus-mask, paired-reference and baseline variants inherit that split. Repeated noise acquisitions require a declared conditional-noise model; shared drift does not establish Noise2Noise eligibility. Do not spatially augment targets without physically corresponding measurements. Segmentation-only spatial augmentation must transform every volume/context/label together.

## Clinical schema

Default mutually exclusive classes: `0 healthy_background`, `1 ischemia`, `2 hemorrhage`, `3 tumor`, `4 edema`. Use binary Dice+BCE when labels only distinguish lesion/non-lesion; multiclass Dice+CE when voxel classes are exclusive; independent sigmoid heads only for genuinely overlapping labelled voxels. Multi-lesion volumes alone do not imply overlapping voxels. Missing labels use validity masks and do not become healthy cases.

## Supplied pilot

The MAT contains complex128 `[18,18,10,5]`, ten frequencies 0.60–1.05 GHz, completion metadata and four sphere geometries. Reference case 0 is not a training example. Train cases 1/4, validation case 2, test case 3. Test access remains sealed by protocol until explicit selection. The conversion CLI exports train/validation only by default; it does not materialize case3 targets. The frequency runner opens test case3 only through its explicit post-selection test command.

Target shape is 128³ over [-60,60] mm. Centres are `-60+0.9375/2+arange(128)*0.9375`. Spheres have epsilon_r=45, sigma=0.8 S/m; outside is 1 and 0. Cases 1–4 are r15 origin, r15 at XYZ(25,-20,15), r25 origin, r25 translated. These are object masks, not clinical diagnoses. The normalized two property maps coincide, so this pilot cannot validate independent two-property recovery. Four geometries are not hundreds of independent examples.

## Structured training acquisition and repeated observations

`acquisition_augmentation` supports `tx_keep`, `rx_keep`, selected `frequency_indices`, failed-channel probability, declared corruption models and imperfect prior-focus displacement/radius scaling. Geometry-grouped split identity is unchanged. Structured Tx/Rx holdout supervises only truly observed selected-frequency entries, not hidden values.

Repeated-acquisition eligibility requires explicit verified `same_clean_state`, `conditionally_unbiased_target`, `independent_given_clean_state`, `shared_drift=false` and distinct acquisition IDs. This is a declared data contract, not validation of a particular SDR noise model. Main inference manifests may use split `inference` without property targets or clinical labels.

C monitoring can load `baseline_spectral_context_path` as `[2,F,Z,Y,X]` only
with `baseline_spectral_provenance=registered_prior_prediction` or
`verified_known_phantom`. This inference context is separate from
`baseline_spectral_path`, which supplies supervised baseline spectral labels.
Current material/lesion labels never populate inference conditioning.
