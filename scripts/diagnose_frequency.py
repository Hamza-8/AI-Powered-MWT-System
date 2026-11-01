"""Inference-only pilot fit/input-sensitivity checks using existing saved weights.

No optimizer is created, no checkpoint is written, and sealed case3 is never
indexed. Targets are used only AFTER inference for diagnostic metrics. Place
this script in scripts/; do not edit microscan/ or the frequency configuration.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from microscan.data import PilotDataset
from microscan.frequency import (canonical_hash, file_sha256, reconstruction_metrics,
                                 source_identity, transform_signal)
from microscan.models import MSRNN


def output_difference(a, b):
    scale = a.new_tensor([44., .8]).reshape(2, 1, 1, 1)
    difference = ((a-b)/scale).double()
    return {"normalized_rms": difference.square().mean().sqrt().item(),
            "epsilon_rms": (a[0]-b[0]).double().square().mean().sqrt().item(),
            "sigma_rms_s_per_m": (a[1]-b[1]).double().square().mean().sqrt().item()}


def describe_map(volume, half_span_mm=60.):
    contrast = (volume-volume.new_tensor([1., 0.]).reshape(2, 1, 1, 1))/volume.new_tensor([44., .8]).reshape(2, 1, 1, 1)
    selected = contrast.mean(0) >= .5
    points = selected.nonzero().double()
    spacing = volume.new_tensor([2*half_span_mm/n for n in volume.shape[-3:]]).double()
    centre = ((points+.5)*spacing-half_span_mm).mean(0).flip(0).tolist() if len(points) else None
    return {"threshold": "mean normalized two-property contrast >=0.5",
            "centre_xyz_mm": centre, "volume_ml": len(points)*spacing.prod().item()/1000.,
            "scope": "sphere-object diagnostic, not clinical segmentation"}


def diagnose(checkpoint, mat, output, *, device="cpu", save_volumes=False):
    checkpoint, mat, output = Path(checkpoint), Path(mat), Path(output)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint_hash = file_sha256(checkpoint)
    # Only open trusted checkpoints produced by your own experiment.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = saved["identity"]
    if canonical_hash(identity) != saved["identity_sha256"]:
        raise ValueError("Checkpoint experiment identity is inconsistent")
    if saved.get("physics_requested") != "A" or saved.get("physics_active") != "none":
        raise ValueError("This diagnostic is for the image-only frequency pilot")
    if identity["source"] != source_identity():
        raise ValueError("Installed microscan source differs from the experiment; use its original source")
    if identity["data_sha256"] != file_sha256(mat):
        raise ValueError("Pilot MAT differs from the checkpoint's experiment")
    config, preprocessing = identity["config"], saved["preprocessing"]
    indices = identity["indices_zero_based"]
    samples = []
    for split in ("train", "validation"):
        samples.extend(PilotDataset(mat, split=split, frequency_indices=indices,
                                    grid_size=config.get("grid_size", 128)))
    case_ids = [sample["metadata"]["case_id"] for sample in samples]
    if case_ids != [1, 4, 2]:
        raise ValueError("Only training cases1/4 and validation case2 are allowed")
    if not all(torch.equal(samples[0]["mask"], sample["mask"]) for sample in samples):
        raise ValueError("The reused swap/zero controls require identical acquisition masks")
    model = MSRNN(saved.get("architecture", config.get("model", {}))).to(target_device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    # Do not load optimizer/scaler state. They play no part in inference.
    predictions = {}
    forward_count = 0
    with torch.inference_mode():
        for sample in samples:
            cid = sample["metadata"]["case_id"]
            mask = sample["mask"].unsqueeze(0).to(target_device)
            signal = transform_signal(sample["signals"].unsqueeze(0).to(target_device), mask, preprocessing)
            result = model(signal, mask, sample["frequencies_ghz"].to(target_device), physics_mode="A")
            predictions[cid] = {key: result[key][0].float().cpu() for key in ("coarse", "refined")}
            forward_count += 1
            print(f"Reconstructed case{cid} ({sample['metadata']['split']}); no weight updates", flush=True)
        sample = samples[0]
        mask = sample["mask"].unsqueeze(0).to(target_device)
        result = model(torch.zeros_like(sample["signals"]).unsqueeze(0).to(target_device), mask,
                       sample["frequencies_ghz"].to(target_device), physics_mode="A")
        zero_prediction = {key: result[key][0].float().cpu() for key in ("coarse", "refined")}
        forward_count += 1
    report = {"scope": "inference-only diagnostic; no retraining or frequency reselection",
              "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
              "k": identity["k"], "seed": identity["seed"], "device": str(target_device),
              "training_progress": {key: saved.get("progress", {}).get(key) for key in ("epoch", "updates")},
              "weight_snapshot": "last_saved_weights" if "progress" in saved else "selected_weights",
              "inference_precision": "float32, AMP disabled to match recorded validation",
              "optimizer_steps": 0, "forward_passes": forward_count, "test_accessed": False,
              "input_policy": "original saved scale, subset and masks; no fit on validation",
              "control_policy": "Zero values retain observed support; zero is an out-of-distribution sensitivity probe. Swapped inputs reuse donor predictions because targets never enter the model.",
              "cases": [], "pairwise_output_differences": []}
    output.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        cid = sample["metadata"]["case_id"]
        actual, target = predictions[cid], sample["target"]
        donor = {1: 4, 4: 2, 2: 1}[cid]
        row = {"case_id": cid, "split": sample["metadata"]["split"], "target_geometry": sample["metadata"]["case"],
               "normal": {}, "zero_measurements": {}, "swapped_measurements": {"donor_case_id": donor},
               "zero_vs_normal": {}, "swapped_vs_normal": {}}
        for key in ("coarse", "refined"):
            row["normal"][key] = {"errors": reconstruction_metrics(actual[key], target), "object": describe_map(actual[key])}
            row["zero_measurements"][key] = {"errors": reconstruction_metrics(zero_prediction[key], target), "object": describe_map(zero_prediction[key])}
            row["swapped_measurements"][key] = {"errors": reconstruction_metrics(predictions[donor][key], target), "object": describe_map(predictions[donor][key])}
            row["zero_vs_normal"][key] = output_difference(zero_prediction[key], actual[key])
            row["swapped_vs_normal"][key] = output_difference(predictions[donor][key], actual[key])
        report["cases"].append(row)
        if save_volumes:
            np.savez_compressed(output/f"case_{cid}.npz", target=target.numpy(), **{key: value.numpy() for key, value in actual.items()})
    for i, left in enumerate(case_ids):
        for right in case_ids[i+1:]:
            report["pairwise_output_differences"].append({"case_ids": [left, right],
                **{key: output_difference(predictions[left][key], predictions[right][key]) for key in ("coarse", "refined")}})
    if file_sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed while being read; stop training before this diagnostic")
    report["checkpoint_unchanged"] = True
    path = output/"diagnostic_report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Saved {path}; {forward_count} forward passes, zero optimizer steps", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="outputs/frequency/K10/seed_17/resume.pt")
    p.add_argument("--mat", default="data/pilot/frequency_pilot.mat")
    p.add_argument("--output", default="outputs/inference_diagnostics/K10_seed17_final")
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-volumes", action="store_true")
    args = p.parse_args()
    diagnose(args.checkpoint, args.mat, args.output, device=args.device, save_volumes=args.save_volumes)


if __name__ == "__main__":
    main()
