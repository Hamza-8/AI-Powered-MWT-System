"""Held-out physical reconstruction and label-aware lesion evaluation."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .metrics import property_metrics, lesion_components


def segmentation_metrics(probabilities, labels, validity, schema):
    if schema == "multiclass":
        if labels.ndim == probabilities.ndim:
            labels = labels[:, 0]
        valid = validity[:, 0] if validity.ndim == probabilities.ndim else validity
        valid = valid.bool() & (labels >= 0)
        prediction = probabilities.argmax(1)
        classes = range(1, probabilities.shape[1])
    else:
        if labels.ndim == probabilities.ndim-1:
            labels = labels[:, None]
        valid = torch.broadcast_to(validity.bool(), labels.shape) & (labels >= 0)
        prediction = probabilities >= .5
        classes = range(probabilities.shape[1])
    rows = {}
    for c in classes:
        if schema == "multiclass":
            v, p, t = valid, prediction == c, labels == c
        else:
            v, p, t = valid[:, c], prediction[:, c], labels[:, c] >= .5
        p, t = p & v, t & v
        tp, fp, fn = (p & t).sum().item(), (p & ~t).sum().item(), (~p & t).sum().item()
        denom = 2*tp+fp+fn
        rows[str(c)] = {"dice": (2*tp/denom if denom else 1.) if v.any() else None,
                        "true_voxels": int(t.sum()), "predicted_voxels": int(p.sum()),
                        "false_positive_voxels": int(fp), "valid_voxels": int(v.sum()),
                        "absent_in_truth": not bool(t.any())}
    return rows


def evaluate_checkpoint(checkpoint, manifest, *, split="validation", device="cpu", phi_config=None):
    from .data import ManifestDataset, collate_samples
    from .training import load_deployment, move_batch, prepare_batch, build_deployment_pipeline, execute_pipeline
    system, config, metadata, phi = load_deployment(checkpoint, device, phi_config=phi_config)
    if split == "test":
        raise ValueError("Use the explicit post-selection sealed-test protocol; evaluate never unseals it")
    dataset = ManifestDataset(manifest, split=split)
    rows = []
    with torch.no_grad():
        for sample in dataset:
            batch = prepare_batch(move_batch(collate_samples([sample]), device), config.get("preprocessing"), config)
            pipeline = build_deployment_pipeline(system, config, phi, batch)
            outputs = execute_pipeline(pipeline, batch, config)
            valid = bool(outputs["absolute_valid"].all())
            row = {"sample_id": sample["metadata"]["id"], "absolute_valid": valid,
                   "observed_channels": int(batch["mask"].sum()), "request_full_scan": outputs.get("request_full_scan", False)}
            if "target" in batch and valid:
                primary = outputs.get("absolute", outputs["refined"])
                row["refined"] = property_metrics(primary, batch["target"], batch.get("object_mask"), config.get("property_scales", (44., .8)))
                coarse = outputs["coarse"]
                if config.get("mode") in ("monitoring", "baseline_tracking"):
                    coarse = coarse+batch["baseline"]
                row["coarse"] = property_metrics(coarse, batch["target"], batch.get("object_mask"), config.get("property_scales", (44., .8)))
            probs = outputs.get("segmentation_probabilities")
            if probs is not None and "labels" in batch and valid:
                validity = batch["label_validity"].bool()
                assessed = outputs.get("segmentation_assessed")
                if assessed is not None:
                    validity = validity & (assessed[:, 0] if validity.ndim == 4 else assessed)
                row["lesion_metrics"] = segmentation_metrics(probs, batch["labels"], validity, config.get("segmentation", {}).get("label_mode", "multiclass"))
            rows.append(row)
    if not rows:
        raise ValueError("No held-out samples to evaluate")
    def aggregate(key):
        available = [r[key] for r in rows if key in r]
        if not available:
            return None
        return {name: sum(r[name] for r in available if r[name] is not None)/sum(r[name] is not None for r in available)
                if any(r[name] is not None for r in available) else None for name in available[0]}
    return {"split": split, "samples": len(rows), "per_sample": rows, "refined_mean": aggregate("refined"),
            "coarse_mean": aggregate("coarse"), "physics_training": metadata.get("physics_active"),
            "physics_inference": "C" if config.get("physics_mode") == "C" else "A_graph",
            "probability_calibration_validated": False, "test_accessed": False}
