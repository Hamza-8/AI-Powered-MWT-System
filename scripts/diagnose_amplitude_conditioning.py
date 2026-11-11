"""Frozen-weight amplitude ablations on train/validation only. Never opens case3.

Branch knockouts are distribution-shifting interventions, not retrained models.
Positive gain tests change magnitude while preserving pattern, not physical size.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import torch

from microscan import frequency as f
from pattern_amplitude_model import PatternAmplitudeMSRNN
from train_expanded_pilot import ExpandedPilot
from train_pattern_pilot import atomic_write, without_score


CASES = (5, 6, 7, 8)
CONDITIONS = ("intact", "raw_off", "rms_off", "both_off", "coarse_off", "enhancement_off", "matched_gain")


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if f.canonical_hash(state["identity"]) != state["identity_sha256"]:
        raise ValueError("Checkpoint identity hash is invalid")
    identity = state["identity"]
    expected = {"reference": [0], "train": [1,2,4,6,7], "validation": [5,8], "sealed_test": [3]}
    if (identity["splits"] != expected or identity["k"] != 10 or identity["seed"] != 17
            or state.get("epoch") != 300 or state.get("optimizer_updates") != 1500):
        raise ValueError("Use the completed native K10 seed17 crossed-v3 checkpoint")
    if (identity["config"]["grid_size"] != 128 or state["architecture"] != identity["config"]["model"]
            or identity["indices_zero_based"] != list(range(10))
            or state.get("checkpoint_policy") != "fixed_final_epoch" or identity.get("test_accessed") is not False):
        raise ValueError("Expected native128 final-epoch architecture and validation-only lineage")
    if identity["source"] != f.source_identity():
        raise ValueError("Legacy package source does not match checkpoint")
    for filename, digest in identity["config"]["experimental_source"].items():
        if f.file_sha256(Path(__file__).parent / filename) != digest:
            raise ValueError(f"Experimental source mismatch: {filename}")
    cls = PatternAmplitudeMSRNN
    if state["architecture"].get("postnorm_amplitude"):
        from postnorm_amplitude_model import PostNormAmplitudeMSRNN
        cls = PostNormAmplitudeMSRNN
    model = cls(state["architecture"]).to(device).eval()
    if model.coarse.output_size != 128 or sum(p.numel() for p in model.parameters()) != state["parameter_count"]:
        raise ValueError("Checkpoint parameter count or native output grid is inconsistent")
    model.load_state_dict(state["model"], strict=True)
    return state, model


def load_samples(state, mat):
    if f.file_sha256(mat) != state["identity"]["data_sha256"]:
        raise ValueError("MAT checksum differs from checkpoint")
    samples = {}
    for split in ("train", "validation"):
        for sample in ExpandedPilot(mat, split, range(10), state["identity"]["config"]):
            samples[sample["metadata"]["case_id"]] = sample
    if set(samples) != {1,2,4,5,6,7,8}:
        raise ValueError("Unexpected train/validation case inventory")
    return samples


@contextlib.contextmanager
def ablate(model, condition):
    """Remove branch outputs, restoring every hook even if forward fails."""
    hooks = []
    conditioners = {"coarse": model.coarse.conditioner,
                    "enhancement": model.enhancement.measurement_encoder.conditioner}
    def zero(module, inputs, output):
        return torch.zeros_like(output)
    def bypass(module, inputs, output):
        return inputs[0]
    try:
        for name, conditioner in conditioners.items():
            if condition == "raw_off":
                hooks.append(conditioner.raw_encoder.register_forward_hook(zero))
            elif condition == "rms_off":
                hooks.append(conditioner.amplitude.register_forward_hook(zero))
            elif condition == "both_off" or condition == name + "_off":
                hooks.append(conditioner.register_forward_hook(bypass))
        yield
    finally:
        for hook in hooks:
            hook.remove()


def predict(model, sample, preprocessing, device="cpu", gain=1.):
    altered = {**sample, "signals": sample["signals"] * gain}
    with torch.inference_mode():
        outputs = f._predict(model, altered, preprocessing, device)
    return {name: outputs[name][0].detach().float().cpu() for name in ("coarse", "refined")}


def describe(prediction, sample):
    target = sample["target"]
    metrics = without_score(f.reconstruction_metrics(prediction, target))
    contrast = f.normalized_contrast(prediction).mean(0)
    metrics["soft_volume_ml"] = float(contrast.clamp(0, 1).sum()) * (120 / target.shape[-1])**3 / 1000
    metrics["equivalent_radius_mm"] = float((3 * metrics["predicted_volume_mm3"] / (4 * torch.pi))**(1/3))
    truth = sample["object_mask"].bool().expand_as(target)
    metrics["property_regions"] = {}
    for c, name in enumerate(("epsilon", "sigma")):
        metrics["property_regions"][name] = {}
        for region_name, region in (("foreground", truth[c]), ("background", ~truth[c])):
            error = prediction[c][region] - target[c][region]
            metrics["property_regions"][name][region_name] = {
                "mean_prediction": float(prediction[c][region].mean()),
                "rmse": float(error.square().mean().sqrt())}
    return metrics


def run_ablation(checkpoint, mat, output, device="cpu", conditions=CONDITIONS):
    state, model = load_checkpoint(checkpoint, device)
    if state["architecture"].get("postnorm_amplitude"):
        raise ValueError("Ablate the existing pattern-v1 baseline before choosing a new model")
    samples = load_samples(state, mat)
    if any(condition not in CONDITIONS for condition in conditions) or "intact" not in conditions:
        raise ValueError("Ablation conditions require intact reference")
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    norms = {cid:float(torch.linalg.vector_norm(s["signals"].double())) for cid,s in samples.items()}
    mates = {5:6, 6:5, 7:8, 8:7}
    report = {"checkpoint_sha256": f.file_sha256(checkpoint), "identity_sha256": state["identity_sha256"],
              "test_accessed": False, "optimizer_updates": 0, "conditions": {}, "real_pairs": {}}
    for small, large in ((5,6), (7,8)):
        a,b = (samples[cid]["signals"].double().flatten() for cid in (small,large))
        report["real_pairs"][f"{small}_{large}"] = {
            "small_case":small, "large_case":large,
            "large_to_small_norm_ratio":norms[large]/norms[small],
            "normalized_realimag_inner_product":float(torch.dot(a,b)/(a.norm()*b.norm()))}
    baseline = {}
    for condition in conditions:
        rows = []
        for cid in CASES:
            gain = norms[mates[cid]] / norms[cid] if condition == "matched_gain" else 1.
            with ablate(model, condition):
                maps = predict(model, samples[cid], state["preprocessing"], device, gain)
            if condition == "intact":
                baseline[cid] = maps
            elif cid not in baseline:
                raise ValueError("Intact condition must run first")
            row = {"case_id":cid, "split":samples[cid]["metadata"]["split"], "gain":gain,
                   "geometry":samples[cid]["metadata"]["geometry"],
                   **{name:describe(v,samples[cid]) for name,v in maps.items()},
                   "map_change_from_intact":{name:{prop:float((v[c]-baseline[cid][name][c]).square().mean().sqrt())
                       for c,prop in enumerate(("epsilon","sigma"))} for name,v in maps.items()}}
            rows.append(row)
            print(f"{condition} case{cid}: centre={row['refined']['localization_error_mm']:.3f}mm "
                  f"volume={row['refined']['predicted_volume_mm3']/1000:.3f}ml "
                  f"epsilon={row['refined']['epsilon_rmse']:.4f}", flush=True)
        report["conditions"][condition] = rows
        atomic_write(output, report)
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items())
    report["weights_unchanged"] = True
    atomic_write(output, report)
    return report


def compare_checkpoints(baseline_path, candidate_path, mat, output, device="cpu"):
    """Native final-epoch comparison; refuses incomplete or altered-budget runs."""
    base_state, base_model = load_checkpoint(baseline_path, device)
    new_state, new_model = load_checkpoint(candidate_path, device)
    for key in ("data_sha256", "csv_sha256", "splits", "preprocessing", "k", "seed"):
        if base_state["identity"][key] != new_state["identity"][key]:
            raise ValueError(f"Comparison lineage differs: {key}")
    for key in ("grid_size", "epochs", "batch_size", "learning_rate", "weight_decay", "gradient_weight", "gradient_clip", "amp"):
        if base_state["identity"]["config"][key] != new_state["identity"]["config"][key]:
            raise ValueError(f"Training budget/objective differs: {key}")
    samples = load_samples(base_state, mat)
    report = {"baseline_sha256":f.file_sha256(baseline_path), "candidate_sha256":f.file_sha256(candidate_path),
              "baseline_execution":base_state["identity"]["execution"],
              "candidate_execution":new_state["identity"]["execution"],
              "inference_device":str(device), "test_accessed":False, "per_case":[]}
    for cid in CASES:
        row = {"case_id":cid, "split":samples[cid]["metadata"]["split"],
               "geometry":samples[cid]["metadata"]["geometry"]}
        for name,model,state in (("baseline",base_model,base_state),("candidate",new_model,new_state)):
            maps = predict(model,samples[cid],state["preprocessing"],device)
            row[name] = {key:describe(v,samples[cid]) for key,v in maps.items()}
        report["per_case"].append(row)
        atomic_write(output,report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("ablate","compare"))
    parser.add_argument("--checkpoint",default="outputs/frequency_pattern_v1/K10/seed_17/final.pt")
    parser.add_argument("--candidate")
    parser.add_argument("--mat",default="data/pilot_expanded/frequency_pilot.mat")
    parser.add_argument("--output",default="outputs/amplitude_diagnostic_v1/report.json")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args(argv)
    if args.command=="compare":
        if args.candidate is None: parser.error("compare requires --candidate")
        result=compare_checkpoints(args.checkpoint,args.candidate,args.mat,args.output,args.device)
    else:
        result=run_ablation(args.checkpoint,args.mat,args.output,args.device)
    print(f"Saved {args.output}; sealed test accessed: {result['test_accessed']}")
    return result


if __name__=="__main__":
    main()
