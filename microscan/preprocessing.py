"""Explicit offline calibration-cohort fitting and physical scan export commands."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .data import ManifestDataset, channels_to_complex
from .subspace import MaskedSubspace
from .acquisition import export_scan


def fit_subspace(manifest, context, output, *, rank=1, strength=.25):
    dataset = ManifestDataset(manifest, split="train")
    if len(dataset) < 2:
        raise ValueError("A subspace needs multiple independent calibration geometries")
    groups, signals, masks = set(), [], []
    for sample in dataset:
        md = sample["metadata"]
        if not md.get("subspace_cohort_eligible", False):
            raise ValueError("Every cohort case must explicitly declare healthy/calibration eligibility")
        group = md["geometry_id"]
        if group in groups:
            raise ValueError("Noise repeats are not independent cohort geometries")
        if list(md["frequencies_ghz"]) != list(context["frequencies_ghz"]):
            raise ValueError("Frequency layout differs from cohort binding")
        groups.add(group)
        signals.append(channels_to_complex(sample["signals"]))
        masks.append(sample["mask"][0].numpy())
    fitted = MaskedSubspace(rank, strength).fit(np.stack(signals), np.stack(masks), context)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fitted.save(output)
    report = {"path": str(output), "training_geometries": sorted(groups), "rank": rank,
              "strength": strength, "context": context, "scientifically_validated": False,
              "evidence": "calibration-cohort fit; lesion-preservation validation remains necessary"}
    Path(str(output)+".json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-subspace")
    fit.add_argument("--manifest", required=True); fit.add_argument("--context", required=True)
    fit.add_argument("--output", required=True); fit.add_argument("--rank", type=int, default=1)
    fit.add_argument("--strength", type=float, default=.25)
    scan = sub.add_parser("export-scan")
    scan.add_argument("--mask", required=True, help="NPY [F,Rx,Tx] observed acquisition mask")
    scan.add_argument("--frequencies", required=True, help="JSON list of GHz values")
    scan.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if args.command == "fit-subspace":
        result = fit_subspace(args.manifest, json.loads(Path(args.context).read_text()), args.output,
                              rank=args.rank, strength=args.strength)
    else:
        result = export_scan(args.output, np.load(args.mask, allow_pickle=False), json.loads(Path(args.frequencies).read_text()))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
