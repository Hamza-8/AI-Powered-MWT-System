"""Isolated K1/K2 extension of the original pattern-v1 frequency experiment.

Default: K1=1.05GHz, seed17, original v3 split and 1500-update budget.
Optional K2=.60/1.05GHz. Existing sources, CSV and outputs remain unchanged.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path

import torch

from microscan import frequency as f
from train_pattern_pilot import Experiment as PatternExperiment, atomic_write


CANDIDATES = {1: [9], 2: [0, 9]}


@contextlib.contextmanager
def lowcount_subsets():
    """Bind explicit new candidates in this process; restore the original reader."""
    original = f.read_subsets
    def read(path):
        subsets = original(path)
        return {**subsets, **copy.deepcopy(CANDIDATES)}
    f.read_subsets = read
    try:
        yield
    finally:
        f.read_subsets = original


class Experiment(PatternExperiment):
    def __init__(self, config, mat, csv, output, device):
        if config.get("extent_loss") or config.get("targeted_adjustment") or config.get("model", {}).get("postnorm_amplitude"):
            raise ValueError("Use the original pattern-v1 model and weighted objective")
        super().__init__(config, mat, csv, output, device)
        self.config["experimental_source"][Path(__file__).name] = f.file_sha256(__file__)
        self.config["lowcount_protocol"] = {
            "name": "pattern_v1_K1_K2_v1", "indices_zero_based": {str(k): v for k, v in CANDIDATES.items()},
            "frequencies_ghz": {str(k): [f.GRID_GHZ[i] for i in slots] for k, slots in CANDIDATES.items()},
            "subset_source": "Explicit candidates in this runner; original K3..10 CSV unchanged",
            "benchmark": "Original v3 known-position, held-out-size combinations",
            "selection_note": "K1 selected by training-only scalar-size screening; neural performance is unproven"}

    def prepare(self):
        path = self.output / "protocol.json"
        if path.exists() and json.loads(path.read_text())["config"] != self.config:
            raise ValueError("Output protocol differs; use a new output folder")
        if not path.exists() and any(self.output.glob("K*/seed_*/*")):
            raise FileExistsError("Existing run artifacts lack this protocol; use a new output folder")
        report = super().prepare()
        report["candidate_frequencies_ghz"] = self.config["lowcount_protocol"]["frequencies_ghz"]
        atomic_write(path, report)
        return report

    def evaluate(self, model, samples, preprocessing, device):
        result = super().evaluate(model, samples, preprocessing, device)
        print("  " + "; ".join(
            f"case{r['case_id']} centre={r['refined']['localization_error_mm']:.3f}mm "
            f"volume={r['refined']['predicted_volume_mm3']/1000:.2f}/{r['refined']['true_volume_mm3']/1000:.2f}ml "
            f"epsilon={r['refined']['epsilon_rmse']:.4f} sigma={r['refined']['sigma_rmse_s_per_m']:.5f}"
            for r in result["per_case"]), flush=True)
        return result

    def run(self, counts=(1,), *, seeds=None, resume=False, max_updates=None):
        counts = list(counts)
        if not counts or len(set(counts)) != len(counts) or any(k not in CANDIDATES for k in counts):
            raise ValueError("Use distinct counts from 1,2 only")
        seeds = [17] if seeds is None else list(seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(s not in self.config["seeds"] for s in seeds):
            raise ValueError("Use distinct configured seeds")
        if max_updates is not None and max_updates < 1:
            raise ValueError("max-updates must be positive")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable here; use the RTX machine")
        self.prepare()
        results = []
        with lowcount_subsets():
            for k in counts:
                for seed in seeds:
                    self.current, self.evaluations = (k, seed), 0
                    results.append(self.run_candidate(k, seed, resume=resume, max_updates=max_updates))
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--config", default="configs/frequency_pilot_pattern_v1.json")
    parser.add_argument("--mat", default="data/pilot_expanded/frequency_pilot.mat")
    parser.add_argument("--csv", default="data/pilot/frequency_subsets.csv")
    parser.add_argument("--output", default="outputs/frequency_pattern_lowcount_v1")
    parser.add_argument("--counts", default="1", help="1 first; 2 for the endpoint pair; or 1,2")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, help="Bounded verification, not a completed experiment")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    experiment = Experiment(config, args.mat, args.csv, args.output, args.device)
    print("Original pattern-v1 model/loss/split; K1=1.05GHz; K2=.60/1.05GHz; new output folder", flush=True)
    result = experiment.prepare() if args.command == "prepare" else experiment.run(
        [int(k) for k in args.counts.split(',')], seeds=[int(s) for s in args.seeds.split(',')],
        resume=args.resume, max_updates=args.max_updates)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
