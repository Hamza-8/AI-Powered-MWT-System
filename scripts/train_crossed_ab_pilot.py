"""Split-only K10/seed17 pilot: both radii at A/B, both radii at held-out C.

Train from scratch for 1,500 updates. Preserve the pattern-v1 model, weighted
objective, and original cosine schedule (one scheduler step per five updates).
Four training cases give 375 complete shuffled passes. Case1 is unused and
case3 stays sealed. Existing runners and their checkpoint identities are intact.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from microscan import frequency as f
from pattern_amplitude_model import PatternAmplitudeMSRNN
from train_expanded_pilot import ExpandedPilot
from train_pattern_pilot import Experiment as PatternExperiment, atomic_write, without_score


SPLITS = {"reference": [0], "train": [2, 4, 5, 6], "validation": [7, 8],
          "sealed_test": [3], "unused": [1]}
BUDGET = {"optimizer_updates": 1500, "updates_per_evaluation": 5,
          "updates_per_scheduler_step": 5, "scheduler_t_max": 300,
          "sampling": "shuffled_passes_without_replacement"}
EXPERIMENT = "fully_crossed_ab_holdout_c_v1"


def build_config(base):
    """Change allocation/budget bookkeeping only; retain the original objective."""
    config = copy.deepcopy(base)
    if (config.get("model") != {"measurement_representation": "pattern_amplitude_v1",
                                "pattern_rms_floor": 1e-8}
            or config.get("epochs") != 300 or config.get("batch_size") != 1
            or config.get("checkpoint_policy") != "fixed_final_epoch"
            or config.get("extent_loss") or config.get("targeted_adjustment")):
        raise ValueError("Use the unchanged configs/frequency_pilot_pattern_v1.json as the base")
    config.update(protocol_version=4, seeds=[17], epochs=375, patience=375,
                  checkpoint_policy="fixed_final_update", split_experiment=EXPERIMENT,
                  training_budget=copy.deepcopy(BUDGET))
    old = config["expanded_protocol"]
    config["expanded_protocol"] = {
        "name": EXPERIMENT, "data_sha256": old["data_sha256"],
        "original_prefix_sha256": old["original_prefix_sha256"],
        "cases": copy.deepcopy(old["cases"]), "splits": copy.deepcopy(SPLITS),
        "validation_geometry_ids": [7, 8],
        "validation_kind": "Both radii at unseen location C; development validation",
        "note": "Every training location A/B has both radii. Case1 unused; case3 sealed. "
                "These cases were inspected in earlier experiments, so validation is not a fresh test."}
    return config


def validate_config(config):
    f.validate_config(config)
    if (config.get("protocol_version") != 4 or config.get("split_experiment") != EXPERIMENT
            or config["expanded_protocol"]["splits"] != SPLITS
            or config.get("seeds") != [17] or config.get("checkpoint_policy") != "fixed_final_update"
            or config.get("representation_experiment") != "pattern_amplitude_v1"
            or config.get("extent_loss") or config.get("targeted_adjustment")
            or config["model"].get("postnorm_amplitude")):
        raise ValueError("Use the explicit split-only K10 seed17 allocation and original model/loss")
    cases = config["expanded_protocol"]["cases"]
    if len(cases) != 9 or [c["id"] for c in cases] != list(range(9)):
        raise ValueError("All nine ordered metadata records are required")
    groups = {}
    for cid in SPLITS["train"]:
        case = cases[cid]
        groups.setdefault(tuple(case["centre_xyz_mm"]), set()).add(case["radius_mm"])
    if len(groups) != 2 or any(radii != {15., 25.} for radii in groups.values()):
        raise ValueError("Every training location must have both radii")
    if any(tuple(cases[cid]["centre_xyz_mm"]) in groups for cid in SPLITS["validation"]):
        raise ValueError("Validation location C must be absent from training")
    budget = config["training_budget"]
    total, interval = budget["optimizer_updates"], budget["updates_per_evaluation"]
    if (not isinstance(total, int) or total < 1 or not isinstance(interval, int) or interval < 1
            or total % 4 or total % interval or total != config["epochs"] * 4
            or budget["updates_per_scheduler_step"] != interval
            or budget["scheduler_t_max"] != total // interval):
        raise ValueError("Require whole four-case passes and matched cosine/evaluation blocks")


def next_sample_index(progress, count):
    """Checkpointable shuffle stream, independent of evaluation block boundaries."""
    if progress["order"] is None:
        progress["order"] = torch.randperm(count).tolist()
    index = progress["order"][progress["position"]]
    progress["position"] += 1
    if progress["position"] == count:
        progress["passes_completed"] += 1
        progress["position"], progress["order"] = 0, None
    return index


class Experiment(PatternExperiment):
    def __init__(self, config, mat, csv, output, device):
        # The old superclass initializer intentionally accepts only its frozen
        # historical split. Validate this separate protocol without changing it.
        validate_config(config)
        self.config = copy.deepcopy(config)
        root = Path(__file__).parent
        self.config["experimental_source"] = {name: f.file_sha256(root / name) for name in
            ("train_crossed_ab_pilot.py", "pattern_amplitude_model.py",
             "train_pattern_pilot.py", "train_expanded_pilot.py")}
        self.mat, self.csv, self.output, self.device = str(mat), str(csv), Path(output), str(device)
        self.samples, self.current, self.evaluations = {}, None, 0

    def dataset(self, mat, split, indices, config, *, allow_test=False):
        if split not in ("train", "validation") or allow_test:
            raise PermissionError("Only training/validation are available; case1 unused and case3 sealed")
        return super().dataset(mat, split, indices, config)

    def prepare(self):
        protocol = self.output / "protocol.json"
        if protocol.exists() and json.loads(protocol.read_text())["config"] != self.config:
            raise ValueError("Output protocol differs; use a new output folder")
        run_root = self.output / "K10/seed_17"
        if not protocol.exists() and run_root.exists() and any(run_root.iterdir()):
            raise FileExistsError("Unrecognized existing run; use a new output folder")
        train = self.dataset(self.mat, "train", range(10), self.config)
        validation = self.dataset(self.mat, "validation", range(10), self.config)
        report = {"status": "pending_selection", "config": self.config, "splits": SPLITS,
                  "counts": {"training": 4, "validation": 2, "sealed_test": 1, "unused": 1},
                  "training_budget": self.config["training_budget"],
                  "data_passes": self.config["epochs"], "checkpoint_policy": "fixed_final_update",
                  "baselines": without_score(f.baseline_metrics(train, validation)), "test_accessed": False}
        atomic_write(protocol, report)
        return report

    def evaluate(self, model, samples, preprocessing, device):
        result = super().evaluate(model, samples, preprocessing, device)
        print("  " + "; ".join(
            f"case{r['case_id']} volume={r['refined']['predicted_volume_mm3']/1000:.2f}/"
            f"{r['refined']['true_volume_mm3']/1000:.2f}ml centre={r['refined']['localization_error_mm']:.2f}mm "
            f"epsilon={r['refined']['epsilon_rmse']:.4f} sigma={r['refined']['sigma_rmse_s_per_m']:.5f}"
            for r in result["per_case"]), flush=True)
        return result

    def run(self, counts=(10,), *, seeds=None, resume=False, max_updates=None):
        if list(counts) != [10] or (seeds is not None and list(seeds) != [17]):
            raise ValueError("This experiment runs only K10 seed17")
        if max_updates is not None and max_updates < 1:
            raise ValueError("max-updates must be positive")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable here; run this script on the RTX machine")
        self.prepare()
        self.current = (10, 17)
        return [self.run_candidate(10, 17, resume=resume, max_updates=max_updates)]

    def run_candidate(self, k, seed, *, resume=False, max_updates=None):
        if (k, seed) != (10, 17):
            raise ValueError("This experiment runs only K10 seed17")
        train, validation, preprocessing, identity = self.inputs(k, seed)
        identity_hash = f.canonical_hash(identity)
        root = self.output / "K10/seed_17"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint = root / "resume.pt"
        result_path = root / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if (result.get("identity_sha256") != identity_hash
                    or f.canonical_hash(result.get("identity")) != identity_hash
                    or result.get("status") != "completed" or result.get("test_accessed") is not False
                    or not (root / "final.pt").exists()
                    or f.file_sha256(root / "final.pt") != result.get("final_checkpoint_sha256")):
                raise ValueError("Completed-run identity/artifact mismatch; use a fresh output")
            return result
        if checkpoint.exists() and not resume:
            raise FileExistsError("Interrupted run exists; use --resume")
        if not checkpoint.exists() and any((root / name).exists() for name in ("final.pt", "initial_metrics.json")):
            raise FileExistsError("Orphaned run artifacts; use a new output folder")
        if any((self.output / name).exists() for name in ("sealed_test_access.json", "sealed_test_results.json")):
            raise PermissionError("Sealed-test access locks this output")

        f.seed_everything(seed)
        model = PatternAmplitudeMSRNN(self.config["model"]).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["learning_rate"],
                                     weight_decay=self.config["weight_decay"])
        budget = self.config["training_budget"]
        total, interval = budget["optimizer_updates"], budget["updates_per_evaluation"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget["scheduler_t_max"])
        amp = bool(self.config.get("amp", True) and self.device.startswith("cuda"))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        progress = {"updates": 0, "passes_completed": 0, "position": 0, "order": None,
                    "pending_evaluation_learning_rate": None,
                    "evaluations": 0, "history": [], "case_update_counts": {str(cid): 0 for cid in SPLITS["train"]}}
        metadata = {"schema_version": 1, "identity": identity, "identity_sha256": identity_hash,
                    "architecture": self.config["model"], "representation": "pattern_amplitude_v1",
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "preprocessing": preprocessing, "splits": identity["splits"],
                    "selected_indices": list(range(10)), "checkpoint_policy": "fixed_final_update",
                    "physics_requested": "A", "physics_active": "none", "test_accessed": False}
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
            if (saved["identity_sha256"] != identity_hash
                    or f.canonical_hash(saved["identity"]) != identity_hash):
                raise ValueError("Resume identity mismatch: config/data/code/execution must match")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            progress = saved["progress"]
            f.restore_rng(saved["rng"])
            del saved
        else:
            atomic_write(root / "initial_metrics.json", {
                "stage": "untrained_initialization", "optimizer_updates": 0,
                "identity_sha256": identity_hash, "test_accessed": False,
                "training": self.metrics(model, train, preprocessing, self.device),
                "validation": self.metrics(model, validation, preprocessing, self.device)})

        self.evaluations = progress["evaluations"]

        def save():
            atomic_write(checkpoint, {**metadata, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": f.rng_state(), "progress": progress}, checkpoint=True)

        started, invocation_updates = time.perf_counter(), 0
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        # A checkpoint may stop exactly after update5 but before evaluation.
        # Scheduling is checkpointed with the optimizer update; process pending
        # evaluation on resume before taking the next optimizer update.
        while True:
            if progress["updates"] == (progress["evaluations"] + 1) * interval:
                lr_used = progress["pending_evaluation_learning_rate"]
                metrics = self.evaluate(model, validation, preprocessing, self.device)
                progress["history"].append({"evaluation": progress["evaluations"] + 1,
                    "optimizer_updates": progress["updates"],
                    "data_passes_completed": progress["passes_completed"],
                    "learning_rate": lr_used, "validation": metrics})
                progress["evaluations"] += 1
                progress["pending_evaluation_learning_rate"] = None
                save()
                atomic_write(root / "training_history.json", {
                    "identity_sha256": identity_hash, "checkpoint_policy": "fixed_final_update",
                    "history": progress["history"], "test_accessed": False})
            if progress["updates"] == total:
                break
            if max_updates is not None and invocation_updates >= max_updates:
                return {"status": "running", "k": 10, "seed": 17, "updates": progress["updates"],
                        "identity_sha256": identity_hash, "bounded_mechanism_check": True}
            model.train()
            sample = train[next_sample_index(progress, len(train))]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                output = f._predict(model, sample, preprocessing, self.device)
                loss = f._objective(output, sample["target"].unsqueeze(0).to(self.device),
                                    sample["object_mask"].unsqueeze(0).to(self.device), self.config["gradient_weight"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training objective")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            progress["updates"] += 1
            progress["case_update_counts"][str(sample["metadata"]["case_id"])] += 1
            invocation_updates += 1
            if progress["updates"] % interval == 0:
                progress["pending_evaluation_learning_rate"] = optimizer.param_groups[0]["lr"]
                scheduler.step()
            save()

        atomic_write(root / "final.pt", {**metadata, "model": model.state_dict(),
            "epoch": progress["passes_completed"], "optimizer_updates": progress["updates"]}, checkpoint=True)
        result = {**metadata, "status": "completed", "k": 10, "seed": 17,
                  "training": progress["history"][-1]["validation"]["training_fit"],
                  "validation": progress["history"][-1]["validation"],
                  "baselines": without_score(f.baseline_metrics(train, validation)),
                  "epochs_executed": progress["passes_completed"], "optimizer_updates": progress["updates"],
                  "case_update_counts": progress["case_update_counts"],
                  "final_checkpoint_sha256": f.file_sha256(root / "final.pt"),
                  "last_invocation_elapsed_seconds": time.perf_counter() - started,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.startswith("cuda") else None,
                  "device": self.device}
        atomic_write(result_path, result)
        print(f"K10 seed17: completed {progress['updates']} updates, {progress['passes_completed']} data passes; final.pt", flush=True)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--base-config", default="configs/frequency_pilot_pattern_v1.json")
    parser.add_argument("--mat", default="data/pilot_expanded/frequency_pilot.mat")
    parser.add_argument("--csv", default="data/pilot/frequency_subsets.csv")
    parser.add_argument("--output", default="outputs/frequency_crossed_ab_v1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, help="Bounded execution; never counts as a completed pilot")
    args = parser.parse_args(argv)
    config = build_config(json.loads(Path(args.base_config).read_text()))
    experiment = Experiment(config, args.mat, args.csv, args.output, args.device)
    print("K10 seed17 | train [2,4,5,6] | validation [7,8] | unused [1] | sealed [3]", flush=True)
    print("1500 updates; 375 four-case passes; original 300-block cosine schedule; unchanged weighted loss", flush=True)
    result = experiment.prepare() if args.command == "prepare" else experiment.run(resume=args.resume, max_updates=args.max_updates)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
