"""Controlled pattern/amplitude pilot: fixed final epoch, no balanced score.

Reuses the crossed-v3 data contract and the EXACT original weighted objective.
No changes to legacy package sources, runners or checkpoint identities.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from microscan import frequency as f
from pattern_amplitude_model import PatternAmplitudeMSRNN, REPRESENTATION
from train_expanded_pilot import Experiment as CrossedExperiment


def without_score(value):
    if isinstance(value, dict):
        return {k: without_score(v) for k, v in value.items() if k != "balanced_contrast_rmse"}
    if isinstance(value, list):
        return [without_score(v) for v in value]
    return value


def atomic_write(path, payload, *, checkpoint=False):
    """Retry transient Windows destination locks; never delete the last checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if checkpoint:
        torch.save(payload, temporary)
    else:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(60):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 59:
                raise
            time.sleep(.5)


class Experiment(CrossedExperiment):
    def __init__(self, config, mat, csv, output, device):
        super().__init__(config, mat, csv, output, device)
        if config.get("protocol_version") != 3:
            raise ValueError("Pattern pilot requires the frozen crossed-v3 split")
        if config.get("checkpoint_policy") != "fixed_final_epoch":
            raise ValueError("Use fixed final-epoch weights, with no validation score selection")
        if config.get("model", {}).get("measurement_representation") != REPRESENTATION:
            raise ValueError("Explicit pattern/amplitude model configuration required")
        if config.get("representation_experiment") != REPRESENTATION:
            raise ValueError("Explicit representation experiment identity required")
        root = Path(__file__).parent
        self.config["experimental_source"] = {name: f.file_sha256(root / name) for name in
            ("pattern_amplitude_model.py", "train_pattern_pilot.py", "train_expanded_pilot.py")}

    def metrics(self, model, samples, preprocessing, device):
        # Keep physical RMSE, geometry, and separate region diagnostics. The old
        # composite is neither reported nor used for scheduling/checkpoint choice.
        return without_score(super().metrics(model, samples, preprocessing, device))

    def prepare(self):
        train = self.dataset(self.mat, "train", range(10), self.config)
        validation = self.dataset(self.mat, "validation", range(10), self.config)
        report = {"status": "pending_selection", "config": self.config,
            "splits": self.config["expanded_protocol"]["splits"],
            "counts": {"training": len(train), "validation": len(validation), "sealed_test": 1},
            "baselines": without_score(f.baseline_metrics(train, validation)),
            "checkpoint_policy": "fixed_final_epoch", "test_accessed": False}
        atomic_write(self.output / "protocol.json", report)
        return report

    def inputs(self, k, seed):
        if seed not in self.config["seeds"]:
            raise ValueError("Seed outside configured paired seeds")
        indices = f.read_subsets(self.csv)[int(k)]
        train = self.dataset(self.mat, "train", indices, self.config)
        validation = self.dataset(self.mat, "validation", indices, self.config)
        preprocessing = f.fit_input_scale(train)
        preprocessing["fit_case_ids"] = list(self.config["expanded_protocol"]["splits"]["train"])
        identity = {"config": self.config, "source": f.source_identity(),
            "data_sha256": f.file_sha256(self.mat), "csv_sha256": f.file_sha256(self.csv),
            "k": int(k), "seed": int(seed), "indices_zero_based": indices,
            "splits": self.config["expanded_protocol"]["splits"], "preprocessing": preprocessing,
            "execution": f.execution_identity(self.device, self.config.get("amp", True)),
            "physics_requested": "A", "physics_active": "none", "test_accessed": False}
        # Plain JSON metadata also makes the model-only final.pt weights-only safe.
        identity = json.loads(json.dumps(identity))
        return train, validation, preprocessing, identity

    def run(self, counts=(10,), *, resume=False, max_updates=None, seeds=None):
        if max_updates is not None and max_updates < 1:
            raise ValueError("max-updates must be positive")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable here; use the RTX machine")
        results = []
        for k in counts:
            if k not in range(3, 11):
                raise ValueError("Frequency count must be 3..10")
            for seed in (self.config["seeds"] if seeds is None else seeds):
                self.current, self.evaluations = (k, seed), 0
                results.append(self.run_candidate(k, seed, resume=resume, max_updates=max_updates))
        return results

    def run_candidate(self, k, seed, *, resume=False, max_updates=None):
        train, validation, preprocessing, identity = self.inputs(k, seed)
        identity_hash = f.canonical_hash(identity)
        root = self.output / f"K{k:02d}" / f"seed_{seed}"
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = root / "resume.pt"
        if (root / "result.json").exists():
            result = json.loads((root / "result.json").read_text())
            if (result.get("identity_sha256") != identity_hash
                    or f.canonical_hash(result.get("identity")) != identity_hash
                    or result.get("status") != "completed" or result.get("test_accessed")
                    or not (root / "final.pt").exists()
                    or f.file_sha256(root / "final.pt") != result.get("final_checkpoint_sha256")):
                raise ValueError("Completed-run identity/artifact mismatch; use a fresh output")
            return result
        if checkpoint_path.exists() and not resume:
            raise FileExistsError("Interrupted run exists; use --resume")
        if any((self.output / name).exists() for name in ("sealed_test_access.json", "sealed_test_results.json")):
            raise PermissionError("Sealed-test access locks new training in this output")
        f.seed_everything(seed)
        model = PatternAmplitudeMSRNN(self.config["model"]).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(self.config["learning_rate"]),
                                     weight_decay=float(self.config["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config["epochs"])
        amp = bool(self.config.get("amp", True) and self.device.startswith("cuda"))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        state = {"epoch": 0, "position": 0, "order": None, "updates": 0, "history": []}
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if (saved["identity_sha256"] != identity_hash
                    or f.canonical_hash(saved["identity"]) != identity_hash):
                raise ValueError("Resume identity mismatch: config/data/code/execution must match")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            scaler.load_state_dict(saved["scaler"])
            state = saved["progress"]
            f.restore_rng(saved["rng"])
            del saved
        else:
            initial = {"stage": "untrained_initialization", "optimizer_updates": 0,
                "identity_sha256": identity_hash, "test_accessed": False,
                "training": self.metrics(model, train, preprocessing, self.device),
                "validation": self.metrics(model, validation, preprocessing, self.device)}
            atomic_write(root / "initial_metrics.json", initial)

        parameter_count = sum(p.numel() for p in model.parameters())
        metadata = {"schema_version": 1, "identity": identity, "identity_sha256": identity_hash,
            "architecture": self.config["model"], "representation": REPRESENTATION,
            "parameter_count": parameter_count, "preprocessing": preprocessing,
            "splits": identity["splits"], "selected_indices": identity["indices_zero_based"],
            "checkpoint_policy": "fixed_final_epoch", "physics_requested": "A", "physics_active": "none"}
        def save():
            atomic_write(checkpoint_path, {**metadata, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "rng": f.rng_state(), "progress": state}, checkpoint=True)

        started, invocation_updates = time.perf_counter(), 0
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)
        while state["epoch"] < self.config["epochs"]:
            if state["order"] is None:
                state["order"] = torch.randperm(len(train)).tolist()
            model.train()
            while state["position"] < len(state["order"]):
                sample = train[state["order"][state["position"]]]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                    prediction = f._predict(model, sample, preprocessing, self.device)
                    loss = f._objective(prediction, sample["target"].unsqueeze(0).to(self.device),
                        sample["object_mask"].unsqueeze(0).to(self.device), self.config["gradient_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training objective")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["gradient_clip"], error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                state["position"] += 1
                state["updates"] += 1
                invocation_updates += 1
                save()
                if max_updates is not None and invocation_updates >= max_updates:
                    return {"status": "running", "k": k, "seed": seed, "updates": state["updates"],
                        "identity_sha256": identity_hash, "bounded_mechanism_check": True}
            metrics = self.evaluate(model, validation, preprocessing, self.device)
            state["history"].append({"epoch": state["epoch"], "optimizer_updates": state["updates"],
                                      "validation": metrics})
            scheduler.step()
            state["epoch"] += 1
            state["position"], state["order"] = 0, None
            save()
            atomic_write(root / "training_history.json", {"identity_sha256": identity_hash,
                "checkpoint_policy": "fixed_final_epoch", "history": state["history"], "test_accessed": False})
        # No best-state restore, early stopping, score ranking or validation-LR hook.
        atomic_write(root / "final.pt", {**metadata, "model": model.state_dict(),
            "epoch": state["epoch"], "optimizer_updates": state["updates"]}, checkpoint=True)
        result = {**metadata, "status": "completed", "k": k, "seed": seed,
            "validation": state["history"][-1]["validation"],
            "baselines": without_score(f.baseline_metrics(train, validation)),
            "epochs_executed": state["epoch"], "optimizer_updates": state["updates"],
            "final_checkpoint_sha256": f.file_sha256(root / "final.pt"),
            "last_invocation_elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.startswith("cuda") else None,
            "device": self.device, "test_accessed": False}
        atomic_write(root / "result.json", result)
        print(f"K{k} seed{seed}: completed {state['updates']} updates; final.pt", flush=True)
        return result


def profile(config, device, *, backward=False):
    """Synthetic structural profile only, including optional AMP backward."""
    f.seed_everything(17)
    model = PatternAmplitudeMSRNN(config["model"]).to(device)
    model.train(backward)
    amp = bool(config.get("amp", True) and str(device).startswith("cuda"))
    cuda = str(device).startswith("cuda")
    signal = torch.randn(1, 2, 10, 18, 18, device=device)
    mask = torch.ones(1, 1, 10, 18, 18, device=device, dtype=torch.bool)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.set_grad_enabled(backward), torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
        output = model(signal, mask)
        loss = output["coarse"].square().mean() + output["refined"].square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite synthetic profile output")
    if backward:
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite synthetic profile gradient")
    if cuda:
        torch.cuda.synchronize(device)
    return {"evidence": "one synthetic-input structural profile; no accuracy evidence",
        "representation": REPRESENTATION, "device": str(device), "torch": str(torch.__version__),
        "amp": amp, "backward": backward, "parameters": sum(p.numel() for p in model.parameters()),
        "shape": list(output["refined"].shape), "seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if cuda else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "profile"))
    parser.add_argument("--config", default="configs/frequency_pilot_pattern_v1.json")
    parser.add_argument("--mat", default="data/pilot_expanded/frequency_pilot.mat")
    parser.add_argument("--csv", default="data/pilot/frequency_subsets.csv")
    parser.add_argument("--output", default="outputs/frequency_pattern_v1")
    parser.add_argument("--counts", default="10")
    parser.add_argument("--seeds", help="Run a subset of configured seeds, e.g. 17, without changing identity")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--backward", action="store_true", help="Include backward in the synthetic profile")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    if args.command == "profile":
        result = profile(config, args.device, backward=args.backward)
        print(json.dumps(result, indent=2))
        return result
    experiment = Experiment(config, args.mat, args.csv, args.output, args.device)
    result = experiment.prepare() if args.command == "prepare" else experiment.run(
        [int(k) for k in args.counts.split(",")], resume=args.resume, max_updates=args.max_updates,
        seeds=None if args.seeds is None else [int(seed) for seed in args.seeds.split(",")])
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
