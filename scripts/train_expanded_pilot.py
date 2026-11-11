"""Versioned eight-geometry pilot using the existing reconstruction trainer.

This standalone process adapter leaves microscan/ unchanged so old checkpoints
retain their original source identity. New identities bind this runner, expanded
MAT, exact split and original-prefix integrity. Default run: K10, three seeds.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from microscan import frequency as f
from microscan.data import (load_mat, sphere_target, calibrate, complex_to_channels,
                            PILOT_FREQUENCIES, PILOT_COMSOL_PORTS)


class ExpandedPilot:
    def __init__(self, mat, split, indices, config, *, allow_test=False):
        protocol = config["expanded_protocol"]
        if split == "test" and not allow_test:
            raise PermissionError("Original case3 remains sealed until explicit frequency selection")
        key = "sealed_test" if split == "test" else split
        self.ids = protocol["splits"][key]
        self.split, self.indices = split, list(indices)
        self.grid_size = config.get("grid_size", 128)
        m = load_mat(mat)
        d, self.cfg = m["data"], m["cfg"]
        self.s, self.cases = np.asarray(d["S"]), d["cases"]
        if self.s.shape != (18,18,10,9) or self.s.dtype != np.complex128:
            raise ValueError("Expanded pilot requires complex128 [18,18,10,9]")
        if not np.isfinite(self.s).all() or not np.asarray(d["done"]).all() or not bool(m["report"]["completed"]):
            raise ValueError("Expanded pilot has incomplete/nonfinite measurements")
        if f.file_sha256(mat) != protocol["data_sha256"]:
            raise ValueError("Expanded MAT hash differs from the frozen generation record")
        prefix = hashlib.sha256(np.ascontiguousarray(self.s[:,:,:,:5]).tobytes()).hexdigest()
        if prefix != protocol["original_prefix_sha256"]:
            raise ValueError("Original measurements were altered in the extension")
        if not np.allclose(d["GHz"], PILOT_FREQUENCIES, rtol=0, atol=1e-8):
            raise ValueError("Frequency grid differs from the original ten slots")
        if list(d["physicalPorts"]) != list(range(1,19)) or list(d["comsolPorts"]) != PILOT_COMSOL_PORTS:
            raise ValueError("Physical/COMSOL port mapping changed")
        if list(d["axes"]) != ["receiver_physical","transmitter_physical","frequency","case"]:
            raise ValueError("Unexpected measurement axes")
        if float(self.cfg["gridHalfSpanMM"]) != 60 or int(self.cfg["gridN"]) != 128:
            raise ValueError("Physical label grid differs from the sphere-pilot contract")
        for actual, expected in zip(self.cases, protocol["cases"]):
            if (int(actual["id"]) != expected["id"] or float(actual["radiusMM"]) != expected["radius_mm"]
                    or not np.array_equal(np.asarray(actual["centreMM"],float), expected["centre_xyz_mm"])):
                raise ValueError("Geometry metadata differs from the allocation")
        if not self.indices or len(set(self.indices)) != len(self.indices) or min(self.indices)<0 or max(self.indices)>9:
            raise ValueError("Invalid frequency slots")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        cid = self.ids[index]
        current = self.s[:,:,:,cid].transpose(2,0,1)
        reference = self.s[:,:,:,0].transpose(2,0,1)
        mask = np.zeros_like(current, dtype=bool)
        mask[self.indices] = True
        difference, mask = calibrate(current, reference, mask, mask)
        target, foreground = sphere_target(self.cases[cid], self.grid_size, 60.)
        case = self.cases[cid]
        geometry = {"id": cid, "radius_mm": float(case["radiusMM"]),
                    "centre_xyz_mm": np.asarray(case["centreMM"],float).tolist()}
        return {"signals": complex_to_channels(difference), "mask": torch.from_numpy(mask[None].copy()),
                "target": target, "object_mask": foreground,
                "frequencies_ghz": torch.tensor(PILOT_FREQUENCIES,dtype=torch.float32),
                "metadata": {"case_id": cid, "geometry_id": f"sphere_case_{cid}", "split": self.split,
                             "geometry": geometry, "domain": "calibrated_delta_s"}}


def validate_allocation(config):
    f.validate_config(config)
    protocol = config.get("expanded_protocol", {})
    required = {"reference":[0], "train":[1,2,4,5,6], "validation":[7,8], "sealed_test":[3]}
    if config.get("protocol_version") != 2 or protocol.get("splits") != required:
        raise ValueError("Use the explicitly frozen v2 allocation; original case3 cannot enter training")
    if len(protocol.get("cases", [])) != 9:
        raise ValueError("All nine reference/object metadata records are required")
    return config


class Experiment:
    def __init__(self, config, mat, csv, output, device):
        self.config = copy.deepcopy(validate_allocation(config))
        self.config["expanded_protocol"]["runner_sha256"] = f.file_sha256(__file__)
        self.mat, self.csv, self.output, self.device = str(mat), str(csv), Path(output), str(device)
        self.samples = {}
        self.current = None
        self.evaluations = 0

    def dataset(self, mat, split, indices, config, *, allow_test=False):
        key = (split, tuple(indices))
        if key not in self.samples:
            self.samples[key] = list(ExpandedPilot(mat, split, indices, config, allow_test=allow_test))
        # A cached test loader can only be used in the explicitly authorized call.
        if split == "test" and not allow_test:
            raise PermissionError("Test access is explicit")
        return self.samples[key]

    def metrics(self, model, samples, preprocessing, device):
        model.eval()
        rows = []
        with torch.inference_mode():
            for sample in samples:
                prediction = f._predict(model, sample, preprocessing, device)
                rows.append({"case_id": sample["metadata"]["case_id"],
                    **{key: f.reconstruction_metrics(prediction[key][0], sample["target"]) for key in ("coarse","refined")},
                    "evaluation_objective_float32": float(f._objective(prediction,
                        sample["target"].unsqueeze(0).to(device), sample["object_mask"].unsqueeze(0).to(device),
                        self.config.get("gradient_weight", .01)).cpu())})
        return {"refined": f.average_metrics([row["refined"] for row in rows]),
                "coarse": f.average_metrics([row["coarse"] for row in rows]),
                "per_case": rows,
                "evaluation_objective_float32": float(np.mean([row["evaluation_objective_float32"] for row in rows]))}

    def evaluate(self, model, samples, preprocessing, device):
        result = self.metrics(model, samples, preprocessing, device)
        if self.current is not None:
            k, seed = self.current
            training = self.dataset(self.mat, "train", f.read_subsets(self.csv)[k], self.config)
            result["training_fit"] = self.metrics(model, training, preprocessing, device)
            self.evaluations += 1
            train = result["training_fit"]["refined"]
            val = result["refined"]
            print(f"K{k} seed{seed} evaluation {self.evaluations}: "
                  f"train epsilon={train['epsilon_rmse']:.4f}, sigma={train['sigma_rmse_s_per_m']:.5f}; "
                  f"validation epsilon={val['epsilon_rmse']:.4f}, sigma={val['sigma_rmse_s_per_m']:.5f}, "
                  f"centre={val['localization_error_mm']:.2f}mm", flush=True)
        return result

    @contextlib.contextmanager
    def adapter(self):
        names = ("SPLITS", "PROTOCOL_VERSION", "_dataset", "_evaluate", "_model")
        old = {name:getattr(f,name) for name in names}
        original_replace = os.replace
        def reliable_replace(source, destination):
            for attempt in range(60):
                try:
                    return original_replace(source, destination)
                except PermissionError:
                    if attempt == 59:
                        raise
                    time.sleep(.5)
        def make_model(config):
            model = old["_model"](config).to(self.device)
            if self.current is not None:
                k, seed = self.current
                train = self.dataset(self.mat, "train", f.read_subsets(self.csv)[k], config)
                validation = self.dataset(self.mat, "validation", f.read_subsets(self.csv)[k], config)
                preprocessing = f.fit_input_scale(train)
                identity = f.run_identity(config,self.mat,self.csv,k,seed,preprocessing,
                    execution=f.execution_identity(self.device,config.get("amp",True)))
                path = self.output/f"K{k:02d}"/f"seed_{seed}"/"initial_metrics.json"
                if path.exists():
                    if json.loads(path.read_text())["identity_sha256"] != f.canonical_hash(identity):
                        raise ValueError("Initial-metric lineage mismatch; use a fresh output directory")
                else:
                    # This is after seeded initialization and before any optimizer update.
                    f.atomic_json(path,{"stage":"untrained_initialization", "optimizer_updates":0,
                        "identity_sha256":f.canonical_hash(identity),"test_accessed":False,
                        "training":self.metrics(model,train,preprocessing,self.device),
                        "validation":self.metrics(model,validation,preprocessing,self.device)})
                model.train()
            return model
        f.SPLITS = copy.deepcopy(self.config["expanded_protocol"]["splits"])
        f.PROTOCOL_VERSION = 2
        f._dataset, f._evaluate, f._model = self.dataset, self.evaluate, make_model
        os.replace = reliable_replace
        try:
            yield
        finally:
            for name,value in old.items():
                setattr(f,name,value)
            os.replace = original_replace

    def prepare(self):
        train = self.dataset(self.mat,"train",range(10),self.config)
        validation = self.dataset(self.mat,"validation",range(10),self.config)
        report = {"status":"pending_selection","protocol_version":2,"config":self.config,
                  "splits":self.config["expanded_protocol"]["splits"],
                  "case_inventory":self.config["expanded_protocol"]["cases"],
                  "counts":{"training":len(train),"validation":len(validation),"sealed_test":1},
                  "baselines":f.baseline_metrics(train,validation),"data_sha256":f.file_sha256(self.mat),
                  "csv_sha256":f.file_sha256(self.csv),"test_accessed":False,
                  "scope":"Eight nondispersive sphere geometries, not independent clinical/headset validation"}
        f.atomic_json(self.output/"protocol.json",report)
        return report

    def run(self, counts=(10,), *, resume=False, max_updates=None):
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable here; run the supplied command on the RTX machine")
        results = []
        with self.adapter():
            for k in counts:
                if k not in range(3,11):
                    raise ValueError("Frequency counts must be3..10")
                for seed in self.config["seeds"]:
                    self.current, self.evaluations = (k,seed), 0
                    results.append(f.run_candidate(self.config,self.mat,self.csv,self.output,k,seed,
                        device=self.device,resume=resume,max_updates=max_updates))
                    if results[-1]["status"] == "completed":
                        root = self.output/f"K{k:02d}"/f"seed_{seed}"
                        state = torch.load(root/"resume.pt",map_location="cpu",weights_only=False)
                        results[-1]["last_epoch_metrics"] = state["progress"]["history"][-1]
                        results[-1]["selected_checkpoint_policy"] = "legacy balanced-contrast validation score; last-epoch metrics retained separately"
                        results[-1]["initial_metrics_file"] = "initial_metrics.json"
                        f.atomic_json(root/"result.json",results[-1])
                        del state
                    print(f"K{k} seed{seed}: {results[-1]['status']}",flush=True)
        return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","run","select","test"))
    p.add_argument("--config",default="configs/frequency_pilot_expanded.json")
    p.add_argument("--mat",default="data/pilot_expanded/frequency_pilot.mat")
    p.add_argument("--csv",default="data/pilot/frequency_subsets.csv")
    p.add_argument("--output",default="outputs/frequency_expanded_v2")
    p.add_argument("--counts",default="10",help="Start with the ten-frequency comparator; explicitly request all K later")
    p.add_argument("--device",default="cpu")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--max-updates",type=int,help="Bounded verification; leaves candidate incomplete")
    args = p.parse_args(argv)
    experiment = Experiment(json.loads(Path(args.config).read_text()),args.mat,args.csv,args.output,args.device)
    if args.command == "prepare":
        result = experiment.prepare()
    elif args.command == "run":
        result = experiment.run([int(k) for k in args.counts.split(',')],resume=args.resume,max_updates=args.max_updates)
    else:
        with experiment.adapter():
            fn = f.select_and_freeze if args.command == "select" else f.evaluate_sealed_test
            kwargs = {"device":args.device} if args.command == "test" else {}
            result = fn(experiment.config,args.mat,args.csv,args.output,**kwargs)
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__ == "__main__":
    main()
