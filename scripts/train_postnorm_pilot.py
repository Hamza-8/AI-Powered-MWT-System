"""One controlled K10/seed17 retraining with unchanged crossed-v3 objective.

The old runner is used through a scoped model factory override, leaving its
source/checkpoint identities intact. Production runs require completed baseline
weights and their frozen-weight ablation report. Full budget is 300 x 5 updates.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

from microscan import frequency as f
import train_pattern_pilot as legacy
from diagnose_amplitude_conditioning import load_checkpoint, compare_checkpoints
from postnorm_amplitude_model import ADJUSTMENT, PostNormAmplitudeMSRNN


@contextlib.contextmanager
def model_factory():
    old = legacy.PatternAmplitudeMSRNN
    legacy.PatternAmplitudeMSRNN = PostNormAmplitudeMSRNN
    try:
        yield
    finally:
        legacy.PatternAmplitudeMSRNN = old


class Experiment(legacy.Experiment):
    def __init__(self, config, mat, csv, output, device, *, baseline, evidence, mechanism_fixture=False):
        super().__init__(config, mat, csv, output, device)
        state, _ = load_checkpoint(baseline)
        old_config = state["identity"]["config"]
        if state["architecture"].get("postnorm_amplitude"):
            raise ValueError("Reference must be the unchanged pattern-v1 checkpoint")
        adjustment = config.get("model", {}).get("postnorm_amplitude", {})
        if adjustment.get("name") != ADJUSTMENT or config["seeds"] != [17]:
            raise ValueError("Only the explicit postnorm K10 seed17 experiment is allowed")
        equal = ("batch_size", "learning_rate", "weight_decay", "gradient_weight", "gradient_clip", "amp", "expanded_protocol")
        if not mechanism_fixture:
            equal += ("epochs", "patience", "grid_size")
            if config["epochs"] != 300 or config["grid_size"] != 128:
                raise ValueError("Native128 with the original 300-epoch budget is required")
        for key in equal:
            if self.config[key] != old_config[key]:
                raise ValueError(f"Baseline training protocol differs: {key}")
        old_model = old_config["model"]
        if not mechanism_fixture and {k:v for k,v in config["model"].items() if k != "postnorm_amplitude"} != old_model:
            raise ValueError("Change only postnorm amplitude routing, leaving the rest of the model fixed")
        record = json.loads(Path(evidence).read_text())
        required = {"intact", "raw_off", "rms_off", "both_off", "coarse_off", "enhancement_off", "matched_gain"}
        if (record.get("checkpoint_sha256") != f.file_sha256(baseline)
                or record.get("identity_sha256") != state["identity_sha256"]
                or not record.get("weights_unchanged") or record.get("test_accessed") is not False
                or record.get("optimizer_updates") != 0 or set(record.get("conditions", {})) != required):
            raise ValueError("Completed, intact train/validation baseline ablation report required")
        for rows in record["conditions"].values():
            if sorted(row["case_id"] for row in rows) != [5,6,7,8]:
                raise ValueError("Ablations must compare both real B/C size pairs")
        self.baseline, self.evidence = str(baseline), str(evidence)
        destination = self.output / "K10/seed_17/final.pt"
        if destination.resolve() == Path(baseline).resolve() or self.output.resolve().is_relative_to(Path(baseline).resolve().parent):
            raise ValueError("Use a new output directory outside the reference run")
        root = Path(__file__).parent
        for filename in ("train_postnorm_pilot.py", "postnorm_amplitude_model.py", "diagnose_amplitude_conditioning.py"):
            self.config["experimental_source"][filename] = f.file_sha256(root / filename)
        self.config["targeted_adjustment"] = {
            "name":ADJUSTMENT, "baseline_checkpoint_sha256":f.file_sha256(baseline),
            "ablation_report_sha256":f.file_sha256(evidence), "mechanism_fixture":bool(mechanism_fixture),
            "training_initialization":"from_scratch_paired_seed17", "loss":"unchanged_weighted_objective"}

    def run(self, counts=(10,), *, seeds=None, **kwargs):
        if list(counts) != [10] or (seeds is not None and list(seeds) != [17]):
            raise ValueError("This experiment trains only K10 seed17")
        with model_factory():
            return super().run(counts, seeds=[17], **kwargs)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","profile","compare"))
    parser.add_argument("--config",default="configs/frequency_pilot_postnorm_v1.json")
    parser.add_argument("--mat",default="data/pilot_expanded/frequency_pilot.mat")
    parser.add_argument("--csv",default="data/pilot/frequency_subsets.csv")
    parser.add_argument("--output",default="outputs/frequency_postnorm_v1")
    parser.add_argument("--baseline",default="outputs/frequency_pattern_v1/K10/seed_17/final.pt")
    parser.add_argument("--evidence",default="outputs/amplitude_diagnostic_v1/report.json")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--max-updates",type=int)
    parser.add_argument("--backward",action="store_true")
    args=parser.parse_args(argv)
    config=json.loads(Path(args.config).read_text())
    if args.command=="profile":
        with model_factory():
            result=legacy.profile(config,args.device,backward=args.backward)
        result["targeted_adjustment"]=ADJUSTMENT
    else:
        experiment=Experiment(config,args.mat,args.csv,args.output,args.device,baseline=args.baseline,evidence=args.evidence)
        if args.command=="run":
            results=experiment.run(resume=args.resume,max_updates=args.max_updates)
            if results[0]["status"] != "completed":
                print(json.dumps(results,indent=2));return results
        candidate=Path(args.output)/"K10/seed_17/final.pt"
        result=compare_checkpoints(args.baseline,candidate,args.mat,Path(args.output)/"comparison.json",args.device)
        for row in result["per_case"]:
            if row["split"] != "validation":continue
            print(f"VALIDATION case{row['case_id']}")
            for name in ("baseline","candidate"):
                m=row[name]["refined"]
                print(f"  {name}: centre={m['localization_error_mm']:.3f}mm, volume={m['predicted_volume_mm3']/1000:.3f}ml, "
                      f"epsilon={m['epsilon_rmse']:.4f}, sigma={m['sigma_rmse_s_per_m']:.5f}")
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=="__main__":
    main()
