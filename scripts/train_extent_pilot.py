"""Loss-only radius experiment: original pattern-v1 model plus soft Dice.

One matched K10/seed17 run. Existing package, model and runners remain unchanged.
Dice is applied to the two refined normalized-contrast maps independently.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import time

import torch

from microscan import frequency as f
from pattern_amplitude_model import PatternAmplitudeMSRNN
from train_pattern_pilot import Experiment as PatternExperiment, atomic_write, without_score


BASE_OBJECTIVE = f._objective
LOSS_NAME = "refined_contrast_squared_soft_dice_v1"
SPLITS = {"reference":[0], "train":[1,2,4,6,7], "validation":[5,8], "sealed_test":[3]}


def extent_per_property(prediction, foreground, smooth=1e-6):
    """[B,2] squared-denominator Dice loss, using float32 spatial reductions.

    These are continuous physical contrasts, not a new segmentation head.
    No clipping/threshold/sigmoid is used: false positives above contrast1 must
    retain a gradient. This pilot's snapshots have nonnegative contrasts.
    """
    if prediction.ndim != 5 or prediction.shape[1] != 2:
        raise ValueError("Expected [B,2,Z,Y,X] physical property maps")
    if foreground.shape != (prediction.shape[0],1,*prediction.shape[2:]):
        raise ValueError("Foreground must be [B,1,Z,Y,X]")
    if not math.isfinite(smooth) or smooth <= 0:
        raise ValueError("Dice smoothing must be finite and positive")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        value = prediction.float()
        offset = value.new_tensor(f.PROPERTY_OFFSET).reshape(1,2,1,1,1)
        scale = value.new_tensor(f.PROPERTY_SCALE).reshape(1,2,1,1,1)
        p = (value-offset)/scale
        target = foreground.to(device=value.device, dtype=torch.bool).expand_as(p).float()
        dims = (2,3,4)
        intersection = (p*target).sum(dims)
        denominator = p.square().sum(dims)+target.square().sum(dims)
        return 1-(2*intersection+smooth)/(denominator+smooth)


def soft_extent_loss(prediction, foreground, smooth=1e-6):
    return extent_per_property(prediction, foreground, smooth).mean()


@contextlib.contextmanager
def augmented_objective(weight, smooth):
    """Scoped trainer hook; no changes to original source or other processes."""
    old = f._objective
    def objective(outputs, target, foreground, edge_weight=.01):
        base = BASE_OBJECTIVE(outputs,target,foreground,edge_weight)
        return base + weight*soft_extent_loss(outputs["refined"],foreground,smooth)
    f._objective = objective
    try:
        yield
    finally:
        f._objective = old


def load_complete(path, device="cpu"):
    state=torch.load(path,map_location="cpu",weights_only=True)
    identity=state["identity"]
    if f.canonical_hash(identity) != state["identity_sha256"]:
        raise ValueError("Checkpoint identity is invalid")
    if (identity["splits"] != SPLITS or identity["k"] != 10 or identity["seed"] != 17
            or state.get("epoch") != 300 or state.get("optimizer_updates") != 1500
            or identity["config"]["grid_size"] != 128 or identity["indices_zero_based"] != list(range(10))
            or state.get("checkpoint_policy") != "fixed_final_epoch" or identity.get("test_accessed") is not False):
        raise ValueError("Completed native128 K10 seed17 validation-only weights required")
    if state["architecture"] != identity["config"]["model"] or state["architecture"].get("postnorm_amplitude"):
        raise ValueError("Use the original pattern-v1 architecture, without postnorm additions")
    if identity["source"] != f.source_identity():
        raise ValueError("Package source mismatch")
    for name,digest in identity["config"]["experimental_source"].items():
        if f.file_sha256(Path(__file__).parent/name) != digest:
            raise ValueError(f"Experimental source mismatch: {name}")
    model=PatternAmplitudeMSRNN(state["architecture"]).to(device).eval()
    if model.coarse.output_size != 128 or sum(p.numel() for p in model.parameters()) != state["parameter_count"]:
        raise ValueError("Native grid or parameter count mismatch")
    model.load_state_dict(state["model"],strict=True)
    return state,model


class Experiment(PatternExperiment):
    def __init__(self,config,mat,csv,output,device,*,baseline,mechanism_fixture=False):
        super().__init__(config,mat,csv,output,device)
        reference,_=load_complete(baseline)
        old=reference["identity"]["config"]
        if old.get("extent_loss") or old.get("targeted_adjustment"):
            raise ValueError("Reference must be the unchanged original pattern-v1 run")
        spec=config.get("extent_loss",{})
        if (spec.get("name") != LOSS_NAME or spec.get("stage") != "refined"
                or config["seeds"] != [17] or config["model"].get("postnorm_amplitude")):
            raise ValueError("Use the explicit loss-only, refined-stage, K10 seed17 protocol")
        weight,smooth=float(spec.get("weight",0)),float(spec.get("smooth",0))
        if not all(math.isfinite(x) and x>0 for x in (weight,smooth)):
            raise ValueError("Positive finite Dice weight and smoothing required")
        equal=("batch_size","learning_rate","weight_decay","gradient_weight","gradient_clip","amp","expanded_protocol")
        if not mechanism_fixture:
            equal+=("epochs","patience","grid_size","model")
            if config["epochs"] != 300 or config["grid_size"] != 128:
                raise ValueError("Preserve native128 and the 300-epoch budget")
        for key in equal:
            if self.config[key] != old[key]:
                raise ValueError(f"Baseline architecture/training protocol differs: {key}")
        if self.output.resolve().is_relative_to(Path(baseline).resolve().parent):
            raise ValueError("Use a new output folder outside the baseline run")
        if (self.output/"K10/seed_17/final.pt").resolve() == Path(baseline).resolve():
            raise ValueError("Baseline output cannot be overwritten")
        self.baseline,self.weight,self.smooth=str(baseline),weight,smooth
        self.config["experimental_source"][Path(__file__).name]=f.file_sha256(__file__)
        self.config["loss_experiment"]={"name":LOSS_NAME,"baseline_checkpoint_sha256":f.file_sha256(baseline),
            "mechanism_fixture":bool(mechanism_fixture),"initialization":"from_scratch_paired_seed17",
            "architecture":"unchanged_pattern_v1","base_objective":"unchanged_weighted_property_and_gradient"}

    def metrics(self,model,samples,preprocessing,device):
        model.eval();rows=[]
        with torch.inference_mode():
            for sample in samples:
                output=f._predict(model,sample,preprocessing,device)
                target=sample["target"].unsqueeze(0).to(device)
                foreground=sample["object_mask"].unsqueeze(0).to(device)
                base=BASE_OBJECTIVE(output,target,foreground,self.config["gradient_weight"])
                extent=extent_per_property(output["refined"],foreground,self.smooth)[0]
                rows.append({"case_id":sample["metadata"]["case_id"],
                    **{key:without_score(f.reconstruction_metrics(output[key][0],sample["target"])) for key in ("coarse","refined")},
                    "evaluation_objective_float32":float(base),
                    "extent_loss_per_property":{"epsilon":float(extent[0]),"sigma":float(extent[1])},
                    "extent_loss":float(extent.mean()),
                    "training_objective_float32":float(base+self.weight*extent.mean())})
        return {"coarse":f.average_metrics([r["coarse"] for r in rows]),
                "refined":f.average_metrics([r["refined"] for r in rows]),"per_case":rows,
                **{key:sum(r[key] for r in rows)/len(rows) for key in
                   ("evaluation_objective_float32","extent_loss","training_objective_float32")}}

    def evaluate(self,model,samples,preprocessing,device):
        result=super().evaluate(model,samples,preprocessing,device)
        print("  "+"; ".join(f"case{row['case_id']} volume={row['refined']['predicted_volume_mm3']/1000:.2f}/"
              f"{row['refined']['true_volume_mm3']/1000:.2f}ml centre={row['refined']['localization_error_mm']:.2f}mm"
              for row in result["per_case"]),flush=True)
        return result

    def run(self,counts=(10,),*,seeds=None,**kwargs):
        if list(counts) != [10] or (seeds is not None and list(seeds) != [17]):
            raise ValueError("This loss experiment trains only K10 seed17")
        with augmented_objective(self.weight,self.smooth):
            return super().run(counts,seeds=[17],**kwargs)


def compare(experiment,output):
    base_state,base_model=load_complete(experiment.baseline,experiment.device)
    path=experiment.output/"K10/seed_17/final.pt"
    candidate,new_model=load_complete(path,experiment.device)
    if candidate["identity"]["config"] != experiment.config:
        raise ValueError("Candidate identity differs from the requested loss experiment")
    if candidate["architecture"] != base_state["architecture"]:
        raise ValueError("A loss-only comparison requires identical architectures")
    for key in ("data_sha256","csv_sha256","splits","preprocessing","k","seed"):
        if candidate["identity"][key] != base_state["identity"][key]:
            raise ValueError(f"Comparison lineage differs: {key}")
    report={"baseline_sha256":f.file_sha256(experiment.baseline),"candidate_sha256":f.file_sha256(path),
            "objective_change":experiment.config["extent_loss"],"architecture_unchanged":True,
            "loss_reporting_note":"Comparison training_objective_float32 evaluates the candidate formula on both frozen models; the baseline was trained without Dice.",
            "baseline_execution":base_state["identity"]["execution"],
            "candidate_execution":candidate["identity"]["execution"],"test_accessed":False,"per_case":[]}
    for split in ("train","validation"):
        for sample in experiment.dataset(experiment.mat,split,range(10),experiment.config):
            row={"case_id":sample["metadata"]["case_id"],"split":split}
            for name,model,state in (("baseline",base_model,base_state),("candidate",new_model,candidate)):
                m=experiment.metrics(model,[sample],state["preprocessing"],experiment.device)["per_case"][0]
                row[name]=m
            report["per_case"].append(row)
    atomic_write(output,report)
    for row in report["per_case"]:
        if row["split"] != "validation":continue
        print(f"VALIDATION case{row['case_id']}")
        for name in ("baseline","candidate"):
            m=row[name]["refined"]
            print(f"  {name}: centre={m['localization_error_mm']:.3f}mm, volume={m['predicted_volume_mm3']/1000:.3f}/"
                  f"{m['true_volume_mm3']/1000:.3f}ml, epsilon={m['epsilon_rmse']:.4f}, sigma={m['sigma_rmse_s_per_m']:.5f}")
    return report


def profile(config,device,backward=False):
    f.seed_everything(17);model=PatternAmplitudeMSRNN(config["model"]).to(device)
    model.train(backward);cuda=str(device).startswith('cuda');amp=bool(config['amp'] and cuda)
    signal=torch.randn(1,2,10,18,18,device=device)
    mask=torch.ones(1,1,10,18,18,dtype=torch.bool,device=device)
    n=model.coarse.output_size;axis=torch.arange(n,device=device)-(n-1)/2
    foreground=((axis[:,None,None]**2+axis[None,:,None]**2+axis[None,None,:]**2)<(n/6)**2)[None,None]
    target=foreground.float()*signal.new_tensor(f.PROPERTY_SCALE).reshape(1,2,1,1,1)+signal.new_tensor(f.PROPERTY_OFFSET).reshape(1,2,1,1,1)
    if cuda:torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
    started=time.perf_counter()
    with torch.set_grad_enabled(backward),torch.autocast('cuda' if amp else 'cpu',enabled=amp):
        output=model(signal,mask)
        base=BASE_OBJECTIVE(output,target,foreground,config['gradient_weight'])
        extent=soft_extent_loss(output['refined'],foreground,config['extent_loss']['smooth'])
        total=base+config['extent_loss']['weight']*extent
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite objective')
    if backward:
        total.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradients')
    if cuda:torch.cuda.synchronize(device)
    return {'scope':'synthetic native objective verification; no accuracy evidence','torch':str(torch.__version__),
            'device':device,'amp':amp,'backward':backward,'shape':list(output['refined'].shape),
            'parameters':sum(p.numel() for p in model.parameters()),'base_loss':float(base.detach()),
            'extent_loss':float(extent.detach()),'total_loss':float(total.detach()),
            'seconds':time.perf_counter()-started,
            'peak_cuda_bytes':torch.cuda.max_memory_allocated(device) if cuda else None}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('run','compare','profile'))
    p.add_argument('--config',default='configs/frequency_pilot_extent_v1.json')
    p.add_argument('--mat',default='data/pilot_expanded/frequency_pilot.mat')
    p.add_argument('--csv',default='data/pilot/frequency_subsets.csv')
    p.add_argument('--output',default='outputs/frequency_extent_v1')
    p.add_argument('--baseline',default='outputs/frequency_pattern_v1/K10/seed_17/final.pt')
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-updates',type=int)
    p.add_argument('--backward',action='store_true')
    args=p.parse_args(argv);config=json.loads(Path(args.config).read_text())
    if args.command=='profile':result=profile(config,args.device,args.backward)
    else:
        experiment=Experiment(config,args.mat,args.csv,args.output,args.device,baseline=args.baseline)
        if args.command=='run':
            results=experiment.run(resume=args.resume,max_updates=args.max_updates)
            if results[0]['status']!='completed':print(json.dumps(results,indent=2));return results
        result=compare(experiment,Path(args.output)/'comparison.json')
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=='__main__':main()
