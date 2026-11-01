"""Real-data stage training, deterministic interruption recovery and bounded profiling.

No command synthesizes production inputs or starts COMSOL. Physics B is a training
loss only. C explicitly adds an independently qualified inference dependency.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .checkpoint import identity_hash, load_checkpoint, save_checkpoint, atomic_torch_save

STAGES = ("gan", "phi", "coarse", "enhance", "physics", "segmentation", "joint")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_batch(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch(v, device) for v in value)
    if isinstance(value, list):
        return [move_batch(v, device) for v in value]
    return value


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _collate(samples):
    from .data import collate_samples
    return collate_samples(samples)


def epoch_batches(dataset, batch_size, seed, epoch, shuffle=True):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(dataset), generator=generator).tolist() if shuffle else list(range(len(dataset)))
    for start in range(0, len(order), batch_size):
        yield _collate([dataset[i] for i in order[start:start + batch_size]])


def _first(value, default=None):
    if value is None:
        return default
    return value[0] if isinstance(value, list) and value else value


def _context(batch):
    value = batch.get("physics_context")
    if value is None:
        raise ValueError("This physics path needs source/layout/calibration context in the export")
    return value


def prepare_batch(batch, preprocessing, config):
    """Keep original physical observations separate from network normalization."""
    batch = dict(batch)
    monitoring = config.get("mode") in ("monitoring", "baseline_tracking")
    if monitoring:
        if "difference" not in batch or "difference_mask" not in batch:
            raise ValueError("Monitoring requires registered current-minus-baseline measurements/masks")
        batch["signals"], batch["mask"] = batch["difference"], batch["difference_mask"]
        batch["domain"] = "current_minus_baseline"
    mask = batch["mask"].bool()
    selected = config.get("frequency_config")
    if selected:
        slots = torch.zeros(mask.shape[2], dtype=torch.bool, device=mask.device)
        slots[selected["indices_zero_based"]] = True
        mask = mask & slots[None, None, :, None, None]
    batch["mask"] = mask
    batch["original_calibrated"] = torch.where(mask, batch["signals"], torch.zeros_like(batch["signals"]))
    batch["signals"] = batch["original_calibrated"]
    if preprocessing:
        from .data import ObservedScale
        scale = ObservedScale.from_state_dict(preprocessing)
        batch["signals"] = scale.transform(batch["original_calibrated"], mask)
        for key in ("clean_signals", "pair_signals", "pair_clean_signals"):
            if key in batch:
                batch[key] = scale.transform(batch[key], mask)
    if config.get("physics_mode") == "C":
        context = dict(_context(batch))
        context["measurement_scale"] = preprocessing["scale"] if preprocessing else 1.
        context["observation_domain"] = "current_minus_baseline" if monitoring else config.get("physics", {}).get("domain", "calibrated_delta")
        if "baseline_spectral" in batch:
            context["baseline_spectral"] = batch["baseline_spectral"]
        batch["physics_context"] = context
    return batch


def forward_inverse(model, batch, config, phi=None):
    return model(batch["signals"], batch["mask"],
                 frequencies_ghz=batch.get("frequencies_ghz"),
                 baseline=batch.get("baseline"), roi=batch.get("roi"),
                 context_validity=batch.get("context_validity"),
                 mode=config.get("mode", "snapshot"), physics_mode=config.get("physics_mode", "A"),
                 phi=phi, physics_context=batch.get("physics_context"),
                 physics_observations=batch.get("original_calibrated"))


def load_phi(config, device, *, required=True):
    """Load a separately trained operator and gate; never produce substitute weights."""
    from .physics import HeadsetForwardOperator, PhysicsConsistency
    spec = config.get("physics", {})
    path, gate = spec.get("checkpoint"), spec.get("qualification")
    if not path or not gate:
        if required:
            raise ValueError("Qualified physics explicitly requested but Phi checkpoint/qualification is absent; use A")
        return None, None
    operator = HeadsetForwardOperator(**spec.get("operator", {})).to(device)
    state = load_checkpoint(path, restore_random=False, map_location=device)
    weights = state["model"]
    if any(k.startswith("phi.") for k in weights):
        weights = {k[len("phi."):]: v for k, v in weights.items() if k.startswith("phi.")}
    operator.load_state_dict(weights)
    report = json.loads(Path(gate).read_text())
    loss = PhysicsConsistency(operator, report, domain=spec.get("domain", "calibrated_delta"), require_qualified=True)
    operator.qualification_report = report
    operator.qualified = True
    return operator, loss


class StageSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        from .models import MSRNN, SegmentationUNet
        self.stage = config["stage"]
        if self.stage == "phi":
            from .physics import HeadsetForwardOperator
            self.phi = HeadsetForwardOperator(**config.get("physics", {}).get("operator", {}))
        elif self.stage == "gan":
            from .gan import CycleGAN3D
            self.gan = CycleGAN3D(config.get("gan", {}))
        else:
            self.inverse = MSRNN(config.get("model", {}))
            if self.stage in ("segmentation", "joint"):
                self.segmentation = SegmentationUNet(**config.get("segmentation", {}))



def attach_processing(system, config, device, *, initialize):
    """Embed trained G and fitted bases in the stage checkpoint; never random routes."""
    processing = config.setdefault("processing", {})
    if not hasattr(system, "inverse"):
        return
    if processing.get("gan_checkpoint") or processing.get("gan_config"):
        from .gan import CycleGAN3D
        prior = None
        if initialize:
            if not processing.get("gan_checkpoint"):
                raise ValueError("Enabled decluttering needs a trained GAN checkpoint")
            prior = load_checkpoint(processing["gan_checkpoint"], restore_random=False, map_location=device)
            if prior["metadata"]["stage"] != "gan":
                raise ValueError("Declutter source is not a trained GAN stage")
            processing["gan_config"] = prior["metadata"]["config"].get("gan", {})
            processing["gan_sha256"] = file_hash(processing["gan_checkpoint"])
            current_scale = config["preprocessing"]["scale"]
            if abs(prior["metadata"]["preprocessing"]["scale"] / current_scale - 1) > 1e-6:
                raise ValueError("GAN and inverse must use the same saved measurement scale")
        gan = CycleGAN3D(processing["gan_config"]).to(device)
        if prior:
            weights = {k[len("gan."):]: v for k, v in prior["model"].items() if k.startswith("gan.")}
            gan.load_state_dict(weights)
        system.declutterer = gan.G
        system.declutterer.requires_grad_(False)
        system.declutterer.eval()
    if processing.get("subspace_path") or processing.get("subspace_state"):
        from .subspace import MaskedSubspace
        if initialize:
            subspace = MaskedSubspace.load(processing["subspace_path"])
            processing["subspace_state"] = {"rank": subspace.rank, "strength": subspace.strength, "ridge": subspace.ridge,
                                            "context": subspace.context, "real": subspace.basis.real.tolist(), "imag": subspace.basis.imag.tolist(),
                                            "observed_support": subspace.observed_support.tolist()}
            processing["subspace_sha256"] = file_hash(processing["subspace_path"])
        else:
            value = processing["subspace_state"]
            subspace = MaskedSubspace(value["rank"], value["strength"], value["ridge"])
            subspace.context = value["context"]
            subspace.basis = np.asarray(value["real"]) + 1j * np.asarray(value["imag"])
            subspace.observed_support = np.asarray(value["observed_support"],dtype=bool)
        system.subspace = subspace
    processing["trained_routing"] = {"declutterer": hasattr(system, "declutterer"), "subspace": hasattr(system, "subspace")}
    if config.get("mode") in ("monitoring", "baseline_tracking") and any(processing["trained_routing"].values()):
        raise ValueError("Default baseline monitoring bypasses GAN/subspace; train a separately specified mode to change it")


def build_deployment_pipeline(system, config, phi=None, batch=None):
    from .pipeline import MicroScanPipeline
    processing = config.get("processing", {})
    routing = processing.get("trained_routing", {"declutterer": False, "subspace": False})
    return MicroScanPipeline(system.inverse, getattr(system, "segmentation", None), getattr(system, "declutterer", None),
                             trained_declutterer=hasattr(system, "declutterer"), subspace=getattr(system, "subspace", None),
                             physics_mode=config.get("physics_mode", "A"), phi=phi,
                             physics_context=batch.get("physics_context") if batch else None,
                             global_refresh_interval=config.get("global_refresh_interval", 10), trained_routing=routing,
                             subspace_context=processing.get("subspace_state", {}).get("context"),
                             signal_scale=config.get("preprocessing",{}).get("scale",1.))


def execute_pipeline(pipeline, batch, config, run_segmentation=True):
    routing = config.get("processing", {}).get("trained_routing", {"declutterer": False, "subspace": False})
    return pipeline(batch["signals"], batch["mask"], batch.get("frequencies_ghz"), baseline=batch.get("baseline"),
                    roi=batch.get("roi"), context_validity=batch.get("context_validity"), mode=config.get("mode", "snapshot"),
                    enable_declutter=routing["declutterer"], enable_subspace=routing["subspace"], run_segmentation=run_segmentation,
                    frame_index=config.get("frame_index", 1), physics_observations=batch.get("original_calibrated"))


def _initialize_stage(system, config, device):
    checkpoint = config.get("initialize_from")
    if checkpoint:
        prior = load_checkpoint(checkpoint, restore_random=False, map_location=device)["model"]
        wanted = system.state_dict()
        compatible = {k: v for k, v in prior.items() if k in wanted and v.shape == wanted[k].shape}
        if not compatible:
            raise ValueError("Initialization checkpoint contains no compatible model tensors")
        if hasattr(system, "inverse"):
            required = {"inverse." + k for k in system.inverse.state_dict()}
            if not required.issubset(compatible):
                raise ValueError("Initialization must contain the complete compatible inverse; partial weights could leave random deployment layers")
            prior_config = load_checkpoint(checkpoint, restore_random=False, map_location="cpu")["metadata"]["config"]
            if prior_config.get("model", {}) != config.get("model", {}):
                raise ValueError("Initialization architecture/frequency configuration mismatch")
            if prior_config.get("preprocessing") != config.get("preprocessing") and not config.get("allow_preprocessing_refit", False):
                raise ValueError("Initialization preprocessing differs; preserve prior scale or explicitly validate allow_preprocessing_refit")
        system.load_state_dict(compatible, strict=False)
    if config["stage"] == "segmentation" and not checkpoint:
        raise ValueError("Segmentation training requires initialize_from with trained inverse predictions, not random maps")
    if config["stage"] == "coarse":
        for name, parameter in system.inverse.named_parameters():
            parameter.requires_grad_(name.startswith("coarse."))
    elif config["stage"] == "segmentation":
        system.inverse.requires_grad_(False)
        system.inverse.eval()
    elif config["stage"] == "joint" and not config.get("freeze_schedule"):
        raise ValueError("Joint training requires an explicit freeze_schedule")


def _joint_schedule(system, config, epoch):
    if config["stage"] != "joint":
        return
    entries = [x for x in config["freeze_schedule"] if int(x["epoch"]) <= epoch]
    if not entries:
        raise ValueError("Joint freeze_schedule must begin at epoch zero")
    item = max(entries, key=lambda x: x["epoch"])
    for name in ("inverse", "segmentation"):
        trainable = name in item["train"]
        getattr(system, name).requires_grad_(trainable)
        getattr(system, name).train(trainable)


def compute_stage_loss(system, batch, config, consistency=None, phi=None):
    from .losses import reconstruction_loss, segmentation_loss
    stage = config["stage"]
    if stage == "phi":
        from .physics import masked_complex_loss
        target = batch.get("physics_observations", batch.get("original_calibrated", batch["signals"]))
        # Spectral labels come from material library, never from invented constant head dispersion.
        properties = batch.get("spectral_target")
        if properties is None:
            if not config.get("physics", {}).get("nondispersive_labels_verified", False):
                raise ValueError("Phi training requires spectral_target or verified nondispersive material labels")
            properties = batch["target"]
        prediction = system.phi(properties, batch["frequencies_ghz"], _context(batch))
        loss = masked_complex_loss(prediction, target, batch["mask"])
        if "baseline_target" in batch and "baseline_observations" in batch:
            baseline_properties = batch.get("baseline_spectral_target",batch["baseline_target"])
            baseline = system.phi(baseline_properties.detach(), batch["frequencies_ghz"], _context(batch))
            loss = loss + masked_complex_loss(prediction - baseline, target - batch["baseline_observations"], batch.get("baseline_mask", batch["mask"]) & batch["mask"].bool())
        return {"total": loss, "data": loss}
    pipeline = build_deployment_pipeline(system, config, phi, batch)
    outputs = execute_pipeline(pipeline, batch, config, run_segmentation=False)
    terms = {}
    if stage != "segmentation":
        data_loss = None
        if consistency is not None:
            from .physics import normalize_domain
            observed_domain = batch.get("domain", config.get("physics", {}).get("domain", "calibrated_delta"))
            expected_domain = "current_minus_baseline" if config.get("mode") in ("monitoring","baseline_tracking") else consistency.domain
            if normalize_domain(observed_domain) != normalize_domain(expected_domain):
                raise ValueError("Physical consistency requires original observations in the operator's explicit domain")
            if "spectral_target" not in batch and not config.get("physics",{}).get("nondispersive_labels_verified",False) and not consistency.allow_fixture:
                raise ValueError("Scientific physics requires supervised frequency-resolved material labels")
            spectral = system.inverse.spectral_properties(outputs, batch["frequencies_ghz"])
            data_loss = consistency(spectral, batch.get("original_calibrated", batch["signals"]), batch["mask"],
                                    batch["frequencies_ghz"], _context(batch),
                                    baseline_properties=batch.get("baseline_spectral_target",batch.get("baseline")) if config.get("mode") in ("monitoring","baseline_tracking") else None,
                                    baseline_valid=batch.get("baseline_valid"), noise_scale=batch.get("noise_scale"),
                                    covariance=batch.get("noise_covariance"),
                                    observation_domain="current_minus_baseline" if config.get("mode") in ("monitoring", "baseline_tracking") else config.get("physics", {}).get("domain", "calibrated_delta"))
            if "withheld_mask" in batch and config.get("acquisition", {}).get("heldout_reprojection_weight",0):
                heldout_loss = consistency(spectral, batch["original_calibrated"],batch["withheld_mask"],batch["frequencies_ghz"],_context(batch),
                                          baseline_properties=batch.get("baseline_spectral_target"),baseline_valid=batch.get("baseline_valid"),
                                          observation_domain=expected_domain)
                data_loss = data_loss + config["acquisition"]["heldout_reprojection_weight"] * heldout_loss
        elif config.get("acquisition", {}).get("heldout_reprojection_weight",0):
            raise ValueError("Held-out physical reprojection requires independently qualified Phi; A remains supervised")
        loss_outputs = dict(outputs)
        baseline_target = batch.get("baseline_target", batch.get("baseline"))
        if config.get("mode") in ("monitoring", "baseline_tracking"):
            if baseline_target is None:
                raise ValueError("Signed monitoring training requires current and baseline property labels")
            loss_outputs["coarse"] = outputs["coarse"] + baseline_target
            loss_outputs["refined"] = outputs["refined"] + baseline_target
        spectral_target = batch.get("spectral_target")
        spectral_prediction = system.inverse.spectral_properties(outputs, batch["frequencies_ghz"]) if spectral_target is not None else None
        if spectral_prediction is not None and config.get("mode") in ("monitoring", "baseline_tracking"):
            if "baseline_spectral_target" not in batch:
                raise ValueError("Monitoring spectral supervision needs baseline spectral labels")
            spectral_prediction = spectral_prediction + batch["baseline_spectral_target"]
        terms = reconstruction_loss(loss_outputs, batch["target"], foreground=batch.get("object_mask"),
                                    baseline_target=baseline_target, stable_mask=batch.get("stable_mask"),
                                    physics_loss=data_loss, weights=config.get("loss_weights"),
                                    property_scales=config.get("property_scales", (44., .8)), validity=batch.get("target_validity"),
                                    spectral_target=spectral_target, spectral_prediction=spectral_prediction)
    if stage in ("segmentation", "joint"):
        if "labels" not in batch or "label_validity" not in batch:
            raise ValueError("Segmentation needs clinical labels and explicit label_validity; missing labels are not background")
        absolute_valid = outputs["absolute_valid"][:,None,None,None,None]
        absolute = torch.where(absolute_valid, outputs["absolute"], torch.zeros_like(outputs["absolute"]))
        logits = system.segmentation(absolute, baseline=batch.get("baseline"), difference=outputs.get("change"),
                                     roi=batch.get("roi"), detach=stage == "segmentation")
        label_validity = batch["label_validity"]
        valid = absolute_valid if label_validity.ndim == 5 else absolute_valid[:,0]
        seg_loss = segmentation_loss(logits, batch["labels"], label_validity & valid, schema=config.get("segmentation", {}).get("label_mode", "multiclass"))
        terms["segmentation"] = seg_loss
        terms["total"] = terms.get("total", seg_loss * 0) + config.get("segmentation_weight", 1.) * seg_loss
    return terms



def apply_structured_holdout(batch, config):
    spec = config.get("acquisition", {}).get("structured_tx_holdout")
    if not spec:
        return batch
    from .acquisition import structured_tx_holdout
    result = dict(batch)
    if isinstance(spec, int):
        available = batch["mask"].any(dim=(0,1,2,3)).nonzero().flatten()
        if not 0 < spec < len(available):
            raise ValueError("Tx holdout must retain at least one observed training transmitter")
        indices = available[torch.randperm(len(available),device=available.device)[:spec]].tolist()
    else:
        indices = list(spec)
    retained, withheld = structured_tx_holdout(batch["mask"], indices)
    result["mask"] = retained
    result["withheld_mask"] = withheld
    result["signals"] = torch.where(retained, batch["signals"], torch.zeros_like(batch["signals"]))
    return result


def validate_gan(system, dataset, config, device):
    from .gan import masked_l1
    system.eval()
    values = []
    with torch.no_grad():
        for sample in epoch_batches(dataset, config.get("batch_size",1), config.get("seed",42), 0, False):
            batch = prepare_batch(move_batch(sample,device), config.get("preprocessing"),config)
            if "clean_signals" not in batch:
                raise ValueError("GAN validation needs declared held-out X/Y domains")
            mask = batch["mask"].bool() & batch.get("clean_mask",batch["mask"]).bool()
            x,y = batch["signals"],batch["clean_signals"]
            fake_y = system.gan.G(x,mask,batch["frequencies_ghz"])
            fake_x = system.gan.F(y,mask,batch["frequencies_ghz"])
            loss = masked_l1(system.gan.F(fake_y,mask,batch["frequencies_ghz"]),x,mask)
            loss = loss + masked_l1(system.gan.G(fake_x,mask,batch["frequencies_ghz"]),y,mask)
            if config.get("gan_paired",False):
                loss = loss + masked_l1(fake_y,y,mask)
            values.append(float(loss))
    if not values:
        raise ValueError("GAN validation split is empty")
    return sum(values)/len(values)


def validate_epoch(system, dataset, config, device, consistency=None, phi=None):
    system.eval()
    values = []
    with torch.no_grad():
        for batch in epoch_batches(dataset, config.get("batch_size", 1), config.get("seed", 42), 0, shuffle=False):
            batch = prepare_batch(move_batch(batch, device), config.get("preprocessing"), config)
            terms = compute_stage_loss(system, batch, config, consistency, phi)
            values.append(float(terms["total"].detach().cpu()))
    if not values:
        raise ValueError("Validation split is empty")
    return sum(values) / len(values)


def train(config, *, resume=False, max_steps=None):
    from .data import ManifestDataset, ObservedScale
    config = dict(config)
    stage = config.get("stage")
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    requested = config.get("physics_mode", "A")
    if requested not in ("A", "B", "C"):
        raise ValueError("physics_mode must be A/B/C")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no GPU validation was performed")
    if not config.get("frequency_config_path") and not config.get("verification_only",False):
        raise ValueError("Main training requires frequency_config_path with an experimentally frozen selection")
    if config.get("frequency_config_path"):
        frozen = json.loads(Path(config["frequency_config_path"]).read_text())
        if frozen.get("status") != "frozen":
            raise ValueError("Main training waits for an experimentally frozen frequency configuration")
        indices = frozen.get("indices_zero_based",[])
        grid = frozen.get("grid_ghz",[])
        if (not grid or not indices or len(set(indices)) != len(indices) or min(indices)<0 or max(indices)>=len(grid)
                or frozen.get("selected_k") != len(indices) or frozen.get("frequencies_ghz") != [grid[i] for i in indices]
                or not frozen.get("selection_sha256")):
            raise ValueError("Invalid frozen frequency selection schema/provenance")
        config["frequency_config"] = frozen
        model_config = dict(config.get("model", {}))
        model_config["frequency_grid_ghz"] = frozen["grid_ghz"]
        config["model"] = model_config
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    dataset = ManifestDataset(config["manifest"], split="train")
    validation = ManifestDataset(config["manifest"], split="validation")
    if not len(dataset):
        raise ValueError("Training split is empty")
    if config.get("verification_only",False) and max_steps is None:
        if not all("fixture" in s["source_domain"].lower() for s in dataset.samples) or config.get("epochs",1)>1:
            raise ValueError("verification_only requires --max-steps or a one-epoch explicit mechanism fixture")
    if not config.get("preprocessing"):
        def fit_samples():
            for sample in dataset:
                # Keep geometry/split metadata while fitting only selected observed support.
                batch_sample = dict(sample)
                if config.get("mode") in ("monitoring","baseline_tracking"):
                    if "difference" not in sample:
                        raise ValueError("Monitoring preprocessing requires registered differences")
                    batch_sample["signals"], batch_sample["mask"] = sample["difference"], sample["difference_mask"]
                if config.get("frequency_config"):
                    slots = torch.zeros(batch_sample["mask"].shape[1], dtype=torch.bool)
                    slots[config["frequency_config"]["indices_zero_based"]] = True
                    batch_sample["mask"] = batch_sample["mask"].bool() & slots[None,:,None,None]
                yield batch_sample
        config["preprocessing"] = ObservedScale.fit(fit_samples()).state_dict()
    system = StageSystem(config).to(device)
    if resume:
        saved = load_checkpoint(Path(config.get("output",f"outputs/{stage}"))/"last.pt",restore_random=False)
        config["processing"] = saved["metadata"]["config"].get("processing",{})
        attach_processing(system, config, device, initialize=False)
        # Reapply freeze policy without reopening initial source checkpoint files.
        freeze_config = dict(config)
        freeze_config.pop("initialize_from",None)
        if stage == "coarse":
            for name,p in system.inverse.named_parameters(): p.requires_grad_(name.startswith("coarse."))
        elif stage == "segmentation":
            system.inverse.requires_grad_(False); system.inverse.eval()
    else:
        attach_processing(system, config, device, initialize=True)
        _initialize_stage(system, config, device)
    phi, consistency = (None, None)
    if requested in ("B", "C") and stage not in ("gan", "phi"):
        phi, consistency = load_phi(config, device, required=True)
    data_fingerprints = {}
    for sample in dataset.samples + validation.samples:
        for name, value in sample.items():
            if name.endswith("_path") and value:
                referenced = Path(config["manifest"]).parent / value
                data_fingerprints[str(referenced)] = file_hash(referenced)
        for value in sample.get("physics_context_paths", {}).values():
            referenced = Path(config["manifest"]).parent / value
            data_fingerprints[str(referenced)] = file_hash(referenced)
    metadata = {"data_sha256": data_fingerprints, "config": json.loads(json.dumps(config)), "architecture": config.get("model", {}), "split": {"train": [s["geometry_id"] for s in dataset.samples], "validation": [s["geometry_id"] for s in validation.samples]},
                "manifest_sha256": file_hash(config["manifest"]), "source_sha256": source_fingerprint(),
                "preprocessing": config.get("preprocessing", {}), "frequency_config": config.get("frequency_config"),
                "mask_policy": config.get("acquisition", {}), "physics_requested": requested,
                "physics_active": requested if consistency is not None else "A", "physics_provenance": config.get("physics", {}),
                "inference_requirements": {"phi_required": requested == "C"}, "stage": stage}
    metadata["run_identity"] = identity_hash(metadata)
    output = Path(config.get("output", f"outputs/{stage}"))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "last.pt"
    start_epoch = start_step = global_step = bad_epochs = 0
    best = math.inf
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    optimizer = None
    gan_trainer = None
    if stage == "gan":
        from .gan import GANTrainer
        gan_kwargs = dict(config.get("gan_training",{}))
        gan_kwargs.setdefault("amp",amp)
        gan_trainer = GANTrainer(system.gan, **gan_kwargs)
        optimizers = {"generator":gan_trainer.optimizer_G,"discriminators":gan_trainer.optimizer_D}
        schedulers = {name:torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,int(config.get("epochs",100)))) for name,opt in optimizers.items()}
    else:
        # Include temporarily frozen joint parameters so schedule changes need no optimizer reset.
        parameters = list(system.parameters()) if stage == "joint" else [p for p in system.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("Stage freeze policy left no trainable parameters")
        optimizer = torch.optim.AdamW(parameters, lr=config.get("learning_rate", 2e-4), weight_decay=config.get("weight_decay", 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config.get("epochs", 100))))
        optimizers, schedulers = {"stage": optimizer}, {"stage": scheduler}
    if resume:
        state = load_checkpoint(path, system, optimizers, schedulers, scaler, expected_identity=metadata["run_identity"], map_location=device)
        start_epoch, start_step, global_step = state["epoch"], state["step_in_epoch"], state["global_step"]
        best, bad_epochs = state["best"], state["bad_epochs"]
        if gan_trainer is not None:
            gan_trainer.load_state_dict(state["extra"]["gan_trainer"])
    elif path.exists():
        raise FileExistsError("Existing training state: use --resume or a new output directory")
    if config.get("acquisition_augmentation"):
        from .data import TrainingAugmentation
        dataset = TrainingAugmentation(dataset, config["acquisition_augmentation"], seed)
    history_path = output / "history.jsonl"
    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 1))
    steps_this_call = 0
    for epoch in range(start_epoch, epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        system.train()
        if hasattr(system, "declutterer"):
            system.declutterer.eval()
        if stage == "segmentation":
            system.inverse.eval()
        _joint_schedule(system, config, epoch)
        if consistency is not None:
            weights = dict(config.get("loss_weights", {}))
            weights["data"] = config.get("physics", {}).get("data_weight", 1.) * min(1., (epoch + 1) / max(1, config.get("physics", {}).get("ramp_epochs", 10)))
            config["loss_weights"] = weights
        last_step = 0
        for index, batch in enumerate(epoch_batches(dataset, batch_size, seed, epoch)):
            last_step = index + 1
            if epoch == start_epoch and index < start_step:
                continue
            batch = prepare_batch(move_batch(batch, device), config.get("preprocessing"), config)
            if not batch["mask"].any():
                raise ValueError("All-missing training acquisition has no usable supervised observation support")
            if gan_trainer is not None:
                if "clean_signals" not in batch:
                    raise ValueError("GAN manifest needs a declared clean/unpaired Y domain as clean_signals")
                if config.get("gan_paired",False) and not all(m.get("clean_pairing",{}).get("aligned") and m.get("clean_pairing",{}).get("pair_id") for m in batch["metadata"]):
                    raise ValueError("Paired GAN supervision requires verified aligned clean_pairing metadata")
                result = gan_trainer.step(batch["signals"], batch["clean_signals"], batch["mask"], batch.get("clean_mask", batch["mask"]),
                                          batch.get("frequencies_ghz"), paired=bool(config.get("gan_paired", False)),
                                          pair_x=batch.get("pair_signals"), pair_y=batch.get("pair_clean_signals"))
                terms = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in result.items()}
            else:
                optimizer.zero_grad(set_to_none=True)
                batch = apply_structured_holdout(batch, config)
                with torch.autocast(device_type=device.type, enabled=amp):
                    result = compute_stage_loss(system, batch, config, consistency, phi)
                if not torch.isfinite(result["total"]):
                    raise FloatingPointError("Nonfinite stage loss; checkpoint remains at last successful update")
                scaler.scale(result["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(system.parameters(), config.get("grad_clip", 5.))
                scaler.step(optimizer)
                scaler.update()
                terms = {k: float(v.detach()) for k, v in result.items() if torch.is_tensor(v)}
            global_step += 1
            steps_this_call += 1
            with history_path.open("a") as stream:
                stream.write(json.dumps({"epoch": epoch, "step": global_step, **terms}) + "\n")
            extra = {"gan_trainer": gan_trainer.state_dict()} if gan_trainer else {}
            save_checkpoint(path, system, optimizers, schedulers, scaler, epoch=epoch, step_in_epoch=index + 1,
                            global_step=global_step, metadata=metadata, best=best, bad_epochs=bad_epochs, extra=extra)
            if max_steps is not None and steps_this_call >= max_steps:
                return {"status": "bounded_interruption", "global_step": global_step, "checkpoint": str(path), "metadata": metadata}
        if stage == "gan":
            val = validate_gan(system, validation, config, device)
        else:
            val = validate_epoch(system, validation, config, device, consistency, phi)
        for scheduler in schedulers.values():
            scheduler.step()
        improved = val < best
        if improved:
            best, bad_epochs = val, 0
        else:
            bad_epochs += 1
        extra = {"gan_trainer": gan_trainer.state_dict()} if gan_trainer else {}
        state = save_checkpoint(path, system, optimizers, schedulers, scaler, epoch=epoch + 1, step_in_epoch=0,
                                global_step=global_step, metadata=metadata, best=best, bad_epochs=bad_epochs, extra=extra)
        if improved:
            atomic_torch_save(state, output / "best.pt")
        if bad_epochs >= int(config.get("patience", 20)):
            break
    return {"status": "completed", "global_step": global_step, "checkpoint": str(path), "best_validation_loss": best, "metadata": metadata}


def load_deployment(checkpoint, device="cpu", *, phi_config=None):
    """B deployment reads no Phi file; C is explicit and never silently downgraded."""
    state = load_checkpoint(checkpoint, restore_random=False, map_location=device)
    config = dict(state["metadata"]["config"])
    mode = config.get("physics_mode", "A")
    system = StageSystem(config).to(device)
    attach_processing(system, config, device, initialize=False)
    system.load_state_dict(state["model"])
    system.eval()
    phi = None
    if mode == "C":
        if phi_config is None:
            raise ValueError("Experimental C deployment requires explicit compatible phi_config")
        phi, _ = load_phi({"physics": phi_config}, device, required=True)
    return system, config, state["metadata"], phi


def infer(checkpoint, manifest, output, *, device="cpu", split="inference", phi_config=None):
    from .data import ManifestDataset
    system, config, metadata, phi = load_deployment(checkpoint, device, phi_config=phi_config)
    if not hasattr(system, "inverse"):
        raise ValueError("Inference requires a reconstruction/enhancement/segmentation checkpoint")
    dataset = ManifestDataset(manifest, split=split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with torch.no_grad():
        for index, sample in enumerate(dataset):
            batch = prepare_batch(move_batch(_collate([sample]), device), config.get("preprocessing"), config)
            pipeline = build_deployment_pipeline(system, config, phi, batch)
            result = execute_pipeline(pipeline, batch, config)
            result["calibrated"] = batch["original_calibrated"]
            if result.get("segmentation_probabilities") is not None:
                from .pipeline import connected_components
                md = batch["metadata"][0]
                grid = md["grid"]
                xyz = [np.linspace(bounds[0], bounds[1], n, endpoint=False) + (bounds[1]-bounds[0])/(2*n)
                       for bounds, n in zip(grid["bounds_xyz_mm"], reversed(grid["shape_zyx"]))]
                components = connected_components(result["segmentation_probabilities"], xyz,
                                                   label_mode=config.get("segmentation", {}).get("label_mode", "multiclass"),
                                                   assessed=result.get("segmentation_assessed"))
                (output / f"case_{index:05d}_components.json").write_text(json.dumps(components, indent=2))
                baseline_valid = batch.get("context_validity", {}).get("baseline")
                if "baseline" in batch and baseline_valid is not None and bool(baseline_valid.all()):
                    from .pipeline import component_changes
                    base_logits = system.segmentation(batch["baseline"], baseline=batch["baseline"],
                                                      difference=torch.zeros_like(batch["baseline"]), roi=batch.get("roi"))
                    base_probabilities = system.segmentation.probabilities(base_logits)
                    prior = connected_components(base_probabilities, xyz,
                        label_mode=config.get("segmentation", {}).get("label_mode", "multiclass"),
                        assessed=result.get("segmentation_assessed"))
                    changes = component_changes(components, prior, roi_valid=bool(result["absolute_valid"].all()))
                    tracking = {"baseline_valid": True, "changes": changes,
                                "matching_method": "experimental_class_and_distance_heuristic", "clinically_validated": False}
                else:
                    tracking = {"baseline_valid": False, "changes": None, "reason": "missing_or_unregistered_prior_volume"}
                (output / f"case_{index:05d}_tracking.json").write_text(json.dumps(tracking, indent=2))
            arrays = {k: v.detach().cpu().numpy() for k, v in result.items() if torch.is_tensor(v)}
            arrays["observation_mask"] = batch["mask"].cpu().numpy()
            if "roi" in batch:
                arrays["assessed_roi"] = batch["roi"].cpu().numpy()
            np.savez_compressed(output / f"case_{index:05d}.npz", **arrays)
            validity = {"sample_id": sample["metadata"]["id"], "grid": sample["metadata"]["grid"],
                        "mode": config.get("mode", "snapshot"), "routing": result.get("routing"),
                        "absolute_valid": bool(result["absolute_valid"].all()),
                        "request_full_scan": bool(result.get("request_full_scan", False)),
                        "outside_roi_assessed": "roi" not in batch, "probability_calibration_validated": False}
            (output / f"case_{index:05d}_validity.json").write_text(json.dumps(validity, indent=2))
            count += 1
    report = {"cases": count, "checkpoint": str(checkpoint), "physics_training": metadata.get("physics_active"),
              "physics_inference": "C" if config.get("physics_mode") == "C" else "A_graph", "device": str(device),
              "probability_calibration_validated": False}
    (output / "inference.json").write_text(json.dumps(report, indent=2))
    return report


def evaluate(checkpoint, manifest, *, split="validation", device="cpu", phi_config=None):
    from .evaluation import evaluate_checkpoint
    return evaluate_checkpoint(checkpoint, manifest, split=split, device=device, phi_config=phi_config)


def profile(config, *, device="cuda", steps=2, backward=False):
    """Structural compute only: generated tensors never count as scientific data."""
    from .models import MSRNN
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: target-GPU AMP/memory/profile checks remain unrun")
    model = MSRNN(config.get("model", {})).to(target)
    frequencies = config.get("model", {}).get("frequency_grid_ghz", [.6 + .05 * i for i in range(10)])
    ports = int(config.get("profile_ports", 18))
    signals = torch.zeros(1, 2, len(frequencies), ports, ports, device=target)
    mask = torch.ones(1, 1, len(frequencies), ports, ports, device=target)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    amp = target.type == "cuda" and config.get("amp", True)
    elapsed = []
    for index in range(steps + 1):
        if target.type == "cuda": torch.cuda.synchronize(target)
        start = time.perf_counter()
        with torch.set_grad_enabled(backward), torch.autocast(device_type=target.type, enabled=amp):
            result = model(signals, mask, physics_mode="A")
            if backward:
                (result["refined"].square().mean() + result["coarse"].square().mean()).backward()
                model.zero_grad(set_to_none=True)
        if target.type == "cuda": torch.cuda.synchronize(target)
        if index: elapsed.append(time.perf_counter() - start)
    return {"evidence": "structural synthetic-input profile; no accuracy evidence", "device": str(target), "torch": torch.__version__,
            "amp": amp, "backward": backward, "parameters": sum(p.numel() for p in model.parameters()),
            "shape": list(result["refined"].shape), "mean_seconds": sum(elapsed) / len(elapsed),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(target) if target.type == "cuda" else None}


def qualify(checkpoint, manifest, config, *, output, device="cpu"):
    """Independent held-out gate: absent differential/noise/ROI evidence fails it."""
    from .data import ManifestDataset
    from .physics import qualification_report, validation_signal_metrics
    state = load_checkpoint(checkpoint, restore_random=False, map_location=device)
    training_config = state["metadata"]["config"]
    if training_config["stage"] != "phi":
        raise ValueError("Qualification needs an independently trained Phi checkpoint")
    system = StageSystem(training_config).to(device)
    system.load_state_dict(state["model"])
    system.eval()
    validation = ManifestDataset(manifest, split="validation")
    train_data = ManifestDataset(manifest, split="train")
    train_groups = [s["geometry_id"] for s in train_data.samples]
    if sorted(train_groups) != sorted(state["metadata"]["split"]["train"]):
        raise ValueError("Qualification training-group lineage differs from Phi checkpoint")
    validation_groups = [s["geometry_id"] for s in validation.samples]
    metrics = []
    coverage = set()
    contexts = []
    frequencies = None
    with torch.no_grad():
        for batch in epoch_batches(validation, 1, 0, 0, False):
            batch = move_batch(batch, device)
            context = _context(batch)
            contexts.append(context)
            properties = batch.get("spectral_target", batch.get("target"))
            if properties is None:
                raise ValueError("Qualification requires actual physical-property labels")
            frequencies = batch["frequencies_ghz"]
            prediction = system.phi(properties, frequencies, context)
            baseline = None
            if "baseline_spectral_target" in batch or "baseline_target" in batch:
                baseline = system.phi(batch.get("baseline_spectral_target", batch.get("baseline_target")), frequencies, context)
            metric = validation_signal_metrics(prediction, batch["signals"], batch["mask"], baseline_predicted=baseline,
                                               baseline_observed=batch.get("baseline_observations"), baseline_mask=batch.get("baseline_mask"),
                                               noise_scale=batch.get("noise_scale"), roi_channel_mask=batch.get("roi_channel_mask"))
            metrics.append(metric)
            md = batch["metadata"][0]
            coverage.update(md.get("qualification_coverage", []))
    if not metrics:
        raise ValueError("Empty independent qualification split")
    from .physics import context_hash
    if len({context_hash(c) for c in contexts}) != 1:
        raise ValueError("A qualification report binds one hardware/source/calibration context; qualify separate contexts separately")
    aggregate = {name: max(m[name] for m in metrics) if all(math.isfinite(m[name]) for m in metrics) else float("nan") for name in metrics[0]}
    report = qualification_report(system.phi, aggregate, contexts[0], frequencies.flatten(),
                                  training_groups=train_groups, validation_groups=validation_groups, coverage=coverage,
                                  thresholds=config.get("thresholds"), scientific=config.get("scientific", True),
                                  property_bounds=config.get("property_bounds"), geometry_feature_bounds=config.get("geometry_feature_bounds"))
    if report["qualified"]:
        from .physics import require_qualified_operator
        for context in contexts:
            require_qualified_operator(system.phi,report,context,system.phi.domain)
    report["operator_checkpoint"] = str(checkpoint)
    report["checkpoint_sha256"] = file_hash(checkpoint)
    report["manifest_sha256"] = file_hash(manifest)
    report["metric_aggregation"] = "conservative_max_over_held_out_geometries; missing evidence remains nonfinite"
    report["metrics"] = {k:(v if v is not None and math.isfinite(float(v)) else None) for k,v in report["metrics"].items()}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
