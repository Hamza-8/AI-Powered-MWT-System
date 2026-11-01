"""Reconstruction-only, geometry-grouped frequency pilot.

This prepares experiments; importing it never trains. Test case 3 is accessible only
through ``test`` after a successful, explicitly requested selection. This tiny
sphere study cannot establish clinical performance or independent property recovery.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

GRID_GHZ = [round(.60 + .05 * i, 2) for i in range(10)]
SPLITS = {"reference": [0], "train": [1, 4], "validation": [2], "sealed_test": [3]}
PROPERTY_OFFSET = (1., 0.)
PROPERTY_SCALE = (44., .8)
PROTOCOL_VERSION = 1


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_subsets(path):
    """Validate supplied CSV; return fixed-grid zero-based indices without reordering ports."""
    result = {k: [] for k in range(3, 11)}
    positions = {k: [] for k in result}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            k, slot = int(row["frequency_count"]), int(row["frequency_axis_index"]) - 1
            if k not in result or slot not in range(10):
                raise ValueError("Frequency CSV must contain only K=3..10 and MATLAB slots 1..10")
            if not math.isclose(float(row["GHz"]), GRID_GHZ[slot], abs_tol=1e-8):
                raise ValueError("CSV frequency/axis mismatch")
            result[k].append(slot)
            positions[k].append(int(row["subset_position"]))
    for k, indices in result.items():
        if len(indices) != k or len(set(indices)) != k or positions[k] != list(range(1, k + 1)):
            raise ValueError(f"Invalid or repeated K={k} subset")
    if result[10] != list(range(10)):
        raise ValueError("Ten-frequency comparator must cover the original full grid")
    return result


def validate_config(config):
    if config.get("physics_mode", "A") != "A":
        raise ValueError("The frequency pilot uses image-only A for every candidate")
    if config.get("use_cyclegan", False) or config.get("use_segmentation", False):
        raise ValueError("Frequency selection bypasses CycleGAN and excludes segmentation")
    if config.get("fixed_frequency_slots", 10) != 10:
        raise ValueError("The input representation must have ten fixed frequency slots")
    model = config.get("model", {})
    frequency_grid = model.get("frequency_grid_ghz", GRID_GHZ)
    if len(frequency_grid) != 10 or not np.allclose(frequency_grid, GRID_GHZ, rtol=0., atol=1e-8):
        raise ValueError("Every candidate must preserve the ten original frequency slots")
    if model.get("correction_depth", 1) not in (1, 2, 3):
        raise ValueError("Frequency selection includes enhancement, with fixed correction depth 1..3")
    size = int(model.get("latent_size", 8)) * 2 ** len(model.get("decoder_channels", [128,64,32,16]))
    if int(config.get("grid_size", 128)) != size:
        raise ValueError("Target grid must match native learned decoder size")
    if not config.get("seeds") or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Use distinct paired seeds")
    if int(config.get("patience", config.get("epochs", 0))) < int(config.get("epochs", 0)):
        raise ValueError("Frequency candidates use equal full epoch budgets; patience must be at least epochs")
    if int(config.get("epochs", 0)) < 1 or int(config.get("batch_size", 1)) != 1:
        raise ValueError("Positive training budget and batch size one required for this pilot")
    if config.get("frequency_config", {}).get("status", "pending_selection") != "pending_selection":
        raise ValueError("Experiment config must remain pending_selection; selection writes a separate frozen artifact")
    return config


def source_identity():
    root = Path(__file__).parent
    return {p.name: file_sha256(p) for p in sorted(root.glob("*.py"))}


def execution_identity(device, amp_requested):
    target = torch.device(device)
    cuda = target.type == "cuda"
    return {"torch_version": torch.__version__, "numpy_version": np.__version__,
            "device_type": target.type, "amp_active": bool(amp_requested and cuda),
            "cuda_version": torch.version.cuda if cuda else None,
            "cuda_capability": list(torch.cuda.get_device_capability(target)) if cuda and torch.cuda.is_available() else None}


def run_identity(config, mat, csv_path, k, seed, preprocessing, execution=None):
    return {
        "protocol_version": PROTOCOL_VERSION, "config": copy.deepcopy(config),
        "data_sha256": file_sha256(mat), "csv_sha256": file_sha256(csv_path),
        "source": source_identity(), "k": int(k), "seed": int(seed),
        "indices_zero_based": read_subsets(csv_path)[int(k)], "splits": SPLITS,
        "preprocessing": preprocessing, "physics_requested": "A", "physics_active": "none",
        "cyclegan": "bypass", "segmentation": "excluded", "fixed_slots": 10,
        "property_reference_ghz": .85, "execution": execution,
    }


def fit_input_scale(training_samples):
    """One global complex RMS, fitted on observed entries in training geometries only.

    No centering or sample-wise renormalization removes signed lesion amplitudes.
    Zero observed values contribute to the denominator; hidden NaNs never contribute.
    """
    total, count = 0., 0
    for sample in training_samples:
        signal = torch.as_tensor(sample["signals"], dtype=torch.float64)
        mask = torch.as_tensor(sample["mask"], dtype=torch.bool).expand_as(signal)
        selected = signal[mask]
        if selected.numel() and not torch.isfinite(selected).all():
            raise ValueError("Nonfinite observed training measurement")
        total += selected.square().sum().item()
        count += selected.numel()
    if count == 0:
        raise ValueError("No observed training support")
    return {"algorithm": "train_observed_global_realimag_rms_v1", "scale": max(math.sqrt(total / count), 1e-12),
            "fit_case_ids": SPLITS["train"], "observed_real_values": count,
            "property_offset": list(PROPERTY_OFFSET), "property_scale": list(PROPERTY_SCALE)}


def transform_signal(signal, mask, preprocessing):
    mask = mask.bool()
    observed = torch.where(mask, signal, torch.zeros_like(signal))
    if not torch.isfinite(observed).all():
        raise ValueError("Observed signals must be finite")
    return observed / preprocessing["scale"]


def normalized_contrast(volume):
    offset = volume.new_tensor(PROPERTY_OFFSET).reshape(-1, 1, 1, 1)
    scale = volume.new_tensor(PROPERTY_SCALE).reshape(-1, 1, 1, 1)
    return (volume - offset) / scale


def reconstruction_metrics(prediction, target, *, extent_mm=120.):
    """Both physical-property errors and foreground-balanced normalized contrast.

    Localization/volume are sphere object diagnostics, not a clinical segmentation
    score. Threshold 0.5 is declared before running the protocol.
    """
    prediction, target = prediction.detach().double().cpu(), target.detach().double().cpu()
    if prediction.shape != target.shape or prediction.shape[0] != 2:
        raise ValueError("Expected matching [2,Z,Y,X] property volumes")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite reconstruction or labels")
    p, t = normalized_contrast(prediction), normalized_contrast(target)
    foreground = t.mean(0) >= .5
    background = ~foreground
    error = (p - t).square().mean(0)
    regions = [error[region].mean() for region in (foreground, background) if region.any()]
    balanced = torch.stack(regions).mean().sqrt().item()
    estimated = p.mean(0) >= .5
    spacing = torch.tensor([extent_mm / n for n in target.shape[-3:]], dtype=torch.float64)
    true_points = torch.nonzero(foreground).double()
    estimated_points = torch.nonzero(estimated).double()
    if true_points.numel() and estimated_points.numel():
        center_error = torch.linalg.vector_norm((true_points.mean(0) - estimated_points.mean(0)) * spacing).item()
    elif not true_points.numel() and not estimated_points.numel():
        center_error = 0.
    else:
        center_error = extent_mm * math.sqrt(3)
    true_volume = float(foreground.sum()) * spacing.prod().item()
    predicted_volume = float(estimated.sum()) * spacing.prod().item()
    squared = (prediction - target).square().flatten(1).mean(1).sqrt()
    return {"epsilon_rmse": squared[0].item(), "sigma_rmse_s_per_m": squared[1].item(),
            "balanced_contrast_rmse": balanced, "localization_error_mm": center_error,
            "relative_volume_error": abs(predicted_volume - true_volume) / max(true_volume, spacing.prod().item()),
            "true_volume_mm3": true_volume, "predicted_volume_mm3": predicted_volume,
            "foreground_contrast_rmse": error[foreground].mean().sqrt().item() if foreground.any() else 0.,
            "background_contrast_rmse": error[background].mean().sqrt().item() if background.any() else 0.}


def baseline_metrics(training_samples, validation_samples):
    mean = torch.stack([torch.as_tensor(s["target"]) for s in training_samples]).mean(0)
    zero = torch.zeros_like(mean)
    zero[0] = 1.
    return {"zero_contrast": average_metrics([reconstruction_metrics(zero, s["target"]) for s in validation_samples]),
            "training_mean_volume": average_metrics([reconstruction_metrics(mean, s["target"]) for s in validation_samples])}


def average_metrics(rows):
    if not rows:
        raise ValueError("No metrics")
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _dataset(mat, split, indices, config, *, allow_test=False):
    from .data import PilotDataset
    return PilotDataset(mat, split=split, frequency_indices=indices, allow_test=allow_test,
                        grid_size=int(config.get("grid_size", 128)))


def _model(config):
    from .models import MSRNN
    return MSRNN(config.get("model", {}))


def _predict(model, sample, preprocessing, device):
    signal = torch.as_tensor(sample["signals"], device=device).unsqueeze(0)
    mask = torch.as_tensor(sample["mask"], device=device).bool().unsqueeze(0)
    signal = transform_signal(signal, mask, preprocessing)
    frequencies = torch.as_tensor(sample["frequencies_ghz"], device=device).unsqueeze(0)
    return model(signal, mask, frequencies, physics_mode="A")


def _objective(outputs, target, object_mask, edge_weight=.01):
    scale = target.new_tensor(PROPERTY_SCALE).reshape(1, 2, 1, 1, 1)
    foreground = object_mask.bool().expand_as(target)
    total = target.sum() * 0
    for key, weight in (("coarse", .5), ("refined", 1.)):
        prediction = outputs[key]
        error = ((prediction - target) / scale).square()
        terms = [error[region].mean() for region in (foreground, ~foreground) if region.any()]
        total = total + weight * torch.stack(terms).mean()
    refined, normalized_target = outputs["refined"] / scale, target / scale
    for axis in (-3, -2, -1):
        total = total + edge_weight / 3 * (torch.diff(refined, dim=axis) - torch.diff(normalized_target, dim=axis)).abs().mean()
    return total


def _evaluate(model, samples, preprocessing, device):
    model.eval()
    refined, coarse = [], []
    with torch.no_grad():
        for sample in samples:
            output = _predict(model, sample, preprocessing, device)
            refined.append(reconstruction_metrics(output["refined"][0], sample["target"]))
            coarse.append(reconstruction_metrics(output["coarse"][0], sample["target"]))
    return {"refined": average_metrics(refined), "coarse": average_metrics(coarse)}


def prepare(config, mat, csv_path, output):
    validate_config(config)
    subsets = read_subsets(csv_path)
    # Intentionally never request the test split here.
    validation = list(_dataset(mat, "validation", list(range(10)), config))
    training = list(_dataset(mat, "train", list(range(10)), config))
    report = {"status": "pending_selection", "protocol_version": PROTOCOL_VERSION, "config": config,
              "splits": SPLITS, "frequency_grid_ghz": GRID_GHZ, "subsets_zero_based": subsets,
              "baselines": baseline_metrics(training, validation), "data_sha256": file_sha256(mat),
              "csv_sha256": file_sha256(csv_path), "test_accessed": False,
              "scope": "Four object geometries; normalized properties coincide. No independent property or clinical validation."}
    atomic_json(Path(output) / "protocol.json", report)
    return report


def run_candidate(config, mat, csv_path, output, k, seed, *, device="cpu", resume=False, max_updates=None):
    """Train from scratch, or exactly resume an identity-matched candidate.

    max_updates supports bounded mechanism verification and leaves status running;
    it is not a scientific completed result and cannot be consumed by selection.
    """
    validate_config(config)
    if seed not in config["seeds"]:
        raise ValueError("Seed is not part of the predeclared paired protocol")
    indices = read_subsets(csv_path)[int(k)]
    training = list(_dataset(mat, "train", indices, config))
    validation = list(_dataset(mat, "validation", indices, config))
    preprocessing = fit_input_scale(training)
    identity = run_identity(config, mat, csv_path, k, seed, preprocessing,
                            execution=execution_identity(device, config.get("amp", True)))
    identity_hash = canonical_hash(identity)
    root = Path(output) / f"K{k:02d}" / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    result_path, checkpoint_path = root / "result.json", root / "resume.pt"
    if result_path.exists():
        completed = json.loads(result_path.read_text())
        if (completed.get("identity_sha256") != identity_hash
                or canonical_hash(completed.get("identity")) != identity_hash
                or completed.get("status") != "completed" or completed.get("test_accessed")
                or completed.get("physics_requested") != "A" or completed.get("physics_active") != "none"):
            raise ValueError("Completed work identity differs; use a separate output directory")
        best_path = root / "best.pt"
        if not best_path.exists() or file_sha256(best_path) != completed.get("best_checkpoint_sha256"):
            raise ValueError("Completed checkpoint is missing or its bytes changed")
        if not resume:
            raise FileExistsError("Completed candidate exists; use --resume to reuse matching completed work")
        return completed
    if any((Path(output) / name).exists() for name in ("sealed_test_results.json", "sealed_test_access.json")):
        raise PermissionError("Test has been opened; new candidate training is locked in this output directory")
    seed_everything(seed)
    model = _model(config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 2e-4)),
                                  weight_decay=float(config.get("weight_decay", 1e-5)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    amp = bool(config.get("amp", True) and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    state = {"epoch": 0, "position": 0, "order": None, "updates": 0, "best": float("inf"),
             "bad_epochs": 0, "best_model": None, "history": []}
    if checkpoint_path.exists():
        if not resume:
            raise FileExistsError("Interrupted candidate exists; use --resume")
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["identity_sha256"] != identity_hash:
            raise ValueError("Resume identity mismatch: data/config/code/preprocessing/seed must match")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["progress"]
        restore_rng(saved["rng"])
    started, invocation_updates = time.perf_counter(), 0
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def save():
        atomic_checkpoint(checkpoint_path, {"schema_version": 1, "identity": identity, "identity_sha256": identity_hash,
            "architecture": config.get("model", {}), "parameter_count": parameter_count, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": rng_state(), "progress": state, "splits": SPLITS, "preprocessing": preprocessing,
            "frequency_grid_ghz": GRID_GHZ, "selected_indices": indices, "physics_requested": "A", "physics_active": "none",
            "inference_requirements": {"phi": False, "cyclegan": False, "segmentation": False}})

    while state["epoch"] < config["epochs"]:
        if state["order"] is None:
            state["order"] = torch.randperm(len(training)).tolist()
        model.train()
        while state["position"] < len(state["order"]):
            sample = training[state["order"][state["position"]]]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if amp else "cpu", enabled=amp):
                output_model = _predict(model, sample, preprocessing, device)
                target = torch.as_tensor(sample["target"], device=device).unsqueeze(0)
                foreground = torch.as_tensor(sample["object_mask"], device=device).unsqueeze(0)
                loss = _objective(output_model, target, foreground, float(config.get("gradient_weight", .01)))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite frequency training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.)))
            scaler.step(optimizer)
            scaler.update()
            state["position"] += 1
            state["updates"] += 1
            invocation_updates += 1
            save()
            if max_updates is not None and invocation_updates >= max_updates:
                return {"status": "running", "identity_sha256": identity_hash, "updates": state["updates"],
                        "bounded_mechanism_check": True}
        metrics = _evaluate(model, validation, preprocessing, device)
        current = metrics["refined"]["balanced_contrast_rmse"]
        state["history"].append({"epoch": state["epoch"], "validation": metrics})
        if current < state["best"] - float(config.get("minimum_delta", 0.)):
            state["best"], state["bad_epochs"] = current, 0
            state["best_model"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            state["bad_epochs"] += 1
        scheduler.step()
        state["epoch"] += 1
        state["position"], state["order"] = 0, None
        save()
        if state["bad_epochs"] >= int(config.get("patience", config["epochs"])):
            break
    model.load_state_dict(state["best_model"])
    metrics = _evaluate(model, validation, preprocessing, device)
    baselines = baseline_metrics(training, validation)
    result = {"status": "completed", "identity": identity, "identity_sha256": identity_hash, "k": k, "seed": seed,
              "parameter_count": parameter_count, "validation": metrics, "baselines": baselines,
              "epochs_executed": state["epoch"], "optimizer_updates": state["updates"],
              "last_invocation_elapsed_seconds": time.perf_counter() - started,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
              "device": str(device), "test_accessed": False, "physics_requested": "A", "physics_active": "none"}
    # Best and resume checkpoints are distinct: resuming optimizer state never uses best-model weights.
    atomic_checkpoint(root / "best.pt", {"schema_version": 1, "model": state["best_model"], "identity": identity,
                      "identity_sha256": identity_hash, "preprocessing": preprocessing,
                      "architecture": config.get("model", {}), "physics_requested": "A", "physics_active": "none"})
    result["best_checkpoint_sha256"] = file_sha256(root / "best.pt")
    atomic_json(result_path, result)
    return result


def select_results(results, config):
    """Pure predeclared selection; incomplete evidence cannot freeze a frequency count."""
    validate_config(config)
    grouped = {k: [] for k in range(3, 11)}
    shared_identity = None
    expected_seeds = sorted(config["seeds"])
    for result in results:
        if result.get("status") != "completed" or result.get("test_accessed"):
            raise ValueError("Selection accepts completed validation-only runs")
        if result.get("physics_requested") != "A" or result.get("physics_active") != "none":
            raise ValueError("Every frequency candidate must use image-only A")
        identity = result.get("identity")
        if identity is None or canonical_hash(identity) != result.get("identity_sha256"):
            raise ValueError("Selection requires intact complete run identity")
        if identity.get("config") != config or identity.get("k") != result["k"] or identity.get("seed") != result["seed"]:
            raise ValueError("Run/config/seed identity mismatch")
        comparable = copy.deepcopy(identity)
        for candidate_key in ("k", "seed", "indices_zero_based"):
            comparable.pop(candidate_key, None)
        # Numeric RMS differs with the observed selected support; the algorithm,
        # fit geometries and output scales must agree across every K.
        comparable["preprocessing"].pop("scale", None)
        comparable["preprocessing"].pop("observed_real_values", None)
        if shared_identity is None:
            shared_identity = comparable
        elif comparable != shared_identity:
            raise ValueError("Candidate data/source/architecture/budget/preprocessing policies differ")
        if int(result["k"]) not in grouped:
            raise ValueError("Unexpected frequency candidate")
        grouped[int(result["k"])].append(result)
    for k, runs in grouped.items():
        if sorted(r["seed"] for r in runs) != expected_seeds:
            raise ValueError(f"K={k}: require exactly the predeclared paired seeds")
    counts = {run["parameter_count"] for runs in grouped.values() for run in runs}
    if len(counts) != 1:
        raise ValueError("Frequency candidates must have equal parameter counts")
    summaries = {}
    for k, runs in grouped.items():
        errors = [r["validation"]["refined"]["balanced_contrast_rmse"] for r in runs]
        summaries[k] = {"mean": float(np.mean(errors)), "std": float(np.std(errors)),
                        "metrics": average_metrics([r["validation"]["refined"] for r in runs]),
                        "coarse": average_metrics([r["validation"]["coarse"] for r in runs])}
    thresholds = config.get("selection", {})
    ten = summaries[10]
    baselines = [r["baselines"] for r in grouped[10]]
    zero = np.mean([b["zero_contrast"]["balanced_contrast_rmse"] for b in baselines])
    mean = np.mean([b["training_mean_volume"]["balanced_contrast_rmse"] for b in baselines])
    required_gain = float(thresholds.get("baseline_improvement_fraction", .05))
    max_center = float(thresholds.get("max_localization_error_mm", 10.))
    max_volume = float(thresholds.get("max_relative_volume_error", .5))

    def localized(summary):
        m = summary["metrics"]
        return m["localization_error_mm"] <= max_center and m["relative_volume_error"] <= max_volume

    useful = ten["mean"] < min(zero, mean) * (1 - required_gain) and localized(ten)
    report = {"schema_version": 1, "status": "inconclusive", "selected_k": None, "summaries": summaries,
              "thresholds": {"within_ten_frequency_fraction": .05, "baseline_improvement_fraction": required_gain,
                             "max_localization_error_mm": max_center, "max_relative_volume_error": max_volume},
              "ten_frequency_useful": useful, "test_accessed": False,
              "reason": "Ten-frequency reconstruction must beat both baselines and satisfy predeclared localization/volume checks"}
    if useful:
        eligible = [k for k in range(3, 11) if summaries[k]["mean"] <= 1.05 * ten["mean"] and localized(summaries[k])]
        report.update(status="selected", selected_k=min(eligible), reason="Smallest eligible K within 5% of useful ten-frequency validation mean")
    return report


def select_and_freeze(config, mat, csv_path, output):
    if any((Path(output) / name).exists() for name in ("sealed_test_results.json", "sealed_test_access.json")):
        raise PermissionError("Test has been opened; frequency selection is immutable in this output directory")
    results = []
    for k in range(3, 11):
        for seed in config["seeds"]:
            path = Path(output) / f"K{k:02d}" / f"seed_{seed}" / "result.json"
            result = json.loads(path.read_text())
            train = list(_dataset(mat, "train", read_subsets(csv_path)[k], config))
            expected = run_identity(config, mat, csv_path, k, seed, fit_input_scale(train),
                                    execution=result["identity"].get("execution"))
            if result["identity_sha256"] != canonical_hash(expected):
                raise ValueError(f"Stale or incompatible result: {path}")
            best_path = path.parent / "best.pt"
            if not best_path.exists() or file_sha256(best_path) != result.get("best_checkpoint_sha256"):
                raise ValueError(f"Missing or modified best checkpoint: {best_path}")
            results.append(result)
    report = select_results(results, config)
    report["results_sha256"] = canonical_hash(results)
    atomic_json(Path(output) / "selection.json", report)
    if report["status"] == "selected":
        indices = read_subsets(csv_path)[report["selected_k"]]
        frozen = {"schema_version": 1, "status": "frozen", "selected_k": report["selected_k"],
                  "grid_ghz": GRID_GHZ, "indices_zero_based": indices, "frequencies_ghz": [GRID_GHZ[i] for i in indices],
                  "fixed_slots": 10, "reference_frequency_ghz": .85, "port_order": "physical_1_to_18",
                  "selection_sha256": canonical_hash(report), "data_sha256": file_sha256(mat),
                  "csv_sha256": file_sha256(csv_path), "requires_all_stages_to_match": True}
        atomic_json(Path(output) / "frozen_frequencies.json", frozen)
    return report


def evaluate_sealed_test(config, mat, csv_path, output, *, device="cpu"):
    """Explicit post-selection access; preflight every artifact before opening test.

    The access marker is durable before materializing test labels, so an interrupted
    evaluation cannot accidentally unlock selection or further candidate training.
    """
    validate_config(config)
    root = Path(output)
    report = json.loads((root / "selection.json").read_text())
    frozen = json.loads((root / "frozen_frequencies.json").read_text())
    if report.get("status") != "selected" or frozen.get("selection_sha256") != canonical_hash(report):
        raise ValueError("Sealed test requires a successful explicit matching selection")
    k = int(frozen["selected_k"])
    indices = read_subsets(csv_path)[k]
    if (report.get("selected_k") != k or frozen.get("status") != "frozen"
            or frozen.get("data_sha256") != file_sha256(mat)
            or frozen.get("csv_sha256") != file_sha256(csv_path)
            or frozen.get("indices_zero_based") != indices
            or frozen.get("grid_ghz") != GRID_GHZ
            or frozen.get("frequencies_ghz") != [GRID_GHZ[i] for i in indices]):
        raise ValueError("Frozen selection/data/frequency binding mismatch")
    training = list(_dataset(mat, "train", indices, config))
    preprocessing = fit_input_scale(training)
    # Confirm every run that participated in selection remains intact. No test
    # loader call is made while validating sources, preprocessing or checkpoints.
    all_results = []
    selected_states = []
    for candidate_k in range(3, 11):
        for seed in config["seeds"]:
            candidate_root = root / f"K{candidate_k:02d}" / f"seed_{seed}"
            result = json.loads((candidate_root / "result.json").read_text())
            all_results.append(result)
            if candidate_k != k:
                continue
            best_path = candidate_root / "best.pt"
            if file_sha256(best_path) != result.get("best_checkpoint_sha256"):
                raise ValueError("Sealed-test best-checkpoint bytes changed")
            saved = torch.load(best_path, map_location=device, weights_only=False)
            expected = run_identity(config, mat, csv_path, k, seed, preprocessing,
                                    execution=saved["identity"].get("execution"))
            if (saved["identity_sha256"] != canonical_hash(expected)
                    or canonical_hash(saved.get("identity")) != saved["identity_sha256"]
                    or saved.get("preprocessing") != preprocessing
                    or result.get("identity_sha256") != saved["identity_sha256"]):
                raise ValueError("Sealed-test checkpoint identity mismatch")
            selected_states.append((seed, saved))
    if canonical_hash(all_results) != report.get("results_sha256"):
        raise ValueError("Validation results changed since explicit selection")
    atomic_json(root / "sealed_test_access.json", {"selected_k": k, "test_access_requested": True,
                "selection_sha256": canonical_hash(report), "selection_locked": True})
    samples = list(_dataset(mat, "test", indices, config, allow_test=True))
    rows = []
    for seed, saved in selected_states:
        model = _model(config).to(device)
        model.load_state_dict(saved["model"])
        rows.append({"seed": seed, "metrics": _evaluate(model, samples, saved["preprocessing"], device)})
    result = {"selected_k": k, "test_accessed": True, "seed_results": rows,
              "selection_sha256": canonical_hash(report), "warning": "Test results cannot revise K selection"}
    atomic_json(root / "sealed_test_results.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "select", "test"))
    parser.add_argument("--config", default="configs/frequency_pilot.json")
    parser.add_argument("--mat", default="data/pilot/frequency_pilot.mat")
    parser.add_argument("--csv", default="data/pilot/frequency_subsets.csv")
    parser.add_argument("--output", default="outputs/frequency")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--counts", default="3,4,5,6,7,8,9,10")
    parser.add_argument("--max-updates", type=int, help="Bounded verification only; leaves runs incomplete")
    args = parser.parse_args(argv)
    config = validate_config(json.loads(Path(args.config).read_text()))
    if args.command == "prepare":
        result = prepare(config, args.mat, args.csv, args.output)
    elif args.command == "run":
        result = []
        for k in (int(x) for x in args.counts.split(",")):
            for seed in config["seeds"]:
                result.append(run_candidate(config, args.mat, args.csv, args.output, k, seed,
                    device=args.device, resume=args.resume, max_updates=args.max_updates))
    elif args.command == "select":
        result = select_and_freeze(config, args.mat, args.csv, args.output)
    else:
        result = evaluate_sealed_test(config, args.mat, args.csv, args.output, device=args.device)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
