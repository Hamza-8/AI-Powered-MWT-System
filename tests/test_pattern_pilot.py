"""Representation invariants and controlled training mechanics, not accuracy."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pattern_amplitude_model import PatternAmplitudeMSRNN, factor_measurements
from train_pattern_pilot import Experiment, atomic_write
from microscan import frequency as f
from microscan.models import MSRNN, measurement_features, tiny_config

MAT = ROOT / "data/pilot_expanded/frequency_pilot.mat"
CSV = ROOT / "data/pilot/frequency_subsets.csv"


def features(values=None, mask=None):
    if values is None:
        values = torch.randn(2, 2, 10, 6, 6)
    if mask is None:
        mask = torch.ones(values.shape[0], 1, *values.shape[2:], dtype=torch.bool)
    return measurement_features(values, mask, f.GRID_GHZ), mask


def test_common_complex_rms_preserves_phase_relative_frequencies_and_all_observations():
    raw, mask = features()
    raw[:, :2, 0] = 0  # Observed zero remains in denominator.
    pattern, descriptors, rms = factor_measurements(raw, mask)
    expected = (raw[:, :2].square().sum((1, 2, 3, 4)) / mask.flatten(1).sum(1)).sqrt()
    torch.testing.assert_close(rms, expected)
    torch.testing.assert_close(pattern[:, :2] * rms[:, None, None, None, None], raw[:, :2])
    torch.testing.assert_close(pattern[:, 2:], raw[:, 2:])
    z = torch.complex(raw[:, 0], raw[:, 1])
    u = torch.complex(pattern[:, 0], pattern[:, 1])
    torch.testing.assert_close(torch.angle(z), torch.angle(u))
    gained = raw.clone(); gained[:, :2] *= 4.8
    gained_pattern, gained_descriptors, gained_rms = factor_measurements(gained, mask)
    torch.testing.assert_close(pattern, gained_pattern)
    torch.testing.assert_close(gained_rms, rms * 4.8)
    torch.testing.assert_close(gained_descriptors[:, 0] - descriptors[:, 0], torch.full((2,), 4.8).log())
    # A global phase change is retained, rather than silently phase-aligned away.
    rotated = raw.clone(); rotated[:, 0], rotated[:, 1] = -raw[:, 1], raw[:, 0]
    rotated_pattern, _, _ = factor_measurements(rotated, mask)
    torch.testing.assert_close(rotated_pattern[:, 0], -pattern[:, 1])


@pytest.mark.parametrize("magnitude", [0., 1e-20, 1e20])
def test_extreme_and_empty_support_are_finite_with_gradients(magnitude):
    values = torch.full((2, 2, 10, 6, 6), magnitude, requires_grad=True)
    mask = torch.ones(2, 1, 10, 6, 6, dtype=torch.bool); mask[1] = False
    raw, _ = features(values, mask)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        pattern, descriptors, rms = factor_measurements(raw, mask)
    assert pattern.dtype == descriptors.dtype == rms.dtype == torch.float32
    assert all(torch.isfinite(x).all() for x in (pattern, descriptors, rms))
    (pattern.square().mean() + descriptors.square().mean()).backward()
    assert torch.isfinite(values.grad).all()
    assert not pattern[1, :2].any() and rms[1] == 0 and descriptors[1, 1] == 0


def test_hidden_nan_inf_isolation_and_observed_nonfinite_rejection():
    raw, mask = features(); mask[:, :, 3:7] = False
    clean = raw.clone(); clean[:, :2, 3:7] = 0; clean[:, 2:3] = mask
    dirty = clean.clone(); dirty[:, 0, 3:7] = float("nan"); dirty[:, 1, 3:7] = float("inf")
    for left, right in zip(factor_measurements(clean, mask), factor_measurements(dirty, mask)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    dirty[:, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Observed"):
        factor_measurements(dirty, mask)


def test_model_masking_seeded_core_gradients_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17); old = MSRNN(tiny_config())
    torch.manual_seed(17); model = PatternAmplitudeMSRNN(tiny_config())
    assert all(torch.equal(v, model.coarse.core.state_dict()[k]) for k, v in old.coarse.state_dict().items())
    signal = torch.randn(1, 2, 10, 6, 6)
    mask = torch.ones(1, 1, 10, 6, 6, dtype=torch.bool); mask[:, :, 2:6] = False
    altered = signal.clone(); altered[:, :, 2:6] = float("nan")
    output = model(signal, mask)
    torch.testing.assert_close(output["refined"], model(altered, mask)["refined"], rtol=0, atol=0)
    assert output["refined"].shape == (1, 2, 16, 16, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    target = torch.ones_like(output["refined"]); target[:, 1] = 0
    loss = f._objective(output, target, torch.ones(1, 1, 16, 16, 16, dtype=torch.bool))
    loss.backward()
    for branch in (model.coarse.conditioner.raw_encoder, model.coarse.conditioner.amplitude,
                   model.coarse.core.encoder, model.enhancement.measurement_encoder.pattern_encoder,
                   model.enhancement.measurement_encoder.conditioner.raw_encoder,
                   model.enhancement.measurement_encoder.conditioner.amplitude):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        assert sum(p.grad.abs().sum() for p in branch.parameters()) > 0
    optimizer.step()
    atomic_write(tmp_path / "model.pt", model.state_dict(), checkpoint=True)
    restored = PatternAmplitudeMSRNN(tiny_config())
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True), strict=True)
    torch.testing.assert_close(model(signal, mask)["refined"], restored(signal, mask)["refined"], rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in restored.parameters())


@pytest.mark.skipif(not MAT.exists(), reason="Expanded simulation MAT stays local")
def test_real_training_exact_resume_final_policy_history_and_sealed_test(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/frequency_pilot_pattern_v1.json").read_text())
    config.update(grid_size=16, epochs=2, patience=2, amp=False)
    config["model"].update(tiny_config())
    source = f.source_identity(); splits = copy.deepcopy(f.SPLITS)
    interrupted = Experiment(config, MAT, CSV, tmp_path / "resumed", "cpu")
    touched = []
    from train_expanded_pilot import ExpandedPilot
    original = ExpandedPilot.__getitem__
    def track(self, index):
        if index < len(self): touched.append(self.ids[index])
        return original(self, index)
    monkeypatch.setattr(ExpandedPilot, "__getitem__", track)
    assert interrupted.run(max_updates=1, seeds=[17])[0]["status"] == "running"
    root = tmp_path / "resumed/K10/seed_17"
    initial = (root / "initial_metrics.json").read_bytes()
    interrupted.run(resume=True, max_updates=1, seeds=[17])
    result = interrupted.run(resume=True, seeds=[17])[0]
    saved = torch.load(root / "resume.pt", weights_only=False)
    final = torch.load(root / "final.pt", weights_only=True)
    assert result["optimizer_updates"] == 10 and result["epochs_executed"] == 2
    assert result["checkpoint_policy"] == "fixed_final_epoch" and not (root / "best.pt").exists()
    assert all(torch.equal(v, final["model"][key]) for key, v in saved["model"].items())
    assert result["validation"] == saved["progress"]["history"][-1]["validation"]
    assert saved["preprocessing"]["fit_case_ids"] == [1, 2, 4, 6, 7]
    assert 3 not in touched and set(touched) == {1, 2, 4, 5, 6, 7, 8}
    assert [r["case_id"] for r in result["validation"]["per_case"]] == [5, 8]
    assert len(json.loads((root / "training_history.json").read_text())["history"]) == 2
    assert (root / "initial_metrics.json").read_bytes() == initial
    assert "balanced_contrast_rmse" not in json.dumps(result)
    assert "balanced_contrast_rmse" not in (root / "training_history.json").read_text()
    direct = Experiment(config, MAT, CSV, tmp_path / "direct", "cpu")
    direct.run(seeds=[17])
    uninterrupted = torch.load(tmp_path / "direct/K10/seed_17/final.pt", weights_only=True)
    assert all(torch.equal(v, uninterrupted["model"][key]) for key, v in final["model"].items())
    assert interrupted.run(resume=True, seeds=[17])[0] == result
    changed = copy.deepcopy(config); changed["model"]["pattern_rms_floor"] *= 10
    with pytest.raises(ValueError, match="identity"):
        Experiment(changed, MAT, CSV, tmp_path / "resumed", "cpu").run(resume=True, seeds=[17])
    assert f.source_identity() == source and f.SPLITS == splits


def test_config_preserves_baseline_training_budget_objective_and_split():
    old = json.loads((ROOT / "configs/frequency_pilot_crossed_v3.json").read_text())
    new = json.loads((ROOT / "configs/frequency_pilot_pattern_v1.json").read_text())
    for key in ("epochs", "batch_size", "seeds", "learning_rate", "weight_decay", "gradient_weight",
                "gradient_clip", "amp", "grid_size", "expanded_protocol", "physics_mode"):
        assert old[key] == new[key]


def test_amp_model_backward_fixed_frequency_slots_and_empty_validity():
    torch.set_num_threads(1)
    model = PatternAmplitudeMSRNN(tiny_config())
    parameters = sum(p.numel() for p in model.parameters())
    signal = torch.randn(1, 2, 10, 6, 6)
    for slots in ([0, 5, 9], list(range(10))):
        mask = torch.zeros(1, 1, 10, 6, 6, dtype=torch.bool); mask[:, :, slots] = True
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = model(signal, mask, physics_mode="B")
            loss = output["refined"].float().square().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        assert output["refined"].shape == (1, 2, 16, 16, 16)
        assert sum(p.numel() for p in model.parameters()) == parameters
    with torch.inference_mode():
        empty = model(torch.full_like(signal, float("nan")), torch.zeros_like(mask))
    assert not empty["measurement_valid"].any() and not empty["absolute_valid"].any()
    assert torch.isfinite(empty["refined"]).all()
