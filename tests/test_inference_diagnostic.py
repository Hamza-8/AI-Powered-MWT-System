"""Mechanism tests only: an untrained fixture never supplies research evidence."""
import importlib.util
from pathlib import Path

import pytest
import torch

from microscan.frequency import canonical_hash, file_sha256, source_identity
from microscan.models import MSRNN, tiny_config


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("diagnose_frequency", ROOT/"scripts/diagnose_frequency.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def fixture_checkpoint(path):
    config = {"model": tiny_config(), "grid_size": 16}
    identity = {"config": config, "source": source_identity(), "data_sha256": file_sha256(ROOT/"data/pilot/frequency_pilot.mat"),
                "k": 3, "seed": 17, "indices_zero_based": [0, 4, 9]}
    torch.manual_seed(1)
    model = MSRNN(tiny_config())
    torch.save({"identity": identity, "identity_sha256": canonical_hash(identity), "architecture": config["model"],
                "preprocessing": {"scale": .002}, "model": model.state_dict(), "progress": {"epoch": 0, "updates": 0},
                "physics_requested": "A", "physics_active": "none"}, path)


def test_inference_only_same_source_mask_controls_and_sealed_test(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    checkpoint = tmp_path/"resume.pt"
    fixture_checkpoint(checkpoint)
    before = checkpoint.read_bytes()
    def no_optimizer(*args, **kwargs):
        raise AssertionError("No optimizer or training step is allowed")
    monkeypatch.setattr(torch.optim, "AdamW", no_optimizer)
    accesses = []
    original = diagnostic.PilotDataset.__getitem__
    def tracked(self, index):
        if index < len(self):
            assert self.split in ("train", "validation")
            accesses.append(self.ids[index])
        return original(self, index)
    monkeypatch.setattr(diagnostic.PilotDataset, "__getitem__", tracked)
    report = diagnostic.diagnose(checkpoint, ROOT/"data/pilot/frequency_pilot.mat", tmp_path/"diagnostic")
    assert report["optimizer_steps"] == 0 and report["forward_passes"] == 4
    assert accesses == [1, 4, 2] and not report["test_accessed"]
    assert checkpoint.read_bytes() == before and report["checkpoint_unchanged"]
    assert [c["case_id"] for c in report["cases"]] == [1, 4, 2]
    assert all("epsilon_rmse" in c["normal"]["refined"]["errors"] for c in report["cases"])
    assert (tmp_path/"diagnostic/diagnostic_report.json").exists()
    assert len(report["pairwise_output_differences"]) == 3


def test_wrong_source_checkpoint_is_rejected_without_inference(tmp_path):
    path = tmp_path/"resume.pt"
    fixture_checkpoint(path)
    saved = torch.load(path, weights_only=False)
    saved["identity"]["source"]["models.py"] = "wrong-source"
    saved["identity_sha256"] = canonical_hash(saved["identity"])
    torch.save(saved, path)
    with pytest.raises(ValueError, match="source differs"):
        diagnostic.diagnose(path, ROOT/"data/pilot/frequency_pilot.mat", tmp_path/"out")


def test_geometric_centroid_reports_xyz_not_zyx():
    volume = torch.zeros(2, 8, 8, 8); volume[0] = 1.
    volume[:, 1, 2, 3] = torch.tensor([45., .8])
    result = diagnostic.describe_map(volume)
    assert result["centre_xyz_mm"] == [-7.5, -22.5, -37.5]
