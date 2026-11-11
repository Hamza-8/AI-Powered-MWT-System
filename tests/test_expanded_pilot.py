"""Real expanded-MAT and bounded training mechanics, not scientific accuracy."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from microscan import frequency as f
from microscan.data import load_mat, voxel_grid
from microscan.models import tiny_config

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("train_expanded_pilot",ROOT/"scripts/train_expanded_pilot.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
MAT = ROOT/"data/pilot_expanded/frequency_pilot.mat"
CSV = ROOT/"data/pilot/frequency_subsets.csv"


def config():
    return json.loads((ROOT/"configs/frequency_pilot_expanded.json").read_text())


def test_real_extension_preserves_original_data_and_crosses_sizes():
    m = load_mat(MAT)
    old = load_mat(ROOT/"data/pilot/frequency_pilot.mat")
    assert m["data"]["S"].shape == (18,18,10,9)
    np.testing.assert_array_equal(m["data"]["S"][:,:,:,:5],old["data"]["S"])
    ds = runner.ExpandedPilot(MAT,"train",range(10),config())
    assert ds.ids == [1,2,4,5,6]
    pairs = {}
    for sample in ds:
        geometry = sample["metadata"]["geometry"]
        pairs.setdefault(tuple(geometry["centre_xyz_mm"]),set()).add(geometry["radius_mm"])
    assert pairs[(25.,-20.,15.)] == pairs[(-25.,15.,10.)] == {15.,25.}
    assert runner.ExpandedPilot(MAT,"validation",range(10),config()).ids == [7,8]
    with pytest.raises(PermissionError):
        runner.ExpandedPilot(MAT,"test",range(10),config())


def test_expanded_complex128_subtraction_fixed_slots_and_translated_labels():
    ds = runner.ExpandedPilot(MAT,"validation",[0,5,9],config())
    sample = ds[0]
    expected = (ds.s[:,:,:,7]-ds.s[:,:,:,0]).transpose(2,0,1)
    np.testing.assert_allclose(sample["signals"][0,[0,5,9]],expected.real[[0,5,9]],rtol=1e-6,atol=1e-9)
    assert sample["mask"].sum() == 3*18*18
    assert not sample["signals"][:,1:5].any()
    x,y,z = voxel_grid()
    foreground = sample["object_mask"][0].numpy()
    np.testing.assert_allclose([x[foreground].mean(),y[foreground].mean(),z[foreground].mean()],[10,25,-20],atol=.1)


def test_original_test_cannot_be_moved_and_wrong_mat_rejected():
    cfg = config(); cfg["expanded_protocol"]["splits"]["train"].append(3)
    with pytest.raises(ValueError,match="allocation"):
        runner.validate_allocation(cfg)
    with pytest.raises(ValueError,match=r"\[18,18,10,9\]"):
        runner.ExpandedPilot(ROOT/"data/pilot/frequency_pilot.mat","train",range(10),config())


def test_prepare_does_not_materialize_test_or_change_legacy_protocol(tmp_path,monkeypatch):
    touched = []
    original = runner.ExpandedPilot.__getitem__
    def track(self,index):
        if index < len(self): touched.append(self.ids[index])
        return original(self,index)
    monkeypatch.setattr(runner.ExpandedPilot,"__getitem__",track)
    cfg = config(); cfg.update(model=tiny_config(),grid_size=16)
    experiment = runner.Experiment(cfg,MAT,CSV,tmp_path,"cpu")
    source = f.source_identity(); old_splits = f.SPLITS.copy()
    report = experiment.prepare()
    assert report["counts"] == {"training":5,"validation":2,"sealed_test":1}
    assert touched == [1,2,4,5,6,7,8] and not report["test_accessed"]
    assert f.SPLITS == old_splits and f.source_identity() == source


def test_bounded_expanded_optimizer_resume_history_initialization_and_identity(tmp_path):
    torch.set_num_threads(1)
    cfg = config(); cfg.update(model=tiny_config(),grid_size=16,seeds=[17],epochs=1,patience=1,amp=False)
    experiment = runner.Experiment(cfg,MAT,CSV,tmp_path/"interrupted","cpu")
    first = experiment.run(max_updates=1)[0]
    assert first["status"] == "running"
    initial = tmp_path/"interrupted/K10/seed_17/initial_metrics.json"
    initial_bytes = initial.read_bytes()
    record = json.loads(initial_bytes)
    assert record["optimizer_updates"] == 0
    assert [row["case_id"] for row in record["training"]["per_case"]] == [1,2,4,5,6]
    experiment.run(resume=True,max_updates=1)
    result = experiment.run(resume=True)[0]
    saved = torch.load(tmp_path/"interrupted/K10/seed_17/resume.pt",weights_only=False)
    assert saved["progress"]["updates"] == 5 and result["status"] == "completed"
    assert saved["preprocessing"]["fit_case_ids"] == [1,2,4,5,6]
    assert saved["splits"]["sealed_test"] == [3] and saved["identity"]["protocol_version"] == 2
    history = saved["progress"]["history"][0]["validation"]
    assert [row["case_id"] for row in history["per_case"]] == [7,8]
    assert [row["case_id"] for row in history["training_fit"]["per_case"]] == [1,2,4,5,6]
    assert initial.read_bytes() == initial_bytes
    direct = runner.Experiment(cfg,MAT,CSV,tmp_path/"direct","cpu")
    direct.run()
    uninterrupted = torch.load(tmp_path/"direct/K10/seed_17/resume.pt",weights_only=False)
    assert all(torch.equal(value,uninterrupted["model"][key]) for key,value in saved["model"].items())
    before = f.source_identity()
    assert experiment.run(resume=True)[0]["status"] == "completed"
    assert f.source_identity() == before and f.SPLITS == {"reference":[0],"train":[1,4],"validation":[2],"sealed_test":[3]}
