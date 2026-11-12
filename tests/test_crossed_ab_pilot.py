"""Split isolation, matched schedule, and exact interrupted-run recovery."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from microscan import frequency as f
from microscan.models import tiny_config
from train_crossed_ab_pilot import BUDGET, SPLITS, Experiment, build_config, next_sample_index, validate_config

MAT = ROOT / "data/pilot_expanded/frequency_pilot.mat"
CSV = ROOT / "data/pilot/frequency_subsets.csv"


def config():
    return build_config(json.loads((ROOT / "configs/frequency_pilot_pattern_v1.json").read_text()))


def test_allocation_is_fully_crossed_and_changes_no_model_or_loss():
    old = json.loads((ROOT / "configs/frequency_pilot_pattern_v1.json").read_text())
    new = config()
    validate_config(new)
    assert new["training_budget"] == BUDGET
    assert new["epochs"] * 4 == old["epochs"] * 5 == 1500
    assert new["expanded_protocol"]["splits"] == SPLITS
    for key in ("model", "learning_rate", "weight_decay", "gradient_weight", "gradient_clip",
                "amp", "batch_size", "grid_size", "representation_experiment"):
        assert new[key] == old[key]
    for cid in (1, 3):
        assert cid not in SPLITS["train"] + SPLITS["validation"]
    with pytest.raises(ValueError, match="allocation"):
        changed = copy.deepcopy(new)
        changed["expanded_protocol"]["splits"]["train"].append(1)
        validate_config(changed)


def test_sampling_is_balanced_and_cosine_matches_every_original_update():
    torch.manual_seed(17)
    state = {"order": None, "position": 0, "passes_completed": 0}
    indices = [next_sample_index(state, 4) for _ in range(1500)]
    assert all(set(indices[i:i+4]) == set(range(4)) for i in range(0, 1500, 4))
    assert [indices.count(i) for i in range(4)] == [375] * 4
    assert state == {"order": None, "position": 0, "passes_completed": 375}
    old_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    new_opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=2e-4)
    old = torch.optim.lr_scheduler.CosineAnnealingLR(old_opt, T_max=300)
    new = torch.optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=BUDGET["scheduler_t_max"])
    original_lrs = []
    for _ in range(300):
        for _ in range(5):
            original_lrs.append(old_opt.param_groups[0]["lr"])
            old_opt.step()
        old.step()
    actual_lrs = []
    for update in range(1500):
        actual_lrs.append(new_opt.param_groups[0]["lr"])
        new_opt.step()
        if (update + 1) % BUDGET["updates_per_scheduler_step"] == 0:
            new.step()
    assert original_lrs == actual_lrs


@pytest.mark.skipif(not MAT.exists(), reason="Private MAT stays local")
def test_real_data_resume_boundary_is_exact_and_excluded_cases_never_materialized(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = config()
    cfg.update(grid_size=16, epochs=5, patience=5, amp=False)
    cfg["model"].update(tiny_config())
    cfg["training_budget"].update(optimizer_updates=20, scheduler_t_max=4)
    source, original_splits, objective = f.source_identity(), copy.deepcopy(f.SPLITS), f._objective
    from train_expanded_pilot import ExpandedPilot
    touched = []
    getitem = ExpandedPilot.__getitem__
    def track(self, index):
        if index < len(self):
            touched.append(self.ids[index])
        return getitem(self, index)
    monkeypatch.setattr(ExpandedPilot, "__getitem__", track)
    run = Experiment(cfg, MAT, CSV, tmp_path / "resumed", "cpu")
    with pytest.raises(PermissionError):
        run.dataset(MAT, "test", range(10), cfg, allow_test=True)
    with pytest.raises(PermissionError):
        run.dataset(MAT, "unused", range(10), cfg)
    with pytest.raises(ValueError, match="only K10 seed17"):
        run.run([9])
    assert run.run(max_updates=3)[0]["status"] == "running"
    root = tmp_path / "resumed/K10/seed_17"
    initial = (root / "initial_metrics.json").read_bytes()
    evaluate = run.evaluate
    def interrupted_evaluation(*args, **kwargs):
        raise RuntimeError("Simulated interruption after the fifth update")
    monkeypatch.setattr(run, "evaluate", interrupted_evaluation)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        run.run(resume=True)
    boundary = torch.load(root / "resume.pt", weights_only=False)
    assert boundary["progress"]["updates"] == 5 and boundary["progress"]["evaluations"] == 0
    assert boundary["scheduler"]["last_epoch"] == 1
    monkeypatch.setattr(run, "evaluate", evaluate)
    result = run.run(resume=True)[0]
    saved = torch.load(root / "resume.pt", weights_only=False)
    final = torch.load(root / "final.pt", weights_only=True)
    assert result["optimizer_updates"] == 20 and result["epochs_executed"] == 5
    assert result["case_update_counts"] == {str(cid): 5 for cid in SPLITS["train"]}
    assert saved["preprocessing"]["fit_case_ids"] == [2, 4, 5, 6]
    assert saved["scheduler"]["last_epoch"] == 4
    assert [r["case_id"] for r in result["validation"]["per_case"]] == [7, 8]
    assert set(touched) == {2, 4, 5, 6, 7, 8}
    assert (root / "initial_metrics.json").read_bytes() == initial
    history = json.loads((root / "training_history.json").read_text())["history"]
    assert [h["optimizer_updates"] for h in history] == [5, 10, 15, 20]
    assert "balanced_contrast_rmse" not in json.dumps(result)
    assert all(torch.equal(value, saved["model"][key]) for key, value in final["model"].items())
    assert not (root / "best.pt").exists()
    direct = Experiment(cfg, MAT, CSV, tmp_path / "direct", "cpu")
    direct.run()
    expected = torch.load(tmp_path / "direct/K10/seed_17/final.pt", weights_only=True)
    assert all(torch.equal(value, expected["model"][key]) for key, value in final["model"].items())
    assert run.run(resume=True)[0] == result
    assert f.source_identity() == source and f.SPLITS == original_splits and f._objective is objective
    with pytest.raises(ValueError, match="protocol differs"):
        changed = copy.deepcopy(cfg)
        changed["learning_rate"] *= 2
        Experiment(changed, MAT, CSV, tmp_path / "resumed", "cpu").prepare()


@pytest.mark.skipif(not MAT.exists(), reason="Private MAT stays local")
def test_old_output_is_rejected_before_writing(tmp_path):
    (tmp_path / "K10/seed_17").mkdir(parents=True)
    old = tmp_path / "K10/seed_17/resume.pt"
    old.write_bytes(b"untouched original checkpoint")
    with pytest.raises(FileExistsError, match="new output"):
        Experiment(config(), MAT, CSV, tmp_path, "cpu").prepare()
    assert old.read_bytes() == b"untouched original checkpoint"
