"""The K1/K2 adapter preserves old sources and exact training/resume mechanics."""
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
from train_pattern_lowcount import CANDIDATES, Experiment, lowcount_subsets

MAT = ROOT / "data/pilot_expanded/frequency_pilot.mat"
CSV = ROOT / "data/pilot/frequency_subsets.csv"


def config():
    cfg = json.loads((ROOT / "configs/frequency_pilot_pattern_v1.json").read_text())
    cfg.update(grid_size=16, epochs=2, patience=2, amp=False)
    cfg["model"].update(tiny_config())
    return cfg


def test_explicit_subsets_are_scoped_and_restore_on_failure():
    original = f.read_subsets
    old = original(CSV)
    with pytest.raises(RuntimeError):
        with lowcount_subsets():
            assert f.read_subsets(CSV) == {**old, **CANDIDATES}
            raise RuntimeError("interrupted")
    assert f.read_subsets is original and f.read_subsets(CSV) == old


@pytest.mark.skipif(not MAT.exists(), reason="Private expanded MAT stays local")
@pytest.mark.parametrize('k', [1, 2])
def test_real_data_mask_scale_exact_resume_and_lineage(tmp_path, monkeypatch, k):
    torch.set_num_threads(1)
    cfg = config(); source = f.source_identity(); reader = f.read_subsets
    old_splits = copy.deepcopy(f.SPLITS); old_csv = f.file_sha256(CSV)
    from train_expanded_pilot import ExpandedPilot
    touched = []; original = ExpandedPilot.__getitem__
    def track(self, index):
        if index < len(self): touched.append(self.ids[index])
        return original(self, index)
    monkeypatch.setattr(ExpandedPilot, '__getitem__', track)
    run = Experiment(cfg, MAT, CSV, tmp_path / 'resumed', 'cpu')
    with pytest.raises(ValueError, match='1,2 only'):
        run.run([3])
    assert run.run([k], max_updates=1)[0]['status'] == 'running'
    root = tmp_path / f'resumed/K{k:02d}/seed_17'
    initial = (root / 'initial_metrics.json').read_bytes()
    result = run.run([k], resume=True)[0]
    final = torch.load(root / 'final.pt', weights_only=True)
    saved = torch.load(root / 'resume.pt', weights_only=False)
    assert result['optimizer_updates'] == 10 and result['epochs_executed'] == 2
    assert final['selected_indices'] == CANDIDATES[k]
    assert final['preprocessing']['fit_case_ids'] == [1,2,4,6,7]
    assert final['preprocessing']['observed_real_values'] == 5 * 2 * k * 18 * 18
    assert final['identity']['config']['lowcount_protocol']['indices_zero_based'][str(k)] == CANDIDATES[k]
    assert 'train_pattern_lowcount.py' in final['identity']['config']['experimental_source']
    assert f.canonical_hash(final['identity']) == final['identity_sha256']
    assert [r['case_id'] for r in result['validation']['per_case']] == [5,8]
    assert set(touched) == {1,2,4,5,6,7,8} and 3 not in touched
    assert (root / 'initial_metrics.json').read_bytes() == initial
    assert 'balanced_contrast_rmse' not in json.dumps(result)
    assert all(torch.equal(value, saved['model'][key]) for key,value in final['model'].items())
    direct = Experiment(cfg, MAT, CSV, tmp_path / 'direct', 'cpu')
    direct.run([k])
    expected = torch.load(tmp_path / f'direct/K{k:02d}/seed_17/final.pt', weights_only=True)
    assert all(torch.equal(value, expected['model'][key]) for key,value in final['model'].items())
    assert run.run([k], resume=True)[0] == result
    assert f.source_identity() == source and f.SPLITS == old_splits
    assert f.read_subsets is reader and f.file_sha256(CSV) == old_csv
    assert reader(CSV).keys() == set(range(3,11))


@pytest.mark.skipif(not MAT.exists(), reason="Private expanded MAT stays local")
def test_old_output_is_rejected_without_overwriting(tmp_path):
    old = tmp_path / 'K10/seed_17/result.json'
    old.parent.mkdir(parents=True)
    old.write_text('{"old": true}')
    with pytest.raises(FileExistsError, match='new output'):
        Experiment(config(), MAT, CSV, tmp_path, 'cpu').prepare()
    assert old.read_text() == '{"old": true}'
