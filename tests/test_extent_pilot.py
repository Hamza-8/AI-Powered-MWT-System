"""Loss behavior and matched training mechanics; not validation improvement."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from microscan import frequency as f
from microscan.models import tiny_config
from train_extent_pilot import (BASE_OBJECTIVE, Experiment, augmented_objective,
                                extent_per_property, load_complete, soft_extent_loss)


def physical(contrast):
    return contrast*contrast.new_tensor([44.,.8]).reshape(1,2,1,1,1)+contrast.new_tensor([1.,0.]).reshape(1,2,1,1,1)


def masks():
    axis=torch.arange(9)-4
    small=((axis[:,None,None].abs()<=1)&(axis[None,:,None].abs()<=1)&(axis[None,None,:].abs()<=1))[None,None]
    large=((axis[:,None,None].abs()<=2)&(axis[None,:,None].abs()<=2)&(axis[None,None,:].abs()<=2))[None,None]
    return small,large


def test_perfect_empty_and_same_centre_wrong_extent():
    small,large=masks()
    a=physical(small.expand(-1,2,-1,-1,-1).float())
    b=physical(large.expand(-1,2,-1,-1,-1).float())
    assert soft_extent_loss(a,small)==0 and soft_extent_loss(b,large)==0
    assert soft_extent_loss(physical(torch.zeros_like(a)),torch.zeros_like(small))==0
    expected=1-2*float(small.sum())/float(small.sum()+large.sum())
    assert float(soft_extent_loss(b,small))==pytest.approx(expected,abs=1e-6)
    assert float(soft_extent_loss(a,large))==pytest.approx(expected,abs=1e-6)


def test_false_positive_above_one_and_missing_foreground_have_correct_gradients():
    small,_=masks();p=small.expand(-1,2,-1,-1,-1).float().clone()
    p[:,:,0,0,0]=2. # Would lose extent gradient if clamped to [0,1].
    p[:,:,4,4,4]=0. # Missing material at the true centre.
    prediction=physical(p).detach().requires_grad_()
    soft_extent_loss(prediction,small).backward()
    assert (prediction.grad[:,:,0,0,0]>0).all()
    assert (prediction.grad[:,:,4,4,4]<0).all()
    assert torch.isfinite(prediction.grad).all()


def test_channels_cannot_cancel_and_contrast_error_is_not_pure_volume():
    small,_=masks();p=small.expand(-1,2,-1,-1,-1).float().clone()
    p[:,0]*=2;p[:,1]*=0 # Mean contrast is correct, individual maps are wrong.
    each=extent_per_property(physical(p),small)
    assert each.shape==(1,2) and (each>0).all()
    half=small.expand(-1,2,-1,-1,-1).float()*.5
    assert soft_extent_loss(physical(half),small)>0


def test_amp_reductions_float32_and_finite_on_native_volume():
    torch.set_num_threads(1)
    p=torch.full((1,2,128,128,128),.7,dtype=torch.float16)
    prediction=physical(p).requires_grad_()
    foreground=torch.zeros(1,1,128,128,128,dtype=torch.bool);foreground[:,:,55:73,55:73,55:73]=True
    with torch.autocast('cpu',dtype=torch.bfloat16):loss=soft_extent_loss(prediction,foreground)
    assert loss.dtype==torch.float32 and torch.isfinite(loss)
    loss.backward();assert torch.isfinite(prediction.grad).all()


def test_scoped_objective_is_exact_addition_and_restores_on_error():
    small,_=masks();target=physical(small.expand(-1,2,-1,-1,-1).float())
    output={'coarse':target+.1,'refined':target+.2}
    original=f._objective
    expected=BASE_OBJECTIVE(output,target,small,.01)+.1*soft_extent_loss(output['refined'],small)
    with pytest.raises(RuntimeError):
        with augmented_objective(.1,1e-6):
            torch.testing.assert_close(f._objective(output,target,small,.01),expected,rtol=0,atol=0)
            raise RuntimeError('interrupted')
    assert f._objective is original


def test_config_changes_only_one_loss_and_single_seed():
    old=json.loads((ROOT/'configs/frequency_pilot_pattern_v1.json').read_text())
    new=json.loads((ROOT/'configs/frequency_pilot_extent_v1.json').read_text())
    assert new.pop('seeds')==[17];old.pop('seeds')
    assert new.pop('extent_loss')=={'name':'refined_contrast_squared_soft_dice_v1','stage':'refined','weight':.1,'smooth':1e-6}
    assert old==new


MAT=ROOT/'data/pilot_expanded/frequency_pilot.mat'
BASE=ROOT/'work/pattern_seed17_analysis/final.pt'
CSV=ROOT/'data/pilot/frequency_subsets.csv'


@pytest.mark.skipif(not all(p.exists() for p in (MAT,BASE)),reason='Private baseline/MAT stay local')
def test_real_optimizer_exact_resume_loss_accounting_identity_and_case3_sealed(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    cfg=json.loads((ROOT/'configs/frequency_pilot_extent_v1.json').read_text())
    cfg.update(grid_size=16,epochs=2,patience=2)
    cfg['model'].update(tiny_config())
    kwargs={'baseline':BASE,'mechanism_fixture':True}
    from train_expanded_pilot import ExpandedPilot
    touched=[];original=ExpandedPilot.__getitem__
    def track(self,index):
        if index<len(self):touched.append(self.ids[index])
        return original(self,index)
    monkeypatch.setattr(ExpandedPilot,'__getitem__',track)
    source=f.source_identity();baseline_hash=f.file_sha256(BASE);objective=f._objective
    run=Experiment(cfg,MAT,CSV,tmp_path/'resumed','cpu',**kwargs)
    with pytest.raises(ValueError,match='only K10 seed17'):run.run([9])
    with pytest.raises(ValueError,match='only K10 seed17'):run.run(seeds=[29])
    assert run.run(max_updates=1)[0]['status']=='running'
    assert f._objective is objective
    root=tmp_path/'resumed/K10/seed_17';initial=(root/'initial_metrics.json').read_bytes()
    run.run(resume=True,max_updates=1)
    result=run.run(resume=True)[0]
    final=torch.load(root/'final.pt',weights_only=True)
    resume=torch.load(root/'resume.pt',weights_only=False)
    assert final['optimizer_updates']==10 and final['epoch']==2
    assert all(torch.equal(v,resume['model'][key]) for key,v in final['model'].items())
    assert (root/'initial_metrics.json').read_bytes()==initial
    assert 3 not in touched and set(touched)=={1,2,4,5,6,7,8}
    history=json.loads((root/'training_history.json').read_text())['history']
    assert len(history)==2 and 'balanced_contrast_rmse' not in json.dumps(result)
    for row in history[-1]['validation']['per_case']:
        assert row['training_objective_float32']==pytest.approx(row['evaluation_objective_float32']+.1*row['extent_loss'],abs=1e-6)
    direct=Experiment(cfg,MAT,CSV,tmp_path/'direct','cpu',**kwargs);direct.run()
    other=torch.load(tmp_path/'direct/K10/seed_17/final.pt',weights_only=True)
    assert all(torch.equal(v,other['model'][key]) for key,v in final['model'].items())
    assert run.run(resume=True)[0]==result
    changed=copy.deepcopy(cfg);changed['extent_loss']['weight']=.2
    with pytest.raises(ValueError,match='identity'):
        Experiment(changed,MAT,CSV,tmp_path/'resumed','cpu',**kwargs).run(resume=True)
    with pytest.raises(ValueError,match='new output'):
        Experiment(cfg,MAT,CSV,BASE.parent,'cpu',**kwargs)
    with pytest.raises(ValueError,match='Completed native128'):
        load_complete(root/'final.pt')
    assert f.source_identity()==source and f.file_sha256(BASE)==baseline_hash and f._objective is objective
