"""Controlled routing mechanics; no claim that new routing fixes validation size."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import train_pattern_pilot as legacy
from pattern_amplitude_model import PatternAmplitudeMSRNN
from postnorm_amplitude_model import PostNormAmplitudeMSRNN, PostNormFiLM
from train_postnorm_pilot import Experiment, model_factory
from microscan import frequency as f
from microscan.models import tiny_config


def model_config():
    c=tiny_config()
    c["postnorm_amplitude"]={"name":"postnorm_amplitude_v1","hidden":32,"init_std":.02}
    return c


def test_only_added_decoder_modulation_changes_paired_initialization_and_is_after_norm():
    torch.set_num_threads(1)
    torch.manual_seed(17);old=PatternAmplitudeMSRNN(model_config())
    old_rng=torch.get_rng_state().clone()
    torch.manual_seed(17);new=PostNormAmplitudeMSRNN(model_config())
    assert torch.equal(torch.get_rng_state(),old_rng)
    for key,value in old.state_dict().items():
        assert torch.equal(value,new.state_dict()[key])
    signal=torch.randn(1,2,10,6,6)
    mask=torch.ones(1,1,10,6,6,dtype=torch.bool)
    hooks=[];events=[]
    for i,mod in enumerate(new.coarse.postnorm):
        hooks.append(new.coarse.core.decoder[3*i+1].register_forward_hook(lambda m,a,o,i=i:events.append(('norm',i))))
        hooks.append(mod.register_forward_hook(lambda m,a,o,i=i:events.append(('film',i))))
        hooks.append(new.coarse.core.decoder[3*i+2].register_forward_hook(lambda m,a,o,i=i:events.append(('activation',i))))
    with torch.inference_mode():new(signal,mask)
    assert events==[(name,i) for i in range(3) for name in ('norm','film','activation')]
    for hook in hooks:hook.remove()
    # Bypassing only the added stages exactly recovers the original model.
    for mod in new.coarse.postnorm:
        hooks.append(mod.register_forward_hook(lambda m,args,out:args[0]))
    with torch.inference_mode():
        torch.testing.assert_close(new(signal,mask)['refined'],old(signal,mask)['refined'],rtol=0,atol=0)
    for hook in hooks:hook.remove()


def test_late_amplitude_has_direct_effect_and_finite_amp_gradients_with_mask_isolation():
    torch.set_num_threads(1)
    torch.manual_seed(17);model=PostNormAmplitudeMSRNN(model_config())
    signal=torch.randn(1,2,10,6,6)
    mask=torch.ones(1,1,10,6,6,dtype=torch.bool);mask[:,:,3:7]=False
    dirty=signal.clone();dirty[:,:,3:7]=float('nan')
    with torch.inference_mode():
        torch.testing.assert_close(model(signal,mask)['refined'],model(dirty,mask)['refined'],rtol=0,atol=0)
    # Isolate the added FiLM: a fixed spatial feature receives distinct log RMS.
    stage=model.coarse.postnorm[-1]
    normalized=torch.randn(1,4,16,16,16)
    low=torch.tensor([[-1.,1.,1.]]);high=torch.tensor([[1.,1.,1.]])
    assert (stage(normalized,low)-stage(normalized,high)).abs().max()>1e-4
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=model(signal,mask)
        target=torch.ones_like(out['refined']);target[:,1]=0
        loss=f._objective(out,target,torch.ones(1,1,16,16,16,dtype=torch.bool))
    loss.backward()
    for stage in model.coarse.postnorm:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in stage.parameters())
        assert sum(p.grad.abs().sum() for p in stage.parameters())>0


def test_scoped_factory_restores_legacy_model_on_exception():
    original=legacy.PatternAmplitudeMSRNN
    with pytest.raises(RuntimeError):
        with model_factory():
            assert legacy.PatternAmplitudeMSRNN is PostNormAmplitudeMSRNN
            raise RuntimeError('interrupted')
    assert legacy.PatternAmplitudeMSRNN is original


def test_config_changes_only_routing_and_single_seed_not_objective_or_budget():
    old=json.loads((ROOT/'configs/frequency_pilot_pattern_v1.json').read_text())
    new=json.loads((ROOT/'configs/frequency_pilot_postnorm_v1.json').read_text())
    assert new.pop('seeds')==[17];old.pop('seeds')
    assert new['model'].pop('postnorm_amplitude')=={'name':'postnorm_amplitude_v1','hidden':32,'init_std':.02}
    assert old==new


MAT=ROOT/'data/pilot_expanded/frequency_pilot.mat'
BASE=ROOT/'work/pattern_seed17_analysis/final.pt'
EVIDENCE=ROOT/'work/amplitude_ablation_v1/report.json'


@pytest.mark.skipif(not all(p.exists() for p in (MAT,BASE,EVIDENCE)),reason='Private baseline and completed ablations stay local')
def test_exact_resume_final_weights_guards_and_no_sealed_sample(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    cfg=json.loads((ROOT/'configs/frequency_pilot_postnorm_v1.json').read_text())
    cfg.update(grid_size=16,epochs=2,patience=2)
    cfg['model'].update(tiny_config())
    kwargs=dict(baseline=BASE,evidence=EVIDENCE,mechanism_fixture=True)
    from train_expanded_pilot import ExpandedPilot
    touched=[];original=ExpandedPilot.__getitem__
    def track(self,index):
        if index<len(self):touched.append(self.ids[index])
        return original(self,index)
    monkeypatch.setattr(ExpandedPilot,'__getitem__',track)
    source=f.source_identity();baseline_hash=f.file_sha256(BASE)
    run=Experiment(cfg,MAT,ROOT/'data/pilot/frequency_subsets.csv',tmp_path/'resumed','cpu',**kwargs)
    assert run.run(max_updates=1)[0]['status']=='running'
    run.run(resume=True,max_updates=1)
    result=run.run(resume=True)[0]
    root=tmp_path/'resumed/K10/seed_17'
    final=torch.load(root/'final.pt',map_location='cpu',weights_only=True)
    resume=torch.load(root/'resume.pt',map_location='cpu',weights_only=False)
    assert result['optimizer_updates']==10 and result['epochs_executed']==2
    assert all(torch.equal(v,resume['model'][key]) for key,v in final['model'].items())
    direct=Experiment(cfg,MAT,ROOT/'data/pilot/frequency_subsets.csv',tmp_path/'direct','cpu',**kwargs)
    direct.run()
    other=torch.load(tmp_path/'direct/K10/seed_17/final.pt',weights_only=True)
    assert all(torch.equal(v,other['model'][key]) for key,v in final['model'].items())
    assert 3 not in touched and set(touched)=={1,2,4,5,6,7,8}
    assert f.source_identity()==source and f.file_sha256(BASE)==baseline_hash
    assert legacy.PatternAmplitudeMSRNN is PatternAmplitudeMSRNN
    assert run.run(resume=True)[0]==result
    for counts,seeds in (([9],[17]),([10],[29])):
        with pytest.raises(ValueError,match='only K10 seed17'):run.run(counts,seeds=seeds)
    changed=copy.deepcopy(cfg);changed['gradient_weight']=.02
    with pytest.raises(ValueError,match='protocol differs'):
        Experiment(changed,MAT,ROOT/'data/pilot/frequency_subsets.csv',tmp_path/'bad','cpu',**kwargs)
    with pytest.raises(ValueError,match='new output'):
        Experiment(cfg,MAT,ROOT/'data/pilot/frequency_subsets.csv',BASE.parent,'cpu',**kwargs)
