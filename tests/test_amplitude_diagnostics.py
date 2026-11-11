"""Intervention semantics; output changes do not establish trained efficacy."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from diagnose_amplitude_conditioning import ablate, load_checkpoint
from pattern_amplitude_model import PatternAmplitudeMSRNN
from microscan.models import tiny_config
from microscan import frequency as f


@pytest.mark.parametrize("condition", ["raw_off","rms_off","both_off","coarse_off","enhancement_off"])
def test_ablation_removes_exact_branches_and_restores_hooks(condition):
    torch.set_num_threads(1)
    model=PatternAmplitudeMSRNN(tiny_config())
    coarse=model.coarse.conditioner
    enhancer=model.enhancement.measurement_encoder.conditioner
    features=torch.randn(1,6,10,6,6)
    mask=torch.ones(1,1,10,6,6,dtype=torch.bool)
    descriptors=torch.tensor([[.7,1.,1.]])
    pattern=torch.randn(1,coarse.amplitude[-1].out_features//2,4)
    before={k:v.clone() for k,v in model.state_dict().items()}
    baseline=coarse(pattern,features,mask,descriptors)
    with pytest.raises(RuntimeError,match="test interruption"):
        with ablate(model,condition):
            if condition in ("both_off","coarse_off"):
                torch.testing.assert_close(coarse(pattern,features,mask,descriptors),pattern,rtol=0,atol=0)
            elif condition=="raw_off":
                assert not coarse.raw_encoder(features,mask).any()
                assert not enhancer.raw_encoder(features,mask).any()
            elif condition=="rms_off":
                assert not coarse.amplitude(descriptors).any()
                assert not enhancer.amplitude(descriptors).any()
            elif condition=="enhancement_off":
                torch.testing.assert_close(coarse(pattern,features,mask,descriptors),baseline,rtol=0,atol=0)
            raise RuntimeError("test interruption")
    torch.testing.assert_close(coarse(pattern,features,mask,descriptors),baseline,rtol=0,atol=0)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("k,seed,epoch,updates", [(9,17,300,1500),(10,29,300,1500),(10,17,299,1495)])
def test_diagnostic_rejects_other_runs_and_incomplete_weights(tmp_path,k,seed,epoch,updates):
    identity={"splits":{"reference":[0],"train":[1,2,4,6,7],"validation":[5,8],"sealed_test":[3]},"k":k,"seed":seed}
    state={"identity":identity,"identity_sha256":f.canonical_hash(identity),"epoch":epoch,"optimizer_updates":updates}
    path=tmp_path/"final.pt";torch.save(state,path)
    with pytest.raises(ValueError,match="completed native K10"):
        load_checkpoint(path)
