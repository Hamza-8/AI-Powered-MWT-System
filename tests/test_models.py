import torch
import pytest
from microscan.models import MSRNN,SegmentationUNet,tiny_config
from microscan.pipeline import MicroScanPipeline,connected_components

torch.set_num_threads(2)

def inputs():
    y=torch.randn(1,2,10,6,6)
    mask=torch.ones(1,1,10,6,6,dtype=torch.bool)
    return y,mask

def test_native_decoder_shapes_mask_isolation_and_finite_gradients():
    model=MSRNN(tiny_config())
    y,mask=inputs()
    mask[:,:,2:5]=False
    y2=y.clone(); y2[:,:,2:5]=float('nan')
    a=model(y,mask); b=model(y2,mask)
    assert a['refined'].shape==(1,2,16,16,16)
    assert torch.equal(a['refined'],b['refined'])
    a['refined'].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.coarse.parameters())

def test_ab_identical_deployment_and_b_ignores_phi():
    model=MSRNN(tiny_config()).eval(); y,mask=inputs()
    class Forbidden:
        def __call__(self,*args): raise AssertionError('B must never call Phi')
    with torch.no_grad():
        a=model(y,mask,physics_mode='A')
        b=model(y,mask,physics_mode='B',phi=Forbidden())
    assert torch.equal(a['refined'],b['refined'])
    assert a['correction_depth']==b['correction_depth']
    with pytest.raises(ValueError,match='qualified'):
        model(y,mask,physics_mode='C')

def test_frequency_fixed_slots_all_k_same_parameter_count():
    model=MSRNN(tiny_config()).eval(); y,mask=inputs()
    count=sum(p.numel() for p in model.parameters())
    for k in range(3,11):
        selected=mask.clone(); selected[:,:,k:]=False
        with torch.no_grad(): output=model(y,selected)
        assert output['refined'].shape==(1,2,16,16,16)
        assert sum(p.numel() for p in model.parameters())==count

def test_spectral_reference_and_signed_monitoring():
    model=MSRNN(tiny_config()); y,mask=inputs()
    output=model(y,mask)
    spectral=model.spectral_properties(output,[.6,.85,1.05])
    assert spectral.shape==(1,2,3,16,16,16)
    assert torch.equal(spectral[:,:,1],output['refined'])
    loss=spectral[:,:,0].square().mean(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.spectral.parameters())
    base=torch.ones(1,2,16,16,16)
    change=model(y,mask,mode='monitoring',baseline=base)
    assert torch.allclose(change['absolute']-base,change['change'],atol=1e-6)
    missing=model(y,mask,mode='monitoring')
    assert not missing['absolute_valid'].any()
    assert torch.isnan(missing['absolute']).all()

def test_segmentation_detached_and_pipeline_routing():
    model=MSRNN(tiny_config()); segmentation=SegmentationUNet(widths=[4,8,8])
    y,mask=inputs()
    pipeline=MicroScanPipeline(model,segmentation)
    output=pipeline(y,mask)
    assert output['segmentation_logits'].shape==(1,5,16,16,16)
    output['segmentation_logits'].square().mean().backward()
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError,match='trained'):
        pipeline(y,mask,enable_declutter=True)
    assert output['routing']['declutterer']=='bypassed'

def test_all_missing_invalid_and_no_hidden_nan():
    model=MSRNN(tiny_config()); y,mask=inputs(); mask[:]=False; y[:]=float('nan')
    output=model(y,mask)
    assert not output['measurement_valid'].any()
    assert not output['absolute_valid'].any()
    assert torch.isfinite(output['refined']).all()

def test_pipeline_subspace_adapter_bindings_and_hidden_isolation():
    import numpy as np
    from microscan.subspace import MaskedSubspace
    cohort=np.random.default_rng(4).normal(size=(3,10,6,6)).astype(np.complex128)
    observed=np.ones(cohort.shape,dtype=bool)
    context={'hardware':'fixture','calibration':'fixture','frequencies_ghz':[.6+.05*i for i in range(10)]}
    subspace=MaskedSubspace(rank=1).fit(cohort,observed,context)
    model=MSRNN(tiny_config()).eval()
    pipeline=MicroScanPipeline(model,subspace=subspace,subspace_context=context)
    y,mask=inputs();mask[:,:,2:5]=False
    nan=y.clone();nan[:,:,2:5]=float('nan')
    with torch.no_grad():
        a=pipeline(y,mask,enable_subspace=True)
        b=pipeline(nan,mask,enable_subspace=True)
    assert torch.equal(a['refined'],b['refined'])
    assert a['routing']['subspace']=='active'
    pipeline.trained_routing={'declutterer':False,'subspace':False}
    with pytest.raises(ValueError,match='routing'):
        pipeline(y,mask,enable_subspace=True)
