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

def test_batched_spectral_queries_keep_query_axis():
    model=MSRNN(tiny_config());y,mask=inputs()
    output=model(y.expand(2,-1,-1,-1,-1),mask.expand(2,-1,-1,-1,-1))
    q=torch.tensor([[.6,.85],[.7,.85]])
    spectral=model.spectral_properties(output,q)
    assert spectral.shape==(2,2,2,16,16,16)
    assert torch.equal(spectral[:,:,1],output['refined'])

def test_c_explicit_fixture_original_domain_and_gradients():
    from microscan.physics import HeadsetForwardOperator,qualification_report
    phi=HeadsetForwardOperator(width=2,latent_dim=8)
    context={'tx_positions_mm':[[0.,0.,0.]]*6,'rx_positions_mm':[[0.,0.,0.]]*6,
             'source_excitation':[[1.,0.]]*6,'layout_id':'synthetic_C_mechanism_only',
             'calibration_id':'fixture','constant_spectrum':True,
             'observation_domain':phi.domain,'mechanism_fixture':True}
    names=['full_relative_rmse','differential_relative_rmse','noise_relative_rmse','worst_channel_relative_rmse','roi_relative_rmse']
    phi.qualification_report=qualification_report(phi,{n:0. for n in names},context,[.6+.05*i for i in range(10)],
        training_groups=['synthetic_train'],validation_groups=['synthetic_validation'],
        coverage=['healthy','lesion','positive_change','negative_change','sources','regions'],scientific=False,
        property_bounds=[[-1000.,1000.],[-1000.,1000.]])
    assert not phi.qualification_report['qualified']
    model=MSRNN(tiny_config());y,mask=inputs()
    with pytest.raises(ValueError,match='physics_observations'):
        model(y,mask,physics_mode='C',phi=phi,physics_context=context)
    output=model(y,mask,physics_mode='C',phi=phi,physics_context=context,physics_observations=y*.01)
    assert output['physics_active']=='C'
    output['refined'].mean().backward()
    assert phi.calls>0
    assert all(p.grad is None and not p.requires_grad for p in phi.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    with pytest.raises(ValueError,match='domain'):
        model(y,mask,physics_mode='C',phi=phi,physics_context={**context,'observation_domain':'decluttered_target'},physics_observations=y)

def test_monitoring_invalid_registration_and_baseline_gate():
    model=MSRNN(tiny_config());y,mask=inputs();baseline=torch.ones(1,2,16,16,16)
    bad=model(y,mask,baseline=baseline,mode='monitoring',context_validity={'baseline':True,'measurement_registration':False})
    assert not bad['measurement_valid'].any()
    assert not bad['absolute_valid'].any()
    valid_measurement=model(y,mask,baseline=baseline,mode='monitoring',context_validity={'baseline':False,'measurement_registration':True})
    assert valid_measurement['measurement_valid'].all()
    assert not valid_measurement['absolute_valid'].any()

def test_component_physical_coordinates_signed_growth_and_missing_roi():
    from microscan.pipeline import component_changes
    import numpy as np
    probability=torch.zeros(1,1,4,4,4);probability[0,0,1,2,3]=.9
    grid=(np.arange(4)*2.,np.arange(4)*3.,np.arange(4)*4.)
    component=connected_components(probability,grid,label_mode='binary')[0]
    assert component['center_xyz_mm']==[6.,6.,4.]
    assert component['volume_mm3']==24.
    baseline={**component,'volume_mm3':48.}
    result=component_changes([component],[baseline])
    assert result[0]['volume_change_mm3']==-24.
    assert component_changes([],[baseline],roi_valid=False)[0]['volume_change_mm3'] is None
