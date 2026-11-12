import numpy as np
import pytest
import torch
from microscan.acquisition import (structured_mask,scan_list,roi_from_context,focus_selection,
    baseline_difference,focused_validity,structured_tx_holdout,acquisition_diversity)
from microscan.subspace import MaskedSubspace


def test_structured_scan_failure_and_actual_solve_budget():
    failed=np.zeros((3,4,4),bool);failed[0,1,2]=True
    m=structured_mask([.6,.8,1.],4,4,frequency_indices=[0,2],tx_indices=[2],rx_indices=[1,3],failed=failed)
    assert m.sum()==3
    s=scan_list(m,[.6,.8,1.],already_solved_tx=[(0,2),(2,2)])
    assert s['requested_source_excitations']==2 and s['new_source_excitations']==0
    assert s['prior_solves_saved']==0
    assert set(row['tx_physical'] for row in s['channels'])=={3}
    inferred=np.zeros_like(m);inferred[0,0,1]=True
    s=scan_list(m,[.6,.8,1.],inferred_mask=inferred)
    assert s['inferred_channels']==1


def test_roi_provenance_and_xyz_centres():
    roi,md=roi_from_context((20,20,20),[[-60,60]]*3,[25,-20,15],15,source='previous_frame_prediction')
    assert roi.any() and md['assessed_inside_only']
    with pytest.raises(ValueError):roi_from_context((20,)*3,[[-60,60]]*3,[0,0,0],10,source='current_target')
    f=focused_validity(roi,torch.ones(1,3,4,4),frame=5,refresh_every=5)
    assert f['request_full_scan'] and f['outside_roi_status']=='unassessed'
    assert focused_validity(None,torch.ones(1))['request_full_scan']


def test_focus_selection_no_target_and_labelled_fallback():
    a=np.array([[10,0,0],[0,10,0],[-10,0,0],[0,-10,0]])
    out=focus_selection(a,[9,0,0],1,2)
    assert out['tx_indices']==[0] and 'fallback' in out['method']


@pytest.mark.parametrize('delta',[0.,2.,-2.])
def test_signed_baseline_monitoring_and_registration(delta):
    a=np.ones((2,3,3),np.complex128);m=np.ones(a.shape,bool)
    reg=dict(same_subject=True,measurement_compatible=True,calibration_compatible=True,grid_compatible=True)
    out=baseline_difference(a+delta,a,m,m,reg)
    np.testing.assert_array_equal(out['difference'],np.ones_like(a)*delta)
    assert out['baseline_valid']
    assert baseline_difference(a,None,m,m,reg)['request_full_scan']
    reg['measurement_compatible']=False
    assert baseline_difference(a,a,m,m,reg)['difference'] is None


def test_structured_holdout_is_selected_observed_only():
    m=torch.zeros(1,10,3,3,dtype=torch.bool);m[:,[0,5,9]]=True
    kept,held=structured_tx_holdout(m,[1])
    assert not (kept&held).any() and torch.equal(kept|held,m)
    assert not held[:,1:5].any()


def test_subspace_observed_ridge_hidden_nan_isolation_roundtrip(tmp_path):
    rng=np.random.default_rng(7);v=np.array([1,2,3,4,5],dtype=np.complex128)
    a=rng.normal(size=(5,1,1,1))*v[None,:,None,None]
    m=np.ones_like(a,bool);m[0,1,0,0]=False
    b=a.copy();b[~m]=np.nan
    ctx={'hardware':'fixture','calibration':'empty','frequencies_ghz':[.6,.7,.8,.9,1.]}
    s=MaskedSubspace(1).fit(b,m,ctx)
    y=a[0].copy();ym=m[0];y[~ym]=np.nan
    r=s.transform(y,ym,ctx);assert np.isfinite(r['residual']).all()
    y[~ym]=1e99;r2=s.transform(y,ym,ctx);np.testing.assert_array_equal(r['residual'],r2['residual'])
    p=tmp_path/'subspace.npz';s.save(p);loaded=MaskedSubspace.load(p)
    np.testing.assert_array_equal(loaded.transform(y,ym,ctx)['residual'],r['residual'])
    np.testing.assert_array_equal(s.transform(y,ym,ctx,bypass=True)['residual'],np.where(ym,y,0))
    with pytest.raises(ValueError):s.transform(y,ym,{**ctx,'hardware':'other'})
    with pytest.raises(ValueError):MaskedSubspace(1).fit(a,m,ctx,split='test')
    with pytest.raises(ValueError):MaskedSubspace(1).fit(a[:1],m[:1],ctx)


def test_diversity_train_only_no_hidden_values():
    a=np.arange(30).reshape(3,2,5).astype(complex);m=np.ones_like(a,bool);m[:,0,0]=False
    a[~m]=np.nan
    out=acquisition_diversity(a,m);assert out['valid'] and out['independent_geometries']==3
    with pytest.raises(ValueError):acquisition_diversity(a,m,split='validation')
