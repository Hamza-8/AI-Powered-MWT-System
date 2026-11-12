import json
from pathlib import Path
import numpy as np
import pytest
import torch
from microscan.data import (PilotDataset, ObservedScale, calibrate, complex_to_channels,
    channels_to_complex, sphere_target, load_mat, voxel_grid, convert_pilot, ManifestDataset,
    validate_manifest, collate_samples)

PILOT=Path(__file__).parents[1]/'data/pilot/frequency_pilot.mat'


def test_real_mat_axes_ports_reference_and_completion():
    ds=PilotDataset(PILOT)
    a=ds[0]
    assert ds.s.dtype==np.complex128 and ds.s.shape==(18,18,10,5)
    assert a['signals'].shape==(2,10,18,18)
    assert a['metadata']['case_id']==1
    expected=(ds.s[:,:,:,1]-ds.s[:,:,:,0]).transpose(2,0,1)
    np.testing.assert_allclose(channels_to_complex(a['signals']),expected,rtol=1e-7,atol=1e-8)
    # Non-symmetric axis checks preserve reflection channels and physical order.
    np.testing.assert_allclose(channels_to_complex(a['signals'])[4,2,13],expected[4,2,13],rtol=1e-7)
    assert a['mask'][0,4,2,2]


def test_cell_centred_translated_sphere_xyz_zyx():
    ds=PilotDataset(PILOT,split='validation');a=ds[0]
    mask=a['object_mask'][0].numpy();x,y,z=voxel_grid()
    centre=np.array([x[mask].mean(),y[mask].mean(),z[mask].mean()])
    np.testing.assert_allclose(centre,[25,-20,15],atol=.1)
    assert x.min()==-60+.9375/2 and x.max()==60-.9375/2
    assert torch.all(a['target'][0][a['object_mask'][0]]==45)
    assert torch.all(a['target'][1][~a['object_mask'][0]]==0)


def test_reference_subtracts_in_complex128_before_cast():
    current=np.array([[[1e8+1e-3+1j*(1e8+2e-3)]]],np.complex128)
    reference=np.array([[[1e8+1j*1e8]]],np.complex128)
    out,m=calibrate(current,reference)
    assert np.all(out.real>0) and np.all(out.imag>0)
    a=complex_to_channels(out)
    np.testing.assert_allclose(a.numpy().ravel(),[.001,.002],rtol=1e-5)


def test_mask_intersection_hides_nans_measured_zero():
    c=np.array([[[0, np.nan],[5+2j,8]]]);r=np.array([[[0,4],[np.nan,1]]])
    cm=np.array([[[1,0],[1,1]]],bool);rm=np.array([[[1,1],[0,1]]],bool)
    d,m=calibrate(c,r,cm,rm)
    assert m[0,0,0] and d[0,0,0]==0
    np.testing.assert_array_equal(m,cm&rm)
    assert np.isfinite(d).all() and d[0,1,1]==7
    with pytest.raises(ValueError):calibrate(c,r)


def test_frequency_fixed_slots_and_test_seal():
    a=PilotDataset(PILOT,frequency_indices=[0,4,9])[0]
    assert a['signals'].shape==(2,10,18,18)
    assert a['mask'].sum()==3*18*18
    assert not a['signals'][:,1:4].any()
    with pytest.raises(PermissionError):PilotDataset(PILOT,split='test')


def test_prior_spectral_context_is_separate_from_supervised_labels(tmp_path):
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text());s=m['samples'][0]
    prior=np.ones((2,10,8,8,8),np.float32)
    np.save(tmp_path/'prior.npy',prior)
    s['baseline_spectral_context_path']='prior.npy'
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='provenance'):ManifestDataset(p)[0]
    s['baseline_spectral_provenance']='registered_prior_prediction';p.write_text(json.dumps(m))
    a=ManifestDataset(p)[0]
    assert a['baseline_spectral'].shape==(2,10,8,8,8)
    assert 'baseline_spectral_target' not in a


def test_observed_training_rms_hidden_nans_and_roundtrip():
    a=PilotDataset(PILOT,frequency_indices=[0,9])[0]
    b={**a,'signals':a['signals'].clone()}
    b['signals'][:,1:9]=float('nan')
    s=ObservedScale.fit([a]);s2=ObservedScale.fit([b]);assert s.scale==s2.scale
    transformed=s.transform(b['signals'],b['mask']);assert torch.isfinite(transformed).all()
    torch.testing.assert_close(s.inverse(transformed),a['signals'])
    assert ObservedScale.from_state_dict(s.state_dict()).scale==s.scale
    with pytest.raises(ValueError):ObservedScale.fit([PilotDataset(PILOT,'validation')[0]])
    with pytest.raises(ValueError):ObservedScale.fit([{**a,'mask':torch.zeros_like(a['mask'])}])


def test_hdf5_matlab_complex_axes_struct_and_refs(tmp_path):
    h5py=pytest.importorskip('h5py')
    p=tmp_path/'v73.mat';expected=np.arange(24).reshape(2,3,4)+1j
    with h5py.File(p,'w') as f:
        g=f.create_group('data');d=np.empty(expected.T.shape,dtype=[('real','f8'),('imag','f8')]);d['real']=expected.T.real;d['imag']=expected.T.imag
        g.create_dataset('S',data=d)
        txt=g.create_dataset('axes',data=np.array([[ord(c)] for c in 'FRxTx'],np.uint16));txt.attrs['MATLAB_class']=np.bytes_('char')
        refer=f.create_dataset('single',shape=(1,1),dtype=h5py.ref_dtype);refer[0,0]=g.ref
    out=load_mat(p)
    np.testing.assert_array_equal(out['data']['S'],expected)
    assert out['data']['axes']=='FRxTx'
    np.testing.assert_array_equal(out['single']['S'],expected)


def test_convert_manifest_real_roundtrip_group_isolation(tmp_path):
    path=convert_pilot(PILOT,tmp_path/'export',grid_size=16)
    a=ManifestDataset(path)[0];b=PilotDataset(PILOT,grid_size=16)[0]
    torch.testing.assert_close(a['signals'],b['signals'])
    torch.testing.assert_close(a['target'],b['target'])
    manifest=json.loads(path.read_text());assert validate_manifest(manifest)['samples']==3
    manifest['samples'][2]['geometry_id']=manifest['samples'][0]['geometry_id']
    with pytest.raises(ValueError,match='leakage'):validate_manifest(manifest)


def test_manifest_missing_labels_not_fake_background(tmp_path):
    p=convert_pilot(PILOT,tmp_path,grid_size=8)
    a=ManifestDataset(p)[0]
    assert 'labels' not in a and 'label_validity' not in a
    out=collate_samples([a,a]);assert out['signals'].shape==(2,2,10,18,18)
    assert isinstance(out['metadata'],list)


def test_material_spectral_labels_are_frequency_specific(tmp_path):
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text())
    ids=np.zeros((8,8,8),np.int16);ids[2:5]=1
    table=np.ones((2,10,2),np.float32);table[1,:,0]=np.arange(10)+30;table[1,:,1]=np.arange(10)*.1
    np.save(tmp_path/'ids.npy',ids);np.save(tmp_path/'table.npy',table)
    m['samples'][0]['material_ids_path']='ids.npy';m['samples'][0]['spectral_table_path']='table.npy'
    p.write_text(json.dumps(m));a=ManifestDataset(p)[0]
    assert a['spectral_target'].shape==(2,10,8,8,8)
    assert a['spectral_target'][0,4,3,3,3]==34
    assert a['spectral_target'][0,4,0,0,0]==1


def test_conversion_keeps_test_case_sealed(tmp_path,monkeypatch):
    accessed=[];original=PilotDataset.__getitem__
    def tracked(self,index):
        accessed.append(self.ids[index]);return original(self,index)
    monkeypatch.setattr(PilotDataset,'__getitem__',tracked)
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text())
    assert 3 not in accessed and m['sealed_test']['status']=='sealed'
    bad=tmp_path/'selection.json';bad.write_text(json.dumps({'status':'pending_selection'}))
    with pytest.raises(PermissionError):convert_pilot(PILOT,tmp_path/'bad',grid_size=8,selection_path=bad)


def test_augmentation_grouping_structured_masks_and_signed_corruption():
    from microscan.data import TrainingAugmentation
    ds=PilotDataset(PILOT,frequency_indices=[0,5,9],grid_size=8)
    aug=TrainingAugmentation(ds,{'tx_keep':4,'rx_keep':5,'corruption':{'gain_std':.02,'phase_std_rad':.02,'noise_std':.001,'reference_drift_std':.001}},seed=3)
    a=aug[0];b=ds[0]
    assert a['mask'].sum()==3*4*5 and not a['mask'][:,1:5].any()
    torch.testing.assert_close(a['target'],b['target'])
    assert a['metadata']['augmentation_geometry_id']==b['metadata']['geometry_id']
    assert a['metadata']['corruption']['validated_sdr_noise'] is False
    assert torch.isfinite(a['signals']).all()
    aug.set_epoch(1);assert not torch.equal(a['mask'],aug[0]['mask'])


def test_noise2noise_requires_independent_conditional_target_assumptions():
    from microscan.data import repeated_acquisition_target
    a=PilotDataset(PILOT,grid_size=8)[0];a['repeated_signals']=a['signals'].clone()
    with pytest.raises(ValueError):repeated_acquisition_target(a)
    a['metadata'].update(acquisition_id='frame1_repeat1',repeated_acquisition_id='frame1_repeat2',repeat_assumptions={'same_clean_state':True,'conditionally_unbiased_target':True,'independent_given_clean_state':True,'shared_drift':False})
    target,mask=repeated_acquisition_target(a);torch.testing.assert_close(target,a['signals'])
    a['metadata']['repeat_assumptions']['shared_drift']=True
    with pytest.raises(ValueError):repeated_acquisition_target(a)


def test_inference_manifest_needs_no_supervised_target(tmp_path):
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text())
    md=m['samples'][0];md['split']='inference';md.pop('target_path')
    m['samples']=[md];p.write_text(json.dumps(m))
    a=ManifestDataset(p,split='inference')[0]
    assert 'target' not in a and a['mode']=='snapshot'
    assert a['context_validity']['observations'] and not a['context_validity']['baseline']


def test_context_masks_baseline_original_domain_and_inference_no_label_conditioning(tmp_path):
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text());s=m['samples'][0]
    s['domain']='raw_s';s['baseline_signal_path']=s['signal_path'];s['baseline_target_path']=s['target_path']
    s['baseline_valid']=True;s['registration']=dict(same_subject=True,measurement_compatible=True,calibration_compatible=True,grid_compatible=True)
    s['physics_context']={'layout_id':'headset18','calibration_id':'empty','tx_positions_mm':[[0.,0.,0.]]*18,'rx_positions_mm':[[0.,0.,0.]]*18,'source_excitation':[[1.,0.]]*18,'constant_spectrum':True}
    p.write_text(json.dumps(m));a=ManifestDataset(p)[0]
    assert a['domain']=='calibrated_delta_s' and a['metadata']['stored_measurement_domain']=='raw_s'
    torch.testing.assert_close(a['baseline_observations'],a['signals']);assert not a['difference'].any()
    assert a['context_validity']['baseline']
    batched=collate_samples([a,a]);assert batched['physics_context']['tx_positions_mm'].shape==(2,18,3)
    assert batched['physics_context']['layout_id']=='headset18'
    s['roi_path']=s['target_path'];s['roi_source']='current_target'
    with pytest.raises(ValueError,match='ROI'):validate_manifest(m)


def test_prior_roi_augmentation_keeps_target_and_never_wraps(tmp_path):
    from microscan.data import TrainingAugmentation
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text());s=m['samples'][0]
    roi=np.zeros((1,8,8,8),np.float32);roi[:,2:5,2:5,2:5]=1;np.save(tmp_path/'roi.npy',roi)
    s['roi_path']='roi.npy';s['roi_source']='previous_frame_prediction';p.write_text(json.dumps(m))
    ds=ManifestDataset(p);aug=TrainingAugmentation(ds,{'focus_shift_mm':5,'focus_radius_scale':[.8,1.3]},seed=10)
    a=aug[0];torch.testing.assert_close(a['target'],ds[0]['target'])
    assert a['metadata']['roi_source']=='previous_frame_prediction'
    assert a['roi'].shape==(1,8,8,8) and a['roi'].dtype==torch.float32
    assert a['metadata']['prior_roi_radius_scale']>0


def test_material_reference_labels_independent_input_frequency_subset(tmp_path):
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text());s=m['samples'][0]
    s.pop('target_path');s['frequencies_ghz']=[.6,.8,1.05]
    sig=np.load(tmp_path/s['signal_path'])[[0,4,9]];ref=np.load(tmp_path/s['reference_path'])[[0,4,9]]
    np.save(tmp_path/'sparse.npy',sig);np.save(tmp_path/'sparse_ref.npy',ref);s['signal_path']='sparse.npy';s['reference_path']='sparse_ref.npy'
    ids=np.ones((8,8,8),np.int16);table=np.ones((2,4,2),np.float32);table[1,:,0]=[30,32,33,35];table[1,:,1]=[.2,.3,.4,.5]
    np.save(tmp_path/'ids.npy',ids);np.save(tmp_path/'table.npy',table)
    s.update(material_ids_path='ids.npy',spectral_table_path='table.npy',spectral_table_frequencies_ghz=[.6,.8,.85,1.05])
    p.write_text(json.dumps(m));a=ManifestDataset(p)[0]
    assert a['target'][0,0,0,0]==33 and a['spectral_target'][0,1,0,0,0]==32
    assert a['spectral_target'].shape==(2,3,8,8,8)


@pytest.mark.parametrize('invalid_flag',['same_subject','measurement_compatible','calibration_compatible'])
def test_wrong_baseline_must_gate_measurement_subtraction(tmp_path,invalid_flag):
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text());s=m['samples'][0]
    s['mode']='baseline_tracking';s['baseline_signal_path']=s['signal_path']
    s['registration']=dict(same_subject=True,measurement_compatible=True,calibration_compatible=True,grid_compatible=True)
    s['registration'][invalid_flag]=False;p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='baseline subtraction'):ManifestDataset(p)[0]


def test_grid_mismatch_only_invalidates_image_baseline_context(tmp_path):
    p=convert_pilot(PILOT,tmp_path,grid_size=8);m=json.loads(p.read_text());s=m['samples'][0]
    s['mode']='monitoring';s['baseline_signal_path']=s['signal_path'];s['baseline_target_path']=s['target_path'];s['baseline_valid']=True
    s['registration']=dict(same_subject=True,measurement_compatible=True,calibration_compatible=True,grid_compatible=False)
    p.write_text(json.dumps(m));a=ManifestDataset(p)[0]
    assert a['context_validity']['measurement_registration']
    assert not a['context_validity']['baseline'] and not a['difference'].any()
