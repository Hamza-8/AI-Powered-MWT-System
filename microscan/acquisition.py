"""Observation selection and baseline/ROI validity, independent of labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .data import voxel_grid, calibrate


def structured_mask(frequencies,rx_count,tx_count,frequency_indices=None,rx_indices=None,tx_indices=None,failed=None):
    mask=np.zeros((len(frequencies),rx_count,tx_count),bool)
    f=list(range(len(frequencies))) if frequency_indices is None else list(frequency_indices)
    r=list(range(rx_count)) if rx_indices is None else list(rx_indices)
    t=list(range(tx_count)) if tx_indices is None else list(tx_indices)
    for indices,n in ((f,len(frequencies)),(r,rx_count),(t,tx_count)):
        if len(set(indices))!=len(indices) or any(i<0 or i>=n for i in indices):raise ValueError('invalid acquisition indices')
    mask[np.ix_(f,r,t)]=True
    if failed is not None:
        if np.shape(failed)!=mask.shape:raise ValueError('failed-channel mask shape')
        mask &= ~np.asarray(failed,bool)
    return mask


def scan_list(mask,frequencies_ghz,physical_ports=None,inferred_mask=None,already_solved_tx=None):
    """Tx/frequency excitations count solves; Rx deletion saves no prior solves."""
    m=np.asarray(mask,bool)
    if m.ndim==4 and m.shape[0]==1:m=m[0]
    if m.ndim!=3:raise ValueError('mask must be [F,Rx,Tx]')
    ports=list(range(1,m.shape[1]+1)) if physical_ports is None else list(physical_ports)
    if m.shape[1]!=m.shape[2] or len(ports)!=m.shape[1] or len(frequencies_ghz)!=m.shape[0]:raise ValueError('scan metadata mismatch')
    inf=np.zeros_like(m) if inferred_mask is None else np.asarray(inferred_mask,bool)
    rows=[{'frequency_ghz':float(frequencies_ghz[f]),'tx_physical':int(ports[t]),'rx_physical':int(ports[r]),'provenance':'inferred_reciprocity' if inf[f,r,t] else 'acquired'} for f,r,t in np.argwhere(m|inf)]
    excitations={(int(f),int(t)) for f,r,t in np.argwhere(m)}
    already=set(tuple(x) for x in (already_solved_tx or []))
    return {'channels':rows,'independently_acquired_channels':int(m.sum()),'inferred_channels':int((inf&~m).sum()),'requested_source_excitations':len(excitations),'new_source_excitations':len(excitations-already),'prior_solves_saved':0,'full_source_excitations':m.shape[0]*m.shape[2],'note':'Rx exclusion after a completed full-Tx solve does not save that solve.'}


def roi_from_context(shape,bounds_xyz_mm,centre_xyz_mm,radius_mm,margin_mm=0.,source='clinician'):
    if source not in ('clinician','prior_baseline_prediction','previous_frame_prediction'):raise ValueError('ROI may not come from current ground truth')
    c=np.asarray(centre_xyz_mm,float);r=np.asarray(radius_mm,float)+float(margin_mm)
    if c.shape!=(3,) or np.any(r<=0):raise ValueError('invalid focus geometry')
    x,y,z=voxel_grid(shape,bounds_xyz_mm)
    if r.ndim==0:roi=(x-c[0])**2+(y-c[1])**2+(z-c[2])**2<=r*r
    else:
        if r.shape!=(3,):raise ValueError('ROI radius shape')
        roi=((x-c[0])/r[0])**2+((y-c[1])/r[1])**2+((z-c[2])/r[2])**2<=1
    return torch.from_numpy(roi[None]),{'source':source,'centre_xyz_mm':c.tolist(),'radius_mm':np.asarray(r).tolist(),'assessed_inside_only':True}


def focus_selection(antenna_xyz_mm,centre_xyz_mm,tx_count,rx_count,sensitivity=None):
    """Offline sensitivity ranking or labelled geometric fallback; never targets."""
    a=np.asarray(antenna_xyz_mm,float);c=np.asarray(centre_xyz_mm,float)
    if a.ndim!=2 or a.shape[1]!=3 or c.shape!=(3,):raise ValueError('invalid antenna/ROI geometry')
    if min(tx_count,rx_count)<1 or max(tx_count,rx_count)>len(a):raise ValueError('invalid focus budget')
    if sensitivity is None:
        score=-np.linalg.norm(a-c,axis=1);method='geometric_distance_fallback_not_validated_sensitivity'
    else:
        score=np.asarray(sensitivity,float)
        if score.shape!=(len(a),) or not np.isfinite(score).all():raise ValueError('sensitivity shape')
        method='supplied_offline_sensitivity'
    order=np.argsort(-score,kind='stable')
    return {'tx_indices':order[:tx_count].tolist(),'rx_indices':order[:rx_count].tolist(),'method':method}


def baseline_difference(current,baseline,current_mask,baseline_mask,registration):
    """Image-only registration does not authorize measurement subtraction."""
    required=('same_subject','measurement_compatible','calibration_compatible','grid_compatible')
    valid=baseline is not None and all(registration.get(x,False) for x in required)
    if not valid:
        return {'difference':None,'difference_mask':np.zeros_like(current_mask,dtype=bool),'baseline_valid':False,'request_full_scan':True,'reason':'missing_or_incompatible_baseline'}
    d,m=calibrate(current,baseline,current_mask,baseline_mask)
    return {'difference':d,'difference_mask':m,'baseline_valid':True,'request_full_scan':False,'reason':'registered_compatible_acquisition'}


def focused_validity(roi,mask,previous_roi=None,frame=0,refresh_every=None,displacement_mm=0.,maximum_displacement_mm=10.):
    valid=roi is not None and bool(torch.as_tensor(roi).any()) and bool(torch.as_tensor(mask).any())
    refresh=bool(refresh_every and frame%refresh_every==0)
    drift=displacement_mm>maximum_displacement_mm
    return {'roi_valid':valid,'assessed_region':roi if valid else None,'outside_roi_status':'unassessed','request_full_scan':not valid or refresh or drift,'global_refresh_due':refresh,'roi_drift':drift}


def structured_tx_holdout(mask,tx_indices):
    """Heldout support is drawn only from already observed selected frequencies."""
    m=torch.as_tensor(mask).bool();held=torch.zeros_like(m)
    held[...,list(tx_indices)]=m[...,list(tx_indices)]
    return m&~held,held


def acquisition_diversity(responses,mask,split='train'):
    if split!='train':raise ValueError('diversity diagnostics are training-only')
    a=np.asarray(responses,dtype=np.complex128);m=np.broadcast_to(np.asarray(mask,bool),a.shape)
    # Only common observed entries enter correlations; no zero-filled missing data.
    common=m.all(axis=0);x=np.where(m,a,0)[:,common]
    if not x.size:return {'valid':False,'reason':'no_common_observed_support'}
    if not np.isfinite(x).all():raise ValueError('nonfinite observed diagnostic')
    singular=np.linalg.svd(x-x.mean(0,keepdims=True),compute_uv=False)
    p=singular/max(float(singular.sum()),1e-12)
    return {'valid':True,'singular_values':singular.tolist(),'effective_rank':float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())),'independent_geometries':len(x),'scope':'training_response_diagnostics_not_frequency_selection'}


def export_scan(path,*args,**kwargs):
    result=scan_list(*args,**kwargs);Path(path).write_text(json.dumps(result,indent=2));return result


def corrupt_measurements(signal,mask,*,gain_std=0.,phase_std_rad=0.,noise_std=0.,reference_drift_std=0.,seed=0):
    """Explicit illustrative hardware model, not a validated SDR-noise emulator.

    Gain/phase are per Tx; reference drift is correlated across Rx/frequency.
    All transformations preserve observed zeros as observations and return the
    same mask. Hidden values never participate in arithmetic.
    """
    a=np.asarray(signal,dtype=np.complex128);m=np.broadcast_to(np.asarray(mask,bool),a.shape)
    if a.ndim!=3 or not np.isfinite(a[m]).all():raise ValueError('expected finite observed [F,Rx,Tx]')
    if min(gain_std,phase_std_rad,noise_std,reference_drift_std)<0:raise ValueError('negative corruption standard deviation')
    rng=np.random.default_rng(seed);tx=a.shape[-1]
    gain=(1+rng.normal(0,gain_std,(1,1,tx)))*np.exp(1j*rng.normal(0,phase_std_rad,(1,1,tx)))
    noise=(rng.normal(size=a.shape)+1j*rng.normal(size=a.shape))*noise_std/np.sqrt(2)
    drift=(rng.normal(size=(1,1,tx))+1j*rng.normal(size=(1,1,tx)))*reference_drift_std/np.sqrt(2)
    return {'signal':np.where(m,np.where(m,a,0)*gain+noise+drift,0),'mask':m.copy(),'model':'illustrative_gain_phase_iid_complex_noise_correlated_reference_drift','validated_sdr_noise':False,'seed':seed}
