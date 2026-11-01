"""Explicitly routed calibrated MicroScan deployment and monitoring outputs."""
from __future__ import annotations
import torch
from torch import nn
from .models import MSRNN, SegmentationUNet, sanitize_measurements


class MicroScanPipeline(nn.Module):
    def __init__(self,reconstruction=None,segmentation=None,declutterer=None,*,
                 trained_declutterer=False,subspace=None,physics_mode="A",phi=None,
                 physics_context=None,global_refresh_interval=10,trained_routing=None,subspace_context=None,signal_scale=1.):
        super().__init__()
        self.reconstruction=reconstruction if reconstruction is not None else MSRNN()
        self.segmentation=segmentation
        self.declutterer=declutterer
        self.trained_declutterer=trained_declutterer
        self.subspace=subspace
        self.subspace_context=subspace_context
        self.signal_scale=signal_scale
        self.physics_mode,self.phi=physics_mode,phi if physics_mode=="C" else None
        self.physics_context=physics_context
        self.global_refresh_interval=global_refresh_interval
        self.trained_routing=trained_routing
    def forward(self,measurements,mask=None,frequencies_ghz=None,*,baseline=None,roi=None,
                context_validity=None,mode="snapshot",enable_declutter=False,enable_subspace=False,
                run_segmentation=True,frame_index=0,physics_observations=None):
        if isinstance(measurements,dict):
            batch=measurements
            measurements=batch.get("measurements",batch.get("signals"))
            mask=batch["mask"]
            frequencies_ghz=batch.get("frequencies_ghz",frequencies_ghz)
            baseline=batch.get("baseline_volume",batch.get("baseline",baseline))
            roi=batch.get("roi",roi)
            context_validity=batch.get("context_validity",context_validity)
            mode=batch.get("mode",mode)
            physics_observations=batch.get("original_calibrated",physics_observations)
        if self.trained_routing is not None:
            actual={"declutterer":bool(enable_declutter),"subspace":bool(enable_subspace)}
            if actual!=self.trained_routing:
                raise ValueError("Inference routing differs from the trained reconstruction checkpoint")
        calibrated=sanitize_measurements(measurements,mask)
        processed=calibrated
        monitoring=mode in ("monitoring","baseline_tracking")
        routing={"declutterer":"bypassed","subspace":"bypassed"}
        if enable_subspace:
            if monitoring:
                raise ValueError("Monitoring subspace processing must be separately trained and explicitly configured")
            if self.subspace is None:
                raise ValueError("No fitted subspace supplied")
            if self.subspace_context is None:
                raise ValueError("Subspace processing requires saved hardware/calibration/frequency bindings")
            if processed.requires_grad:
                raise ValueError("Fixed subspace preprocessing must precede differentiable model inputs")
            residuals=[]
            for i in range(processed.shape[0]):
                signal=torch.complex(processed[i,0],processed[i,1]).detach().cpu().numpy()
                observed=mask[i,0].detach().cpu().numpy()
                result=self.subspace.transform(signal,observed,self.subspace_context)
                complex_residual=torch.as_tensor(result["residual"],device=processed.device)
                residuals.append(torch.stack((complex_residual.real,complex_residual.imag),0).to(processed))
            processed=torch.stack(residuals,0)
            processed=sanitize_measurements(processed,mask)
            routing["subspace"]="active"
        if enable_declutter:
            if monitoring:
                raise ValueError("Default monitoring checkpoint bypasses CycleGAN; train a dedicated route to enable it")
            if self.declutterer is None or not self.trained_declutterer:
                raise ValueError("Decluttering requires explicitly loaded trained weights; random G is prohibited")
            processed=self.declutterer(processed,mask,frequencies_ghz if frequencies_ghz is not None else self.reconstruction.frequency_grid)
            processed=sanitize_measurements(processed,mask)
            routing["declutterer"]="active"
        outputs=self.reconstruction(processed,mask,frequencies_ghz,baseline,roi,context_validity,mode,
                                    self.physics_mode,self.phi,self.physics_context,physics_observations=calibrated if physics_observations is None else physics_observations)
        physical_calibrated=calibrated*torch.as_tensor(self.signal_scale,device=calibrated.device,dtype=calibrated.dtype) if physics_observations is None else sanitize_measurements(physics_observations,mask)
        outputs.update(calibrated=physical_calibrated,network_calibrated=calibrated,decluttered=processed*torch.as_tensor(self.signal_scale,device=processed.device,dtype=processed.dtype),routing=routing,
                       segmentation_logits=None,segmentation_probabilities=None)
        if self.segmentation is not None and run_segmentation:
            # Invalid absolute states must not become confident clinical labels.
            absolute_valid=outputs["absolute_valid"]
            absolute=torch.where(absolute_valid[:,None,None,None,None],outputs["absolute"],torch.zeros_like(outputs["absolute"]))
            logits=self.segmentation(absolute,baseline,outputs["change"],roi)
            probabilities=self.segmentation.probabilities(logits)
            assessed=torch.ones_like(probabilities[:,:1],dtype=torch.bool)
            if roi is not None: assessed=assessed & roi.bool()
            assessed=assessed & absolute_valid[:,None,None,None,None]
            outputs.update(segmentation_logits=logits,segmentation_probabilities=probabilities,
                           segmentation_assessed=assessed)
        roi_valid=True if roi is None else bool(roi.bool().any())
        if context_validity is not None:
            roi_valid=roi_valid and bool(torch.as_tensor(context_validity.get("roi",True)).all())
        focused=mode in ("focused","baseline_tracking")
        outputs["request_full_scan"]=bool(focused and (not roi_valid or frame_index%max(1,self.global_refresh_interval)==0))
        return outputs


def connected_components(probabilities,grid_xyz_mm,threshold=.5,label_mode="multiclass",assessed=None):
    """Physical object descriptors; no probability calibration is claimed."""
    import numpy as np
    from scipy.ndimage import label
    values=probabilities.detach().cpu().numpy() if torch.is_tensor(probabilities) else np.asarray(probabilities)
    if values.ndim==5:
        if values.shape[0]!=1: raise ValueError("Component export accepts one volume")
        values=values[0]
    x,y,z=[np.asarray(v,dtype=np.float64) for v in grid_xyz_mm]
    spacing=[float(abs(v[1]-v[0])) if len(v)>1 else 1. for v in (x,y,z)]
    voxel_volume=float(np.prod(spacing))
    region=np.ones(values.shape[1:],bool) if assessed is None else np.asarray(assessed.detach().cpu() if torch.is_tensor(assessed) else assessed).squeeze().astype(bool)
    components=[]
    classes=range(1,values.shape[0]) if label_mode=="multiclass" else range(values.shape[0])
    winner=values.argmax(0)
    for cls in classes:
        foreground=(values[cls]>=threshold)&region
        if label_mode=="multiclass": foreground &= winner==cls
        labels,count=label(foreground)
        for i in range(1,count+1):
            zz,yy,xx=np.where(labels==i)
            components.append(dict(class_id=int(cls),voxel_count=len(xx),center_xyz_mm=[float(x[xx].mean()),float(y[yy].mean()),float(z[zz].mean())],volume_mm3=len(xx)*voxel_volume,volume_ml=len(xx)*voxel_volume/1000,mean_probability=float(values[cls,zz,yy,xx].mean())))
    return components


def component_changes(current,baseline,max_match_distance_mm=30.,roi_valid=True):
    """Greedy class-matched prior/current descriptors; signed growth and shrinkage.

    Match validity is explicit: an unmatched component is not silently declared
    resolved outside an assessed ROI. For clinical tracking a verified temporal
    registration and independently validated matcher remain prerequisites.
    """
    import math
    remaining=set(range(len(baseline)))
    changes=[]
    for component in current:
        choices=[]
        for index in remaining:
            prior=baseline[index]
            if prior['class_id']!=component['class_id']:continue
            distance=math.dist(component['center_xyz_mm'],prior['center_xyz_mm'])
            if distance<=max_match_distance_mm:choices.append((distance,index))
        if choices:
            distance,index=min(choices);remaining.remove(index);prior=baseline[index]
            changes.append(dict(current=component,baseline=prior,status='matched',
                volume_change_mm3=component['volume_mm3']-prior['volume_mm3'],
                displacement_xyz_mm=[a-b for a,b in zip(component['center_xyz_mm'],prior['center_xyz_mm'])],
                tracking_valid=bool(roi_valid)))
        else:
            changes.append(dict(current=component,baseline=None,status='new_or_unmatched',
                                volume_change_mm3=None,tracking_valid=False))
    for index in sorted(remaining):
        changes.append(dict(current=None,baseline=baseline[index],status='absent_in_assessed_roi' if roi_valid else 'unassessed',
                            volume_change_mm3=-baseline[index]['volume_mm3'] if roi_valid else None,
                            tracking_valid=bool(roi_valid)))
    return changes
