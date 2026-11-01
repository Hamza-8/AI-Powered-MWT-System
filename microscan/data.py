"""Calibrated complex data, explicit axes and geometry-isolated datasets.

Stored signals: complex128 [F,Rx,Tx]. Network signals: float [2,F,Rx,Tx].
Property volumes: [epsilon_r,sigma_S_per_m,Z,Y,X], XYZ physical coordinates.
No data-dependent transform is fitted implicitly by a dataset.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from torch.utils.data import Dataset

PILOT_FREQUENCIES = np.round(np.arange(.60, 1.051, .05), 2)
PILOT_COMSOL_PORTS = [7,8,9,10,11,1,2,3,4,6,12,13,14,15,16,17,18,19]
PILOT_SPLITS = {"train": [1,4], "validation": [2], "val": [2], "test": [3], "reference": [0]}


def load_mat(path):
    """Read v5 with SciPy or v7.3 via explicit MATLAB HDF5 conventions.

    v7.3 numeric axes are reversed on disk by MATLAB, including references in
    cells/structs. Compound real/imaginary datasets retain complex128.
    """
    path = Path(path)
    with path.open('rb') as f:
        header=f.read(520)
    if header.startswith(b'\x89HDF') or header[512:520] == b'\x89HDF\r\n\x1a\n' or b'MATLAB 7.3' in header[:128]:
        import h5py
        with h5py.File(path, 'r') as f:
            def read(obj):
                if isinstance(obj, h5py.Group):
                    return {k:read(v) for k,v in obj.items() if not k.startswith('#')}
                a=obj[()]
                if h5py.check_dtype(ref=a.dtype) is not None:
                    vals=[read(f[r]) if r else None for r in a.T.ravel()]
                    return vals[0] if len(vals)==1 else vals
                if a.dtype.names and set(a.dtype.names)>={'real','imag'}:
                    a=a['real'].astype(np.float64)+1j*a['imag'].astype(np.float64)
                a=a.transpose(tuple(reversed(range(a.ndim)))) if a.ndim>1 else a
                cls=obj.attrs.get('MATLAB_class', b'')
                if cls in (b'char','char'):
                    return ''.join(chr(int(x)) for x in a.ravel() if x)
                return a.item() if a.size==1 else a
            return read(f)
    from scipy.io import loadmat
    return loadmat(path, simplify_cells=True)


def mask_complex(values, mask):
    """Mask before arithmetic: NaN*0 is not a mask operation."""
    v=np.asarray(values, dtype=np.complex128)
    m=np.broadcast_to(np.asarray(mask,dtype=bool),v.shape)
    if not np.isfinite(v[m]).all(): raise ValueError('nonfinite observed measurement')
    return np.where(m,v,0j)


def complex_to_channels(values, mask=None):
    v=np.asarray(values,dtype=np.complex128)
    if mask is not None: v=mask_complex(v,mask)
    if not np.isfinite(v).all(): raise ValueError('nonfinite measurement')
    return torch.from_numpy(np.stack([v.real,v.imag],axis=0).astype(np.float32))


def channels_to_complex(values):
    a=values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    if a.shape[0]!=2: raise ValueError('expected [2,F,Rx,Tx]')
    return a[0].astype(np.float64)+1j*a[1].astype(np.float64)


def calibrate(current, reference=None, current_mask=None, reference_mask=None):
    """Complex128 subtraction on mask intersection; zero is valid data."""
    c=np.asarray(current,dtype=np.complex128)
    m=np.ones(c.shape,bool) if current_mask is None else np.broadcast_to(current_mask,c.shape).astype(bool)
    if reference is None: return mask_complex(c,m),m
    r=np.asarray(reference,dtype=np.complex128)
    if r.shape!=c.shape: raise ValueError('reference shape mismatch')
    rm=np.ones(r.shape,bool) if reference_mask is None else np.broadcast_to(reference_mask,r.shape).astype(bool)
    m=m&rm
    return mask_complex(c,m)-mask_complex(r,m),m


def voxel_grid(shape=(128,128,128), bounds_xyz_mm=((-60,60),(-60,60),(-60,60))):
    """Cell centres in ZYX mesh order, each output holds XYZ coordinate values."""
    zyx=tuple(int(n) for n in shape)
    if len(zyx)!=3 or min(zyx)<=0: raise ValueError('invalid ZYX grid')
    b=np.asarray(bounds_xyz_mm,dtype=np.float64)
    if b.shape!=(3,2) or np.any(b[:,1]<=b[:,0]):raise ValueError('invalid XYZ bounds')
    xyz=[b[i,0]+(np.arange(zyx[2-i])+.5)*(b[i,1]-b[i,0])/zyx[2-i] for i in range(3)]
    z,y,x=np.meshgrid(xyz[2],xyz[1],xyz[0],indexing='ij')
    return x,y,z


def sphere_target(case, grid_size=128, half_span_mm=60):
    shape=(grid_size,)*3 if np.isscalar(grid_size) else tuple(grid_size)
    x,y,z=voxel_grid(shape,[[-half_span_mm,half_span_mm]]*3)
    centre=np.asarray(case['centreMM'],dtype=np.float64)
    radius=float(case['radiusMM'])
    inside=((x-centre[0])**2+(y-centre[1])**2+(z-centre[2])**2 <= radius**2) if radius>0 else np.zeros(shape,bool)
    target=np.stack([1+44*inside,.8*inside]).astype(np.float32)
    return torch.from_numpy(target),torch.from_numpy(inside[None].copy())


class ObservedScale:
    """Training-only shared RMS scale, no per-sample amplitude removal."""
    def __init__(self, scale=1., fitted_geometry_ids=None):
        self.scale=float(scale)
        self.fitted_geometry_ids=list(fitted_geometry_ids or [])
        if not np.isfinite(self.scale) or self.scale<=0:raise ValueError('invalid scale')
    @classmethod
    def fit(cls,samples:Iterable[dict],floor=1e-12):
        total=0.;count=0;ids=[]
        for s in samples:
            md=s.get('metadata',{})
            if md.get('split')!='train':raise ValueError('preprocessing fit requires training geometries only')
            a=s['signals'].double();m=s['mask'].bool().expand_as(a)
            observed=a[m]
            if not torch.isfinite(observed).all():raise ValueError('nonfinite observed training value')
            total+=float(observed.square().sum());count+=observed.numel();ids.append(md.get('geometry_id',md.get('case_id')))
        if count==0:raise ValueError('cannot fit scale on all-missing data')
        return cls(max((total/count)**.5,floor),sorted(set(str(x) for x in ids)))
    def transform(self, signals, mask):
        return torch.where(mask.bool(),signals,torch.zeros_like(signals))/self.scale
    def inverse(self,signals,mask=None):
        a=signals*self.scale
        return a if mask is None else torch.where(mask.bool(),a,torch.zeros_like(a))
    def state_dict(self):return {'scale':self.scale,'fitted_geometry_ids':self.fitted_geometry_ids,'algorithm':'observed_training_rms_v1'}
    @classmethod
    def from_state_dict(cls,d):return cls(d['scale'],d['fitted_geometry_ids'])


class PilotDataset(Dataset):
    """Four geometries; reference is never an independent training example."""
    def __init__(self,mat_path,split='train',frequency_indices=None,allow_test=False,grid_size=128,preprocessing=None):
        if split not in PILOT_SPLITS:raise ValueError('unknown split')
        if split=='test' and not allow_test:raise PermissionError('pilot test geometry is sealed until explicit selection')
        m=load_mat(mat_path);d=m['data'];cfg=m['cfg'];r=m['report']
        s=np.asarray(d['S'])
        if s.shape!=(18,18,10,5) or s.dtype!=np.complex128:raise ValueError('pilot requires complex128 [Rx,Tx,F,case]=[18,18,10,5]')
        if not np.isfinite(s).all() or not np.asarray(d['done']).all() or not bool(r['completed']):raise ValueError('incomplete/nonfinite pilot')
        if not np.allclose(d['GHz'],PILOT_FREQUENCIES):raise ValueError('unexpected pilot frequencies')
        if list(np.asarray(d['physicalPorts']).astype(int))!=list(range(1,19)):raise ValueError('physical port order mismatch')
        if list(np.asarray(d['comsolPorts']).astype(int))!=PILOT_COMSOL_PORTS:raise ValueError('COMSOL mapping mismatch')
        if list(d['axes'])!=['receiver_physical','transmitter_physical','frequency','case']:raise ValueError('pilot axes mismatch')
        self.s=s;self.cases=d['cases'];self.ids=PILOT_SPLITS[split];self.split='validation' if split=='val' else split
        self.grid_size=grid_size;self.half_span=float(cfg['gridHalfSpanMM']);self.preprocessing=preprocessing
        self.indices=list(range(10)) if frequency_indices is None else list(map(int,frequency_indices))
        if not self.indices or len(set(self.indices))!=len(self.indices) or min(self.indices)<0 or max(self.indices)>9:raise ValueError('invalid frequency indices')
        self.path=str(mat_path)
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        cid=self.ids[index];current=self.s[:,:,:,cid].transpose(2,0,1);reference=self.s[:,:,:,0].transpose(2,0,1)
        mask=np.zeros(current.shape,bool);mask[self.indices]=True
        calibrated,mask=calibrate(current,reference,mask,mask)
        target,obj=sphere_target(self.cases[cid],self.grid_size,self.half_span)
        signals=complex_to_channels(calibrated);mt=torch.from_numpy(mask[None].copy())
        if self.preprocessing: signals=self.preprocessing.transform(signals,mt)
        return {'signals':signals,'mask':mt,'target':target,'object_mask':obj,
                'raw_signals':complex_to_channels(current,mask),'reference':complex_to_channels(reference,mask),
                'frequencies_ghz':torch.tensor(PILOT_FREQUENCIES,dtype=torch.float32),
                'metadata':{'case_id':cid,'geometry_id':f'pilot_case_{cid}','subject_id':f'pilot_case_{cid}',
                'split':self.split,'domain':'calibrated_delta_s','mode':'snapshot','source_domain':'COMSOL_sphere_pilot',
                'calibration':'empty_reference_case_0_complex128','reference_frequency_ghz':.85,
                'physical_ports':list(range(1,19)),'comsol_ports':PILOT_COMSOL_PORTS,
                'axes_signal':'C,F,Rx,Tx','axes_target':'property,Z,Y,X','bounds_xyz_mm':[[-60,60]]*3,
                'grid_shape_zyx':list(target.shape[1:]),'target_kind':'object_properties_not_clinical_lesions',
                'case':_jsonable(self.cases[cid]),'preprocessing':self.preprocessing.state_dict() if self.preprocessing else None}}


def _jsonable(a):
    if isinstance(a,dict):return {k:_jsonable(v) for k,v in a.items()}
    if isinstance(a,(tuple,list)):return [_jsonable(x) for x in a]
    if isinstance(a,np.ndarray):return a.tolist()
    if isinstance(a,np.generic):return a.item()
    return a


def validate_manifest(manifest):
    """Validate units, grids, grouping, label absence, and physical provenance."""
    if manifest.get('schema_version')!=1:raise ValueError('unsupported manifest schema')
    samples=manifest.get('samples',[])
    if not samples:raise ValueError('empty manifest')
    seen={};ids=set()
    required=('id','split','subject_id','head_id','geometry_id','sequence_id','source_domain','domain','signal_path','grid','frequencies_ghz','physical_ports','reference_frequency_ghz')
    for s in samples:
        missing=[k for k in required if k not in s]
        if missing:raise ValueError(f'missing sample metadata {missing}')
        if s['id'] in ids:raise ValueError('duplicate sample id')
        ids.add(s['id'])
        if s['split'] not in ('train','validation','test'):raise ValueError('invalid split')
        for key in ('subject_id','head_id','geometry_id','sequence_id'):
            identity=(key,str(s[key]))
            if identity in seen and seen[identity]!=s['split']:raise ValueError(f'{key} leakage across splits')
            seen[identity]=s['split']
        g=s['grid'];voxel_grid(tuple(g['shape_zyx']),g['bounds_xyz_mm'])
        if g.get('array_axes')!='ZYX' or g.get('coordinate_axes')!='XYZ' or g.get('units')!='mm':raise ValueError('explicit grid axes/units required')
        if s['domain'] not in ('raw_s','calibrated_delta_s','decluttered_target','current_minus_baseline'):raise ValueError('unknown measurement domain')
        f=np.asarray(s['frequencies_ghz'],float)
        if f.ndim!=1 or (f<=0).any() or len(np.unique(f))!=len(f):raise ValueError('invalid frequency grid')
        if s.get('mode')=='monitoring' and not s.get('registration',{}).get('measurement_compatible',False):
            if s.get('difference_path'):raise ValueError('cannot use precomputed unregistered differences')
        if s.get('label_schema') and not s.get('label_path'):raise ValueError('label schema requires actual labels')
    return {'schema_version':1,'samples':len(samples),'split_counts':{k:sum(s['split']==k for s in samples) for k in ('train','validation','test')},'valid':True}


class ManifestDataset(Dataset):
    """Scalable per-sample NPY/NPZ arrays; no implicit dummy targets or labels.

    signal_path complex128 [F,Rx,Tx], target_path float [2,Z,Y,X]. Optional
    fields use *_path. Whole-head extents come from each grid, never the pilot.
    """
    def __init__(self,path,split='train',preprocessing=None,allow_test=False):
        self.path=Path(path);self.manifest=json.loads(self.path.read_text());validate_manifest(self.manifest)
        if split=='test' and not allow_test:raise PermissionError('test split requires explicit access')
        self.samples=[s for s in self.manifest['samples'] if s['split']==split];self.root=self.path.parent;self.preprocessing=preprocessing
    def __len__(self):return len(self.samples)
    def _array(self,path):
        p=self.root/path
        if p.suffix=='.npy':return np.load(p,mmap_mode='r',allow_pickle=False)
        if p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                if len(z.files)!=1:raise ValueError('NPZ sample must contain one named array')
                return z[z.files[0]]
        if p.suffix=='.mat':
            d=load_mat(p)
            if 'array' not in d:raise ValueError('MAT array field missing')
            return d['array']
        raise ValueError('arrays must use NPY/NPZ/MAT')
    def __getitem__(self,index):
        s=self.samples[index];raw=np.asarray(self._array(s['signal_path']),dtype=np.complex128)
        shape=(len(s['frequencies_ghz']),len(s['physical_ports']),len(s['physical_ports']))
        if raw.shape!=shape:raise ValueError('measurement shape/metadata mismatch')
        mask=np.asarray(self._array(s['mask_path']),bool) if s.get('mask_path') else np.ones(shape,bool)
        reference=self._array(s['reference_path']) if s.get('reference_path') else None
        rm=self._array(s['reference_mask_path']) if s.get('reference_mask_path') else None
        if s.get('domain')=='raw_s' and reference is None:calibrated,mask=calibrate(raw,None,mask)
        else:calibrated,mask=calibrate(raw,reference,mask,rm)
        if s.get('mode')=='monitoring' and not s.get('registration',{}).get('measurement_compatible',False):raise ValueError('monitoring needs registered acquisition or recalibration')
        signals=complex_to_channels(calibrated);mt=torch.from_numpy(mask[None].copy())
        if self.preprocessing:signals=self.preprocessing.transform(signals,mt)
        out={'signals':signals,'mask':mt,'frequencies_ghz':torch.tensor(s['frequencies_ghz'],dtype=torch.float32),'raw_signals':complex_to_channels(raw,mask),'metadata':s.copy()}
        for source,key in [('target_path','target'),('roi_path','roi'),('baseline_target_path','baseline'),('label_path','labels'),('label_validity_path','label_validity'),('clean_signal_path','clean_signals'),('material_ids_path','material_ids'),('repeated_signal_path','repeated_signals'),('stable_mask_path','stable_mask'),('baseline_spectral_path','baseline_spectral_target')]:
            if s.get(source):
                a=np.array(self._array(s[source]),copy=True)
                if key.endswith('signals'):a=complex_to_channels(a,mask)
                else:a=torch.from_numpy(a)
                out[key]=a
        if 'target' in out and tuple(out['target'].shape)!=(2,*s['grid']['shape_zyx']):raise ValueError('target shape/metadata mismatch')
        if 'labels' in out and 'label_validity' not in out:out['label_validity']=out['labels']>=0
        if s.get('baseline_signal_path'):
            base=self._array(s['baseline_signal_path']);bm=self._array(s['baseline_mask_path']) if s.get('baseline_mask_path') else np.ones(shape,bool)
            diff,dm=calibrate(raw,base,mask,bm)
            out['difference']=complex_to_channels(diff);out['difference_mask']=torch.from_numpy(dm[None].copy())
        if s.get('spectral_table_path') and 'material_ids' in out:
            table=np.array(self._array(s['spectral_table_path']),copy=True)
            # [materials,F,2] -> [2,F,Z,Y,X], labels only.
            mids=out['material_ids'].long()
            if mids.min()<0 or mids.max()>=table.shape[0]:raise ValueError('invalid supervised material id')
            out['spectral_target']=torch.from_numpy(table)[mids].movedim((-1,-2),(0,1))
        if s.get('spectral_target_path'):
            out['spectral_target']=torch.from_numpy(np.array(self._array(s['spectral_target_path']),copy=True))
        # Context never comes from current supervised labels or material IDs.
        out['mode']=s.get('mode','snapshot');out['domain']=s['domain']
        registration=s.get('registration',{})
        bv=bool(s.get('baseline_valid',False)) and all(registration.get(k,False) for k in ('same_subject','measurement_compatible','calibration_compatible','grid_compatible'))
        out['baseline_valid']=torch.tensor(bv)
        out['context_validity']={'baseline':torch.tensor(bv),'roi':torch.tensor('roi' in out),'observations':torch.tensor(bool(mask.any()))}
        if s.get('context_validity'):
            for key,value in s['context_validity'].items():
                if key!='baseline':out['context_validity'][key]=torch.as_tensor(value)
        if s.get('physics_context') or s.get('physics_context_paths'):
            out['physics_context']=dict(s.get('physics_context',{}))
            for key,path in s.get('physics_context_paths',{}).items():
                out['physics_context'][key]=torch.from_numpy(np.array(self._array(path),copy=True))
            for key in ('tx_positions_mm','rx_positions_mm','source_excitation','tx_orientations','rx_orientations','geometry_features','calibration_features'):
                if key in out['physics_context'] and not torch.is_tensor(out['physics_context'][key]):
                    out['physics_context'][key]=torch.as_tensor(out['physics_context'][key],dtype=torch.float32)
        return out


def collate_samples(samples):
    """Stack arrays, recurse context, retain physical IDs as case-aligned lists."""
    def combine(values,key=None):
        first=values[0]
        if torch.is_tensor(first):return torch.stack(values)
        if isinstance(first,dict) and key!='metadata':
            keys=set(first)
            if any(set(v)!=keys for v in values):raise ValueError('mixed context schema requires explicit missing-context fields')
            return {k:combine([v[k] for v in values],k) for k in keys}
        # Equal acquisition mode/domain IDs are scalar semantics; metadata remains per case.
        if key in ('mode','domain','layout_id','calibration_id','grid_id','material_library_id','source_id','constant_spectrum') and all(v==first for v in values):return first
        return values
    keys=set(samples[0])
    if any(set(s)!=keys for s in samples):raise ValueError('mixed optional fields require explicit missing-context arrays')
    return {k:combine([s[k] for s in samples],k) for k in keys}


def convert_pilot(mat_path,output_dir,grid_size=128):
    """Lossless calibrated real-data export with untouched sealed-test flag."""
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True);samples=[]
    for split in ('train','validation','test'):
        ds=PilotDataset(mat_path,split,allow_test=True,grid_size=grid_size)
        for a in ds:
            cid=a['metadata']['case_id'];prefix=f'case_{cid}'
            # Export raw current + reference, so subtraction remains complex128.
            raw=ds.s[:,:,:,cid].transpose(2,0,1);ref=ds.s[:,:,:,0].transpose(2,0,1)
            np.save(root/f'{prefix}_signal.npy',raw);np.save(root/f'{prefix}_reference.npy',ref);np.save(root/f'{prefix}_target.npy',a['target'].numpy())
            md=a['metadata'];samples.append({'id':prefix,'split':split,'subject_id':prefix,'head_id':prefix,'geometry_id':prefix,'sequence_id':prefix,'source_domain':md['source_domain'],'domain':'calibrated_delta_s','mode':'snapshot','signal_path':f'{prefix}_signal.npy','reference_path':f'{prefix}_reference.npy','target_path':f'{prefix}_target.npy','grid':{'shape_zyx':list(a['target'].shape[1:]),'bounds_xyz_mm':md['bounds_xyz_mm'],'array_axes':'ZYX','coordinate_axes':'XYZ','units':'mm'},'frequencies_ghz':PILOT_FREQUENCIES.tolist(),'physical_ports':list(range(1,19)),'comsol_ports':PILOT_COMSOL_PORTS,'reference_frequency_ghz':.85,'target_kind':'sphere_object_only','sealed_test':split=='test','lesion_configuration':md['case']})
    manifest={'schema_version':1,'source_mat':str(Path(mat_path).name),'samples':samples};validate_manifest(manifest)
    p=root/'manifest.json';p.write_text(json.dumps(manifest,indent=2));return p
