"""Optional per-Rx/Tx complex frequency subspace (a project adaptation).

The source temporal-cohort method is not copied as a universal rank-two rule.
Missing-data bases are fitted by observed-only alternating least squares.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class MaskedSubspace:
    def __init__(self,rank=1,strength=.25,ridge=1e-5):
        if rank<1 or not 0<=strength<=1 or ridge<=0:raise ValueError('invalid subspace configuration')
        self.rank=rank;self.strength=strength;self.ridge=ridge;self.basis=None;self.context=None
    def _coeff(self,b,y,m):
        if not m.any():return np.zeros(self.rank,dtype=np.complex128)
        a=b[m];v=y[m]
        return np.linalg.solve(a.conj().T@a+self.ridge*np.eye(self.rank),a.conj().T@v)
    def fit(self,cohort,masks,context,split='train',iterations=8):
        if split!='train':raise ValueError('subspace fit requires calibration training cohort')
        a=np.asarray(cohort,dtype=np.complex128);m=np.broadcast_to(np.asarray(masks,bool),a.shape)
        if a.ndim!=4 or a.shape[0]<2:raise ValueError('need >=2 independent cohort configurations [N,F,Rx,Tx]')
        if self.rank>=min(a.shape[:2]):raise ValueError('rank must be below cohort/frequency count')
        if not np.isfinite(a[m]).all():raise ValueError('nonfinite observed cohort')
        if not all(k in context for k in ('hardware','calibration','frequencies_ghz')):raise ValueError('subspace physical bindings required')
        a=np.where(m,a,0);n,f,r,t=a.shape;bases=np.empty((r,t,f,self.rank),np.complex128)
        # Deterministic Fourier initialization avoids treating zeros as observations.
        initial=np.exp(2j*np.pi*np.arange(f)[:,None]*np.arange(1,self.rank+1)[None]/f)/np.sqrt(f)
        for rx in range(r):
            for tx in range(t):
                y=a[:,:,rx,tx];obs=m[:,:,rx,tx];b=initial.copy()
                for _ in range(iterations):
                    c=np.stack([self._coeff(b,y[i],obs[i]) for i in range(n)])
                    for fi in range(f):
                        valid=obs[:,fi]
                        if valid.any():
                            cv=c[valid];b[fi]=np.linalg.solve(cv.conj().T@cv+self.ridge*np.eye(self.rank),cv.conj().T@y[valid,fi])
                    b,_=np.linalg.qr(b)
                bases[rx,tx]=b
        self.basis=bases;self.context=json.loads(json.dumps(context));return self
    def transform(self,signal,mask,context,bypass=False):
        a=np.asarray(signal,dtype=np.complex128);m=np.broadcast_to(np.asarray(mask,bool),a.shape)
        if not np.isfinite(a[m]).all():raise ValueError('nonfinite observed signal')
        original=np.where(m,a,0)
        if bypass:return {'original':original,'clutter':np.zeros_like(original),'residual':original.copy(),'mask':m.copy(),'active':False}
        if self.basis is None:raise ValueError('subspace is not fitted')
        if context!=self.context:raise ValueError('subspace hardware/calibration/frequency binding mismatch')
        f,r,t=a.shape
        if self.basis.shape!=(r,t,f,self.rank):raise ValueError('subspace input shape mismatch')
        clutter=np.zeros_like(original)
        for rx in range(r):
            for tx in range(t):
                b=self.basis[rx,tx];c=self._coeff(b,original[:,rx,tx],m[:,rx,tx]);clutter[:,rx,tx]=b@c
        clutter=np.where(m,clutter,0)
        return {'original':original,'clutter':clutter,'residual':np.where(m,original-self.strength*clutter,0),'mask':m.copy(),'active':True,'strength':self.strength,'source_validity':self.context}
    def save(self,path):
        if self.basis is None:raise ValueError('subspace not fitted')
        np.savez_compressed(path,basis=self.basis,metadata=json.dumps({'rank':self.rank,'strength':self.strength,'ridge':self.ridge,'context':self.context}))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as d:
            md=json.loads(str(d['metadata']));o=cls(md['rank'],md['strength'],md['ridge']);o.context=md['context'];o.basis=d['basis'];return o
