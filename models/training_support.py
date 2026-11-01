"""Shared training, waveform, feature, and checkpoint utilities."""
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

from models.comparison_training import write_json
from models.pressure_metrics import CUTS
from models.train_feature_mlp import ResidentBatches


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, output, hours=8):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / 'execution_budget.json'
        if path.exists():
            self.identity = json.loads(path.read_text())
        else:
            now = time.time()
            self.identity = dict(started_unix=now, deadline_unix=now+hours*3600,
                hours=hours, reserve_seconds=3600., test_accessed=False)
            write_json(path, self.identity)
        self.reserve = self.identity['reserve_seconds']

    def remaining(self, selection=True):
        return self.identity['deadline_unix'] - time.time() - (self.reserve if selection else 0)

    def admit(self, seconds, selection=True):
        return self.remaining(selection) > 1.5*seconds

    def check(self, selection=True):
        if self.remaining(selection) <= 0:
            raise BudgetExpired('Selection reserve reached' if selection else 'Execution deadline reached')

    def update_reserve(self, projected_refits):
        self.reserve = max(3600., 1.5*projected_refits+1200.)
        self.identity['reserve_seconds'] = self.reserve
        write_json(self.output/'execution_budget.json', self.identity)


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_callback(directory, budget, selection=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def callback(state):
        model = state.pop('model'); optimizer = state.pop('optimizer'); loader = state.pop('loader')
        generator = loader.order_loader.generator if isinstance(loader, ResidentBatches) else loader.generator
        payload = dict(state, model=model.state_dict(), optimizer=optimizer.state_dict(),
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
            order_rng=generator.get_state())
        atomic_torch_save(payload, directory/'last_checkpoint.pt')
        pd.DataFrame(state['history']).to_csv(directory/'live_history.csv', index=False)
        write_json(directory/'progress.json', dict(epoch=state['epoch'], best_epoch=state['best_epoch'],
            best_score=state['best_score'], updated_unix=time.time(), complete=False))
        budget.check(selection)
    return callback


def stable_key(value, seed=2026):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def representative_subset(frame, count, seed=2026):
    """Proportional joint recording-median strata, with deterministic tail coverage."""
    if not frame.split.eq('train').all():
        raise ValueError('Screening selection is training-only')
    records = frame.groupby('source_record_id')[['sbp','dbp']].median()
    count = min(count, len(records))
    strata = pd.Series(list(zip(np.searchsorted(CUTS['sbp'],records.sbp,side='left'),
                                np.searchsorted(CUTS['dbp'],records.dbp,side='left'))), index=records.index)
    members = {k: sorted(v.index, key=lambda x:stable_key(x,seed)) for k,v in strata.groupby(strata)}
    exact = {k:count*len(v)/len(records) for k,v in members.items()}
    allocation = {k:int(np.floor(v)) for k,v in exact.items()}
    for k in sorted(members, key=lambda k:(-(exact[k]-allocation[k]),k))[:count-sum(allocation.values())]:
        allocation[k] += 1
    chosen = set(r for k,v in members.items() for r in v[:allocation[k]])
    masks = [np.searchsorted(CUTS[t],frame[t],side='left') == b for t in ('sbp','dbp') for b in range(5)]
    coverage = [set(frame.loc[m,'source_record_id']) for m in masks]
    for available in coverage:
        if available and not (chosen & available):
            add = min(available,key=lambda r:stable_key(r,seed))
            removable = [r for r in chosen if all(not (r in g and len(chosen&g)==1) for g in coverage)]
            if not removable: raise ValueError('Cannot preserve pressure coverage at requested subset size')
            # Largest current stratum first, then stable recording hash.
            selected_counts = strata.loc[list(chosen)].value_counts()
            remove = min(removable,key=lambda r:(-selected_counts[strata[r]],stable_key(r,seed)))
            chosen.remove(remove); chosen.add(add)
    return frame.loc[frame.source_record_id.isin(chosen)].reset_index(drop=True)


def candidate_parameters(winners):
    m = copy.deepcopy(winners['mlp']['parameters']); l = copy.deepcopy(winners['lightgbm']['parameters'])
    mlp=[('control',m)]
    for name,key,value in [('wd_0001','weight_decay',.0001),('wd_003','weight_decay',.003),
        ('wd_01','weight_decay',.01),('drop_0','dropout',0.),('drop_02','dropout',.2),
        ('smaller','hidden_sizes',[128,64]),('larger','hidden_sizes',[256,128,64])]:
        mlp.append((name,{**m,key:value}))
    lgb=[('control',l)]
    for name,key,value in [('l2_0','reg_lambda',0.),('l2_1','reg_lambda',1.),
        ('l2_30','reg_lambda',30.),('l2_100','reg_lambda',100.),
        ('leaves_63','num_leaves',63),('leaves_255','num_leaves',255),
        ('leaf_25','min_child_samples',25),('leaf_200','min_child_samples',200)]:
        lgb.append((name,{**l,key:value}))
    return {'mlp':mlp,'lightgbm':lgb}


def waveform_channels(ppg, representation='ppg'):
    ppg=np.asarray(ppg,dtype=np.float32)
    if ppg.ndim!=2 or ppg.shape[1]!=625 or not np.isfinite(ppg).all():
        raise ValueError('Expected finite N x 625 cached PPG')
    if representation=='ppg': return ppg[:,None,:]
    if representation!='derivatives': raise ValueError('Unknown representation')
    # Float64 polynomial fitting avoids float32 edge cancellation amplified by fs**2.
    return np.stack([ppg]+[savgol_filter(ppg.astype(np.float64),11,3,deriv=d,delta=1/125,axis=1,mode='interp')
                          for d in (1,2)],axis=1).astype(np.float32)


def scale_waveforms(values, state):
    x=np.asarray(values,dtype=np.float32)
    if state['normalization']=='window':
        if x.shape[1]!=1: raise ValueError('Window scaling is a PPG-only comparison')
        return ((x-x.mean(axis=2,keepdims=True))/np.maximum(x.std(axis=2,keepdims=True),
            1e-6*np.asarray(state['channel_scale'])[None,:,None])).astype(np.float32)
    return ((x-np.asarray(state['channel_mean'])[None,:,None])/
            np.asarray(state['channel_scale'])[None,:,None]).astype(np.float32)


def fit_waveform_scaler(values, targets, normalization='global'):
    mean=np.mean(values,axis=(0,2),dtype=np.float64)
    scale=np.std(values,axis=(0,2),dtype=np.float64)
    if not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Degenerate/nonfinite training waveform channel')
    return dict(normalization=normalization,channel_mean=mean.tolist(),channel_scale=scale.tolist(),
        target_mean=np.mean(targets,axis=0).tolist(),target_scale=np.std(targets,axis=0).tolist())


class WaveCache:
    """Only PPG/peak-mask staging arrays; ABP is never opened."""
    def __init__(self, source, eligible):
        self.source=Path(source)
        with np.load(self.source/'cleaned_dataset.npz',allow_pickle=False) as a:
            ids=a['window_ids'].astype(str)
            if eligible.window_id.duplicated().any() or set(ids)!=set(eligible.window_id):
                raise ValueError('Cached waveform/feature identities differ')
            eligible=eligible.set_index('window_id',drop=False).loc[ids].reset_index(drop=True)
            for key,column in [('source_record_ids','source_record_id'),('split_assignments','split')]:
                if not np.array_equal(a[key],eligible[column]): raise ValueError('Cached identities differ')
            for t in ('sbp','dbp'):
                # CSV decimal round-tripping differs from the NPZ by at most a few ULPs.
                # Labels used by every model remain the unchanged original CSV values.
                np.testing.assert_allclose(a[t],eligible[t].to_numpy(),rtol=0,atol=1e-10)
        self.index=pd.Series(np.arange(len(ids)),index=ids)
        self.ppg=np.load(self.source/'staging/ppg.npy',mmap_mode='r')
        self.peaks=np.load(self.source/'staging/ppg_peak_mask.npy',mmap_mode='r')
        if self.ppg.shape!=(len(ids),625) or self.peaks.shape!=self.ppg.shape:
            raise ValueError('Staging shape mismatch')

    def rows(self,frame): return self.index.loc[frame.window_id].to_numpy()

    def values(self,frame,representation='ppg'):
        return waveform_channels(self.ppg[self.rows(frame)],representation)


def pulse_additions(clean, peaks, fs=125, shift=(0,0)):
    """Two optional measurements, retaining the existing complete-pulse definition."""
    feet=[a+int(np.argmin(clean[a:b+1])) for a,b in zip(peaks[:-1],peaks[1:])]
    values=[]
    for l,p,r in zip(feet[:-1],peaks[1:-1],feet[1:]):
        l=max(0,l+shift[0]);r=min(len(clean)-1,r+shift[1])
        if not l<p<r:continue
        y=np.maximum(clean[l:r+1]-np.linspace(clean[l],clean[r],r-l+1),0)
        h=y[p-l];area=np.trapezoid(y,dx=1/fs)
        if h<=1e-12 or area<=1e-12:continue
        values.append([np.trapezoid(y[p-l:],dx=1/fs)/area,
            ((p-l)/fs)*np.max(np.diff(y[:p-l+1])*fs)/h])
    return np.median(values,axis=0) if values else np.full(2,np.nan)
