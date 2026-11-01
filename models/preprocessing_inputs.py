"""Versioned PPG-only input preparation; never labels or changes eligibility."""
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, savgol_filter

from cleaning.clean_data import read_record
from cleaning.pipeline import interpolate_zeros
from models.comparison_training import digest, validate_request, write_json
from models.training_support import waveform_channels

PREPARATION = dict(version='ppg_boundaries_v1', samples=625, sampling_rate=125,
    padding=250, reflection='odd', lowcut=.5, highcut=8, order=2,
    derivative_window=11, derivative_polynomial=3, calculation='float64',storage='float32',
    zero_interpolation='existing per-window np.interp before reflection',
    filtered_extension='quantized float32 before derivatives')
SOS=butter(2,[.5,8],btype='bandpass',fs=125,output='sos')


def prepare_values(original):
    original=np.asarray(original,dtype=np.float32)
    if original.ndim!=2 or original.shape[1]!=625:raise ValueError('Expected N x 625')
    raw=np.stack([interpolate_zeros(x) for x in original])
    extended=np.pad(raw,((0,0),(250,250)),mode='reflect',reflect_type='odd')
    filtered=sosfiltfilt(SOS,extended,axis=1).astype(np.float32)
    b=filtered[:,250:875].copy()
    def extended_channels(x):
        x=np.asarray(x,dtype=np.float32)
        return np.stack([x]+[savgol_filter(x.astype(float),11,3,deriv=d,delta=1/125,axis=1,mode='interp') for d in (1,2)],axis=1).astype(np.float32)
    c=extended_channels(filtered)[:,:,250:875]
    d=extended_channels(extended)[:,:,250:875]
    def edges(values):return np.concatenate([values[:,1:,:5],values[:,1:,-5:]],axis=2)
    result=dict(minimal=raw.astype(np.float32),bandpass=b,c_edges=edges(c),d_edges=edges(d))
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite input preparation')
    return result


def record_filename(record):
    return hashlib.sha256(record['source_record_id'].encode()).hexdigest()+'.npz'


def prepare_record(record,frame,directory,baseline):
    path=Path(directory)/'records'/record_filename(record)
    marker=path.with_suffix('.json')
    ids=frame.window_id.tolist()
    request=dict(source_hash=record['content_hash'],ids=ids,preparation=PREPARATION)
    if marker.exists():
        saved=json.loads(marker.read_text());validate_request(saved['request'],request,'Prepared record')
        if not path.exists() or digest(path)!=saved['sha256']:raise ValueError('Prepared record corrupted: '+str(path))
        return saved['review']
    raw=read_record(record)
    indices=np.array([int(w.rsplit('_',1)[1]) for w in ids])
    # Matches the old pipeline's float32 cast before per-window interpolation.
    original=raw[:record['window_count']*625,0].reshape(-1,625)[indices].astype(np.float32)
    values=prepare_values(original)
    first=np.stack([interpolate_zeros(x) for x in original[:2]])
    reproduced=sosfiltfilt(SOS,first,axis=1).astype(np.float32)
    cached=baseline.values(frame.iloc[:2],'ppg')[:,0]
    error=float(abs(reproduced-cached).max())
    if error>1e-6:raise ValueError(f'Cached A reproduction failed: {error}')
    a=baseline.values(frame,'derivatives')
    bc=waveform_channels(values['bandpass'],'derivatives')
    edge=np.r_[0:25,600:625];interior=np.arange(25,600)
    review=dict(record=record['source_record_id'],windows=len(frame),a_reproduction_max=error,
        edge_ppg_difference=float(abs(bc[:,0,edge]-a[:,0,edge]).mean()),
        interior_ppg_difference=float(abs(bc[:,0,interior]-a[:,0,interior]).mean()),
        b_channel_std=np.std(bc.astype(float),axis=(0,2)).tolist(),
        minimal_range=[float(values['minimal'].min()),float(values['minimal'].max())])
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.npz');np.savez(temporary,window_ids=np.asarray(ids),**values);temporary.replace(path)
    write_json(marker,dict(request=request,sha256=digest(path),review=review))
    return review


def prepare_cache(directory,frame,records,baseline,workers=4):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=dict(preparation=PREPARATION,ids=hashlib.sha256('\n'.join(frame.window_id).encode()).hexdigest(),
        source_hashes={r['source_record_id']:r['content_hash'] for r in records},
        source_code_sha256=digest(__file__))
    if (directory/'identity.json').exists():validate_request(json.loads((directory/'identity.json').read_text()),identity,'Input cache')
    else:write_json(directory/'identity.json',identity)
    complete=directory/'complete.json'
    if complete.exists():
        for name,sha in json.loads(complete.read_text())['sha256'].items():
            if digest(directory/name)!=sha:raise ValueError('Input cache hash mismatch: '+name)
        return
    grouped={str(k):v for k,v in frame.groupby('source_record_id',sort=False)}
    records=[r for r in records if r['source_record_id'] in grouped]
    start=time.perf_counter();reviews=[]
    def work(r):return prepare_record(r,grouped[r['source_record_id']],directory,baseline)
    # Four readers, bounded chunks; verified record checkpoints are reusable.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0,len(records),200):
            reviews.extend(pool.map(work,records[first:first+200]))
            elapsed=time.perf_counter()-start
            print(f'Prepared {min(first+200,len(records))}/{len(records)} records in {elapsed:.1f}s',flush=True)
            if first==0:write_json(directory/'benchmark_200.json',dict(records=min(200,len(records)),seconds=elapsed,
                projected_seconds=elapsed*len(records)/min(200,len(records)),workers=workers))
    shapes={'minimal':(len(frame),625),'bandpass':(len(frame),625),
            'c_edges':(len(frame),2,10),'d_edges':(len(frame),2,10)}
    arrays={k:np.lib.format.open_memmap(directory/f'{k}.npy',mode='w+',dtype=np.float32,shape=v) for k,v in shapes.items()}
    positions=pd.Series(np.arange(len(frame)),index=frame.window_id.to_numpy(dtype=object))
    for r in records:
        with np.load(directory/'records'/record_filename(r)) as data:
            rows=positions.loc[data['window_ids']].to_numpy()
            for name,a in arrays.items():a[rows]=data[name]
    for a in arrays.values():a.flush()
    pd.DataFrame(reviews).to_csv(directory/'training_signal_checks.csv',index=False)
    frame[['window_id','source_record_id','split','sbp','dbp']].to_csv(directory/'ids.csv',index=False)
    names=[f'{k}.npy' for k in arrays]+['ids.csv','identity.json','training_signal_checks.csv']
    write_json(complete,dict(sha256={n:digest(directory/n) for n in names},seconds=time.perf_counter()-start,
        windows=len(frame),records=len(records),all_finite=True,test_accessed=False))


class PreparedCache:
    def __init__(self,directory,baseline,version):
        if version not in 'ABCD' or len(version)!=1:raise ValueError('Unknown input version')
        self.baseline=baseline;self.version=version
        if version!='A':
            directory=Path(directory)
            ids=pd.read_csv(directory/'ids.csv',usecols=['window_id']).window_id
            self.index=pd.Series(np.arange(len(ids)),index=ids)
            self.ppg=np.load(directory/('minimal.npy' if version=='D' else 'bandpass.npy'),mmap_mode='r')
            self.edges=np.load(directory/('d_edges.npy' if version=='D' else 'c_edges.npy'),mmap_mode='r') if version in 'CD' else None
    def values(self,frame,representation='derivatives'):
        if self.version=='A':return self.baseline.values(frame,representation)
        rows=self.index.loc[frame.window_id].to_numpy()
        x=waveform_channels(self.ppg[rows],representation)
        if representation=='derivatives' and self.edges is not None:
            edge=self.edges[rows];x[:,1:,:5]=edge[:,:,:5];x[:,1:,-5:]=edge[:,:,5:]
        return x


def gentle_weights(frame):
    from models.pressure_metrics import CUTS,RANGES,TARGETS
    if frame.empty or not frame.split.eq('train').all():raise ValueError('Fitting training rows required')
    y=frame[list(TARGETS)].to_numpy(float)
    if not np.isfinite(y).all():raise ValueError('Nonfinite fitting labels')
    weights=np.empty_like(y);rows=[]
    for j,target in enumerate(TARGETS):
        bins=np.searchsorted(CUTS[target],y[:,j],side='left');counts=np.bincount(bins,minlength=5)
        raw=np.zeros(5);occupied=counts>0
        raw[occupied]=np.minimum(1.5,(counts.max()/counts[occupied])**.25)
        norm=raw[bins].mean();weights[:,j]=raw[bins]/norm
        for k,label in enumerate(RANGES[target]):rows.append(dict(target=target,range=label,count=int(counts[k]),
            raw_weight=float(raw[k]),normalizer=float(norm),weight=float(raw[k]/norm)))
    return weights.astype(np.float32),rows
