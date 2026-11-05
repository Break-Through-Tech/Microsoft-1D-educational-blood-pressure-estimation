"""Fixed pressure groups and training-only capped, target-specific weights."""
import numpy as np
import pandas as pd

TARGETS = ('sbp', 'dbp')
CUTS = {'sbp': [90.,120.,140.,180.], 'dbp': [60.,80.,90.,120.]}
RANGES = {'sbp': ['<=90','(90,120]','(120,140]','(140,180]','>180'],
          'dbp': ['<=60','(60,80]','(80,90]','(90,120]','>120']}
WEIGHT_VERSION = 'capped_sqrt_frequency_per_target_v1'


def pressure_weights(train):
    """Return N×2 weights in input row order; accepts training rows only."""
    if train.empty or not train.split.eq('train').all():
        raise ValueError('Weights require eligible training rows only')
    if 'usable' in train and not train.usable.astype(str).str.lower().isin(['true','1']).all():
        raise ValueError('Weights require eligible training rows only')
    y=train[list(TARGETS)].to_numpy(float)
    if not np.isfinite(y).all():raise ValueError('Nonfinite training labels')
    weights=np.empty_like(y);rows=[];summary=[]
    for j,target in enumerate(TARGETS):
        indices=np.searchsorted(CUTS[target],y[:,j],side='left')
        counts=np.bincount(indices,minlength=5)
        raw=np.zeros(5);occupied=counts>0
        raw[occupied]=np.minimum(3.,np.sqrt(counts.max()/counts[occupied]))
        normalizer=float(np.mean(raw[indices]));weights[:,j]=raw[indices]/normalizer
        for b,label in enumerate(RANGES[target]):
            m=indices==b
            rows.append({'target':target,'range':label,'windows':int(counts[b]),
                'recordings':int(train.loc[m,'source_record_id'].nunique()),
                'frequency':float(counts[b]/len(train)),'raw_weight':float(raw[b]),
                'normalizer':normalizer,'weight':float(raw[b]/normalizer),
                'weighted_mass':float(weights[m,j].sum()/len(train))})
        w=weights[:,j]
        summary.append({'target':target,'windows':len(train),'mean_weight':float(w.mean()),
            'min_weight':float(w.min()),'max_weight':float(w.max()),
            'effective_sample_size':float(w.sum()**2/np.square(w).sum())})
    return weights,pd.DataFrame(rows),summary


def pressure_groups(frame):
    """Cache masks/counts once; bins depend only on unchanged reference labels."""
    groups=[]
    for j,target in enumerate(TARGETS):
        values=frame[target].to_numpy(float)
        index=np.searchsorted(CUTS[target],values,side='left')
        masks=[(label,index==b) for b,label in enumerate(RANGES[target])]
        masks += [('(90,140]',(values>90)&(values<=140))] if target=='sbp' else [('>90',values>90)]
        for label,mask in masks:
            groups.append((j,target,label,mask,int(frame.loc[mask,'source_record_id'].nunique())))
    return groups


def pressure_errors(predictions, answers, groups):
    p,a=np.asarray(predictions),np.asarray(answers)
    if p.shape!=a.shape or p.ndim!=2 or p.shape[1]!=2:
        raise ValueError('Predictions/answers must be aligned N×2 arrays')
    error=p-a
    rows=[]
    for j,target,label,mask,records in groups:
        e=error[mask,j];n=len(e)
        rows.append({'target':target,'range':label,'windows':n,'recordings':records,
            'mae':float(abs(e).mean()) if n else np.nan,
            'rmse':float(np.sqrt(np.square(e).mean())) if n else np.nan,
            'bias':float(e.mean()) if n else np.nan})
    return rows
