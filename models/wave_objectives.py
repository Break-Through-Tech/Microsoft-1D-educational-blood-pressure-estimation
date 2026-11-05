"""Explicit standardized regression objectives and PPG-only auxiliary heads."""
import numpy as np
import torch
from torch.nn import functional as F

from models.pressure_metrics import CUTS, TARGETS


def class_targets(answers):
    values=np.asarray(answers,dtype=float)
    if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():
        raise ValueError('Classification targets require finite N x 2 reference labels')
    return np.column_stack([np.searchsorted(CUTS[t],values[:,j],side='left')
        for j,t in enumerate(TARGETS)]).astype(np.int64)


class WaveObjective:
    def __init__(self,parameters,state):
        self.kind=parameters.get('loss','mse')
        if self.kind not in ('mse','huber','l1'):raise ValueError('Unknown regression loss')
        self.auxiliary=parameters.get('auxiliary_classification',False)
        self.strength=float(parameters.get('auxiliary_lambda',0.))
        if self.strength<0 or not np.isfinite(self.strength):raise ValueError('Invalid auxiliary lambda')
        if not self.auxiliary and self.strength:raise ValueError('Auxiliary lambda requires classification heads')
        self.delta=np.asarray(parameters.get('huber_delta_mmHg',[10.,5.]),dtype=float)/np.asarray(state['target_scale'])
        if self.delta.shape!=(2,) or not np.isfinite(self.delta).all() or (self.delta<=0).any():
            raise ValueError('Huber thresholds must be two positive finite values')
        self.multiplier=float(parameters.get('huber_multiplier',2.))
        if self.multiplier!=2.:raise ValueError('This campaign defines twice-conventional Huber')

    def regression(self,predictions,answers):
        error=predictions-answers
        if self.kind=='mse':return F.mse_loss(predictions,answers)
        if self.kind=='l1':return error.abs().mean()
        delta=torch.as_tensor(self.delta,device=error.device,dtype=error.dtype)
        absolute=error.abs()
        return self.multiplier*torch.where(absolute<=delta,.5*error.square(),
            delta*(absolute-.5*delta)).mean()

    def __call__(self,model,batch):
        inputs,answers,*extra=batch
        if self.auxiliary:
            if len(extra)!=1 or extra[0].dtype!=torch.int64 or extra[0].shape!=answers.shape:
                raise ValueError('Auxiliary batches require aligned integer class targets, not weights')
            predicted,sbp,dbp=model.forward_tasks(inputs)
            ce_sbp=F.cross_entropy(sbp,extra[0][:,0]);ce_dbp=F.cross_entropy(dbp,extra[0][:,1])
        else:
            if extra:raise ValueError('Alternative waveform losses do not accept target weights')
            predicted=model(inputs);ce_sbp=ce_dbp=predicted.new_zeros(())
        regression=self.regression(predicted,answers)
        loss=regression+self.strength*(ce_sbp+ce_dbp)/2
        return loss,F.mse_loss(predicted,answers),dict(regression_loss=regression,
            sbp_cross_entropy=ce_sbp,dbp_cross_entropy=ce_dbp)

    def configuration(self):
        return dict(loss=self.kind,huber_delta_standardized=self.delta.tolist(),
            huber_multiplier=self.multiplier,auxiliary_classification=self.auxiliary,
            auxiliary_lambda=self.strength)
