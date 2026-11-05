"""Matched PR41 CNN fitting; receives training/inner tables, never selects on validation."""
import copy
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.cnn_model import PPGCNN, build_wave_model
from models.comparison_training import digest, mean_mae, validate_request, write_json
from models.training_support import (BudgetExpired, atomic_torch_save, checkpoint_callback,
    fit_waveform_scaler, scale_waveforms)
from models.pressure_metrics import pressure_groups, pressure_errors
from models.train_feature_mlp import (ResidentBatches, TARGETS, TrainConfig,
    predict_array, regression_metrics, train_epoch)
from models.wave_objectives import WaveObjective, class_targets


BASELINE = dict(representation='ppg',normalization='global',channels=[16,32,64],
    learning_rate=.001,weight_decay=.0001,batch_size=256)


def synchronize(device):
    if device=='cuda':torch.cuda.synchronize()


def seed_training(seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')


def load_wave_model(directory,device='cpu'):
    bundle=torch.load(Path(directory)/'model.pt',map_location=device,weights_only=False)
    p=bundle['parameters']
    model=build_wave_model(p).to(device)
    model.load_state_dict(bundle['state_dict']);model.eval()
    return model,bundle['preprocessing'],bundle


def predict_wave(directory,frame,cache,device='cpu'):
    model,state,bundle=load_wave_model(directory,device)
    # Chunk input transfers bound RAM/VRAM during full train/validation evaluation.
    predictions=[]
    for start in range(0,len(frame),8192):
        x=scale_waveforms(cache.values(frame.iloc[start:start+8192],bundle['parameters']['representation']),state)
        predictions.append(predict_array(model,x,state,bundle['parameters']['batch_size']))
    del model
    if device=='cuda':torch.cuda.empty_cache()
    return np.concatenate(predictions)


def predict_wave_classes(directory,frame,cache,device='cpu'):
    """No reference BP is read: class probabilities depend only on PPG and saved state."""
    model,state,bundle=load_wave_model(directory,device)
    if not bundle['parameters'].get('auxiliary_classification',False):raise ValueError('No classification heads')
    probabilities=[]
    with torch.no_grad():
        for start in range(0,len(frame),8192):
            x=scale_waveforms(cache.values(frame.iloc[start:start+8192],bundle['parameters']['representation']),state)
            values=torch.from_numpy(x).to(device)
            for k in range(0,len(values),bundle['parameters']['batch_size']):
                _,a,b=model.forward_tasks(values[k:k+bundle['parameters']['batch_size']])
                probabilities.append(torch.stack((a.softmax(-1),b.softmax(-1)),dim=1).cpu().numpy())
    del model
    if device=='cuda':torch.cuda.empty_cache()
    return np.concatenate(probabilities)


def benchmark_waveform(cache,frame,output):
    """Identical short optimization sequences; no model is retained for selection."""
    results=[]
    raw=cache.values(frame.iloc[:4096]);target=frame.iloc[:4096][TARGETS].to_numpy(float)
    state=fit_waveform_scaler(raw,target)
    x=scale_waveforms(raw,state);y=((target-state['target_mean'])/state['target_scale']).astype(np.float32)
    for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
        seed_training(2026,device)
        model=PPGCNN().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        a=torch.from_numpy(x[:256]).to(device);b=torch.from_numpy(y[:256]).to(device)
        for _ in range(3):
            opt.zero_grad();nn.functional.mse_loss(model(a),b).backward();opt.step()
        samples=[]
        for _ in range(3):
            synchronize(device);start=time.perf_counter()
            for _ in range(20):
                opt.zero_grad();nn.functional.mse_loss(model(a),b).backward();opt.step()
            synchronize(device);samples.append(time.perf_counter()-start)
        results.append(dict(device=device,seconds_per_20_batches=samples,median=float(np.median(samples))))
        del model,opt,a,b
        if device=='cuda':torch.cuda.empty_cache()
    chosen=min(results,key=lambda r:r['median'])['device']
    write_json(Path(output)/'cnn_hardware_benchmark.json',dict(measurements=results,chosen=chosen,
        precision='float32; no AMP/TF32',warmup_batches=3,timed_batches_per_repeat=20,repeats=3))
    return chosen


class PilotPaused(RuntimeError):
    """A saved pilot checkpoint is intentionally paused, not a completed fit."""


def fit_wave(train,holdout,cache,parameters,seed,device,directory,budget,*,limit=100,duration=None,provenance=None,target_weights=None,pause_after_epoch=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    request=dict(parameters=parameters,seed=seed,device=device,limit=limit,duration=duration,
        fitting_ids=digest_ids(train),holdout_ids=digest_ids(holdout) if holdout is not None else None,
        provenance=provenance)
    if target_weights is not None:
        import hashlib
        target_weights=np.asarray(target_weights,dtype=np.float32)
        if target_weights.shape!=(len(train),2) or not np.isfinite(target_weights).all() or (target_weights<=0).any():
            raise ValueError('Target weights must be aligned positive finite N x 2')
        request['target_weights_sha256']=hashlib.sha256(target_weights.tobytes()).hexdigest()
    request_path=directory/'request.json'
    if request_path.exists():
        validate_request(json.loads(request_path.read_text()),request,'Waveform fit')
    if (directory/'complete.json').exists():
        marker=json.loads((directory/'complete.json').read_text())
        required={'model.pt','fit.json','request.json','history.csv','pressure_history.csv'}
        if not required<=set(marker['sha256']):raise ValueError('Completed CNN integrity manifest is incomplete')
        for name,sha in marker['sha256'].items():
            if digest(directory/name)!=sha:raise ValueError('Completed CNN artifact changed')
        return json.loads((directory/'fit.json').read_text())
    if not request_path.exists():write_json(request_path,request)
    budget.check(selection=duration is None)
    started=time.perf_counter();seed_training(seed,device)
    raw=cache.values(train,parameters['representation']);answers=train[TARGETS].to_numpy(float)
    state=fit_waveform_scaler(raw,answers,parameters['normalization'])
    objective=WaveObjective(parameters,state) if ('loss' in parameters or parameters.get('auxiliary_classification')) else None
    if objective is not None and target_weights is not None:raise ValueError('Campaign objectives cannot be combined with target weights')
    x=scale_waveforms(raw,state);del raw
    target=((answers-state['target_mean'])/state['target_scale']).astype(np.float32)
    cfg=TrainConfig(seed=seed,device=device,batch_size=parameters['batch_size'])
    tensors=[torch.from_numpy(x),torch.from_numpy(target)]
    if target_weights is not None:tensors.append(torch.from_numpy(target_weights))
    if parameters.get('auxiliary_classification',False):tensors.append(torch.from_numpy(class_targets(answers)))
    loader=ResidentBatches(tensors,cfg) if device=='cuda' else DataLoader(TensorDataset(*tensors),
        batch_size=cfg.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
    del x,target,tensors
    other=None;other_y=None
    if holdout is not None:
        other_y=holdout[TARGETS].to_numpy(float)
        other=torch.from_numpy(scale_waveforms(cache.values(holdout,parameters['representation']),state)).to(device)
    model=build_wave_model(parameters).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=parameters['learning_rate'],weight_decay=parameters['weight_decay'])
    callback=checkpoint_callback(directory,budget,selection=duration is None)
    best_state=None;best_score=float('inf');best_epoch=0;stale=0;history=[];pressure=[]
    epochs=int(duration or limit);first=1
    checkpoint=directory/'last_checkpoint.pt'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        validate_request(saved['preprocessing'],state,'Waveform checkpoint preprocessing')
        if 'request' in saved:
            # A continuation may increase its ceiling; every other requested field stays fixed.
            expected=dict(request);expected['limit']=saved['request']['limit']
            validate_request(saved['request'],expected,'Waveform checkpoint request')
            if request['limit']<saved['request']['limit']:raise ValueError('Checkpoint ceiling cannot shrink')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best_state=saved['best_state'];best_score=saved['best_score'];best_epoch=saved['best_epoch'];stale=saved['stale']
        history=saved['history'];pressure=saved['pressure_history'];first=saved['epoch']+1
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'].cpu())
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        (loader.order_loader.generator if isinstance(loader,ResidentBatches) else loader.generator).set_state(saved['order_rng'].cpu())
    loading=time.perf_counter()-started;optimization=validation=diagnostics=0.
    epoch=first-1
    groups=pressure_groups(holdout) if holdout is not None else []
    training_groups=pressure_groups(train)
    for current in range(first,epochs+1):
        if duration is None and stale>=10:break
        epoch=current;synchronize(device);tic=time.perf_counter()
        losses=train_epoch(model,loader,optimizer,nn.MSELoss(),return_metrics=True,objective_fn=objective)
        if not all(np.isfinite(v) for v in losses.values()):
            raise FloatingPointError('Nonfinite waveform optimization loss')
        synchronize(device);optimization+=time.perf_counter()-tic
        row=dict(epoch=epoch,train_mse_standardized=losses['mse'],optimization_objective=losses['objective'])
        row.update({k:v for k,v in losses.items() if k not in ('mse','objective')})
        if other is not None:
            tic=time.perf_counter();yp=predict_array(model,other,state,cfg.batch_size)
            scores=regression_metrics(yp,other_y);score=mean_mae(scores)
            if not np.isfinite(yp).all() or not np.isfinite(score):
                raise FloatingPointError('Nonfinite waveform holdout prediction or score')
            row.update(scores,validation_mean_mae=score)
            row.update({t+'_bias':float((yp-other_y)[:,j].mean()) for j,t in enumerate(TARGETS)})
            pressure.extend(dict(epoch=epoch,split='inner',**r) for r in pressure_errors(yp,other_y,groups))
            validation+=time.perf_counter()-tic
        else:score=0.
        if other is None or score<best_score:
            best_state=copy.deepcopy(model.state_dict());best_score=score;best_epoch=epoch;stale=0
        else:stale+=1
        if epoch==1 or epoch%5==0 or epoch==epochs or stale>=10:
            tic=time.perf_counter();yp=predict_array(model,loader.dataset.tensors[0],state,cfg.batch_size)
            row.update({'training_'+k:v for k,v in regression_metrics(yp,answers).items()})
            pressure.extend(dict(epoch=epoch,split='train',**r) for r in pressure_errors(yp,answers,training_groups))
            diagnostics+=time.perf_counter()-tic
        history.append(row)
        callback(dict(epoch=epoch,model=model,optimizer=optimizer,loader=loader,best_state=best_state,
            best_score=best_score,best_epoch=best_epoch,stale=stale,history=history,
            pressure_history=pressure,preprocessing=state,request=request))
        if pause_after_epoch is not None and epoch==pause_after_epoch and epoch<epochs:
            finite_gradients=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            if not finite_gradients:raise FloatingPointError('Nonfinite pilot gradients')
            write_json(directory/'pilot.json',dict(epoch=epoch,seconds=time.perf_counter()-started,
                loading_seconds=loading,optimization_seconds=optimization,validation_seconds=validation,
                diagnostics_seconds=diagnostics,finite=True,checkpoint_sha256=digest(checkpoint),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                finite_gradients=finite_gradients,complete=False))
            del loader,other,model;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
            raise PilotPaused('Five-epoch checkpoint saved; resume the same request')
        if epoch==1 or epoch%5==0:
            print(f'CNN {directory.name} epoch{epoch} best{best_epoch} inner={best_score:.4f}',flush=True)
    if not best_state:raise ValueError('No trained CNN checkpoint')
    model.load_state_dict(best_state);model.eval()
    yp=predict_array(model,loader.dataset.tensors[0],state,cfg.batch_size)
    metrics={'train':regression_metrics(yp,answers)}
    pressure.extend(dict(epoch=best_epoch,split='train_restored',**r) for r in pressure_errors(yp,answers,training_groups))
    if other is not None:
        expected=predict_array(model,other,state,cfg.batch_size)
        metrics['inner']=regression_metrics(expected,other_y)
    else:expected=yp[:512]
    atomic_torch_save(dict(state_dict={k:v.cpu() for k,v in best_state.items()},parameters=parameters,
        preprocessing=state,seed=seed,duration=[best_epoch],provenance=provenance,
        objective_configuration=objective.configuration() if objective else None),directory/'model.pt')
    restored,_,_=load_wave_model(directory,device)
    reload_values=other if other is not None else loader.dataset.tensors[0][:512]
    actual=predict_array(restored,reload_values,state,cfg.batch_size)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-5)
    pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
    pd.DataFrame(pressure).to_csv(directory/'pressure_history.csv',index=False)
    elapsed=time.perf_counter()-started
    info=dict(family='cnn',parameters=parameters,seed=seed,device=device,directory=str(directory),duration=[best_epoch],
        stopped=[epoch],ceiling_censored=bool(holdout is not None and epoch==limit and stale<10),
        metrics=metrics,inner_mean_mae=mean_mae(metrics['inner']) if holdout is not None else None,
        fitting_seconds=elapsed,total_seconds=elapsed,loading_seconds=loading,optimization_seconds=optimization,
        validation_seconds=validation,training_diagnostics_seconds=diagnostics,epochs_this_attempt=len(history)-first+1,
        reload_max_absolute_difference=float(abs(actual-expected).max()))
    write_json(directory/'fit.json',info)
    write_json(directory/'complete.json',dict(sha256={n:digest(directory/n) for n in
        ['model.pt','fit.json','request.json','history.csv','pressure_history.csv']},test_accessed=False))
    del loader,other,model,restored;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return info


def digest_ids(frame):
    import hashlib
    return hashlib.sha256('\n'.join(frame.window_id).encode()).hexdigest()
