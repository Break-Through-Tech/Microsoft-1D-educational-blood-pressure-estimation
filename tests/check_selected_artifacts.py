"""Explicit, read-only saved-artifact audit; no fitting or reserved test inputs.

python -m tests.check_selected_artifacts --bundle artifacts/selected-v1 --output data/derived/artifact_audit.json
Add --evidence artifacts/evaluation-v1 to recompute full saved development metrics.
"""
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.comparison_training import digest
from models.selected_inputs import prepare_prediction_inputs
from models.selected_models import load_selected_model


def verify_manifest(directory):
    directory=Path(directory)
    files=json.loads((directory/'manifest.json').read_text())['sha256']
    for name,expected in files.items():
        if digest(directory/name)!=expected:raise AssertionError('Artifact hash changed: '+name)
    return len(files)


def check_bundle(directory):
    directory=Path(directory)
    count=verify_manifest(directory)
    with np.load(directory/'samples/ppg.npz',allow_pickle=False) as source:
        ppg,ids=source['ppg'],source['window_ids']
    positions,wave,features,reasons=prepare_prediction_inputs(ppg)
    assert len(positions)==len(ppg) and not any(reasons)
    with np.load(directory/'samples/expected_input_c.npz',allow_pickle=False) as original:
        np.testing.assert_array_equal(wave,original['X'])
    original=pd.read_csv(directory/'samples/expected_features.csv',float_precision='round_trip')
    np.testing.assert_allclose(features,original[features.columns],rtol=0,atol=1e-10,equal_nan=True)
    references=pd.read_csv(directory/'samples/expected_predictions.csv',float_precision='round_trip')
    differences=[]
    for seed in (2026,2027,2028):
        for name in ('recommended','resnet_l1','lightgbm_16'):
            model=load_selected_model(name,seed,directory)
            predictions=model.predict_ppg(ppg,ids)
            expected=references[references.model.eq(name)&references.seed.eq(seed)]
            assert predictions.window_id.tolist()==expected.window_id.tolist()
            a=predictions[['predicted_sbp','predicted_dbp']].to_numpy()
            b=expected[['predicted_sbp','predicted_dbp']].to_numpy()
            np.testing.assert_allclose(a,b,rtol=0,atol=.001)
            differences.append(dict(model=name,seed=seed,max_absolute_difference_mmhg=float(abs(a-b).max())))
            del model
    return dict(files_verified=count,raw_ppg_input_c_exact=True,feature_preparation_matches=True,replays=differences)


def chunks(path):
    return pd.read_csv(path,chunksize=100000,float_precision='round_trip')


def check_evidence(directory, metrics_path):
    directory=Path(directory)
    count=verify_manifest(directory)
    metrics=pd.read_csv(metrics_path)
    id_hashes={};checks=[]
    for name in ('recommended','resnet_l1','lightgbm_16'):
        for seed in (2026,2027,2028):
            for split in ('train','validation'):
                path=directory/f'predictions/{name}/seed_{seed}/{split}_predictions.csv'
                total=0;absolute=np.zeros(2);squares=np.zeros(2);bias=np.zeros(2);identity=hashlib.sha256()
                recording_errors={};seen=set()
                for frame in chunks(path):
                    assert frame.split.eq(split).all()
                    ids=frame.window_id.tolist()
                    assert not frame.window_id.duplicated().any() and not seen.intersection(ids)
                    seen.update(ids);identity.update(('\n'.join(ids)+'\n').encode())
                    answers=frame[['sbp','dbp']].to_numpy()
                    predictions=frame[['predicted_sbp','predicted_dbp']].to_numpy()
                    assert np.isfinite(answers).all() and np.isfinite(predictions).all()
                    error=predictions-answers
                    total+=len(frame);absolute+=abs(error).sum(axis=0);squares+=(error**2).sum(axis=0);bias+=error.sum(axis=0)
                    groups=pd.DataFrame({'record':frame.source_record_id,'a':abs(error[:,0]),'b':abs(error[:,1])})
                    for record,group in groups.groupby('record',sort=False):
                        saved=recording_errors.setdefault(record,np.zeros(3))
                        saved+=np.array([group.a.sum(),group.b.sum(),len(group)])
                actual=dict(sbp_mae=absolute[0]/total,dbp_mae=absolute[1]/total,
                    sbp_rmse=np.sqrt(squares[0]/total),dbp_rmse=np.sqrt(squares[1]/total),
                    sbp_bias=bias[0]/total,dbp_bias=bias[1]/total)
                by_record=np.array(list(recording_errors.values()))
                actual.update(recording_macro_sbp_mae=np.mean(by_record[:,0]/by_record[:,2]),
                              recording_macro_dbp_mae=np.mean(by_record[:,1]/by_record[:,2]))
                expected=metrics[metrics.model.eq(name)&metrics.seed.eq(seed)&metrics.split.eq(split)].iloc[0]
                assert total==int(expected.windows) and len(recording_errors)==int(expected.recordings)
                for field,value in actual.items():np.testing.assert_allclose(value,expected[field],rtol=0,atol=1e-8)
                value=identity.hexdigest()
                if split in id_hashes:assert value==id_hashes[split]
                else:id_hashes[split]=value
                checks.append(dict(model=name,seed=seed,split=split,windows=total,recordings=len(recording_errors)))
                print('Recomputed',name,seed,split,flush=True)
    max_blend_error=0.
    for seed in (2026,2027,2028):
        for split in ('train','validation'):
            readers=[chunks(directory/f'predictions/{name}/seed_{seed}/{split}_predictions.csv')
                     for name in ('recommended','multitask_resnet','cnn_c','lightgbm_16')]
            for parts in zip_longest(*readers):
                assert all(p is not None for p in parts)
                reference=parts[0]
                for frame in parts[1:]:
                    assert frame.window_id.tolist()==reference.window_id.tolist()
                    np.testing.assert_array_equal(frame[['sbp','dbp']],reference[['sbp','dbp']])
                arrays=[p[['predicted_sbp','predicted_dbp']].to_numpy() for p in parts]
                blended=.8*arrays[1]+.13*arrays[2]+.07*arrays[3]
                error=float(abs(blended-arrays[0]).max());max_blend_error=max(error,max_blend_error)
                assert error<1e-9
    return dict(files_verified=count,metric_checks=checks,aligned_id_sha256=id_hashes,
                maximum_blend_arithmetic_difference=max_blend_error,test_accessed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--evidence',type=Path)
    parser.add_argument('--metrics',type=Path,default=Path('reports/final/selected_metrics.csv'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=dict(bundle=check_bundle(args.bundle),test_accessed=False)
    if args.evidence:result['evidence']=check_evidence(args.evidence,args.metrics)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Saved-artifact audit passed:',args.output,flush=True)


if __name__=='__main__':main()
