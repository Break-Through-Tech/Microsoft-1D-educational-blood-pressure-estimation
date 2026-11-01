"""Supported selected-model training, development evaluation, and PPG-only prediction.

Run ``python -m models.selected_models --help`` from the repository root.
The reusable loader performs no training and accepts CPU inference by default.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from models.comparison_training import digest, fit_and_save, tree_inputs, write_json
from models.pressure_metrics import pressure_errors, pressure_groups
from models.selected_inputs import (load_prepared_split, prepare_prediction_inputs,
                                    prepare_selected_inputs)
from models.training_support import scale_waveforms
from models.train_feature_mlp import TARGETS, predict_array, regression_metrics
from models.train_waveform_comparison import fit_wave, load_wave_model, seed_training

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/selected_models.json'
DEFAULT_ARTIFACTS = ROOT / 'artifacts/selected-v1'
MODEL_NAMES = ('recommended', 'resnet_l1', 'lightgbm_16')


def component_directory(root, name, seed):
    return Path(root) / 'components' / name / f'seed_{seed}'


def seal_bundle(directory):
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix():digest(p) for p in directory.rglob('*')
             if p.is_file() and p != directory / 'manifest.json'}
    write_json(directory / 'manifest.json', dict(format_version=1, sha256=files))


class SelectedPredictor:
    """Loaded weights and fitted preprocessing, reusable across prediction calls."""

    def __init__(self, directory=DEFAULT_ARTIFACTS, model='recommended', seed=2026, device='cpu'):
        self.directory, self.device = Path(directory), device
        self.config = json.loads((self.directory / 'config.json').read_text())
        if model not in self.config['models'] or seed not in self.config['seeds']:
            raise ValueError('Unknown selected model or seed')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; choose --device cpu')
        self.model_name, self.seed = model, seed
        self.weights = self.config['models'][model]
        manifest = json.loads((self.directory / 'manifest.json').read_text())['sha256']
        required = ['config.json']
        for name in self.weights:
            base = component_directory(self.directory, name, seed)
            names = ['model.pt'] if self.config['components'][name]['family'] == 'cnn' else ['sbp.txt','dbp.txt','preprocessing.json']
            required.extend((base / n).relative_to(self.directory).as_posix() for n in names)
        for name in required:
            path = self.directory / name
            if name not in manifest or not path.is_file() or digest(path) != manifest[name]:
                raise ValueError('Missing or changed model artifact: ' + name)
        # Match the recorded deterministic float32 inference settings on CPU/CUDA.
        seed_training(seed, device)
        self.components = {}
        for name in self.weights:
            path = component_directory(self.directory, name, seed)
            spec = self.config['components'][name]
            if spec['family'] == 'cnn':
                net, state, bundle = load_wave_model(path, device)
                if bundle['parameters'] != spec['parameters'] or bundle['seed'] != seed:
                    raise ValueError('Checkpoint parameters/seed differ from the selected configuration')
                if bundle['duration'] != spec['durations'][str(seed)]:
                    raise ValueError('Checkpoint duration differs from the frozen duration')
                self.components[name] = (net, state)
            else:
                import lightgbm as lgb
                state = json.loads((path / 'preprocessing.json').read_text())
                if state['feature_names'] != spec['feature_names']:
                    raise ValueError('LightGBM feature order changed')
                self.components[name] = ([lgb.Booster(model_file=str(path / f'{t}.txt')) for t in TARGETS], state)

    def _predict(self, frame, waveform_values):
        result = np.zeros((len(frame), 2), dtype=float)
        for name, weight in self.weights.items():
            model, state = self.components[name]
            spec = self.config['components'][name]
            if spec['family'] == 'cnn':
                values = scale_waveforms(waveform_values, state)
                prediction = predict_array(model, values, state, spec['parameters']['batch_size'])
            else:
                values = tree_inputs(frame, state)
                prediction = np.column_stack([m.predict(values, num_threads=2) for m in model])
            result += weight * prediction
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite selected-model prediction')
        return result

    def predict_prepared(self, frame, cache):
        """Predict from existing ID-aligned development features and Input C."""
        parts = []
        for start in range(0, len(frame), 8192):
            batch = frame.iloc[start:start+8192]
            needs_wave = any(self.config['components'][n]['family']=='cnn' for n in self.weights)
            values = cache.values(batch, 'derivatives') if needs_wave else None
            parts.append(self._predict(batch, values))
        return np.concatenate(parts) if parts else np.empty((0, 2))

    def predict_ppg(self, ppg, window_ids):
        """Return ID-matched SBP/DBP in mmHg; invalid PPG rows retain a reason.

        Inputs are N x 625 source-amplitude PPG samples at 125 Hz. ABP, labels,
        ECG, and dataset split information are not prediction inputs.
        """
        ids = np.asarray(window_ids, dtype=str)
        if ids.ndim != 1 or len(ids) != len(ppg) or len(set(ids)) != len(ids):
            raise ValueError('Supply one unique window ID per PPG row')
        needs_features = any(self.config['components'][n]['family']=='lightgbm' for n in self.weights)
        needs_wave = any(self.config['components'][n]['family']=='cnn' for n in self.weights)
        positions, values, frame, reasons = prepare_prediction_inputs(ppg, features=needs_features, waveforms=needs_wave)
        result = pd.DataFrame({'window_id':ids, 'predicted_sbp':np.nan, 'predicted_dbp':np.nan,
                               'rejection_reason':reasons})
        if len(positions):
            result.loc[positions, ['predicted_sbp','predicted_dbp']] = self._predict(frame, values)
        return result


def load_selected_model(model='recommended', seed=2026, artifact_dir=DEFAULT_ARTIFACTS, device='cpu'):
    """Load once, then call predictor.predict_ppg(ppg, window_ids) repeatedly."""
    return SelectedPredictor(artifact_dir, model, seed, device)


class FixedRun:
    def check(self, selection=True):
        pass


def train_selected(data_dir, inputs_dir, output, model='recommended', seeds=(2026,), device='cpu', config_path=DEFAULT_CONFIG):
    """Refit only frozen settings/durations, without a search or validation stopping."""
    config = json.loads(Path(config_path).read_text())
    if model not in (*MODEL_NAMES, 'all') or not seeds or any(s not in config['seeds'] for s in seeds):
        raise ValueError('Unknown model or seed')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new training output directory; frozen runs are never overwritten')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    train, cache = load_prepared_split(data_dir, inputs_dir, config, 'train')
    names = MODEL_NAMES if model == 'all' else (model,)
    needed = sorted({n for m in names for n in config['models'][m]})
    output.mkdir(parents=True)
    shutil.copyfile(config_path, output / 'config.json')
    provenance = dict(source_table_sha256=config['source_table_sha256'], label_version=config['label_version'],
        input_preparation=config['waveform_preparation'], selection='frozen fixed-duration reproduction', test_accessed=False,
        code_sha256={p.name:digest(p) for p in (ROOT/'models').glob('*.py')})
    for seed in dict.fromkeys(seeds):
        seed_training(seed, device)
        for name in needed:
            spec = config['components'][name]
            directory = component_directory(output, name, seed)
            duration = spec['durations'][str(seed)]
            if spec['family'] == 'cnn':
                fit_wave(train, None, cache, spec['parameters'], seed, device, directory,
                         FixedRun(), duration=duration[0], provenance=provenance)
            else:
                fit_and_save('lightgbm', train, spec['feature_names'], spec['parameters'], seed,
                    {'device':'cpu'}, directory, duration=duration, provenance=provenance)
    write_json(output / 'reproduction.json', dict(models=list(names), seeds=list(dict.fromkeys(seeds)), provenance=provenance))
    seal_bundle(output)
    return output


def evaluate_selected(data_dir, inputs_dir, artifact_dir, output, model='recommended', seeds=(2026,), splits=('validation',), device='cpu'):
    if any(s not in ('train','validation') for s in splits):
        raise ValueError('Test evaluation is not part of this workflow')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new evaluation output directory')
    config = json.loads((Path(artifact_dir)/'config.json').read_text())
    names = MODEL_NAMES if model == 'all' else (model,)
    output.mkdir(parents=True)
    metrics, pressure = [], []
    for split in dict.fromkeys(splits):
        frame, cache = load_prepared_split(data_dir, inputs_dir, config, split)
        for name in names:
            for seed in dict.fromkeys(seeds):
                predictor = load_selected_model(name, seed, artifact_dir, device)
                predictions = predictor.predict_prepared(frame, cache)
                saved = frame[['window_id','source_record_id','split','sbp','dbp']].copy()
                saved[['predicted_sbp','predicted_dbp']] = predictions
                saved.to_csv(output/f'{name}_{seed}_{split}_predictions.csv', index=False)
                row = dict(model=name, seed=seed, split=split, **regression_metrics(predictions, frame[TARGETS].to_numpy(float)))
                for j,t in enumerate(TARGETS):
                    row[t+'_bias'] = float((predictions[:,j]-frame[t].to_numpy()).mean())
                    errors = pd.Series(abs(predictions[:,j]-frame[t].to_numpy()), index=frame.source_record_id)
                    row['recording_macro_'+t+'_mae'] = float(errors.groupby(level=0).mean().mean())
                metrics.append(row)
                pressure.extend(dict(model=name, seed=seed, split=split, **r) for r in pressure_errors(
                    predictions, frame[TARGETS].to_numpy(float), pressure_groups(frame)))
                del predictor
    pd.DataFrame(metrics).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(pressure).to_csv(output/'pressure_metrics.csv', index=False)
    write_json(output/'evaluation.json', dict(test_accessed=False, source_table_sha256=config['source_table_sha256']))
    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Explicitly build selected inputs after shared cleaning')
    prepare.add_argument('--data-dir', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--workers', type=int, default=4)
    for name in ('train','evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--data-dir', type=Path, required=True)
        command.add_argument('--inputs-dir', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--model', choices=(*MODEL_NAMES,'all'), default='recommended')
        command.add_argument('--seeds', nargs='+', type=int, choices=(2026,2027,2028), default=[2026])
        command.add_argument('--device', choices=('cpu','cuda'), default='cpu')
        if name == 'evaluate':
            command.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
            command.add_argument('--split', nargs='+', choices=('train','validation'), default=['validation'])
    predict = commands.add_parser('predict')
    predict.add_argument('--input', type=Path, required=True, help='NPZ containing ppg and window_ids')
    predict.add_argument('--output', type=Path, required=True)
    predict.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    predict.add_argument('--model', choices=MODEL_NAMES, default='recommended')
    predict.add_argument('--seed', type=int, choices=(2026,2027,2028), default=2026)
    predict.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_selected_inputs(args.data_dir, args.output, json.loads(DEFAULT_CONFIG.read_text()), args.workers)
    elif args.command == 'train':
        print(train_selected(args.data_dir, args.inputs_dir, args.output, args.model, args.seeds, args.device))
    elif args.command == 'evaluate':
        print(evaluate_selected(args.data_dir, args.inputs_dir, args.artifacts, args.output, args.model, args.seeds, args.split, args.device).to_string(index=False))
    else:
        if args.output.exists():
            raise FileExistsError('Prediction output already exists')
        model = load_selected_model(args.model, args.seed, args.artifacts, args.device)
        with np.load(args.input, allow_pickle=False) as data:
            result = model.predict_ppg(data['ppg'], data['window_ids'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(result.to_string(index=False))


if __name__ == '__main__':
    main()
