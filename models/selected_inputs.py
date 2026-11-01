"""PPG-only prediction inputs and explicit preparation for selected-model reruns."""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from cleaning.pipeline import ALL_FEATURE_NAMES, interpolate_zeros, ppg_features, raw_signal_reason
from models.comparison_training import digest, write_json
from models.preprocessing_inputs import PREPARATION, PreparedCache, prepare_cache, prepare_values
from models.training_support import WaveCache, pulse_additions, waveform_channels
from models.train_feature_mlp import prepare_rows, read_feature_table

EXTRA_FEATURES = ['median_post_peak_area_fraction', 'median_normalized_max_upstroke_slope']


def input_c(ppg):
    """Reconstruct the frozen reflected/filter/derivative input, including its edges."""
    prepared = prepare_values(ppg)
    values = waveform_channels(prepared['bandpass'], 'derivatives')
    values[:, 1:, :5] = prepared['c_edges'][:, :, :5]
    values[:, 1:, -5:] = prepared['c_edges'][:, :, 5:]
    return values


def prepare_prediction_inputs(ppg, *, features=True, waveforms=True):
    """Return retained positions, Input C, feature rows, and per-input failure reasons.

    Features 1-14 use the original float64 NeuroKit result; additions 15-16 use
    its float32 cache representation, exactly as the completed comparison did.
    ABP-dependent training eligibility cannot be established during inference.
    """
    values = np.asarray(ppg, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 625:
        raise ValueError('PPG must have shape N x 625 at 125 Hz')
    positions, rows = [], []
    reasons = [''] * len(values)
    for index, original in enumerate(values):
        reason = raw_signal_reason(original)
        if reason:
            reasons[index] = 'ppg:' + reason
            continue
        if features:
            import neurokit2 as nk
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    signals, _ = nk.ppg_process(interpolate_zeros(original), sampling_rate=125, method='elgendi')
                clean = signals['PPG_Clean'].to_numpy(dtype=float)
                peaks = np.flatnonzero(signals['PPG_Peaks'].to_numpy() > 0)
                if not np.isfinite(clean).all():
                    raise ValueError('Nonfinite cleaned PPG')
                basic, _ = ppg_features(clean, peaks)
                additions = pulse_additions(clean.astype(np.float32), peaks)
                row = np.r_[basic, additions]
                if np.isinf(row).any() or np.isnan(row).all():
                    raise ValueError('Unusable PPG features')
                rows.append(row)
            except Exception as error:
                reasons[index] = 'ppg:feature_extraction:' + type(error).__name__
                continue
        positions.append(index)
    positions = np.asarray(positions, dtype=int)
    inputs = input_c(values[positions]) if waveforms and len(positions) else np.empty((0, 3, 625), np.float32)
    frame = pd.DataFrame(rows, columns=ALL_FEATURE_NAMES + EXTRA_FEATURES) if features else pd.DataFrame(index=range(len(positions)))
    return positions, inputs, frame, reasons


def read_cohort(data_dir, config):
    """Require the frozen development cohort; never introduce or score test rows."""
    data_dir = Path(data_dir)
    table_path = data_dir / 'model_table.csv'
    if digest(table_path) != config['source_table_sha256']:
        raise ValueError('Prepared table differs from the frozen cohort; use the documented pipeline/configuration')
    table = read_feature_table(table_path)
    if set(table['split']) != {'train', 'validation'}:
        raise ValueError('Only the saved train and validation cohort is supported; test is reserved')
    splits, _ = prepare_rows(table, ALL_FEATURE_NAMES, config['label_version'])
    for split, expected in config['expected_windows'].items():
        if len(splits[split]) != expected:
            raise ValueError('Frozen cohort window count changed: ' + split)
    return splits


def prepare_selected_inputs(data_dir, output, config, workers=4):
    """Build Input C and two feature additions using existing shared algorithms.

    This is deliberately explicit: normal review/prediction needs only the
    downloadable bundle, and does not prepare the full source dataset.
    """
    data_dir, output = Path(data_dir), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new selected-input directory; existing data is preserved')
    splits = read_cohort(data_dir, config)
    summary = json.loads((data_dir / 'preparation_summary.json').read_text())
    records = summary['records']
    eligible = pd.concat([splits['train'], splits['validation']], ignore_index=True)
    baseline = WaveCache(data_dir, eligible)
    output.mkdir(parents=True)
    for split, frame in splits.items():
        allowed = set(frame.source_record_id)
        selected = [r for r in records if r['source_record_id'] in allowed]
        if any(r['split'] != split for r in selected) or {r['source_record_id'] for r in selected} != allowed:
            raise ValueError('Source recording assignments changed')
        prepare_cache(output / split, frame, selected, baseline, workers=workers)
        additions = np.stack([pulse_additions(baseline.ppg[i], np.flatnonzero(baseline.peaks[i]))
                              for i in baseline.rows(frame)])
        np.savez_compressed(output / split / 'feature_additions.npz',
            window_ids=frame.window_id.to_numpy(str), X=additions)
    write_json(output / 'manifest.json', dict(source_table_sha256=config['source_table_sha256'],
        preparation=PREPARATION, feature_names=EXTRA_FEATURES,
        additions_sha256={s:digest(output/s/'feature_additions.npz') for s in splits}, test_accessed=False))


def load_prepared_split(data_dir, inputs_dir, config, split):
    if split not in ('train', 'validation'):
        raise ValueError('Only train or validation may be evaluated')
    frame = read_cohort(data_dir, config)[split].copy()
    inputs_dir = Path(inputs_dir)
    manifest = json.loads((inputs_dir / 'manifest.json').read_text())
    if manifest['source_table_sha256'] != config['source_table_sha256'] or manifest['preparation'] != PREPARATION:
        raise ValueError('Selected inputs do not match the frozen source/preparation')
    directory = inputs_dir / split
    marker = json.loads((directory / 'complete.json').read_text())
    for name, checksum in marker['sha256'].items():
        if digest(directory / name) != checksum:
            raise ValueError('Prepared input changed: ' + name)
    path = directory / 'feature_additions.npz'
    if digest(path) != manifest['additions_sha256'][split]:
        raise ValueError('Feature additions changed')
    with np.load(path, allow_pickle=False) as saved:
        lookup = pd.Series(np.arange(len(saved['window_ids'])), index=saved['window_ids'])
        positions = lookup.loc[frame.window_id].to_numpy()
        for i, name in enumerate(EXTRA_FEATURES):
            frame[name] = saved['X'][positions, i]
    return frame, PreparedCache(directory, None, 'C')
