"""Small synthetic inference and mocked fixed-fit wiring; no source/test dataset."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from models.cnn_model import build_wave_model
from models.comparison_training import write_json
from models.selected_inputs import input_c, prepare_prediction_inputs
from models.selected_models import (DEFAULT_CONFIG, component_directory, evaluate_selected,
    load_selected_model, seal_bundle, train_selected)


class SelectedModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def bundle(self):
        parameters = dict(representation='derivatives', normalization='global', channels=[4,4,4], batch_size=2)
        config = dict(seeds=[2026], components={'cnn_c':dict(family='cnn',parameters=parameters,durations={'2026':[1]})},
                      models={'recommended':{'cnn_c':1.0}})
        (self.root/'config.json').write_text(json.dumps(config))
        directory = component_directory(self.root, 'cnn_c', 2026)
        directory.mkdir(parents=True)
        model = build_wave_model(parameters)
        state = dict(normalization='global',channel_mean=[0,0,0],channel_scale=[1,1,1],
                     target_mean=[130,70],target_scale=[1,1])
        torch.save(dict(parameters=parameters,state_dict=model.state_dict(),preprocessing=state,seed=2026,duration=[1]),directory/'model.pt')
        seal_bundle(self.root)

    def test_raw_inference_keeps_ids_and_reasons_and_reuses_loaded_weights(self):
        self.bundle()
        t = np.arange(625)/125
        original = np.stack([np.sin(t*7),np.zeros(625),np.cos(t*7)])
        with patch('models.selected_models.load_wave_model', wraps=__import__('models.train_waveform_comparison',fromlist=['load_wave_model']).load_wave_model) as loader:
            model = load_selected_model(artifact_dir=self.root)
            first = model.predict_ppg(original, ['a','invalid','b'])
            second = model.predict_ppg(original, ['a','invalid','b'])
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(first.window_id.tolist(), ['a','invalid','b'])
        self.assertTrue(first.loc[1,'rejection_reason'])
        self.assertTrue(np.isnan(first.loc[1,'predicted_sbp']))
        self.assertTrue(np.isfinite(first.loc[[0,2],['predicted_sbp','predicted_dbp']]).all().all())
        pd.testing.assert_frame_equal(first,second)
        with self.assertRaisesRegex(ValueError,'unique'):
            model.predict_ppg(original[:2],['a','a'])

    def test_artifact_tampering_is_rejected_before_loading(self):
        self.bundle()
        path = component_directory(self.root,'cnn_c',2026)/'model.pt'
        path.write_bytes(b'changed')
        with patch('models.selected_models.load_wave_model',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'changed model artifact'):
                load_selected_model(artifact_dir=self.root)

    def test_loader_applies_recorded_precision_settings(self):
        self.bundle()
        with patch('models.selected_models.seed_training') as configure:
            load_selected_model(artifact_dir=self.root)
        configure.assert_called_once_with(2026,'cpu')

    def test_empty_bad_shape_and_nonfinite_rows(self):
        self.bundle()
        model=load_selected_model(artifact_dir=self.root)
        self.assertTrue(model.predict_ppg(np.empty((0,625)),[]).empty)
        with self.assertRaisesRegex(ValueError,'N x 625'):
            model.predict_ppg(np.ones((1,624)),['a'])
        bad=np.ones((1,625));bad[0,20]=np.nan
        result=model.predict_ppg(bad,['a'])
        self.assertEqual(result.rejection_reason.iloc[0],'ppg:nonfinite_signal')

    def test_feature_preparation_preserves_training_precision_paths(self):
        from cleaning.pipeline import ppg_features
        from models.training_support import pulse_additions
        t=np.arange(625)/125
        clean=0.7+np.sin(t*7)+0.123456789*np.sin(t*14)
        peaks=np.array([28,140,252,364,476,588])
        signals=pd.DataFrame({'PPG_Clean':clean,'PPG_Peaks':np.isin(np.arange(625),peaks).astype(int)})
        with patch('neurokit2.ppg_process',return_value=(signals,{})):
            positions,wave,features,reasons=prepare_prediction_inputs(clean[None,:])
        np.testing.assert_array_equal(positions,[0])
        expected,_=ppg_features(clean,peaks)
        np.testing.assert_allclose(features.iloc[0,:14],expected,rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(features.iloc[0,14:],pulse_additions(clean.astype(np.float32),peaks),rtol=0,atol=0,equal_nan=True)
        np.testing.assert_array_equal(wave,input_c(clean[None,:]))
        self.assertEqual(reasons,[''])

    def test_frozen_training_passes_fixed_durations_and_never_validation(self):
        config=json.loads(DEFAULT_CONFIG.read_text())
        train=pd.DataFrame({'split':['train']})
        with patch('models.selected_models.load_prepared_split',return_value=(train,object())) as source, \
             patch('models.selected_models.fit_wave') as wave, patch('models.selected_models.fit_and_save') as tree:
            train_selected(self.root,self.root,self.root/'new','recommended')
        self.assertEqual(source.call_args.args[-1],'train')
        self.assertEqual(wave.call_count,2)
        for call in wave.call_args_list:
            self.assertIsNone(call.args[1])
            name=call.args[6].parent.name
            self.assertEqual(call.kwargs['duration'],config['components'][name]['durations']['2026'][0])
        self.assertEqual(tree.call_args.kwargs['duration'],config['components']['lightgbm_16']['durations']['2026'])
        self.assertNotIn('holdout',tree.call_args.kwargs)
        with self.assertRaises(FileExistsError):
            train_selected(self.root,self.root,self.root/'new')

    def test_test_evaluation_rejected_before_any_file_is_opened(self):
        with patch('models.selected_models.load_prepared_split',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'Test evaluation'):
                evaluate_selected(self.root,self.root,self.root,self.root/'evaluation',splits=('test',))
        self.assertFalse((self.root/'evaluation').exists())

    def test_atomic_json_retries_transient_windows_lock(self):
        target=self.root/'status.json';target.write_text('{"old": true}')
        original=Path.replace
        busy=PermissionError('temporary Windows lock');busy.winerror=32
        attempts=[]
        def replace(path,destination):
            attempts.append(path)
            if len(attempts)==1:raise busy
            return original(path,destination)
        with patch.object(Path,'replace',replace),patch('models.comparison_training.time.sleep'):
            write_json(target,{'complete':True})
        self.assertEqual(len(attempts),2)
        self.assertEqual(json.loads(target.read_text()),{'complete':True})

    def test_atomic_json_persistent_denial_preserves_previous_status(self):
        target=self.root/'status.json';target.write_text('{"old": true}')
        busy=PermissionError('Windows access denied');busy.winerror=5
        with patch.object(Path,'replace',side_effect=busy) as replace,patch('models.comparison_training.time.sleep'):
            with self.assertRaises(PermissionError):write_json(target,{'complete':True})
        self.assertEqual(replace.call_count,3)
        self.assertEqual(json.loads(target.read_text()),{'old':True})


if __name__=='__main__':
    unittest.main()
