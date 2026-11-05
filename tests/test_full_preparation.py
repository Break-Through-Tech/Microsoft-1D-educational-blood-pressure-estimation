"""Behavior checks for the team-label production path."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from cleaning.clean_data import (ROOT, digest, prepare, read_record, selected_records,
                                 save_npz, verify_review, write_json)
from cleaning.pipeline import (CLEANING_CONFIG, ALL_FEATURE_NAMES, clean_window,
                               label_definition, ppg_features, raw_signal_reason)
from data.bp_utils import complete_beats, extract_bp_labels
from tests.test_bp_utils import make_abp


class FullPreparationTests(unittest.TestCase):
    def test_checkpoint_retries_temporary_windows_sharing_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_replace=Path.replace
            attempts=[]
            def intermittent(path,target):
                attempts.append(path)
                if len(attempts)<3:raise PermissionError('sharing violation')
                return real_replace(path,target)
            with patch.object(Path,'replace',intermittent),patch('cleaning.clean_data.time.sleep'):
                write_json(Path(tmp)/'checkpoint.json',{'complete':True})
            self.assertEqual(len(attempts),3)
            self.assertTrue(json.loads((Path(tmp)/'checkpoint.json').read_text())['complete'])

    def review_fixture(self,root):
        evidence={'accepted':True,'label_configuration':label_definition(),'cleaning':CLEANING_CONFIG,
                  'code_hashes':{f:digest(ROOT/f) for f in ('data/bp_utils.py','cleaning/pipeline.py')}}
        path=root/'review.json';write_json(path,evidence)
        return {'dataset_version':'unit_fixture','label_configuration':label_definition(),'cleaning':CLEANING_CONFIG,
                'review_file':str(path),'review_sha256':digest(path)}

    def test_review_gate_rejects_changed_method_settings_and_evidence(self):
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.review_fixture(root)
            self.assertEqual(verify_review(cfg),CLEANING_CONFIG)
            changed=copy.deepcopy(cfg);changed['label_configuration']['min_complete_beats']=99
            with self.assertRaisesRegex(ValueError,'label method'):verify_review(changed)
            changed=copy.deepcopy(cfg);changed['cleaning']={**CLEANING_CONFIG,'min_std':.01}
            with self.assertRaisesRegex(ValueError,'not accepted'):verify_review(changed)
            (root/'review.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'evidence changed'):verify_review(cfg)

    def test_real_preparation_resume_and_corruption_guard(self):
        import neurokit2 as nk
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=self.review_fixture(root);records=[]
            ppg=nk.ppg_simulate(duration=5,sampling_rate=125,heart_rate=75,random_state=2026)
            with h5py.File(root/'Part_1.mat','w') as f:
                refs=f.create_dataset('Part_1',(3,1),dtype=h5py.ref_dtype)
                for i,split in enumerate(('train','validation','test')):
                    raw=np.column_stack((ppg,make_abp(sbp=120+i),np.zeros(625)))
                    data=f.create_dataset('record'+str(i),data=raw);refs[i,0]=data.ref
                    h=hashlib.sha256();h.update(str(raw.shape).encode());h.update(raw.dtype.str.encode());h.update(raw.tobytes(order='C'))
                    rid=f'Part_1:record_{i}'
                    records.append({'source_record_id':rid,'canonical_record_id':rid,'split':split,
                                    'part_number':1,'record_index':i,'window_count':1,'content_hash':h.hexdigest()})
            with patch('cleaning.clean_data.validate_assignments',return_value={'records':records}):
                prepare(root/'out',cfg,workers=1,raw_dir=root)
                first=pd_read(root/'out/model_table.csv')
                with patch('cleaning.clean_data.process_record',side_effect=AssertionError('Completed records should not be cleaned again')):
                    prepare(root/'out',cfg,workers=1,resume=True,raw_dir=root)
                self.assertEqual(first,pd_read(root/'out/model_table.csv'))
                changed={**cfg,'dataset_version':'different'}
                with self.assertRaisesRegex(ValueError,'mismatch'):prepare(root/'out',changed,workers=1,resume=True,raw_dir=root)
                with (root/'out/chunks/00000.csv').open('a') as f:f.write('\n')
                with self.assertRaisesRegex(ValueError,'Checkpoint hash'):prepare(root/'out',cfg,workers=1,resume=True,raw_dir=root)

    def test_invalid_and_short_labels_fail_without_fallback(self):
        for x in ([],[1,2],np.full(625,np.nan),np.full(625,np.inf),np.ones(625)):
            s,d,n,b=extract_bp_labels(x)
            self.assertTrue(np.isnan(s) and np.isnan(d))
            self.assertEqual((n,b),(0,0))

    def test_complete_pairs_have_matching_lows_and_no_endpoint_peak(self):
        x=make_abp(bump=15)
        peaks,lows=complete_beats(x)
        self.assertTrue(np.all((peaks>0)&(peaks<len(x)-1)))
        self.assertEqual(len(lows),len(peaks)-1)
        for a,b,l in zip(peaks[:-1],peaks[1:],lows):
            self.assertTrue(a<=l<b)
            self.assertEqual(x[l],x[a:b].min())
        s,d,_,_=extract_bp_labels(x)
        self.assertEqual(s,np.median(x[peaks[1:]]))
        self.assertEqual(d,np.median(x[lows]))

    def test_actual_settings_control_quality_checks(self):
        x=np.arange(625,dtype=float)+1;x[:20]=0
        self.assertEqual(raw_signal_reason(x),"")
        self.assertEqual(raw_signal_reason(x,{"max_zero_fraction":.01}),"excess_zeros")
        self.assertEqual(raw_signal_reason(np.arange(10)+1,{"window_samples":10}),"")
        with self.assertRaises(ValueError):raw_signal_reason(x,{"unused_setting":1})

    def test_known_triangular_pulse_morphology(self):
        peaks=np.arange(25,625,100)
        positions=np.arange(625)%100
        x=np.where(positions<=25,positions/25,(100-positions)/75)
        values,n=ppg_features(x,peaks)
        self.assertEqual(len(values),len(ALL_FEATURE_NAMES))
        self.assertGreater(n,1)
        np.testing.assert_allclose(values[6:12],[.2,.6,5,1/.6,.4,.4],atol=1e-8)
        self.assertAlmostEqual(values[12],0)
        shifted,n2=ppg_features(x+10,peaks)
        np.testing.assert_allclose(values[6:],shifted[6:],atol=1e-10)

    def test_missing_pulse_morphology_stays_missing(self):
        values,n=ppg_features(np.arange(625),[100])
        self.assertEqual(n,0)
        self.assertTrue(np.isnan(values[3:]).all())

    def test_one_neurokit_pass_and_shared_label_function(self):
        import neurokit2 as nk
        ppg=nk.ppg_simulate(duration=5,sampling_rate=125,heart_rate=75,random_state=2026)
        with patch.object(nk,"ppg_process",wraps=nk.ppg_process) as process:
            with patch("cleaning.pipeline.bp_utils.extract_bp_labels",wraps=extract_bp_labels) as label:
                result=clean_window(ppg,make_abp())
        self.assertTrue(result['usable'],result['reason'])
        self.assertEqual(process.call_count,1);self.assertEqual(label.call_count,1)
        np.testing.assert_allclose(result['features'][:6],ppg_features(result['ppg'],result['ppg_peaks'])[0][:6],atol=1e-6)

    def test_test_and_duplicate_signals_cannot_be_read(self):
        base={'split':'test','source_record_id':'a','canonical_record_id':'a'}
        with self.assertRaisesRegex(ValueError,'Refusing'):read_record(base)
        with self.assertRaises(ValueError):selected_records({'records':[]},('test',))
        base['split']='excluded';base['canonical_record_id']='b'
        with self.assertRaises(ValueError):read_record(base)

    def test_source_record_bytes_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=np.arange(1875,dtype=float).reshape(625,3)
            with h5py.File(Path(tmp)/'Part_1.mat','w') as f:
                data=f.create_dataset('record',data=x)
                refs=f.create_dataset('Part_1',(1,1),dtype=h5py.ref_dtype);refs[0,0]=data.ref
            h=hashlib.sha256();h.update(str(x.shape).encode());h.update(x.dtype.str.encode());h.update(x.tobytes(order='C'))
            r={'split':'train','source_record_id':'a','canonical_record_id':'a','part_number':1,'record_index':0,'content_hash':h.hexdigest()}
            np.testing.assert_array_equal(read_record(r,tmp),x)
            r['content_hash']='incorrect'
            with self.assertRaisesRegex(ValueError,'hash changed'):read_record(r,tmp)

    def test_archive_streaming_preserves_order_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            x=np.lib.format.open_memmap(Path(tmp)/'x.npy',mode='w+',dtype=np.float32,shape=(5,625))
            x[:]=np.arange(3125).reshape(5,625)
            save_npz(Path(tmp)/'out.npz',{'x':x,'ids':np.array(['a','b','c','d','e'])},{'label_version':'v'})
            with np.load(Path(tmp)/'out.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(x,z['x'])
                self.assertEqual(json.loads(z['metadata_json'].item())['label_version'],'v')
            del x


def pd_read(path):
    return Path(path).read_bytes()


if __name__=='__main__':unittest.main()
