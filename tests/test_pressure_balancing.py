"""Weighting mathematics, row alignment, diagnostics and unchanged controls."""
import unittest
import numpy as np
import pandas as pd
import torch

from models.pressure_metrics import pressure_weights, pressure_groups, pressure_errors
from models.train_feature_mlp import (TrainConfig, make_demo_data, prepare_rows, DEMO_LABEL_VERSION,
    fit_preprocessing, make_loader, train_model, predict, train_epoch)


class PressureBalancingTests(unittest.TestCase):
    def setUp(self):
        table,self.names=make_demo_data()
        self.splits,_=prepare_rows(table,self.names,DEMO_LABEL_VERSION)

    def test_frequency_formula_cap_normalization_and_target_specificity(self):
        t=self.splits['train'].copy()
        t['sbp']=130.;t['dbp']=70.
        t.loc[:3,'sbp']=190.;t.loc[:7,'dbp']=100.
        w,bins,summary=pressure_weights(t)
        np.testing.assert_allclose(w.mean(axis=0),1.,atol=1e-14)
        np.testing.assert_allclose(w[0]/w[20],[3.,3.])
        self.assertGreater(w[5,1],w[20,1]);self.assertEqual(w[5,0],w[20,0])
        self.assertLessEqual(w.max(),3.)
        self.assertTrue((bins.loc[bins.windows.eq(0),'weight']==0).all())
        self.assertEqual(summary[0]['windows'],len(t))
        with self.assertRaisesRegex(ValueError,'training rows'):
            pressure_weights(self.splits['validation'])
        with self.assertRaisesRegex(ValueError,'training rows'):
            pressure_weights(pd.concat(self.splits.values()))

    def test_boundaries_and_empty_groups_have_explicit_counts(self):
        t=pd.DataFrame({'sbp':[90.,120.,140.,180.,181.],'dbp':[60.,80.,90.,120.,121.],
                        'split':'train','source_record_id':['a','a','b','c','d']})
        w,bins,_=pressure_weights(t)
        np.testing.assert_array_equal(w,np.ones((5,2)))
        self.assertTrue(bins.windows.eq(1).all())
        rows=pressure_errors(t[['sbp','dbp']].to_numpy()+[2.,-3.],t[['sbp','dbp']],pressure_groups(t))
        r=next(r for r in rows if r['target']=='dbp' and r['range']=='>90')
        self.assertEqual(r['windows'],2);self.assertEqual(r['mae'],3.);self.assertEqual(r['bias'],-3.)
        central=t.iloc[[2]].copy();rows=pressure_errors(central[['sbp','dbp']],central[['sbp','dbp']],pressure_groups(central))
        self.assertTrue(any(r['windows']==0 and np.isnan(r['mae']) for r in rows))
        with self.assertRaises(ValueError):pressure_errors(np.zeros((1,2)),np.zeros((5,2)),pressure_groups(t))

    def test_loader_shuffles_features_targets_and_weights_together(self):
        t=self.splits['train'];state=fit_preprocessing(t,self.names);cfg=TrainConfig(batch_size=31)
        w=np.column_stack([np.arange(len(t))+1,np.arange(len(t))+2.])
        loader=make_loader(t,state,cfg,w)
        source=loader.dataset.tensors[0].numpy()
        for x,y,weight in loader:
            for values,answer,actual in zip(x.numpy(),y.numpy(),weight.numpy()):
                i=np.flatnonzero(np.all(source==values,axis=1))[0]
                np.testing.assert_array_equal(actual,w[i])
                np.testing.assert_array_equal(answer,loader.dataset.tensors[1][i])
        for invalid in (w[:-1],w*0,np.full_like(w,np.nan)):
            with self.assertRaisesRegex(ValueError,'target_weights'):make_loader(t,state,cfg,invalid)

    def test_ones_and_diagnostic_cadence_preserve_predictions_and_selection(self):
        t,v=self.splits['train'],self.splits['validation']
        cfg=TrainConfig(epochs=7,patience=10,dropout=.1,threads=1)
        plain=train_model(t,v,self.names,cfg)
        logged=train_model(t,v,self.names,cfg,target_weights=np.ones((len(t),2)),
            track_training_metrics=True,training_metrics_interval=5,track_pressure_metrics=True)
        np.testing.assert_array_equal(predict(plain['model'],v,plain['preprocessing']),predict(logged['model'],v,logged['preprocessing']))
        self.assertEqual(plain['best_epoch'],logged['best_epoch'])
        h=logged['history'];expected={1,5,7,logged['best_epoch']}
        self.assertEqual(set(h.loc[h.training_sbp_mae.notna(),'epoch']),expected)
        ph=logged['pressure_history']
        self.assertEqual(set(ph.loc[ph.split.eq('validation'),'epoch']),set(range(1,8)))
        self.assertEqual(set(ph.loc[ph.split.eq('train'),'epoch']),expected)
        self.assertTrue(h.train_mse_standardized.equals(h.train_objective_standardized))
        # Eval logging must not shift features/target statistics, even with nonuniform weights.
        weights,_,_=pressure_weights(t)
        weighted=train_model(t,v,self.names,TrainConfig(epochs=1,threads=1),target_weights=weights)
        self.assertEqual(weighted['preprocessing'],plain['preprocessing'])

    def test_final_and_restored_best_diagnostics_on_early_stop(self):
        t,v=self.splits['train'],self.splits['validation']
        result=train_model(t,v,self.names,TrainConfig(epochs=20,patience=2,learning_rate=1e-40,dropout=0,threads=1),
            track_training_metrics=True,training_metrics_interval=5,track_pressure_metrics=True)
        self.assertEqual(result['best_epoch'],1)
        self.assertEqual(result['history'].epoch.max(),3)
        self.assertEqual(set(result['history'].loc[result['history'].training_sbp_mae.notna(),'epoch']),{1,3})
        self.assertTrue((result['history'].loc[result['history'].epoch.eq(3),'training_diagnostics_stage']=='final').all())

    def test_nonuniform_weighted_logging_preserves_learning_and_best_prediction_cache(self):
        t,v=self.splits['train'],self.splits['validation'];w,_,_=pressure_weights(t)
        self.assertFalse(np.all(w==1))
        cfg=TrainConfig(epochs=7,patience=10,dropout=.1,threads=1)
        plain=train_model(t,v,self.names,cfg,target_weights=w)
        logged=train_model(t,v,self.names,cfg,target_weights=w,track_training_metrics=True,
            training_metrics_interval=5,track_pressure_metrics=True)
        self.assertEqual(plain['best_epoch'],logged['best_epoch'])
        np.testing.assert_array_equal(predict(plain['model'],v,plain['preprocessing']),predict(logged['model'],v,logged['preprocessing']))
        np.testing.assert_array_equal(logged['training_predictions'],predict(logged['model'],t,logged['preprocessing']))

    def test_weighted_gradient_uses_global_weights_without_batch_renormalization(self):
        x=torch.tensor([[1.,2.],[3.,4.]])
        y=torch.tensor([[10.,5.],[20.,9.]])
        w=torch.tensor([[3.,1.],[1.,2.]])
        model=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():model.weight.zero_()
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y,w),batch_size=2)
        expected=-(2/4)*(w*y).T@x
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        losses=train_epoch(model,loader,optimizer,torch.nn.MSELoss(),return_metrics=True)
        np.testing.assert_allclose(model.weight.detach().numpy(),(-.01*expected).numpy(),atol=1e-7)
        self.assertAlmostEqual(losses['objective'],float((w*y.square()).mean()))
        self.assertAlmostEqual(losses['mse'],float(y.square().mean()))
        self.assertNotEqual(losses['objective'],float((w*y.square()).sum()/w.sum()))


if __name__=='__main__':unittest.main()
