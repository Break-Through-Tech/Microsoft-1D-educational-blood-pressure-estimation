"""Eval-mode logging must not alter stochastic training or checkpoint selection."""
import unittest
import numpy as np

from models.train_feature_mlp import (TrainConfig, make_demo_data, prepare_rows,
    train_model, predict, regression_metrics, DEMO_LABEL_VERSION)


class TrainingMetricTests(unittest.TestCase):
    def test_logging_preserves_weights_and_reports_checkpoint_training_error(self):
        table,names=make_demo_data()
        splits,_=prepare_rows(table,names,DEMO_LABEL_VERSION)
        cfg=TrainConfig(epochs=3,patience=3,dropout=.1)
        normal=train_model(splits['train'],splits['validation'],names,cfg)
        logged=train_model(splits['train'],splits['validation'],names,cfg,track_training_metrics=True)
        a=predict(normal['model'],splits['validation'],normal['preprocessing'])
        b=predict(logged['model'],splits['validation'],logged['preprocessing'])
        np.testing.assert_array_equal(a,b)
        self.assertEqual(normal['best_epoch'],logged['best_epoch'])
        predictions=predict(logged['model'],splits['train'],logged['preprocessing'])
        scores=regression_metrics(predictions,splits['train'][['sbp','dbp']])
        row=logged['history'].loc[logged['history'].epoch.eq(logged['best_epoch'])].iloc[0]
        for name,value in scores.items():self.assertAlmostEqual(row['training_'+name],value,places=12)


if __name__=='__main__':unittest.main()
