import unittest
import numpy as np
import pandas as pd
import torch

from models.feature_importance import correlation_groups, permutation_importance


class FeatureImportanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(2026)
        self.frame=pd.DataFrame({'signal':rng.normal(size=200),'noise':rng.normal(size=200)})
        self.frame['sbp']=130+10*self.frame['signal'];self.frame['dbp']=70+5*self.frame['signal']
        self.state={'feature_names':['signal','noise'],'feature_medians':[0,0],
                    'feature_mean':[0,0],'feature_scale':[1,1],'target_mean':[130,70],'target_scale':[10,5]}

    def test_importance_detects_model_reliance_without_changing_weights(self):
        model=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():model.weight[:]=torch.tensor([[1.,0],[1.,0]])
        before=model.weight.clone()
        scores=permutation_importance(model,self.state,self.frame,repeats=3)
        self.assertGreater(scores.loc[scores.feature=='signal','sbp_mae_increase'].mean(),5)
        np.testing.assert_allclose(scores.loc[scores.feature=='noise','sbp_mae_increase'],0,atol=1e-7)
        self.assertTrue(torch.equal(before,model.weight))
        pd.testing.assert_frame_equal(scores,permutation_importance(model,self.state,self.frame,repeats=3))

    def test_constant_predictor_has_zero_importance(self):
        model=torch.nn.Linear(2,2,bias=False)
        with torch.no_grad():model.weight.zero_()
        scores=permutation_importance(model,self.state,self.frame,repeats=2)
        np.testing.assert_array_equal(scores[['sbp_mae_increase','dbp_mae_increase']],0)

    def test_correlations_form_joint_groups_from_training_inputs(self):
        self.frame['copy']=-self.frame['signal']
        corr,groups=correlation_groups(self.frame,['signal','noise','copy'])
        self.assertIn(['signal','copy'],groups)
        self.assertIn(['noise'],groups)
        self.assertEqual(corr.loc['signal','copy'],-1)


if __name__=='__main__':unittest.main()
