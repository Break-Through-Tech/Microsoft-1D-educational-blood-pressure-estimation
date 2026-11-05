"""Fixed five-bin evaluation; auxiliary classification is not a diagnosis."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from models.pressure_metrics import TARGETS,RANGES
from models.wave_objectives import class_targets


def classification_summary(answers,continuous,probabilities=None):
    truth=class_targets(answers)
    predictions={'regression_bins':class_targets(continuous)}
    if probabilities is not None:
        p=np.asarray(probabilities)
        if p.shape!=(len(truth),2,5) or not np.isfinite(p).all() or (p<0).any():
            raise ValueError('Expected finite N x 2 x 5 probabilities')
        np.testing.assert_allclose(p.sum(-1),1.,rtol=0,atol=1e-6)
        predictions['auxiliary_head']=p.argmax(-1)
    rows=[];matrices=[]
    for method,predicted in predictions.items():
        for j,target in enumerate(TARGETS):
            matrix=confusion_matrix(truth[:,j],predicted[:,j],labels=np.arange(5))
            counts=matrix.sum(1);recall=np.divide(matrix.diagonal(),counts,out=np.zeros(5,dtype=float),where=counts>0)
            rows.append(dict(method=method,target=target,macro_f1=float(f1_score(truth[:,j],predicted[:,j],
                labels=np.arange(5),average='macro',zero_division=0)),
                balanced_accuracy=float(recall[counts>0].mean()),windows=len(truth),
                supported_classes=int((counts>0).sum())))
            for a in range(5):
                for b in range(5):matrices.append(dict(method=method,target=target,
                    reference_class=a,predicted_class=b,reference_range=RANGES[target][a],
                    predicted_range=RANGES[target][b],count=int(matrix[a,b]),support=int(counts[a]),
                    recall=float(recall[a]),row_fraction=float(matrix[a,b]/counts[a]) if counts[a] else 0.))
    return pd.DataFrame(rows),pd.DataFrame(matrices)
