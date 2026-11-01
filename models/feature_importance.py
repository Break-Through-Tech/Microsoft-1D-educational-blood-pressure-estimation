"""Permutation reliance of frozen MLPs; no retraining or feature pruning."""
import numpy as np
import pandas as pd

from models.train_feature_mlp import predict_array, transform_features


def correlation_groups(training, names, threshold=.9):
    corr=training[names].corr(method="spearman")
    unseen=set(names);groups=[]
    while unseen:
        first=next(n for n in names if n in unseen)
        unseen.remove(first);group=[first];pending=[first]
        while pending:
            parent=pending.pop()
            linked=[n for n in names if n in unseen and np.isfinite(corr.loc[parent,n]) and abs(corr.loc[parent,n])>=threshold]
            for n in linked:unseen.remove(n);group.append(n);pending.append(n)
        groups.append(group)
    return corr,groups


def permutation_importance(model, preprocessing, validation, groups=None, repeats=20, seed=2026):
    """Return MAE increase in mmHg. Same row shuffle within each feature group."""
    names=preprocessing["feature_names"]
    values=transform_features(validation,preprocessing)
    answers=validation[["sbp","dbp"]].to_numpy(float)
    baseline=np.abs(predict_array(model,values,preprocessing,4096)-answers).mean(axis=0)
    groups=groups or [[n] for n in names]
    rng=np.random.default_rng(seed);rows=[]
    for group in groups:
        indices=[names.index(n) for n in group]
        for repeat in range(repeats):
            perm=rng.permutation(len(values));changed=values.copy()
            changed[:,indices]=values[perm][:,indices]
            error=np.abs(predict_array(model,changed,preprocessing,4096)-answers).mean(axis=0)-baseline
            rows.append({"feature":" + ".join(group),"repeat":repeat,
                         "sbp_mae_increase":error[0],"dbp_mae_increase":error[1]})
    return pd.DataFrame(rows)


def plot_importance(frame,destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    summary=frame.groupby("feature")[["sbp_mae_increase","dbp_mae_increase"]].agg(["mean","std"])
    order=summary[("sbp_mae_increase","mean")].sort_values().index
    summary=summary.loc[order]
    fig,axes=plt.subplots(1,2,figsize=(15,max(4,len(order)*.4)),sharey=True)
    for ax,target in zip(axes,("sbp","dbp")):
        c=target+"_mae_increase"
        ax.barh(np.arange(len(order)),summary[(c,"mean")],xerr=summary[(c,"std")].fillna(0),capsize=2)
        ax.axvline(0,color="black",lw=.7);ax.set(title=target.upper(),xlabel="Validation MAE increase (mmHg)")
    axes[0].set_yticks(np.arange(len(order)),order)
    fig.suptitle("Permutation reliance; bars show mean, whiskers shuffle/seed variation")
    fig.tight_layout();fig.savefig(destination,dpi=150);plt.close(fig)
    return summary
