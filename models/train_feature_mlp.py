"""Run the issue #29 feature network on artificial data or a teammate's CSV.

Demo: python -m models.train_feature_mlp --demo
Real: python -m models.train_feature_mlp --data TABLE.csv --feature-config CONFIG.json

Only training and validation rows are modeled. Test values are never used for
preprocessing, tuning, or scoring. Each invocation starts a fresh model.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.feature_mlp import FeatureMLP


ISSUE_URL = "https://github.com/Break-Through-Tech/Microsoft-1D-educational-blood-pressure-estimation/issues/29"
TARGETS = ["sbp", "dbp"]
IDENTITY_COLUMNS = ["window_id", "source_record_id", "split", "usable", "label_version"]
RESERVED_COLUMNS = set(IDENTITY_COLUMNS + TARGETS + ["dataset_version"])
DEMO_LABEL_VERSION = "artificial_known_relationship_v1"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    patience: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 256
    dropout: float = 0.1
    seed: int = 2026
    threads: int = 2
    hidden_sizes: tuple[int, ...] = (32, 16)
    device: str = "cpu"

    def validate(self):
        for name in ("epochs", "patience", "batch_size", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if not isinstance(self.hidden_sizes, (tuple, list)) or not self.hidden_sizes or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers")


def check_feature_names(feature_names):
    """Use an explicit list, with guards against common target/metadata leakage."""
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("feature_names must be a nonempty JSON list")
    if any(not isinstance(name, str) or not name.strip() for name in feature_names):
        raise ValueError("Every feature name must be a nonempty string")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names contains duplicates")
    forbidden = ("abp", "ecg", "sbp", "dbp", "systolic", "diastolic", "pressure",
                 "target", "quality", "status", "error", "_qc")
    for name in feature_names:
        if name.lower() in RESERVED_COLUMNS or any(token in name.lower() for token in forbidden):
            raise ValueError(f"Target, identity, or diagnostic column cannot be an input: {name}")


def _usable_values(series):
    # Do not use astype(bool): the string "False" would become True.
    normalized = series.astype("string").str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError("usable must contain only true/false or 1/0, without missing values")
    return normalized.isin(["true", "1"]).to_numpy(dtype=bool)


def prepare_rows(table, feature_names, label_version):
    """Check IDs once, then select eligible train/validation rows without resplitting.

    The returned row log includes excluded development rows and exact reasons.
    Test identity/split metadata is checked for overlap, but its values are not
    converted, inspected for quality, fitted, or scored.
    """
    check_feature_names(feature_names)
    if not table.columns.is_unique:
        raise ValueError("Table contains duplicate column names")
    required = IDENTITY_COLUMNS + TARGETS + feature_names
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(label_version, str) or not label_version.strip():
        raise ValueError("Specify the agreed label_version")
    for name in ("window_id", "source_record_id", "split"):
        values = table[name].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"{name} contains missing or blank values")
        if not values.eq(values.str.strip()).all():
            raise ValueError(f"{name} contains surrounding whitespace")
    if table["window_id"].duplicated().any():
        raise ValueError("Duplicate window_id: inputs and answers must match one-to-one")
    if not table["split"].isin(["train", "validation", "test", "excluded"]).all():
        raise ValueError("split must be train, validation, test, or excluded")
    retained = table[table["split"] != "excluded"]
    if retained.groupby("source_record_id")["split"].nunique().gt(1).any():
        raise ValueError("A recording appears in more than one split")

    splits, logs = {}, []
    for split in ("train", "validation"):
        frame = table.loc[table["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"No {split} rows in the supplied table")
        usable = _usable_values(frame["usable"])
        frame[TARGETS + feature_names] = frame[TARGETS + feature_names].apply(pd.to_numeric, errors="raise")
        labels = frame[TARGETS].to_numpy(dtype=float, na_value=np.nan)
        finite_labels = np.isfinite(labels).all(axis=1)
        ordered = labels[:, 0] > labels[:, 1]
        label_matches = frame["label_version"].eq(label_version).fillna(False).to_numpy(dtype=bool)
        if (usable & ~label_matches).any():
            raise ValueError(f"Usable {split} rows do not match label_version={label_version!r}")
        keep = usable & finite_labels & ordered
        reasons = [";".join(reason for flag, reason in (
            (not usable[i], "marked_unusable"),
            (not finite_labels[i], "nonfinite_label"),
            (finite_labels[i] and not ordered[i], "sbp_not_above_dbp"),
        ) if flag) for i in range(len(frame))]
        log = frame[["window_id", "source_record_id", "split"]].copy()
        log["included"] = keep
        log["exclusion_reason"] = reasons
        logs.append(log)
        selected = frame.loc[keep].reset_index(drop=True)
        if selected.empty:
            raise ValueError(f"No usable {split} rows remain")
        inputs = selected[feature_names].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(inputs).any():
            raise ValueError(f"Infinite {split} feature value; fix the feature export rather than filling infinity")
        if np.isnan(inputs).all(axis=1).any():
            raise ValueError(f"A usable {split} window has no features; mark failed extraction unusable")
        splits[split] = selected
    return splits, pd.concat(logs, ignore_index=True)


def fit_preprocessing(train, feature_names):
    """Calculate replacement/scaling values ONLY from eligible training examples."""
    values = train[feature_names].to_numpy(dtype=float, na_value=np.nan)
    empty = np.isnan(values).all(axis=0)
    if empty.any():
        raise ValueError(f"Features have no usable training values: {np.asarray(feature_names)[empty].tolist()}")
    imputer = SimpleImputer(strategy="median")
    filled = imputer.fit_transform(values)
    scaler = StandardScaler().fit(filled)
    targets = StandardScaler().fit(train[TARGETS].to_numpy(dtype=float))
    # Plain lists keep the saved checkpoint independent of pickle-only sklearn objects.
    return {
        "feature_names": list(feature_names),
        "feature_medians": imputer.statistics_.tolist(),
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "target_mean": targets.mean_.tolist(),
        "target_scale": targets.scale_.tolist(),
    }


def transform_features(frame, preprocessing):
    """Apply saved training statistics; inference needs PPG features, not ABP."""
    names = preprocessing["feature_names"]
    values = frame[names].to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any() or np.isnan(values).all(axis=1).any():
        raise ValueError("Cannot predict from infinite or entirely missing feature rows")
    values = np.where(np.isnan(values), preprocessing["feature_medians"], values)
    scaled = (values - preprocessing["feature_mean"]) / preprocessing["feature_scale"]
    result = np.asarray(scaled, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Feature transformation produced nonfinite float32 values")
    return result


class ResidentBatches:
    """GPU-resident tensors, using the historical CPU DataLoader's batch order."""

    def __init__(self, tensors, config):
        self.dataset = TensorDataset(*(t.to(config.device) for t in tensors))
        self.batch_size = config.batch_size
        self.order_loader = DataLoader(torch.arange(len(tensors[0])),
            batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(config.seed), num_workers=0)

    def __iter__(self):
        order = torch.cat(list(self.order_loader)).to(self.dataset.tensors[0].device)
        for indices in order.split(self.batch_size):
            yield tuple(t[indices] for t in self.dataset.tensors)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def make_loader(frame, preprocessing, config, target_weights=None):
    inputs = torch.from_numpy(transform_features(frame, preprocessing))
    answers = (frame[TARGETS].to_numpy(dtype=float) - preprocessing["target_mean"]) / preprocessing["target_scale"]
    answers = np.asarray(answers, dtype=np.float32)
    if not np.isfinite(answers).all():
        raise ValueError("Target transformation produced nonfinite float32 values")
    tensors=[inputs,torch.from_numpy(answers)]
    if target_weights is not None:
        weights=np.asarray(target_weights,dtype=np.float32)
        if weights.shape!=answers.shape or not np.isfinite(weights).all() or (weights<=0).any():
            raise ValueError('target_weights must be positive finite N×2 values in training row order')
        tensors.append(torch.from_numpy(weights))
    if config.device == "cuda":
        return ResidentBatches(tensors, config)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=0,
    )


def train_epoch(model, loader, optimizer, loss_fn, *, return_metrics=False, objective_fn=None):
    """Learning checkpoint: predict -> error -> clear -> backward -> update."""
    model.train()  # Enables dropout during training.
    gpu = next(model.parameters()).is_cuda
    total_loss, total_mse = 0.0, 0.0
    accumulated = torch.zeros(2, device="cuda") if gpu else None
    extra_totals={}
    for inputs, answers, *weights in loader:
        if objective_fn is None:
            predictions = model(inputs)
            mse = nn.functional.mse_loss(predictions,answers)
            ordinary_loss=loss_fn(predictions,answers)
            loss = (weights[0]*(predictions-answers).square()).mean() if weights and not torch.all(weights[0]==1) else ordinary_loss
        else:
            loss,mse,components=objective_fn(model,(inputs,answers,*weights))
            for key,value in components.items():
                extra_totals[key]=extra_totals.get(key,0)+value.detach()*len(inputs)
        if not gpu and not torch.isfinite(loss):
            raise ValueError("Training loss became nonfinite")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if gpu:
            accumulated += torch.stack((loss.detach(), mse.detach())) * len(inputs)
        else:
            total_loss += loss.item() * len(inputs)
            total_mse += mse.item() * len(inputs)
    if gpu:
        total_loss, total_mse = accumulated.cpu().tolist()
        if not np.isfinite([total_loss, total_mse]).all():
            raise ValueError("Training loss became nonfinite")
    metrics={'objective':total_loss/len(loader.dataset),'mse':total_mse/len(loader.dataset)}
    metrics.update({k:float(v.cpu())/len(loader.dataset) for k,v in extra_totals.items()})
    return metrics if return_metrics else metrics['objective']


def predict(model, frame, preprocessing, batch_size=256):
    """Validation/prediction updates no weights. Return SBP/DBP in mmHg."""
    if batch_size < 1 or frame.empty:
        raise ValueError("Prediction needs a positive batch size and at least one row")
    return predict_array(model, transform_features(frame, preprocessing), preprocessing, batch_size)


def predict_array(model, values, preprocessing, batch_size=256):
    """Predict cached standardized features, also used by permutation inspection."""
    if batch_size < 1 or len(values) == 0:
        raise ValueError("Prediction needs nonempty inputs and positive batch size")
    model.eval()
    device = next(model.parameters()).device
    if isinstance(values, torch.Tensor):
        inputs = values.to(device=device, dtype=torch.float32)
    else:
        inputs = torch.from_numpy(np.ascontiguousarray(values)).to(device)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            predictions.append(model(inputs[start:start + batch_size]))
    result = torch.cat(predictions).cpu().numpy() * preprocessing["target_scale"] + preprocessing["target_mean"]
    if not np.isfinite(result).all():
        raise ValueError("Prediction became nonfinite")
    return result


def regression_metrics(predictions, answers):
    error = np.asarray(predictions, dtype=float) - np.asarray(answers, dtype=float)
    return {
        **{f"{target}_mae": float(np.abs(error[:, i]).mean()) for i, target in enumerate(TARGETS)},
        **{f"{target}_rmse": float(np.sqrt(np.mean(error[:, i] ** 2))) for i, target in enumerate(TARGETS)},
    }


def initialize_training(train, feature_names, config):
    """Shared initialization for early-stopped fits and validation-free refits."""
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    preprocessing = fit_preprocessing(train, feature_names)
    model = FeatureMLP(len(feature_names), config.dropout, config.hidden_sizes).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return model, optimizer, preprocessing


def train_model(train, validation, feature_names, config=TrainConfig(), *, progress=False,
                track_training_metrics=False, target_weights=None, training_metrics_interval=1,
                track_pressure_metrics=False, epoch_callback=None, resume_state=None):
    """Fit a fresh network and restore its best validation-MAE checkpoint."""
    config.validate()
    if not isinstance(training_metrics_interval,int) or isinstance(training_metrics_interval,bool) or training_metrics_interval<1:
        raise ValueError('training_metrics_interval must be a positive integer')
    from models.pressure_metrics import pressure_groups, pressure_errors
    started=time.perf_counter()
    model, optimizer, preprocessing = initialize_training(train, feature_names, config)
    loader = make_loader(train, preprocessing, config, target_weights)
    history, best_state, best_score, best_epoch, stale = [], None, float("inf"), 0, 0
    answers = validation[TARGETS].to_numpy(dtype=float)
    validation_inputs = torch.from_numpy(transform_features(validation, preprocessing)).to(config.device)
    training_inputs = loader.dataset.tensors[0] if track_training_metrics else None
    training_answers = train[TARGETS].to_numpy(dtype=float) if track_training_metrics else None
    groups={s:pressure_groups(f) for s,f in (('train',train),('validation',validation))} if track_pressure_metrics else {}
    pressure_history=[]
    timing={'optimization_seconds':0.,'validation_seconds':0.,'training_diagnostics_seconds':0.}
    last_training_epoch=0
    training_cache={}
    best_training_predictions=None

    def training_diagnostics(epoch, stage='epoch'):
        nonlocal last_training_epoch, best_training_predictions
        tic=time.perf_counter()
        y=predict_array(model,training_inputs,preprocessing,config.batch_size)
        row=history[epoch-1]
        row.update({'training_'+k:v for k,v in regression_metrics(y,training_answers).items()})
        standardized=np.square((y-training_answers)/preprocessing['target_scale'])
        row['training_eval_mse_standardized']=float(standardized.mean())
        row['training_eval_objective_standardized']=float((standardized*target_weights).mean()) if target_weights is not None else float(standardized.mean())
        row['training_diagnostics_stage']=stage
        if track_pressure_metrics:
            pressure_history.extend({'epoch':epoch,'split':'train','stage':stage,**r} for r in pressure_errors(y,training_answers,groups['train']))
        last_training_epoch=epoch
        training_cache.update(epoch=epoch,predictions=y)
        if epoch==best_epoch:best_training_predictions=y
        timing['training_diagnostics_seconds']+=time.perf_counter()-tic

    first_epoch = 1
    if resume_state is not None:
        model.load_state_dict(resume_state['model'])
        optimizer.load_state_dict(resume_state['optimizer'])
        best_state, best_score, best_epoch, stale = (resume_state[k] for k in ('best_state','best_score','best_epoch','stale'))
        history, pressure_history = resume_state['history'], resume_state['pressure_history']
        first_epoch = resume_state['epoch'] + 1
        random.setstate(resume_state['python_rng']); np.random.set_state(resume_state['numpy_rng'])
        torch.set_rng_state(resume_state['torch_rng'])
        if config.device == 'cuda': torch.cuda.set_rng_state_all(resume_state['cuda_rng'])
        generator = loader.order_loader.generator if isinstance(loader, ResidentBatches) else loader.generator
        generator.set_state(resume_state['order_rng'])
    epoch = first_epoch - 1
    for next_epoch in range(first_epoch, config.epochs + 1):
        if stale >= config.patience:
            break
        epoch = next_epoch
        tic=time.perf_counter()
        train_losses = train_epoch(model, loader, optimizer, nn.MSELoss(),return_metrics=True)
        timing['optimization_seconds']+=time.perf_counter()-tic
        tic=time.perf_counter()
        predictions = predict_array(model, validation_inputs, preprocessing, config.batch_size)
        scores = regression_metrics(predictions, answers)
        validation_loss = float(np.mean(((predictions - answers) / preprocessing["target_scale"]) ** 2))
        score = (scores["sbp_mae"] + scores["dbp_mae"]) / 2
        history.append({"epoch": epoch, "train_mse_standardized": train_losses['mse'],
                        'train_objective_standardized':train_losses['objective'],
                        "validation_mse_standardized": validation_loss,
                        "validation_mean_mae": score, **scores})
        history[-1].update({t+'_bias':float((predictions-answers)[:,j].mean()) for j,t in enumerate(TARGETS)})
        if track_pressure_metrics:
            pressure_history.extend({'epoch':epoch,'split':'validation','stage':'epoch',**r} for r in pressure_errors(predictions,answers,groups['validation']))
        timing['validation_seconds']+=time.perf_counter()-tic
        if track_training_metrics and (epoch==1 or epoch%training_metrics_interval==0):
            training_diagnostics(epoch)
        if progress and (epoch == 1 or epoch % 5 == 0):
            print(f"epoch {epoch}: SBP MAE {scores['sbp_mae']:.3f}; DBP MAE {scores['dbp_mae']:.3f}", flush=True)
        if score < best_score:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), score, epoch, 0
            best_training_predictions=training_cache.get('predictions') if training_cache.get('epoch')==epoch else None
        else:
            stale += 1
        if epoch_callback is not None:
            epoch_callback(dict(epoch=epoch, model=model, optimizer=optimizer, loader=loader,
                best_state=best_state, best_score=best_score, best_epoch=best_epoch, stale=stale,
                history=history, pressure_history=pressure_history, preprocessing=preprocessing))
        if stale >= config.patience:
            break
    if track_training_metrics and last_training_epoch!=epoch:
        training_diagnostics(epoch,'final')
    model.load_state_dict(best_state)
    model.eval()
    if track_training_metrics and best_training_predictions is None:
        training_diagnostics(best_epoch,'restored_best')
    timing['total_training_seconds']=time.perf_counter()-started
    return {"model": model, "preprocessing": preprocessing, "config": asdict(config),
            "history": pd.DataFrame(history), "best_epoch": best_epoch,
            'pressure_history':pd.DataFrame(pressure_history),'timing':timing,
            'training_predictions':best_training_predictions}


def fit_fixed_epochs(train, feature_names, config=TrainConfig(), *, training_metrics_interval=5, progress=False,
                     epoch_callback=None):
    """Refit for a previously selected duration. No validation argument or access."""
    if training_metrics_interval < 1:
        raise ValueError("training_metrics_interval must be positive")
    from models.pressure_metrics import pressure_groups, pressure_errors
    started = time.perf_counter()
    model, optimizer, preprocessing = initialize_training(train, feature_names, config)
    loader = make_loader(train, preprocessing, config)
    answers = train[TARGETS].to_numpy(float)
    groups = pressure_groups(train)
    history, pressure_history, predictions = [], [], None
    optimization, diagnostics = 0., 0.
    for epoch in range(1, config.epochs + 1):
        tic = time.perf_counter()
        losses = train_epoch(model, loader, optimizer, nn.MSELoss(), return_metrics=True)
        optimization += time.perf_counter() - tic
        row = {"epoch": epoch, "train_mse_standardized": losses["mse"],
               "train_objective_standardized": losses["objective"]}
        if epoch == 1 or epoch % training_metrics_interval == 0 or epoch == config.epochs:
            tic = time.perf_counter()
            predictions = predict_array(model, loader.dataset.tensors[0], preprocessing, config.batch_size)
            row.update({"training_" + k: v for k, v in regression_metrics(predictions, answers).items()})
            pressure_history.extend({"epoch": epoch, "split": "train", **r}
                                    for r in pressure_errors(predictions, answers, groups))
            diagnostics += time.perf_counter() - tic
        history.append(row)
        if epoch_callback is not None:
            epoch_callback(dict(epoch=epoch, model=model, optimizer=optimizer, loader=loader,
                best_state=model.state_dict(), best_score=0., best_epoch=epoch, stale=0,
                history=history, pressure_history=pressure_history, preprocessing=preprocessing))
        if progress and (epoch == 1 or epoch % 10 == 0 or epoch == config.epochs):
            print(f"Fixed refit epoch {epoch}/{config.epochs}", flush=True)
    model.eval()
    return {"model": model, "preprocessing": preprocessing, "config": asdict(config),
            "history": pd.DataFrame(history), "pressure_history": pd.DataFrame(pressure_history),
            "best_epoch": config.epochs, "training_predictions": predictions,
            "selection_mode": "fixed_duration_from_inner_holdout",
            "timing": {"optimization_seconds": optimization, "training_diagnostics_seconds": diagnostics,
                       "total_training_seconds": time.perf_counter() - started}}


def make_demo_data(seed=2026):
    """Artificial feature rows with a known relationship, NOT physiological data."""
    rng = np.random.default_rng(seed)
    feature_names = [f"demo_feature_{i}" for i in range(14)]
    frames = []
    for split, count in (("train", 512), ("validation", 128)):
        values = rng.normal(size=(count, len(feature_names)))
        sbp = 125 + 12 * values[:, 0] + 5 * values[:, 1] + 2 * np.maximum(values[:, 2], 0)
        dbp = 75 + 6 * values[:, 0] - 4 * values[:, 3]
        frame = pd.DataFrame(values, columns=feature_names)
        frame["sbp"], frame["dbp"] = sbp, dbp
        # A few missing features exercise training-only median replacement.
        frame.loc[frame.index % 19 == 0, "demo_feature_4"] = np.nan
        frame["source_record_id"] = [f"artificial:{split}:record_{i // 8}" for i in range(count)]
        frame["window_id"] = [f"artificial:{split}:record_{i // 8}:window_{i % 8}" for i in range(count)]
        frame["split"], frame["usable"], frame["label_version"] = split, True, DEMO_LABEL_VERSION
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), feature_names


def plot_history(history, destination=None, *, demo=False):
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for column, label in (("train_mse_standardized", "Training"),
                          ("validation_mse_standardized", "Validation")):
        axes[0].plot(history["epoch"], history[column], label=label)
    axes[0].set(xlabel="Epoch", ylabel="MSE (standardized targets)", title="Learning curves")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["sbp_mae"], label="SBP")
    axes[1].plot(history["epoch"], history["dbp_mae"], label="DBP")
    axes[1].set(xlabel="Epoch", ylabel="Validation MAE (mmHg)", title="Pressure errors")
    axes[1].legend()
    fig.suptitle("Artificial data: programming check only" if demo else "Development validation")
    fig.tight_layout()
    if destination is not None:
        fig.savefig(destination, dpi=150)
        plt.close(fig)
    return fig


def load_bundle(path, device="cpu"):
    """Reload a state_dict and plain preprocessing data; no sklearn pickle needed."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1:
        raise ValueError("Unsupported feature-MLP bundle version")
    model = FeatureMLP(len(bundle["preprocessing"]["feature_names"]), bundle["config"]["dropout"],
                       bundle["config"].get("hidden_sizes", (32, 16)))
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    return model, bundle["preprocessing"], bundle


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_feature_table(path):
    """Check the CSV header before pandas can silently rename duplicate columns."""
    import csv

    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("CSV header is empty or contains duplicate column names")
    return pd.read_csv(path, dtype={"window_id": "string", "source_record_id": "string"})


def save_run(result, splits, row_log, directory, provenance):
    """Write a fresh run directory; never replace an earlier experiment."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    model, preprocessing = result["model"], result["preprocessing"]
    validation = splits["validation"]
    predictions = predict(model, validation, preprocessing, result["config"]["batch_size"])
    answers = validation[TARGETS].to_numpy(dtype=float)
    baseline = np.broadcast_to(preprocessing["target_mean"], answers.shape)
    metrics = pd.DataFrame([
        {"model": "feature_mlp", **regression_metrics(predictions, answers)},
        {"model": "training_mean_diagnostic", **regression_metrics(baseline, answers)},
    ])
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "matplotlib", "torch")}
    bundle = {
        "format_version": 1, "issue_url": ISSUE_URL,
        "state_dict": model.state_dict(), "preprocessing": preprocessing,
        "config": result["config"], "best_epoch": result["best_epoch"],
        "targets": TARGETS, "provenance": provenance, "package_versions": versions,
    }
    torch.save(bundle, directory / "model.pt")
    # The metadata duplicates small non-tensor fields so they are human-readable.
    metadata = {key: value for key, value in bundle.items() if key != "state_dict"}
    metadata["predictions_sbp_not_above_dbp"] = int((predictions[:, 0] <= predictions[:, 1]).sum())
    (directory / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result["history"].to_csv(directory / "history.csv", index=False)
    metrics.to_csv(directory / "validation_metrics.csv", index=False)
    row_log.to_csv(directory / "window_usage.csv", index=False)
    coverage = row_log.groupby("split")["included"].agg(total="size", included="sum")
    coverage["excluded"] = coverage["total"] - coverage["included"]
    coverage.to_csv(directory / "coverage.csv")
    output = validation[["window_id", "source_record_id", "split"] + TARGETS].copy()
    output[["predicted_sbp", "predicted_dbp"]] = predictions
    output[["baseline_sbp", "baseline_dbp"]] = baseline
    output.to_csv(directory / "validation_predictions.csv", index=False)
    plot_history(result["history"], directory / "learning_curves.png", demo=provenance["demo"])
    restored, restored_preprocessing, _ = load_bundle(directory / "model.pt", str(next(model.parameters()).device))
    np.testing.assert_allclose(predict(restored, validation, restored_preprocessing,
                                      result["config"]["batch_size"]), predictions, rtol=0, atol=1e-6)
    return metrics


def run_experiment(table, feature_names, label_version, dataset_version, directory,
                   config=TrainConfig(), *, demo=False, data_sha256=None, feature_config_sha256=None,
                   preparation_metadata=None):
    if any(not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_")
           for value in (dataset_version, label_version)):
        raise ValueError("Set the actual teammate-confirmed dataset_version and label_version")
    if Path(directory).exists():
        raise FileExistsError(f"Run directory already exists; choose a new directory: {directory}")
    splits, row_log = prepare_rows(table, feature_names, label_version)
    result = train_model(splits["train"], splits["validation"], feature_names, config)
    provenance = {"demo": demo, "dataset_version": dataset_version, "label_version": label_version,
                  "data_sha256": data_sha256, "feature_config_sha256": feature_config_sha256,
                  "scope": "artificial programming check" if demo else
                  "record-disjoint validation; patient independence is not established"}
    if preparation_metadata is not None:
        provenance["preparation"] = preparation_metadata
    metrics = save_run(result, splits, row_log, directory, provenance)
    return result, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Use artificial data; no project accuracy claim")
    source.add_argument("--data", type=Path, help="Teammate's feature/label CSV")
    parser.add_argument("--feature-config", type=Path, help="JSON with feature_names, dataset_version, label_version")
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/feature_mlp"))
    parser.add_argument("--compare", action="store_true", help="Run the four planned one-change configurations")
    defaults = TrainConfig()
    parser.add_argument("--device", choices=("cpu", "cuda"), default=defaults.device)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=defaults.hidden_sizes)
    for name, kind in (("epochs", int), ("patience", int), ("batch_size", int), ("seed", int),
                       ("threads", int), ("learning_rate", float), ("weight_decay", float), ("dropout", float)):
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=getattr(defaults, name))
    args = parser.parse_args()
    config = TrainConfig(**{key: getattr(args, key) for key in asdict(defaults)})
    config.validate()
    if args.demo:
        if args.feature_config:
            parser.error("--demo uses its own feature definitions; omit --feature-config")
        table, feature_names = make_demo_data(config.seed)
        definition = {"feature_names": feature_names, "dataset_version": "artificial_v1",
                      "label_version": DEMO_LABEL_VERSION}
        print("ARTIFICIAL DATA ONLY: these results are programming checks, not blood-pressure performance.")
    else:
        if not args.feature_config:
            parser.error("--data requires --feature-config; features are never inferred from numeric columns")
        definition = json.loads(args.feature_config.read_text(encoding="utf-8"))
        for key in ("feature_names", "dataset_version", "label_version"):
            if key not in definition:
                parser.error(f"Feature configuration is missing {key}")
        table = read_feature_table(args.data)
    variants = [("default", config)]
    if args.compare:
        variants += [("lr_0.0003", replace(config, learning_rate=0.0003)),
                     ("dropout_0", replace(config, dropout=0)),
                     ("batch_128", replace(config, batch_size=128))]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    summaries = []
    for name, settings in variants:
        destination = args.output_dir / run_id / name
        result, metrics = run_experiment(
            table, definition["feature_names"], definition["label_version"], definition["dataset_version"],
            destination, settings, demo=args.demo,
            data_sha256=file_sha256(args.data) if args.data else None,
            feature_config_sha256=file_sha256(args.feature_config) if args.feature_config else None,
            preparation_metadata={"label_configuration": definition["label_configuration"]}
            if "label_configuration" in definition else None,
        )
        scores = metrics.loc[metrics["model"] == "feature_mlp"].iloc[0].to_dict()
        summaries.append({"configuration": name, "best_epoch": result["best_epoch"], **scores})
        print(f"\n{name}: best epoch {result['best_epoch']}; saved {destination}")
        print(metrics.to_string(index=False))
    summary = pd.DataFrame(summaries)
    summary["validation_mean_mae"] = (summary["sbp_mae"] + summary["dbp_mae"]) / 2
    summary.sort_values("validation_mean_mae").to_csv(args.output_dir / run_id / "comparison.csv", index=False)


if __name__ == "__main__":
    main()
