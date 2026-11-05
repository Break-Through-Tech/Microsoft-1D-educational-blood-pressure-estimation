"""Model fitting and portable artifacts for the frozen-feature comparison.

Only supplied fitting/inner-holdout tables are used. A fixed-duration refit has
no holdout parameter and cannot select weights using original validation data.
"""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from models.train_feature_mlp import (TrainConfig, TARGETS, fit_fixed_epochs,
    fit_preprocessing, load_bundle, predict, regression_metrics, train_model)

SEEDS = (2026, 2027, 2028)
MLP_ANCHOR = {"hidden_sizes": [32, 16], "learning_rate": .001,
              "weight_decay": .0001, "dropout": 0., "batch_size": 256}
LGB_ANCHOR = {"learning_rate": .05, "num_leaves": 31,
              "min_child_samples": 200, "reg_lambda": 0.}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(3):
        try:
            temporary.replace(path)
            break
        except PermissionError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 2:
                raise
            # Windows scanners can briefly hold a just-written status file.
            time.sleep(.05 * (attempt + 1))


def mean_mae(metrics):
    return (metrics["sbp_mae"] + metrics["dbp_mae"]) / 2


def mlp_config(parameters, seed, device, epochs):
    return TrainConfig(**{**parameters, "hidden_sizes": tuple(parameters["hidden_sizes"])},
                       seed=seed, device=device, epochs=int(epochs), patience=10, threads=2)


def lgb_parameters(parameters, seed, device, platform_id=None):
    result = {**parameters, "objective": "regression_l2", "metric": "l1",
              "verbosity": -1, "num_threads": min(8, os.cpu_count() or 1),
              "bagging_fraction": .8, "bagging_freq": 1, "feature_fraction": 1.,
              "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              "data_random_seed": seed, "device_type": device, "max_bin": 255}
    if device == "cpu":
        result.update(deterministic=True, force_col_wise=True)
    else:
        result.update(gpu_platform_id=int(platform_id), gpu_device_id=0, gpu_use_dp=False)
    # Use canonical LightGBM parameter names throughout saved configurations.
    result["min_data_in_leaf"] = result.pop("min_child_samples")
    result["lambda_l2"] = result.pop("reg_lambda")
    return result


def tree_inputs(frame, preprocessing):
    values = frame[preprocessing["feature_names"]].to_numpy(float)
    return np.asarray(np.where(np.isnan(values), preprocessing["feature_medians"], values), np.float32)


def fit_lightgbm(train, names, parameters, seed, backend, *, holdout=None,
                 limit=2000, duration=None, progress=False, round_callback=None):
    import lightgbm as lgb
    state = fit_preprocessing(train, names)
    inputs = tree_inputs(train, state)
    other = tree_inputs(holdout, state) if holdout is not None else None
    settings = lgb_parameters(parameters, seed, backend["device"], backend.get("platform_id"))
    models, histories, best, stopped = [], [], [], []
    for j, target in enumerate(TARGETS):
        training = lgb.Dataset(inputs, label=train[target].to_numpy(float), free_raw_data=False)
        sets, labels = [training], ["train"]
        if holdout is not None:
            sets.append(lgb.Dataset(other, label=holdout[target].to_numpy(float), reference=training))
            labels.append("inner")
        history = {}
        callbacks = [lgb.record_evaluation(history)]
        if round_callback is not None:
            callbacks.append(round_callback)
        if holdout is not None:
            callbacks.append(lgb.early_stopping(50, first_metric_only=True, verbose=False))
        if progress:
            callbacks.append(lgb.log_evaluation(250))
        rounds = int(duration[j]) if duration is not None else int(limit)
        model = lgb.train(settings, training, num_boost_round=rounds,
                          valid_sets=sets, valid_names=labels, callbacks=callbacks)
        n = len(history["train"]["l1"])
        chosen = int(model.best_iteration or model.current_iteration())
        histories.extend({"target": target, "epoch": i + 1,
                          "training_mae": history["train"]["l1"][i],
                          **({"inner_mae": history["inner"]["l1"][i]} if holdout is not None else {})}
                         for i in range(n))
        best.append(chosen); stopped.append(n); models.append(model)
    return {"models": models, "preprocessing": state, "config": settings,
            "history": pd.DataFrame(histories), "duration": best, "stopped": stopped,
            "ceiling_censored": bool(holdout is not None and any(
                n == limit and n - b < 50 for n, b in zip(stopped, best)))}


def predictions_from_result(family, result, frame, batch_size=256):
    if family == "mlp":
        return predict(result["model"], frame, result["preprocessing"], batch_size)
    values = tree_inputs(frame, result["preprocessing"])
    return np.column_stack([m.predict(values, num_iteration=n, num_threads=min(8, os.cpu_count() or 1))
                            for m, n in zip(result["models"], result["duration"])])


def verify_complete(directory):
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    for name, expected in json.loads(marker.read_text())["sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Completed fit changed: {directory / name}")
    return True


def build_fit_request(family, train, parameters, seed, backend, *, holdout=None,
                      limit=None, duration=None, provenance=None):
    """Keep the existing on-disk request schema shared by fitting and reuse."""
    return json.loads(json.dumps(dict(family=family, parameters=parameters, seed=seed,
        backend=backend, limit=limit, fixed_duration=duration, fitting_rows=len(train),
        holdout_rows=len(holdout) if holdout is not None else 0, provenance=provenance),
        allow_nan=False))


def validate_request(actual, expected, context="Completed fit"):
    """Reject mismatches with field paths, without altering either artifact."""
    def differences(a, b, prefix=""):
        if isinstance(a, dict) and isinstance(b, dict):
            return [p for key in sorted(set(a) | set(b)) for p in
                ([prefix + str(key)] if key not in a or key not in b else
                 differences(a[key], b[key], prefix + str(key) + "."))]
        return [] if a == b else [prefix.rstrip(".") or "request"]
    fields = differences(actual, expected)
    if fields:
        raise ValueError(f"{context} request changed: {', '.join(fields)}")


def load_completed_fit(directory, expected_request, feature_names):
    """Validate hashes, both saved request copies, and feature order before reuse."""
    directory = Path(directory)
    if not verify_complete(directory):
        return None
    marker = json.loads((directory / 'complete.json').read_text())['sha256']
    required = {'fit.json', 'request.json', 'preprocessing.json'}
    if not required <= set(marker):
        raise ValueError('Completed fit integrity manifest missing: ' + ', '.join(sorted(required-set(marker))))
    info = json.loads((directory / 'fit.json').read_text())
    request = json.loads((directory / 'request.json').read_text())
    validate_request(request, expected_request)
    validate_request(info['request'], expected_request, 'Embedded fit')
    models = {'model.pt'} if expected_request['family'] == 'mlp' else {'sbp.txt', 'dbp.txt'}
    if not models <= set(marker):
        raise ValueError('Completed fit integrity manifest missing: ' + ', '.join(sorted(models-set(marker))))
    state = json.loads((directory / 'preprocessing.json').read_text())
    validate_request({'feature_names': state['feature_names']},
                     {'feature_names': list(feature_names)}, 'Preprocessing')
    return info


def predict_saved(directory, frame, device="cpu"):
    directory = Path(directory)
    info = json.loads((directory / "fit.json").read_text())
    if info["family"] == "mlp":
        model, state, _ = load_bundle(directory / "model.pt", device=device)
        return predict(model, frame, state, info["parameters"]["batch_size"])
    import lightgbm as lgb
    state = json.loads((directory / "preprocessing.json").read_text())
    values = tree_inputs(frame, state)
    return np.column_stack([lgb.Booster(model_file=str(directory / f"{t}.txt")).predict(
        values, num_threads=min(8, os.cpu_count() or 1)) for t in TARGETS])


def fit_and_save(family, train, names, parameters, seed, backend, directory,
                 *, holdout=None, limit=None, duration=None, provenance=None,
                 epoch_callback=None, resume_state=None, round_callback=None):
    """A completed fit is immutable; predictions are regenerated from its bundle."""
    directory = Path(directory)
    expected = build_fit_request(family, train, parameters, seed, backend,
        holdout=holdout, limit=limit, duration=duration, provenance=provenance)
    info = load_completed_fit(directory, expected, names)
    if info is not None:
        return info
    if resume_state is not None:
        request_path = directory / 'request.json'
        if not request_path.exists():
            raise ValueError('Interrupted checkpoint request is missing')
        validate_request(json.loads(request_path.read_text()), expected, 'Interrupted checkpoint')
        validate_request(resume_state['preprocessing'], fit_preprocessing(train, names),
                         'Checkpoint preprocessing')
    if directory.exists() and resume_state is None:
        # Preserve any interrupted attempt rather than overwriting partial files.
        i = 0
        while directory.with_name(directory.name + f"_interrupted_{i}").exists():
            i += 1
        directory.rename(directory.with_name(directory.name + f"_interrupted_{i}"))
    directory.mkdir(parents=True, exist_ok=resume_state is not None)
    write_json(directory / 'request.json', expected)
    started = time.perf_counter()
    if family == "mlp":
        cfg = mlp_config(parameters, seed, backend["device"], duration[0] if duration else limit)
        if holdout is None:
            result = fit_fixed_epochs(train, names, cfg, progress=True, epoch_callback=epoch_callback)
        else:
            result = train_model(train, holdout, names, cfg, progress=False,
                track_training_metrics=True, training_metrics_interval=5, track_pressure_metrics=True,
                epoch_callback=epoch_callback, resume_state=resume_state)
        chosen = [int(result["best_epoch"])]
        stopped = [int(result["history"].epoch.max())]
        censored = bool(holdout is not None and stopped[0] == limit and stopped[0] - chosen[0] < cfg.patience)
        payload = {"format_version": 1, "state_dict": {k: v.cpu() for k, v in result["model"].state_dict().items()},
                   "config": asdict(cfg), "preprocessing": result["preprocessing"],
                   "best_epoch": chosen[0], "targets": TARGETS, "provenance": provenance}
        torch.save(payload, directory / "model.pt")
        pressure_history = result["pressure_history"].copy()
        if holdout is not None and "split" in pressure_history:
            pressure_history["split"] = pressure_history["split"].replace({"validation": "inner"})
        pressure_history.to_csv(directory / "pressure_history.csv", index=False)
    else:
        result = fit_lightgbm(train, names, parameters, seed, backend,
                              holdout=holdout, limit=limit, duration=duration, round_callback=round_callback)
        chosen, stopped, censored = result["duration"], result["stopped"], result["ceiling_censored"]
        for t, model, n in zip(TARGETS, result["models"], chosen):
            model.save_model(str(directory / f"{t}.txt"), num_iteration=n)
    fitting_seconds = time.perf_counter() - started
    result["history"].to_csv(directory / "history.csv", index=False)
    write_json(directory / "preprocessing.json", result["preprocessing"])
    batch_size = parameters.get("batch_size", 256)
    training_predictions = result.get("training_predictions")
    if training_predictions is None:
        training_predictions = predictions_from_result(family, result, train, batch_size)
    metrics = {"train": regression_metrics(training_predictions, train[TARGETS].to_numpy(float))}
    inner_predictions = None
    if holdout is not None:
        inner_predictions = predictions_from_result(family, result, holdout, batch_size)
        metrics["inner"] = regression_metrics(inner_predictions, holdout[TARGETS].to_numpy(float))
    info = {"request": expected, "family": family, "parameters": parameters, "seed": seed,
            "duration": chosen, "stopped": stopped, "ceiling_censored": censored,
            "metrics": metrics, "inner_mean_mae": mean_mae(metrics["inner"]) if holdout is not None else None,
            "fitting_seconds": fitting_seconds, "total_seconds": time.perf_counter() - started,
            "backend": backend, "selection": "inner_mae_mmhg" if holdout is not None else "fixed_duration_inner_only"}
    write_json(directory / "fit.json", info)
    frame = holdout if holdout is not None else train
    original = inner_predictions if holdout is not None else training_predictions
    restored = predict_saved(directory, frame, backend["device"] if family == "mlp" else "cpu")
    maximum = float(np.max(np.abs(restored - original)))
    np.testing.assert_allclose(restored, original, rtol=0, atol=1e-6)
    info["reload_max_absolute_difference"] = maximum
    info["total_seconds"] = time.perf_counter() - started
    write_json(directory / "fit.json", info)
    files = [p for p in directory.iterdir() if p.is_file()]
    write_json(directory / "complete.json", {"sha256": {p.name: digest(p) for p in files},
               "reload_verified": True, "test_accessed": False})
    print(f"{family} seed{seed} duration={chosen} "
          f"inner_MAE={info['inner_mean_mae']} seconds={info['total_seconds']:.1f}", flush=True)
    return info


def converged_fit(family, train, holdout, names, parameters, seed, backend,
                  parent, *, ensure_converged=False, provenance=None):
    limit = 100 if family == "mlp" else 2000
    while True:
        directory = Path(parent) / f"limit_{limit}"
        info = fit_and_save(family, train, names, parameters, seed, backend, directory,
                            holdout=holdout, limit=limit, provenance=provenance)
        if not ensure_converged or not info["ceiling_censored"]:
            return {**info, "directory": str(directory)}
        print(f"Extending censored {family} seed{seed}: {limit} -> {limit * 2}", flush=True)
        limit *= 2
