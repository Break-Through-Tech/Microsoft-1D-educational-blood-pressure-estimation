"""Train the chosen models, save them before testing, and predict from PPG."""

import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from gradient_boosting import cleaning, data, features

MODEL_NAMES = ("training_mean", "training_median", "ridge", "gradient_boosting")
TARGETS = ("sbp", "dbp")


def package_versions():
    requirements = Path(__file__).with_name("requirements.txt").read_text().splitlines()
    return {line.split("==")[0]: importlib.metadata.version(line.split("==")[0])
            for line in requirements if "==" in line and not line.startswith("#")}


def signature(config):
    return {"processing": data.processing_signature(config), "seed": config["seed"],
            "gradient_boosting": config["gradient_boosting"], "ridge_alpha": config["ridge_alpha"],
            "model_code": data.checksum(Path(__file__)),
            "split_file": data.checksum(data.path(config["paths"]["split_config"]))}


def fit_regressor(x, y, kind, config, columns=None, parameters=None):
    """Only training data enters this function, including imputation and scaling."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != 17 or not len(x) or np.isinf(x).any() or y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("Expected 17 training features and one finite target per row")
    if kind not in ("gb", "ridge"):
        raise ValueError("Choose gb or ridge")
    columns = list(range(17)) if columns is None else list(columns)
    columns = [i for i in columns if np.isfinite(x[:, i]).any()]
    estimator = None
    if columns:
        if kind == "gb":
            settings = {**config["gradient_boosting"], **(parameters or {}),
                        "random_state": config["seed"], "categorical_features": None}
            if settings["early_stopping"] is not False:
                raise ValueError("Keep early stopping disabled; validation is separate")
            estimator = HistGradientBoostingRegressor(**settings)
        else:
            alpha = (parameters or {}).get("alpha", config["ridge_alpha"])
            estimator = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                      StandardScaler(), Ridge(alpha=alpha, solver="svd"))
        with threadpool_limits(limits=1):
            estimator.fit(x[:, columns], y)
    return {"columns": columns, "estimator": estimator,
            "constant": float(np.median(y)) if not columns else None, "kind": kind}


def predict_regressor(model, x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 17 or np.isinf(x).any():
        raise ValueError("Expected 17 features; missing values are allowed, infinities are not")
    with threadpool_limits(limits=1):
        predicted = model["estimator"].predict(x[:, model["columns"]]) if model["columns"] else np.full(len(x), model["constant"])
    if predicted.shape != (len(x),) or not np.isfinite(predicted).all():
        raise ValueError("The model returned invalid predictions")
    return predicted


def predict(bundle, x):
    if tuple(bundle["feature_names"]) != features.FEATURE_NAMES or tuple(bundle["target_names"]) != TARGETS:
        raise ValueError("Model features or targets are in the wrong order")
    values = []
    for name in MODEL_NAMES:
        model = bundle["models"][name]
        values.append(np.tile(model, (len(x), 1)) if name.startswith("training_") else
                      np.column_stack([predict_regressor(target, x) for target in model]))
    return np.stack(values)


def prediction_table(bundle, table):
    keep = table["ppg_eligible"]
    if not keep.any():
        raise ValueError("There are no usable PPG windows")
    return {"predictions": predict(bundle, table["X"][keep]),
            "truth": np.column_stack([table[t][keep] for t in TARGETS]),
            "window_ids": table["window_ids"][keep], "source_record_ids": table["source_record_ids"][keep],
            "reference_valid": table["reference_valid"][keep],
            "model_names": np.array(MODEL_NAMES), "target_names": np.array(TARGETS)}


def save_predictions(filename, payload):
    with Path(filename).open("xb") as output:
        np.savez_compressed(output, **payload)


def load_predictions(config, split):
    folder = data.run_dir(config)
    metadata = json.loads((folder / ("model_info.json" if split == "validation" else "test_results.json")).read_text())
    filename = folder / f"{split}_predictions.npz"
    if data.checksum(filename) != metadata["predictions_sha256"]:
        raise ValueError("Saved predictions changed")
    with np.load(filename, allow_pickle=False) as saved:
        return dict(saved)


def load_model(config):
    folder = data.run_dir(config)
    if not (folder / "model_info.json").exists():
        raise ValueError("Train and save a model first: python -m gradient_boosting train")
    info = json.loads((folder / "model_info.json").read_text())
    if info["fit_split"] != "train" or info["signature"] != signature(config) or info["packages"] != package_versions():
        raise ValueError("Model settings or versions changed; use a new output directory")
    if data.checksum(folder / "model.joblib") != info["model_sha256"]:
        raise ValueError("The saved model changed")
    data.check_backup(config)
    bundle = joblib.load(folder / "model.joblib")
    if tuple(bundle["feature_names"]) != features.FEATURE_NAMES or tuple(bundle["target_names"]) != TARGETS:
        raise ValueError("Wrong saved feature or target order")
    return bundle


def train(config):
    from gradient_boosting.evaluate import evaluate_predictions
    folder = data.run_dir(config)
    if (folder / "model_info.json").exists():
        load_model(config)
        print("Using the saved model; no retraining needed.", flush=True)
        return
    report = data.source_report(config)
    training = data.load_table(config, "train", report)
    validation = data.load_table(config, "validation", report)
    if set(training["source_record_ids"]) & set(validation["source_record_ids"]):
        raise ValueError("A recording crosses training and validation")
    keep = training["included"]
    x, y = training["X"][keep], np.column_stack([training[t][keep] for t in TARGETS])
    if not len(x) or not len(validation["X"][validation["included"]]):
        raise ValueError("Both development splits need usable labelled windows")
    before = signature(config)
    models = {"training_mean": y.mean(axis=0), "training_median": np.median(y, axis=0)}
    for name, kind in (("ridge", "ridge"), ("gradient_boosting", "gb")):
        models[name] = [fit_regressor(x, y[:, t], kind, config) for t in range(2)]
    bundle = {"feature_names": list(features.FEATURE_NAMES), "target_names": list(TARGETS),
              "fit_split": "train", "models": models}
    predictions = prediction_table(bundle, validation)
    summary = evaluate_predictions(predictions, validation, config, "validation")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "model.joblib").exists() or (folder / "validation_predictions.npz").exists():
        raise FileExistsError("An incomplete model run exists; preserve it and use a new output directory")
    joblib.dump(bundle, folder / "model.joblib")
    np.testing.assert_allclose(predict(joblib.load(folder / "model.joblib"), validation["X"][validation["ppg_eligible"]]),
                               predictions["predictions"], rtol=1e-6, atol=1e-6)
    save_predictions(folder / "validation_predictions.npz", predictions)
    data.save_json(folder / "validation_results.json", summary)
    if signature(config) != before:
        raise ValueError("Processing settings changed during training")
    data.save_json(folder / "model_info.json", {"fit_split": "train", "signature": before,
        "packages": package_versions(), "source_id": data.source_id(report),
        "model_sha256": data.checksum(folder / "model.joblib"),
        "predictions_sha256": data.checksum(folder / "validation_predictions.npz")})
    print("Models saved before test preparation. See validation_results.json.", flush=True)


def test(config):
    from gradient_boosting.evaluate import evaluate_predictions
    bundle = load_model(config)
    folder = data.run_dir(config)
    if (folder / "test_results.json").exists():
        verify_predictions(config, "test")
        print("Existing test results verified; no new model was selected.", flush=True)
        return
    model_hash = data.checksum(folder / "model.joblib")
    report = data.source_report(config)
    info = json.loads((folder / "model_info.json").read_text())
    if data.source_id(report) != info["source_id"]:
        raise ValueError("The prepared dataset changed after training")
    data.prepare_split(config, "test", report)
    table = data.load_table(config, "test", report)
    payload = prediction_table(bundle, table)
    result = evaluate_predictions(payload, table, config, "test")
    save_predictions(folder / "test_predictions.npz", payload)
    data.save_json(folder / "test_results.json", {**result, "model_sha256": model_hash,
        "predictions_sha256": data.checksum(folder / "test_predictions.npz")})
    if data.checksum(folder / "model.joblib") != model_hash:
        raise ValueError("The model changed during testing")
    print("Test evaluation saved. The model was not fitted or tuned on test data.", flush=True)


def verify_predictions(config, split):
    from gradient_boosting.evaluate import evaluate_predictions
    bundle = load_model(config)
    table = data.load_table(config, split)
    expected, saved = prediction_table(bundle, table), load_predictions(config, split)
    if set(expected) != set(saved):
        raise ValueError("Wrong prediction-file fields")
    for key, value in expected.items():
        same = np.array_equal(value, saved[key], equal_nan=True) if value.dtype.kind == "f" else np.array_equal(value, saved[key])
        if not same:
            raise ValueError(f"Saved predictions differ: {key}")
    result = evaluate_predictions(saved, table, config, split)
    stored = json.loads((data.run_dir(config) / f"{split}_results.json").read_text())
    if any(stored[key] != value for key, value in result.items()):
        raise ValueError("The saved results do not match their predictions")
    if split == "test" and stored["model_sha256"] != data.checksum(data.run_dir(config) / "model.joblib"):
        raise ValueError("Test results belong to a different model")
    print(f"{split}: predictions and results match; no fitting.", flush=True)
    return result


def predict_ppg(bundle, raw, config):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 625 or not len(raw):
        raise ValueError("Provide a raw PPG array with shape (n, 625)")
    values = np.full((4, len(raw), 2), np.nan)
    measured = np.full((len(raw), 17), np.nan)
    reasons, eligible = [], np.zeros(len(raw), dtype=bool)
    for i, window in enumerate(raw):
        ppg = cleaning.clean_ppg(window, config)
        reasons.append(ppg["reason"])
        eligible[i] = ppg["eligible"]
        if eligible[i]:
            measured[i] = features.extract_window(ppg["cleaned"], config["sample_rate_hz"])["values"]
    if eligible.any():
        values[:, eligible] = predict(bundle, measured[eligible])
    return {"predictions": values, "ppg_eligible": eligible, "ppg_reason": np.array(reasons),
            "model_names": np.array(MODEL_NAMES), "target_names": np.array(TARGETS)}
