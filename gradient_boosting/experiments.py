"""Two useful validation experiments: extra features and a four-setting search."""

import itertools
import json
from datetime import datetime, timezone

import numpy as np

from gradient_boosting import data, models
from gradient_boosting.evaluate import regression_metrics


def experiment_signature(config):
    return {"model": models.signature(config), "grid": config["tuning"]}


def choose_best(rows, target):
    candidates = [row for row in rows if row["target"] == target]
    if not candidates or any(not np.isfinite(row["macro_mae"]) for row in candidates):
        raise ValueError("Missing or invalid validation scores")
    return min(candidates, key=lambda row: (row["macro_mae"], row["order"]))


def run_experiment(config, name, rerun=False):
    if name not in ("features", "tuning"):
        raise ValueError("Choose features or tuning")
    folder = data.run_dir(config) / "experiments"
    filename = folder / f"{name}.json"
    signature = experiment_signature(config)
    if filename.exists() and not rerun:
        result = json.loads(filename.read_text())
        if result["signature"] != signature:
            raise ValueError("Experiment settings changed; use --rerun to save a separate result")
        print(f"Using saved {name} experiment.", flush=True)
        return result
    training, validation = data.load_table(config, "train"), data.load_table(config, "validation")
    if set(training["source_record_ids"]) & set(validation["source_record_ids"]):
        raise ValueError("Training and validation recordings overlap")
    a, b = training["included"], validation["included"]
    x, vx = training["X"][a], validation["X"][b]
    choices = [("six_features", list(range(6)), {}), ("all_features", list(range(17)), {})] if name == "features" else [
        ("all_features", list(range(17)), {"learning_rate": rate, "max_leaf_nodes": leaves})
        for rate, leaves in itertools.product(config["tuning"]["learning_rates"], config["tuning"]["leaf_counts"])]
    rows = []
    for order, (feature_set, columns, parameters) in enumerate(choices):
        for target in models.TARGETS:
            model = models.fit_regressor(x, training[target][a], "gb", config, columns, parameters)
            score = regression_metrics(models.predict_regressor(model, vx), validation[target][b], validation["source_record_ids"][b])
            rows.append({"order": order, "features": feature_set, "target": target, "parameters": parameters, **score})
        print(f"{name}: finished {order + 1}/{len(choices)} settings", flush=True)
    result = {"experiment": name, "signature": signature, "rows": rows,
              "best": {target: choose_best(rows, target) for target in models.TARGETS},
              "uses_test_data": False, "changes_saved_model": False}
    if rerun:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = folder / f"{name}_{stamp}.json"
    data.save_json(filename, result)
    return result
