"""MAE/RMSE, recording balance, pressure ranges, and paired uncertainty."""

import numpy as np

from gradient_boosting import data

NAMES = ("training_mean", "training_median", "ridge", "gradient_boosting")
TARGETS = ("sbp", "dbp")
BANDS = {"sbp": ("below_100", "100_to_139", "140_to_179", "180_or_above", "unknown"),
         "dbp": ("below_60", "60_to_89", "90_to_109", "110_or_above", "unknown")}


def regression_metrics(predicted, truth, records):
    predicted, truth, records = np.asarray(predicted, float), np.asarray(truth, float), np.asarray(records, str)
    if predicted.ndim != 1 or predicted.shape != truth.shape or truth.shape != records.shape:
        raise ValueError("Predictions, labels and recording IDs must line up")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all() or (records == "").any():
        raise ValueError("Invalid predictions, labels or recording IDs")
    result = {"n_windows": len(truth), "n_recordings": len(np.unique(records))}
    keys = ("pooled_mae", "pooled_rmse", "pooled_mean_error", "macro_mae", "macro_rmse", "macro_mean_error")
    if not len(truth):
        return {**result, **dict.fromkeys(keys)}
    error = predicted - truth
    _, inverse, counts = np.unique(records, return_inverse=True, return_counts=True)
    mae = np.bincount(inverse, weights=np.abs(error)) / counts
    mse = np.bincount(inverse, weights=error ** 2) / counts
    mean = np.bincount(inverse, weights=error) / counts
    return {**result, "pooled_mae": float(np.abs(error).mean()), "pooled_rmse": float(np.sqrt(np.mean(error ** 2))),
            "pooled_mean_error": float(error.mean()), "macro_mae": float(mae.mean()),
            "macro_rmse": float(np.sqrt(mse.mean())), "macro_mean_error": float(mean.mean())}


def paired_intervals(truth, predictions, records, config, split):
    """Resample whole recordings together for every model and both BP targets."""
    unique, inverse, counts = np.unique(records, return_inverse=True, return_counts=True)
    if not len(unique):
        return []
    mae = np.empty((4, len(unique), 2))
    for model in range(4):
        for target in range(2):
            mae[model, :, target] = np.bincount(inverse, weights=np.abs(predictions[model, :, target] - truth[:, target])) / counts
    rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 7, 0 if split == "validation" else 1]))
    draws = rng.integers(0, len(unique), size=(config["bootstrap_samples"], len(unique)), dtype=np.int32)
    rows = []
    for baseline in ("training_median", "ridge"):
        base = NAMES.index(baseline)
        for target, name in enumerate(TARGETS):
            differences = mae[3, :, target] - mae[base, :, target]
            sampled = differences[draws].mean(axis=1)
            low, high = np.quantile(sampled, [.025, .975], method="linear")
            rows.append({"candidate": "gradient_boosting", "baseline": baseline, "target": name,
                         "delta_mae": float(differences.mean()), "ci_lower": float(low), "ci_upper": float(high)})
    return rows


def evaluate_predictions(payload, table, config, split):
    if split not in ("validation", "test") or not (table["split_assignments"] == split).all():
        raise ValueError("Evaluation received the wrong split")
    eligible, retained = table["ppg_eligible"], table["included"]
    for key in ("window_ids", "source_record_ids", "reference_valid"):
        if not np.array_equal(payload[key], table[key][eligible]):
            raise ValueError(f"Prediction rows do not match the data: {key}")
    truth = np.column_stack([table[t][eligible] for t in TARGETS])
    predictions = payload["predictions"]
    if (not np.array_equal(payload["truth"], truth, equal_nan=True) or tuple(payload["model_names"]) != NAMES
            or tuple(payload["target_names"]) != TARGETS or predictions.shape != (4, len(truth), 2)
            or not np.isfinite(predictions).all()):
        raise ValueError("Wrong labels, model order or prediction values")
    valid = payload["reference_valid"]
    y, pred, records = truth[valid], predictions[:, valid], payload["source_record_ids"][valid]
    result = {"split": split, "coverage": {"source_windows": len(table["X"]),
        "selected_recordings": len(np.unique(table["source_record_ids"])),
        "ppg_eligible_windows": int(eligible.sum()), "retained_windows": int(retained.sum()),
        "retained_recordings": len(np.unique(records)), "excluded_windows": int((~retained).sum()),
        "ppg_failures": int((~eligible).sum()), "reference_failures": int((~table["reference_valid"]).sum())},
        "metrics": [], "pressure_ranges": [], "recordings": [], "ordering": {}}
    for m, model in enumerate(NAMES):
        result["ordering"][model] = int((predictions[m, :, 0] <= predictions[m, :, 1]).sum())
        for t, target in enumerate(TARGETS):
            result["metrics"].append({"model": model, "target": target,
                **regression_metrics(pred[m, :, t], y[:, t], records)})
    for t, target in enumerate(TARGETS):
        bands = np.array([data.pressure_band(value, target) for value in table[target]])
        for band in BANDS[target]:
            selected = bands == band
            scored = bands[retained] == band
            result["pressure_ranges"].append({"target": target, "band": band,
                "source_windows": int(selected.sum()), "retained_windows": int((selected & retained).sum()),
                **regression_metrics(pred[3, scored, t], y[scored, t], records[scored])})
    unique, inverse = np.unique(table["source_record_ids"], return_inverse=True)
    source_counts = np.bincount(inverse)
    included_counts = np.bincount(inverse, weights=retained.astype(int), minlength=len(unique)).astype(int)
    errors = np.zeros((2, len(unique)))
    for t in range(2):
        errors[t] = np.bincount(inverse[retained], weights=np.abs(pred[3, :, t] - y[:, t]), minlength=len(unique))
    for i, record in enumerate(unique):
        result["recordings"].append({"recording": str(record), "source_windows": int(source_counts[i]),
            "retained_windows": int(included_counts[i]),
            "sbp_mae": float(errors[0, i] / included_counts[i]) if included_counts[i] else None,
            "dbp_mae": float(errors[1, i] / included_counts[i]) if included_counts[i] else None})
    result["intervals"] = paired_intervals(y, pred, records, config, split)
    return result
