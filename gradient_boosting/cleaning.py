"""Boosting-owned adaptation of cleaning/clean_data.ipynb, without editing it.

Signals are first cast to float32, matching the notebook's source loading.
PPG eligibility and ABP reference validity are deliberately independent.
"""

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]
                                        / "data/derived/gradient_boosting/matplotlib_cache"))

import neurokit2 as nk
import numpy as np

from data import bp_utils


def validate_settings(config):
    settings = config["cleaning"]
    for name in ("max_zero_fraction", "min_std", "abp_low_pass_hz", "abp_filter_order",
                 "zero_policy", "ppg_method", "sbp_range", "dbp_range"):
        if name not in settings:
            raise ValueError(f"Missing cleaning setting: {name}")
    if not 0 <= settings["max_zero_fraction"] <= 1 or not settings["min_std"] > 0:
        raise ValueError("Invalid zero-fraction or minimum-standard-deviation threshold")
    if not 0 < settings["abp_low_pass_hz"] < config["sample_rate_hz"] / 2:
        raise ValueError("ABP low-pass must be between zero and the Nyquist frequency")
    if not isinstance(settings["abp_filter_order"], int) or settings["abp_filter_order"] < 1:
        raise ValueError("ABP filter order must be a positive integer")
    if settings["ppg_method"] != "elgendi":
        raise ValueError("Only the reference notebook's elgendi PPG method is supported")
    if settings["zero_policy"] != "linear_interpolation_with_constant_edge_extension":
        raise ValueError("Unsupported zero interpolation policy")
    for name in ("sbp_range", "dbp_range"):
        bounds = np.asarray(settings[name], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not bounds[0] < bounds[1]:
            raise ValueError(f"Invalid {name}")


def interpolate_zeros(signal):
    """Match the notebook, including constant extension at leading/trailing zeros."""
    result = np.asarray(signal, dtype=np.float64).copy()
    zero = result == 0
    if zero.any() and not zero.all():
        positions = np.arange(result.size)
        result = np.interp(positions, positions[~zero], result[~zero])
    return result


def raw_status(signal, config):
    values = np.asarray(signal, dtype=np.float32)
    if values.shape != (config["window_samples"],):
        raise ValueError("Expected a single 625-sample signal")
    finite = bool(np.isfinite(values).all())
    zero_fraction = float(np.mean(values == 0))
    std = float(np.std(values)) if finite else float("nan")
    if not finite:
        reason = "nonfinite_signal"
    elif zero_fraction > config["cleaning"]["max_zero_fraction"]:
        reason = "excess_zeros"
    elif std < config["cleaning"]["min_std"]:
        reason = "flat_signal"
    else:
        reason = "ok"
    return values, {"reason": reason, "zero_fraction": zero_fraction, "raw_std": std}


def warning_text(caught):
    return "; ".join(sorted({f"{item.category.__name__}: {item.message}" for item in caught}))


def clean_ppg(ppg, config):
    """PPG-only inference eligibility; neither ABP nor its labels are arguments."""
    values, status = raw_status(ppg, config)
    result = {**status, "eligible": False, "cleaned": None, "peak_count": 0,
              "warnings": "", "error": ""}
    if status["reason"] != "ok":
        return result
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            signals, _ = nk.ppg_process(interpolate_zeros(values),
                                         sampling_rate=config["sample_rate_hz"],
                                         method=config["cleaning"]["ppg_method"])
        result["warnings"] = warning_text(caught)
        cleaned = signals["PPG_Clean"].to_numpy(dtype=np.float64)
        if cleaned.shape != values.shape or not np.isfinite(cleaned).all():
            raise ValueError("NeuroKit returned invalid PPG samples")
        result.update(eligible=True, cleaned=cleaned.astype(np.float32),
                      peak_count=int((signals["PPG_Peaks"].to_numpy() > 0).sum()))
    except Exception as error:
        result.update(reason="processing_error", error=f"{type(error).__name__}: {error}")
    return result




def clean_reference(abp, config):
    """Reference validity for research evaluation; never an inference gate."""
    values, status = raw_status(abp, config)
    result = {**status, "valid": False, "cleaned": None, "sbp": float("nan"),
              "dbp": float("nan"), "unfiltered_sbp": float("nan"),
              "unfiltered_dbp": float("nan"), "peak_count": 0, "beat_count": 0,
              "warnings": "", "error": ""}
    if status["reason"] != "ok":
        return result
    try:
        unfiltered = interpolate_zeros(values)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw_sbp, raw_dbp, _, _ = bp_utils.extract_bp_labels(unfiltered, fs=config["sample_rate_hz"])
            filtered = np.asarray(nk.signal_filter(
                unfiltered, sampling_rate=config["sample_rate_hz"],
                highcut=config["cleaning"]["abp_low_pass_hz"], method="butterworth",
                order=config["cleaning"]["abp_filter_order"]), dtype=np.float64)
            if filtered.shape != values.shape or not np.isfinite(filtered).all():
                raise ValueError("NeuroKit returned invalid ABP samples")
            sbp, dbp, peaks, beats = bp_utils.extract_bp_labels(filtered, fs=config["sample_rate_hz"])
        result.update(cleaned=filtered.astype(np.float32), sbp=float(sbp), dbp=float(dbp),
                      unfiltered_sbp=float(raw_sbp), unfiltered_dbp=float(raw_dbp),
                      peak_count=int(peaks), beat_count=int(beats), warnings=warning_text(caught))
        if not np.isfinite([sbp, dbp]).all():
            result["reason"] = "insufficient_complete_beats"
        elif not config["cleaning"]["sbp_range"][0] <= sbp <= config["cleaning"]["sbp_range"][1]:
            result["reason"] = "sbp_outside_configured_range"
        elif not config["cleaning"]["dbp_range"][0] <= dbp <= config["cleaning"]["dbp_range"][1]:
            result["reason"] = "dbp_outside_configured_range"
        elif sbp <= dbp:
            result["reason"] = "sbp_not_above_dbp"
        else:
            result["valid"] = True
    except Exception as error:
        result.update(reason="processing_error", error=f"{type(error).__name__}: {error}")
    return result


def clean_window(ppg, abp, config):
    """Evaluate stages independently; joint inclusion matches the notebook's rules."""
    ppg_result = clean_ppg(ppg, config)
    reference = clean_reference(abp, config)
    return ppg_result, reference, bool(ppg_result["eligible"] and reference["valid"])
