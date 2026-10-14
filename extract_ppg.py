"""PPG-only feature extraction with NeuroKit2.

The functions in this module intentionally do not accept ABP or labels.  That
keeps the engineered inputs available at prediction time and makes accidental
target leakage harder.  Each call processes one fixed-length PPG window and
returns both features and quality-control metadata.
"""

from __future__ import annotations

from collections.abc import Mapping

import neurokit2 as nk
import numpy as np
from scipy.stats import skew


DEFAULT_FEATURE_CONFIG = {
    "sampling_rate": 125,
    "method": "elgendi",
    "include_raw_summary": False,
}

FEATURE_NAMES = (
    "ppg_clean_mean",
    "ppg_clean_std",
    "ppg_clean_range",
    "median_interpeak_interval_s",
    "median_rate_bpm",
    "median_pulse_amplitude",
    "median_rise_time_s",
    "median_decay_time_s",
    "median_rise_slope",
    "median_decay_slope",
    "median_half_amplitude_width_s",
    "median_pulse_area",
    "pulse_amplitude_cv",
    "median_pulse_skewness",
)

RAW_SUMMARY_FEATURE_NAMES = (
    "ppg_raw_mean",
    "ppg_raw_std",
    "ppg_raw_range",
)


def _empty_features(include_raw_summary: bool) -> dict[str, float]:
    names = FEATURE_NAMES + (RAW_SUMMARY_FEATURE_NAMES if include_raw_summary else ())
    return {name: np.nan for name in names}


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else np.nan


def _pulse_morphology(
    clean: np.ndarray,
    peaks: np.ndarray,
    sampling_rate: int,
) -> dict[str, np.ndarray]:
    """Calculate beat-level pulse-shape measurements from adjacent peaks.

    A beat is represented by the interval between two detected peaks.  The
    minimum between each pair is used as that beat's trough.  This is a
    transparent, reproducible morphology approximation for short PPG windows;
    the resulting values are summarized across beats by the caller.
    """
    values = {
        "amplitude": [],
        "rise_time": [],
        "decay_time": [],
        "rise_slope": [],
        "decay_slope": [],
        "half_width": [],
        "area": [],
        "skewness": [],
    }
    for previous_peak, current_peak, next_peak in zip(peaks[:-2], peaks[1:-1], peaks[2:]):
        rise_segment = clean[previous_peak : current_peak + 1]
        decay_segment = clean[current_peak : next_peak + 1]
        if rise_segment.size < 2 or decay_segment.size < 2:
            continue

        trough_before = previous_peak + int(np.argmin(rise_segment))
        trough_after = current_peak + int(np.argmin(decay_segment))
        amplitude = float(clean[current_peak] - clean[trough_before])
        if not np.isfinite(amplitude) or amplitude <= 0:
            continue

        rise_time = (current_peak - trough_before) / sampling_rate
        decay_time = (trough_after - current_peak) / sampling_rate
        full_segment = clean[trough_before : trough_after + 1]
        baseline = float(min(clean[trough_before], clean[trough_after]))
        normalized = full_segment - baseline
        half_level = amplitude / 2
        above_half = np.flatnonzero(normalized >= half_level)
        area = float(np.trapz(normalized, dx=1 / sampling_rate))

        values["amplitude"].append(amplitude)
        values["rise_time"].append(rise_time)
        values["decay_time"].append(decay_time)
        values["rise_slope"].append(amplitude / rise_time if rise_time > 0 else np.nan)
        values["decay_slope"].append(amplitude / decay_time if decay_time > 0 else np.nan)
        values["half_width"].append(
            (above_half[-1] - above_half[0]) / sampling_rate if above_half.size >= 2 else np.nan
        )
        values["area"].append(area)
        values["skewness"].append(float(skew(full_segment, bias=False)) if full_segment.size > 2 else np.nan)

    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def extract_ppg_features(
    ppg_window: np.ndarray,
    *,
    config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Extract NeuroKit2 PPG features from one window.

    Parameters
    ----------
    ppg_window:
        One-dimensional PPG signal.  The current project uses 625 samples at
        125 Hz, but the function only requires a one-dimensional finite array.
    config:
        Optional settings.  Supported keys are ``sampling_rate``, ``method``,
        and ``include_raw_summary``.

    Returns
    -------
    dict
        Feature values plus ``processing_status``, ``processing_error``,
        ``detected_peak_count``, ``ppg_zero_count``, and ``ppg_zero_fraction``.
        A failed window is retained with NaN feature values instead of being
        silently discarded.
    """
    settings = dict(DEFAULT_FEATURE_CONFIG)
    if config:
        settings.update(config)
    include_raw_summary = bool(settings["include_raw_summary"])

    signal = np.asarray(ppg_window, dtype=float)
    result: dict[str, object] = {
        **_empty_features(include_raw_summary),
        "ppg_zero_count_qc": int(np.count_nonzero(signal == 0)) if signal.size else 0,
        "ppg_zero_fraction_qc": float(np.mean(signal == 0)) if signal.size else np.nan,
        "processing_status": "failed",
        "processing_error": "",
        "detected_peak_count_qc": 0,
    }

    if include_raw_summary and signal.size and np.isfinite(signal).all():
        result.update(
            {
                "ppg_raw_mean": float(np.mean(signal)),
                "ppg_raw_std": float(np.std(signal)),
                "ppg_raw_range": float(np.ptp(signal)),
            }
        )

    try:
        if signal.ndim != 1 or signal.size < 3:
            raise ValueError("PPG must be a one-dimensional signal with at least 3 samples")
        if not np.isfinite(signal).all():
            raise ValueError("PPG contains non-finite values")

        signals, _ = nk.ppg_process(
            signal,
            sampling_rate=int(settings["sampling_rate"]),
            method=str(settings["method"]),
        )
        clean = signals["PPG_Clean"].to_numpy(dtype=float)
        peak_mask = signals["PPG_Peaks"].to_numpy(dtype=int) > 0
        peaks = np.flatnonzero(peak_mask)
        intervals = np.diff(peaks) / int(settings["sampling_rate"])
        rates = 60.0 / intervals if intervals.size else np.array([], dtype=float)
        morphology = _pulse_morphology(
            clean, peaks, sampling_rate=int(settings["sampling_rate"])
        )
        amplitudes = morphology["amplitude"]
        amplitude_mean = float(np.mean(amplitudes)) if amplitudes.size else np.nan

        result.update(
            {
                "ppg_clean_mean": float(np.mean(clean)),
                "ppg_clean_std": float(np.std(clean)),
                "ppg_clean_range": float(np.ptp(clean)),
                "median_interpeak_interval_s": _finite_median(intervals),
                "median_rate_bpm": _finite_median(rates),
                "median_pulse_amplitude": _finite_median(amplitudes),
                "median_rise_time_s": _finite_median(morphology["rise_time"]),
                "median_decay_time_s": _finite_median(morphology["decay_time"]),
                "median_rise_slope": _finite_median(morphology["rise_slope"]),
                "median_decay_slope": _finite_median(morphology["decay_slope"]),
                "median_half_amplitude_width_s": _finite_median(morphology["half_width"]),
                "median_pulse_area": _finite_median(morphology["area"]),
                "pulse_amplitude_cv": (
                    float(np.std(amplitudes, ddof=1) / amplitude_mean)
                    if amplitudes.size > 1 and amplitude_mean != 0
                    else np.nan
                ),
                "median_pulse_skewness": _finite_median(morphology["skewness"]),
                "detected_peak_count_qc": int(peaks.size),
                "processing_status": "ok",
            }
        )
    except Exception as error:  # retain the row so failures are auditable
        result["processing_error"] = f"{type(error).__name__}: {error}"

    return result


def extract_ppg_feature_rows(
    ppg_windows: np.ndarray,
    *,
    config: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    """Extract one auditable feature row for each row in a 2-D window array."""
    windows = np.asarray(ppg_windows)
    if windows.ndim != 2:
        raise ValueError(f"Expected a 2-D array of windows, got shape {windows.shape}")
    return [extract_ppg_features(window, config=config) for window in windows]
