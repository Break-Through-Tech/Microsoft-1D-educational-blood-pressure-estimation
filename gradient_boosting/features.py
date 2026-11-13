"""The 17 PPG inputs: six signal summaries and eleven pulse-shape measurements.

Only cleaned PPG enters these functions. A failed pulse measurement stays NaN;
other useful features are kept. Times are in seconds; normalized shapes have no units.
"""

import warnings
from collections import Counter

import numpy as np
from gradient_boosting.cleaning import nk, warning_text

BASIC_NAMES = (
    "ppg_clean_mean_feature", "ppg_clean_std_feature", "ppg_clean_range_feature",
    "median_interpeak_interval_s_feature", "median_rate_from_peaks_bpm_feature",
    "median_pulse_amplitude_feature",
)


def from_peaks(cleaned, peaks, fs=125):
    """Notebook formulas, including population std and inclusive pulse intervals.

    Rate is median(60 / interval), not 60 / median(interval). Pulse amplitude
    uses current_peak minus min(previous_peak:current_peak+1), as in the starter.
    The first peak alone supplies neither an interval nor a pulse amplitude.
    """
    values = np.asarray(cleaned, dtype=float)
    peaks = np.asarray(peaks)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or fs <= 0:
        raise ValueError("Expected finite one-dimensional cleaned PPG and positive sample rate")
    if peaks.ndim != 1 or (peaks.size and (
            peaks.dtype.kind not in "iu" or peaks[0] < 0 or peaks[-1] >= len(values)
            or (np.diff(peaks) <= 0).any())):
        raise ValueError("Peak indices must be increasing, unique integers within the waveform")
    peaks = peaks.astype(int)
    intervals = np.diff(peaks) / fs
    amplitudes = [values[right] - np.min(values[left:right + 1])
                  for left, right in zip(peaks[:-1], peaks[1:])]
    return np.array([
        values.mean(), values.std(), np.ptp(values),
        np.median(intervals) if intervals.size else np.nan,
        np.median(60.0 / intervals) if intervals.size else np.nan,
        np.median(amplitudes) if amplitudes else np.nan,
    ])


MORPH_NAMES = (
    "median_rise_time_s", "median_beat_duration_s", "median_rise_fraction",
    "median_width_50_s", "median_width_75_s", "median_width_50_fraction",
    "median_width_75_fraction", "median_pulse_area", "median_normalized_area",
    "median_normalized_rise_slope", "median_normalized_decay_slope_magnitude",
)


def crossing_width(residual, peak, fraction, fs):
    """Nearest threshold crossings around the designated peak, linearly interpolated."""
    level = residual[peak] * fraction
    left = np.flatnonzero(residual[:peak] <= level)
    right = np.flatnonzero(residual[peak + 1:] <= level)
    if not len(left) or not len(right):
        return np.nan
    a, b = int(left[-1]), int(peak + 1 + right[0])
    left_change, right_change = residual[a + 1] - residual[a], residual[b - 1] - residual[b]
    if left_change <= 0 or right_change <= 0:
        return np.nan
    left_x = a + (level - residual[a]) / left_change
    right_x = b - (level - residual[b]) / right_change
    return (right_x - left_x) / fs


def pulse_features(signal, left, peak, right, fs=125):
    """A complete trough-to-trough pulse with a linear baseline between troughs.

    Height is the designated peak above that baseline. Area integrates positive
    residual only. Slopes are maxima of rise/fall magnitude after normalizing
    residual by height and time by total beat duration. Invalid boundaries or
    nonpositive height produce missing features, never invented boundary beats.
    """
    values = np.asarray(signal, dtype=float)
    missing = np.full(len(MORPH_NAMES), np.nan)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Expected finite PPG and positive sample rate")
    if not all(isinstance(i, (int, np.integer)) for i in (left, peak, right)):
        raise ValueError("Pulse indices must be integers")
    if not 0 <= left < peak < right < len(values):
        return missing, "invalid_boundaries"
    segment = values[left:right + 1]
    residual = segment - np.linspace(segment[0], segment[-1], len(segment))
    local_peak = peak - left
    height = residual[local_peak]
    if not height > 0:
        return missing, "nonpositive_height"
    duration, rise = (right - left) / fs, (peak - left) / fs
    width50, width75 = (crossing_width(residual, local_peak, f, fs) for f in (.5, .75))
    area = float(np.trapezoid(np.maximum(residual, 0), dx=1 / fs))
    normalized = residual / height
    rise_slope = float(np.max(np.diff(normalized[:local_peak + 1])) * (right - left))
    decay_slope = float(np.max(-np.diff(normalized[local_peak:])) * (right - left))
    return np.array([rise, duration, rise / duration, width50, width75, width50 / duration,
                     width75 / duration, area, area / (height * duration), rise_slope, decay_slope]), "ok"


def from_boundaries(cleaned, peaks, fs=125):
    """Use only interior peaks with troughs between adjacent detected peaks.

    Each interpeak trough is the first minimum in the inclusive peak interval.
    The first and last detected peaks cannot have both boundaries. A trough
    landing on a peak is invalid; window edges are never substituted for troughs.
    """
    from_peaks(cleaned, peaks, fs)  # Reuse the frozen input/peak validation.
    values, peaks = np.asarray(cleaned, dtype=float), np.asarray(peaks, dtype=int)
    troughs = np.array([a + int(np.argmin(values[a:b + 1]))
                        for a, b in zip(peaks[:-1], peaks[1:])], dtype=int)
    candidates, issues, pulses, boundaries = max(0, len(peaks) - 2), Counter(), [], []
    for i in range(1, len(peaks) - 1):
        boundary = (int(troughs[i - 1]), int(peaks[i]), int(troughs[i]))
        features, reason = pulse_features(values, *boundary, fs)
        if reason == "ok":
            pulses.append(features)
            boundaries.append(boundary)
        else:
            issues[reason] += 1
    aggregated = np.full(len(MORPH_NAMES), np.nan)
    if pulses:
        pulse_values = np.stack(pulses)
        for i in range(len(MORPH_NAMES)):
            finite = pulse_values[:, i][np.isfinite(pulse_values[:, i])]
            if len(finite):
                aggregated[i] = np.median(finite)
    status = ("no_complete_pulses" if not candidates else "no_valid_pulses" if not pulses
              else "partial_pulses" if issues else "ok")
    return {"values": aggregated, "status": status, "candidate_pulses": candidates,
            "valid_pulses": len(pulses), "issues": dict(issues), "boundaries": boundaries,
            "peaks": peaks, "troughs": troughs}


def extract_window(cleaned, fs=125):
    base = from_peaks(cleaned, [], fs)
    result = {"values": np.r_[base, np.full(len(MORPH_NAMES), np.nan)],
              "status": "peak_detection_error", "candidate_pulses": 0, "valid_pulses": 0,
              "peak_count": 0, "error": "", "warnings": "", "issues": {}, "boundaries": [],
              "peaks": np.array([], dtype=int), "troughs": np.array([], dtype=int)}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            peaks = nk.ppg_findpeaks(np.asarray(cleaned, dtype=float), sampling_rate=fs,
                                     method="elgendi", show=False)["PPG_Peaks"]
        result["warnings"] = warning_text(caught)
        pulse = from_boundaries(cleaned, peaks, fs)
        result.update(pulse, values=np.r_[from_peaks(cleaned, peaks, fs), pulse["values"]],
                      peak_count=len(peaks))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


FEATURE_NAMES = (*BASIC_NAMES, *MORPH_NAMES)
