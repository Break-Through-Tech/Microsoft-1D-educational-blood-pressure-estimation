"""Shared team cleaning and PPG-only features. Historical demo code is separate."""
import warnings

import numpy as np
from scipy.stats import skew

from data import bp_utils

FS = 125
FEATURE_NAMES = [
    "ppg_clean_mean", "ppg_clean_std", "ppg_clean_range",
    "median_interpeak_interval_s", "median_rate_bpm", "median_pulse_amplitude",
]
MORPHOLOGY_NAMES = [
    "median_rise_time_s", "median_decay_time_s", "median_rise_slope",
    "median_decay_slope", "median_half_amplitude_width_s", "median_pulse_area",
    "pulse_amplitude_cv", "median_pulse_skewness",
]
ALL_FEATURE_NAMES = FEATURE_NAMES + MORPHOLOGY_NAMES
CLEANING_CONFIG = {
    "sampling_rate": FS, "window_samples": 625, "max_zero_fraction": 0.05,
    "min_std": 1e-6, "abp_lowpass_hz": 40, "abp_filter_order": 4,
    "ppg_method": "elgendi", "sbp_range": [60., 250.], "dbp_range": [30., 150.],
    "reject_repeated_max_plateaus": False,
}
FEATURE_DEFINITIONS = {
    "starter": "Cleaned PPG mean/std(ddof=0)/range; median diff(peaks)/fs; median 60/interval; median clean[next_peak]-min(clean[peak:next_peak+1]).",
    "morphology": "Interior peaks bounded by minima between adjacent peaks; linear baseline between feet; positive baseline-corrected pulse; nearest interpolated half-height crossings. Rise/decay times run from feet to peak; slopes are corrected peak height/time; area uses numpy.trapezoid; skew is scipy.stats.skew(bias=False). Pulse summaries are medians; amplitude CV is std(ddof=0)/mean of valid corrected heights.",
    "units": "Times and widths seconds; rate beats/min; PPG amplitude arbitrary source units; slopes units/s; area units*s; CV and skew dimensionless.",
}


def label_definition():
    return {"label_version": "pr41_complete_beats_median_v2",
            "prominence_floor_mmhg": bp_utils.MIN_PROMINENCE,
            "relative_prominence": bp_utils.RELATIVE_PROMINENCE,
            "spread_percentiles": [5, 95], "minimum_peak_distance_s": bp_utils.MIN_PEAK_DISTANCE_S,
            "distance_rounding": "round(seconds*sampling_rate)", "min_complete_beats": bp_utils.MIN_BEATS,
            "aggregation": "median", "dbp": "minimum in [previous_peak, peak)",
            "source_pr": 41, "source_commit": "e1b273a10200342e2d55532854f146e8f516e93a"}


def settings_checked(settings=None):
    cfg = {**CLEANING_CONFIG, **(settings or {})}
    if set(cfg) != set(CLEANING_CONFIG):
        raise ValueError("Unknown cleaning configuration field")
    for name in ("sampling_rate", "window_samples", "abp_filter_order"):
        if not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] < 1:
            raise ValueError(f"Invalid {name}")
    if not 0 <= cfg["max_zero_fraction"] <= 1 or not np.isfinite(cfg["min_std"]) or cfg["min_std"] <= 0:
        raise ValueError("Invalid raw quality settings")
    if not 0 < cfg["abp_lowpass_hz"] < cfg["sampling_rate"] / 2 or cfg["ppg_method"] != "elgendi":
        raise ValueError("Invalid filter settings")
    for name in ("sbp_range", "dbp_range"):
        bounds = cfg[name]
        if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]):
            raise ValueError(f"Invalid {name}")
    return cfg


def raw_signal_reason(signal, settings=None):
    cfg = settings_checked(settings)
    x = np.asarray(signal)
    if x.shape != (cfg["window_samples"],): return "wrong_window_shape"
    if not np.isfinite(x).all(): return "nonfinite_signal"
    if np.mean(x == 0) > cfg["max_zero_fraction"]: return "excess_zeros"
    if np.std(x) < cfg["min_std"]: return "flat_signal"
    return ""


def interpolate_zeros(signal):
    x = np.array(signal, dtype=np.float64, copy=True)
    zero = x == 0
    if zero.any() and not zero.all():
        t = np.arange(x.size)
        x = np.interp(t, t[~zero], x[~zero])
    return x


def clean_abp(signal, settings=None):
    import neurokit2 as nk
    cfg = settings_checked(settings)
    return np.asarray(nk.signal_filter(interpolate_zeros(signal), sampling_rate=cfg["sampling_rate"],
                      highcut=cfg["abp_lowpass_hz"], method="butterworth", order=cfg["abp_filter_order"]), dtype=float)


def repeated_max_plateaus(signal):
    x = np.asarray(signal)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all(): return False
    edges = np.diff(np.r_[False, x == np.max(x), False].astype(int))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return bool(np.count_nonzero(lengths >= 5) >= 3)


def ppg_features(clean, peaks, fs=FS):
    """Six unchanged starter features, then eight complete-pulse features."""
    clean, peaks = np.asarray(clean, dtype=float), np.asarray(peaks, dtype=int)
    intervals = np.diff(peaks) / fs
    amplitudes = [clean[b] - np.min(clean[a:b + 1]) for a, b in zip(peaks[:-1], peaks[1:])]
    initial = [np.mean(clean), np.std(clean), np.ptp(clean),
               np.median(intervals) if intervals.size else np.nan,
               np.median(60 / intervals) if intervals.size else np.nan,
               np.median(amplitudes) if amplitudes else np.nan]
    pulse_rows, heights = [], []
    feet = [a + int(np.argmin(clean[a:b + 1])) for a, b in zip(peaks[:-1], peaks[1:])]
    for left, peak, right in zip(feet[:-1], peaks[1:-1], feet[1:]):
        if not left < peak < right: continue
        segment = clean[left:right + 1]
        y = np.maximum(segment - np.linspace(clean[left], clean[right], len(segment)), 0)
        top = peak - left
        height = y[top]
        if height <= 1e-12 or np.ptp(y) <= 1e-12: continue
        rise, decay = (peak - left) / fs, (right - peak) / fs
        half = height / 2
        li = np.flatnonzero(y[:top] <= half)
        ri = np.flatnonzero(y[top + 1:] <= half) + top + 1
        width = np.nan
        if len(li) and len(ri):
            a, b = li[-1], ri[0]
            if y[a + 1] != y[a] and y[b] != y[b - 1]:
                l = a + (half-y[a])/(y[a+1]-y[a])
                r = b-1 + (half-y[b-1])/(y[b]-y[b-1])
                width = (r-l)/fs
        pulse_rows.append([rise, decay, height/rise, height/decay, width,
                           np.trapezoid(y, dx=1/fs), skew(y, bias=False)])
        heights.append(height)
    extra = np.full(8, np.nan)
    if pulse_rows:
        rows = np.asarray(pulse_rows)
        for j in range(7):
            finite = rows[:, j][np.isfinite(rows[:, j])]
            if finite.size: extra[j if j < 6 else 7] = np.median(finite)
        if len(heights) >= 2: extra[6] = np.std(heights) / np.mean(heights)
    return np.r_[initial, extra], len(pulse_rows)


def clean_window(ppg, abp, cleaning_config=None):
    """One NeuroKit pass; team ABP labels; all exclusions explicit."""
    import neurokit2 as nk
    cfg = settings_checked(cleaning_config)
    result = {"usable": False, "reason": "", "sbp": np.nan, "dbp": np.nan,
              "features": np.full(14, np.nan), "n_ppg_peaks": 0, "n_abp_peaks": 0,
              "n_abp_complete_beats": 0, "n_ppg_complete_pulses": 0,
              "saturation_flag": repeated_max_plateaus(abp), "warnings": []}
    for name, signal in (("ppg", ppg), ("abp", abp)):
        reason = raw_signal_reason(signal, cfg)
        if reason: result["reason"] = f"{name}:{reason}"; return result
    if cfg["reject_repeated_max_plateaus"] and result["saturation_flag"]:
        result["reason"] = "abp:repeated_maximum_saturation_plateaus"; return result
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            signals, _ = nk.ppg_process(interpolate_zeros(ppg), sampling_rate=cfg["sampling_rate"], method=cfg["ppg_method"])
            clean_ppg = signals["PPG_Clean"].to_numpy(dtype=float)
            peaks = np.flatnonzero(signals["PPG_Peaks"].to_numpy() > 0)
            clean_pressure = clean_abp(abp, cfg)
        result["warnings"] = sorted(set(str(w.message) for w in caught))
        if not np.isfinite(clean_ppg).all() or not np.isfinite(clean_pressure).all():
            raise ValueError("nonfinite cleaned signal")
        sbp, dbp, npk, nb = bp_utils.extract_bp_labels(clean_pressure, fs=cfg["sampling_rate"])
        result.update(sbp=sbp, dbp=dbp, n_ppg_peaks=len(peaks), n_abp_peaks=npk, n_abp_complete_beats=nb)
        if not np.isfinite([sbp, dbp]).all(): result["reason"] = "label:insufficient_complete_beats"; return result
        if sbp <= dbp: result["reason"] = "label:sbp_not_above_dbp"; return result
        for name, value in (("sbp", sbp), ("dbp", dbp)):
            bounds = cfg[f"{name}_range"]
            if bounds is not None and not bounds[0] <= value <= bounds[1]:
                result["reason"] = f"label:{name}_outside_range"; return result
        features, n_pulses = ppg_features(clean_ppg, peaks, cfg["sampling_rate"])
        result.update(usable=True, ppg=clean_ppg.astype(np.float32), abp=clean_pressure.astype(np.float32),
                      ppg_peaks=peaks, features=features, n_ppg_complete_pulses=n_pulses)
    except Exception as error:
        result["reason"] = f"processing:{type(error).__name__}:{error}"
    return result
