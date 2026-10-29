"""
Shared blood pressure label extraction utilities.

This module provides functions to extract systolic (SBP) and diastolic (DBP)
blood pressure values from ABP (arterial blood pressure) signal windows using
peak detection via scipy.signal.find_peaks.

All notebooks should import from this module to maintain a single source of
truth for the extraction logic.
"""

import numpy as np
from scipy.signal import find_peaks

FS = 125                    # sampling rate (Hz)
MIN_PROMINENCE = 10.0       # mmHg; smallest prominence a systolic peak may have
RELATIVE_PROMINENCE = 0.5   # fraction of the window's pulse amplitude a peak must stand out by
MIN_PEAK_DISTANCE_S = 0.33  # seconds between systolic peaks (about 180 bpm max)
MIN_BEATS = 2               # fewer complete beats -> NaN labels


def extract_bp_labels(abp_window, fs=FS):
    """Extract SBP and DBP from a single ABP window using complete heartbeats.

    1. Systolic peaks must stand out by at least
       max(10 mmHg, 0.5 x (95th - 5th percentile of the window)) and be at
       least 0.33 s apart. This skips small secondary bumps such as the
       dicrotic notch.
    2. A complete beat is a peak plus the lowest point between it and the
       previous peak (its DBP). The first peak has no previous peak, so it
       is not a complete beat.
    3. SBP and DBP are the medians over complete beats.

    Parameters
    ----------
    abp_window : array-like, shape (n_samples,)
        A single ABP signal window (e.g., 625 samples at 125 Hz).
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    sbp : float
        Median systolic peak over complete beats, or NaN if fewer than 2 beats.
    dbp : float
        Median diastolic low over complete beats, or NaN if fewer than 2 beats.
    n_peaks : int
        Number of systolic peaks detected.
    n_beats : int
        Number of complete beats used for the labels.
    """
    abp_window = np.asarray(abp_window, dtype=np.float64)

    low, high = np.percentile(abp_window, [5, 95])
    prominence = max(MIN_PROMINENCE, RELATIVE_PROMINENCE * (high - low))
    distance = int(round(MIN_PEAK_DISTANCE_S * fs))
    peaks, _ = find_peaks(abp_window, prominence=prominence, distance=distance)

    beat_sbp = []
    beat_dbp = []
    for previous_peak, peak in zip(peaks[:-1], peaks[1:]):
        beat_sbp.append(abp_window[peak])
        beat_dbp.append(abp_window[previous_peak:peak].min())

    n_peaks = len(peaks)
    n_beats = len(beat_sbp)
    if n_beats < MIN_BEATS:
        return np.nan, np.nan, n_peaks, n_beats

    sbp = float(np.median(beat_sbp))
    dbp = float(np.median(beat_dbp))
    return sbp, dbp, n_peaks, n_beats


def extract_bp_labels_v1(abp_window):
    """Original method, kept for comparison: mean of all peaks and all valleys.

    Counts every local maximum as a peak, so small secondary bumps are
    included. Returns (sbp, dbp, n_peaks, n_valleys).
    """
    peaks, _ = find_peaks(abp_window)
    valleys, _ = find_peaks(-abp_window)

    sbp = np.mean(abp_window[peaks]) if len(peaks) > 0 else np.nan
    dbp = np.mean(abp_window[valleys]) if len(valleys) > 0 else np.nan
    n_peaks = len(peaks)
    n_valleys = len(valleys)

    return sbp, dbp, n_peaks, n_valleys


def extract_bp_labels_batch(abp_dataset):
    """Extract SBP/DBP for all windows in a dataset.

    Parameters
    ----------
    abp_dataset : numpy.ndarray, shape (n_windows, n_samples)
        2D array where each row is one ABP signal window.

    Returns
    -------
    sbp_arr : numpy.ndarray, shape (n_windows,)
    dbp_arr : numpy.ndarray, shape (n_windows,)
    n_peaks_arr : numpy.ndarray, shape (n_windows,)
    n_beats_arr : numpy.ndarray, shape (n_windows,)
    """
    n_windows = abp_dataset.shape[0]
    sbp_arr = np.full(n_windows, np.nan)
    dbp_arr = np.full(n_windows, np.nan)
    n_peaks_arr = np.zeros(n_windows, dtype=int)
    n_beats_arr = np.zeros(n_windows, dtype=int)

    for i in range(n_windows):
        if i % 100_000 == 0:
            print(f"  Processing window {i:,} / {n_windows:,} ...")
        sbp_arr[i], dbp_arr[i], n_peaks_arr[i], n_beats_arr[i] = extract_bp_labels(
            abp_dataset[i]
        )

    print("Done!")
    return sbp_arr, dbp_arr, n_peaks_arr, n_beats_arr
