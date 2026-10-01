"""
Shared blood pressure label extraction utilities.

This module provides functions to extract systolic (SBP) and diastolic (DBP)
blood pressure values from ABP (arterial blood pressure) signal windows using
peak/valley detection via scipy.signal.find_peaks.

All notebooks should import from this module to maintain a single source of
truth for the extraction logic.
"""

import numpy as np
from scipy.signal import find_peaks


def extract_bp_labels(abp_window):
    """Extract SBP (mean of peaks) and DBP (mean of valleys) from a single ABP window.

    Parameters
    ----------
    abp_window : array-like, shape (n_samples,)
        A single ABP signal window (e.g., 625 samples at 125 Hz).

    Returns
    -------
    sbp : float
        Systolic blood pressure (mean of detected peak values), or NaN if no peaks found.
    dbp : float
        Diastolic blood pressure (mean of detected valley values), or NaN if no valleys found.
    n_peaks : int
        Number of peaks detected.
    n_valleys : int
        Number of valleys detected.
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
    n_valleys_arr : numpy.ndarray, shape (n_windows,)
    """
    n_windows = abp_dataset.shape[0]
    sbp_arr = np.full(n_windows, np.nan)
    dbp_arr = np.full(n_windows, np.nan)
    n_peaks_arr = np.zeros(n_windows, dtype=int)
    n_valleys_arr = np.zeros(n_windows, dtype=int)

    for i in range(n_windows):
        if i % 100_000 == 0:
            print(f"  Processing window {i:,} / {n_windows:,} ...")
        sbp_arr[i], dbp_arr[i], n_peaks_arr[i], n_valleys_arr[i] = extract_bp_labels(
            abp_dataset[i]
        )

    print("Done!")
    return sbp_arr, dbp_arr, n_peaks_arr, n_valleys_arr
