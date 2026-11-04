"""
Load Part_1.mat - Part_4.mat (MATLAB v7.3 / HDF5 format) and split every recording
into non-overlapping 5-second (625-sample @ 125 Hz) windows, producing three aligned
datasets: PPG, ABP, and ECG. Row i in each dataset is the same time window, so the
datasets can be joined by index (e.g. ppg_dataset[i] <-> abp_dataset[i]).

Column counts (samples per recording) VARY across records with no fixed pattern -
across all 4 files (12,000 records total) N ranges from 1,000 to 74,000.
Only ~16% of records (1,918 / 12,000) have a column count
that is an exact multiple of 625, so most recordings produce one or more full 625-sample
windows plus a leftover remainder. This script drops leftover samples that do not fill
a complete 625-sample window.
"""

from pathlib import Path

import h5py
import neurokit2 as nk
import numpy as np

DATA_DIR = Path(__file__).parent
RAW_DATA_DIR = DATA_DIR / "raw"
MAT_FILES = ["Part_1.mat", "Part_2.mat", "Part_3.mat", "Part_4.mat"]
OUTPUT_PATH = DATA_DIR / "processed_dataset.npz"

WINDOW_SIZE = 625  # 125 Hz * 5 seconds
SAMPLING_RATE = 125

# Column order within each (N, 3) record, confirmed against notes.md / value ranges.
PPG_COL, ABP_COL, ECG_COL = 0, 1, 2


def load_mat_records(filepath):
    """Yield each recording in a v7.3 .mat file as an (N, 3) array of PPG/ABP/ECG samples."""
    with h5py.File(filepath, "r") as f:
        var_name = filepath.stem  # e.g. "Part_1", matches the top-level HDF5 key
        refs = f[var_name]
        for i in range(refs.shape[0]):
            yield f[refs[i, 0]][()]


def split_into_windows(record, window_size=WINDOW_SIZE):
    """Split an (N, 3) recording into non-overlapping (window_size, 3) windows, dropping any remainder."""
    n_samples = record.shape[0]
    n_windows = n_samples // window_size
    usable = record[: n_windows * window_size]
    return usable.reshape(n_windows, window_size, 3)


def clean_ecg_signal(ecg, sampling_rate=SAMPLING_RATE):
    """Clean one continuous ECG recording with NeuroKit2.

    NeuroKit's ECG filters require finite input.  The source data contains a
    small number of missing ECG samples, so interpolate those samples before
    filtering.  Cleaning the continuous recording before windowing avoids
    introducing a filter edge at every five-second window.
    """
    ecg = np.asarray(ecg, dtype=float)
    finite = np.isfinite(ecg)
    if not finite.any():
        raise ValueError("ECG recording contains no finite samples")
    if not finite.all():
        sample_indices = np.arange(ecg.size)
        ecg = np.interp(sample_indices, sample_indices[finite], ecg[finite])
    return np.asarray(nk.ecg_clean(ecg, sampling_rate=sampling_rate), dtype=float)


def build_datasets(data_dir=RAW_DATA_DIR, mat_files=MAT_FILES, window_size=WINDOW_SIZE):
    """Load all mat files and return (ppg, abp, ecg) arrays of shape (n_windows, window_size)."""
    ppg_windows, abp_windows, ecg_windows = [], [], []

    for filename in mat_files:
        filepath = data_dir / filename
        for record in load_mat_records(filepath):
            record = np.asarray(record, dtype=float).copy()
            record[:, ECG_COL] = clean_ecg_signal(record[:, ECG_COL])
            windows = split_into_windows(record, window_size)
            if windows.shape[0] == 0:
                continue
            ppg_windows.append(windows[:, :, PPG_COL])
            abp_windows.append(windows[:, :, ABP_COL])
            ecg_windows.append(windows[:, :, ECG_COL])

    ppg_dataset = np.concatenate(ppg_windows, axis=0)
    abp_dataset = np.concatenate(abp_windows, axis=0)
    ecg_dataset = np.concatenate(ecg_windows, axis=0)
    return ppg_dataset, abp_dataset, ecg_dataset


def save_datasets(ppg_dataset, abp_dataset, ecg_dataset, output_path=OUTPUT_PATH):
    """Save the PPG, ABP, and ECG datasets to a single compressed .npz file."""
    np.savez_compressed(output_path, ppg=ppg_dataset, abp=abp_dataset, ecg=ecg_dataset)


def main():
    """Build the PPG/ABP/ECG window datasets from all mat files and save them to disk."""
    ppg_dataset, abp_dataset, ecg_dataset = build_datasets()
    print(f"PPG dataset shape: {ppg_dataset.shape}")
    print(f"ABP dataset shape: {abp_dataset.shape}")
    print(f"ECG dataset shape: {ecg_dataset.shape}")
    save_datasets(ppg_dataset, abp_dataset, ecg_dataset)
    print(f"Saved datasets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
