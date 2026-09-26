"""Audit raw PPG/ABP records and check the team's windowed data.

Run from the repository root with the project's virtual environment:
    .venv/Scripts/python.exe eda/audit.py

Prints a short summary. It does not rewrite the team's data or notebook.
Beat-based labels are an exploratory comparison, not approved training labels.
"""

import argparse
import hashlib
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import format as npy_format
from scipy.signal import find_peaks


ROOT = Path(__file__).resolve().parents[1]
FS = 125
WINDOW = 5 * FS
PARTS = [f"Part_{i}.mat" for i in range(1, 5)]
QUANTILES = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))},
    }


def exploratory_beat_labels(abp):
    """Median local extrema for a comparison sample, without clinical QC."""
    peaks, _ = find_peaks(abp, distance=40, prominence=10)
    troughs, _ = find_peaks(-abp, distance=40, prominence=10)
    if len(peaks) < 2 or len(troughs) < 2:
        return None
    return float(np.median(abp[peaks])), float(np.median(abp[troughs]))


def verify_processed(path, expected_rows, raw_hashes):
    """Read PPG/ABP from the NPZ in small row batches and compare every byte."""
    results = {}
    with zipfile.ZipFile(path) as archive:
        for channel in ("ppg", "abp"):
            with archive.open(f"{channel}.npy") as stream:
                version = npy_format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran_order, dtype = npy_format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"Unsupported NPY header version for {channel}: {version}")
                if shape != (expected_rows, WINDOW) or fortran_order or dtype != np.dtype("float64"):
                    raise ValueError(f"Unexpected {channel} array: shape={shape}, order={fortran_order}, dtype={dtype}")
                digest = hashlib.blake2b(digest_size=16)
                rows_read = 0
                nonfinite = 0
                windows_with_zero = 0
                baseline_labels = []
                while rows_read < expected_rows:
                    rows = min(128, expected_rows - rows_read)
                    block = stream.read(rows * WINDOW * dtype.itemsize)
                    if len(block) != rows * WINDOW * dtype.itemsize:
                        raise ValueError(f"Short read in {channel} array at row {rows_read}")
                    digest.update(block)
                    values = np.frombuffer(block, dtype=dtype).reshape(rows, WINDOW)
                    nonfinite += int((~np.isfinite(values)).sum())
                    if channel == "ppg":
                        windows_with_zero += int(np.any(values == 0, axis=1).sum())
                    else:
                        baseline_labels.extend(zip(values.max(axis=1).tolist(), values.min(axis=1).tolist()))
                    rows_read += rows
                if stream.read(1):
                    raise ValueError(f"Extra bytes after {channel} array")
                same = digest.hexdigest() == raw_hashes[channel]
                results[channel] = {
                    "shape": list(shape), "dtype": str(dtype),
                    "matches_raw_windows_exactly": same,
                    "nonfinite_samples": nonfinite,
                }
                if channel == "ppg":
                    results[channel]["windows_with_zero"] = windows_with_zero
                else:
                    labels = np.asarray(baseline_labels)
                    results[channel]["median_window_max_sbp"] = float(np.median(labels[:, 0]))
                    results[channel]["median_window_min_dbp"] = float(np.median(labels[:, 1]))
    return results


def audit(raw_dir):
    stats = Counter()
    zero_counts = []
    raw_window_digests = {channel: hashlib.blake2b(digest_size=16) for channel in ("ppg", "abp")}
    part_counts = {}
    lengths = []
    windows_per_record = []
    metric_names = ("sbp_max", "dbp_min", "pulse_pressure", "ppg_min", "ppg_max", "ppg_range", "ppg_std")
    metrics = {k: [] for k in metric_names}
    part_metrics = {filename: {k: [] for k in ("sbp_max", "dbp_min", "ppg_range")} for filename in PARTS}
    seen_record_hashes = {}
    duplicate_records = []
    examples = {k: [] for k in ("lowest_dbp", "highest_sbp", "lowest_ppg_range")}
    beat = {k: [] for k in ("sbp_difference", "dbp_difference", "beat_sbp", "beat_dbp")}
    sample_per_part = Counter()

    for part_number, filename in enumerate(PARTS, start=1):
        with h5py.File(raw_dir / filename, "r") as source:
            refs = source[f"Part_{part_number}"]
            part_counts[filename] = {"records": int(refs.shape[0]), "windows": 0}
            for record_index in range(refs.shape[0]):
                record = source[refs[record_index, 0]][()]
                if record.ndim != 2 or record.shape[1] != 3:
                    raise ValueError(f"Unexpected shape in {filename} record {record_index}: {record.shape}")
                digest = hashlib.blake2b(record.tobytes(), digest_size=16).hexdigest()
                source_key = [filename, int(record_index)]
                duplicate_of = seen_record_hashes.get(digest)
                if duplicate_of is not None:
                    duplicate_records.append({"first": duplicate_of, "duplicate": source_key})
                else:
                    seen_record_hashes[digest] = source_key
                n = record.shape[0]
                n_windows = n // WINDOW
                if duplicate_of is not None and n_windows:
                    stats["duplicate_copy_windows"] += n_windows
                    stats["cross_part_duplicate_records"] += duplicate_of[0] != filename
                lengths.append(n)
                windows_per_record.append(n_windows)
                stats["raw_samples_per_channel"] += n
                stats["dropped_tail_samples_per_channel"] += n % WINDOW
                stats["records_with_tail"] += (n % WINDOW) != 0
                stats["ppg_nonfinite_samples"] += int((~np.isfinite(record[:, 0])).sum())
                stats["abp_nonfinite_samples"] += int((~np.isfinite(record[:, 1])).sum())
                stats["ppg_nonpositive_samples"] += int((record[:, 0] <= 0).sum())
                stats["abp_nonpositive_samples"] += int((record[:, 1] <= 0).sum())
                stats["ppg_zero_samples"] += int((record[:, 0] == 0).sum())
                stats["abp_exactly_50_samples"] += int((record[:, 1] == 50).sum())
                stats["abp_at_least_199_9_samples"] += int((record[:, 1] >= 199.9).sum())
                if not n_windows:
                    continue

                windows = record[: n_windows * WINDOW].reshape(n_windows, WINDOW, 3)
                ppg = windows[:, :, 0]
                abp = windows[:, :, 1]
                raw_window_digests["ppg"].update(ppg.tobytes(order="C"))
                raw_window_digests["abp"].update(abp.tobytes(order="C"))
                sbp = abp.max(axis=1)
                dbp = abp.min(axis=1)
                ppg_min = ppg.min(axis=1)
                ppg_max = ppg.max(axis=1)
                ppg_range = ppg_max - ppg_min
                abp_range = sbp - dbp
                window_zero_counts = (ppg == 0).sum(axis=1)
                zero_counts.extend(window_zero_counts[window_zero_counts > 0].tolist())
                batch = {
                    "sbp_max": sbp, "dbp_min": dbp, "pulse_pressure": abp_range,
                    "ppg_min": ppg_min, "ppg_max": ppg_max, "ppg_range": ppg_range,
                    "ppg_std": ppg.std(axis=1),
                }
                for key, values in batch.items():
                    metrics[key].extend(values.tolist())
                    if key in part_metrics[filename]:
                        part_metrics[filename][key].extend(values.tolist())
                stats["windows"] += n_windows
                part_counts[filename]["windows"] += n_windows
                stats["windows_with_nonpositive_ppg"] += int((ppg_min <= 0).sum())
                stats["windows_with_nonpositive_abp"] += int((dbp <= 0).sum())
                stats["windows_with_zero_ppg"] += int((ppg_min == 0).sum())
                stats["windows_with_one_zero_ppg_sample"] += int((window_zero_counts == 1).sum())
                stats["windows_with_32_or_more_zero_ppg_samples"] += int((window_zero_counts >= 32).sum())
                stats["windows_with_dbp_exactly_50"] += int((dbp == 50).sum())
                stats["windows_with_sbp_at_least_199_9"] += int((sbp >= 199.9).sum())
                stats["flat_ppg_windows"] += int((ppg_range == 0).sum())
                stats["flat_abp_windows"] += int((abp_range == 0).sum())
                stats["abp_range_below_5_windows"] += int((abp_range < 5).sum())
                stats["sbp_above_250_windows"] += int((sbp > 250).sum())
                stats["dbp_below_40_windows"] += int((dbp < 40).sum())

                for key, values, order in (
                    ("lowest_dbp", dbp, 1),
                    ("highest_sbp", sbp, -1),
                    ("lowest_ppg_range", ppg_range, 1),
                ):
                    idx = int(np.argmin(values) if order == 1 else np.argmax(values))
                    examples[key].append({
                        "part": part_number, "record_index_zero_based": record_index,
                        "window_index_zero_based": idx, "value": float(values[idx]),
                    })

                # At least one window per record, plus every 40th local window.
                # The sample covers all four parts without loading the full NPZ.
                for window_index in range(0, n_windows, 40):
                    sample_per_part[filename] += 1
                    labels = exploratory_beat_labels(abp[window_index])
                    if labels is None:
                        stats["sampled_windows_without_two_extrema_of_each_type"] += 1
                        continue
                    beat["beat_sbp"].append(labels[0])
                    beat["beat_dbp"].append(labels[1])
                    beat["sbp_difference"].append(float(sbp[window_index] - labels[0]))
                    beat["dbp_difference"].append(float(dbp[window_index] - labels[1]))

    stats["records"] = len(lengths)
    stats["duplicate_records"] = len(duplicate_records)
    stats["beat_sample_windows"] = sum(sample_per_part.values())
    stats["beat_sample_usable_windows"] = len(beat["beat_sbp"])
    for key in examples:
        examples[key] = sorted(examples[key], key=lambda row: row["value"], reverse=key == "highest_sbp")[:5]
    return {
        "method": {
            "sample_rate_hz": FS, "window_samples": WINDOW,
            "windowing": "complete nonoverlapping windows per source record; tails dropped",
            "beat_sample": "first and then every 40th window within each record",
            "beat_peaks": "scipy.find_peaks; minimum distance 40 samples; prominence 10 mmHg; at least two peaks and two troughs",
            "indices": "zero based record and window indices within each file",
        },
        "counts": dict(stats),
        "parts": part_counts,
        "record_lengths": distribution(lengths),
        "windows_per_record": distribution(windows_per_record),
        "window_metrics": {key: distribution(values) for key, values in metrics.items()},
        "zero_samples_per_affected_window": distribution(zero_counts),
        "part_metrics": {part: {key: distribution(values) for key, values in part_metrics[part].items()} for part in PARTS},
        "beat_comparison": {key: distribution(values) for key, values in beat.items()},
        "beat_sample_per_part": dict(sample_per_part),
        "examples": examples,
        "duplicate_record_examples": duplicate_records[:10],
        "raw_window_hashes": {channel: digest.hexdigest() for channel, digest in raw_window_digests.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--processed-path", type=Path, default=ROOT / "data" / "processed_dataset.npz")
    args = parser.parse_args()
    result = audit(args.raw_dir)
    if args.processed_path.exists():
        result["processed_archive_check"] = verify_processed(
            args.processed_path, result["counts"]["windows"], result["raw_window_hashes"]
        )
    else:
        result["processed_archive_check"] = {"status": "archive not found; raw audit only"}
    counts = result["counts"]
    print(f"Source records: {counts['records']:,}; five-second windows: {counts['windows']:,}")
    print(f"PPG/ABP nonfinite samples: {counts['ppg_nonfinite_samples']}/{counts['abp_nonfinite_samples']}")
    print(f"Windows with any zero PPG sample: {counts['windows_with_zero_ppg']:,}")
    print(f"  One zero sample: {counts['windows_with_one_zero_ppg_sample']:,}; 32 or more: {counts['windows_with_32_or_more_zero_ppg_samples']:,}")
    print(f"Duplicate source-record copies: {counts['duplicate_records']}; cross-part: {counts['cross_part_duplicate_records']}; windows in duplicate copies: {counts['duplicate_copy_windows']:,}")
    cross_part_example = next((pair for pair in result["duplicate_record_examples"] if pair["first"][0] != pair["duplicate"][0]), None)
    if cross_part_example:
        first, copy = cross_part_example["first"], cross_part_example["duplicate"]
        print(f"  Example (zero-based): {first[0]} record {first[1]} = {copy[0]} record {copy[1]}")
    labels = result["window_metrics"]
    print(f"Baseline label medians (mmHg): SBP {labels['sbp_max']['quantiles']['0.5']:.2f}; DBP {labels['dbp_min']['quantiles']['0.5']:.2f}")
    beat = result["beat_comparison"]
    print(f"Exploratory whole-window minus beat-median labels (mmHg): SBP {beat['sbp_difference']['quantiles']['0.5']:+.2f}; DBP {beat['dbp_difference']['quantiles']['0.5']:+.2f}")
    check = result["processed_archive_check"]
    if "ppg" in check:
        print(f"Local .npz matches raw-derived PPG/ABP windows: {check['ppg']['matches_raw_windows_exactly']}/{check['abp']['matches_raw_windows_exactly']}")
    else:
        print(check["status"])


if __name__ == "__main__":
    main()
