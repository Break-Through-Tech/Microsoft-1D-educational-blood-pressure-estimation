# Data Structure And Processing Notes

## Dataset Summary

The project uses the UCI Cuff-Less Blood Pressure Estimation dataset. It contains 12,000 physiological signal records split across `Part_1.mat` through `Part_4.mat`. Each record contains synchronized PPG, ABP, and ECG signals sampled at 125 Hz.

The current project goal is to train a regression model that receives a PPG signal and estimates systolic blood pressure (SBP) and diastolic blood pressure (DBP). ABP provides the reference labels during training. ECG is retained for possible future experiments but is not required by the current scope.

## Raw Record Structure

Conceptually, one raw record is a `3 x N` MATLAB matrix:

```text
                 time/sample position
             0        1        2       ...      N - 1
PPG row      ppg_0    ppg_1    ppg_2   ...      ppg_N-1
ABP row      abp_0    abp_1    abp_2   ...      abp_N-1
ECG row      ecg_0    ecg_1    ecg_2   ...      ecg_N-1
```

When read through `h5py`, each record appears as an `(N, 3)` array, so the loader uses columns `0`, `1`, and `2` for PPG, ABP, and ECG respectively.

`N` varies by record. Inspection of all four files found record lengths from 1,000 to 74,000 samples. The three channels in a record share the same `N`, which keeps them aligned at every time position.

## Observed Data Validation

A full record-by-record inspection found:

```text
Records checked:             12,000
Records with invalid shape:       0
PPG NaN values:                   0
ABP NaN values:                   0
ECG NaN values:                  16 (within one record)
Infinite values:                  0
```

The current PPG and ABP data therefore have no missing numeric values. The known missing values occur only in ECG, which is outside the current model scope. The preprocessing pipeline should still validate shapes and finite values rather than assume future copies of the data are identical.

## Five-Second Windowing

At 125 samples per second, one five-second window contains:

```text
125 samples/second x 5 seconds = 625 samples
```

The loader divides each record into non-overlapping complete windows. For example:

```text
Record with 1,000 synchronized samples

samples 0-624     -> one complete 625-sample window
samples 625-999   -> 375-sample remainder, dropped
```

Remainders are dropped from all channels together. They are not padded because artificial padding could introduce fake physiological patterns. Only 1,918 of the 12,000 records have lengths exactly divisible by 625.

After processing:

```text
528,828 windows x 625 samples/window
= 330,517,500 retained samples per channel

528,828 windows x 5 seconds/window
= 2,644,140 seconds
= 734.483 hours
= 30.603 days of synchronized recording time
```

The raw records contain 333,690,000 time positions per channel. Windowing drops 3,172,500 trailing positions across all records, equivalent to about 7.05 hours.

## Processed Arrays

`python data/load_files.py` writes archive format 2 to `data/processed_dataset.npz`. Its three original aligned arrays remain:

```text
ppg: (528828, 625)
abp: (528828, 625)
ecg: (528828, 625)
```

Each row is one five-second example and each column is a sample position from `0` through `624`:

```text
                   sample position within the window
             0        1        2       ...      624
window 0     ppg      ppg      ppg     ...      ppg
window 1     ppg      ppg      ppg     ...      ppg
window 2     ppg      ppg      ppg     ...      ppg
```

For any row index `i`, `ppg[i]`, `abp[i]`, and `ecg[i]` represent the same five-second period.

The archive also contains `window_ids`, `source_record_ids`, and `split_assignments`, each with 528,828 entries in that same row order. `format_version` is the scalar 2. `preparation_json` stores the complete recording map, exact-duplicate relationships, dropped tails, per-channel nonfinite counts, source checksums, split configuration, and package versions. None of these fields is a prediction feature.

IDs use `Part_1:record_1055:window_0`. Its source ID is `Part_1:record_1055`; local window 0 uses samples `[0, 625)`. Window `j` uses `[j * 625, (j + 1) * 625)`. Each recording's `first_npz_row` identifies its original archive offset, including duplicate copies.

Exact-record comparison includes shape, dtype, and all three channels. Keep the earliest source copy for modeling; the other 14 records / 700 windows have split `excluded`. Their signals remain in the archive for existing notebook reads. The split loader returns 11,986 recordings / 528,128 windows before signal cleaning or label/feature failures.

Canonical recordings are allocated approximately 70/15/15 using seed 2026 and largest-remainder integer rounding. Identifiable examples already reviewed in the EDA notebooks, including historical saved examples, are listed in `configs/split.json`. Their canonical recordings are forced into training, counting toward the training allocation. This is recording separation, not verified patient separation; shifted/partial overlaps remain possible. Overall EDA happened before the split was reserved.

The builder reads one recording at a time and streams NPY entries into a temporary compressed NPZ. Before replacing an existing archive, it checks every channel's shape, dtype, and full waveform-byte digest against the old file, validates metadata, and confirms source checksums have not changed. Failure leaves the existing archive intact. The command needs temporary space for another compressed archive.

## Accessing a split

From the repository root:

```python
from data.dataset_splits import iter_split_windows

for batch in iter_split_windows("train", batch_size=1024):
    ppg = batch.ppg
    abp = batch.abp
    window_ids = batch.window_ids
    source_ids = batch.source_record_ids
    # Extract PPG features and ABP labels separately; retain window_ids with both.
```

The loader scans the archive sequentially with bounded waveform memory, returns at most 1,024 selected windows per batch, and excludes duplicate copies. It does not clean signals or derive labels. Smaller batches can result when original rows belong to other splits. Reading a complete split scans all PPG/ABP rows; this prioritizes compatibility and memory use over random access. Learn normalization/imputation only from training rows, then apply the fitted transformation to validation/test. Keep test out of feature selection and tuning.

Run `python data/example_split_usage.py --split train --limit 1000` for an example preserving IDs through a finite-signal filter. Use `validation` or `test` to exercise those loaders; the example calculates no model scores. Legacy archives without metadata must be rebuilt. Notebook code that reads the entire `ppg` array still sees every original row and must adopt this loader before claiming split-based results.

## PPG, ABP, And Blood Pressure Labels

PPG is an optical signal that tracks changes in blood volume near the fingertip. Its amplitude values are signal measurements, not pressure in mmHg. PPG does contain systolic and diastolic phases and useful pulse-shape information, but its maxima and minima are not automatically SBP and DBP measurements.

ABP is an invasively measured arterial pressure waveform in mmHg. Its peaks and valleys provide the reference SBP and DBP labels. There is no direct unit conversion from PPG amplitude to mmHg in the current code. Instead, supervised learning uses many synchronized examples to learn a relationship from PPG morphology to ABP-derived pressure labels.

The intended model example is:

```text
Input X:  625 PPG amplitude samples
Target y: [SBP in mmHg, DBP in mmHg] derived from the aligned ABP window
```

During model training and evaluation, ABP supplies the correct answers. At inference time, the intended model receives only a new PPG window and estimates SBP and DBP.

## Current Label Extraction

The current notebooks call `data/bp_utils.py`, which uses `scipy.signal.find_peaks`:

```text
SBP label = mean of detected ABP peak values
DBP label = mean of detected ABP valley values
```

Missing peaks or valleys produce NaN. This implementation does not pair complete beats, set detection parameters, or validate signal quality. Whole-window maxima/minima remain a separate proposed baseline export. The archive contains neither label method; callers must calculate and version labels using aligned ABP. New label methods require evidence before adoption.

`notebooks/read_data.ipynb` currently drops windows with more than 10% zeros in either signal and replaces remaining zeros with each row's mean of nonzero samples, after calculating labels. These changes are in notebook memory only. The split preparation does not apply or endorse those rules; zero semantics and suitable cleaning remain open.

## Pipeline Roles

```text
Raw MATLAB records
        |
        v
Load, window, identify duplicates, save splits  data/load_files.py
        |
        v
Select train/validation/test windows             iter_split_windows
        |
        v
Clean signals and derive labels from ABP         teammates' preprocessing
        |
        v
Clean or extract features from PPG               feature engineering (to finalize)
        |
        v
Train PPG -> SBP/DBP regression models           model development
```

The loader should focus on reading, alignment, windowing, provenance, and basic validity checks. Signal cleaning and ABP peak/valley label extraction belong in the preprocessing stage after windowing and before model training.

## Questions Requiring Team Clarification

- Confirm whether a source record can be treated as a unique patient. The available files clearly identify records, but patient identity must not be assumed without supporting metadata.
- Define ABP beat detection, noisy-window rejection, and physiologically plausible SBP/DBP limits.
- Decide whether labels use whole-window max/min, mean valid beat extrema, or median valid beat extrema.
- Decide how PPG is normalized and whether models use engineered features, raw windows, or both.

Project-wide accepted and pending choices are tracked in the root [DECISIONS.md](../DECISIONS.md).
