# PPG/ABP exploratory data analysis

Run `.venv\Scripts\python.exe eda/kushagra_eda.py` from the repository root to regenerate [the machine-readable summary](kushagra_eda_summary.json). The script reads all four local raw MATLAB v7.3 files, one record at a time, then checks the existing processed archive in small batches. It writes only the summary JSON. This report describes that run; it does not establish clinical signal-quality thresholds or finalized labels.

## What is being processed

- Each `data/raw/Part_*.mat` file contains a `(3000, 1)` HDF5 array of object references. Dereferencing one cell with `h5py` yields an `(N, 3)` `float64` record. Columns 0 and 1 are synchronized PPG and ABP. Column 2 is ECG and was excluded from this EDA.
- `data/load_files.py` currently creates complete, nonoverlapping 625-sample windows (5 seconds at 125 Hz), drops each record's final short tail, and writes aligned `ppg`, `abp`, and `ecg` arrays to the Git-ignored `data/processed_dataset.npz`. The local archive has three `(528828, 625)` `float64` arrays. It contains no labels or source-record identifiers.
- `notebooks/read_data.ipynb` currently derives SBP from the maximum and DBP from the minimum of each ABP window, then joins these labels to PPG by row position. ABP is the label source, not a model input for the intended PPG-only model.
- Signal cleaning, ABP beat detection, label validation, and PPG feature extraction belong after loading/windowing and before modeling. Their rules remain open in `DECISIONS.md`; this audit does not rewrite the team's files or regenerate the processed archive.

The raw recordings have different lengths, sometimes called *ragged*. That prevents stacking them directly into one fixed-width model input, but they can still be read and analyzed one record at a time. The processed archive is the appropriate fixed-width input for initial modeling. The raw files were useful here because they still identify which source record produced each window.

A **source record** is one original continuous recording referenced by a cell in a raw `.mat` file. For example, a 12-second source record produces two complete five-second windows and leaves a two-second tail. The record is not proven to equal one unique patient. **Provenance** means retaining the part, source-record index, and window index for each resulting window.

**Processed archive verification:** The audit compared the complete ordered bytes of all 528,828 PPG windows and all 528,828 ABP windows in `processed_dataset.npz` against windows independently produced from the raw records. Both channels match exactly. The archive therefore represents the loader's documented windowing correctly for the two channels in current scope. Direct checks on the archive also found 5,407 PPG windows with zero samples, no nonfinite PPG/ABP samples, and median max/min ABP labels of 131.10/62.03 mmHg, matching the raw audit. This verification does not assess ECG or decide whether the current labels are the best choice.

## How this fits the team's work (reviewed after pulling `main` on 2026-09-26)

- `data/load_files.py` is the team's **loading/windowing** step. It turns the variable-length raw records into aligned, fixed-size PPG/ABP/ECG arrays. The existing local processed archive is consistent with it for PPG and ABP. This step does not create final SBP/DBP labels or a cleaned modeling dataset.
- `notebooks/read_data.ipynb` is the merged `jliu/clean-data` work. It reads the processed archive, adds whole-window ABP max/min labels, and experiments with zero-PPG handling. It does **not** train a model or save a cleaned dataset. Its saved output reports 5,407 PPG rows containing zero and 53 rows dropped by a greater-than-5%-of-627 rule. The subsequent `fillna` acts on NaNs rather than zeros, and its chained assignment emits a pandas warning in the saved output. After row deletion, using the original row labels with `iloc` can select different rows; the row mean also includes the two pressure-label columns. Treat these cells as an experiment, not an approved cleaning pipeline.
- `eda/kushagra_eda.py` and this report audit the source data, verify the processed PPG/ABP arrays, and compare provisional labels on a defined sample. They do not replace the loader or produce a new cleaned dataset.

## Full-data audit

| Measure | Observation |
| --- | ---: |
| Source records | 12,000; 3,000 per part |
| Record length | 1,000–74,000 samples; median 20,000 |
| Complete five-second windows | 528,828 |
| Windows per record | 1–118; median 32 |
| Trailing samples dropped, per channel | 3,172,500 of 333,690,000 (0.95%) |
| Records with a dropped tail | 10,082 |
| PPG or ABP nonfinite samples | 0 |
| Zero PPG samples | 45,654; no negative PPG samples |
| Windows containing a zero PPG sample | 5,407 (1.02%) |
| Nonpositive ABP samples | 0 |
| Entirely flat PPG or ABP windows | 0 |
| Source records with a matching full-record hash | 14 duplicate occurrences |

These counts cover the raw PPG and ABP channels, including samples in tails for sample-level checks. Window-level checks cover only complete windows. A zero in PPG is a signal value in this dataset, not a missing-value marker established by the source metadata; discarding every such window needs a reason. The existing notebook's `<= 0` check identifies the same 5,407 windows, but its later `627`-entry count includes the two joined labels, whereas a PPG waveform has 625 samples.

The duplicate scan hashes each complete three-channel record, including ECG, with BLAKE2b. Its 14 matches include adjacent records within Part 3 and records repeated between Part 1 and Part 3 (see JSON examples). A cryptographic hash match is strong evidence of identical record contents; the audit did not infer patient identity from these matches.

## Current whole-window labels and signal ranges

| Quantity across 528,828 windows | Minimum | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: | ---: |
| ABP maximum / baseline SBP (mmHg) | 63.36 | 131.10 | 174.42 | 199.99 |
| ABP minimum / baseline DBP (mmHg) | 50.00 | 62.03 | 84.72 | 184.51 |
| ABP window range (mmHg) | 5.13 | 67.50 | 103.26 | 148.49 |
| PPG window range (native amplitude) | 0.098 | 2.075 | 2.663 | 4.002 |

There are 38 complete windows with a DBP minimum exactly 50 mmHg, and 16 with an SBP maximum at least 199.9 mmHg. The raw ABP channel has 814 exact 50 readings and 22 readings at least 199.9. These values could reflect dataset selection or measurement bounds; the audit cannot establish clipping from counts alone. Part-wise median baseline SBP varies from 127.12 to 134.86 mmHg, and median baseline DBP from 59.94 to 63.84 mmHg, so preserve part provenance when comparing models.

## Exploratory beat-extrema comparison

For a reproducible comparison sample, the script takes window 0 and every 40th window *within each record* (20,336 windows). `scipy.signal.find_peaks` detects ABP peaks and inverted troughs with a minimum 40-sample separation and 10 mmHg prominence. A window needs at least two of each; 20,263 meet that simple condition, while 73 do not. The comparison label is the median detected peak or trough amplitude. Peaks and troughs have not yet been paired into valid beats, and no waveform-quality review was applied.

| Difference: whole-window extreme minus median detected extrema | Median | 95th percentile | Most extreme |
| --- | ---: | ---: | ---: |
| SBP (mmHg) | +2.35 | +9.23 | +94.00 |
| DBP (mmHg) | -1.76 | -0.34 | -55.85 |

The direction is expected mathematically: a full-window maximum is at least as high as a median of local peaks, and the minimum is at most as low as a median of local troughs. The large tails show that labeling choice matters for some windows, but this comparison does **not** show that the median-extrema method is clinically correct. The 73 rejected comparison windows are useful candidates for waveform inspection, not automatically invalid records.

## Processing implications

1. Add `part_number`, zero-based `record_index`, and `window_index` to a future **versioned** processed dataset. Keep the current `data/processed_dataset.npz` available for team work until a replacement and its consumer code are agreed. Record identity is not confirmed patient identity.
2. Resolve repeated source records before any group-aware split; grouping only by `(part, record)` still leaves the cross-part copies in separate groups. Derive groups from source-record identity and the duplicate equivalence class, then check for any other patient linkage information.
3. Keep full-window max/min as a documented baseline. Compare it with a beat-based candidate after inspecting waveforms, pairing valid beats, and choosing quality rules on training data only. Record how many windows each rule retains and how the SBP/DBP distributions change.
4. Treat PPG zeros, very small PPG ranges, and boundary ABP values as review candidates. Do not silently drop or impute them. Fit any normalization using training groups only; apply the fitted transform to validation and test groups.
5. The notebook's row deletion and `627`-entry heuristic should not be adopted as shared preprocessing without revisiting its assumptions. Its subsequent `fillna` call cannot replace zero PPG values because they are not NaNs; the saved notebook output also shows a pandas chained-assignment error. After dropping rows by label, `iloc[row]` can additionally refer to a different position. Keep final processing in reusable Python code so notebooks can call the same rules.

This is an educational dataset audit. The proposed label and quality checks need further validation before they are used for model evaluation or any health-related interpretation.
