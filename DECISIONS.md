# Project Decision Log

This file records important loading, preprocessing, EDA, and modeling decisions. Items marked **Open** require team or advisor clarification before they should be treated as final.

## Data Loading

| Status | Decision | Reason or follow-up |
|---|---|---|
| Accepted | Store the original UCI files under `data/raw/` using the names `Part_1.mat` through `Part_4.mat`. | Keeps immutable source data separate from code and generated artifacts. |
| Accepted | Keep raw `.mat` files and `data/processed_dataset.npz` out of Git. | These are large local data artifacts. |
| Accepted | Use `h5py` to read the MATLAB v7.3 files. | MATLAB v7.3 uses HDF5 internally. |
| Accepted | Save aligned PPG, ABP, and ECG arrays in `data/processed_dataset.npz`. | Preserves synchronized channels for current and possible future experiments. |
| Accepted | Use complete, non-overlapping five-second windows of 625 samples at 125 Hz. | Matches the challenge guidance and creates fixed-length examples. |
| Accepted | Drop trailing samples that cannot fill a complete window; do not pad them. | Padding could create artificial waveform patterns. |
| Open | Preserve `part_number` and `record_index` for every output window. | Needed to group related windows and avoid train/test leakage. Determine the metadata format before regenerating data. |
| Open | Confirm whether source records correspond one-to-one with patients. | Do not claim patient-level splitting without reliable identity metadata. |

## Data Processing

| Status | Decision | Reason or follow-up |
|---|---|---|
| Accepted | Treat PPG as the primary model input and ABP as the source of reference SBP/DBP labels. | PPG amplitude is not pressure in mmHg; ABP is the synchronized pressure waveform. |
| Accepted | Retain ECG in the processed archive, but exclude it from the initial project scope. | The challenge overview says ECG is not required for the initial model. |
| Current baseline | Derive one SBP label from the maximum and one DBP label from the minimum of each ABP window. | Implemented in `read_data.ipynb`; useful as a baseline but sensitive to artifacts. |
| Open | Replace or compare whole-window max/min with beat-based ABP label extraction. | Detect peaks and valleys for each beat, reject invalid beats, then aggregate valid SBP/DBP values. |
| Open | Choose mean versus median aggregation for valid beat-level labels. | Median is more robust to remaining outliers; this must be evaluated. |
| Open | Define noisy-window and physiologically implausible-value rules. | Requires documented thresholds and signal-quality criteria rather than guesses. |
| Open | Choose PPG cleaning, normalization, and feature extraction methods. | Compare engineered features with models that learn from raw windows. |

## Exploratory Data Analysis

| Status | Decision | Reason or follow-up |
|---|---|---|
| Accepted | Use notebooks for visual inspection and EDA, while moving finalized repeatable preprocessing into reusable Python code. | Keeps exploration flexible and the final pipeline reproducible. |
| Open | Plot aligned PPG, ABP, and ECG windows and inspect representative normal, noisy, and extreme examples. | Confirms alignment and informs quality-control rules. |
| Open | Compare label distributions produced by whole-window and beat-based extraction. | Quantifies how much the labeling choice changes the target data. |
| Open | Audit record lengths, windows per record, missing values, duplicates, and outliers in a reproducible report. | Establishes a shared data-quality baseline. |

## Modeling And Evaluation

| Status | Decision | Reason or follow-up |
|---|---|---|
| Accepted | Frame the initial task as supervised regression from PPG to SBP and DBP. | Matches the challenge goal. |
| Accepted | Evaluate both SBP and DBP using MAE and RMSE, with at least two models and a naive baseline. | Matches the challenge success criteria. |
| Open | Define a group-aware train/validation/test split using the strongest available provenance key. | Neighboring windows from one source record must not be spread across splits. |
| Open | Decide whether the first models use engineered PPG features, raw 625-sample windows, or both. | The choice affects preprocessing, interpretability, and model families. |

## Confirmed Dataset Findings

- All 12,000 records have the expected three-channel shape when loaded through HDF5.
- Record lengths range from 1,000 to 74,000 synchronized samples.
- PPG and ABP contain no observed `NaN` or infinite values; ECG contains 16 `NaN` values in one record.
- Complete-window processing produces 528,828 aligned windows and 330,517,500 retained samples per channel.
- Tail dropping removes 3,172,500 samples per channel across all records, about 7.05 hours of recording time.

## Suggested GitHub Issue

**Title:** Document project decisions and open preprocessing questions

**Suggested branch:** `docs/project-decisions`

The issue should cover adding this decision log, documenting accepted loading behavior, and tracking unresolved provenance, label extraction, signal-quality, and dataset-splitting choices.
