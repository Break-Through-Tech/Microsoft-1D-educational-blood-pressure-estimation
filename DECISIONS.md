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
| Implemented | Save window IDs, source-record IDs, and split assignments alongside the original signal arrays. | `data/load_files.py` builds archive format 2; IDs encode the original part, record index, and local window index. |
| Open | Confirm whether source records correspond one-to-one with patients. | Do not claim patient-level splitting without reliable identity metadata. |

## Data Processing

| Status | Decision | Reason or follow-up |
|---|---|---|
| Accepted | Treat PPG as the primary model input and ABP as the source of reference SBP/DBP labels. | PPG amplitude is not pressure in mmHg; ABP is the synchronized pressure waveform. |
| Accepted | Retain ECG in the processed archive, but exclude it from the initial project scope. | The challenge overview says ECG is not required for the initial model. |
| Implemented, not validated | The notebooks use mean detected ABP peaks for SBP and mean detected valleys for DBP. | Shared `data/bp_utils.py` uses `scipy.signal.find_peaks` without beat pairing or quality rules; a missing peak/valley produces NaN. |
| Implemented, not validated | `read_data.ipynb` drops aligned windows with more than 10% zeros in either signal and replaces remaining zeros with each row's nonzero mean. | This affects notebook memory, not the saved archive. Numeric zeros are not established missing values; the rule still needs evidence. |
| Open | Compare explicitly named max/min labels with the current peak/valley method and complete-beat alternatives. | Keep target versions separate; detector quality and boundary handling remain unresolved. |
| Open | Choose mean versus median aggregation for valid beat-level labels. | Median is more robust to remaining outliers; this must be evaluated. |
| Open | Define noisy-window and physiologically implausible-value rules. | Requires documented thresholds and signal-quality criteria rather than guesses. |
| Open | Choose PPG cleaning, normalization, and feature extraction methods. | Compare engineered features with models that learn from raw windows. |

## Exploratory Data Analysis

| Status | Decision | Reason or follow-up |
|---|---|---|
| Accepted | Use notebooks for visual inspection and EDA, while moving finalized repeatable preprocessing into reusable Python code. | Keeps exploration flexible and the final pipeline reproducible. |
| Completed for PPG/ABP | Plot aligned PPG/ABP examples and descriptive distributions. | `eda/edaplots_rawdata.ipynb`; ECG remains outside initial modeling scope. |
| Exploratory | Compare window extrema with unpaired detected-beat medians. | The audit comparison measures target sensitivity; it does not validate new labels. |
| Completed | Audit record lengths, windows per record, missing values, and exact-record copies. | `eda/audit.py` and `eda/EDA-readme.md`; unusual values are review candidates, not automatically artifacts. |

## Modeling And Evaluation

| Status | Decision | Reason or follow-up |
|---|---|---|
| Accepted | Frame the initial task as supervised regression from PPG to SBP and DBP. | Matches the challenge goal. |
| Accepted | Evaluate both SBP and DBP using MAE and RMSE, with at least two models and a naive baseline. | Matches the challenge success criteria. |
| Implemented | Assign canonical recordings approximately 70/15/15 to train/validation/test, seed 2026. | `configs/split.json`; saved reviewed recordings and their identical copies force the retained recording into training. Overall EDA preceded splitting. |
| Implemented | Keep the earliest `(part_number, record_index)` copy of each exact three-channel record for modeling. | Original rows remain in the archive; `iter_split_windows` excludes redundant copies. Shifted/partial overlaps are not exhaustively checked. |
| Open | Decide whether the first models use engineered PPG features, raw 625-sample windows, or both. | The choice affects preprocessing, interpretability, and model families. |

## Confirmed Dataset Findings

- All 12,000 records have the expected three-channel shape when loaded through HDF5.
- Record lengths range from 1,000 to 74,000 synchronized samples.
- PPG and ABP contain no observed `NaN` or infinite values; ECG contains 16 `NaN` values in one record.
- Complete-window processing produces 528,828 aligned windows and 330,517,500 retained samples per channel.
- Tail dropping removes 3,172,500 samples per channel across all records, about 7.05 hours of recording time.

## Using the prepared data

Run `python data/load_files.py`, then consume `data.dataset_splits.iter_split_windows("train")`. Use the same IDs and keep/exclude mask for signals, labels, and features. Learn scaling and imputation on training data only. Existing notebooks that load all rows directly do not automatically use the saved split.

Evaluation scope: **Duplicate-aware, record-disjoint evaluation on the UCI release. Patient independence and external-device generalization are not established.**
