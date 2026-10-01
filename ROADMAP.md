# CufflessAI: fixes and next steps

**We have mostly finished exploring the data. Next, we need to fix the cleaning notebook, give every window a source ID, handle repeated recordings, split by recording, and then train simple models.**

Descriptive EDA means summarizing and plotting the data. Most of that work is done. The cleaning fixes, source IDs, duplicate handling, and fair split are still pending. The tasks below describe work to do; they do not mean those fixes have already been made.

Prepared September 30, 2026 from the repository, the supplied *Evidence Review for an Educational PPG-to-Blood-Pressure Project Using the UCI Cuff-Less Dataset*, and the readings at the end. Dates, models, sample sizes, and split percentages are proposed starting points. Record accepted choices in [DECISIONS.md](DECISIONS.md). Treat the research report as evidence to assess.

## Start here

1. **Fix setup and cleaning.** FIX-01 installs the missing packages. FIX-02 removes the unsupported deletion of 53 windows and the broken attempt to fill missing values.
2. **Track where each window came from.** DATA-01 gives every five-second window a recording ID and sample positions. This source history is called **provenance**.
3. **Handle copies, then save the split.** DATA-02 keeps one copy of each identical recording. DATA-03 puts all windows from a recording in the same training, validation, or test set. This is a **record-disjoint split**: no recording appears in more than one set. We cannot guarantee different sets contain different patients because patient IDs are unavailable.
4. **Save labels and build the first models.** LABEL-01 saves the current maximum/minimum pressure answers. FEATURE-01 prepares PPG measurements for MODEL-01 to train simple predictors.
5. **Resolve signal questions alongside modeling.** FIX-03 measures zero runs. QC-01 records which signals are usable. LABEL-02 tries **beat-based labels**: measure pressure for each complete heartbeat, then summarize those measurements into one SBP/DBP pair per window.
6. **Finish with a reserved test and runnable project.** EVAL-01 runs the final comparison after model choices are fixed. DOC-01 updates the instructions and results.

FIX-01, FIX-02, and DATA-01 can start together. FIX-03 and LABEL-01 can follow DATA-01 while duplicate handling is underway. After the split exists, signal review and model development can run alongside each other. The first models can use max/min labels while LABEL-02 is still an experiment.

## Choose a task

Add your name and a pull-request link when you start. Mark a task done after its completion checks pass. **P0** blocks trustworthy modeling; **P1** builds or evaluates models and cleaning choices; **P2** prepares the final handoff. Existing files are linked. New paths are files to create, not claims that they already exist.

| ID | Priority / type | Claimable task | Depends on | Owner / status |
|---|---|---|---|---|
| FIX-01 | P0 / setup | [Install packages and rerun the starter notebook](#fix-01) | None | Unclaimed / open |
| FIX-02 | P0 / bugs | [Fix the cleaning notebook and separate signals from answers](#fix-02) | None | Unclaimed / open |
| DATA-01 | P0 / missing capability | [Give windows source IDs and process data in batches](#data-01) | None | Unclaimed / open |
| DATA-02 | P0 / leakage risk | [List duplicates and keep one copy of each recording](#data-02) | DATA-01 | Unclaimed / open |
| DATA-03 | P0 / missing capability | [Split by recording and save the assignments](#data-03) | DATA-02 | Unclaimed / open |
| FIX-03 | P1 / incomplete audit | [Count consecutive zeros in every window](#fix-03) | DATA-01 for final export | Unclaimed / open |
| LABEL-01 | P0 / missing capability | [Save the current blood-pressure answers in a table](#label-01) | DATA-01 | Unclaimed / open |
| QC-01 | P1 / unfinished EDA decision | [Review signals and write the cleaning rules](#qc-01) | DATA-03, FIX-03, LABEL-01 | Unclaimed / open |
| LABEL-02 | P1 / experiment | [Try labels calculated from complete heartbeats](#label-02) | DATA-03, QC-01 | Unclaimed / open |
| FEATURE-01 | P1 / prototype to pipeline | [Turn the feature starter into reusable code](#feature-01) | FIX-01, DATA-01; DATA-03 before selecting features | Unclaimed / open |
| MODEL-01 | P1 / missing capability | [Train simple models and compare their errors](#model-01) | FIX-02, DATA-03, LABEL-01, FEATURE-01 | Unclaimed / open |
| EVAL-01 | P1 / missing capability | [Choose the final setup and run the reserved test](#eval-01) | MODEL-01; QC-01 before final freeze | Unclaimed / open |
| DOC-01 | P2 / stale documentation | [Update the notes and make the full project runnable](#doc-01) | Start now; finish after EVAL-01 | Unclaimed / open |

## Terms used in the tasks

- **Record / window:** a record is one downloaded signal segment. A window is a complete five-second piece containing 625 samples at 125 Hz. The final short remainder is the record's tail.
- **PPG / ABP / ECG:** PPG is the pulse signal used as model input. ABP is the measured arterial-pressure signal used to calculate answers. ECG is retained in the archive but is outside the initial modeling scope.
- **Label or target (`y`):** the answer to predict. SBP is systolic pressure and DBP is diastolic pressure, both in mmHg. **Features (`X`)** are measurements calculated from PPG and given to the model.
- **Max/min or extrema labels:** the highest and lowest ABP sample in a window. A beat median describes the typical accepted heartbeat. These are different targets.
- **QC:** quality checks. A flag marks something for review; it does not automatically mean delete the window.
- **Training / validation / test:** training fits the model; validation helps choose settings; the reserved test measures the final result. "Development data" means training and validation together.
- **Manifest / canonical record:** a manifest is a table listing data and source IDs. A canonical record is the one copy kept to represent identical recordings.
- **Nonfinite:** a NaN or infinity value. Numeric zero differs from a missing value. **Imputation** means replacing a missing value.
- **MAE / RMSE:** mean absolute error is the average size of the pressure error. Root mean squared error gives large mistakes more weight. Report both separately for SBP and DBP.

## What is already done

| Work | Evidence and current result |
|---|---|
| Loading and windowing | [data/load_files.py](data/load_files.py) reads the four MAT/HDF5 files into aligned PPG, ABP, and ECG windows. It drops incomplete tails. |
| Saved signals | `data/processed_dataset.npz` has three float64 arrays of shape `(528828, 625)`. No source IDs, labels, QC flags, or split assignments. |
| Dataset checks | [eda/audit.py](eda/audit.py) and [eda/EDA-readme.md](eda/EDA-readme.md) report completeness, numeric values, duplicates, and agreement with the archive. Copies are detected but still present. |
| Plots | [eda/edaplots_rawdata.ipynb](eda/edaplots_rawdata.ipynb) includes distributions, unusual examples, source coordinates, zero cases, and an early peak comparison. |
| Labels and cleaning attempt | [notebooks/read_data.ipynb](notebooks/read_data.ipynb) calculates ABP max/min in memory. It also contains the deletion and imputation problems in FIX-02. |
| Feature starter | [eda/neurokit_feature_starter.ipynb](eda/neurokit_feature_starter.ipynb) has six candidate features and QC fields. Saved results cover **10 windows from four records**, using NeuroKit2 0.2.13. |
| Still missing | A reusable feature export, trained benchmark, saved split, model-selection procedure, and final test results. [requirements.txt](requirements.txt) is effectively empty; [README.md](README.md) is mostly a template. |

The audit found **12,000 records, 528,828 complete windows, and 14 extra exact-record copies containing 700 windows**. Removing those copies should leave **11,986 records and 528,128 windows**, before other exclusions.

There are **5,407 PPG windows containing zeros**. Of those, 53 contain at least 32 zeros; those zeros need not be consecutive. Our copy has no observed PPG/ABP NaNs or infinities. Dropped tails total **3,172,500 sample positions per channel**, about **0.95%**. Check these local counts when rebuilding data.

During the original review, `.venv\Scripts\python.exe eda/audit.py` ran successfully and confirmed exact PPG/ABP agreement, in order, between source windows and the archive. Notebook code and saved outputs were read, but plotting/NeuroKit notebooks were not rerun: the inspected environment lacked matplotlib, NeuroKit2, and scikit-learn. This roadmap has not changed waveform files or implemented model code.

## Rules shared by all tasks

Use `window_id` to join signals, labels, QC, features, and splits. Row numbers can change after filtering. Keep original MAT files and the existing NPZ intact.

Write generated tables to Git-ignored `data/derived/` and compact reports to `reports/`. Commit code, small artificial test datasets, settings, and documentation. Keep waveforms and large feature tables out of Git. Store manifests and checksums in the team's chosen artifact location, and document how to recreate them.

Coordinate changes to shared files such as `eda/audit.py`. Put repeated calculations in reusable functions that the notebooks call.

## Task instructions

<a id="fix-01"></a>

### FIX-01: Install packages and rerun the starter notebook

**Files:** [requirements.txt](requirements.txt), [.gitignore](.gitignore), [data/README.md](data/README.md), [eda/neurokit_feature_starter.ipynb](eda/neurokit_feature_starter.ipynb).

**Why this matters:** requirements are empty; the inspected environment cannot import plotting, NeuroKit, or model packages. Saved notebook outputs came from another environment.

**Steps:**

1. Create a fresh virtual environment and record its Python version. Install numpy, pandas, scipy, h5py, matplotlib, NeuroKit2, scikit-learn, Jupyter, and the kernel/execution packages used to run notebooks.
2. Start with the pilot's recorded `neurokit2==0.2.13`. Install package versions that work together, run `python -m pip check`, and write the successfully tested package versions to `requirements.txt`. Do not copy an unrelated environment's entire package list.
3. Add `.venv/`, `__pycache__/`, and the new generated-data/model directories to `.gitignore`. Keep small source/configuration files trackable.
4. Document setup and run commands that start from the repository root. Run `python eda/audit.py`; execute the NeuroKit starter after restarting the notebook kernel using this environment, not the notebook's saved outputs.
5. Record the selected record IDs, sample count, processing failures, package versions, and elapsed time in a short reproducibility note.

**Done when:**

a second fresh environment installs from the file, `pip check` passes, the audit runs, and the pilot produces a fresh feature table and plots. Record any difference from the saved 10-window output rather than editing results to match it.

<a id="fix-02"></a>

### FIX-02: Fix the cleaning notebook and separate signals from answers

**File:** [notebooks/read_data.ipynb](notebooks/read_data.ipynb). Find cells using `leniency_percent`, `num_entries = 627`, `to_drop_rows`, and `fillna` rather than relying on cell numbers.

**Why this matters:** the notebook deletes 53 windows under an unsupported rule, mixes labels into signal operations, then attempts a repair that neither replaces zeros nor reliably addresses the original row.

**Steps:**

1. Keep waveform arrays at exactly 625 columns. Calculate both target arrays from the unchanged ABP waveform: `sbp = abp.max(axis=1)` and `dbp = abp.min(axis=1)`. Put targets in a separate table; do not append them to either waveform table.
2. Delete the automatic 5% row-dropping loop from the normal notebook run. Replace it with a QC table containing zero count, negative count, NaN/infinity count, and zero fraction (`zero_count / 625`). Preserve all rows at this stage.
3. Delete the `ppg_dataset.iloc[row].fillna(..., inplace=True)` loop and the calculation of the mean across that row. The current data have no PPG/ABP NaNs to repair; numeric zeros must remain unchanged until QC-01 establishes a handling rule. Do not convert zeros into NaNs to make imputation run.
4. Once DATA-01 is merged, attach `window_id` to targets and QC. Apply future approved exclusions using one table of window IDs and keep/exclude decisions, joined to every data table. Do not reuse pre-deletion positions or separately drop rows from `X` and `y`.
5. Replace the notebook's full-array display with a small preview and summary counts. Clear stale outputs and rerun the changed path after restarting the notebook kernel, using small batches after DATA-01 is available.

**Done when:**

baseline preparation retains 528,828 windows before removing duplicate copies; signal inputs remain 625 columns; zeros remain unchanged; no chained-assignment warnings occur; and target/QC joins preserve exact window IDs. Changing a target value in a small synthetic example must not change any PPG QC value. Using 625 instead of 627 still gives the old integer cutoff of 31. Explain that the deletion rule lacks evidence; do not claim changing the denominator would restore those 53 rows.

<a id="data-01"></a>

### DATA-01: Give windows source IDs and process data in batches

**Why this matters:** The saved arrays do not identify their source recordings. The loader also holds large arrays in memory; the three final arrays alone need about 7.93 GB before copies and DataFrames.

**Files:** [data/load_files.py](data/load_files.py); create `data/build_manifest.py` and reusable record/window iteration functions.

**Steps:**

1. Yield `(part_number, record_index, record)` from a function that reads one record at a time and returns its source information. Keep existing callers working through a wrapper or update all callers together. Sort by part 1–4, then zero-based record index.
2. Require each record to have shape `(N, 3)` before indexing it. Count NaN and infinity values separately for PPG, ABP, and ECG. For a wrong shape, raise an error naming the part and record. For NaN or infinity, record which signal is affected. Do not discard valid PPG/ABP windows solely because ECG contains NaNs.
3. Use stable IDs such as `Part_1:record_1055` and `Part_1:record_1055:window_0`. For each complete window store `start_sample = window_index * 625`, `end_sample = start_sample + 625` (exclusive), and `legacy_npz_row_index` counted across **all original records**, including copies.
4. Write `data/derived/records.csv` and `data/derived/windows.csv` in batches. Include the fields in the shared-table specification below. Save hashes that account for shape and data type, plus source-file checksums (fingerprints used to detect changes). The record table must account for trailing samples even when a record contributes zero complete windows.
5. Reuse the functions that read one record at a time for later label/feature exports. Do not load the entire NPZ into two pandas DataFrames. If a new waveform cache is needed, write HDF5 in chunks or save arrays in manageable parts; keep the legacy archive intact.
6. Add small tests with fixed inputs for lengths 624, 625, 626, and 1,250: expected window/tail counts are `(0,624)`, `(1,0)`, `(1,1)`, and `(2,0)`. Include a malformed shape and an ECG-only NaN case.

**Done when:**

the tables describing the original data contain 12,000 records and 528,828 windows, with 3,172,500 trailing samples per channel. Reload the first/last window of each part and 100 IDs chosen with a saved random seed from MAT files; assert exact PPG/ABP equality with the corresponding legacy rows. Run the existing streaming archive check. Record peak memory for the new export to demonstrate that it does not retain all waveforms.

<a id="data-02"></a>

### DATA-02: List duplicates and keep one copy of each recording

**Why this matters:** The audit found 14 extra copies, but they are still in the data. Copies give those signals extra weight and can place the same signal in training and testing.

**Files:** [eda/audit.py](eda/audit.py), record manifest from DATA-01; create `data/deduplicate.py`.

**Steps:**

1. Preserve the full `duplicate_records` list in the audit result/export. Keep `duplicate_records[:10]` only as a display preview; do not use that truncated field to build the dataset.
2. Group candidate duplicates by record shape, dtype, and full three-channel bytes. Compare matching candidates directly before declaring them identical. Keep the record with the lowest `(part_number, record_index)` as the canonical record, the one copy retained from an identical set.
3. Export `data/derived/duplicates.csv` with canonical ID, redundant ID, matching hash, and redundant window count. Add `canonical_record_id` and `is_canonical` to the record manifest; join those fields onto windows by record ID.
4. Build the dataset used for analysis by selecting the retained records, without deleting source files or changing legacy row numbering. Report removed records and windows per part.
5. Screen exact paired PPG/ABP windows across different canonical records. Reload matches and compare bytes and surrounding sequence positions. Save confirmed overlaps as links and give each connected set of overlapping records one `split_group_id`; single-channel flatness or high correlation alone is not confirmation. Document that shifted excerpts outside this exact-window screen remain possible.

**Done when:**

all 14 known redundant copies, including 13 cross-part copies, are exported; excluding those copies removes 700 windows and leaves 11,986 records / 528,128 windows before other exclusions. Duplicating a record in a small test dataset must select one canonical record and preserve the relationship regardless of the order in which records are read. Any newly confirmed overlap links are kept in one split group by DATA-03.

<a id="data-03"></a>

### DATA-03: Split by recording and save the assignments

**Why this matters:** There is no saved split. Neighboring windows are closely related, so randomly splitting windows would make the test too easy.

**Files to create:** `data/make_splits.py`, `configs/split.json`, `data/derived/splits.csv`, `reports/split_summary.md`.

**Steps:**

1. Read canonical records and `split_group_id`; use canonical record ID as the group when no overlap links exist. Make a table with one row per group. Assign groups to splits; never randomly split individual windows.
2. List records whose individual waveforms/features were already reviewed, including the four NeuroKit pilot records and plotted EDA examples. Keep those groups in training or validation, document the exception, and allocate the remaining groups with seed 2026 toward overall 70/15/15 train/validation/test proportions. Save exact assignments; report deviations caused by forced development groups and rounding.
3. Join assignments onto every window by source/group ID. Assert one split per source record, canonical duplicate group, and confirmed-overlap component. Add assertions that no group appears in two different splits.
4. Export both group/record counts and window counts for each split. Report counts before and after keep/exclude rules. Do not try different seeds to obtain better scores or preferred BP distributions.
5. Save the split configuration, input-table checksums, and output checksum. Add small test datasets with multiple windows per record and a cross-part duplicate; deliberately split one duplicate across sets and make the assertion fail.

**Done when:**

repeated runs generate the same assignments; all overlap assertions pass; downstream training takes the saved file as input; and test records are excluded from subsequent development plots/review sets. Describe the split as record-disjoint. Patient identity is unknown. State that overall EDA happened before this split was saved.

<a id="fix-03"></a>

### FIX-03: Count consecutive zeros in every window

**Why this matters:** The notebook searches for long runs only in windows with at least 32 zeros. Its result does not cover every window. The audit counter would also miss a window containing both negative values and zeros.

**Files:** [eda/audit.py](eda/audit.py), [eda/edaplots_rawdata.ipynb](eda/edaplots_rawdata.ipynb); export `data/derived/window_qc.csv`.

**Steps:**

1. Move zero-run calculation outside the `many_zero_indices` selection. For each window, form `mask = ppg == 0`, pad it with `False` on both ends, find transitions, and subtract start indices from end indices. Store the largest run, or zero if there are no runs.
2. Store total zero count, zero fraction, longest run in samples/seconds (`samples / 125`), negative count, NaN/infinity count, range, standard deviation, and largest absolute adjacent-sample change. Keep these as measurements, not deletion decisions.
3. In `audit.py`, replace the zero-containing-window count based on `ppg_min == 0` with `window_zero_counts > 0` or `np.any(ppg == 0, axis=1)`. The old expression misses a window containing both a negative sample and a zero; this is a robustness bug even if current counts agree.
4. Select and plot the longest-run example across **all** windows. Label the old 21-sample result as a result for the restricted 32+-zero subset until the full calculation completes.
5. Add small test datasets for no zeros, one zero, separated zeros, a run touching either boundary, all zeros, and mixed negative/zero values. For `[0,0,1,0,0,0]`, require count 5 and longest run 3.

**Done when:**

QC has one row identified by window ID per original window; total zero-containing counts match 5,407 on the current copy; all test cases pass; and the notebook uses the exported metrics rather than a second inconsistent implementation.

<a id="label-01"></a>

### LABEL-01: Save the current blood-pressure answers in a table

**Why this matters:** The notebook calculates pressure answers in memory but does not save a reusable label table.

**Files to create:** `data/derive_labels.py`, `data/derived/labels_window_extrema_v1.csv`.

**Steps:**

1. Read ABP through the DATA-01 iterator and compute maxima/minima over only the 625 waveform samples. Keep pressure units unchanged. Do not derive labels from cleaned/normalized PPG or a DataFrame containing label columns.
2. Export `window_id`, `label_version=window_extrema_v1`, `sbp`, `dbp`, `abp_finite`, `label_valid`, and `failure_reason`. Do not silently use `nanmax`/`nanmin` to hide missing ABP; mark labels invalid if ABP contains NaN or infinity.
3. Keep one row for every original window. Join duplicate and keep/exclude tables by ID to select model data. Max/min labels describe the highest and lowest samples; they do not prove the pressure signal is reliable.
4. For a known synthetic ABP window, assert the two target values equal its exact maximum/minimum. Reload 100 source windows chosen with a saved random seed and compare their exported targets to direct array reductions.

**Done when:**

the current copy produces 528,828 SBP/DBP pairs with window IDs and no NaN/infinity before exclusions, labels reproduce exactly on rerun, and no target column is included in an list of model inputs. This task does not depend on completing LABEL-02.

<a id="qc-01"></a>

### QC-01: Review signals and write the cleaning rules

**Why this matters:** A zero or unusual pressure value does not establish whether a signal is unusable. We need recorded examples and decisions before deleting data.

**Files to create:** `eda/review_windows.ipynb`, `data/derived/review_windows.csv`, `data/derived/review_annotations.csv`, `reports/qc_policy.md`.

**Steps:**

1. Load the saved split and restrict review candidates to train/validation records. Select 200 unique windows from at least 50 records: 80 random examples chosen with a saved seed and 20 additional examples from each of six categories—one zero, multiple zeros, longest runs, lowest PPG range, lowest/highest max/min pressure labels, and smallest/largest pulse pressure. Resolve category overlaps by selecting the next eligible ID; record shortages. Cover all four parts and spread selections across records rather than filling the set from one long recording.
2. Save the chosen IDs, categories, seed, and train/validation designation. Create PPG and ABP plots with window boundaries, one second of available context on either side, exact-zero markers, and source coordinates. Save figures keyed by window ID.
3. Have reviewers enter separate PPG and ABP ratings (`usable`, `uncertain`, `unusable`), reason codes, and notes. Annotate ABP onset/peak/trough sample indices and incomplete boundary beats for LABEL-02. Independently double-review 40 windows, including difficult categories, and save both original ratings and the final agreed decision.
4. Write an exclusion policy that distinguishes input failure from unreliable reference labels. For each proposed rule, calculate agreement with training annotations and the number of retained/rejected windows and records by part and pressure ranges defined using training/validation data. Keep uncertain cases explicitly labeled. Do not fit thresholds on the final test.
5. Apply the chosen rules to the reserved validation annotations without retuning. Save disagreements, included/excluded counts, and BP histograms before and after filtering. If the evidence does not justify a threshold, retain its flag and state that it is not an exclusion rule.

**Done when:**

the saved annotations and policy explain exactly which conditions exclude an example, with rule-specific counts and validation disagreements. Focus the remaining EDA on these specific questions and save the decisions. MODEL-01 can produce development baselines before this finishes; final evaluation cannot freeze an undocumented cleaning policy.

<a id="label-02"></a>

### LABEL-02: Try labels calculated from complete heartbeats

**Why this matters:** The current detector finds peaks and troughs separately. It does not establish which points belong to the same complete heartbeat.

**Files:** [eda/audit.py](eda/audit.py), [eda/edaplots_rawdata.ipynb](eda/edaplots_rawdata.ipynb); create `data/abp_beats.py` and a separate beat-label export.

**Steps:**

1. Keep the current `exploratory_beat_labels` explicitly exploratory. Implement a detector that marks the start (onset) of each beat in a continuous ABP record. Save the implementation version, settings, sampling rate, and how it handles pressure units. Store detected onsets in original sample coordinates.
2. Define a beat between two consecutive onsets. Reject intervals with reversed or identical start/end positions. Specify whether the closing onset is included when locating the diastolic trough, and store the exact samples used. Detect on context if needed, but exclude a beat from a target window when its defining boundaries or pressure samples lie outside that window's stored sample interval.
3. Store one row per beat with onset/end indices, SBP/DBP values and locations, duration, quality flags, and accepted/rejected status. For each window, calculate **separate** median and mean labels from the accepted complete beats. Begin with a configurable minimum of two beats; emit invalid status when insufficient beats remain.
4. Use training annotations from QC-01 for development. Before evaluating validation annotations, write the matching tolerance, quality rules, and numerical criteria for accepting the new method in `reports/label_protocol.md`. Report missed/extra beats, onset timing errors, boundary errors, differences from labels marked by reviewers, and the fraction of windows that receive valid labels. Set acceptance criteria before seeing validation results.
5. Save extrema/median/mean comparisons on identical window IDs, including percentiles of the differences and the 20 largest validation disagreements with plots showing the marked points. Keep all failure rows. Never fill an invalid median target with an extrema target under the same version name.

**Done when:**

manually marked examples and artificial boundary tests have recorded outcomes, every beat target is traceable to accepted beats, and a written decision either adopts the new labels or keeps max/min labels. Two independent peak/trough lists and a lower model MAE alone do not close this task.

<a id="feature-01"></a>

### FEATURE-01: Turn the feature starter into reusable code

**Why this matters:** The starter has only run on 10 windows from four records. It needs reusable code, a full export, and clear failure handling.

**Files:** [eda/neurokit_feature_starter.ipynb](eda/neurokit_feature_starter.ipynb); create `features/extract_ppg.py`, `configs/features.json`, and `data/derived/features_v1.csv`.

**Steps:**

1. Move extraction into a function whose output depends only on its arguments: PPG samples, sampling rate, and configuration. Do not pass ABP or labels into this function. Start with one 625-sample PPG window per prediction. If an experiment needs surrounding samples, document how many and provide them during prediction too.
2. Start with the pilot's six features and explicit NeuroKit methods/version. Keep source IDs, QC measurements, processing errors, and targets out of the explicit list of allowed model inputs. Add raw summary features as a separately named comparison set.
3. Process records incrementally and emit one row identified by window ID per requested window, including failed windows. Record separately whether processing failed entirely or only one feature was undefined. Save the exception or reason code. Do not treat every returned table as successful just because no exception occurred.
4. Export a table describing each feature with formula, units, required peak count, and missing-value behavior. Keep undefined values as missing for MODEL-01's missing-feature handling, learned from training data only; do not calculate fill values over the full dataset.
5. Run on a saved subset of training/validation records first, then the full requested export. Save success rates and missing-feature counts by record and part. Recompute a sample twice and assert equal outputs; change ABP labels externally and assert unchanged feature values. Replace the notebook's claim that filtering preserves shape/amplitude with the actual method and measured raw-versus-clean summaries/overlays.

**Done when:**

features regenerate by ID, failure counts are explicit, making a prediction needs only PPG, and the notebook calls the shared extraction function. High correlations in the four-record pilot do not qualify a feature for inclusion by themselves.

<a id="model-01"></a>

### MODEL-01: Train simple models and compare their errors

**Why this matters:** There is no reproducible model comparison. Start by finding out whether PPG features improve on guessing a typical pressure.

**Files to create:** `models/train_baselines.py`, `configs/baseline.json`, `reports/validation_metrics.csv`; save fitted bundles under an ignored model-output directory.

**Steps:**

1. Join split assignments, duplicate status, max/min labels, features, and keep/exclude decisions by `window_id` and assert that each ID has at most one matching row in each table. Stop with an error if an ID is duplicated or has no expected matching row. Save the list of allowed features and the exact window IDs included in the run.
2. Use the same selected windows for every model. Require labels without NaN/infinity and apply the documented PPG-processing rules. Report excluded failures. Learn missing-feature replacement values and scaling values from training rows only. Apply the same values to validation and test rows.
3. Fit constant training mean and training median predictors per target. Fit Ridge with a small predefined set of alpha values (for example 0.1, 1, 10, 100). Fit one histogram-gradient-boosting regressor per target with a small saved set of parameter combinations; disable automatic early stopping that creates its own random window split.
4. Choose model settings using validation MAE, report RMSE as well, and save per-window validation predictions for all four predictors on the same IDs. Keep the test set out of the script's default evaluation path.
5. Write the comparison table with separate SBP/DBP metrics, target version, feature/QC versions, record/window counts, parameters, seed, and runtime. Save fitted preprocessing and estimators together. Run twice with the same configuration and compare predictions within documented numerical tolerance.

**Done when:**

both regressors and both constants have reproducible validation results on the same selected windows. Include results even if the constants win. This task proceeds with max/min labels while LABEL-02 is still an experiment.

<a id="eval-01"></a>

### EVAL-01: Choose the final setup and run the reserved test

**Why this matters:** We need a final test after choosing the model. Repeatedly adjusting the model using test scores would make the result misleading.

**Files to create:** `models/evaluate.py`, `configs/final_evaluation.json`, `reports/model_comparison.csv`, `reports/error_analysis.md`.

**Steps:**

1. Run the comparisons listed later in this roadmap on training/validation data. For each comparison save IDs of windows included in every method, failures for each method, label version, and the parameters selected. Compare different label definitions against manual references as well as their own regression errors.
2. Save the final preprocessing, label/QC rules, feature list, model settings, and metric definitions in one configuration, and stop adjusting them. Record whether the final model remains trained on train only or is refitted on train+validation after selection; refit every learned transform consistently if using the latter.
3. Add an explicit final-test mode that loads the frozen configuration and saved split. Save predictions once for the chosen models and constant predictors fitted on the same final training data. Compute per-window MAE/RMSE and signed error separately for SBP/DBP.
4. Also compute each record's MAE and report their equal-weight mean. For equal-record RMSE, average record MSEs and then take the square root. Generate 95% intervals from 1,000 seeded resamples of test split groups, carrying all windows with each sampled group; use identical resamples when calculating differences between models.
5. Save plots of prediction errors and metric/count tables by predefined BP band, part, and PPG quality category. List the 20 largest errors and all `predicted_sbp <= predicted_dbp` cases; generate waveform examples without using the findings to quietly retune the model. Document post-test corrections as a new exploratory run.

**Done when:**

saved predictions reproduce every table, interval resampling uses groups rather than rows, included and excluded example counts are reported, and the claim says record-disjoint rather than patient-independent.

<a id="doc-01"></a>

### DOC-01: Update the notes and make the full project runnable

**Why this matters:** Some notes describe completed EDA as pending, and the README has placeholders. Instructions need to match the final code.

**Files:** [README.md](README.md), [data/DATA_NOTES.md](data/DATA_NOTES.md), [eda/EDA-readme.md](eda/EDA-readme.md), [DECISIONS.md](DECISIONS.md), the affected notebooks; create a final end-to-end notebook and `reports/model_card.md`.

**Steps:**

1. Mark completed descriptive EDA as completed in the decision log. Keep label/QC decisions open until their task evidence exists. Link each resolved issue to its code/report instead of leaving contradictory old instructions.
2. Update data notes with the table columns and ID-based joins and heartbeat detection on complete ABP recordings followed by window assignment for the beat experiment. State that known patient mapping and exact-zero semantics are unavailable. Describe original and counts after removing exact copies separately.
3. Replace README placeholders with actual setup, dataset preparation, feature extraction, training, and evaluation commands. Label proposed commands as proposed until implemented; publish no fabricated accuracy numbers.
4. Create an end-to-end notebook that calls the reusable modules rather than copying their implementations. Include a small trial-run configuration using training/validation records and a full-run configuration using the same saved split.
5. Run the documented small trial run from a fresh environment. Load the saved model bundle and feed a single PPG window from a reserved test record; assert its output matches batch inference without supplying ABP. Add final metrics, exclusions, inference context, and limitations to the model card and presentation.

**Done when:**

a teammate can follow the README to reproduce a small run and understand how the full results were generated; obsolete cleaning claims and stale notebook outputs are removed or explicitly labeled historical.

## Shared tables to build

DATA-01 and the later exports use these fields. A separate source table can make the existing NPZ usable without rewriting it, once source-to-row matching passes the checks.

| Table | Fields to keep |
|---|---|
| Records | `source_record_id`, part number, zero-based record index, length, complete-window count, tail count, content hash, canonical duplicate ID |
| Windows | `window_id`, source ID, local window index, start sample, exclusive end sample, existing NPZ row index |
| Splits | group ID, train/validation/test assignment, seed, split-policy version |
| Labels | window ID, target version, SBP, DBP, valid-beat count when applicable, validity flag, failure reason |
| QC and features | window ID, separate PPG/ABP flags, processing status, feature version, explicitly named PPG input columns |
| Run details | source checksums, code revision, package/settings versions, allowed feature list, split/label/QC versions, model settings |

## Choices that guide implementation

**Keep the current window size and task.** Predict SBP/DBP for the same five-second interval as the PPG input. This is not future-pressure forecasting. Keep dropping only incomplete tails; do not add padding. Recovering tails is lower priority than fixing labels and leakage, where related signals appear on both sides of an evaluation.

**Keep label versions separate.** Use `window_extrema_v1` first. Test `complete_beat_median_v1`, with beat means as a comparison. Never fill a missing median label with max/min under the same name. The current unpaired peak experiment found median differences of **+2.35 mmHg for SBP and -1.76 mmHg for DBP**. Label choice matters, but these differences do not establish detector accuracy. If the beat method remains inconclusive by October 16, keep max/min as the main target and report the beat experiment separately.

**Reserve test records before further tuning.** Overall EDA has already happened; disclose that. Keep individually reviewed examples in training/validation where practical and document the assignments. The proposed 70/15/15 split and seed 2026 apply to recording groups. Window percentages will differ because recordings have different lengths. Any extra validation folds must also keep groups together. [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) is one implementation option.

**Support quality rules with evidence.** The meaning of exact PPG zeros is unknown. Do not automatically delete or interpolate them. Compare PPG shape, local variation, abrupt changes, flat sections, and peak counts with the separate ABP reference. Unusual real pressure is not automatically a bad label. Report how each removal rule changes counts and pressure ranges, including counts by recording length. The proposed review size and two-beat minimum are practical starting points, not published standards. Include different beat rates and detector failures in review; expand the sample when a failure category remains unresolved.

**Use the right signal context.** Detect ABP beats on continuous records, then assign complete accepted beats to windows. [PhysioNet's `wabp`](https://physionet.org/content/wfdb/10.7.0/app/wabp.c) is a relevant adult-ABP detector designed for 125 Hz. It is a WFDB program: document how to run or adapt it, its units, startup behavior, and license before adopting it. [Sun et al.](https://www.cinc.org/archives/2006/pdf/0013.pdf) offers starting quality checks; an abnormality flag alone does not prove artifact.

For PPG, start with one 625-sample input window. If filtering uses surrounding or future samples, provide the same context during prediction and describe the system accordingly. A fair recording split does not make that a real-time five-second estimator. ABP may supply labels and offline review, but it must not be required to calculate prediction features, align incoming PPG, or decide whether an incoming PPG signal is acceptable. Report exclusions caused by poor ABP separately from PPG rejection.

**Add features when they help.** Begin with the six existing features and simple PPG summaries. Later candidates include amplitude spread, skewness, time between peaks, pulse amplitude, rise time, width, area, and derivative summaries. The overview suggests around ten features; justify each one instead of adding features to hit a count. Filtering can change amplitude and shape. A few pulse intervals in five seconds do not establish reliable long-duration heart-rate variability (HRV).

**Keep the first model comparison small.** Use constant training mean, constant training median, Ridge, and one histogram-boosting model per target. A bounded random forest is an alternative. See the [histogram boosting documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). Use grouped validation when supported by the installed version, or disable automatic random-window early stopping. Start with a saved subset of development records, then scale. Do not use the overall medians **131.10/62.03 mmHg** as fitted constants; those include reserved test data.

## Model comparisons to run

Use training/validation data for these choices. Save the same windows for comparisons where possible and report each method's additional failures.

| Comparison | Keep the same | What we learn |
|---|---|---|
| Ridge, boosting, and constants | Split, labels, features, included windows | Whether PPG features help on new recordings |
| Basic summaries and pulse-shape features | Model/settings-search budget, split, labels, included windows | Whether more complicated features help |
| Original and filtered/normalized PPG | Split, labels, shared windows, model budget | Whether extra processing helps |
| Max/min, beat median, and beat mean | Split and model family; show shared and method-specific windows | How the answer definition changes results and data coverage |
| Minimal removals and reviewed QC removals | Label definition and modeling procedure | How much improvement comes with removing difficult examples |

Lower error against a different label does not prove that label is more accurate. Compare labels to manual markings too. A method should not appear better simply because it rejects harder windows. Choose settings and features using validation, then freeze the final setup before testing. Optional repeated validation must keep recordings together. A raw-window Ridge or small 1D CNN can follow once this comparison is reproducible.

Use validation-based permutation importance (shuffle one feature and measure the change in error) or standardized Ridge coefficients to explain what the model uses. Mention correlated features. Show large errors and `SBP <= DBP` predictions without silently clipping them. Record bug fixes prompted by test results; later scores then count as exploratory results.

Resampling whole test groups measures uncertainty while keeping related windows together. It cannot account for unknown cases where separate records came from the same patient. We also lack demographic data for demographic fairness analysis; do not infer demographics from signals.

Use this scope statement: **"Duplicate-aware, record-disjoint evaluation on the UCI release. Patient independence and external-device generalization are not established."** In plain language: we handle known copies and test on different recordings, but cannot claim success on new people or different sensors. A well-run experiment that barely beats the constants, or does not beat them, is still a useful educational result.

## Proposed schedule

These dates follow the overview's October modeling and November presentation milestones. Complete the checks before moving on; December provides handoff buffer time.

| Phase | Proposed dates | Output and completion check |
|---|---|---|
| 0. Setup and cleaning fixes | Oct 1-3 | Another teammate can run the audit and small feature starter using documented commands. |
| 1. IDs, duplicates, and splits | Oct 1-7 | Every window has source information; no recording or linked-copy group crosses splits. |
| 2. Labels and quality rules | Oct 5-16 | Reviewed examples, detector/boundary results, label choice, and removal counts are saved. |
| 3. Features and first models | Oct 8-21 | Features repeat correctly; failures are explicit; preprocessing learns only from training; validation results are comparable. |
| 4. Controlled comparisons | Oct 19-30 | Select the setup using validation; record label and included-window differences; freeze the final test procedure. |
| 5. Final evaluation | Nov 1-13 | Reproducible reserved-test results, uncertainty based on recordings, and error analysis. |
| 6. Handoff and optional demo | Nov 14-30 | A fresh environment reproduces the comparison and PPG-only prediction; notebook, report, slides, and model card are ready. |

**If time is short:** finish setup/cleaning, source IDs, duplicate handling, recording splits, max/min labels, documented minimal QC, features, two regressors plus constants, and the final report. Beat labels can remain an experiment. Source IDs, group separation, and exclusion counts remain required.

**Optional later work:** CNNs, exhaustive searches for similar excerpts, extra datasets, classification, phone/webcam capture, deployment, and dashboards. A replay demo can show reserved prerecorded PPG, reference labels, predictions, quality status, and model comparisons. Phone/webcam signals come from a different acquisition setup and need separate evaluation. Prediction intervals also need their own validation; the spread of past errors is not automatically a confidence guarantee for one new prediction.

## Research and reading priorities

The bibliography is in [Challenge-Project-Overview.md](Challenge-Project-Overview.md); the root README currently has no substantive reading list.

The [official UCI page](https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation) establishes the format, channels, 125 Hz rate, and that the creators already did some processing. It does not provide patient mapping, zero semantics, or enough detail to recreate every earlier step. Call these the "original downloaded UCI files." The creators' later experimental database differs from this release; do not assume its filters or patient separation are already present in our files.

The [2023 benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/) uses five-second segments and median systolic-peak/diastolic-boundary labels, and lists UCI subject identity as unknown. This adds a close precedent to the supplied report. It supports testing the idea but does not validate our detector or justify copying thresholds. Its UCI subset has different counts. [Khurana et al., 2026](https://pubmed.ncbi.nlm.nih.gov/42546757/) compares max/min with averages of four peaks/four troughs in ten-second segments; that rule should not automatically become our five-second default.

| Read when | Source | What to take from it |
|---|---|---|
| Choosing labels and splits | [2023 benchmark](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/) | Read preprocessing, dataset characteristics, and splitting. Account for different subsets when comparing published results. |
| Documenting data | [UCI documentation](https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation) | Distinguish documented facts from unanswered source-data questions. |
| Building beat detection/QC | [Sun et al.](https://www.cinc.org/archives/2006/pdf/0013.pdf), [WFDB `wabp`](https://physionet.org/content/wfdb/10.7.0/app/wabp.c) | Design annotations, detector checks, and candidate quality flags. |
| Extracting PPG features | [NeuroKit PPG API](https://neuropsychology.github.io/NeuroKit/functions/ppg.html) | Record the actual filter, peak detector, quality method, and package version. |
| Adding feature ideas | [Chowdhury et al., 2020](https://arxiv.org/abs/2005.03357), [PPG features, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10963242/) | Both describe 219-participant studies. Their data and inputs differ; our files lack demographic features. |
| Optional follow-up | [Non-fiducial features and derivatives, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12511243/) | Full text was unavailable during the original review. This plan does not rely on its accuracy or splitting claims. |
| Comparing label summaries | [Khurana et al., 2026](https://pubmed.ncbi.nlm.nih.gov/42546757/) | Keep the difference between its ten-second segments and our five-second windows explicit. |
