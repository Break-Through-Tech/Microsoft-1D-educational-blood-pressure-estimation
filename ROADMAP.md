# CufflessAI: priorities for a trustworthy model result

**Recording identities, exact-duplicate exclusion, and recording-based splits are implemented. Next, export labels and PPG features by window ID, then compare models on the saved training/validation sets.**

Updated October 3, 2026 against current code. The loader preserves every original PPG/ABP/ECG row and adds identities and split assignments. Direct full-array notebook reads still include duplicate copies and all splits; use the split loader for modeling. No trained comparison or final test score exists.

The notebooks now use mean detected ABP peaks/valleys through `data/bp_utils.py`. The old zero-handling code was replaced: current code drops aligned rows with more than 10% zeros and replaces remaining zeros with the row's nonzero mean. Neither method has established signal-quality validity. The max/min target and zero-preserving minimal policy below remain proposed baseline comparisons, not descriptions of the current notebook.

## Do these next

1. Run `python data/load_files.py` to prepare the shared dataset. DATA-01/02/03 are implemented together: identities are saved, exact copies excluded by the split loader, and whole recordings assigned to splits.
2. LABEL-01: export an explicitly versioned target with window IDs. Current notebook peak/valley means and a named max/min baseline are different targets; do not switch silently.
3. FEATURE-01 and MODEL-01: consume saved training/validation batches and retain IDs. Compare constants, Ridge, and boosting; learn preprocessing from training only.
4. EVAL-01: reserve test batches for the frozen final setup. Report errors, coverage, and recording-level limitations.

FIX-01 supports reproducibility. QC-01 still needs a shared inclusion policy. Dataset preparation does not normalize signals, apply zero replacement, reject outliers, or extract features; these remain teammates' work.

## Choose a task

**P0:** next steps needed for the first trustworthy model comparison. **P1:** enabling work and required final reporting. **P2:** conditional cleanup or optional experiments; do these when their trigger applies. A task's priority reflects current impact, while its dependencies determine when it can run.

Every task distinguishes an **observed problem** (confirmed in this repository), a **future risk** (what could happen if used that way), and an **optional improvement**. An observed code issue is not automatically a major project blocker. Add your name and a PR link when starting; mark done after its checks pass. New file paths are proposed files to create.

| ID | Priority and current impact | Task | Depends on | Owner / status |
|---|---|---|---|---|
| DATA-01 | P0 | [Save source IDs](#data-01) | None | KR-0000 / implemented |
| DATA-02 | P0 | [Handle duplicate records](#data-02) | DATA-01 | KR-0000 / implemented |
| DATA-03 | P0 | [Split whole recordings](#data-03) | DATA-02 | KR-0000 / implemented |
| LABEL-01 | P0 - no reusable label export | [Export versioned pressure labels](#label-01) | DATA-01 | Unclaimed / open |
| FEATURE-01 | P0 - only a tiny feature pilot | [Export PPG-only features](#feature-01) | FIX-01, DATA-01; DATA-03 before selecting features | Unclaimed / open |
| MODEL-01 | P0 - no model benchmark | [Compare constants and two regressors](#model-01) | DATA-03, LABEL-01, FEATURE-01 | Unclaimed / open |
| EVAL-01 | P1 - needed for final results | [Run the final test](#eval-01) | MODEL-01; QC-01 minimal policy before final freeze | Unclaimed / open |
| FIX-01 | P1 - reproducibility support | [Document a working environment](#fix-01) | None; run alongside DATA-01 | Unclaimed / open |
| QC-01 | P1 - shared inclusion policy missing | [Record quality rules and coverage](#qc-01) | DATA-03, LABEL-01; FIX-03 only for zero-run thresholds | Unclaimed / open |
| DOC-01 | P1 - required project handoff | [Keep instructions and claims accurate](#doc-01) | Start now; finish after EVAL-01 | Unclaimed / open |
| FIX-02 | P2 - notebook training integration | [Adopt split access](#fix-02) | DATA-03; only if training from this notebook | Unclaimed / open |
| FIX-03 | P2 - incomplete optional QC metric | [Extend the zero-run audit](#fix-03) | DATA-01 for final export | Unclaimed / open |
| LABEL-02 | P2 - alternative target experiment | [Try labels from complete beats](#label-02) | DATA-03, MODEL-01, QC-01 extended annotations | Unclaimed / open |

**Changed dependencies:** MODEL-01 uses fresh label/feature exports and does not depend on fixing the old exploratory notebook. If the team chooses to reuse that notebook, finish FIX-02 first. The first baseline keeps numeric zeros and uses a documented minimal policy; it does not require a global zero-run audit or a new beat detector. **Beat-based labels** measure each complete heartbeat and summarize those measurements into one pressure pair per window; this is an optional alternative to the selected baseline target.

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
| Saved signals | `data/processed_dataset.npz` has three float64 arrays of shape `(528828, 625)`. Format 2 also saves window/source IDs, split assignments, and the preparation record. No labels or engineered features. |
| Dataset checks | [eda/audit.py](eda/audit.py) and [eda/EDA-readme.md](eda/EDA-readme.md) report completeness, numeric values, duplicates, and agreement with the archive. Original copies remain in the archive; split access excludes the 14 redundant recordings. |
| Plots | [eda/edaplots_rawdata.ipynb](eda/edaplots_rawdata.ipynb) includes distributions, unusual examples, source coordinates, zero cases, and an early peak comparison. |
| Labels and cleaning attempt | [notebooks/read_data.ipynb](notebooks/read_data.ipynb) uses shared peak/valley means and revised zero handling in memory; suitability remains unresolved. |
| Feature starter | [eda/neurokit_feature_starter.ipynb](eda/neurokit_feature_starter.ipynb) has six candidate features and QC fields. Saved results cover **10 windows from four records**, using NeuroKit2 0.2.13. |
| Still missing | Reusable label/feature exports, a trained benchmark, model-selection procedure, and final test results. [requirements.txt](requirements.txt) is effectively empty; [README.md](README.md) is mostly a template. |

The audit found **12,000 records, 528,828 complete windows, and 14 extra exact-record copies containing 700 windows**. The split loader excludes those copies, leaving **11,986 records and 528,128 windows**, before other exclusions.

There are **5,407 PPG windows containing zeros**. Of those, 53 contain at least 32 zeros; those zeros need not be consecutive. Our copy has no observed PPG/ABP NaNs or infinities. Dropped tails total **3,172,500 sample positions per channel**, about **0.95%**. Check these local counts when rebuilding data.

During the original review, `.venv\Scripts\python.exe eda/audit.py` ran successfully and confirmed exact PPG/ABP agreement, in order, between source windows and the archive. Notebook code and saved outputs were read, but plotting/NeuroKit notebooks were not rerun: the inspected environment lacked matplotlib, NeuroKit2, and scikit-learn. Those environment findings are historical. Dataset preparation is now implemented; no model result is implied.

## Rules shared by all tasks

Use `window_id` to join signals, labels, QC, features, and splits. Row numbers can change after filtering. Keep original MAT files intact. The builder replaces the NPZ with added metadata only after checking identical original waveform rows.

Write generated tables to Git-ignored `data/derived/` and compact reports to `reports/`. Commit code, small artificial test datasets, settings, and documentation. Keep waveforms and large feature tables out of Git. Store manifests and checksums in the team's chosen artifact location, and document how to recreate them.

Coordinate changes to shared files such as `eda/audit.py`. Put repeated calculations in reusable functions that the notebooks call.

## Task instructions

<a id="data-01"></a>

### DATA-01: Save a source ID for every window

**Implemented:** [data/load_files.py](data/load_files.py) contains both the streaming builder and the split-reading function. Existing array-returning loader functions remain available.

The archive contains `source_record_ids`, `window_ids`, and `split_assignments` aligned with every original signal row. IDs encode part, zero-based record index, and local window index. Window `j` uses `[j * 625, (j + 1) * 625)`; each recording's `first_npz_row` preserves its original offset.

Embedded `preparation_json` contains the complete recording map, tail counts, per-channel NaN/infinity counts, hashes, source checksums, package versions, and configuration. Malformed shapes fail with a source-specific message. ECG-only NaNs do not exclude PPG/ABP windows.

**Checks:** small boundary/shape/ECG tests, full waveform-byte comparison before replacement, metadata validation, and the existing streaming audit. Full-data results are in [reports/split_summary.md](reports/split_summary.md).

<a id="data-02"></a>

### DATA-02: Keep one copy of each exact recording for modeling

**Implemented:** hash shape, dtype, and all three channels, then directly compare matching records' bytes. Keep the lowest `(part_number, record_index)`. The complete duplicate map is embedded, not limited to the audit's ten-example preview.

Redundant records receive `excluded`; original signal rows stay in the archive. The split loader omits them automatically. Expected exclusion: 14 recordings / 700 windows, leaving 11,986 recordings / 528,128 windows before cleaning.

**Limit:** shifted/partial overlaps are not exhaustively searched; patient identities are unknown.

<a id="data-03"></a>

### DATA-03: Save assignments and provide split access

**Implemented:** `configs/split.json` fixes seed 2026 and approximately 70/15/15 recording proportions, using largest-remainder rounding. Identifiable reviewed records, including the NeuroKit pilot and historical EDA examples, force their canonical records into training; these count toward the training allocation. Window proportions differ.

```python
from data.load_files import iter_split_windows

for batch in iter_split_windows("train", batch_size=1024):
    ppg, abp = batch.ppg, batch.abp
    window_ids = batch.window_ids
    # Retain IDs with features and labels; apply one filtering mask to all.
```

The reader validates assignments, streams PPG/ABP and IDs, and returns only the requested retained records. Batches have at most the requested size. Missing metadata produces an instruction to rebuild. No cleaning or labels are calculated.

**Handoff:** [data/DATA_NOTES.md](data/DATA_NOTES.md) describes access costs and preprocessing boundaries. `python data/example_split_usage.py --split train --limit 1000` demonstrates access and IDs; `validation` and `test` exercise those sets without calculating model scores.

**Evaluation:** overall EDA preceded splitting. These are duplicate-aware, record-disjoint splits; patient independence and external-device generalization are not established.

<a id="label-01"></a>

### LABEL-01: Export versioned blood-pressure answers in a table

**Priority:** P0 - before model training

**Observed problem:** Current notebooks calculate peak/valley mean labels, but no reusable ID-based target table is saved. A max/min export is a separately named proposed baseline.

**Future risk:** Different notebooks could join the wrong rows or silently use different targets. Unreliable ABP can affect labels, but the current evidence does not establish that every max/min label is bad.

**Optional improvement:** Complete-beat median/mean labels are an alternative experiment in LABEL-02. A new detector is not required to export the current method or a separately named max/min comparison.

**Files to create:** `data/derive_labels.py`; generated label tables under `data/derived/`.

**Steps:**

1. Read aligned ABP and window IDs through `iter_split_windows`. Export the current `data/bp_utils.py` peak/valley means under an explicit method name, such as `peak_valley_mean_v1`. Coordinate with the owner of that shared implementation; do not change its detector as part of exporting results.
2. Save `window_id`, split, label version, SBP, DBP, detected peak/valley counts, validity, and failure reason. Require finite ABP and finite targets; missing detections remain invalid rather than receiving a different method's answer.
3. Export one row for every requested retained window, including failures. Use training/validation exports for development; generate reserved test labels only when needed for frozen evaluation. Keep pressure units unchanged and join features by window ID.
4. If the team chooses a whole-window max/min comparison, export it separately as `window_extrema_v1`. Never substitute it for peak/valley means under the same method name. Verify each method against its direct calculation on artificial windows and a reproducible development subset.

**Done when:**

one command reproduces ID-based label tables, failures are explicit, and training selects exactly one documented target version. The split loader supplies 528,128 retained windows across all three sets; the 700 duplicate-copy windows are excluded. New beat detection remains LABEL-02, not a prerequisite for this export.

<a id="feature-01"></a>

### FEATURE-01: Turn the feature starter into reusable code

**Priority:** P0 - needed for the first learned models

**Observed problem:** The feature starter has six candidate features and saved results for only 10 windows from four records. There is no reusable dataset-wide feature export.

**Future risk:** Selecting features from four-record correlations, silently dropping failures, or letting labels enter the input list could produce misleading comparisons. The pilot alone does not show those failures have occurred in a trained model.

**Optional improvement:** More pulse-shape features, multiple filters, and larger-context processing can follow the simple PPG-only export. Correct the claim that filtering preserves shape/amplitude when converting the starter.

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

**Priority:** P0 - first useful model result

**Observed problem:** There is no training module or reproducible comparison against a constant predictor.

**Future risk:** A complicated model can look useful without beating a simple guess. Comparisons can also be unfair if models use different windows or preprocessing is learned from validation/test data.

**Optional improvement:** Large parameter searches, additional model families, and CNNs can follow constants plus two regressors on the same split and windows.

**Files to create:** `models/train_baselines.py`, `configs/baseline.json`, `reports/validation_metrics.csv`; save fitted bundles under an ignored model-output directory.

**Steps:**

1. Join split assignments, duplicate status, versioned labels, features, and keep/exclude decisions by `window_id` and assert that each ID has at most one matching row in each table. Stop with an error if an ID is duplicated or has no expected matching row. Save the list of allowed features and the exact window IDs included in the run.
2. Use the same selected windows for every model. Require labels without NaN/infinity and apply the documented PPG-processing rules. Report excluded failures. Learn missing-feature replacement values and scaling values from training rows only. Apply the same values to validation and test rows.
3. Fit constant training mean and training median predictors per target. Fit Ridge with a small predefined set of alpha values (for example 0.1, 1, 10, 100). Fit one histogram-gradient-boosting regressor per target with a small saved set of parameter combinations; disable automatic early stopping that creates its own random window split.
4. Choose model settings using validation MAE, report RMSE as well, and save per-window validation predictions for all four predictors on the same IDs. Keep the test set out of the script's default evaluation path.
5. Write the comparison table with separate SBP/DBP metrics, target version, feature/QC versions, record/window counts, parameters, seed, and runtime. Save fitted preprocessing and estimators together. Run twice with the same configuration and compare predictions within documented numerical tolerance.

**Done when:**

both regressors and both constants have reproducible validation results on the same selected windows. Include results even if the constants win. Use fresh ID-based exports; fixing the old notebook is required only if its cells are reused. This task uses one explicitly selected label version and a documented inclusion policy while LABEL-02 is still an experiment. The current peak/valley method and an optional max/min comparison must be reported separately.

<a id="eval-01"></a>

### EVAL-01: Choose the final setup and run the reserved test

**Priority:** P1 - required for final results

**Observed problem:** There is no final evaluation script or results table because the models have not been built yet.

**Future risk:** Tuning against the final test or treating neighboring windows as independent subjects would overstate what results establish.

**Optional improvement:** Ablations, 1,000 group-bootstrap resamples, and extended subgroup plots add evidence after the core metrics and coverage counts work. If uncertainty intervals are reported, they must respect groups.

**Files to create:** `models/evaluate.py`, `configs/final_evaluation.json`, `reports/model_comparison.csv`, `reports/error_analysis.md`.

**Steps:**

1. Finish the constants/Ridge/boosting comparison on training/validation data. Additional comparisons listed later are optional; select only those the team implements. For each comparison save IDs of windows included in every method, failures for each method, label version, and the parameters selected. Compare different label definitions against manual references as well as their own regression errors.
2. Save the final preprocessing, label/QC rules, feature list, model settings, and metric definitions in one configuration, and stop adjusting them. Record whether the final model remains trained on train only or is refitted on train+validation after selection; refit every learned transform consistently if using the latter.
3. Add an explicit final-test mode that loads the frozen configuration and saved split. Save predictions once for the chosen models and constant predictors fitted on the same final training data. Compute per-window MAE/RMSE and signed error separately for SBP/DBP.
4. Report record/window counts and exclusions alongside the core metrics. Optional: compute each record's MAE and report their equal-weight mean; for equal-record RMSE, average record MSEs and take the square root. If adding 95% uncertainty intervals, use 1,000 seeded resamples of test split groups, carrying all windows with each sampled group. Use identical resamples for model differences. Otherwise state that uncertainty intervals were not estimated.
5. Save a prediction-error plot and a short error analysis. Optional breakdowns can use predefined BP bands, parts, and PPG quality categories, with counts. List the 20 largest errors and all `predicted_sbp <= predicted_dbp` cases; generate waveform examples without using the findings to quietly retune the model. Document post-test corrections as a new exploratory run.

**Done when:**

saved predictions reproduce every table, any reported uncertainty intervals resample groups rather than rows, included and excluded example counts are reported, and the claim says record-disjoint rather than patient-independent.

<a id="fix-01"></a>

### FIX-01: Install packages and rerun the starter notebook

**Files:** [requirements.txt](requirements.txt), [.gitignore](.gitignore), [data/README.md](data/README.md), [eda/neurokit_feature_starter.ipynb](eda/neurokit_feature_starter.ipynb).

**Priority:** P1 - run alongside the data tasks

**Observed problem:** Requirements are effectively empty; the inspected environment lacks matplotlib, NeuroKit2, and scikit-learn. Saved notebook outputs alone do not provide a reproducible environment.

**Future risk:** Teammates may be unable to reproduce features or models, or may get different outputs from different package versions. This is not evidence of corrupted signals.

**Optional improvement:** A lockfile and automated environment checks can follow documented, tested package versions.

**Steps:**

1. Create a fresh virtual environment and record its Python version. Install numpy, pandas, scipy, h5py, matplotlib, NeuroKit2, scikit-learn, Jupyter, and the kernel/execution packages used to run notebooks.
2. Start with the pilot's recorded `neurokit2==0.2.13`. Install package versions that work together, run `python -m pip check`, and write the successfully tested package versions to `requirements.txt`. Do not copy an unrelated environment's entire package list.
3. Add `.venv/`, `__pycache__/`, and the new generated-data/model directories to `.gitignore`. Keep small source/configuration files trackable.
4. Document setup and run commands that start from the repository root. Run `python eda/audit.py`; execute the NeuroKit starter after restarting the notebook kernel using this environment, not the notebook's saved outputs.
5. Record the selected record IDs, sample count, processing failures, package versions, and elapsed time in a short reproducibility note.

**Done when:**

a second fresh environment installs from the file, `pip check` passes, the audit runs, and the pilot produces a fresh feature table and plots. Record any difference from the saved 10-window output rather than editing results to match it.

<a id="qc-01"></a>

### QC-01: Record inclusion rules, failures, and quality evidence

**Priority:** P1 - document a minimal policy before final reporting

**Observed problem:** Numeric completeness and descriptive EDA are already checked. The project has no shared saved policy for signal exclusions and feature-processing failures; zero meaning and some waveform-quality questions remain unresolved.

**Future risk:** New deletion thresholds could remove real but uncommon pressure values. Unreliable reference signals could affect labels. The proportion of affected model examples is not established yet.

**Optional improvement:** The 200-window annotation study is for developing new exclusion thresholds or beat labels. It is not a prerequisite for the initial versioned-label baseline with a documented inclusion policy.

**Files to create:** `eda/review_windows.ipynb`, `data/derived/review_windows.csv`, `data/derived/review_annotations.csv`, `reports/qc_policy.md`.

**Core steps for the first baseline:**

1. Write `reports/qc_policy.md` with the starting rules: use complete 625-sample PPG/ABP windows from canonical records, require finite PPG/ABP and finite labels, and retain numeric zeros and unusual pressure values unless a justified rule says otherwise. ECG-only missing values do not exclude a PPG/ABP example.
2. Export one QC row per `window_id` with separate PPG/ABP status, feature-processing status, keep/exclude decision, and reason. Specify which missing feature values can be replaced using training-only statistics and which extraction failures exclude a window. Keep failed rows in the log.
3. Report windows and records retained/excluded for each reason, by part and pressure range. Save the exact eligible IDs used by MODEL-01. Agree and version this policy before freezing final evaluation; do not silently change it between models.

**Optional extended review:** Do the following when proposing new exclusion thresholds or developing LABEL-02. Complete FIX-03 before selecting longest-zero-run cases or using a run-length threshold. This study does not block the minimal-policy baseline.

1. Load the saved split and restrict review candidates to train/validation records. Select 200 unique windows from at least 50 records: 80 random examples chosen with a saved seed and 20 additional examples from each of six categories—one zero, multiple zeros, longest runs, lowest PPG range, lowest/highest max/min pressure labels, and smallest/largest pulse pressure. Resolve category overlaps by selecting the next eligible ID; record shortages. Cover all four parts and spread selections across records rather than filling the set from one long recording.
2. Save the chosen IDs, categories, seed, and train/validation designation. Create PPG and ABP plots with window boundaries, one second of available context on either side, exact-zero markers, and source coordinates. Save figures keyed by window ID.
3. Have reviewers enter separate PPG and ABP ratings (`usable`, `uncertain`, `unusable`), reason codes, and notes. Annotate ABP onset/peak/trough sample indices and incomplete boundary beats for LABEL-02. Independently double-review 40 windows, including difficult categories, and save both original ratings and the final agreed decision.
4. Write an exclusion policy that distinguishes input failure from unreliable reference labels. For each proposed rule, calculate agreement with training annotations and the number of retained/rejected windows and records by part and pressure ranges defined using training/validation data. Keep uncertain cases explicitly labeled. Do not fit thresholds on the final test.
5. Apply the chosen rules to the reserved validation annotations without retuning. Save disagreements, included/excluded counts, and BP histograms before and after filtering. If the evidence does not justify a threshold, retain its flag and state that it is not an exclusion rule.

**Done when:**

the baseline policy, per-window decisions, failure reasons, and counts are saved and MODEL-01 uses them consistently. If new thresholds or beat labels are proposed, also save the extended annotations, rule-specific counts, and validation disagreements before adopting them. An unsupported threshold remains a flag, not a deletion rule.

<a id="doc-01"></a>

### DOC-01: Update the notes and make the full project runnable

**Priority:** P1 - finish the handoff as code lands

**Observed problem:** The README has placeholders, the decision log still marks some completed EDA as open, and data notes need to describe the implemented label/split workflow.

**Future risk:** Teammates could follow stale instructions or mistake proposed work and patient-independence claims for established results.

**Optional improvement:** A replay app, extra diagrams, and publication-style polish can follow the reproducible notebook, commands, and result explanation.

**Files:** [README.md](README.md), [data/DATA_NOTES.md](data/DATA_NOTES.md), [eda/EDA-readme.md](eda/EDA-readme.md), [DECISIONS.md](DECISIONS.md), the affected notebooks; create a final end-to-end notebook and `reports/model_card.md`.

**Steps:**

1. Mark completed descriptive EDA as completed in the decision log. Keep label/QC decisions open until their task evidence exists. Link each resolved issue to its code/report instead of leaving contradictory old instructions.
2. Update data notes with the table columns and ID-based joins and heartbeat detection on complete ABP recordings followed by window assignment for the beat experiment. State that known patient mapping and exact-zero semantics are unavailable. Describe the original counts and the counts after removing exact copies separately.
3. Replace README placeholders with actual setup, dataset preparation, feature extraction, training, and evaluation commands. Label proposed commands as proposed until implemented; publish no fabricated accuracy numbers.
4. Create an end-to-end notebook that calls the reusable modules rather than copying their implementations. Include a small trial-run configuration using training/validation records and a full-run configuration using the same saved split.
5. Run the documented small trial run from a fresh environment. Load the saved model bundle and feed a single PPG window from a reserved test record; assert its output matches batch inference without supplying ABP. Add final metrics, exclusions, inference context, and limitations to the model card and presentation.

**Done when:**

a teammate can follow the README to reproduce a small run and understand how the full results were generated; obsolete cleaning claims and stale notebook outputs are removed or explicitly labeled historical.

<a id="fix-02"></a>

### FIX-02: Adopt shared split access before notebook-based training

**Priority:** P2 - only if reusing `notebooks/read_data.ipynb` for training.

The previous 5% deletion and ineffective `fillna()` code have been replaced. Current code uses a 10% zero threshold, nonzero row means for replacement, and shared peak/valley mean labels. Historical saved outputs do not establish current counts.

Before training, use saved split access, retain IDs through filtering, and separate waveform inputs from target columns. Document and evaluate zero handling rather than treating zeros as proven missing data. Keep the shared label method or introduce a separately named comparison; do not silently revert it. Clear stale outputs when rerunning the affected notebook.

**Done when:** notebook training consumes saved splits, feature inputs exclude targets, and filtering preserves identities. Dataset preparation does not depend on rewriting this notebook.

<a id="fix-03"></a>

### FIX-03: Count consecutive zeros in every window

**Priority:** P2 - supporting audit improvement

**Observed problem:** The notebook measures longest zero runs only among windows with at least 32 zeros. The audit counter uses ppg_min == 0, while its separate archive check correctly counts any zero. Current reported counts agree at 5,407.

**Future risk:** The minimum-based counter would miss a future window containing both negatives and zero. Using the restricted longest-run example as a universal cutoff would also be unjustified.

**Optional improvement:** Compute runs for every window when developing a run-length exclusion rule or fuller QC table. This does not block a baseline that keeps zeros and documents that choice.

**Files:** [eda/audit.py](eda/audit.py), [eda/edaplots_rawdata.ipynb](eda/edaplots_rawdata.ipynb); export `data/derived/window_qc.csv`.

**Steps:**

1. Move zero-run calculation outside the `many_zero_indices` selection. For each window, form `mask = ppg == 0`, pad it with `False` on both ends, find transitions, and subtract start indices from end indices. Store the largest run, or zero if there are no runs.
2. Store total zero count, zero fraction, longest run in samples/seconds (`samples / 125`), negative count, NaN/infinity count, range, standard deviation, and largest absolute adjacent-sample change. Keep these as measurements, not deletion decisions.
3. In `audit.py`, replace the zero-containing-window count based on `ppg_min == 0` with `window_zero_counts > 0` or `np.any(ppg == 0, axis=1)`. The old expression misses a window containing both a negative sample and a zero; this is a robustness bug even if current counts agree.
4. Select and plot the longest-run example across **all** windows. Label the old 21-sample result as a result for the restricted 32+-zero subset until the full calculation completes.
5. Add small test datasets for no zeros, one zero, separated zeros, a run touching either boundary, all zeros, and mixed negative/zero values. For `[0,0,1,0,0,0]`, require count 5 and longest run 3.

**Done when:**

QC has one row identified by window ID per original window; total zero-containing counts match 5,407 on the current copy; all test cases pass; and the notebook uses the exported metrics rather than a second inconsistent implementation.

<a id="label-02"></a>

### LABEL-02: Try labels calculated from complete heartbeats

**Priority:** P2 - optional target experiment

**Observed problem:** The exploratory detector finds unpaired peaks and troughs within each window. Its +2.35/-1.76 mmHg differences show target sensitivity, not validated complete-beat labels.

**Future risk:** Promoting that detector without boundary/quality checks could create unreliable new targets. This is a risk when adopting the alternative, not proof that the existing max/min calculation is broken.

**Optional improvement:** Test a complete-beat median/mean target after the source split and first baseline exist. Promote it only with documented evidence; retaining the named max/min baseline is a valid outcome.

**Files:** [eda/audit.py](eda/audit.py), [eda/edaplots_rawdata.ipynb](eda/edaplots_rawdata.ipynb); create `data/abp_beats.py` and a separate beat-label export.

**Steps:**

1. Keep the current `exploratory_beat_labels` explicitly exploratory. Implement a detector that marks the start (onset) of each beat in a continuous ABP record. Save the implementation version, settings, sampling rate, and how it handles pressure units. Store detected onsets in original sample coordinates.
2. Define a beat between two consecutive onsets. Reject intervals with reversed or identical start/end positions. Specify whether the closing onset is included when locating the diastolic trough, and store the exact samples used. Detect on context if needed, but exclude a beat from a target window when its defining boundaries or pressure samples lie outside that window's stored sample interval.
3. Store one row per beat with onset/end indices, SBP/DBP values and locations, duration, quality flags, and accepted/rejected status. For each window, calculate **separate** median and mean labels from the accepted complete beats. Begin with a configurable minimum of two beats; emit invalid status when insufficient beats remain.
4. Use training annotations from QC-01 for development. Before evaluating validation annotations, write the matching tolerance, quality rules, and numerical criteria for accepting the new method in `reports/label_protocol.md`. Report missed/extra beats, onset timing errors, boundary errors, differences from labels marked by reviewers, and the fraction of windows that receive valid labels. Set acceptance criteria before seeing validation results.
5. Save extrema/median/mean comparisons on identical window IDs, including percentiles of the differences and the 20 largest validation disagreements with plots showing the marked points. Keep all failure rows. Never fill an invalid median target with an extrema target under the same version name.

**Done when:**

manually marked examples and artificial boundary tests have recorded outcomes, every beat target is traceable to accepted beats, and a written decision either adopts the new labels or keeps max/min labels. Two independent peak/trough lists and a lower model MAE alone do not close this task.

## Shared tables to build

The archive embeds the recording map and per-window IDs/splits. The following are conceptual fields for later exports; separate record/window CSVs are optional.

| Table | Fields to keep |
|---|---|
| Records | `source_record_id`, part number, zero-based record index, length, complete-window count, tail count, content hash, canonical duplicate ID |
| Windows | `window_id`, source ID, local window index, start sample, exclusive end sample, existing NPZ row index |
| Splits | group ID, train/validation/test assignment, seed, split-policy version |
| Labels | window ID, target version, SBP, DBP, valid-beat count when applicable, validity flag, failure reason |
| QC and features | window ID, separate PPG/ABP flags, processing status, feature version, explicitly named PPG input columns |
| Run details | source checksums, code revision, package/settings versions, allowed feature list, split/label/QC versions, model settings |

## Choices that guide implementation

**Keep the current window size and task.** Predict SBP/DBP for the same five-second interval as the PPG input. This is not future-pressure forecasting. Keep dropping only incomplete tails; do not add padding. Recovering tails is lower priority than exporting labels and preventing related signals from appearing on both sides of an evaluation.

**Keep label versions separate.** Export the current method as `peak_valley_mean_v1`; choose and record the model target version explicitly. `window_extrema_v1` is an optional separate comparison. Test `complete_beat_median_v1`, with beat means as a comparison. Never fill a missing median label with max/min under the same name. The current unpaired peak experiment found median differences of **+2.35 mmHg for SBP and -1.76 mmHg for DBP**. Label choice matters, but these differences do not establish detector accuracy. Run the beat experiment after the first baseline, if time allows. If it remains inconclusive, retain the selected baseline target and report the experiment separately.

**Keep the saved test assignments reserved.** Overall EDA has already happened; disclose that. The implemented 70/15/15 split and seed 2026 apply to canonical recordings. Identified reviewed examples are assigned to training. Window percentages differ because recordings have different lengths. Any extra validation folds must also keep recordings together.

**Support quality rules with evidence.** The meaning of exact PPG zeros is unknown. Do not automatically delete or interpolate them. Compare PPG shape, local variation, abrupt changes, flat sections, and peak counts with the separate ABP reference. Unusual real pressure is not automatically a bad label. Report how each removal rule changes counts and pressure ranges, including counts by recording length. The proposed review size and two-beat minimum are practical starting points, not published standards. Include different beat rates and detector failures in review; expand the sample when a failure category remains unresolved.

**Use the right signal context.** Detect ABP beats on continuous records, then assign complete accepted beats to windows. [PhysioNet's `wabp`](https://physionet.org/content/wfdb/10.7.0/app/wabp.c) is a relevant adult-ABP detector designed for 125 Hz. It is a WFDB program: document how to run or adapt it, its units, startup behavior, and license before adopting it. [Sun et al.](https://www.cinc.org/archives/2006/pdf/0013.pdf) offers starting quality checks; an abnormality flag alone does not prove artifact.

For PPG, start with one 625-sample input window. If filtering uses surrounding or future samples, provide the same context during prediction and describe the system accordingly. A fair recording split does not make that a real-time five-second estimator. ABP may supply labels and offline review, but it must not be required to calculate prediction features, align incoming PPG, or decide whether an incoming PPG signal is acceptable. Report exclusions caused by poor ABP separately from PPG rejection.

**Add features when they help.** Begin with the six existing features and simple PPG summaries. Later candidates include amplitude spread, skewness, time between peaks, pulse amplitude, rise time, width, area, and derivative summaries. The overview suggests around ten features; justify each one instead of adding features to hit a count. Filtering can change amplitude and shape. A few pulse intervals in five seconds do not establish reliable long-duration heart-rate variability (HRV).

**Keep the first model comparison small.** Use constant training mean, constant training median, Ridge, and one histogram-boosting model per target. A bounded random forest is an alternative. See the [histogram boosting documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). Use grouped validation when supported by the installed version, or disable automatic random-window early stopping. Start with a saved subset of development records, then scale. Do not use the overall medians **131.10/62.03 mmHg** as fitted constants; those include reserved test data.

## Model comparisons to run

The first row is the required baseline comparison; the remaining rows are optional follow-ups. Use training/validation data for these choices. Save the same windows for comparisons where possible and report each method's additional failures.

| Comparison | Keep the same | What we learn |
|---|---|---|
| Ridge, boosting, and constants | Split, labels, features, included windows | Whether PPG features help on new recordings |
| Basic summaries and pulse-shape features | Model/settings-search budget, split, labels, included windows | Whether more complicated features help |
| Original and filtered/normalized PPG | Split, labels, shared windows, model budget | Whether extra processing helps |
| Max/min, beat median, and beat mean | Split and model family; show shared and method-specific windows | How the answer definition changes results and data coverage |
| Minimal removals and reviewed QC removals | Label definition and modeling procedure | How much improvement comes with removing difficult examples |

Lower error against a different label does not prove that label is more accurate. Compare labels to manual markings too. A method should not appear better simply because it rejects harder windows. Choose settings and features using validation, then freeze the final setup before testing. Optional repeated validation must keep recordings together. A raw-window Ridge or small 1D CNN can follow once this comparison is reproducible.

Optionally use validation-based permutation importance (shuffle one feature and measure the change in error) or standardized Ridge coefficients to explain what the model uses. Mention correlated features. Show large errors and `SBP <= DBP` predictions without silently clipping them. Record bug fixes prompted by test results; later scores then count as exploratory results.

Resampling whole test groups measures uncertainty while keeping related windows together. It cannot account for unknown cases where separate records came from the same patient. We also lack demographic data for demographic fairness analysis; do not infer demographics from signals.

Use this scope statement: **"Duplicate-aware, record-disjoint evaluation on the UCI release. Patient independence and external-device generalization are not established."** In plain language: we handle known copies and test on different recordings, but cannot claim success on new people or different sensors. A well-run experiment that barely beats the constants, or does not beat them, is still a useful educational result.

## Proposed schedule

These dates follow the overview's October modeling and November presentation milestones. Complete the checks before moving on; December provides handoff buffer time.

| Phase | Proposed dates | Output and completion check |
|---|---|---|
| 1. IDs, duplicates, and splits | Implemented Oct 3 | DATA-01/02/03: IDs and assignments saved; split access excludes exact copies. No exhaustive overlap screen is claimed. |
| 2. Setup, labels, and features in parallel | Oct 1-10 | FIX-01, LABEL-01, FEATURE-01: install commands work; labels/features join by ID. |
| 3. First model comparison | Oct 8-21 | MODEL-01 and core QC-01: constants and two regressors use the same eligible windows and training-only preprocessing. |
| 4. Select and freeze the setup | Oct 19-30 | Select using validation. Add optional QC/label/feature experiments only if useful and feasible; freeze the final procedure. |
| 5. Final evaluation | Nov 1-13 | EVAL-01: reproducible reserved-test metrics, coverage counts, limitations, and error analysis; any uncertainty estimates respect recordings. |
| 6. Handoff | Nov 14-30 | DOC-01: a fresh environment reproduces the comparison and PPG-only prediction; report and presentation are ready. |

**If time is short:** finish source IDs, known-duplicate handling, the saved recording split, versioned label export, basic features, and two regressors plus constants. Document the environment, minimal inclusion policy, exclusion counts, and final results. Fix the old notebook only if reusing it for training. Extended annotations, new beat labels, storage redesign, and extra comparisons can wait.

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
