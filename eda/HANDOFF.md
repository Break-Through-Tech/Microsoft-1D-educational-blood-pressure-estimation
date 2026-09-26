# PPG/ABP EDA handoff

This handoff uses the team's merged [loading and labeling notebook](../notebooks/read_data.ipynb) together with [Kushagra's EDA report](kushagra_eda.md). The notebook shows the current baseline workflow; the report checks the data and records questions to settle before model evaluation. No alternate EDA branch is required.

## Current workflow

```text
data/raw/Part_1.mat ... Part_4.mat
  -> data/load_files.py
  -> data/processed_dataset.npz (aligned 625-sample PPG and ABP windows)
  -> notebooks/read_data.ipynb (whole-window ABP max/min labels joined to PPG)
```

One source record is one continuous recording inside a raw file. A record may yield several five-second windows. It is not known whether each record represents a different patient. The processed archive is Git-ignored and must be generated locally from the raw files; see [dataset setup](../data/README.md).

The current PPG and ABP arrays each contain **528,828 windows of 625 samples**. The [reproducible audit](kushagra_eda.py) compared every processed PPG and ABP window with windows independently produced from the raw records: both channels match exactly and in order. The team's processed file is suitable for continuing the current PPG/ABP analysis.

## Findings to carry forward

| Finding | Why it matters |
| --- | --- |
| 12,000 source records produce 528,828 complete windows. | Each source record can contribute multiple related examples. |
| No nonfinite PPG or ABP samples were observed. | Missing-value imputation is not currently needed for these channels. |
| 5,407 windows contain at least one zero PPG sample. | A zero has not been established as a missing-value marker; inspect before applying a removal rule. |
| 14 source-record duplicate hash matches were found, including cross-part matches. | Duplicates and related windows can leak across a random train/test split. |
| The notebook's whole-window ABP max/min labels have medians of 131.10/62.03 mmHg. | Keep this simple label method as the documented baseline. |
| In a defined 20,263-window sample, whole-window extrema differ from median detected beat extrema by a median of +2.35 mmHg SBP and -1.76 mmHg DBP. | Beat-based labels are worth comparing, but the exploratory detection settings are not final labels. |

Full methods, distributions, and limitations are in the [report](kushagra_eda.md); exact counts are in the [JSON summary](kushagra_eda_summary.json).

## What is settled and what is open

**Use now:** the team's 625-sample windowing and aligned processed PPG/ABP arrays; whole-window ABP max/min as a baseline label calculation.

**Do not treat as settled:** the notebook's zero-handling cells. They drop 53 rows by an unvalidated rule, then call `fillna` on zero values, which are not NaNs. The saved output shows a pandas chained-assignment warning. These cells do not save a cleaned dataset. The notebook itself remains unchanged by this handoff.

Before training or reporting model performance, agree on:

1. How each window will retain `part_number`, `record_index`, and `window_index`; how duplicate source records will be grouped; and what information, if any, identifies patients.
2. Which PPG/ABP quality rules to apply, especially for zeros and unusual waveforms. Record the number of windows removed by each rule.
3. Whether max/min labels remain the training labels or will be compared with a validated beat-based method.
4. A group-aware train/validation/test split so adjacent or duplicate recordings cannot appear on both sides of an evaluation.

This handoff is an audit and a current baseline, not a finalized cleaned or model-ready dataset.
