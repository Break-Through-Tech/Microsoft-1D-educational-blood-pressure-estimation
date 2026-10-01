# PPG/ABP exploratory data analysis

Run `.venv\Scripts\python.exe eda/audit.py` from the repository root after following [dataset setup](../data/README.md). The audit reads the four downloaded `.mat` files and the local `data/processed_dataset.npz`; it does not change them. The downloaded files are called *raw* here, although [UCI describes the source dataset as already preprocessed](https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation).

## Output from the latest full run

```text
PPG/ABP dataset audit (125 samples/second; 625 samples = 5 seconds)

Window completeness
  Source recordings examined: 12,000
  Complete 625-sample windows kept: 528,828
  Incomplete or empty windows saved: 0 (the loader keeps only complete windows)
  Recordings with a short trailing segment discarded: 10,082
  Samples discarded in those tails: 3,172,500 per channel
  A discarded tail has fewer than 625 samples; it is not a saved window.

Signal values
  PPG / ABP NaN or infinite samples: 0 / 0
  Complete PPG windows containing a numeric zero: 5,407 (1.02%)
    Exactly one zero: 1,419; at least 32 zeros: 53
  A numeric zero is a sample value, not an empty or incomplete window.

Repeated recordings
  Identical source-record copies found: 14 (13 across part files)
  Windows in those repeated copies: 700
  Example, zero-based indices: Part_1.mat record 1055 = Part_3.mat record 2965
  Keep matching recordings together in train/test splits to avoid leakage.

Provisional ABP labels
  Median of window maxima (SBP): 131.10 mmHg
  Median of window minima (DBP): 62.03 mmHg
  In 20,263 sampled windows, median difference from detected beat medians:
    Window maximum minus beat-peak median: +2.35 mmHg
    Window minimum minus beat-trough median: -1.76 mmHg
  The beat detector is exploratory; these differences do not validate new labels.

Existing processed data file
  PPG / ABP arrays match raw-derived windows in order: yes / yes
  This checks loading consistency, not signal quality.
```

## How to read the results

**Window completeness.** A source recording has a variable number of samples. [load_files.py](../data/load_files.py) cuts each recording into non-overlapping, five-second windows and discards its remaining 1–624 samples. Thus all **528,828 saved windows are complete**; **10,082 recordings have a discarded short tail**, totaling 3,172,500 discarded sample positions per channel. These tails are not missing samples inside saved windows.

**Signal values.** There are no NaN or infinite PPG/ABP samples in the source records. The 5,407 PPG windows contain at least one *numeric zero*, which is different from an empty window. A single zero and a run of zeros may need different handling. The audit does not establish why those zeros occur or a rule for discarding them.

**Repeated recordings.** The audit hashed the full bytes of each three-channel source recording. Fourteen later recordings have the same full-record hash as an earlier one, including 13 matches across part files. The **700 windows** count covers the later copies only; their matching originals also contribute windows. These matches indicate repeated recordings, not merely similar blood-pressure values. Why the released files repeat them is unknown. The current `.npz` omits record IDs, so a random row split can put repeated windows on both sides. Preserve source-record IDs and group matching records when creating train/test splits. The files do not provide a verified patient-level split here.

**ABP labels.** The merged [read_data.ipynb](../notebooks/read_data.ipynb) computes a baseline label from the maximum and minimum ABP sample in each window; it does not save a labeled dataset. The audit reports the median of those window labels over the dataset. It also compares window extrema with median detected beat extrema in a sample. The +2.35/-1.76 mmHg values are medians of **paired differences**, not proposed corrections. The simple beat detector has no clinical quality validation, so label selection remains open.

**Processed file check.** `data/processed_dataset.npz` contains aligned PPG, ABP, and ECG arrays, each with 625 samples per row and no stored SBP/DBP labels or source-record IDs. The audit checked that its PPG and ABP arrays match the windows reconstructed from the source files exactly and in order. This confirms the local loader output, not the physiological quality of every window.

## Before modeling

Decide how to flag poor PPG/ABP windows and derive ABP labels. Add source-record identity when preparing split-ready data so repeated recordings and nearby windows can remain together. Keep these choices explicit in the data preparation step; this audit only describes the current data.
