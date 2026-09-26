# PPG/ABP EDA: start here

**Read this page first.** The team's merged [read_data.ipynb](../notebooks/read_data.ipynb) shows the current baseline. This folder checks that baseline and records what still needs a decision. For full methods and distributions, see [METHODS.md](METHODS.md).

## What each piece does

| Owner/work | Files | Purpose |
| --- | --- | --- |
| Earlier team loading work | [data/load_files.py](../data/load_files.py) | Cuts each variable-length raw recording into aligned 625-sample PPG, ABP, and ECG windows and writes `data/processed_dataset.npz`. |
| Merged `clean-data` branch | [notebooks/read_data.ipynb](../notebooks/read_data.ipynb) | Reads those windows, creates baseline SBP/DBP labels using each ABP window's maximum/minimum, and experiments with zero-PPG handling. It does not save a cleaned dataset or train a model. |
| This branch, `kushagra-eda` | [audit.py](audit.py), [results.json](results.json), [METHODS.md](METHODS.md) | Audits raw PPG/ABP records, checks the processed archive, and compares baseline labels with an exploratory beat-extrema method. |

**These were combined as a handoff, not merged into one notebook.** This branch starts from `main`, which already contains `clean-data`. We left the teammate's notebook unchanged and kept the audit separate so its results and caveats are easy to review.

## Main findings

- The processed PPG and ABP arrays each contain **528,828 five-second windows**. Every window matches the raw-derived windows exactly and in order, so the team's processed file is sound for current PPG/ABP analysis.
- PPG and ABP have no nonfinite samples. **5,407 windows contain zero PPG samples**; zero has not been established as a missing-value marker.
- **14 source-record hash matches** suggest duplicate recordings, including across raw files. Windows from one record or duplicate records should not be split across training and test sets.
- Whole-window max/min is a useful **baseline label method**. The sampled beat-extrema comparison changes labels, but its settings are exploratory and are not approved training labels.

## Decisions before model evaluation

Preserve each window's source-file, record, and window indices; define a group-aware split that handles duplicates; inspect zero and unusual waveforms before choosing cleaning rules; and agree on the final ABP label method. The notebook's zero-handling cells are experimental: `fillna` cannot replace zeros, and the saved output includes a pandas chained-assignment warning.

The raw `.mat` files and generated `.npz` are Git-ignored. To reproduce the audit after [setting up the dataset](../data/README.md), run `.venv\Scripts\python.exe eda/audit.py` from the repository root. Its output is [results.json](results.json).
