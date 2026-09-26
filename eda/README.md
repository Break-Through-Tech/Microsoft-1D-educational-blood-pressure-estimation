# PPG/ABP EDA results

`data/processed_dataset.npz` is the team's **windowed data file**: `data/load_files.py` reads the four downloaded `.mat` files and saves synchronized PPG, ABP, and ECG arrays with 625 samples per row (five seconds at 125 Hz). The `.npz` file has no SBP/DBP labels and is Git-ignored. The downloaded files are called *raw* in this repository, although [UCI says the source dataset was already preprocessed](https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation).

The merged `clean-data` change was to [notebooks/read_data.ipynb](../notebooks/read_data.ipynb). It loads the windowed file, calculates a **baseline** SBP/DBP pair as the maximum/minimum ABP sample in each window, and tries a zero-PPG cleanup. It does not save a new dataset. This EDA did **not** merge code into that notebook; [audit.py](audit.py) independently checked its inputs and label definition.

## Findings from all four files

| Result | Why it matters |
| --- | --- |
| 12,000 source recordings yield **528,828** complete five-second windows. The local `.npz` PPG/ABP rows match those raw-derived windows exactly. | The existing loader output is consistent with the downloaded files; it is fine to use for PPG/ABP work. |
| **No NaN or infinite PPG/ABP samples.** **5,407** windows contain a zero PPG sample; **1,419** have just one zero, while **53** have at least 32 zeros. | A blanket rule that removes every zero-containing window is not supported. Inspect examples and decide a signal-quality rule. |
| **14 duplicate source-record copies**, including **13 across files**, account for **700 windows in the repeated copies**. For example, zero-based Part 1 record 1055 matches Part 3 record 2965. | Preserve source-record identity and keep repeated recordings together when creating train/test splits. The current `.npz` does not store that identity. |
| Baseline max/min ABP labels have medians of **131.10/62.03 mmHg**. In an exploratory 20,263-window sample, max/min differs from median detected beat extrema by **+2.35/-1.76 mmHg** at the median. | Label choice changes the targets. The exploratory beat detector is not validated and should not replace the baseline yet. |

ABP systolic peaks and diastolic troughs are the relevant physiological points. For a multi-beat window, taking its single highest and lowest sample is a simple baseline; published PPG/ABP benchmarks also detect beats, apply quality checks, and aggregate beat-level values within a window ([example study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10030661/)). The team's max/min calculation is useful for starting analysis, but its exact labels have not been validated as the final choice.

**Before model evaluation:** decide how to handle zero PPG samples and label extraction, then attach source-record IDs to windows so training and test sets can be separated by recording. The notebook's `fillna` step cannot replace zeros, and it does not create a saved cleaned dataset.

To reproduce these counts, run `.venv\Scripts\python.exe eda/audit.py` from the repository root after following [dataset setup](../data/README.md). The script prints a short summary and does not modify the team's data.
