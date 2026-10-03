# Dataset Setup

This project uses the UCI [Cuff-Less Blood Pressure Estimation](https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation) dataset. The original files are several gigabytes, so they remain local and are not committed to Git.

See [DATA_NOTES.md](DATA_NOTES.md) for the raw and processed schemas, windowing math, signal roles, validation findings, and open preprocessing questions.

## Expected Directory Structure

Download the four MATLAB v7.3 files and keep their original names:

```text
data/
|-- README.md
|-- DATA_NOTES.md
|-- load_files.py
`-- raw/
    |-- Part_1.mat
    |-- Part_2.mat
    |-- Part_3.mat
    `-- Part_4.mat
```

The raw files are loaded with `h5py`. MATLAB v7.3 files use HDF5 internally and are not supported by the usual `scipy.io.loadmat` workflow.

## Environment And Commands

From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install h5py numpy pandas scipy jupyter
python data/load_files.py
jupyter notebook notebooks/read_data.ipynb
```

`load_files.py` reads from `data/raw/` and creates `data/processed_dataset.npz` with original signal rows, stable identities, and saved recording-based splits. It streams the build and safely replaces the archive only after verification. Raw files and the archive remain Git-ignored. Existing notebooks still read all original rows; use the split loader for model development:

```python
from data.load_files import iter_split_windows

for batch in iter_split_windows("train"):
    ppg, abp = batch.ppg, batch.abp
    window_ids = batch.window_ids
    # Retain these IDs with features, labels, and filtering decisions.
```

Use `validation` to choose settings and `test` only for the frozen final model. Duplicate copies are excluded automatically. Scaling/imputation must be fitted on training data only. Details and limitations are in [DATA_NOTES.md](DATA_NOTES.md).

```powershell
python data/example_split_usage.py --split train --limit 1000
python -m unittest discover -s tests -v
```

Use `notebooks/read_data.ipynb`, not a file under `.ipynb_checkpoints/`. The checkpoint directory is Jupyter's automatic recovery area.
