# Dataset Setup

This project uses the **Cuff-Less Blood Pressure Estimation** dataset from the UCI Machine Learning Repository.

The raw dataset is **not stored in this GitHub repository** because the full dataset is several gigabytes in size. Each team member who needs to run the data-loading, preprocessing, or modeling pipeline should download the dataset locally.

## 1. Download the Dataset

Download the dataset from the official UCI repository:

**Dataset:** Cuff-Less Blood Pressure Estimation
**Source:** https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation

The dataset contains four MATLAB v7.3 `.mat` files:

```text
Part_1.mat
Part_2.mat
Part_3.mat
Part_4.mat
```

Each part contains physiological signal records including:

* PPG: photoplethysmogram signal
* ABP: arterial blood pressure signal
* ECG: electrocardiogram signal

For the current project scope, we primarily use the **PPG** and **ABP** signals.

## 2. Place the Files in the Project

After downloading the four files, place them inside the following directory:

```text
data/raw/
```

The expected structure is:

```text
data/
├── README.md
└── raw/
    ├── Part_1.mat
    ├── Part_2.mat
    ├── Part_3.mat
    └── Part_4.mat
```

## 3. Keep the Original Filenames

Do not rename the downloaded `.mat` files.

The data-loading code expects the filenames to follow this convention:

```text
Part_1.mat
Part_2.mat
Part_3.mat
Part_4.mat
```

Using different capitalization, spacing, or naming may cause the loader to fail.

## 4. Loading the Dataset

The MATLAB files use the MATLAB v7.3 format and are loaded in Python using `h5py`.

The team data-loading utility is located at:

```text
[PLACEHOLDER: path/to/loader_file.py]
```

Example usage:

```python
[PLACEHOLDER: loader usage]
```

Once the loader is finalized, this section will include the exact command or Python code required to load the dataset.

## 5. Raw Data and Git

The files inside `data/raw/` should remain local and should **not be committed to GitHub**.

The repository's `.gitignore` should exclude the raw dataset while allowing this README and the directory structure to remain tracked.

Expected `.gitignore` configuration:

```gitignore
data/raw/*
!data/raw/.gitkeep
```

The original UCI files should remain unchanged. Any cleaned, windowed, or otherwise transformed versions of the data should be stored separately from the raw files.

## Dataset Reference

UCI Machine Learning Repository: **Cuff-Less Blood Pressure Estimation**

https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation
