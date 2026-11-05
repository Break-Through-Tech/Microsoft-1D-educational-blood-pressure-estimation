# Setup and model workflow

Clone the released version below, then run all commands from its root folder. This checks out the release tag, which is suitable for running the demo; create a working branch before making contributions. The notebook imports the shared Python files, so keep it alongside the repository.

```powershell
git clone --branch modeling-v1 https://github.com/Break-Through-Tech/Microsoft-1D-educational-blood-pressure-estimation.git
cd Microsoft-1D-educational-blood-pressure-estimation
```


Use Python **3.13**. From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, open the notebook and select `.venv` as its Python kernel. No hosted notebook account is needed.

Download **`selected-v1.zip`** and **`SHA256SUMS.txt`** from the [modeling-v1 release](https://github.com/Break-Through-Tech/Microsoft-1D-educational-blood-pressure-estimation/releases/tag/modeling-v1). Move both downloaded files into the repository root, then run:

```powershell
Get-FileHash .\selected-v1.zip -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
Expand-Archive .\selected-v1.zip -DestinationPath .\artifacts
```

Compare the ZIP hash with its line in the checksum file. The ZIP already contains the `selected-v1` folder, so extract into `artifacts`, not into `artifacts/selected-v1`. The result should be:

```text
Microsoft-1D-educational-blood-pressure-estimation/
  artifacts/
    selected-v1/
      config.json
      manifest.json
      components/
      samples/
  notebooks/
    project_walkthrough.ipynb
```

Now open the notebook and choose **Run All**:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks/project_walkthrough.ipynb
```

The default notebook displays saved findings and predicts a small set of included training examples on CPU. Full preparation and training are disabled until explicitly enabled. Raw MATLAB files are needed only for full reproduction.

```powershell
.\.venv\Scripts\python.exe -m models.selected_models predict --input artifacts/selected-v1/samples/ppg.npz --output data/derived/demo_predictions.csv
```

Use a new output filename when rerunning. Prediction defaults to the recommended ensemble and seed 2026.

## Prepare data and train locally

Download the [UCI dataset](https://archive.ics.uci.edu/dataset/340/cuff%2Bless%2Bblood%2Bpressure%2Bestimation) and place Part_1.mat through Part_4.mat in data/raw/. These large files are needed only for preparation and training. Then run from the repository root:

```powershell
$projectPython = ".\.venv\Scripts\python.exe"
& $projectPython data/load_files.py
& $projectPython -m cleaning.clean_data --config configs/pipeline.json --output data/derived/prepared --workers 4
& $projectPython -m cleaning.verify_artifacts data/derived/prepared
& $projectPython -m models.selected_models prepare --data-dir data/derived/prepared --output data/derived/selected_inputs --workers 4
& $projectPython -m models.selected_models train --data-dir data/derived/prepared --inputs-dir data/derived/selected_inputs --output data/derived/reproduction_v1 --model recommended
& $projectPython -m models.selected_models evaluate --data-dir data/derived/prepared --inputs-dir data/derived/selected_inputs --artifacts data/derived/reproduction_v1 --output data/derived/evaluation_v1 --split train validation
```

The completed cleaning/export took about 50 minutes on the original machine. CPU training can take longer than the recorded GPU fits. The notebook explains each optional step before its call.

Training reuses frozen settings and per-seed durations. `--model all` reproduces all three supported choices, sharing components within the run. `--seeds 2026 2027 2028` uses all recorded seeds. `--device cuda` is optional. Choose new output directories.


## Evaluate downloaded models

After preparing data and selected inputs, evaluate the downloaded bundle without training:

```powershell
& $projectPython -m models.selected_models evaluate --data-dir data/derived/prepared --inputs-dir data/derived/selected_inputs --artifacts artifacts/selected-v1 --output data/derived/saved_model_evaluation --split train validation
```

## Optional NVIDIA GPU

For a CUDA-compatible NVIDIA GPU, install the CUDA wheel instead of the CPU wheel before installing the other requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Add `--device cuda` to training, evaluation or prediction commands. CPU remains the default.

## Files and downloads

`selected-v1.zip` contains trained models, configuration and example PPG. Download it to use the models without training. `evaluation-v1.zip` contains saved predictions for deeper review; the history archive contains earlier experiments. Neither is required for prediction. Checksums verify that downloaded files are intact.

Prediction input is an NPZ file with `ppg` (one row of 625 samples per window, sampled at 125 Hz) and unique `window_ids`. Output contains those IDs, estimated SBP/DBP in mmHg, and rejection reasons when applicable.

JSON files hold settings and file inventories used by the code. A manifest lists expected files and their checksums. Virtual environments and caches remain local; teammates need only the single environment above.

For Python inference, reuse a loaded model across calls:

```python
from models.selected_models import load_selected_model
predictor = load_selected_model()
predictions = predictor.predict_ppg(ppg, window_ids)
```

The predictor applies the saved preprocessing. Pass source-amplitude PPG, not ABP or already extracted features. `data.bp_utils` is the shared `data/bp_utils.py` module used to derive training reference pressures from ABP.

## Verification

Run the retained synthetic tests with `.\.venv\Scripts\python.exe -m unittest discover -s tests`. For a saved-model replay, run `.\.venv\Scripts\python.exe -m tests.check_selected_artifacts --bundle artifacts/selected-v1 --output data/derived/model_audit.json`.
