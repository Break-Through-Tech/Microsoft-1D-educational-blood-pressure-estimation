# Gradient boosting: what I did

Predict SBP (upper BP number) and DBP (lower number) from five seconds of PPG.
Start with [the short results page](results.html). It includes the experiments and conclusions.

## What I learned

I kept three main comparisons. Feature and parameter choices used validation data only.

| Question | What I found | Choice |
| --- | --- | --- |
| Do pulse-shape features help? | Validation SBP/DBP MAE fell from 17.19/8.43 to 14.86/7.48. | Use 17 features instead of six. |
| Which settings work best? | Of four settings, learning rate 0.1 and 31 leaves gave the lowest errors for both targets. | Keep those settings, with 100 iterations. |
| Does boosting beat simpler models? | It beat constant guesses and Ridge on the same test windows. | Use boosting; keep the baselines for comparison. |

The model is scikit-learn's `HistGradientBoostingRegressor`, trained separately for SBP and DBP.
It handles missing features without scaling. Ridge uses training-only median filling and scaling, with alpha 100 selected in the original search.
`config.json` contains the chosen settings: absolute-error loss, minimum 20 samples per leaf, L2 = 1, seed 2026, no early stopping.
Earlier small pilots gave mixed results. The main comparisons above use the full saved training/validation splits.

## Data handling

I reuse `data/load_files.py`, `configs/split.json`, and `data/bp_utils.py` 
My cleaning follows `cleaning/clean_data.ipynb`; the first six features follow the existing feature notebook.

1. Keep the existing recording splits, recording/window IDs, and exact-duplicate removal.
2. Check each five-second window (625 samples at 125 Hz). Reject flat/nonfinite signals or more than 5% zeros; interpolate smaller zero gaps and clean PPG with NeuroKit.
3. Extract six signal summaries plus 11 pulse-shape features: timing, widths, areas and slopes. Failed measurements stay missing.
4. Filter ABP at 40 Hz and reuse the team's complete-beat label function. Check label ranges and SBP > DBP. ABP supplies labels only; prediction needs PPG alone.
5. Train on usable, labelled training windows. Save the model before testing. Keep rejected rows and reasons in the preparation CSVs.

PPG eligibility and label validity are checked separately: usable PPG can still receive predictions when its ABP label is invalid. Those rows cannot be scored.

## Final result

Test MAE in mmHg; lower is better. These numbers give every recording equal weight (macro MAE).

| Model | SBP | DBP |
| --- | ---: | ---: |
| Always predict training median | 18.54 | 8.89 |
| Ridge | 17.14 | 8.68 |
| Gradient boosting | **15.10** | **7.62** |

Scored 78,667 of 78,946 test windows from 1,793 recordings. Boosting's window-averaged (pooled) MAE is 14.51/7.24.
The HTML also includes RMSE, recording-bootstrap intervals, pressure ranges and plots.

**Conclusion:** boosting learns useful patterns, but low/high-pressure errors remain large. Recordings are separate; different recordings may belong to the same patient. Labels can contain artifacts. This does not establish new-patient or new-device performance. The test has already been scored; later model changes are exploratory.

## Run it

Run PowerShell commands from the repository root. Put `Part_1.mat` through `Part_4.mat` in `data/raw/` for a fresh run.
Use the existing isolated environment, or create it once with Python 3.11:

```powershell
# Only if the environment does not exist yet:
py -3.11 -m venv gradient_boosting/.venv
.\gradient_boosting\.venv\Scripts\python.exe -m pip install -r gradient_boosting/requirements.txt
```

```powershell
$gbPython = ".\gradient_boosting\.venv\Scripts\python.exe"
& $gbPython -m gradient_boosting run                  # Prepare -> train chosen models -> test -> HTML
& $gbPython -m gradient_boosting verify               # Reproduce predictions and check raw samples
```

`run` reuses checked tables and the saved model when they exist. It does not retune or overwrite the model. A fresh run processes all data and can take time.

```powershell
& $gbPython -m gradient_boosting prepare              # Training/validation feature tables
& $gbPython -m gradient_boosting experiment features  # Six versus 17 features; validation only
& $gbPython -m gradient_boosting experiment tuning    # Four parameter settings; validation only
& $gbPython -m gradient_boosting train                # Fit and save the chosen models
& $gbPython -m gradient_boosting test                 # Score the saved models; never refit
& $gbPython -m gradient_boosting report               # Rebuild short HTML; works from saved summary too
& $gbPython -m unittest discover -s gradient_boosting/tests -v
```

Experiments show saved results when available. Add `--rerun` to recompute and save a separate result without changing the final model. For changed model/data settings, copy `config.json`, choose a new `paths.output_dir`, then use `python -m gradient_boosting --config YOUR_CONFIG run`. Tune on validation before testing a new model.

PPG-only prediction: save an `(n, 625)` NumPy array, then run:

```powershell
& $gbPython -m gradient_boosting predict your_ppg.npy --output data/derived/gradient_boosting/new_predictions.npz
```

## Where to look

| File | Purpose |
| --- | --- |
| `__main__.py` | Commands above |
| `cleaning.py`, `features.py`, `data.py` | Clean signals, measure pulses, prepare checked tables |
| `models.py`, `evaluate.py` | Train/save/predict; calculate errors and coverage |
| `experiments.py`, `report.py` | Two optional validation experiments; short HTML |
| `config.json`, `requirements.txt` | Settings and dependencies |
| `results.html`, `results.json`, `tests/` | Readable results, exact numbers, automated checks |

Generated tables, CSV rejection reasons, model and predictions live under `data/derived/gradient_boosting/`, outside Git. New runs use its `run/` folder. This checkout reuses verified tables from `final_v1/`; keep that folder and `archive/` to reuse this saved run.
The old pilot/repeat/gain reports and patient-audit code are retired from the active package. Original source is backed up in `data/derived/gradient_boosting/archive/before_simplification.zip`; earlier outputs remain under `data/derived/gradient_boosting/` for reference. Teammates' files are unchanged.
