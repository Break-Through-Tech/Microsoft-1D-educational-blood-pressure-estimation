"""Minimal live CPU demo. Run from the repository root with streamlit run."""
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"

st.set_page_config(page_title="CufflessAI | Live prediction", page_icon="〰️")
st.title("CufflessAI")
st.subheader("From a pulse waveform to a pressure estimate")
st.write("Choose a five-second optical pulse signal, then run a trained ResNet and Short CNN on CPU.")
st.info("Educational prototype. These estimates are not for medical decisions.")

LIVE_MODELS = {"resnet_l1": "ResNet (L1)", "cnn_c": "Short CNN"}


@st.cache_resource
def predictor_resource(model):
    # One shared copy of each model per server process, cached separately.
    from models.selected_models import load_selected_model
    return load_selected_model(model, seed=2026, artifact_dir=ASSETS)


@st.cache_resource
def prediction_lock():
    # Bound concurrent CPU work across sessions.
    return Lock()


def log_failure(model, error):
    # Detailed diagnostics stay in server logs.
    import logging
    logging.getLogger(__name__).exception("Live prediction failed for %s: %s", model, error)


@st.cache_data
def examples():
    with np.load(ASSETS / "samples/ppg.npz", allow_pickle=False) as data:
        signals, ids = data["ppg"], data["window_ids"].astype(str)
    references = pd.read_csv(ASSETS / "samples/references.csv").set_index("window_id")
    return signals, ids, references


# Loaders for saved validation results; they run no model.
REPORTS = ROOT / "reports/final"
COMPARED_MODELS = {
    "training_median": "Training-median baseline",
    "lightgbm_16": "LightGBM (16 PPG features)",
    "cnn_c": "Short CNN",
    "resnet_l1": "ResNet (L1, standalone)",
    "recommended": "Recommended ensemble",
}
ENSEMBLE_PARTS = {"multitask_resnet": "multi-task ResNet", "cnn_c": "Short CNN", "lightgbm_16": "LightGBM"}
METRIC_COLUMNS = {"sbp_mae": "SBP MAE", "dbp_mae": "DBP MAE", "sbp_rmse": "SBP RMSE", "dbp_rmse": "DBP RMSE"}


@st.cache_data
def validation_comparison():
    metrics = pd.concat([pd.read_csv(REPORTS / "selected_metrics.csv"),
                         pd.read_csv(REPORTS / "cnn_metrics.csv")], ignore_index=True)
    validation = metrics[metrics["split"].eq("validation")]
    return validation.groupby("model")[list(METRIC_COLUMNS)].mean().reindex(list(COMPARED_MODELS))


@st.cache_data
def ensemble_weights():
    import json
    config = json.loads((ROOT / "configs/selected_models.json").read_text())
    return config["models"]["recommended"]


signals, ids, references = examples()
selected = st.selectbox("Example PPG signal", range(len(ids)), format_func=lambda i: f"Example {i + 1}")
window_id = ids[selected]
st.caption("Bundled examples come from training recordings. They demonstrate inference, not unseen-data accuracy.")
st.line_chart(pd.DataFrame({"Time (seconds)": np.arange(625) / 125,
                            "PPG (source units)": signals[selected]}),
              x="Time (seconds)", y="PPG (source units)")

if st.button("Predict blood pressure", type="primary"):
    window = signals[selected:selected + 1]  # Both models receive this same window.
    results = {}
    with st.spinner("Running live prediction…"):
        predictors = {}
        for model in LIVE_MODELS:
            try:
                predictors[model] = predictor_resource(model)
            except (OSError, ValueError, RuntimeError) as error:
                log_failure(model, error)
        started = perf_counter()
        with prediction_lock():
            for model, predictor in predictors.items():
                try:
                    results[model] = predictor.predict_ppg(window, [window_id]).iloc[0]
                except (OSError, ValueError, RuntimeError) as error:
                    log_failure(model, error)
        elapsed = perf_counter() - started
    rejected = [result["rejection_reason"] for result in results.values() if result["rejection_reason"]]
    if not results:
        st.error("Prediction failed. Please retry or contact the project team.")
    elif rejected:
        st.warning(f"This signal could not be processed: {rejected[0]}")
    else:
        for column, (model, label) in zip(st.columns(2), LIVE_MODELS.items()):
            column.markdown(f"#### {label}")
            if model in results:
                column.metric("Estimated systolic blood pressure", f"{results[model]['predicted_sbp']:.1f} mmHg")
                column.metric("Estimated diastolic blood pressure", f"{results[model]['predicted_dbp']:.1f} mmHg")
            else:
                column.warning(f"The {label} is unavailable right now. Other results are unaffected.")
        reference = references.loc[window_id]
        st.write(f"Reference for this example: **{reference['sbp']:.1f} / {reference['dbp']:.1f} mmHg**.")
        st.caption(f"Both models used the same selected waveform, calculated now in {elapsed:.2f} seconds "
                   "(excluding model loading). This example comes from a training recording, not unseen patient data.")

st.divider()
st.caption("Live models: standalone ResNet (L1) and Short CNN (`cnn_c`), seed 2026. "
           "The recommended three-model ensemble is not run live in this demo.")
with st.expander("How to interpret this demo"):
    saved = validation_comparison()
    resnet, cnn = saved.loc["resnet_l1"], saved.loc["cnn_c"]
    beaten = [COMPARED_MODELS[name] for name in ("training_median", "lightgbm_16")
              if (cnn[["sbp_mae", "dbp_mae"]] < saved.loc[name, ["sbp_mae", "dbp_mae"]]).all()]
    resnet_lower = (resnet[["sbp_mae", "dbp_mae"]] < cnn[["sbp_mae", "dbp_mae"]]).all()
    cnn_note = ""
    if beaten:
        cnn_note = f" It had lower errors than the {' and '.join(beaten)}"
        cnn_note += ", although the ResNet's errors were lower." if resnet_lower else "."
    st.write("**Input:** 625 PPG samples at 125 Hz (5 seconds). The saved pipeline filters the waveform, "
             "calculates its first and second derivatives, and applies training-based scaling before "
             "predicting systolic and diastolic blood pressure.")
    st.write(f"**ResNet (L1):** validation mean absolute error (MAE) of **{resnet['sbp_mae']:.2f} mmHg systolic** "
             f"and **{resnet['dbp_mae']:.2f} mmHg diastolic**.")
    st.write(f"**Short CNN:** validation MAE of **{cnn['sbp_mae']:.2f} mmHg systolic** and "
             f"**{cnn['dbp_mae']:.2f} mmHg diastolic**.{cnn_note}")
    st.write("**How to read the predictions:** both models receive the same selected PPG example and estimate "
             "blood pressure independently. Compare each estimate with the shared reference shown with the "
             "predictions: the closer estimate has the smaller error for that example.")
    st.write("These validation results average three training seeds and do not describe the error on every "
             "example. The bundled examples come from training recordings, so they are not evidence of "
             "performance on new patients. The test set remains reserved, and patient identities are unavailable.")
    st.write("See **How the models compare** below for the full results.")

st.divider()
st.subheader("How the models compare")
st.caption("Validation results averaged over three training seeds (the constant baseline needs one). "
           "These are not final test results: the test set remains reserved.")
comparison = validation_comparison()
table = comparison.rename(index=COMPARED_MODELS, columns=METRIC_COLUMNS).rename_axis("Model").reset_index()
st.dataframe(table.round(2), hide_index=True)
st.caption("Errors in mmHg on the validation split; lower is better.")
st.image(str(REPORTS / "plots/selected_comparison.png"), caption="Validation MAE by model (saved project chart).")

with st.expander("What MAE and RMSE mean"):
    st.write("**MAE** (mean absolute error) is the average distance between the estimate and the reference "
             "pressure, in mmHg. **RMSE** (root mean squared error) also measures typical error, but squares "
             "each error first, so a few large misses raise it more than MAE.")

cnn, baseline = comparison.loc["cnn_c"], comparison.loc["training_median"]
better = [COMPARED_MODELS[name] for name in comparison.index if comparison.loc[name, "sbp_mae"] < cnn["sbp_mae"]]
with st.expander("How the Short CNN compares"):
    st.write(f"The Short CNN (`cnn_c`) expands the project's original 1D CNN. On validation it averaged "
             f"{cnn['sbp_mae']:.2f} mmHg SBP and {cnn['dbp_mae']:.2f} mmHg DBP MAE, "
             f"{1 - cnn['sbp_mae'] / baseline['sbp_mae']:.0%} and {1 - cnn['dbp_mae'] / baseline['dbp_mae']:.0%} "
             f"lower than always predicting the training median.")
    if better:
        st.write(f"Lower SBP MAE than the Short CNN: {', '.join(better)}.")
    st.write("The ResNet (L1) row is a standalone model. It is not the multi-task ResNet inside the ensemble, "
             "which has no separate saved result here.")
    weights = ensemble_weights()
    parts = " + ".join(f"{weight:.0%} {ENSEMBLE_PARTS.get(name, name)}" for name, weight in weights.items())
    st.write(f"The recommended ensemble averages its components' estimates in mmHg: {parts}. "
             "The weights come from the project configuration; how they were selected is not documented "
             "in the repository.")
    st.write("None of these results establish clinical accuracy or performance on new patients.")
