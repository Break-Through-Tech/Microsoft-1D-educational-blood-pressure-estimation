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
st.write("Choose a five-second optical pulse signal, then run a trained ResNet on CPU.")
st.info("Educational prototype. These estimates are not for medical decisions.")


@st.cache_resource
def predictor_resource():
    # One shared model per server process; bound concurrent CPU work.
    from models.selected_models import load_selected_model
    return load_selected_model("resnet_l1", seed=2026, artifact_dir=ASSETS), Lock()


@st.cache_data
def examples():
    with np.load(ASSETS / "samples/ppg.npz", allow_pickle=False) as data:
        signals, ids = data["ppg"], data["window_ids"].astype(str)
    references = pd.read_csv(ASSETS / "samples/references.csv").set_index("window_id")
    return signals, ids, references


signals, ids, references = examples()
selected = st.selectbox("Example PPG signal", range(len(ids)), format_func=lambda i: f"Example {i + 1}")
window_id = ids[selected]
st.caption("Bundled examples come from training recordings. They demonstrate inference, not unseen-data accuracy.")
st.line_chart(pd.DataFrame({"Time (seconds)": np.arange(625) / 125,
                            "PPG (source units)": signals[selected]}),
              x="Time (seconds)", y="PPG (source units)")

if st.button("Predict blood pressure", type="primary"):
    try:
        with st.spinner("Running live prediction…"):
            predictor, lock = predictor_resource()
            started = perf_counter()
            with lock:
                result = predictor.predict_ppg(signals[selected:selected + 1], [window_id]).iloc[0]
            elapsed = perf_counter() - started
        if result["rejection_reason"]:
            st.warning(f"This signal could not be processed: {result['rejection_reason']}")
        else:
            sbp, dbp = st.columns(2)
            sbp.metric("Estimated systolic pressure", f"{result['predicted_sbp']:.1f} mmHg")
            dbp.metric("Estimated diastolic pressure", f"{result['predicted_dbp']:.1f} mmHg")
            reference = references.loc[window_id]
            st.write(f"Reference for this example: **{reference['sbp']:.1f} / {reference['dbp']:.1f} mmHg**.")
            st.caption(f"Calculated now from the selected waveform in {elapsed:.2f} seconds (excluding model loading).")
    except (OSError, ValueError, RuntimeError) as error:
        st.error("Prediction failed. Please retry or contact the project team.")
        # Detailed diagnostics stay in server logs.
        import logging
        logging.getLogger(__name__).exception("Live prediction failed: %s", error)

st.divider()
st.caption("MVP: standalone ResNet, seed 2026. The recommended three-model ensemble is not loaded in this demo.")
with st.expander("How to interpret this demo"):
    st.write("Input: 625 PPG samples at 125 Hz. The saved pipeline filters the waveform, calculates derivatives, "
             "and applies training-fitted scaling before predicting systolic and diastolic pressure.")
    st.write("The standalone ResNet averaged validation MAE of 11.51 mmHg systolic and 5.97 mmHg diastolic "
             "across three seeds. Those aggregate results do not describe the error on every example. "
             "The test set remains reserved, and patient identities are unavailable.")
