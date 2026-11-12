# CufflessAI live prediction MVP

Select a bundled five-second PPG waveform and click **Predict blood pressure**.
The app runs the standalone ResNet (seed 2026) on CPU, using the project's shared
preprocessing and prediction code. Outputs are calculated live; the bundle has
no saved predictions. It contains one checkpoint and twelve training examples,
about 288 KB total. Reference pressures are displayed only for comparison.

The checkpoint and PPG examples are copied unchanged from the `modeling-v1`
release's `selected-v1` bundle. The demo config retains only `resnet_l1`, seed
2026; reference columns are extracted from the release's example table. The
local manifest records checksums, and the shared loader verifies the config and
checkpoint before loading.

## Run locally

From the repository root, using Python **3.13**:

```powershell
python -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r demo/requirements.txt
.\.venv-demo\Scripts\python.exe -m streamlit run demo/streamlit_app.py
```

This separate environment installs CPU-only PyTorch. It does not require the
dataset downloads, LightGBM, NeuroKit, notebook packages, or a GPU. Some training
dependencies remain because the app reuses the supported loader.

Verify the UI and bundled inference:

```powershell
.\.venv-demo\Scripts\python.exe -m unittest tests.test_demo_app -v
```

## Deploy to Streamlit Community Cloud

1. Push the demo and shared repository code to a GitHub branch.
2. Open https://share.streamlit.io/ and choose **Create app**.
3. Select the repository and that branch; set the main file to
   `demo/streamlit_app.py`.
4. In **Advanced settings**, choose **Python 3.13**, then deploy.
5. Select an example and click **Predict blood pressure**. Confirm that two
   pressure estimates appear; switch examples and predict again.

Community Cloud finds `demo/requirements.txt` beside the entrypoint before the
root requirements file. Do not select the training requirements. Model assets
are tracked with the demo, so startup needs no release download or credentials.
Repository administrator access may be required by Streamlit. If the fellowship
organization cannot grant it, an authorized fork is another deployment option.

## Hosting constraints and incremental rollout

Checked November 11, 2026: Streamlit's current documentation lists approximately
690 MB minimum to 2.7 GB maximum RAM and up to two CPU cores. These figures are
explicitly variable, not a guaranteed allocation. Only a successful deployment
and live request can establish fit for this account. Apps sleep after 12 idle
hours; open and test the app before presenting.

Local CPU-only Windows smoke test: the complete UI test process peaked at
449 MiB physical RAM (about 1.46 GiB peak Windows committed memory). The first
click, including model imports/loading, took 9.8 seconds; a subsequent click
took 0.37 seconds. Windows memory accounting differs from Linux container
accounting, so these numbers are evidence for trying deployment, not a promise
that the lowest cloud allocation is sufficient.

Start with this single model. After a cloud smoke test, check the app logs and
memory behavior before adding the short CNN, then LightGBM/the full ensemble.
Keep one cached model per process, serialize inference, and bound requests to
one bundled waveform. Do not load full datasets or train on the host.

The bundled examples are training data and do not demonstrate generalization.
This is an educational prototype, not a clinical tool. The held-out test set
remains untouched.

Sources:
- [Streamlit resources and sleep policy](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Dependency file discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Python selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Repository permissions](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)
