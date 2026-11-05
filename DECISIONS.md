# Project decisions

- Use five-second PPG windows at 125 Hz. ABP supplies reference pressures; ECG is not a model input.
- Exclude known exact duplicate recordings and keep each recording in one split. Patient identities are unavailable.
- Derive reference pressures from complete ABP beats using the accepted preprocessing configuration.
- Fit preprocessing on training data only. Reuse selected settings and durations for reproduction.
- Recommend the 80% multi-task ResNet, 13% short CNN and 7% LightGBM ensemble. Reported results average metrics from three seed-specific fits.
- Reserve the test set. Supported evaluation accepts training and validation only.
- Keep shared code and selected results in Git. Distribute saved models and optional history separately.

See [SETUP.md](SETUP.md) for the workflow and README for findings.
