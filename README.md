# NeuroKit2 PPG morphological features

`extract_ppg_features` converts one 625-sample PPG window at 125 Hz into
window-level features. It first uses NeuroKit2 0.2.13 with the Elgendi PPG
processing method to clean the signal and detect pulse peaks. It then measures
the shape of each usable pulse and aggregates those measurements across the
window.

| Feature | Meaning |
|---|---|
| `ppg_clean_mean`, `ppg_clean_std`, `ppg_clean_range` | Summary statistics of the cleaned PPG waveform |
| `median_interpeak_interval_s` | Median time between detected pulse peaks |
| `median_rate_bpm` | Median pulse rate calculated from peak intervals |
| `median_pulse_amplitude` | Median trough-to-peak pulse amplitude |
| `median_rise_time_s` | Median time from the preceding trough to the pulse peak |
| `median_decay_time_s` | Median time from the pulse peak to the following trough |
| `median_rise_slope` | Median pulse amplitude divided by rise time |
| `median_decay_slope` | Median pulse amplitude divided by decay time |
| `median_half_amplitude_width_s` | Median width of the pulse above half amplitude |
| `median_pulse_area` | Median area above the local trough baseline |
| `pulse_amplitude_cv` | Beat-to-beat amplitude variability, measured as coefficient of variation |
| `median_pulse_skewness` | Median skewness of the individual pulse waveform |

For morphology, a pulse is defined using adjacent detected peaks and the
minimum cleaned PPG value between peaks as the trough. Features are summarized
across usable pulses using medians, which reduces the effect of one abnormal
beat. Windows with failed detection remain in the export with a status and
error message; undefined values remain `NaN` for model-side handling.

These are research features for an educational model, not clinically validated
blood-pressure biomarkers. Feature selection and imputation must be learned on
training data only.
