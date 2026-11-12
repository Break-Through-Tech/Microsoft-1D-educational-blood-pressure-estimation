"""Regenerate models/figures/cnn_input_channels.png: the three Input C channels for one training window.

Run from the repository root:
    python models/figures/make_cnn_input_channels.py

Needs data/processed_dataset.npz (built by `python data/load_files.py`) and data/raw/Part_1.mat.
Only the waveform of training recording Part_1:record_0 is read; no validation or test waveform is opened.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

WINDOW_ID = "Part_1:record_0:window_2"
OUTPUT = ROOT / "models" / "figures" / "cnn_input_channels.png"
PANELS = [("Channel 0: filtered PPG (0.5–8 Hz bandpass)", "PPG units", "#1f77b4"),
          ("Channel 1: first derivative (slope)", "PPG units / s", "#d62728"),
          ("Channel 2: second derivative (curvature)", "PPG units / s²", "#2ca02c")]


def load_training_window(window_id):
    """Return the source PPG for one training window, read the same way as the PR #46 pipeline."""
    from data.load_files import DEFAULT_ARCHIVE, RAW_DATA_DIR, validate_assignments
    from cleaning.clean_data import read_record

    if not Path(DEFAULT_ARCHIVE).is_file():
        sys.exit(f"Missing {DEFAULT_ARCHIVE}. Place Part_1.mat-Part_4.mat in data/raw/, "
                 "then run: python data/load_files.py")
    record_id, local = window_id.rsplit(":window_", 1)
    records = {r["source_record_id"]: r for r in validate_assignments(DEFAULT_ARCHIVE)["records"]}
    record = records.get(record_id)
    if record is None:
        sys.exit(f"{record_id} is not in {DEFAULT_ARCHIVE}")
    if record["split"] != "train":
        sys.exit(f"{record_id} is in the {record['split']} split; this figure must use training data")
    mat_file = Path(RAW_DATA_DIR) / f"Part_{record['part_number']}.mat"
    if not mat_file.is_file():
        sys.exit(f"Missing {mat_file}. Download the UCI dataset and place Part_{record['part_number']}.mat in data/raw/")
    index = int(local)
    if not 0 <= index < record["window_count"]:
        sys.exit(f"{window_id} does not exist; {record_id} has {record['window_count']} windows")
    raw = read_record(record)
    # Same slicing and float32 cast as cleaning.clean_data.process_record and prepare_record.
    windows = raw[:record["window_count"] * 625].reshape(-1, 625, 3).astype(np.float32)
    return windows[index, :, 0], windows[index, :, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cleaning.pipeline import clean_window
    from models.selected_inputs import input_c, prepare_prediction_inputs

    ppg, abp = load_training_window(WINDOW_ID)
    result = clean_window(ppg, abp)
    if not result["usable"]:
        sys.exit(f"{WINDOW_ID} no longer passes cleaning: {result['reason']}")

    values = input_c(ppg[None, :])
    if values.shape != (1, 3, 625) or not np.isfinite(values).all():
        sys.exit(f"Unexpected Input C output: shape {values.shape}")
    # Confirm this is the array the prediction path passes to the CNN before training-set scaling.
    _, predicted_inputs, _, reasons = prepare_prediction_inputs(ppg[None, :], features=False)
    if reasons != [""] or not np.array_equal(predicted_inputs, values):
        sys.exit("Input C differs from the prediction path")

    t = np.arange(625) / 125
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, channel, (title, unit, color) in zip(axes, values[0], PANELS):
        ax.plot(t, channel, color=color, linewidth=1.3)
        ax.axhline(0, color="0.6", linewidth=0.6, zorder=0)
        ax.set_ylabel(unit, fontsize=10)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=4)
        ax.grid(axis="x", color="0.9", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlim(0, 5)
    axes[-1].set_xticks(np.arange(0, 5.5, 0.5))
    axes[-1].set_xlabel("Time (s) — 625 samples at 125 Hz", fontsize=10)
    fig.suptitle(f"Input C for one training window ({WINDOW_ID})", fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.935, "Shape 3 × 625, before training-set normalization", fontsize=10, color="0.35", ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    print(f"Saved {args.output}: channels {values.shape[1]} x {values.shape[2]} samples; "
          f"reference SBP {result['sbp']:.1f} / DBP {result['dbp']:.1f} mmHg "
          f"from {result['n_abp_complete_beats']} complete beats")


if __name__ == "__main__":
    main()
