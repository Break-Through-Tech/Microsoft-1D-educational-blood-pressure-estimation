"""Compare a training-median SBP/DBP guess with validation labels.

Run from the repository root: python -m models.naive_baseline_model
First generate data/cleaned_dataset.npz with cleaning/clean_data.ipynb.
"""

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "cleaned_dataset.npz"


def load_data(path):
    """Load labels and split metadata from the cleaned dataset."""
    required = ("ppg", "sbp", "dbp", "source_record_ids", "split_assignments")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Dataset is missing arrays: {sorted(missing)}")
        data = {name: archive[name] for name in required}

    # Keep every recording wholly within its saved split.
    for recording in np.unique(data["source_record_ids"]):
        splits = np.unique(data["split_assignments"][data["source_record_ids"] == recording])
        if len(splits) != 1:
            raise ValueError(f"Recording appears in more than one split: {recording}")
    return data


def select_recordings(data, seed, train_count, validation_count):
    """Choose the same seeded recording subsets as the CNN proof of concept."""
    rng_train, rng_validation = [
        np.random.default_rng(seed_part)
        for seed_part in np.random.SeedSequence(seed).spawn(2)
    ]
    selected = []
    for split, count, rng in (
        ("train", train_count, rng_train),
        ("validation", validation_count, rng_validation),
    ):
        recordings = np.unique(data["source_record_ids"][data["split_assignments"] == split])
        if count > len(recordings):
            raise ValueError(f"Requested {count} {split} recordings; only {len(recordings)} exist")
        selected.append(set(rng.choice(np.sort(recordings), count, replace=False)))
    return selected


def valid_labels(data, split, recordings):
    """Get usable [SBP, DBP] labels from selected recordings in one split."""
    sbp, dbp = data["sbp"], data["dbp"]
    keep = (
        (data["split_assignments"] == split)
        & np.isin(data["source_record_ids"], list(recordings))
        & np.isfinite(data["ppg"]).all(axis=1)
        & np.isfinite(sbp)
        & np.isfinite(dbp)
        & (sbp > dbp)
    )
    labels = np.column_stack((sbp[keep], dbp[keep]))
    if not len(labels):
        raise ValueError(f"No valid {split} labels found")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-records", type=int, default=300)
    parser.add_argument("--validation-records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"Cleaned dataset not found: {args.data}")

    data = load_data(args.data)
    train_ids, validation_ids = select_recordings(
        data, args.seed, args.train_records, args.validation_records
    )
    train = valid_labels(data, "train", train_ids)
    validation = valid_labels(data, "validation", validation_ids)

    # A constant median is the simplest baseline for absolute error (MAE).
    prediction = np.median(train, axis=0)
    error = prediction - validation

    print(f"Training windows: {len(train):,}; validation windows: {len(validation):,}")
    print(f"Baseline prediction: SBP {prediction[0]:.2f}, DBP {prediction[1]:.2f} mmHg")
    for index, name in enumerate(("SBP", "DBP")):
        mae = np.mean(np.abs(error[:, index]))
        rmse = np.sqrt(np.mean(error[:, index] ** 2))
        print(f"{name}: MAE {mae:.2f} mmHg, RMSE {rmse:.2f} mmHg")


if __name__ == "__main__":
    main()
