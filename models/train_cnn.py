"""Provisional proof of concept: train the raw-PPG 1D CNN on small recording subsets (Issue #35).

Run from the repository root: python -m models.train_cnn
Labels use window_extrema_v1 (ABP window maximum -> SBP, minimum -> DBP). This target is
provisional until the team selects a label version (LABEL-01), so every metric printed here is
provisional. Only the train and validation splits are read; the test split is never accessed.
"""

import argparse
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.load_files import DEFAULT_ARCHIVE, iter_split_windows, validate_assignments
from models.cnn_model import PPGCNN

LABEL_METHOD = "window_extrema_v1"
TARGETS = ("SBP", "DBP")


def sample_recordings(records, split, count, rng):
    """Choose `count` whole recordings from one saved split with a seeded generator."""
    ids = sorted(r["source_record_id"] for r in records if r["split"] == split and r["window_count"])
    if count > len(ids):
        raise ValueError(f"Requested {count} {split} recordings; only {len(ids)} exist")
    return set(rng.choice(ids, size=count, replace=False).tolist())


def select_recordings(records, seed, train_count, validation_count):
    """Pick training and validation recordings with two independent generators from one seed.

    Each split gets its own generator, so changing train_count never changes the validation choice.
    """
    train_seed, validation_seed = np.random.SeedSequence(seed).spawn(2)
    train_ids = sample_recordings(records, "train", train_count, np.random.default_rng(train_seed))
    validation_ids = sample_recordings(records, "validation", validation_count, np.random.default_rng(validation_seed))
    return train_ids, validation_ids


def window_extrema_v1(abp):
    """Provisional labels: SBP = window maximum, DBP = window minimum of ABP (mmHg)."""
    return np.stack([abp.max(axis=1), abp.min(axis=1)], axis=1)


def load_windows(split, recording_ids):
    """Read every window of the selected recordings; return PPG, labels, window IDs, source IDs."""
    if split not in ("train", "validation"):
        raise ValueError("This proof of concept reads only the train and validation splits")
    wanted = np.array(sorted(recording_ids))
    ppg, labels, window_ids, sources = [], [], [], []
    for batch in iter_split_windows(split):
        keep = np.isin(batch.source_record_ids, wanted)
        if keep.any():
            ppg.append(batch.ppg[keep])
            labels.append(window_extrema_v1(batch.abp[keep]))
            window_ids.append(batch.window_ids[keep])
            sources.append(batch.source_record_ids[keep])
    return tuple(np.concatenate(parts) for parts in (ppg, labels, window_ids, sources))


def valid_mask(ppg, labels):
    """Minimal inclusion rule: finite PPG, finite labels, and SBP above DBP. Zeros are kept."""
    return np.isfinite(ppg).all(axis=1) & np.isfinite(labels).all(axis=1) & (labels[:, 0] > labels[:, 1])


def normalize_ppg(ppg):
    """Per-window z-score: subtract each window's own mean and divide by its own standard deviation.

    No statistic is shared between windows, so validation data cannot influence training inputs.
    """
    mean = ppg.mean(axis=1, keepdims=True)
    std = ppg.std(axis=1, keepdims=True)
    return (ppg - mean) / np.maximum(std, 1e-6)


def to_tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


def predict(model, ppg, batch_size):
    """Predict in batches so memory use stays small however many windows are evaluated."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(ppg), batch_size):
            outputs.append(model(to_tensor(ppg[start:start + batch_size, None, :])).numpy())
    return np.concatenate(outputs)


def metrics(pred, true):
    error = pred - true
    return {name: (float(np.abs(error[:, i]).mean()), float(np.sqrt((error[:, i] ** 2).mean())))
            for i, name in enumerate(TARGETS)}


def print_metrics(title, values):
    print(f"  {title}")
    for name, (mae, rmse) in values.items():
        print(f"    {name}: MAE {mae:6.2f} mmHg   RMSE {rmse:6.2f} mmHg")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-records", type=int, default=300)
    parser.add_argument("--validation-records", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    records = validate_assignments(DEFAULT_ARCHIVE)["records"]
    train_ids, validation_ids = select_recordings(records, args.seed, args.train_records, args.validation_records)
    assert not train_ids & validation_ids, "A recording was selected for both training and validation"

    print(f"PROVISIONAL RUN - labels: {LABEL_METHOD}; device: cpu; seed: {args.seed}")
    print("Reading selected recordings...", flush=True)
    data = {}
    for split, ids in (("train", train_ids), ("validation", validation_ids)):
        ppg, labels, window_ids, sources = load_windows(split, ids)
        keep = valid_mask(ppg, labels)
        data[split] = (normalize_ppg(ppg[keep]), labels[keep], window_ids[keep], sources[keep])
        print(f"  {split}: {len(set(sources[keep])):,} recordings, {int(keep.sum()):,} windows used "
              f"({int((~keep).sum()):,} excluded by the minimal inclusion rule)")
    assert not set(data["train"][3]) & set(data["validation"][3]), "Training and validation recordings overlap"

    train_x, train_y = data["train"][:2]
    validation_x, validation_y = data["validation"][:2]
    # Targets are standardized with training statistics only, then converted back to mmHg for metrics.
    target_mean, target_std = train_y.mean(axis=0), train_y.std(axis=0)

    loader = DataLoader(
        TensorDataset(to_tensor(train_x[:, None, :]), to_tensor((train_y - target_mean) / target_std)),
        batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed),
    )
    model = PPGCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.L1Loss()

    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        print(f"  epoch {epoch}: training L1 loss {total / len(loader.dataset):.4f} (standardized units)")

    predictions = predict(model, validation_x, args.batch_size) * target_std + target_mean
    baseline = np.broadcast_to(target_mean, validation_y.shape)

    print(f"Validation results on {len(validation_y):,} windows (PROVISIONAL, {LABEL_METHOD}):")
    print_metrics("1D CNN", metrics(predictions, validation_y))
    print_metrics(f"Training-mean baseline (SBP {target_mean[0]:.1f}, DBP {target_mean[1]:.1f})",
                  metrics(baseline, validation_y))
    print(f"Elapsed: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
