"""Read development windows and retain their IDs through a finite-signal filter.

Run: python data/example_split_usage.py --split train --limit 1000
This demonstrates access, not a new quality policy, feature method, or model.
"""

import argparse

import numpy as np

try:
    from .dataset_splits import iter_split_windows
except ImportError:
    from dataset_splits import iter_split_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    examined = accepted = 0
    for batch in iter_split_windows(args.split, batch_size=args.batch_size):
        count = min(len(batch.ppg), args.limit - examined)
        ppg, abp = batch.ppg[:count], batch.abp[:count]
        keep = np.isfinite(ppg).all(axis=1) & np.isfinite(abp).all(axis=1)
        # Use this SAME mask for signals, features, labels, and identities.
        ppg, abp = ppg[keep], abp[keep]
        window_ids = batch.window_ids[:count][keep]
        source_ids = batch.source_record_ids[:count][keep]
        assert len(ppg) == len(abp) == len(window_ids) == len(source_ids)
        if examined == 0 and len(window_ids):
            print(f"First selected window: {window_ids[0]}; PPG/ABP shape: {ppg.shape[1:]}")
        examined += count
        accepted += len(window_ids)
        if examined == args.limit:
            break
    print(f"{args.split}: read {examined} windows; finite PPG/ABP filter retained {accepted}")


if __name__ == "__main__":
    main()
