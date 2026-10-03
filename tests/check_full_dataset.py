"""Opt-in full local-data verification; writes the compact split handoff report.

Run from the repository root: python -m tests.check_full_dataset
"""

import bisect
import zipfile
from collections import Counter
from contextlib import ExitStack

import numpy as np

from data.dataset_splits import (
    DEFAULT_ARCHIVE, ROOT, WINDOW, _open_array, _read_rows, _reload,
    iter_split_windows, scan_records, validate_assignments,
)


def main():
    report = validate_assignments(DEFAULT_ARCHIVE)
    records = report["records"]
    assert len(records) == 12000
    assert sum(r["window_count"] for r in records) == 528828
    assert sum(r["tail_count"] for r in records) == 3172500
    duplicates = [r for r in records if r["split"] == "excluded"]
    assert len(duplicates) == 14
    assert sum(r["window_count"] for r in duplicates) == 700
    raw_dir = ROOT / "data" / "raw"
    files = [f"Part_{i}.mat" for i in range(1, 5)]
    repeated, _, _ = scan_records(raw_dir, files, report["split_config"])
    assert repeated == records, "Repeated source scan changed identities, duplicate map, or assignments"
    print("Full source rescan reproduces saved assignments", flush=True)

    rows = set(map(int, np.random.default_rng(2026).choice(528828, size=100, replace=False)))
    for part in range(1, 5):
        subset = [r for r in records if r["part_number"] == part and r["window_count"]]
        rows.add(subset[0]["first_npz_row"])
        rows.add(subset[-1]["first_npz_row"] + subset[-1]["window_count"] - 1)
    starts = [r["first_npz_row"] for r in records]
    expected = {}
    for row in sorted(rows):
        r = records[bisect.bisect_right(starts, row) - 1]
        local = row - r["first_npz_row"]
        record = _reload(raw_dir, r["part_number"], r["record_index"])
        expected[row] = record[local * WINDOW:(local + 1) * WINDOW, :2]
    with zipfile.ZipFile(DEFAULT_ARCHIVE) as archive, ExitStack() as stack:
        for name, col in (("ppg", 0), ("abp", 1)):
            stream, shape, dtype = _open_array(stack, archive, name)
            for start in range(0, shape[0], 1024):
                count = min(1024, shape[0] - start)
                data = _read_rows(stream, dtype, count, WINDOW)
                for row in rows:
                    if start <= row < start + count:
                        np.testing.assert_array_equal(data[row - start], expected[row][:, col])
    print(f"{len(rows)} source-coordinate samples match exactly", flush=True)

    counts = {}
    groups = {}
    for split in ("train", "validation", "test"):
        count = 0
        for batch in iter_split_windows(split):
            assert batch.ppg.shape == batch.abp.shape
            assert len(batch.ppg) == len(batch.window_ids) == len(batch.source_record_ids)
            assert np.isfinite(batch.ppg).all() and np.isfinite(batch.abp).all()
            for source in np.unique(batch.source_record_ids):
                assert groups.setdefault(str(source), split) == split
            count += len(batch.ppg)
        wanted = sum(r["window_count"] for r in records if r["split"] == split)
        assert count == wanted
        counts[split] = count
        print(f"{split}: consumed all {count:,} selected windows", flush=True)
    assert sum(counts.values()) == 528128
    assert len(groups) == 11986
    record_counts = Counter(r["split"] for r in records)
    forced = [r["source_record_id"] for r in records if r["source_record_id"] == r["canonical_record_id"] and r["forced_development"]]
    lines = ["# Prepared recording-based datasets", "", "Full verification against the local UCI files.", "",
             "Run `python data/load_files.py`, then `python -m tests.check_full_dataset`.", "",
             "| Split | Recordings | Windows |", "|---|---:|---:|"]
    for split in ("train", "validation", "test", "excluded"):
        count = sum(r["window_count"] for r in records if r["split"] == split)
        lines.append(f"| {split} | {record_counts[split]:,} | {count:,} |")
    lines += ["", "The archive retains all 12,000 original recordings / 528,828 windows. The split loader returns 11,986 recordings / 528,128 windows; 14 exact copies account for the 700 exclusions. Incomplete tails total 3,172,500 samples per channel.", "",
              "Verification: all channel waveform bytes matched the previous archive before replacement; staged metadata passed validation. The full source rescan reproduces identities, canonical copies, and split assignments. First/last windows per part plus 100 seeded random rows matched PPG/ABP exactly. Every selected window was consumed through the public batch loader, with finite PPG/ABP, aligned IDs, and no source recording crossing splits. Small-data tests cover rebuild reproducibility, malformed inputs, metadata tampering, and failed-build preservation.", "",
              "The existing streaming audit is run separately with `python eda/audit.py`; see its output for waveform agreement and historical label comparisons.", "",
              "Seed: 2026. Recording proportions: 70/15/15 with largest-remainder rounding. Known reviewed source recordings (including historical saved examples) and their canonical equivalents are assigned to training; no seed search was performed.", "",
              "Forced training recordings: " + ", ".join(f"`{key}`" for key in forced) + ".", "",
              "Source SHA-256 checksums:", ""]
    for name, checksum in report["source_checksums"].items():
        lines.append(f"- `{name}`: `{checksum}`")
    lines += ["", "Overall EDA preceded splitting. These are exact-duplicate-aware, record-disjoint datasets. Patient independence, additional shifted/partial overlaps, and generalization to other devices remain unestablished. The split loader applies no zero handling, outlier removal, normalization, feature extraction, or label method.", ""]
    destination = ROOT / "reports" / "split_summary.md"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
