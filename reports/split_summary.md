# Prepared recording-based datasets

Full verification against the local UCI files.

Run `python data/load_files.py`, then `python -m tests.check_full_dataset`.

| Split | Recordings | Windows |
|---|---:|---:|
| train | 8,390 | 372,295 |
| validation | 1,798 | 76,887 |
| test | 1,798 | 78,946 |
| excluded | 14 | 700 |

The archive retains all 12,000 original recordings / 528,828 windows. The split loader returns 11,986 recordings / 528,128 windows; 14 exact copies account for the 700 exclusions. Incomplete tails total 3,172,500 samples per channel.

Verification: all channel waveform bytes matched the previous archive before replacement; staged metadata passed validation. The full source rescan reproduces identities, canonical copies, and split assignments. First/last windows per part plus 100 seeded random rows matched PPG/ABP exactly. Every selected window was consumed through the public batch loader, with finite PPG/ABP, aligned IDs, and no source recording crossing splits. Small-data tests cover rebuild reproducibility, malformed inputs, metadata tampering, and failed-build preservation.

The existing streaming audit passed: PPG and ABP match all raw-derived windows in order, with 12,000 records, 528,828 windows, 14 extra copies, and 3,172,500 dropped tail samples per channel. Run it separately with `python eda/audit.py`; its label comparisons remain exploratory. All nine small-data tests passed.

Seed: 2026. Recording proportions: 70/15/15 with largest-remainder rounding. Known reviewed source recordings (including historical saved examples) and their canonical equivalents are assigned to training; no seed search was performed.

Forced training recordings: `Part_1:record_0`, `Part_1:record_1`, `Part_1:record_86`, `Part_1:record_87`, `Part_1:record_88`, `Part_1:record_95`, `Part_1:record_99`, `Part_1:record_105`, `Part_1:record_1613`, `Part_1:record_2016`, `Part_1:record_2055`, `Part_1:record_2555`, `Part_2:record_536`, `Part_3:record_79`, `Part_3:record_1103`, `Part_4:record_1919`, `Part_4:record_2608`.

Source SHA-256 checksums:

- `Part_1.mat`: `c065c97a98729c1e3426be7131d046ce48ef551a7812ad570891ed9ab9547281`
- `Part_2.mat`: `31e6c4be3468ea0dd2d295e7b2b8adf77b748c8f89a4e1a4433796e2e38bed68`
- `Part_3.mat`: `169f93eea380c1bb79f7b83f6eeb3c2df7aefbd7da734537dfc66aa6c7a1be88`
- `Part_4.mat`: `9b5123d14960725675ce0b73757636bbeacfbb0b75201a74965c009c733579a9`

Overall EDA preceded splitting. These are exact-duplicate-aware, record-disjoint datasets. Patient independence, additional shifted/partial overlaps, and generalization to other devices remain unestablished. The split loader applies no zero handling, outlier removal, normalization, feature extraction, or label method.
