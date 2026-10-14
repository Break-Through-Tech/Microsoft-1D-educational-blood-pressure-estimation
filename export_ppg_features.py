"""Export PPG-only NeuroKit2 features from the prepared split archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.load_files import DEFAULT_ARCHIVE, SPLITS, iter_split_windows
from features.extract_ppg import DEFAULT_FEATURE_CONFIG, extract_ppg_feature_rows


def export_split(
    split: str,
    *,
    archive_path: Path = DEFAULT_ARCHIVE,
    output_path: Path | None = None,
    batch_size: int = 1024,
    config: dict[str, object] | None = None,
) -> Path:
    """Stream one split into a CSV while retaining failed feature rows."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if output_path is None:
        output_path = Path("data/derived") / f"features_{split}_ppg_neurokit_v2.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    settings = dict(DEFAULT_FEATURE_CONFIG)
    if config:
        settings.update(config)
    settings["feature_version"] = "ppg_neurokit_v2"

    wrote_header = False
    row_count = 0
    for batch in iter_split_windows(split, batch_size=batch_size, archive_path=archive_path):
        extracted = extract_ppg_feature_rows(batch.ppg, config=settings)
        rows = []
        for window_id, source_record_id, result in zip(
            batch.window_ids, batch.source_record_ids, extracted, strict=True
        ):
            rows.append(
                {
                    "window_id": window_id,
                    "source_record_id": source_record_id,
                    "split": split,
                    "feature_version": settings["feature_version"],
                    **result,
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        row_count += len(frame)
        print(f"{split}: processed {row_count:,} windows", flush=True)

    if not wrote_header:
        raise ValueError(f"No windows found for split {split!r}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="train")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--config", type=Path, help="Optional JSON feature configuration")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None

    splits = SPLITS if args.split == "all" else (args.split,)
    for split in splits:
        output = args.output_dir / f"features_{split}_ppg_neurokit_v2.csv"
        print(f"Writing {split} features to {output}")
        export_split(
            split,
            archive_path=args.archive,
            output_path=output,
            batch_size=args.batch_size,
            config=config,
        )


if __name__ == "__main__":
    main()
