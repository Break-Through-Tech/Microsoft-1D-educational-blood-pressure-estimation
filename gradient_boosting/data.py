"""Prepare PPG features and BP labels, keeping recording IDs and rejected rows."""

import csv
import hashlib
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from data.load_files import MAT_FILES, iter_split_windows, prepare_dataset, validate_assignments
from gradient_boosting import cleaning, features

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "data/derived/gradient_boosting"
CONFIG = Path(__file__).with_name("config.json")
TABLE_KEYS = ("X", "sbp", "dbp", "window_ids", "source_record_ids", "split_assignments",
              "ppg_eligible", "reference_valid", "included", "feature_status", "feature_errors")


def path(value):
    value = Path(value)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def output_path(value):
    result = path(value)
    if not result.is_relative_to(OUTPUT_ROOT.resolve()) or result == OUTPUT_ROOT.resolve():
        raise ValueError("Save generated files inside data/derived/gradient_boosting/")
    return result


def run_dir(config):
    return output_path(config["paths"]["output_dir"])


def checksum(filename):
    digest = hashlib.sha256()
    with Path(filename).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(filename, value, overwrite=False):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def load_config(filename=CONFIG):
    config = json.loads(Path(filename).read_text(encoding="utf-8"))
    if config["sample_rate_hz"] != 125 or config["window_samples"] != 625:
        raise ValueError("This pipeline uses five-second windows: 625 samples at 125 Hz")
    cleaning.validate_settings(config)
    run_dir(config)
    output_path(config["paths"]["prepared_archive"])
    for key in ("workers", "batch_size", "bootstrap_samples"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    return config


def processing_signature(config):
    # Reports can change without invalidating models. Data/feature changes cannot.
    files = [Path(__file__), Path(cleaning.__file__), Path(features.__file__),
             ROOT / "data/bp_utils.py", ROOT / "data/load_files.py"]
    return {"sample_rate_hz": config["sample_rate_hz"], "window_samples": config["window_samples"],
            "cleaning": config["cleaning"], "feature_names": list(features.FEATURE_NAMES),
            "code": {p.relative_to(ROOT).as_posix(): checksum(p) for p in files}}


def source_report(config, create=False):
    archive = output_path(config["paths"]["prepared_archive"])
    if not archive.exists():
        if not create:
            raise FileNotFoundError("Prepare the dataset first: python -m gradient_boosting prepare")
        prepare_dataset(path(config["paths"]["raw_dir"]), MAT_FILES, archive,
                        path(config["paths"]["split_config"]))
    report = validate_assignments(archive)
    if report["split_config"] != json.loads(path(config["paths"]["split_config"]).read_text()):
        raise ValueError("The archive and split configuration disagree")
    return report


def source_id(report):
    return hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()


def pressure_band(value, target):
    if value is None or not np.isfinite(value):
        return "unknown"
    cuts, labels = ((100, 140, 180), ("below_100", "100_to_139", "140_to_179", "180_or_above")) if target == "sbp" else (
        (60, 90, 110), ("below_60", "60_to_89", "90_to_109", "110_or_above"))
    return labels[int(np.searchsorted(cuts, value, side="right"))]


def check_backup(config):
    """Keep the original run auditable after retiring its old Python files."""
    imported = run_dir(config) / "imported_run.json"
    if not imported.exists():
        return None
    info = json.loads(imported.read_text())
    if info["processing_signature"] != processing_signature(config):
        raise ValueError("Processing differs from the verified saved run")
    archive = output_path(info["source_backup"])
    if checksum(archive) != info["backup_sha256"]:
        raise ValueError("The original source backup changed")
    old = output_path(info["original_run"])
    freeze = json.loads((old / "frozen/freeze.json").read_text())
    if checksum(old / "frozen/freeze.json") != info["original_freeze_sha256"]:
        raise ValueError("The original frozen run changed")
    with zipfile.ZipFile(archive) as backup:
        for name, expected in freeze["code"].items():
            actual = hashlib.sha256(backup.read(name)).hexdigest() if name.startswith("gradient_boosting/") else checksum(ROOT / name)
            if actual != expected:
                raise ValueError(f"Original source provenance changed: {name}")
    return info


def table_folder(config, split):
    if split not in ("train", "validation", "test"):
        raise ValueError("Choose train, validation or test")
    imported = check_backup(config)
    return output_path(imported["original_run"]) / "tables" / split if imported else run_dir(config) / "tables" / split


def validate_table(arrays, split, report):
    n = len(arrays["X"])
    if arrays["X"].shape != (n, len(features.FEATURE_NAMES)) or not n or np.isinf(arrays["X"]).any():
        raise ValueError("Expected a nonempty 17-column feature table without infinities")
    if any(arrays[key].shape != (n,) for key in TABLE_KEYS[1:]):
        raise ValueError("Table columns have different lengths")
    for key in ("window_ids", "source_record_ids", "split_assignments"):
        if arrays[key].dtype.kind != "U" or (arrays[key] == "").any():
            raise ValueError("IDs must be nonempty strings")
    if len(np.unique(arrays["window_ids"])) != n or not (arrays["split_assignments"] == split).all():
        raise ValueError("Duplicate windows or wrong split")
    source = {r["source_record_id"]: r for r in report["records"]}
    for window, record in zip(arrays["window_ids"], arrays["source_record_ids"], strict=True):
        prefix, _, local = window.rpartition(":window_")
        row = source.get(record)
        if (row is None or row["split"] != split or row["canonical_record_id"] != record
                or prefix != record or not local.isdecimal() or local != str(int(local))
                or int(local) >= row["window_count"]):
            raise ValueError("Window IDs do not match the saved recording split")
    if n != sum(r["window_count"] for r in report["records"] if r["split"] == split):
        raise ValueError("Some source windows are missing")
    for key in ("ppg_eligible", "reference_valid", "included"):
        if arrays[key].dtype.kind != "b":
            raise ValueError("Eligibility fields must be Boolean")
    if not np.array_equal(arrays["included"], arrays["ppg_eligible"] & arrays["reference_valid"]):
        raise ValueError("PPG and reference checks must stay independent")
    valid = arrays["reference_valid"]
    if (not np.isfinite(arrays["sbp"][valid]).all() or not np.isfinite(arrays["dbp"][valid]).all()
            or not (arrays["sbp"][valid] > arrays["dbp"][valid]).all()
            or not np.isnan(arrays["sbp"][~valid]).all() or not np.isnan(arrays["dbp"][~valid]).all()
            or not np.isnan(arrays["X"][~arrays["ppg_eligible"]]).all()):
        raise ValueError("Invalid references or fabricated features")


def load_table(config, split, report=None):
    if split == "test":
        from gradient_boosting.models import load_model
        load_model(config)  # A frozen model must exist before test preparation/use.
    report = report or source_report(config)
    folder = table_folder(config, split)
    metadata = json.loads((folder / "metadata.json").read_text())
    imported = check_backup(config)
    if imported:
        if checksum(folder / "metadata.json") != imported["table_metadata"][split] or source_id(report) != imported["source_id"]:
            raise ValueError("Saved table or source metadata changed")
    elif metadata["signature"] != {"processing": processing_signature(config), "source": source_id(report), "batch_size": config["batch_size"]}:
        raise ValueError("Data settings changed; choose a new output directory")
    chunks = []
    for receipt in metadata["batches"]:
        for key, suffix in (("arrays", ".npz"), ("ledger", ".csv")):
            if checksum(folder / (receipt["stem"] + suffix)) != receipt[key]:
                raise ValueError("Saved data or rejection ledger changed")
        with np.load(folder / (receipt["stem"] + ".npz"), allow_pickle=False) as saved:
            chunks.append({key: saved[key] for key in TABLE_KEYS})
    arrays = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in TABLE_KEYS}
    validate_table(arrays, split, report)
    return arrays


def load_ledger(config, split):
    folder = table_folder(config, split)
    metadata = json.loads((folder / "metadata.json").read_text())
    rows = []
    for receipt in metadata["batches"]:
        with (folder / (receipt["stem"] + ".csv")).open(newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                for key in ("ppg_eligible", "reference_valid", "included"):
                    if row[key] not in ("True", "False"):
                        raise ValueError("Invalid eligibility flag in ledger")
                    row[key] = row[key] == "True"
                rows.append(row)
    return rows


def process_window(ppg, abp, window, record, split, config):
    cleaned, reference, included = cleaning.clean_window(ppg, abp, config)
    measured = features.extract_window(cleaned["cleaned"], config["sample_rate_hz"]) if cleaned["eligible"] else None
    row = {"window_id": str(window), "source_record_id": str(record), "split": split,
           "ppg_eligible": cleaned["eligible"], "reference_valid": reference["valid"], "included": included,
           "ppg_reason": cleaned["reason"], "reference_reason": reference["reason"],
           "sbp": reference["sbp"] if reference["valid"] else None,
           "dbp": reference["dbp"] if reference["valid"] else None,
           "feature_status": measured["status"] if measured else "ppg_ineligible",
           "feature_error": measured["error"] if measured else "", "ppg_error": cleaned["error"],
           "reference_error": reference["error"], "ppg_warnings": cleaned["warnings"],
           "reference_warnings": reference["warnings"], "feature_warnings": measured["warnings"] if measured else ""}
    row.update({t + "_band": pressure_band(row[t], t) for t in ("sbp", "dbp")})
    return measured["values"] if measured else np.full(17, np.nan), row


def process_chunk(task):
    ppg, abp, ids, records, split, config = task
    with threadpool_limits(limits=1):
        return [process_window(p, a, w, r, split, config) for p, a, w, r in zip(ppg, abp, ids, records, strict=True)]


def prepare_split(config, split, report=None):
    if split == "test":
        from gradient_boosting.models import load_model
        load_model(config)
    report = report or source_report(config, create=True)
    folder = table_folder(config, split)
    if (folder / "metadata.json").exists():
        arrays = load_table(config, split, report)
        print(f"Using saved {split} data: {len(arrays['X']):,} windows; {arrays['included'].sum():,} retained", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    signature = {"processing": processing_signature(config), "source": source_id(report), "batch_size": config["batch_size"]}
    settings_file = folder / "settings.json"
    if settings_file.exists():
        if json.loads(settings_file.read_text()) != signature:
            raise ValueError("Preparation settings changed; use a new output directory")
    else:
        save_json(settings_file, signature)
    batches, total = [], 0
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        for index, batch in enumerate(iter_split_windows(split, config["batch_size"], output_path(config["paths"]["prepared_archive"]))):
            stem = f"batch_{index:05d}"
            receipt_path = folder / (stem + ".json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                for key, suffix in (("arrays", ".npz"), ("ledger", ".csv")):
                    if checksum(folder / (stem + suffix)) != receipt[key]:
                        raise ValueError("A completed batch changed")
                with np.load(folder / (stem + ".npz"), allow_pickle=False) as saved:
                    if not np.array_equal(saved["window_ids"], batch.window_ids):
                        raise ValueError("Resumed batch IDs differ")
            else:
                if any((folder / (stem + suffix)).exists() for suffix in (".npz", ".csv")):
                    raise ValueError("An incomplete batch exists; preserve it and use a new output directory")
                slices = [slice(i, i + 64) for i in range(0, len(batch.ppg), 64)]
                tasks = [(batch.ppg[s], batch.abp[s], batch.window_ids[s], batch.source_record_ids[s], split, config) for s in slices]
                results = [item for chunk in pool.map(process_chunk, tasks) for item in chunk]
                rows = [r for _, r in results]
                arrays = {"X": np.stack([x for x, _ in results])}
                fields = {"window_ids": "window_id", "source_record_ids": "source_record_id",
                          "split_assignments": "split", "feature_errors": "feature_error"}
                for key in TABLE_KEYS[1:]:
                    dtype = float if key in ("sbp", "dbp") else bool if key in ("ppg_eligible", "reference_valid", "included") else str
                    arrays[key] = np.array([r[fields.get(key, key)] if r[fields.get(key, key)] is not None else np.nan for r in rows], dtype=dtype)
                with (folder / (stem + ".npz")).open("xb") as output:
                    np.savez_compressed(output, **arrays)
                with (folder / (stem + ".csv")).open("x", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                receipt = {"arrays": checksum(folder / (stem + ".npz")), "ledger": checksum(folder / (stem + ".csv"))}
                save_json(receipt_path, receipt)
            batches.append({"stem": stem, **receipt})
            total += len(batch.ppg)
            print(f"Preparing {split}: {total:,} windows", flush=True)
    if signature["processing"] != processing_signature(config):
        raise ValueError("Processing code changed during preparation")
    save_json(folder / "metadata.json", {"signature": signature, "batches": batches})
    load_table(config, split, report)


def verify_raw_samples(config, split, count=8):
    """Check that cached features still match processing of the original signals."""
    import h5py
    report = source_report(config)
    arrays = load_table(config, split, report)
    records = {r["source_record_id"]: r for r in report["records"]}
    chosen = np.random.default_rng(config["seed"]).choice(len(arrays["X"]), min(count, len(arrays["X"])), replace=False)
    for i in chosen:
        row = records[arrays["source_record_ids"][i]]
        local = int(arrays["window_ids"][i].rpartition(":window_")[2])
        with h5py.File(path(config["paths"]["raw_dir"]) / f"Part_{row['part_number']}.mat", "r") as source:
            record = source[source[f"Part_{row['part_number']}"][row["record_index"], 0]]
            samples = record[local * 625:(local + 1) * 625, :2]
        values, decision = process_window(samples[:, 0], samples[:, 1], arrays["window_ids"][i], arrays["source_record_ids"][i], split, config)
        if not np.array_equal(values, arrays["X"][i], equal_nan=True):
            raise ValueError("Cached features differ from the raw signal")
        for key in ("ppg_eligible", "reference_valid", "included"):
            if decision[key] != arrays[key][i]:
                raise ValueError("Cached quality decisions differ")
        if decision["reference_valid"] and not np.array_equal([decision["sbp"], decision["dbp"]], [arrays["sbp"][i], arrays["dbp"][i]]):
            raise ValueError("Cached reference labels differ")
    print(f"{split}: {len(chosen)} raw windows reproduce their features and labels.", flush=True)
