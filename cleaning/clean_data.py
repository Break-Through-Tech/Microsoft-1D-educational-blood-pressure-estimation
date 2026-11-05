"""Record-streamed, resumable train/validation preparation. Never reads test signals.

python -m cleaning.clean_data --config configs/pipeline.json --output data/derived/prepared --workers 4
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import zipfile

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from cleaning.pipeline import (ALL_FEATURE_NAMES, FEATURE_NAMES, FEATURE_DEFINITIONS,
                               clean_window, label_definition, settings_checked)
from data.load_files import DEFAULT_ARCHIVE, RAW_DATA_DIR, validate_assignments

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ["data/bp_utils.py", "cleaning/pipeline.py", "cleaning/clean_data.py"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Windows readers/virus scanners can briefly deny replacement of an open file.
    for attempt in range(10):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def environment():
    return {name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "pandas", "h5py", "neurokit2", "joblib", "torch", "scikit-learn")}


def selected_records(report, splits=("train", "validation")):
    if not splits or not set(splits) <= {"train", "validation"}:
        raise ValueError("Only training and validation signals may be read")
    return [r for r in report["records"] if r["split"] in splits and
            r["source_record_id"] == r["canonical_record_id"] and r["window_count"] > 0]


def read_record(record, raw_dir=RAW_DATA_DIR):
    if record["split"] not in ("train", "validation") or record["source_record_id"] != record["canonical_record_id"]:
        raise ValueError("Refusing test or duplicate waveform access")
    part = record["part_number"]
    with h5py.File(Path(raw_dir) / f"Part_{part}.mat", "r") as source:
        raw = source[source[f"Part_{part}"][record["record_index"], 0]][()]
    h = hashlib.sha256()
    h.update(str(raw.shape).encode()); h.update(raw.dtype.str.encode()); h.update(raw.tobytes(order="C"))
    if h.hexdigest() != record["content_hash"]:
        raise ValueError(f"Source record hash changed: {record['source_record_id']}")
    return raw


def process_record(record, cfg, raw_dir=RAW_DATA_DIR):
    raw = read_record(record, raw_dir)
    n = record["window_count"]
    windows = raw[:n * 625].reshape(n, 625, 3).astype(np.float32)
    results = [clean_window(w[:, 0], w[:, 1], cfg) for w in windows]
    ids = [f"{record['source_record_id']}:window_{i}" for i in range(n)]
    rows = []
    kept = []
    for wid, result in zip(ids, results):
        rows.append({"window_id": wid, "source_record_id": record["source_record_id"],
                     "split": record["split"], "usable": result["usable"], "exclusion_reason": result["reason"],
                     "label_version": label_definition()["label_version"],
                     "sbp": result["sbp"], "dbp": result["dbp"],
                     "abp_peak_counts": result["n_abp_peaks"],
                     "abp_complete_beat_counts": result["n_abp_complete_beats"],
                     "ppg_peak_counts": result["n_ppg_peaks"],
                     "ppg_complete_pulse_counts": result["n_ppg_complete_pulses"],
                     "saturation_flag": result["saturation_flag"],
                     "processing_warnings": json.dumps(result["warnings"]),
                     "missing_feature_count": int(np.isnan(result["features"]).sum()),
                     **dict(zip(ALL_FEATURE_NAMES, result["features"]))})
        if result["usable"]:
            mask = np.zeros(625, dtype=bool); mask[result["ppg_peaks"]] = True
            kept.append((result["ppg"], result["abp"], mask))
    arrays = {"ppg": np.stack([x[0] for x in kept]) if kept else np.empty((0,625),np.float32),
              "abp": np.stack([x[1] for x in kept]) if kept else np.empty((0,625),np.float32),
              "ppg_peak_mask": np.stack([x[2] for x in kept]) if kept else np.empty((0,625),bool)}
    return rows, arrays


def verify_review(config):
    if config["label_configuration"] != label_definition():
        raise ValueError("Configured label method differs from the actual shared implementation")
    cfg = settings_checked(config["cleaning"])
    if cfg["sampling_rate"] != 125 or cfg["window_samples"] != 625:
        raise ValueError("Source windows are fixed at 625 samples / 125 Hz")
    review = ROOT / config["review_file"]
    if digest(review) != config["review_sha256"]:
        raise ValueError("Review evidence changed")
    evidence = json.loads(review.read_text())
    if not evidence.get("accepted") or evidence["label_configuration"] != label_definition() or evidence["cleaning"] != cfg:
        raise ValueError("Label/cleaning review is not accepted for this configuration")
    if evidence["code_hashes"] != {f: digest(ROOT/f) for f in ("data/bp_utils.py", "cleaning/pipeline.py")}:
        raise ValueError("Reviewed label or cleaning implementation changed")
    return cfg


def save_npz(path, arrays, metadata):
    """Stream NPY members into ZIP, including memmaps, without full RAM copies."""
    path = Path(path); temporary = path.with_suffix(".npz.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for name, array in {**arrays, "metadata_json": np.array(json.dumps(metadata, sort_keys=True))}.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    temporary.replace(path)


def export(output, manifest):
    chunks = sorted(manifest["completed"], key=int)
    frames = [pd.read_csv(output/"chunks"/f"{int(i):05d}.csv", keep_default_na=False) for i in chunks]
    table = pd.concat(frames, ignore_index=True)
    table["usable"] = table["usable"].astype(str).str.lower().eq("true")
    if table["window_id"].duplicated().any() or len(table) != manifest["input_windows"]:
        raise ValueError("Checkpoint rows lost or duplicated")
    for name in ALL_FEATURE_NAMES + ["sbp", "dbp"]:
        table[name] = pd.to_numeric(table[name], errors="coerce")
    table.to_csv(output/"model_table.csv", index=False)
    keep = table.loc[table["usable"]].reset_index(drop=True)
    if keep.empty: raise ValueError("No eligible windows")
    if keep.groupby("source_record_id")["split"].nunique().gt(1).any(): raise ValueError("Recording crosses splits")
    staging = output/"staging"; staging.mkdir(exist_ok=True)
    big = {name: np.lib.format.open_memmap(staging/f"{name}.npy", mode="w+", dtype=dtype, shape=(len(keep),625))
           for name, dtype in (("ppg",np.float32),("abp",np.float32),("ppg_peak_mask",bool))}
    offset = 0
    for i in chunks:
        with np.load(output/"chunks"/f"{int(i):05d}.npz", allow_pickle=False) as chunk:
            n = len(chunk["ppg"])
            for name in big: big[name][offset:offset+n] = chunk[name]
            offset += n
    if offset != len(keep): raise ValueError("Cleaned signals and feature rows do not align")
    for a in big.values(): a.flush()
    shared = {"window_ids": keep["window_id"].to_numpy(dtype=str),
              "source_record_ids": keep["source_record_id"].to_numpy(dtype=str),
              "split_assignments": keep["split"].to_numpy(dtype=str),
              "sbp": keep["sbp"].to_numpy(float), "dbp": keep["dbp"].to_numpy(float)}
    metadata = {k:v for k,v in manifest.items() if k not in ("completed",)}
    metadata["feature_names"] = ALL_FEATURE_NAMES
    metadata["feature_definitions"] = FEATURE_DEFINITIONS
    metadata["label_version"] = label_definition()["label_version"]
    metadata["coverage"] = table.groupby("split").agg(windows=("usable","size"),included=("usable","sum"),recordings=("source_record_id","nunique")).reset_index().to_dict("records")
    metadata["exclusions"] = table.loc[~table["usable"],"exclusion_reason"].value_counts().to_dict()
    save_npz(output/"cleaned_dataset.npz", {**big,**shared,**{name:keep[name].to_numpy(int) for name in
             ("ppg_peak_counts","ppg_complete_pulse_counts","abp_peak_counts","abp_complete_beat_counts")}}, metadata)
    save_npz(output/"feature_eng_dataset.npz", {**shared,"X":keep[ALL_FEATURE_NAMES].to_numpy(float),
             "feature_names":np.array(ALL_FEATURE_NAMES)}, metadata)
    for label,names in (("six",FEATURE_NAMES),("fourteen",ALL_FEATURE_NAMES)):
        write_json(output/f"feature_config_{label}.json", {"feature_names":names,
                   "dataset_version":manifest["configuration"]["dataset_version"],
                   "label_version":label_definition()["label_version"],
                   "label_configuration":label_definition(),"cleaning":manifest["configuration"]["cleaning"],
                   "feature_definitions":FEATURE_DEFINITIONS})
    write_json(output/"feature_config.json",json.loads((output/"feature_config_fourteen.json").read_text()))
    write_json(output/"label_config.json",manifest["configuration"])
    write_json(output/"preparation_summary.json",metadata)
    table.groupby(["split","usable","exclusion_reason"],dropna=False).size().to_csv(output/"coverage.csv")
    return metadata


def prepare(output, config, workers=4, resume=False, archive=DEFAULT_ARCHIVE, raw_dir=RAW_DATA_DIR, record_limit=None):
    started = time.perf_counter()
    cfg = verify_review(config)
    report = validate_assignments(archive)
    records = selected_records(report)
    if record_limit:
        records = [r for split in ("train","validation") for r in [x for x in records if x["split"]==split][:record_limit]]
    if workers < 1: raise ValueError("workers must be positive")
    output=Path(output)
    identity={"configuration":config,"records":records,"package_versions":environment(),
              "review_evidence":json.loads((ROOT/config["review_file"]).read_text()),
              "code_hashes":{f:digest(ROOT/f) for f in CODE_FILES},
              "source_metadata_sha256":hashlib.sha256(json.dumps(report,sort_keys=True).encode()).hexdigest(),
              "test_waveforms_loaded":False,"records_per_chunk":16}
    if output.exists():
        if not resume: raise FileExistsError("Output exists; use --resume or a new output directory")
        manifest=json.loads((output/"manifest.json").read_text())
        if any(manifest[k]!=v for k,v in identity.items()): raise ValueError("Resume source/configuration/code/version mismatch")
    else:
        output.mkdir(parents=True);(output/"chunks").mkdir()
        manifest={**identity,"completed":{},"input_windows":sum(r["window_count"] for r in records),"processing_seconds":0.}
        write_json(output/"manifest.json",manifest)
    groups=[records[i:i+16] for i in range(0,len(records),16)]
    with parallel_config(backend="loky",inner_max_num_threads=1):
        with Parallel(n_jobs=workers) as pool:
            for i, group in enumerate(groups):
                if str(i) in manifest["completed"]:
                    for ext,checksum in manifest["completed"][str(i)].items():
                        if digest(output/"chunks"/f"{i:05d}.{ext}")!=checksum: raise ValueError("Checkpoint hash mismatch")
                    for r in group: read_record(r,raw_dir)
                    continue
                values=pool(delayed(process_record)(r,cfg,raw_dir) for r in group)
                rows=[row for value in values for row in value[0]]
                arrays={name:np.concatenate([value[1][name] for value in values]) for name in values[0][1]}
                path=output/"chunks"/f"{i:05d}"
                pd.DataFrame(rows).to_csv(path.with_suffix(".csv"),index=False)
                np.savez_compressed(path.with_suffix(".npz"),**arrays)
                manifest["completed"][str(i)]={ext:digest(path.with_suffix('.'+ext)) for ext in ("csv","npz")}
                manifest["processing_seconds"]+=time.perf_counter()-started;started=time.perf_counter()
                write_json(output/"manifest.json",manifest)
                done=min((i+1)*16,len(records))
                print(f"Cleaned {done}/{len(records)} recordings; {manifest['processing_seconds']/60:.1f} minutes",flush=True)
    metadata=export(output,manifest)
    manifest["processing_seconds"]+=time.perf_counter()-started
    manifest["export_complete"]=True
    write_json(output/"manifest.json",manifest)
    metadata["processing_seconds"]=manifest["processing_seconds"]
    write_json(output/"preparation_summary.json",metadata)
    print(json.dumps({"coverage":metadata["coverage"],"exclusions":metadata["exclusions"],"seconds":manifest["processing_seconds"]},indent=2),flush=True)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/pipeline.json"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--record-limit",type=int,help="Development check: at most this many records per split")
    args=parser.parse_args()
    prepare(args.output,json.loads(args.config.read_text()),args.workers,args.resume,record_limit=args.record_limit)


if __name__=="__main__": main()
