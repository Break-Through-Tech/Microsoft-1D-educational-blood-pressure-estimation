"""Recording-based preparation and bounded-memory access to the window archive."""

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import format as npy_format

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "processed_dataset.npz"
FORMAT_VERSION = 2
SPLITS = ("train", "validation", "test")
WINDOW = 625


def iter_source_records(raw_dir, mat_files):
    """Yield (part number, original record index, samples), in legacy row order."""
    files = sorted(mat_files, key=lambda name: int(re.fullmatch(r"Part_(\d+)\.mat", name)[1]))
    if len(files) != len(set(files)):
        raise ValueError("Repeated MAT filenames are not allowed")
    for name in files:
        part = int(re.fullmatch(r"Part_(\d+)\.mat", name)[1])
        with h5py.File(Path(raw_dir) / name, "r") as source:
            refs = source[f"Part_{part}"]
            if refs.ndim != 2 or refs.shape[1] != 1:
                raise ValueError(f"Unexpected reference shape in {name}: {refs.shape}")
            for index in range(len(refs)):
                record = source[refs[index, 0]][()]
                if record.ndim != 2 or record.shape[1] != 3 or record.dtype.kind not in "fiu":
                    raise ValueError(f"Invalid shape/dtype in Part_{part}:record_{index}: {record.shape}, {record.dtype}")
                yield part, index, record


def _reload(raw_dir, part, index):
    with h5py.File(Path(raw_dir) / f"Part_{part}.mat", "r") as source:
        return source[source[f"Part_{part}"][index, 0]][()]


def file_checksum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_splits(records, config):
    """Assign canonical recordings; reviewed copies force their canonical into train."""
    fractions = np.asarray(config["fractions"], dtype=float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or (fractions < 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("fractions must contain three nonnegative values summing to one")
    by_id = {r["source_record_id"]: r for r in records}
    reviewed = set(config.get("reviewed_record_ids", []))
    unknown = reviewed - by_id.keys()
    if unknown:
        raise ValueError(f"Reviewed records absent from source files: {sorted(unknown)}")
    forced = {by_id[key]["canonical_record_id"] for key in reviewed}
    canonical = [r for r in records if r["source_record_id"] == r["canonical_record_id"]]
    # Largest remainder allocation gives exact integer totals, without seed searching.
    wanted = fractions * len(canonical)
    counts = np.floor(wanted).astype(int)
    for index in np.argsort(-(wanted - counts), kind="stable")[:len(canonical) - counts.sum()]:
        counts[index] += 1
    if len(forced) > counts[0]:
        raise ValueError("Reviewed records exceed the requested training allocation")
    remaining = [r["source_record_id"] for r in canonical if r["source_record_id"] not in forced]
    rng = np.random.default_rng(config["seed"])
    remaining = list(rng.permutation(remaining))
    assigned = {key: "train" for key in forced}
    offset = 0
    for split, count in zip(SPLITS, (counts[0] - len(forced), counts[1], counts[2])):
        assigned.update({str(key): split for key in remaining[offset:offset + count]})
        offset += count
    for record in records:
        key = record["source_record_id"]
        record["split"] = assigned[key] if key == record["canonical_record_id"] else "excluded"
        record["reviewed"] = key in reviewed
        record["forced_development"] = record["canonical_record_id"] in forced


def scan_records(raw_dir, mat_files, config, window_size=WINDOW):
    records, candidates = [], {}
    row = 0
    dtype = None
    for part, index, record in iter_source_records(raw_dir, mat_files):
        key = f"Part_{part}:record_{index}"
        digest = hashlib.sha256()
        digest.update(str(record.shape).encode())
        digest.update(record.dtype.str.encode())
        digest.update(record.tobytes(order="C"))
        fingerprint = digest.hexdigest()
        canonical = key
        for previous in candidates.get(fingerprint, []):
            original = _reload(raw_dir, previous["part_number"], previous["record_index"])
            if record.shape == original.shape and record.dtype == original.dtype and record.tobytes() == original.tobytes():
                canonical = previous["source_record_id"]
                break
        count, tail = divmod(len(record), window_size)
        info = dict(source_record_id=key, part_number=part, record_index=index,
                    sample_count=len(record), window_count=count, tail_count=tail,
                    first_npz_row=row, content_hash=fingerprint, canonical_record_id=canonical)
        for channel, col in zip(("ppg", "abp", "ecg"), range(3)):
            info[f"{channel}_nan_count"] = int(np.isnan(record[:, col]).sum())
            info[f"{channel}_infinity_count"] = int(np.isinf(record[:, col]).sum())
        records.append(info)
        if canonical == key:
            candidates.setdefault(fingerprint, []).append(info)
        row += count
        dtype = record.dtype if dtype is None else np.result_type(dtype, record.dtype)
    if not records or row == 0:
        raise ValueError("Source files contain no complete windows")
    assign_splits(records, config)
    return records, row, np.dtype(dtype)


def _write_array(archive, name, shape, dtype, chunks):
    dtype = np.dtype(dtype)
    digest = hashlib.blake2b(digest_size=16)
    count = 0
    with archive.open(name + ".npy", "w", force_zip64=True) as stream:
        npy_format.write_array_header_2_0(stream, dict(descr=npy_format.dtype_to_descr(dtype), fortran_order=False, shape=shape))
        for chunk in chunks:
            data = np.ascontiguousarray(chunk, dtype=dtype)
            stream.write(data.tobytes())
            digest.update(data.tobytes())
            count += data.size
    if count != math.prod(shape):
        raise ValueError(f"Incorrect sample count writing {name}")
    return digest.hexdigest()


def _open_array(stack, archive, name):
    stream = stack.enter_context(archive.open(name + ".npy"))
    version = npy_format.read_magic(stream)
    if version == (1, 0):
        shape, order, dtype = npy_format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, order, dtype = npy_format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY version for {name}: {version}")
    if order or dtype.hasobject:
        raise ValueError(f"Unsupported storage for {name}")
    return stream, shape, dtype


def _read_rows(stream, dtype, rows, width=1):
    size = rows * width * dtype.itemsize
    raw = stream.read(size)
    if len(raw) != size:
        raise ValueError("Truncated array in dataset archive")
    data = np.frombuffer(raw, dtype=dtype).copy()
    return data.reshape(rows, width) if width != 1 else data


def archive_signal_hashes(path):
    result = {}
    with zipfile.ZipFile(path) as archive, ExitStack() as stack:
        for name in ("ppg", "abp", "ecg"):
            stream, shape, dtype = _open_array(stack, archive, name)
            digest = hashlib.blake2b(digest_size=16)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            result[name] = (shape, dtype.str, digest.hexdigest())
    return result


@contextmanager
def _temporary_archive(directory):
    # Stage in the destination directory: a private temporary directory on
    # Windows can give a renamed file restrictive inherited directory ACLs.
    descriptor, name = tempfile.mkstemp(prefix="dataset-build-", suffix=".npz", dir=directory)
    os.close(descriptor)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def prepare_dataset(raw_dir, mat_files, output_path=DEFAULT_ARCHIVE, config_path=ROOT / "configs" / "split.json", window_size=WINDOW):
    """Build a versioned archive atomically, retaining every original waveform row."""
    if window_size != WINDOW:
        raise ValueError("Split archive format requires 625-sample windows")
    raw_dir, output_path = Path(raw_dir), Path(output_path)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    print("Scanning recordings, exact duplicates, and split assignments...", flush=True)
    checksums = {name: file_checksum(raw_dir / name) for name in sorted(mat_files)}
    records, total, dtype = scan_records(raw_dir, mat_files, config, window_size)
    report = dict(format_version=FORMAT_VERSION, sample_rate_hz=125, window_samples=WINDOW,
                  split_config=config, source_checksums=checksums, records=records,
                  numpy_version=np.__version__, h5py_version=h5py.__version__)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _temporary_archive(output_path.parent) as staged:
        written = {}
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for name, col in zip(("ppg", "abp", "ecg"), range(3)):
                print(f"Writing {name}: {total:,} original windows...", flush=True)
                def chunks(column=col):
                    for info, (part, index, record) in zip(records, iter_source_records(raw_dir, mat_files), strict=True):
                        if (part, index) != (info["part_number"], info["record_index"]) or len(record) != info["sample_count"]:
                            raise ValueError("Source ordering or length changed during build")
                        count = info["window_count"]
                        yield record[:count * WINDOW, column].reshape(count, WINDOW)
                digest = _write_array(archive, name, (total, WINDOW), dtype, chunks())
                written[name] = ((total, WINDOW), dtype.str, digest)
            for name, field, string_dtype in (("source_record_ids", "source_record_id", "U40"), ("split_assignments", "split", "U10")):
                _write_array(archive, name, (total,), string_dtype,
                             (np.full(r["window_count"], r[field], dtype=string_dtype) for r in records))
            _write_array(archive, "window_ids", (total,), "U64",
                         (np.array([f'{r["source_record_id"]}:window_{i}' for i in range(r["window_count"])], dtype="U64") for r in records))
            for name, value in (("format_version", np.array(FORMAT_VERSION)),
                                ("preparation_json", np.array(json.dumps(report, sort_keys=True)))):
                with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                    npy_format.write_array(stream, value, allow_pickle=False)
        print("Verifying staged signals and metadata...", flush=True)
        if archive_signal_hashes(staged) != written:
            raise ValueError("Staged signal verification failed")
        validate_assignments(staged)
        if output_path.exists():
            print("Comparing all original waveform bytes with existing archive...", flush=True)
            if archive_signal_hashes(output_path) != written:
                raise ValueError("Existing waveform values/order differ; archive was not replaced")
        if any(file_checksum(raw_dir / name) != checksum for name, checksum in checksums.items()):
            raise ValueError("Source files changed during build")
        staged.replace(output_path)
    return report


@dataclass(frozen=True)
class WindowBatch:
    ppg: np.ndarray
    abp: np.ndarray
    window_ids: np.ndarray
    source_record_ids: np.ndarray


def _metadata(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"format_version", "preparation_json", "window_ids", "source_record_ids", "split_assignments"}
        if not required.issubset(archive.files):
            raise ValueError("Archive lacks split metadata. Rebuild with: python data/load_files.py")
        if archive["format_version"].item() != FORMAT_VERSION:
            raise ValueError("Unsupported dataset format; rebuild with python data/load_files.py")
        return json.loads(archive["preparation_json"].item())


def validate_assignments(path):
    """Check metadata against the embedded recording map without loading waveforms."""
    report = _metadata(path)
    records = report["records"]
    total = sum(r["window_count"] for r in records)
    ids = {r["source_record_id"] for r in records}
    by_id = {r["source_record_id"]: r for r in records}
    if len(ids) != len(records):
        raise ValueError("Duplicate source IDs")
    seen = {}
    with zipfile.ZipFile(path) as archive, ExitStack() as stack:
        arrays = {name: _open_array(stack, archive, name) for name in ("window_ids", "source_record_ids", "split_assignments")}
        for name, (_, shape, dtype) in arrays.items():
            if shape != (total,) or dtype.kind != "U":
                raise ValueError(f"Invalid metadata shape/dtype: {name}")
        offset = 0
        for r in records:
            count = r["window_count"]
            key, canonical, split = r["source_record_id"], r["canonical_record_id"], r["split"]
            if canonical not in ids or (key != canonical and split != "excluded") or (key == canonical and split not in SPLITS):
                raise ValueError(f"Invalid duplicate/split assignment: {key}")
            if by_id[canonical]["canonical_record_id"] != canonical or r["first_npz_row"] != offset:
                raise ValueError(f"Invalid canonical reference or original row offset: {key}")
            offset += count
            if key == canonical:
                if canonical in seen and seen[canonical] != split:
                    raise ValueError("Recording crosses splits")
                seen[canonical] = split
                if r["forced_development"] and split == "test":
                    raise ValueError("Reviewed recording is in test")
            expected = {"window_ids": np.array([f"{key}:window_{i}" for i in range(count)]),
                        "source_record_ids": np.full(count, key), "split_assignments": np.full(count, split)}
            for name, (stream, _, dtype) in arrays.items():
                if not np.array_equal(_read_rows(stream, dtype, count), expected[name]):
                    raise ValueError(f"Misaligned {name} for {key}")
    return report


def iter_split_windows(split, batch_size=1024, archive_path=DEFAULT_ARCHIVE):
    """Yield aligned batches from one saved split, automatically excluding copies.

    Batches contain at most batch_size rows and preserve original row order.
    Filtering is applied to signals and identities together. No cleaning is done.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    report = validate_assignments(archive_path)
    total = sum(r["window_count"] for r in report["records"])
    with zipfile.ZipFile(archive_path) as archive, ExitStack() as stack:
        names = ("ppg", "abp", "window_ids", "source_record_ids", "split_assignments")
        arrays = {name: _open_array(stack, archive, name) for name in names}
        for name, (_, shape, dtype) in arrays.items():
            expected = (total, WINDOW) if name in ("ppg", "abp") else (total,)
            if shape != expected or (name in ("ppg", "abp") and dtype.kind not in "fiu"):
                raise ValueError(f"Invalid shape/dtype for {name}")
        for start in range(0, total, batch_size):
            rows = min(batch_size, total - start)
            values = {name: _read_rows(stream, dtype, rows, WINDOW if name in ("ppg", "abp") else 1)
                      for name, (stream, _, dtype) in arrays.items()}
            keep = values["split_assignments"] == split
            if keep.any():
                yield WindowBatch(*(values[name][keep] for name in names[:4]))
