"""Verify streamed preparation outputs and sampled signal/feature provenance."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from cleaning.clean_data import CODE_FILES, ROOT, digest, write_json
from cleaning.pipeline import ALL_FEATURE_NAMES, clean_abp, ppg_features
from data.bp_utils import extract_bp_labels
from data.load_files import _open_array, _read_rows


def verify(directory,sample_count=512):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    summary=json.loads((directory/"preparation_summary.json").read_text())
    if not manifest.get("export_complete") or manifest["test_waveforms_loaded"]:raise ValueError("Preparation not complete or touched test")
    if manifest["code_hashes"]!={f:digest(ROOT/f) for f in CODE_FILES}:raise ValueError("Preparation implementation changed")
    table=pd.read_csv(directory/"model_table.csv")
    usable=table["usable"].astype(str).str.lower().eq("true")
    kept=table.loc[usable].reset_index(drop=True)
    if table["window_id"].duplicated().any() or len(table)!=manifest["input_windows"]:raise ValueError("Input IDs inconsistent")
    expected={r["source_record_id"]:r for r in manifest["records"]}
    counts=table["source_record_id"].value_counts().to_dict()
    if counts!={rid:r["window_count"] for rid,r in expected.items()}:raise ValueError("Recording/window counts differ")
    for rid,frame in table.groupby("source_record_id"):
        if set(frame["split"])!={expected[rid]["split"]}:raise ValueError("Recording split differs")
        if frame["window_id"].tolist()!=[f"{rid}:window_{i}" for i in range(len(frame))]:raise ValueError("Window order differs")
    config=summary["configuration"]
    if not table["label_version"].eq(config["label_configuration"]["label_version"]).all():raise ValueError("Label version mismatch")
    for name in ("six","fourteen"):
        fc=json.loads((directory/f"feature_config_{name}.json").read_text())
        if fc["label_configuration"]!=config["label_configuration"] or fc["cleaning"]!=config["cleaning"]:raise ValueError("Feature config mismatch")
    with np.load(directory/"feature_eng_dataset.npz",allow_pickle=False) as z:
        features=z["X"]
        if z["feature_names"].tolist()!=ALL_FEATURE_NAMES:raise ValueError("Feature order differs")
        np.testing.assert_allclose(features,kept[ALL_FEATURE_NAMES].to_numpy(float),equal_nan=True,rtol=1e-12,atol=1e-12)
        feature_metadata=json.loads(z["metadata_json"].item())
        for name,column in (("window_ids","window_id"),("source_record_ids","source_record_id"),("split_assignments","split")):
            np.testing.assert_array_equal(z[name],kept[column].to_numpy(str))
        for t in ("sbp","dbp"):np.testing.assert_allclose(z[t],kept[t],rtol=0,atol=1e-10)
    with np.load(directory/"cleaned_dataset.npz",allow_pickle=False) as z:
        clean_metadata=json.loads(z["metadata_json"].item())
        if clean_metadata!=feature_metadata:raise ValueError("Dataset configurations differ")
        for name,column in (("window_ids","window_id"),("source_record_ids","source_record_id"),("split_assignments","split")):
            np.testing.assert_array_equal(z[name],kept[column].to_numpy(str))
        for t in ("sbp","dbp"):np.testing.assert_allclose(z[t],kept[t],rtol=0,atol=1e-10)
    sample=set(np.random.default_rng(2026).choice(len(kept),min(sample_count,len(kept)),replace=False).tolist())
    checked=0;max_label_difference=0.;feature_failures=[]
    with zipfile.ZipFile(directory/"cleaned_dataset.npz") as archive,ExitStack() as stack:
        arrays={n:_open_array(stack,archive,n) for n in ("ppg","abp","ppg_peak_mask")}
        for start in range(0,len(kept),2048):
            n=min(2048,len(kept)-start)
            batch={name:_read_rows(stream,dtype,n,625) for name,(stream,shape,dtype) in arrays.items()}
            if not np.isfinite(batch['ppg']).all() or not np.isfinite(batch['abp']).all():raise ValueError("Nonfinite cleaned waveform")
            for i in sorted(sample & set(range(start,start+n))):
                local=i-start
                labels=extract_bp_labels(batch['abp'][local])[:2]
                difference=np.max(abs(np.array(labels)-kept.loc[i,['sbp','dbp']].to_numpy(float)))
                if not np.isfinite(difference) or difference>1e-3:raise ValueError(f"Saved ABP does not reproduce labels: {kept.loc[i,'window_id']}")
                max_label_difference=max(max_label_difference,float(difference))
                values,_=ppg_features(batch['ppg'][local],np.flatnonzero(batch['ppg_peak_mask'][local]))
                # Exported waveforms are float32; preprocessing/features use float64 intermediates.
                np.testing.assert_allclose(values,features[i],equal_nan=True,rtol=2e-4,atol=2e-5)
                checked+=1
    report={"verified":True,"input_windows":len(table),"retained_windows":len(kept),"sampled_signal_feature_label_checks":checked,
            "max_float32_label_difference_mmhg":max_label_difference,"recordings":len(expected),"test_waveforms_loaded":False,
            "artifact_sha256":{name:digest(directory/name) for name in ("cleaned_dataset.npz","feature_eng_dataset.npz","model_table.csv")}}
    write_json(directory/"verification.json",report)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory",type=Path)
    args=p.parse_args();verify(args.directory)


if __name__=="__main__":main()
