"""LIDC-IDRI step 1: index every nodule segmented by three or four radiologists.

Paper:    Table 2
Pipeline: python pipeline.py run lidc_data   (stage 'external')

The CT scans themselves are fetched later, one patient at a time, by
lidc_features.py (125 GB in total; only crops are kept).

Output: data/external/lidc/annotations.csv  (one row per nodule x reader)
        data/external/lidc/ann/<Modality>_<SeriesInstanceUID>/*.dcm
Run:    python scripts/data/lidc_download.py
"""
from __future__ import annotations

import glob
import pathlib
import re
import sys

import pandas as pd
import pydicom
from idc_index import IDCClient

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "external" / "lidc"
ANN = OUT / "ann"
RATINGS = ("Subtlety score", "Internal structure", "Calcification", "Sphericity", "Margin", "Lobular Pattern",
           "Spiculation", "Texture", "Malignancy")


def parse_sr(path) -> dict:
    ds = pydicom.dcmread(path)
    out = {}

    def walk(seq):
        for it in seq:
            name = it.ConceptNameCodeSequence[0].CodeMeaning if "ConceptNameCodeSequence" in it else ""
            if name in RATINGS and "ConceptCodeSequence" in it:
                val = it.ConceptCodeSequence[0].CodeMeaning
                m = re.match(r"(\d) out of 5", val)
                out[name.lower().replace(" ", "_")] = int(m.group(1)) if m else val
            elif name == "Source series for segmentation":
                out["ct_series_uid"] = it.UID
            elif name in ("Volume", "Diameter") and "MeasuredValueSequence" in it:
                out[f"pylidc_{name.lower()}"] = float(it.MeasuredValueSequence[0].NumericValue)
            if "ContentSequence" in it:
                walk(it.ContentSequence)

    walk(ds.ContentSequence)
    return out


def main():
    c = IDCClient()
    s = c.sql_query("""
        SELECT PatientID, StudyInstanceUID, SeriesInstanceUID, Modality, SeriesDescription
        FROM index WHERE collection_id = 'lidc_idri' AND analysis_result_id = 'dicom_lidc_idri_nodules'
          AND Modality IN ('SEG', 'SR')""")
    s[["nodule", "annotation"]] = s.SeriesDescription.str.extract(r"Nodule (\d+) - Annotation (.+?)(?: evaluations)?$")
    n_readers = s[s.Modality == "SEG"].groupby(["PatientID", "nodule"]).annotation.nunique()
    keep = n_readers[n_readers.between(3, 4)].index
    s = s.set_index(["PatientID", "nodule"]).loc[keep].reset_index()
    print(f"{len(keep)} nodules with 3-4 readers in {s.PatientID.nunique()} patients; {len(s)} series")

    ANN.mkdir(parents=True, exist_ok=True)
    if "--no-download" not in sys.argv:
        c.download_from_selection(seriesInstanceUID=s.SeriesInstanceUID.tolist(), downloadDir=str(ANN),
                                  dirTemplate="%Modality_%SeriesInstanceUID", quiet=True)

    rows = []
    # a reader's annotation id can occur twice for one nodule cluster; keep the first mask
    seg = s[s.Modality == "SEG"].drop_duplicates(["PatientID", "nodule", "annotation"]).set_index(
        ["PatientID", "nodule", "annotation"])
    for r in s[s.Modality == "SR"].drop_duplicates(["PatientID", "nodule", "annotation"]).itertuples():
        f = glob.glob(str(ANN / f"SR_{r.SeriesInstanceUID}" / "*.dcm"))
        key = (r.PatientID, r.nodule, r.annotation)
        if not f or key not in seg.index:
            continue
        rows.append({"PatientID": r.PatientID, "StudyInstanceUID": r.StudyInstanceUID, "nodule": int(r.nodule),
                     "annotation": r.annotation, "seg_series_uid": seg.loc[key, "SeriesInstanceUID"],
                     "sr_series_uid": r.SeriesInstanceUID, **parse_sr(f[0])})
    a = pd.DataFrame(rows).sort_values(["PatientID", "nodule", "annotation"])
    a["reader"] = a.groupby(["PatientID", "nodule"]).cumcount()
    a.to_csv(OUT / "annotations.csv", index=False)
    per = a.groupby(["PatientID", "nodule"]).size()
    print(f"annotations.csv: {len(a)} reader contours, {len(per)} nodules, {a.PatientID.nunique()} patients; "
          f"CT series {a.ct_series_uid.nunique()}")
    print("malignancy ratings:", a.malignancy.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
