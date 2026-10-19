"""External validation cohort (Lung1): features at every segmentation adjustment.

Paper:    Table 2
Pipeline: python pipeline.py run lung1_data   (stage 'external')

Input:  data/external/lung1/<PatientID>/{CT_*,SEG_*}/  (scripts/data/lung1_download.py)
Output: data/external/lung1/lung1_features.csv  (patient x shift)
        data/external/lung1/lung1_extraction_log.csv
Run:    python scripts/data/lung1_features.py [n_workers]
"""
from __future__ import annotations

import glob
import pathlib
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom
import pydicom_seg
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import perturbation, viz  # noqa: E402

L1 = ROOT / "data" / "external" / "lung1"
LEVELS = (-2, -1, 0, 1, 2)


def gtv_mask(pdir, ct):
    segs = glob.glob(str(pdir / "SEG_*" / "*.dcm"))
    if not segs:
        return None, "no SEG"
    ds = pydicom.dcmread(segs[0])
    labels = {s.SegmentNumber: s.SegmentLabel for s in ds.SegmentSequence}
    # IDC's DICOM-SEG conversion labels the primary tumour (GTV-1 in the original
    # RTSTRUCT) with the coded name "Neoplasm, Primary"; accept either spelling
    cand = [n for n, lab in labels.items() if lab.strip().lower() in ("neoplasm, primary", "gtv-1", "gtv1")]
    if not cand:
        cand = [n for n, lab in labels.items() if lab.lower().startswith(("gtv", "neoplasm"))]
    if not cand:
        return None, f"no GTV in {sorted(labels.values())}"
    img = pydicom_seg.SegmentReader().read(ds).segment_image(cand[0])
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "m.nii.gz"
        sitk.WriteImage(img, str(p))
        return viz.load_mask_on_ct(p, ct), labels[cand[0]]


def crop_to_tumour(ct, mask, margin_mm=12.0):
    """Crop CT and mask to the tumour bounding box plus a margin wider than the
    largest (2 mm) dilation. Every feature is computed inside the (perturbed)
    mask, so the crop leaves feature values unchanged while cutting memory and
    time for the distance transforms by one to two orders of magnitude."""
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(mask)
    x0, y0, z0, sx, sy, sz = ls.GetBoundingBox(1)
    sp = ct.GetSpacing()
    m = [int(np.ceil(margin_mm / s)) + 1 for s in sp]
    lo = [max(0, v - mm) for v, mm in zip((x0, y0, z0), m)]
    hi = [min(n, v + s + mm) for v, s, mm, n in zip((x0, y0, z0), (sx, sy, sz), m, ct.GetSize())]
    size = [h - l for l, h in zip(lo, hi)]
    return sitk.RegionOfInterest(ct, size, lo), sitk.RegionOfInterest(mask, size, lo)


CACHE = L1 / "cache"


def one(pid):
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    cached = CACHE / f"{pid}.csv"
    if cached.exists():
        d = pd.read_csv(cached)
        return d.to_dict("records"), {"PatientID": pid, "status": "ok", "segment": "cached", "voxels": None}
    pdir = L1 / pid
    try:
        cts = glob.glob(str(pdir / "CT_*"))
        if not cts:
            return [], {"PatientID": pid, "status": "no CT"}
        ct = viz.load_ct_volume(pathlib.Path(cts[0]))
        mask, info = gtv_mask(pdir, ct)
        if mask is None:
            return [], {"PatientID": pid, "status": info}
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        nvox = int(sitk.GetArrayFromImage(mask).sum())
        if nvox == 0:
            return [], {"PatientID": pid, "status": "empty GTV"}
        ct, mask = crop_to_tumour(ct, mask)
        ct_arr = sitk.GetArrayFromImage(ct)
        rows = []
        for mm in LEVELS:
            f = perturbation.extract_perturbed_features(pid, ct_arr, ct, mask, mm)
            if f is not None:
                rows.append(f)
        CACHE.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(cached, index=False)
        return rows, {"PatientID": pid, "status": "ok", "segment": info, "voxels": nvox}
    except Exception as exc:  # keep going; the log records every failure
        return [], {"PatientID": pid, "status": f"error: {exc!r}"[:200]}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    pids = sorted(p.name for p in L1.iterdir() if p.is_dir())
    rows, log = [], []
    with Pool(n) as pool:
        for i, (r, lg) in enumerate(pool.imap_unordered(one, pids), 1):
            rows += r
            log.append(lg)
            print(f"[{i}/{len(pids)}] {lg['PatientID']}: {lg['status']}", flush=True)
    out = pd.DataFrame(rows).rename(columns={"case_id": "PatientID"})
    out.to_csv(L1 / "lung1_features.csv", index=False)
    pd.DataFrame(log).to_csv(L1 / "lung1_extraction_log.csv", index=False)
    print(f"{out.PatientID.nunique()} patients with features; statuses:",
          pd.DataFrame(log).status.str.split(":").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
