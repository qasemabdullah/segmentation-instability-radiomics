"""Patient overlap between the multi-reader NSCLC cohort and Lung1.

Paper:    Table 2, Methods
Pipeline: python pipeline.py run cohort_overlap   (stage 'data')

Both collections come from the same radiotherapy institution and their public
identifiers are not linked, so overlap is determined from the images
themselves. For each of the 21 planning CTs of NSCLC-Radiomics-Interobserver1,
Lung1 series with the same matrix size and pixel spacing are candidates. Each
candidate is ranked by the correlation of its per-slice mean-attenuation
profile with that of the multi-reader CT (best slice offset), and the best
candidate is confirmed voxel by voxel: a match requires a Pearson correlation
of the aligned volumes > 0.99, i.e. the same scan.

A positive control passes three Lung1 scans through the same NIfTI export and
matching (whole and with 10 slices cropped from each end): an identical scan
gives profile and voxel correlations of 1.00.

Output: data/processed/cohort_overlap.csv (one row per multi-reader patient),
        data/processed/cohort_overlap.json (counts), cohort_overlap_positive_control.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
IOB = ROOT / "data" / "interim" / "interobserver_nifti"
LUNG1 = ROOT / "data" / "external" / "lung1"
OUT = ROOT / "data" / "processed"


def lung1_headers():
    rows = []
    for pdir in sorted(LUNG1.glob("LUNG1-*")):
        for sdir in pdir.glob("CT_*"):
            files = sorted(sdir.glob("*.dcm"))
            if not files:
                continue
            h = pydicom.dcmread(files[0], stop_before_pixels=True)
            rows.append({"lung1_id": pdir.name, "series_dir": str(sdir), "n_slices": len(files),
                         "rows": int(h.Rows), "cols": int(h.Columns),
                         "px": float(h.PixelSpacing[0]), "thk": float(getattr(h, "SliceThickness", 0) or 0)})
    return pd.DataFrame(rows)


_PROFILES = {}


def series_profile(sdir):
    """Per-slice mean attenuation of a Lung1 series (cached: each series is read once)."""
    if sdir not in _PROFILES:
        _PROFILES[sdir] = read_series(sdir).mean(axis=(1, 2))
    return _PROFILES[sdir]


def read_series(sdir):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(sdir))
    return sitk.GetArrayFromImage(reader.Execute()).astype(np.float32)


def offsets(a, b, min_frac=0.8):
    for off in range(-(len(b) - 1), len(a)):
        a0, b0 = max(off, 0), max(-off, 0)
        n = min(len(a) - a0, len(b) - b0)
        if n >= max(10, min_frac * min(len(a), len(b))):
            yield a0, b0, n


def profile_match(pa, pb):
    """Best correlation of two per-slice mean profiles over slice offsets, and that offset."""
    best = (-1.0, None)
    for a0, b0, n in offsets(pa, pb):
        r = float(np.corrcoef(pa[a0:a0 + n], pb[b0:b0 + n])[0, 1])
        if r > best[0]:
            best = (r, (a0, b0, n))
    return best


def volume_correlation(a, b, off):
    a0, b0, n = off
    return float(np.corrcoef(a[a0:a0 + n, ::2, ::2].ravel(), b[b0:b0 + n, ::2, ::2].ravel())[0, 1])


def positive_control(ids=("LUNG1-016", "LUNG1-094", "LUNG1-281")):
    import tempfile
    out = []
    for pid in ids:
        sdir = str(next((LUNG1 / pid).glob("CT_*")))
        r = sitk.ImageSeriesReader()
        r.SetFileNames(r.GetGDCMSeriesFileNames(sdir))
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "q.nii.gz"
            sitk.WriteImage(r.Execute(), str(f))
            q = sitk.GetArrayFromImage(sitk.ReadImage(str(f))).astype(np.float32)
        ref = read_series(sdir)
        row = {"lung1_id": pid}
        for name, qq in (("whole", q), ("cropped", q[10:-10])):
            pr, off = profile_match(qq.mean(axis=(1, 2)), ref.mean(axis=(1, 2)))
            row[f"{name}_profile_r"], row[f"{name}_volume_r"] = pr, volume_correlation(qq, ref, off)
        out.append(row)
    (OUT / "cohort_overlap_positive_control.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    if "--control-only" in sys.argv:
        print(json.dumps(positive_control(), indent=2))
        return
    heads = lung1_headers()
    print(f"{len(heads)} Lung1 CT series")
    out = []
    for ct_path in sorted(IOB.glob("*_ct.nii.gz")):
        pid = ct_path.name.replace("_ct.nii.gz", "")
        img = sitk.ReadImage(str(ct_path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        nz, ny, nx = arr.shape
        px = img.GetSpacing()[0]
        cand = heads[(heads.rows == ny) & (heads.cols == nx) & (np.abs(heads.px - px) < 1e-3)]
        # slice counts may differ if one export was cropped, so rank candidates by content, not by count
        prof = arr.mean(axis=(1, 2))
        ranked = []
        for _, c in cand.iterrows():
            r, off = profile_match(prof, series_profile(c.series_dir))
            ranked.append((r, c.lung1_id, off, c.series_dir))
        ranked.sort(key=lambda t: -t[0])
        best_id, best_r, prof_r = None, -1.0, np.nan
        if ranked and ranked[0][2] is not None:
            prof_r, best_id, off, sdir = ranked[0]
            best_r = volume_correlation(arr, read_series(sdir), off)
        match = best_r > 0.99
        print(f"  {pid}: {len(cand)} candidates; best {best_id} r={best_r:.4f} -> {'MATCH' if match else 'no match'}")
        out.append({"interobserver_id": pid, "n_candidates": len(cand), "best_lung1_id": best_id,
                    "profile_correlation": round(prof_r, 5), "best_correlation": round(best_r, 5),
                    "second_profile_correlation": round(ranked[1][0], 5) if len(ranked) > 1 else np.nan,
                    "same_scan": match})
    df = pd.DataFrame(out)
    df.to_csv(OUT / "cohort_overlap.csv", index=False)
    n_shared = int(df.same_scan.sum())
    summary = {"multi_reader_patients": len(df), "shared_with_lung1": n_shared,
               "lung1_patients_analyzed": 419, "unique_nsclc_radiotherapy_patients": 419 + len(df) - n_shared,
               "shared_lung1_ids": sorted(df.loc[df.same_scan, "best_lung1_id"].tolist())}
    summary["max_voxel_correlation"] = float(df.best_correlation.max())
    (OUT / "cohort_overlap.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(json.dumps(positive_control(), indent=2))


if __name__ == "__main__":
    sys.exit(main())
