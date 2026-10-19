"""CT acquisition parameters of every analysed series, read from the DICOM headers.

Paper:    Methods, Suppl. Table S29
Pipeline: python pipeline.py run acquisition   (stage 'external')

Discovery and external validation cohorts: local DICOM. LIDC-IDRI and
liver-metastasis cohorts: one image per series is fetched from the public
NCI Imaging Data Commons bucket (headers only). The multi-reader NSCLC cohort
is summarized from its data descriptor.

Output: data/processed/ct_acquisition_by_series.csv, ct_acquisition_summary.csv
Run:    python scripts/data/acquisition_summary.py [threads]
"""
from __future__ import annotations

import glob
import pathlib
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydicom

ROOT = pathlib.Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
S5 = [shutil.which("s5cmd"), "--no-sign-request", "--endpoint-url", "https://s3.amazonaws.com"]
FIELDS = ("Manufacturer", "ManufacturerModelName", "SliceThickness", "PixelSpacing", "KVP", "ContrastBolusAgent", "ConvolutionKernel")


def header(path) -> dict:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    out = {f: getattr(ds, f, None) for f in FIELDS}
    ps = out.pop("PixelSpacing")
    out["pixel_spacing_mm"] = float(ps[0]) if ps else np.nan
    out["slice_thickness_mm"] = float(out.pop("SliceThickness") or np.nan)
    out["kvp"] = float(out.pop("KVP") or np.nan)
    out["contrast"] = bool(str(out.pop("ContrastBolusAgent") or "").strip() not in ("", "None", "NONE", "No", "NO"))
    out["manufacturer"] = str(out.pop("Manufacturer") or "").strip()
    out["model"] = str(out.pop("ManufacturerModelName") or "").strip()
    out["kernel"] = str(out.pop("ConvolutionKernel") or "").strip()
    return out


def local(series_dir):
    f = sorted(glob.glob(str(pathlib.Path(series_dir) / "*.dcm")))
    return header(f[len(f) // 2]) if f else None


def remote(url):
    try:
        ls = subprocess.run(S5 + ["ls", url], capture_output=True, text=True, timeout=120).stdout.split()
        name = next(x for x in ls if x.endswith(".dcm"))
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(S5 + ["cp", url.replace("*", name), td + "/"], check=True, capture_output=True, timeout=120)
            return header(pathlib.Path(td) / name)
    except Exception:
        return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = []
    feats = pd.read_csv(P / "radiomics_features.csv")
    for uid in feats.ct_series_uid:
        h = local(ROOT / "data" / "interim" / "dicom" / uid)
        if h:
            rows.append({"cohort": "Discovery", **h})
    for d in glob.glob(str(ROOT / "data" / "external" / "lung1" / "LUNG1-*" / "CT_*")):
        h = local(d)
        if h:
            rows.append({"cohort": "External validation (Lung1)", **h})
    lidc_urls = pd.read_csv(ROOT / "data" / "external" / "lidc" / "ct_urls.csv").series_aws_url.tolist()
    crlm = pd.read_csv(ROOT / "data" / "external" / "crlm" / "series.csv")
    crlm_urls = crlm[crlm.kind == "ct"].series_aws_url.tolist()
    with ThreadPoolExecutor(n) as ex:
        for cohort, urls in (("Multi-reader nodules (LIDC-IDRI)", lidc_urls), ("Liver metastases", crlm_urls)):
            for h in ex.map(remote, urls):
                if h:
                    rows.append({"cohort": cohort, **h})
    df = pd.DataFrame(rows)
    df.to_csv(P / "ct_acquisition_by_series.csv", index=False)

    def rng(s, dp=2):
        s = s.dropna()
        if not len(s):
            return "—"
        if s.min() == s.max():
            return f"{s.median():.{dp}f}"
        return f"{s.median():.{dp}f} ({s.min():.{dp}f}–{s.max():.{dp}f})"
    names = {"SIEMENS": "Siemens", "PHILIPS": "Philips", "TOSHIBA": "Toshiba", "CANON": "Canon", "IMATRON": "Imatron"}
    summ = []
    for c, g in df.groupby("cohort", sort=False):
        vendors = (g.manufacturer.str.upper().str.extract(r"(GE|SIEMENS|PHILIPS|TOSHIBA|CANON|IMATRON)")[0]
                   .replace(names).fillna("other").value_counts())
        vendors = pd.concat([vendors.drop("other", errors="ignore"), vendors.reindex(["other"]).dropna()])
        summ.append({"Cohort": c, "CT series": len(g),
                     "Manufacturers": "; ".join(f"{k} {int(v)}" for k, v in vendors.items()),
                     "Slice thickness, mm, median (range)": rng(g.slice_thickness_mm),
                     "Pixel spacing, mm, median (range)": rng(g.pixel_spacing_mm),
                     "Tube voltage, kVp, median (range)": rng(g.kvp, 0),
                     "Contrast agent recorded": f"{int(g.contrast.sum())} ({100 * g.contrast.mean():.0f}%)"})
    s = pd.DataFrame(summ)
    s.to_csv(P / "ct_acquisition_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
