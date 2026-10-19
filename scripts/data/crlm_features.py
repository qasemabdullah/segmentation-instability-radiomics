"""Liver-metastasis cohort: features of the expert, ±1-mm and AI segmentations of the largest lesion.

Paper:    Fig. 5b, c, g, Suppl. Table S19
Pipeline: python pipeline.py run crlm   (stage 'external')

Per patient: the portal-venous CT, the expert DICOM-SEG (liver, vessels,
remnant and one segment per tumour, Tumor_k) and the AIMI nnU-Net liver-tumour
SEG ("Mass"), plus the radiologist-corrected AI SEG where available. The
analysed lesion is the largest expert tumour; the AI lesion is the union of AI
connected components that overlap it (a miss is recorded, never replaced by the
expert segmentation), as in the discovery-cohort analysis. The 26 study features are
extracted from the expert lesion at -1, 0 and +1 mm and from the AI (and
corrected) lesions at 0 mm, with Dice and ASSD against the expert lesion.

Output: data/external/crlm/crlm_features.csv, crlm_agreement.csv, crlm_log.csv
Run:    OPENBLAS_NUM_THREADS=1 python scripts/data/crlm_features.py [workers]
"""
from __future__ import annotations

import glob
import pathlib
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom
import pydicom_seg
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, perturbation, viz  # noqa: E402
from scripts.data.lidc_features import assd  # noqa: E402

CRLM = ROOT / "data" / "external" / "crlm"
CACHE, TMP = CRLM / "cache", CRLM / "tmp"
IDX = str(config.idc_index_file())


def series_table():
    import duckdb
    d = duckdb.sql(f"""SELECT PatientID, Modality, SeriesDescription, SeriesInstanceUID, series_aws_url,
        COALESCE(analysis_result_id, 'orig') AS src FROM '{IDX}'
        WHERE collection_id = 'colorectal_liver_metastases' AND Modality IN ('CT', 'SEG')""").df()
    d["kind"] = np.select([d.Modality == "CT", d.src == "orig",
                           d.SeriesDescription.str.contains("corrected", na=False)], ["ct", "expert", "corrected"], "ai")
    return d


def fetch(url, dst):
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run([shutil.which("s5cmd"), "--no-sign-request", "--endpoint-url", "https://s3.amazonaws.com",
                    "cp", url, str(dst) + "/"], check=True, capture_output=True,
                           timeout=900)  # a stalled cloud download fails this patient instead of blocking the worker
    return dst


def segment(seg_dir, ct, pick):
    """Union of the segments whose label satisfies pick(label), on the CT grid,
    as a labelled array of connected components kept separately for tumours."""
    ds = pydicom.dcmread(glob.glob(str(seg_dir / "*.dcm"))[0])
    res = pydicom_seg.SegmentReader().read(ds)
    out = {}
    for s in ds.SegmentSequence:
        if pick(s.SegmentLabel):
            with tempfile.TemporaryDirectory() as td:
                p = pathlib.Path(td) / "m.nii.gz"
                sitk.WriteImage(res.segment_image(s.SegmentNumber), str(p))
                out[s.SegmentLabel] = sitk.GetArrayFromImage(viz.load_mask_on_ct(p, ct)) > 0
    return out


def as_image(arr, ref):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    return img


def one(args):
    pid, ser = args
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    cached = CACHE / f"{pid}.csv"
    if cached.exists():
        return pid, "cached"
    tmp = TMP / pid
    try:
        kinds = {k: g for k, g in ser.groupby("kind")}
        if "ct" not in kinds or "expert" not in kinds:
            return pid, "missing CT or expert SEG"
        ct = viz.load_ct_volume(fetch(kinds["ct"].series_aws_url.iloc[0], tmp / "ct"))
        tumours = segment(fetch(kinds["expert"].series_aws_url.iloc[0], tmp / "expert"), ct,
                          lambda lab: lab.lower().startswith("tumor"))
        if not tumours:
            return pid, "no expert tumour"
        vox = np.prod(ct.GetSpacing())
        n_tumours = len(tumours)
        exp = max(tumours.values(), key=lambda a: a.sum())
        masks = {"expert": exp}
        for kind in ("ai", "corrected"):
            if kind in kinds:
                seg = segment(fetch(kinds[kind].series_aws_url.iloc[0], tmp / kind), ct,
                              lambda lab: lab.lower() in ("mass", "tumor", "liver tumor", "lesion"))
                if seg:
                    lab_img = sitk.ConnectedComponent(as_image(np.any(list(seg.values()), axis=0), ct))
                    lab = sitk.GetArrayFromImage(lab_img)
                    hit = np.unique(lab[exp & (lab > 0)])
                    masks[kind] = np.isin(lab, hit) if hit.size else np.zeros_like(exp)
        # crop around everything with a margin
        union = np.any(list(masks.values()), axis=0)
        zz, yy, xx = np.nonzero(union)
        sp = ct.GetSpacing()
        pad = [int(np.ceil(15 / s)) + 1 for s in sp]
        lo = [max(0, int(v.min()) - p) for v, p in zip((xx, yy, zz), pad)]
        hi = [min(n, int(v.max()) + 1 + p) for v, p, n in zip((xx, yy, zz), pad, ct.GetSize())]
        size = [h - l for l, h in zip(lo, hi)]
        cct = sitk.RegionOfInterest(ct, size, lo)
        ct_arr = sitk.GetArrayFromImage(cct)
        crop = lambda a: a[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]  # noqa: E731
        rows, agree = [], {"PatientID": pid, "n_expert_tumours": n_tumours,
                           "expert_volume_ml": float(exp.sum() * vox / 1000)}
        cm = {k: as_image(crop(v), cct) for k, v in masks.items()}
        for kind, m in cm.items():
            levels = (-1, 0, 1) if kind == "expert" else (0,)
            detected = sitk.GetArrayFromImage(m).sum() > 0
            if kind != "expert":
                agree[f"{kind}_detected"] = bool(detected)
                if detected:
                    x, y = crop(exp), crop(masks[kind])
                    agree[f"{kind}_dice"] = float(2 * (x & y).sum() / (x.sum() + y.sum()))
                    agree[f"{kind}_assd_mm"] = assd(cm["expert"], m)
                    agree[f"{kind}_volume_ratio"] = float(y.sum() / x.sum())
            if not detected:
                continue
            for mm in levels:
                f = perturbation.extract_perturbed_features(pid, ct_arr, cct, m, mm)
                if f is not None:
                    f.update(PatientID=pid, contour=kind)
                    rows.append(f)
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([agree]).to_csv(CACHE / f"{pid}_agree.csv", index=False)
        pd.DataFrame(rows).to_csv(cached, index=False)
        return pid, f"ok ({len(rows)} rows)"
    except Exception as exc:  # the log records every failure
        return pid, f"error: {exc!r}"[:200]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ser = series_table()
    ser.to_csv(CRLM / "series.csv", index=False)
    jobs = [(pid, g) for pid, g in ser.groupby("PatientID")]
    if "--max-new" in sys.argv:  # only N patients not yet finished (short foreground batches)
        jobs = [j for j in jobs if not (CACHE / f"{j[0]}.csv").exists()][: int(sys.argv[sys.argv.index("--max-new") + 1])]
    if "--first" in sys.argv:
        jobs = jobs[: int(sys.argv[sys.argv.index("--first") + 1])]
    log = []
    with Pool(n) as pool:
        for i, (pid, status) in enumerate(pool.imap_unordered(one, jobs), 1):
            log.append({"PatientID": pid, "status": status})
            print(f"[{i}/{len(jobs)}] {pid}: {status}", flush=True)
    pd.DataFrame(log).to_csv(CRLM / "crlm_log.csv", index=False)
    f = pd.concat([pd.read_csv(x) for x in glob.glob(str(CACHE / "CRLM-CT-????.csv")) if pd.read_csv(x).size])
    a = pd.concat([pd.read_csv(x) for x in glob.glob(str(CACHE / "*_agree.csv"))])
    f.to_csv(CRLM / "crlm_features.csv", index=False)
    a.to_csv(CRLM / "crlm_agreement.csv", index=False)
    print(f"{f.PatientID.nunique()} patients with features; AI detected {a.ai_detected.mean():.1%}; "
          f"median Dice {a.ai_dice.median():.2f}, ASSD {a.ai_assd_mm.median():.2f} mm")


if __name__ == "__main__":
    main()
