"""LIDC-IDRI step 2: features of every reader's segmentation at −1, 0 and +1 mm.

Paper:    Table 2
Pipeline: python pipeline.py run lidc_data   (stage 'external')

Needs data/external/lidc/annotations.csv from lidc_download.py.

Output: data/external/lidc/cache/<PatientID>.csv          (features)
        data/external/lidc/cache/<PatientID>_pairs.csv    (reader agreement)
        data/external/lidc/crops/<PatientID>_n<k>.npz      (CT crop + reader masks)
        data/external/lidc/lidc_features.csv, lidc_pairs.csv, lidc_extraction_log.csv
Run:    OPENBLAS_NUM_THREADS=1 python scripts/data/lidc_features.py [workers]
"""
from __future__ import annotations

import glob
import itertools
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
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import perturbation, viz  # noqa: E402

LIDC = ROOT / "data" / "external" / "lidc"
ANN, CACHE, CROPS, TMP = LIDC / "ann", LIDC / "cache", LIDC / "crops", LIDC / "tmp"
LEVELS = (-1, 0, 1)
MARGIN_MM = 30.0  # wide enough for a 50-mm seed-centred cube (foundation-model features)


def reader_mask(seg_uid: str, ct: sitk.Image) -> sitk.Image:
    ds = pydicom.dcmread(glob.glob(str(ANN / f"SEG_{seg_uid}" / "*.dcm"))[0])
    img = pydicom_seg.SegmentReader().read(ds).segment_image(ds.SegmentSequence[0].SegmentNumber)
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "m.nii.gz"
        sitk.WriteImage(img, str(p))
        return sitk.Cast(viz.load_mask_on_ct(p, ct) > 0, sitk.sitkUInt8)


def union_crop(ct, masks):
    arr = np.any([sitk.GetArrayFromImage(m) > 0 for m in masks], axis=0)
    zz, yy, xx = np.nonzero(arr)
    sp = ct.GetSpacing()
    pad = [int(np.ceil(MARGIN_MM / s)) + 1 for s in sp]
    lo = [max(0, int(v.min()) - p) for v, p in zip((xx, yy, zz), pad)]
    hi = [min(n, int(v.max()) + 1 + p) for v, p, n in zip((xx, yy, zz), pad, ct.GetSize())]
    size = [h - l for l, h in zip(lo, hi)]
    return sitk.RegionOfInterest(ct, size, lo), [sitk.RegionOfInterest(m, size, lo) for m in masks]


def assd(a: sitk.Image, b: sitk.Image) -> float:
    """Average symmetric surface distance (mm) between two binary masks."""
    def surf_dist(x, y):
        contour = sitk.GetArrayFromImage(sitk.LabelContour(x)) > 0
        dmap = np.abs(sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
            y, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)))
        return dmap[contour]
    d = np.concatenate([surf_dist(a, b), surf_dist(b, a)])
    return float(d.mean()) if d.size else float("nan")


def one(pid: str):
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    cached = CACHE / f"{pid}.csv"
    crops_ok = all(np.load(c)["margin"] >= MARGIN_MM if "margin" in np.load(c) else False
                   for c in CROPS.glob(f"{pid}_n*.npz")) and any(CROPS.glob(f"{pid}_n*.npz"))
    if cached.exists() and crops_ok:
        return pid, "cached"
    crop_only = cached.exists()  # features done with a narrower crop: only redo the crops
    ann = pd.read_csv(LIDC / "annotations.csv")
    ann = ann[ann.PatientID == pid]
    tmp = TMP / pid
    try:
        urls = pd.read_csv(LIDC / "ct_urls.csv").set_index("SeriesInstanceUID").series_aws_url
        rows, pairs = [], []
        for ct_uid, a_ct in ann.groupby("ct_series_uid"):
            shutil.rmtree(tmp, ignore_errors=True)
            (tmp / ct_uid).mkdir(parents=True)
            # direct s5cmd copy: loading IDCClient's full index in every worker exhausted the RAM
            subprocess.run([shutil.which("s5cmd"), "--no-sign-request", "--endpoint-url", "https://s3.amazonaws.com",
                            "cp", urls[ct_uid], str(tmp / ct_uid) + "/"], check=True, capture_output=True,
                           timeout=900)  # a stalled cloud download fails this patient instead of blocking the worker
            ct = viz.load_ct_volume(tmp / ct_uid)
            for k, a_n in a_ct.groupby("nodule"):
                a_n = a_n.sort_values("reader")
                masks = [reader_mask(u, ct) for u in a_n.seg_series_uid]
                if any(sitk.GetArrayFromImage(m).sum() == 0 for m in masks):
                    continue
                cct, cmasks = union_crop(ct, masks)
                ct_arr = sitk.GetArrayFromImage(cct)
                CROPS.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(CROPS / f"{pid}_n{k}.npz", ct=ct_arr.astype(np.int16),
                                    masks=np.stack([sitk.GetArrayFromImage(m) for m in cmasks]).astype(np.uint8),
                                    spacing=cct.GetSpacing(), origin=cct.GetOrigin(), direction=cct.GetDirection(),
                                    readers=a_n.reader.to_numpy(), margin=MARGIN_MM)
                if crop_only:
                    continue
                for r, m in zip(a_n.itertuples(), cmasks):
                    for mm in LEVELS:
                        f = perturbation.extract_perturbed_features(f"{pid}_n{k}_r{r.reader}", ct_arr, cct, m, mm)
                        if f is not None:
                            f.update(PatientID=pid, nodule=k, reader=r.reader)
                            rows.append(f)
                for (i, ma), (j, mb) in itertools.combinations(zip(a_n.reader, cmasks), 2):
                    x, y = sitk.GetArrayFromImage(ma) > 0, sitk.GetArrayFromImage(mb) > 0
                    pairs.append({"PatientID": pid, "nodule": k, "reader_a": i, "reader_b": j,
                                  "dice": 2 * (x & y).sum() / (x.sum() + y.sum()), "assd_mm": assd(ma, mb)})
            del ct
        if crop_only:
            return pid, "recropped"
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pairs).to_csv(CACHE / f"{pid}_pairs.csv", index=False)
        pd.DataFrame(rows).to_csv(cached, index=False)
        return pid, f"ok ({len(rows)} rows)"
    except Exception as exc:  # keep going; the log records every failure
        return pid, f"error: {exc!r}"[:200]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def collect():
    read = lambda pat: [pd.read_csv(f) for f in sorted(glob.glob(str(CACHE / pat)))  # noqa: E731
                        if pathlib.Path(f).stat().st_size > 5]  # patients with no usable nodule leave empty files
    feats, pairs = read("LIDC-IDRI-????.csv"), read("*_pairs.csv")
    f = pd.concat([x for x in feats if len(x)], ignore_index=True)
    p = pd.concat([x for x in pairs if len(x)], ignore_index=True)
    f.to_csv(LIDC / "lidc_features.csv", index=False)
    p.to_csv(LIDC / "lidc_pairs.csv", index=False)
    print(f"features: {len(f)} rows, {f.groupby(['PatientID', 'nodule']).ngroups} nodules, "
          f"{f.PatientID.nunique()} patients; reader pairs: {len(p)} (median Dice {p.dice.median():.2f}, "
          f"median ASSD {p.assd_mm.median():.2f} mm)")


def main():
    if "--collect" in sys.argv:
        return collect()
    n = int(next((a for a in sys.argv[1:] if a.isdigit()), 4))
    ann = pd.read_csv(LIDC / "annotations.csv")
    if not (LIDC / "ct_urls.csv").exists():
        from idc_index import IDCClient
        idx = IDCClient().index
        idx[idx.SeriesInstanceUID.isin(ann.ct_series_uid)][["SeriesInstanceUID", "series_aws_url"]].to_csv(
            LIDC / "ct_urls.csv", index=False)
        del idx
    pids = sorted(ann.PatientID.unique())
    if "--max-new" in sys.argv:  # only N patients not yet finished (short foreground batches)
        pids = [x for x in pids if not (CACHE / f"{x}.csv").exists()][: int(sys.argv[sys.argv.index("--max-new") + 1])]
    if "--first" in sys.argv:
        pids = pids[: int(sys.argv[sys.argv.index("--first") + 1])]
    log = []
    with Pool(n) as pool:
        for i, (pid, status) in enumerate(pool.imap_unordered(one, pids), 1):
            log.append({"PatientID": pid, "status": status})
            print(f"[{i}/{len(pids)}] {pid}: {status}", flush=True)
    old = LIDC / "lidc_extraction_log.csv"
    pd.concat(([pd.read_csv(old)] if old.exists() else []) + [pd.DataFrame(log)]).drop_duplicates(
        "PatientID", keep="last").to_csv(old, index=False)
    collect()


if __name__ == "__main__":
    main()
