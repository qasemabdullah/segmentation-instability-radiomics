"""AI segmentation step 1: match the public nnU-Net (AIMI) segmentations to each expert tumor.

Paper:    Fig. 4e-f
Pipeline: python pipeline.py run ai_masks   (stage 'ai')

The AI model segments lungs plus nodules of roughly 3-30 mm, so a large
tumour may be missed or split. For each patient the connected components of
the AI "Nodule" segment that overlap the expert tumour are taken as the AI
lesion; if none overlaps, the case is recorded as a detection failure (it is
not silently replaced by the expert segmentation). Radiologist-corrected versions
of the same AI output (two readers, 26 SEG objects) are processed the same way.

Inputs:  data/interim/bamf_seg_selected.csv  (from the IDC query)
         data/interim/bamf_seg/<SEG SeriesInstanceUID>/*.dcm
Outputs: data/processed/masks_bamf/<case>_<source>.nii.gz
         data/processed/bamf_agreement.csv
         data/processed/radiomics_features_bamf_<source>.csv   (source = ai, r1, r2)
Run:     python scripts/ai_segmentation/ai_segmentation_masks.py
"""
from __future__ import annotations

import glob
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pydicom
import pydicom_seg
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, radiomics, viz  # noqa: E402

SEG_DIR = ROOT / "data" / "interim" / "bamf_seg"
MASK_DIR = config.PROCESSED_DIR / "masks_bamf"
MASK_DIR.mkdir(exist_ok=True)
SOURCE = {"AIMI lung and nodule AI segmentation": "ai",
          "AIMI lung and nodule radiologist 1 corrected segmentation": "r1",
          "AIMI lung and nodule radiologist 2 corrected segmentation": "r2"}


def nodule_on_ct(seg_uid, ct):
    f = glob.glob(str(SEG_DIR / seg_uid / "*.dcm"))[0]
    ds = pydicom.dcmread(f)
    labels = {s.SegmentNumber: s.SegmentLabel.lower() for s in ds.SegmentSequence}
    num = next((n for n, lab in labels.items() if "nodule" in lab), None)
    if num is None:  # the model found no nodule in this scan: an empty lesion mask
        empty = sitk.Image(ct.GetSize(), sitk.sitkUInt8)
        empty.CopyInformation(ct)
        return empty
    img = pydicom_seg.SegmentReader().read(ds).segment_image(num)
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "m.nii.gz"
        sitk.WriteImage(img, str(p))
        return viz.load_mask_on_ct(p, ct)


def surface_distance(a, b, spacing_img):
    """Average symmetric surface distance (mm) between two binary masks."""
    ia = sitk.GetImageFromArray(a.astype(np.uint8))
    ib = sitk.GetImageFromArray(b.astype(np.uint8))
    for im in (ia, ib):
        im.CopyInformation(spacing_img)
    da = np.abs(sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(ia, squaredDistance=False, useImageSpacing=True)))
    db = np.abs(sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(ib, squaredDistance=False, useImageSpacing=True)))
    sa = sitk.GetArrayFromImage(sitk.LabelContour(ia)) > 0
    sb = sitk.GetArrayFromImage(sitk.LabelContour(ib)) > 0
    return float(np.concatenate([db[sa], da[sb]]).mean())


def features(case_id, ct_arr, mask_img):
    m = sitk.GetArrayFromImage(mask_img) > 0
    out = {"case_id": case_id}
    out.update(radiomics.shape_features(sitk.Cast(mask_img, sitk.sitkUInt8)))
    out.update(radiomics.first_order_features(ct_arr[m]))
    out.update(radiomics.texture_features(ct_arr, m.astype(np.uint8)))
    return out


def main():
    sel = pd.read_csv(ROOT / "data" / "interim" / "bamf_seg_selected.csv")
    sel["source"] = sel.SeriesDescription.map(SOURCE)
    rad = pd.read_csv(config.RADIOMICS_OUT).set_index("case_id")
    agree, feats = [], {s: [] for s in SOURCE.values()}
    for i, r in enumerate(sel.sort_values(["PatientID", "source"]).itertuples(), 1):
        case = r.PatientID
        ct = viz.load_ct_volume(ROOT / "data" / "interim" / "dicom" / rad.loc[case, "ct_series_uid"])
        ct_arr = sitk.GetArrayFromImage(ct)
        expert = sitk.GetArrayFromImage(viz.load_mask_on_ct(config.PROCESSED_DIR / "masks" / f"{case}_seg.nii.gz", ct)) > 0
        ai = sitk.GetArrayFromImage(nodule_on_ct(r.seg_uid, ct)) > 0
        cc = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk.GetImageFromArray(ai.astype(np.uint8))))
        hit = np.unique(cc[expert & (cc > 0)])
        lesion = np.isin(cc, hit) & (cc > 0)
        row = {"case_id": case, "source": r.source, "ai_nodule_voxels": int(ai.sum()),
               "n_ai_components": int(cc.max()), "expert_voxels": int(expert.sum()),
               "detected": bool(lesion.any())}
        if lesion.any():
            inter = (lesion & expert).sum()
            row.update(dice=2 * inter / (lesion.sum() + expert.sum()),
                       volume_ratio=lesion.sum() / expert.sum(),
                       assd_mm=surface_distance(lesion, expert, ct))
            mimg = sitk.GetImageFromArray(lesion.astype(np.uint8))
            mimg.CopyInformation(ct)
            sitk.WriteImage(mimg, str(MASK_DIR / f"{case}_{r.source}.nii.gz"))
            feats[r.source].append(features(case, ct_arr, mimg))
        agree.append(row)
        d = row.get("dice", float("nan"))
        print(f"[{i}/{len(sel)}] {case} {r.source}: detected={row['detected']} dice={d:.3f}", flush=True)

    pd.DataFrame(agree).to_csv(config.PROCESSED_DIR / "bamf_agreement.csv", index=False)
    for src, rows in feats.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # keep the same identifying columns as the expert table so modeling.load_dataset accepts it
        df = df.merge(rad[["ct_series_uid", "seg_series_uid"]].reset_index(), on="case_id", how="left")
        df.to_csv(config.PROCESSED_DIR / f"radiomics_features_bamf_{src}.csv", index=False)
        print(f"{src}: features for {len(df)} patients")


if __name__ == "__main__":
    main()
