"""Features of the AI segmentations adjusted by ±1 mm.

Paper:    Fig. 3g, Fig. 4g, Suppl. Table S15
Pipeline: python pipeline.py run ai_models   (stage 'ai')

Output: data/processed/radiomics_features_bamf_ai_perturb.csv (one row per
        patient x shift, shift in column perturbation_mm)
Run:    python scripts/ai_segmentation/run_ai_segmentation_features.py
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, perturbation, viz  # noqa: E402


def main():
    rad = pd.read_csv(config.RADIOMICS_OUT).set_index("case_id")
    ai = pd.read_csv(config.PROCESSED_DIR / "radiomics_features_bamf_ai.csv")
    rows = []
    for i, case in enumerate(sorted(ai.case_id), 1):
        ct = viz.load_ct_volume(ROOT / "data" / "interim" / "dicom" / rad.loc[case, "ct_series_uid"])
        ct_arr = sitk.GetArrayFromImage(ct)
        mask = sitk.ReadImage(str(config.PROCESSED_DIR / "masks_bamf" / f"{case}_ai.nii.gz"))
        mask.CopyInformation(ct)
        for mm in (-1, 1):
            f = perturbation.extract_perturbed_features(case, ct_arr, ct, sitk.Cast(mask, sitk.sitkUInt8), mm)
            if f is not None:
                rows.append(f)
        print(f"[{i}/{len(ai)}] {case}", flush=True)
    out = pd.DataFrame(rows).merge(rad[["ct_series_uid", "seg_series_uid"]].reset_index(), on="case_id", how="left")
    out.to_csv(config.PROCESSED_DIR / "radiomics_features_bamf_ai_perturb.csv", index=False)
    print(f"Saved {len(out)} rows")


if __name__ == "__main__":
    main()
