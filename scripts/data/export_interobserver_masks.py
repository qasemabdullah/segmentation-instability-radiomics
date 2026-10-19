"""Multi-reader NSCLC cohort: export each reader's segmentation and the CT as NIfTI.

Paper:    Fig. 4a-b
Pipeline: python pipeline.py run interobserver   (stage 'readers')

Reuses analyze_interobserver.py's own load_manual_segments() so the mask
resolution logic is identical to what the custom-extractor interobserver
analysis already uses (Methods, "Interobserver validation of the
perturbation model") -- this export changes nothing about how a mask is
read, only where it's written.

Run: python scripts/data/export_interobserver_masks.py
Output: data/interim/interobserver_nifti/{patient_id}_ct.nii.gz,
        data/interim/interobserver_nifti/{patient_id}_obs{N}.nii.gz
"""

from __future__ import annotations

import json
import pathlib
import sys

import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.readers.analyze_interobserver import SERIES_JSON, load_manual_segments  # noqa: E402

OUT_DIR = ROOT / "data" / "interim" / "interobserver_nifti"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with open(SERIES_JSON) as f:
        series = json.load(f)
    by_patient = {}
    for d in series:
        by_patient.setdefault(d["PatientID"], {})[d["Modality"]] = d["SeriesInstanceUID"]

    manifest = []
    n_ok = 0
    for patient_id, mods in sorted(by_patient.items()):
        if "SEG" not in mods or "CT" not in mods:
            continue
        print(f"{patient_id} ...")
        try:
            ct_image, ct_array, masks = load_manual_segments(patient_id, mods["SEG"], mods["CT"])
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        if len(masks) != 5:
            print(f"  expected 5 manual observers, got {len(masks)} -- skipping")
            continue

        ct_path = OUT_DIR / f"{patient_id}_ct.nii.gz"
        sitk.WriteImage(ct_image, str(ct_path))
        for obs, mask_img in masks.items():
            mask_path = OUT_DIR / f"{patient_id}_obs{obs}.nii.gz"
            sitk.WriteImage(sitk.Cast(mask_img, sitk.sitkUInt8), str(mask_path))
            manifest.append({"patient_id": patient_id, "observer": obs, "ct_path": str(ct_path), "mask_path": str(mask_path)})
        n_ok += 1

    import pandas as pd
    manifest_df = pd.DataFrame(manifest)
    manifest_path = OUT_DIR / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nExported {n_ok} patients, {len(manifest)} observer masks -> {OUT_DIR}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
