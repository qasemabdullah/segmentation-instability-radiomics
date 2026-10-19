"""PyRadiomics: reference features at the expert segmentation.

Paper:    Fig. 4d, Suppl. Tables S13-S14
Pipeline: python pipeline.py run pyradiomics_extraction   (stage 'pyradiomics')

This is the feature table that feeds the "full swap" primary re-analysis
(classification, survival, clustering, SHAP) -- distinct from
pyradiomics_perturbation_features.csv (the 22-feature, 5-perturbation-
level table already used for the Fig. 4d cross-check, left
as-is rather than re-extracted, since the range conclusions it supports
are already verified and re-deriving them for two more features is not
worth 720 more extractions).

MUST be run with .venv_pyradiomics's own interpreter:
  .venv_pyradiomics/Scripts/python.exe scripts/pyradiomics/run_pyradiomics_primary_baseline.py

Output: data/processed/pyradiomics_features.csv (case_id + 24 feature
columns, same naming convention as radiomics_features.csv so it drops
into modeling.load_dataset()/survival.load_survival_dataset() via their
radiomics_path override, unchanged).
"""

from __future__ import annotations

import multiprocessing
import pathlib
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
DICOM_DIR = ROOT / "data" / "interim" / "dicom"
MASKS_DIR = DATA_DIR / "masks"

N_WORKERS = 4  # matches the safe worker count established in run_pyradiomics_full_extraction.py
MAX_RETRIES = 2

FEATURE_MAP = {
    "original_shape_VoxelVolume": "shape_volume_mm3",
    "original_shape_SurfaceArea": "shape_surface_area_mm2",
    "original_shape_Sphericity": "shape_sphericity",
    "original_shape_Maximum3DDiameter": "shape_max_diameter_mm",
    "original_shape_Flatness": "shape_flatness",
    "original_shape_Elongation": "shape_elongation",
    "original_firstorder_Mean": "firstorder_mean_hu",
    "original_firstorder_Median": "firstorder_median_hu",
    "original_firstorder_StandardDeviation": "firstorder_std_hu",
    "original_firstorder_Energy": "firstorder_energy",
    "original_firstorder_Entropy": "firstorder_entropy",
    "original_firstorder_Range": "firstorder_range_hu",
    "original_firstorder_Maximum": "firstorder_max_hu",
    "original_firstorder_Minimum": "firstorder_min_hu",
    "original_firstorder_Skewness": "firstorder_skewness",
    "original_firstorder_Kurtosis": "firstorder_kurtosis",
    "original_firstorder_InterquartileRange": "firstorder_iqr_hu",
    "original_firstorder_MeanAbsoluteDeviation": "firstorder_mad_hu",
    "original_glcm_Contrast": "texture_contrast",
    "original_glcm_Correlation": "texture_correlation",
    "original_glcm_JointEnergy": "texture_energy",
    "original_glcm_Idm": "texture_homogeneity",
    "original_glcm_DifferenceAverage": "texture_dissimilarity",
}
# shape_equivalent_spherical_radius_mm is derived below from VoxelVolume,
# not pulled directly from PyRadiomics output -- added after mapping.
# shape_roundness and firstorder_voxel_count have no PyRadiomics equivalent.


def perturb_mask(mask, mm):
    return mask  # baseline only -- no perturbation applied in this script


def load_ct_volume(ct_dir: pathlib.Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(ct_dir))
    if not files:
        raise FileNotFoundError(f"No DICOM series found in {ct_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def load_mask_on_ct(mask_path: pathlib.Path, ct_image: sitk.Image) -> sitk.Image:
    mask = sitk.ReadImage(str(mask_path))
    return sitk.Resample(mask, ct_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())


def make_extractor():
    from radiomics import featureextractor
    settings = {"binWidth": 25, "resampledPixelSpacing": None, "interpolator": "sitkBSpline", "verbose": False}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    return extractor


def process_patient(args):
    case_id, ct_uid = args
    import radiomics
    radiomics.logger.setLevel(50)

    ct_dir = DICOM_DIR / ct_uid
    mask_path = MASKS_DIR / f"{case_id}_seg.nii.gz"
    if not ct_dir.exists() or not mask_path.exists():
        return case_id, None, "missing cached CT or mask"

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            extractor = make_extractor()
            ct_image = load_ct_volume(ct_dir)
            mask_image = load_mask_on_ct(mask_path, ct_image)
            result = extractor.execute(ct_image, mask_image, label=1)
            feats = {"case_id": case_id}
            for pr_name, our_name in FEATURE_MAP.items():
                if pr_name in result:
                    feats[our_name] = float(result[pr_name])
            volume_mm3 = float(result["original_shape_VoxelVolume"])
            feats["shape_equivalent_spherical_radius_mm"] = (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3)
            return case_id, feats, None
        except Exception as exc:
            last_exc = exc
            is_mem_error = "allocate" in str(exc).lower() or "memory" in str(exc).lower()
            if attempt <= MAX_RETRIES and is_mem_error:
                print(f"  {case_id} attempt {attempt} hit a memory error, retrying after backoff: {exc}", flush=True)
                time.sleep(5 * attempt)
                continue
            break
    print(f"  {case_id} FAILED after {attempt} attempt(s): {last_exc}", flush=True)
    return case_id, None, str(last_exc)


def main():
    feat_df = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    patients = list(feat_df[["case_id", "ct_series_uid"]].itertuples(index=False, name=None))
    print(f"Extracting PyRadiomics baseline features for {len(patients)} patients, {len(FEATURE_MAP) + 1} features, {N_WORKERS} workers")

    t0 = time.time()
    rows = []
    failures = []
    done = 0
    with multiprocessing.Pool(N_WORKERS) as pool:
        for case_id, feats, err in pool.imap_unordered(process_patient, patients):
            done += 1
            if feats is not None:
                rows.append(feats)
            if err is not None:
                failures.append((case_id, err))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(patients) - done) / rate if rate > 0 else float("nan")
            print(f"[{done}/{len(patients)}] {case_id} -- {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "pyradiomics_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(df)} patients, {len(df.columns) - 1} features)")
    print(f"Total time: {time.time() - t0:.0f}s")
    if failures:
        print(f"\n{len(failures)} patient(s) FAILED completely:")
        for case_id, err in failures:
            print(f"  {case_id}: {err}")
    else:
        print("\nNo failures -- all patients extracted successfully.")


if __name__ == "__main__":
    main()
