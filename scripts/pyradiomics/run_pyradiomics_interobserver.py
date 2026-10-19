"""PyRadiomics: features of the five radiation oncologists' segmentations.

Paper:    Fig. 4d, Suppl. Tables S13-S14
Pipeline: python pipeline.py run pyradiomics_extraction   (stage 'pyradiomics')

MUST be run with .venv_pyradiomics's own interpreter:
  .venv_pyradiomics/Scripts/python.exe scripts/pyradiomics/run_pyradiomics_interobserver.py

Output: data/processed/pyradiomics_interobserver_features.csv
  (columns: patient_id, observer, + 23 feature columns)
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
NIFTI_DIR = ROOT / "data" / "interim" / "interobserver_nifti"

N_WORKERS = 4
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


def make_extractor():
    from radiomics import featureextractor
    settings = {"binWidth": 25, "resampledPixelSpacing": None, "interpolator": "sitkBSpline", "verbose": False}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    return extractor


def process_row(args):
    patient_id, observer, ct_path, mask_path = args
    import radiomics
    radiomics.logger.setLevel(50)

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            extractor = make_extractor()
            ct_image = sitk.ReadImage(ct_path)
            mask_image = sitk.ReadImage(mask_path)
            result = extractor.execute(ct_image, mask_image, label=1)
            feats = {"patient_id": patient_id, "observer": observer}
            for pr_name, our_name in FEATURE_MAP.items():
                if pr_name in result:
                    feats[our_name] = float(result[pr_name])
            volume_mm3 = float(result["original_shape_VoxelVolume"])
            feats["shape_equivalent_spherical_radius_mm"] = (3 * volume_mm3 / (4 * np.pi)) ** (1 / 3)
            return patient_id, observer, feats, None
        except Exception as exc:
            last_exc = exc
            is_mem_error = "allocate" in str(exc).lower() or "memory" in str(exc).lower()
            if attempt <= MAX_RETRIES and is_mem_error:
                print(f"  {patient_id} obs{observer} attempt {attempt} memory error, retrying: {exc}", flush=True)
                time.sleep(5 * attempt)
                continue
            break
    print(f"  {patient_id} obs{observer} FAILED after {attempt} attempt(s): {last_exc}", flush=True)
    return patient_id, observer, None, str(last_exc)


def main():
    manifest = pd.read_csv(NIFTI_DIR / "manifest.csv")
    tasks = list(manifest[["patient_id", "observer", "ct_path", "mask_path"]].itertuples(index=False, name=None))
    print(f"Extracting PyRadiomics features for {len(tasks)} (patient, observer) pairs, {N_WORKERS} workers")

    t0 = time.time()
    rows = []
    failures = []
    done = 0
    with multiprocessing.Pool(N_WORKERS) as pool:
        for patient_id, observer, feats, err in pool.imap_unordered(process_row, tasks):
            done += 1
            if feats is not None:
                rows.append(feats)
            if err is not None:
                failures.append((patient_id, observer, err))
            elapsed = time.time() - t0
            print(f"[{done}/{len(tasks)}] {patient_id} obs{observer} -- {elapsed:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "pyradiomics_interobserver_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(df)} rows, {df['patient_id'].nunique()} patients)")
    print(f"Total time: {time.time() - t0:.0f}s")
    if failures:
        print(f"\n{len(failures)} FAILED:")
        for patient_id, observer, err in failures:
            print(f"  {patient_id} obs{observer}: {err}")
    else:
        print("\nNo failures.")


if __name__ == "__main__":
    main()
