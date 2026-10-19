"""PyRadiomics: features of all 144 patients at every segmentation adjustment.

Paper:    Fig. 4d, Suppl. Tables S13-S14
Pipeline: python pipeline.py run pyradiomics_extraction   (stage 'pyradiomics')

Only the 22 features with a clean 1:1 conceptual mapping to this study's
own panel are extracted (FEATURE_MAP, shared with the cross-check script);
the other ~4-5 of the original 26 features have no direct PyRadiomics
equivalent and are simply absent from this table -- any downstream model
fit on it will use this 22-feature subset, not the full 26. That is a
real, explicit difference from the main analysis and must be stated
wherever this table's results are reported.

MUST be run with .venv_pyradiomics's own interpreter:
  .venv_pyradiomics/Scripts/python.exe scripts/pyradiomics/run_pyradiomics_full_extraction.py

Parallelized across patients (multiprocessing.Pool) since this machine
has many cores and each patient's extraction is independent; each worker
loads a patient's CT once and reuses it across all 5 perturbation levels.

Output: data/processed/pyradiomics_perturbation_features.csv
"""

from __future__ import annotations

import multiprocessing
import pathlib
import sys
import time

import pandas as pd
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
DICOM_DIR = ROOT / "data" / "interim" / "dicom"
MASKS_DIR = DATA_DIR / "masks"

MM_LEVELS = (-2, -1, 0, 1, 2)
# 12 workers exhausted this machine's 15.5GB RAM and caused real extraction
# FAILURES (SimpleITK/OpenBLAS "failed to allocate memory"), not just
# slowness -- confirmed by a run that silently dropped several patients.
# 4 workers matched a clean, failure-free smoke test; stay well under that.
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


def perturb_mask(mask: sitk.Image, mm: float) -> sitk.Image:
    """The main pipeline's implementation (src/nsclc_pipeline/mask_shift.py),
    loaded by file path because this isolated environment lacks the
    package's other dependencies."""
    import importlib.util
    path = pathlib.Path(__file__).resolve().parents[2] / "src" / "nsclc_pipeline" / "mask_shift.py"
    spec = importlib.util.spec_from_file_location("mask_shift", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.perturb_mask(mask, mm)


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


def _extract_once(case_id, ct_dir, mask_path):
    extractor = make_extractor()
    rows = []
    ct_image = load_ct_volume(ct_dir)
    base_mask = load_mask_on_ct(mask_path, ct_image)
    for mm in MM_LEVELS:
        pmask = perturb_mask(base_mask, mm)
        if sitk.GetArrayFromImage(pmask).sum() == 0:
            continue
        result = extractor.execute(ct_image, pmask, label=1)
        feats = {"case_id": case_id, "perturbation_mm": mm}
        for pr_name, our_name in FEATURE_MAP.items():
            if pr_name in result:
                feats[our_name] = float(result[pr_name])
        rows.append(feats)
    return rows


def process_patient(args):
    case_id, ct_uid = args
    import radiomics
    radiomics.logger.setLevel(50)

    ct_dir = DICOM_DIR / ct_uid
    mask_path = MASKS_DIR / f"{case_id}_seg.nii.gz"
    if not ct_dir.exists() or not mask_path.exists():
        return case_id, [], "missing cached CT or mask"

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            rows = _extract_once(case_id, ct_dir, mask_path)
            return case_id, rows, None
        except Exception as exc:
            last_exc = exc
            is_mem_error = "allocate" in str(exc).lower() or "memory" in str(exc).lower()
            if attempt <= MAX_RETRIES and is_mem_error:
                print(f"  {case_id} attempt {attempt} hit a memory error, retrying after backoff: {exc}", flush=True)
                time.sleep(5 * attempt)
                continue
            break
    print(f"  {case_id} FAILED after {attempt} attempt(s): {last_exc}", flush=True)
    return case_id, [], str(last_exc)


def main():
    feat_df = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    patients = list(feat_df[["case_id", "ct_series_uid"]].itertuples(index=False, name=None))
    out_path = DATA_DIR / "pyradiomics_perturbation_features.csv"
    fail_path = DATA_DIR / "pyradiomics_extraction_failures.csv"
    if "--retry-failures" in sys.argv:
        # re-extract only patients listed in the failures file (e.g. after an out-of-memory
        # episode) and merge them into the existing table
        retry = set(pd.read_csv(fail_path).case_id)
        patients = [p for p in patients if p[0] in retry]
        rows, still = [], []
        for p in patients:
            case_id, r, err = process_patient(p)
            rows.extend(r)
            print(f"{case_id}: {len(r)} rows" + (f" FAILED {err}" if err else ""), flush=True)
            if err is not None:
                still.append((case_id, err))
        old = pd.read_csv(out_path)
        merged = pd.concat([old[~old.case_id.isin(retry)], pd.DataFrame(rows)], ignore_index=True)
        merged.to_csv(out_path, index=False)
        pd.DataFrame(still, columns=["case_id", "error"]).to_csv(fail_path, index=False)
        print(f"Merged -> {out_path} ({merged['case_id'].nunique()} patients); still failing: {len(still)}")
        return
    print(f"Extracting PyRadiomics features for {len(patients)} patients x {len(MM_LEVELS)} levels, {N_WORKERS} workers")

    t0 = time.time()
    all_rows = []
    failures = []
    done = 0
    with multiprocessing.Pool(N_WORKERS) as pool:
        for case_id, rows, err in pool.imap_unordered(process_patient, patients):
            done += 1
            all_rows.extend(rows)
            if err is not None:
                failures.append((case_id, err))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(patients) - done) / rate if rate > 0 else float("nan")
            print(f"[{done}/{len(patients)}] {case_id} ({len(rows)} rows) -- {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    df = pd.DataFrame(all_rows)
    out_path = DATA_DIR / "pyradiomics_perturbation_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(df)} rows, {df['case_id'].nunique()} patients)")
    print(f"Total time: {time.time() - t0:.0f}s")
    if failures:
        print(f"\n{len(failures)} patient(s) FAILED completely (excluded, not silently zero-filled):")
        for case_id, err in failures:
            print(f"  {case_id}: {err}")
        fail_path = DATA_DIR / "pyradiomics_extraction_failures.csv"
        pd.DataFrame(failures, columns=["case_id", "error"]).to_csv(fail_path, index=False)
        print(f"Saved -> {fail_path}")
    else:
        print("\nNo failures -- all patients extracted successfully.")


if __name__ == "__main__":
    main()
