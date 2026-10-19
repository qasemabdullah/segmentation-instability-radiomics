"""PyRadiomics: instability ranking compared with the study extractor (30 patients).

Paper:    Fig. 4d, Suppl. Tables S13-S14
Pipeline: python pipeline.py run pyradiomics_extraction   (stage 'pyradiomics')

MUST be run with .venv_pyradiomics's own interpreter (it has pyradiomics
installed; the main project environment does not and should not):
  .venv_pyradiomics/Scripts/python.exe scripts/pyradiomics/run_pyradiomics_crosscheck.py

Method: for a deterministic random sample of patients (default n=30, a
computational-feasibility subset -- PyRadiomics with true 3D GLCM is far
slower per patient than the custom extractor), regenerate the same -1mm/
+1mm perturbed masks used throughout this paper (identical
SignedMaurerDistanceMap logic, replicated inline here since importing the
main nsclc_pipeline package would pull in dependencies not installed in
this isolated environment), extract PyRadiomics shape/first-order/GLCM
features (no resampling, native voxel spacing, matching this study's
own approach of not resampling to an isotropic grid) at both levels, and
compute the same median-absolute-relative-change instability score
already used for Figs 3b and 4c. Only features with a clean 1:1
conceptual mapping to this study's own 26-feature panel are compared;
features with no clean equivalent (e.g. PyRadiomics has no direct
"roundness" or "voxel count") are left out of the comparison, not forced.

Output: data/processed/pyradiomics_crosscheck.csv
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
DICOM_DIR = ROOT / "data" / "interim" / "dicom"
MASKS_DIR = DATA_DIR / "masks"

N_PATIENTS = 30
SEED = 0

# PyRadiomics feature name -> this study's own custom-extractor feature name,
# restricted to conceptually equivalent pairs only (Methods above).
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
    settings = {
        "binWidth": 25,
        "resampledPixelSpacing": None,  # no resampling -- native spacing, matching this study's own approach
        "interpolator": "sitkBSpline",
        "verbose": False,
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("glcm")
    return extractor


def main():
    import radiomics
    radiomics.logger.setLevel(50)  # suppress PyRadiomics' own verbose per-feature logging

    feat_df = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    sample = feat_df.sample(n=min(N_PATIENTS, len(feat_df)), random_state=SEED).sort_values("case_id")
    print(f"Sampled {len(sample)} of {len(feat_df)} patients (seed={SEED})")

    extractor = make_extractor()
    rows = []
    for i, row in enumerate(sample.itertuples(), start=1):
        case_id, ct_uid = row.case_id, row.ct_series_uid
        ct_dir = DICOM_DIR / ct_uid
        mask_path = MASKS_DIR / f"{case_id}_seg.nii.gz"
        if not ct_dir.exists() or not mask_path.exists():
            print(f"[{i}/{len(sample)}] {case_id}: missing cached CT or mask, skipping")
            continue
        print(f"[{i}/{len(sample)}] {case_id}")
        try:
            ct_image = load_ct_volume(ct_dir)
            base_mask = load_mask_on_ct(mask_path, ct_image)
            for mm in (-1, 1):
                pmask = perturb_mask(base_mask, mm)
                if sitk.GetArrayFromImage(pmask).sum() == 0:
                    continue
                result = extractor.execute(ct_image, pmask, label=1)
                feats = {"case_id": case_id, "perturbation_mm": mm}
                for pr_name, our_name in FEATURE_MAP.items():
                    if pr_name in result:
                        feats[our_name] = float(result[pr_name])
                rows.append(feats)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

    df = pd.DataFrame(rows)
    out_path = DATA_DIR / "pyradiomics_crosscheck_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path} ({len(df)} rows)")

    # per-feature instability, same formula as perturbation.feature_stability_summary
    a = df[df["perturbation_mm"] == -1].set_index("case_id")
    b = df[df["perturbation_mm"] == 1].set_index("case_id")
    common_cases = a.index.intersection(b.index)
    a, b = a.loc[common_cases], b.loc[common_cases]

    stability_rows = []
    for col in FEATURE_MAP.values():
        if col not in a.columns or col not in b.columns:
            continue
        va, vb = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        denom = np.maximum(np.abs(va), 1e-6)
        rel_change = np.abs(va - vb) / denom
        stability_rows.append({"feature": col, "pyradiomics_median_abs_rel_change": float(np.median(rel_change))})
    pd.DataFrame(stability_rows).sort_values("pyradiomics_median_abs_rel_change", ascending=False)

    # compare against this study's own custom-extractor instability ranking (already computed for Fig. 6a)
    pert = pd.read_csv(DATA_DIR / "perturbation_features.csv")
    ap = pert[pert["perturbation_mm"] == -1].set_index("case_id")
    bp = pert[pert["perturbation_mm"] == 1].set_index("case_id")
    common_cases2 = ap.index.intersection(bp.index).intersection(common_cases)
    ap, bp = ap.loc[common_cases2], bp.loc[common_cases2]
    a2, b2 = a.loc[common_cases2], b.loc[common_cases2]

    compare_rows = []
    for col in FEATURE_MAP.values():
        if col not in ap.columns or col not in a2.columns:
            continue
        va_c, vb_c = ap[col].to_numpy(dtype=float), bp[col].to_numpy(dtype=float)
        denom_c = np.maximum(np.abs(va_c), 1e-6)
        custom_rel = np.median(np.abs(va_c - vb_c) / denom_c)

        va_p, vb_p = a2[col].to_numpy(dtype=float), b2[col].to_numpy(dtype=float)
        denom_p = np.maximum(np.abs(va_p), 1e-6)
        pr_rel = np.median(np.abs(va_p - vb_p) / denom_p)

        compare_rows.append({"feature": col, "custom_extractor_instability": custom_rel, "pyradiomics_instability": pr_rel})

    compare = pd.DataFrame(compare_rows)
    from scipy.stats import spearmanr
    rho, p = spearmanr(compare["custom_extractor_instability"], compare["pyradiomics_instability"])
    print(f"\nSpearman rho (custom-extractor instability rank vs. PyRadiomics instability rank): rho={rho:.3f}, p={p:.4f}, n_features={len(compare)}, n_patients={len(common_cases2)}")
    compare_path = DATA_DIR / "pyradiomics_crosscheck.csv"
    compare.to_csv(compare_path, index=False)
    print(compare.sort_values("custom_extractor_instability", ascending=False).to_string(index=False))
    print(f"\nSaved -> {compare_path}")


if __name__ == "__main__":
    main()
