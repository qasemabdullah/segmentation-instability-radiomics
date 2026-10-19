"""Radiomics feature extraction over CT + tumor-mask pairs.

This is NOT PyRadiomics. PyRadiomics is unmaintained and doesn't build on
Python 3.12 (its bundled versioneer.py calls a configparser API removed in
3.12, and there's no prebuilt wheel for 3.12 on PyPI -- confirmed by
trying). Instead this uses:
  - shape features: SimpleITK's LabelShapeStatisticsImageFilter (built in,
    no extra dependency)
  - first-order intensity features: numpy/scipy.stats over the HU values
    inside the mask
  - texture features: scikit-image GLCM (gray-level co-occurrence matrix),
    computed per axial slice that contains tumor and averaged, weighted by
    each slice's tumor pixel count

These are conceptually the same feature *categories* PyRadiomics computes,
but NOT numerically identical -- most importantly, texture here is a
2.5D approximation (per-slice 2D GLCM, averaged), not PyRadiomics' true 3D
GLCM over 13 directions. Do not directly compare these numbers to
published PyRadiomics-based results without accounting for that.

Each patient's CT+mask are resolved and downloaded on demand exactly like
ImageViewerDialog does: the SEG object's own referenced_series_uid picks
the correct CT series (a patient can have several; only one matches).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats as sp_stats
from skimage.feature import graycomatrix, graycoprops

from . import config, download, seg_to_mask, viz

GLCM_LEVELS = 32
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPERTIES = ["contrast", "correlation", "energy", "homogeneity", "dissimilarity"]


def max_diameter_mm(mask_image: sitk.Image, label: int = 1) -> float:
    """Maximum 3D diameter (Feret diameter) of the label: the largest distance
    between two voxel centres. That maximum is always attained between two
    vertices of the point set's convex hull, so only hull vertices are
    compared; this returns exactly ITK's GetFeretDiameter (checked on Lung1
    tumours) without its quadratic cost on large masks."""
    from scipy.spatial import ConvexHull, QhullError
    from scipy.spatial.distance import pdist
    idx = np.argwhere(sitk.GetArrayFromImage(mask_image) == label)[:, ::-1].astype(float)
    if len(idx) < 2:
        return 0.0
    try:
        idx = idx[ConvexHull(idx).vertices]
    except (QhullError, ValueError):  # flat or tiny objects: use all points
        pass
    pts = np.array([mask_image.TransformContinuousIndexToPhysicalPoint(p) for p in idx])
    return float(pdist(pts).max())


def shape_features(mask_image: sitk.Image, label: int = 1) -> dict:
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(mask_image)
    if label not in f.GetLabels():
        return {}
    volume = f.GetPhysicalSize(label)
    surface_area = f.GetPerimeter(label)  # 3D image -> this is surface area, not a 2D perimeter
    sphericity = (
        (np.pi ** (1 / 3)) * (6 * volume) ** (2 / 3) / surface_area if surface_area > 0 else float("nan")
    )
    return {
        "shape_volume_mm3": volume,
        "shape_surface_area_mm2": surface_area,
        "shape_sphericity": sphericity,
        "shape_elongation": f.GetElongation(label),
        "shape_flatness": f.GetFlatness(label),
        "shape_roundness": f.GetRoundness(label),
        "shape_equivalent_spherical_radius_mm": f.GetEquivalentSphericalRadius(label),
        "shape_max_diameter_mm": max_diameter_mm(mask_image, label),
    }


def first_order_features(values: np.ndarray) -> dict:
    v = values.astype(np.float64)
    hist, _ = np.histogram(v, bins=64)
    p = hist[hist > 0] / hist.sum()
    entropy = float(-np.sum(p * np.log2(p)))
    return {
        "firstorder_voxel_count": int(v.size),
        "firstorder_mean_hu": float(v.mean()),
        "firstorder_std_hu": float(v.std()),
        "firstorder_min_hu": float(v.min()),
        "firstorder_max_hu": float(v.max()),
        "firstorder_median_hu": float(np.median(v)),
        "firstorder_range_hu": float(v.max() - v.min()),
        "firstorder_iqr_hu": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "firstorder_mad_hu": float(np.mean(np.abs(v - v.mean()))),
        "firstorder_skewness": float(sp_stats.skew(v)),
        "firstorder_kurtosis": float(sp_stats.kurtosis(v, fisher=False)),
        "firstorder_energy": float(np.sum(v**2)),
        "firstorder_entropy": entropy,
    }


def texture_features(ct_array: np.ndarray, mask_array: np.ndarray, min_pixels: int = 4) -> dict:
    tumor_vals = ct_array[mask_array > 0]
    if tumor_vals.size == 0:
        return {}
    vmin, vmax = float(tumor_vals.min()), float(tumor_vals.max())
    if vmax <= vmin:
        vmax = vmin + 1.0

    z_indices = np.where(mask_array.sum(axis=(1, 2)) > 0)[0]
    per_slice = {p: [] for p in GLCM_PROPERTIES}
    weights = []

    for z in z_indices:
        m = mask_array[z] > 0
        if m.sum() < min_pixels:
            continue
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        roi = ct_array[z, y0:y1, x0:x1].astype(np.float64)
        roi_mask = m[y0:y1, x0:x1]

        quant = np.clip((roi - vmin) / (vmax - vmin) * (GLCM_LEVELS - 1), 0, GLCM_LEVELS - 1)
        quant = quant.astype(np.uint8)
        quant[~roi_mask] = 0  # bounding-box background; a 2.5D approximation, see module docstring

        glcm = graycomatrix(
            quant,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=True,
        )
        for prop in GLCM_PROPERTIES:
            per_slice[prop].append(float(graycoprops(glcm, prop).mean()))
        weights.append(int(m.sum()))

    if not weights:
        return {}
    w = np.array(weights, dtype=np.float64)
    out = {f"texture_{p}": float(np.average(vals, weights=w)) for p, vals in per_slice.items()}
    out["texture_n_slices"] = len(weights)
    return out


def resolve_ct_and_mask(case_id: str, series_index: pd.DataFrame):
    seg_rows = series_index[(series_index["case_id"] == case_id) & (series_index["modality"] == "SEG")]
    if seg_rows.empty:
        raise ValueError(f"{case_id} has no SEG series in the series index")
    seg_series_uid = seg_rows["series_uid"].iloc[0]

    seg_dir = config.DICOM_DIR / seg_series_uid
    if not (seg_dir.exists() and any(seg_dir.iterdir())):
        print(f"Downloading SEG for {case_id}...")
        seg_dir = download.download_series(seg_series_uid)

    mask_path = config.MASKS_DIR / f"{case_id}_seg.nii.gz"
    if not mask_path.exists():
        seg_to_mask.build_one(case_id, seg_dir)

    seg_dicom = seg_to_mask.find_seg_dicom(seg_dir)
    ct_series_uid = seg_to_mask.referenced_ct_series_uid(seg_dicom)

    ct_dir = config.DICOM_DIR / ct_series_uid
    if not (ct_dir.exists() and any(ct_dir.iterdir())):
        print(f"Downloading CT series {ct_series_uid} for {case_id} (typically 50-300MB)...")
        ct_dir = download.download_series(ct_series_uid)

    return ct_dir, mask_path, ct_series_uid, seg_series_uid


def extract_one(case_id: str, series_index: pd.DataFrame | None = None) -> dict:
    series_index = series_index if series_index is not None else pd.read_csv(config.SERIES_INDEX_OUT)
    ct_dir, mask_path, ct_series_uid, seg_series_uid = resolve_ct_and_mask(case_id, series_index)

    ct_image = viz.load_ct_volume(ct_dir)
    mask_image = viz.load_mask_on_ct(mask_path, ct_image)
    ct_array = sitk.GetArrayFromImage(ct_image)
    mask_array = sitk.GetArrayFromImage(mask_image)

    tumor_vals = ct_array[mask_array > 0]
    if tumor_vals.size == 0:
        raise ValueError(f"{case_id}: mask has zero voxels after resampling onto the CT grid")

    features = {"case_id": case_id, "ct_series_uid": ct_series_uid, "seg_series_uid": seg_series_uid}
    features.update(shape_features(mask_image))
    features.update(first_order_features(tumor_vals))
    features.update(texture_features(ct_array, mask_array))
    return features


def build(case_ids=None, out_path=None, progress_cb=None, resume=True) -> pd.DataFrame:
    """Extract features for each case ID (default: every patient with a
    SEG series), saving after every patient so a long batch run (each
    patient may need a CT download) can be interrupted and resumed.

    progress_cb(i, total, case_id), if given, is called before each
    patient is processed (i is 1-based, counting all targets including
    ones about to be skipped because they're already done)."""
    out_path = out_path or config.RADIOMICS_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    series_index = pd.read_csv(config.SERIES_INDEX_OUT)
    seg_case_ids = sorted(series_index[series_index["modality"] == "SEG"]["case_id"].unique())
    targets = case_ids or seg_case_ids

    done = pd.read_csv(out_path) if (resume and out_path.exists()) else pd.DataFrame()
    done_ids = set(done["case_id"]) if len(done) else set()
    rows = done.to_dict("records")

    for i, case_id in enumerate(targets, start=1):
        if progress_cb:
            progress_cb(i, len(targets), case_id)
        if case_id in done_ids:
            print(f"[{i}/{len(targets)}] {case_id}: already done, skipping")
            continue
        print(f"[{i}/{len(targets)}] {case_id}")
        try:
            rows.append(extract_one(case_id, series_index))
        except Exception as e:  # noqa: BLE001 -- keep going across a batch, report at the end
            print(f"FAILED {case_id}: {type(e).__name__}: {e}")
            continue
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = build()
    print(f"Extracted radiomics features for {len(result)} patients -> {config.RADIOMICS_OUT}")
