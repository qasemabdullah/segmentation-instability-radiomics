"""Exploratory 3D intratumoral habitats (supervoxels, k-means).

Paper:    Suppl. Fig. S8
Pipeline: python pipeline.py run habitats   (stage 'discovery')

For each of the 144 segmented patients, the full 3D tumor volume (mask
bounding box, 6-voxel margin) is over-segmented into 3D supervoxels
(SLIC, N-dimensional, spacing-aware, restricted to the tumor mask).
Each supervoxel's mean and standard deviation of Hounsfield units are
pooled across the full cohort and clustered (k-means, k=6) into a shared
set of habitat types, exactly as in the 2D version -- only the
segmentation stage changed from 2D to 3D.

Run in the MAIN environment:
  python scripts/discovery/run_ct_habitat_supervoxels_3d.py

Output: data/processed/ct_habitat3d_supervoxels.csv (one row per supervoxel)
        data/processed/ct_habitat3d_composition.csv (one row per patient,
        habitat volume-fraction columns)
        data/processed/ct_habitat3d_cluster_centers.csv
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.segmentation import slic
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import config, viz  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
N_HABITATS = 6
LUNG_WINDOW = viz.WINDOW_PRESETS["Lung"]


def target_n_segments_3d(n_voxels):
    """~1 supervoxel per 150 tumor voxels (floor 10, cap 100). Calibrated
    against this cohort's voxel-count distribution (median ~10,257,
    min 756, max 357,068; firstorder_voxel_count in radiomics_features.csv),
    aiming for a similar granularity-per-tumor-size ratio as the 2D
    central-slice version's floor/cap (8/60 per ~25 px)."""
    return max(10, min(100, n_voxels // 150))


def full_volume_crop(case_id, ct_series_uid, pad_vox=6):
    ct_dir = ROOT / "data" / "interim" / "dicom" / ct_series_uid
    mask_path = DATA_DIR / "masks" / f"{case_id}_seg.nii.gz"
    ct = viz.load_ct_volume(ct_dir)
    mask = viz.load_mask_on_ct(mask_path, ct)
    ct_arr = sitk.GetArrayFromImage(ct)  # (z, y, x)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    sx, sy, sz = ct.GetSpacing()  # sitk order is (x, y, z)
    spacing_zyx = (sz, sy, sx)

    zs, ys, xs = np.where(mask_arr)
    z0, z1 = max(zs.min() - pad_vox, 0), min(zs.max() + pad_vox + 1, ct_arr.shape[0])
    y0, y1 = max(ys.min() - pad_vox, 0), min(ys.max() + pad_vox + 1, ct_arr.shape[1])
    x0, x1 = max(xs.min() - pad_vox, 0), min(xs.max() + pad_vox + 1, ct_arr.shape[2])

    hu_vol = ct_arr[z0:z1, y0:y1, x0:x1].astype(np.float32)
    mask_vol = mask_arr[z0:z1, y0:y1, x0:x1]
    return hu_vol, mask_vol, spacing_zyx


def extract_supervoxels(case_id, hu_vol, mask_vol, spacing_zyx):
    n_vox = int(mask_vol.sum())
    n_segments = target_n_segments_3d(n_vox)
    windowed = np.clip(hu_vol, LUNG_WINDOW[0] - LUNG_WINDOW[1] / 2, LUNG_WINDOW[0] + LUNG_WINDOW[1] / 2)

    labels = slic(
        windowed, n_segments=n_segments, mask=mask_vol, compactness=0.05,
        sigma=0.6, channel_axis=None, start_label=1, spacing=spacing_zyx,
    )
    voxel_mm3 = spacing_zyx[0] * spacing_zyx[1] * spacing_zyx[2]
    rows = []
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        region_mask = labels == lbl
        vals = hu_vol[region_mask]
        cz, cy, cx = np.mean(np.argwhere(region_mask), axis=0)
        rows.append({
            "case_id": case_id, "supervoxel_id": int(lbl),
            "n_voxels": int(region_mask.sum()), "volume_mm3": float(region_mask.sum() * voxel_mm3),
            "mean_hu": float(vals.mean()), "std_hu": float(vals.std()),
            "cz": float(cz), "cy": float(cy), "cx": float(cx),
        })
    return rows


def main():
    feats = pd.read_csv(config.RADIOMICS_OUT)
    all_rows = []
    failures = []
    for i, row in feats.iterrows():
        case_id, ct_uid = row["case_id"], row["ct_series_uid"]
        try:
            hu_vol, mask_vol, spacing_zyx = full_volume_crop(case_id, ct_uid)
            rows = extract_supervoxels(case_id, hu_vol, mask_vol, spacing_zyx)
            all_rows.extend(rows)
        except Exception as exc:  # keep going; report failures explicitly, never drop silently
            failures.append((case_id, str(exc)))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(feats)} patients processed, {len(all_rows)} supervoxels so far")

    sv = pd.DataFrame(all_rows)
    print(f"\nExtracted {len(sv)} supervoxels from {sv['case_id'].nunique()} patients "
          f"(mean {len(sv) / sv['case_id'].nunique():.1f} per patient)")
    if failures:
        print(f"Failures ({len(failures)}): {failures}")

    scaler = StandardScaler().fit(sv[["mean_hu", "std_hu"]].to_numpy())
    X = scaler.transform(sv[["mean_hu", "std_hu"]].to_numpy())
    km = KMeans(n_clusters=N_HABITATS, random_state=0, n_init=10)
    sv["habitat"] = km.fit_predict(X)
    print(f"\nHabitat cluster sizes (by supervoxel count):\n{sv['habitat'].value_counts().sort_index()}")
    print(f"\nHabitat characterization (mean HU, s.d. HU):\n{sv.groupby('habitat')[['mean_hu', 'std_hu']].mean().round(1)}")

    centers_df = pd.DataFrame(km.cluster_centers_, columns=["mean_hu_z", "std_hu_z"])
    centers_df["scaler_mean_hu"] = scaler.mean_[0]
    centers_df["scaler_std_hu_mean"] = scaler.mean_[1]
    centers_df["scaler_scale_hu"] = scaler.scale_[0]
    centers_df["scaler_scale_std_hu"] = scaler.scale_[1]
    centers_df.to_csv(DATA_DIR / "ct_habitat3d_cluster_centers.csv", index=False)
    print(f"Saved -> {DATA_DIR / 'ct_habitat3d_cluster_centers.csv'}")

    sv_path = DATA_DIR / "ct_habitat3d_supervoxels.csv"
    sv.to_csv(sv_path, index=False)
    print(f"Saved -> {sv_path}")

    comp = sv.pivot_table(index="case_id", columns="habitat", values="volume_mm3", aggfunc="sum", fill_value=0)
    comp = comp.div(comp.sum(axis=1), axis=0)
    comp.columns = [f"habitat_{c}_frac" for c in comp.columns]
    comp = comp.reset_index()
    comp_path = DATA_DIR / "ct_habitat3d_composition.csv"
    comp.to_csv(comp_path, index=False)
    print(f"Saved -> {comp_path}")


if __name__ == "__main__":
    main()
