"""Supplementary Fig. S7: radiomic phenotypes of the discovery cohort.

An exploratory, descriptive analysis; whether the phenotypes are stable
under segmentation adjustment is tested separately (phenotypes.py).

All values are computed from data/processed/radiomics_features.csv,
data/processed/clinical.csv and the CT and mask files. The number of
clusters (k=4) was chosen by silhouette score over k=3..7 (k=4 and k=5
tied at about 0.20; k=4 gave more even cluster sizes), and the silhouette
value is shown in the figure.

Run: python scripts/figures/supp_phenotypes.py
Output: outputs/figures/supplementary/_parts/figure5_phenotype_clusters.png (copied to outputs/figures/supplementary/clusters.png)
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.stats import chi2_contingency
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import viz  # noqa: E402

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.6,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#c9960c", "#7a5cc0"]
INK, SUBINK, RULE = "#14161c", "#5b6470", "#e3e5e9"

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures" / "supplementary" / "_parts"  # full render; render_supplementary_figures.py trims it
FIG_DIR.mkdir(parents=True, exist_ok=True)

NON_FEATURE_COLS = {"case_id", "ct_series_uid", "seg_series_uid", "texture_n_slices"}

RADAR_FEATURES = [
    ("shape_max_diameter_mm", "max diameter"),
    ("shape_sphericity", "sphericity"),
    ("shape_flatness", "flatness"),
    ("firstorder_mean_hu", "mean HU"),
    ("firstorder_energy", "energy"),
    ("firstorder_entropy", "entropy"),
    ("texture_contrast", "GLCM contrast"),
    ("texture_homogeneity", "GLCM homogeneity"),
    ("texture_correlation", "GLCM correlation"),
]

K = 4
FEATURE_NAMES = {
    "shape_volume_mm3": "Volume", "shape_surface_area_mm2": "Surface area", "shape_sphericity": "Sphericity",
    "shape_elongation": "Elongation", "shape_flatness": "Flatness", "shape_roundness": "Roundness",
    "shape_equivalent_spherical_radius_mm": "Equiv. spherical radius", "shape_max_diameter_mm": "Max. 3D diameter",
    "firstorder_voxel_count": "Voxel count", "firstorder_mean_hu": "Mean HU", "firstorder_std_hu": "HU s.d.",
    "firstorder_min_hu": "Min. HU", "firstorder_max_hu": "Max. HU", "firstorder_median_hu": "Median HU",
    "firstorder_range_hu": "HU range", "firstorder_iqr_hu": "HU IQR", "firstorder_mad_hu": "HU MAD",
    "firstorder_skewness": "Skewness", "firstorder_kurtosis": "Kurtosis", "firstorder_energy": "Energy",
    "firstorder_entropy": "Entropy", "texture_contrast": "GLCM contrast", "texture_correlation": "GLCM correlation",
    "texture_energy": "GLCM energy", "texture_homogeneity": "GLCM homogeneity",
    "texture_dissimilarity": "GLCM dissimilarity",
}


CLUSTER_LABELS = {
    0: "large and homogeneous",
    1: "dense and uniform",
    2: "lower density, heterogeneous",
    3: "round, high contrast",
}


# clustering

def run_clustering():
    df = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = SimpleImputer(strategy="median").fit_transform(df[feat_cols].to_numpy())
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    sils = {}
    for k in range(3, 8):
        lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xz)
        sils[k] = silhouette_score(Xz, lbl)

    labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Xz)
    df["cluster"] = labels

    reps = {}
    for c in range(K):
        idx = np.where(labels == c)[0]
        centroid = Xz[idx].mean(axis=0)
        dists = np.linalg.norm(Xz[idx] - centroid, axis=1)
        reps[c] = df.iloc[idx[np.argmin(dists)]]["case_id"]

    clin = pd.read_csv(DATA_DIR / "clinical.csv")
    merged = df.merge(clin[["case_id", "recurrence"]], on="case_id", how="left")
    ct = pd.crosstab(merged["cluster"], merged["recurrence"])
    _, p_recur, _, _ = chi2_contingency(ct)

    return df, feat_cols, Xz, labels, reps, sils[K], p_recur, merged


# radar

def draw_radar(ax, feature_idx, feat_cols, Xz, cluster_labels, cluster_id, color, title):
    n = len(feature_idx)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    sub = Xz[cluster_labels == cluster_id][:, feature_idx]
    mean_z = np.clip(sub.mean(axis=0), -2.4, 2.4)
    shift = 3.0
    r = (mean_z + shift).tolist()
    r += r[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, shift + 2.6)
    ax.set_yticks([shift - 2, shift - 1, shift, shift + 1, shift + 2])
    ax.set_yticklabels(["−2", "−1", "0", "+1", "+2"], fontsize=5.6, color=SUBINK)
    ax.plot(angles[:-1] + angles[:1], [shift] * (n + 1), color="#c7cdd4", linewidth=1.0,
            linestyle=(0, (3, 2)), zorder=1)
    ax.plot(angles, r, color=color, linewidth=2.0, zorder=3)
    ax.fill(angles, r, color=color, alpha=0.22, zorder=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lab for _, lab in RADAR_FEATURES], fontsize=6.2)
    ax.tick_params(axis="x", pad=7)
    ax.set_title(title, fontsize=8.0, pad=8, color=color)
    ax.grid(color="#e3e5e9", linewidth=0.7)
    ax.spines["polar"].set_color("#c7cdd4")


# heatmap

def draw_heatmap(fig, x0, y0, w, h, feat_cols, Xz, labels):
    n_pat, n_feat = Xz.shape
    feat_disp = [FEATURE_NAMES.get(f, f) for f in feat_cols]

    col_linkage = linkage(Xz.T, method="ward")
    col_order = leaves_list(col_linkage)

    order = []
    for c in range(K):
        idx = np.where(labels == c)[0]
        centroid = Xz[idx].mean(axis=0)
        d = np.linalg.norm(Xz[idx] - centroid, axis=1)
        order.extend(idx[np.argsort(d)].tolist())
    order = np.array(order)

    dendro_h = 0.030
    cbar_w = 0.010
    ax_dendro = fig.add_axes([x0 + cbar_w + 0.006, y0 + h, w - cbar_w - 0.006, dendro_h])
    dendrogram(col_linkage, ax=ax_dendro, no_labels=True, color_threshold=0,
                     above_threshold_color="#9aa3ad")
    ax_dendro.axis("off")

    ax_cbar = fig.add_axes([x0, y0, cbar_w, h])
    cluster_strip = labels[order].reshape(-1, 1)
    from matplotlib.colors import ListedColormap
    ax_cbar.imshow(cluster_strip, aspect="auto", cmap=ListedColormap(CLUSTER_COLORS[:K]),
                    vmin=-0.5, vmax=K - 0.5, interpolation="nearest")
    ax_cbar.axis("off")

    ax_hm = fig.add_axes([x0 + cbar_w + 0.006, y0, w - cbar_w - 0.006, h])
    mat = Xz[np.ix_(order, col_order)]
    im = ax_hm.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax_hm.set_xticks(range(n_feat))
    ax_hm.set_xticklabels([feat_disp[i] for i in col_order], rotation=60, ha="right", rotation_mode="anchor", fontsize=5.6)
    ax_hm.set_yticks([])
    ax_hm.set_ylabel("")

    cax = fig.add_axes([x0 + w + 0.012, y0, 0.008, h * 0.6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("z-score", fontsize=6.6)
    cb.ax.tick_params(labelsize=5.8)

    return order


# hero-lite

def ct_thumbnail(case_id):
    feats = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    row = feats.loc[feats["case_id"] == case_id].iloc[0]
    ct_dir = ROOT / "data" / "interim" / "dicom" / row["ct_series_uid"]
    mask_path = DATA_DIR / "masks" / f"{case_id}_seg.nii.gz"

    ct = viz.load_ct_volume(ct_dir)
    mask = viz.load_mask_on_ct(mask_path, ct)
    ct_arr = sitk.GetArrayFromImage(ct)
    mask_arr = sitk.GetArrayFromImage(mask)

    counts = mask_arr.reshape(mask_arr.shape[0], -1).sum(axis=1)
    z = int(np.argmax(counts))
    ys, xs = np.where(mask_arr[z] > 0)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.7), 90)
    half = min(half, cy, cx, ct_arr.shape[1] - cy, ct_arr.shape[2] - cx)
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    gray = viz.window_slice(ct_arr[z], *viz.WINDOW_PRESETS["Lung"])
    crop = gray[y0:y1, x0:x1]
    mask_crop = mask_arr[z, y0:y1, x0:x1] > 0
    return crop, mask_crop


# pca

def draw_pca(fig, x0, y0, w, h, Xz, labels, merged_df):
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Xz)
    var = pca.explained_variance_ratio_ * 100

    ax1 = fig.add_axes([x0, y0, w * 0.46, h])
    for c in range(K):
        m = labels == c
        ax1.scatter(coords[m, 0], coords[m, 1], s=22, color=CLUSTER_COLORS[c], edgecolors="white",
                    linewidths=0.4, alpha=0.9, zorder=3, label=f"Phenotype {c + 1}")
    ax1.set_xlabel(f"PC1 ({var[0]:.0f}%)")
    ax1.set_ylabel(f"PC2 ({var[1]:.0f}%)")
    ax1.set_title("By radiomic phenotype", fontsize=8.2)
    ax1.legend(fontsize=5.8, frameon=False, loc="best", handletextpad=0.3, labelspacing=0.3)

    ax2 = fig.add_axes([x0 + w * 0.54, y0, w * 0.46, h])
    rec = merged_df["recurrence"].to_numpy()
    color_map = {"yes": "#c0392b", "no": "#2a78d6"}
    for val, c in color_map.items():
        m = rec == val
        ax2.scatter(coords[m, 0], coords[m, 1], s=22, color=c, edgecolors="white", linewidths=0.4,
                    alpha=0.85, zorder=3, label={"yes": "Recurrence", "no": "No recurrence"}[val])
    m_unk = ~np.isin(rec, list(color_map))
    if m_unk.any():
        ax2.scatter(coords[m_unk, 0], coords[m_unk, 1], s=16, color="#c7cdd4", edgecolors="none",
                    alpha=0.6, zorder=2, label="Not recorded")
    ax2.set_xlabel(f"PC1 ({var[0]:.0f}%)")
    ax2.set_yticklabels([])
    ax2.set_title("By recurrence status", fontsize=8.2)
    ax2.legend(fontsize=5.8, frameon=False, loc="best", handletextpad=0.3, labelspacing=0.3)

    return var


# main

def main():
    df, feat_cols, Xz, labels, reps, sil, p_recur, merged = run_clustering()
    feature_idx = [feat_cols.index(f) for f, _ in RADAR_FEATURES]

    fig = plt.figure(figsize=(16.0, 15.0))
    fig.patch.set_facecolor("white")

    fig.text(0.025, 0.978, "Fig. 5 | Unsupervised radiomics phenotype clusters",
              fontsize=16, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.025, 0.958,
             "Exploratory, descriptive analysis; robustness of these clusters to segmentation "
             "perturbation is tested separately (Fig. 6b-d). "
             f"k = {K} clusters chosen by silhouette score (k=4: {sil:.2f}, the best-tied value over k=3-7); "
             f"cluster identity vs. recurrence: χ² p = {p_recur:.2f} (n.s., not corrected for multiple comparisons).",
             fontsize=9.0, color=SUBINK, ha="left", va="top")

    # -- a-d: radar charts --------------------------------------------------
    radar_y0, radar_h = 0.700, 0.160
    radar_w = 0.205
    radar_x0s = [0.03 + i * (radar_w + 0.015) for i in range(K)]
    letters = "abcd"
    for c in range(K):
        ax = fig.add_axes([radar_x0s[c], radar_y0, radar_w, radar_h], projection="polar")
        draw_radar(ax, feature_idx, feat_cols, Xz, labels, c, CLUSTER_COLORS[c],
                   f"Phenotype {c + 1} (n = {(labels == c).sum()}): {CLUSTER_LABELS[c]}")
        fig.text(radar_x0s[c] - 0.012, radar_y0 + radar_h + 0.050, letters[c], fontsize=14,
                 fontweight="bold", color=INK, ha="left", va="bottom")

    # -- e: clustered heatmap -----------------------------------------------
    hm_y0, hm_h = 0.470, 0.175
    fig.text(0.03 - 0.012, hm_y0 + hm_h + 0.040, "e", fontsize=14, fontweight="bold", color=INK,
             ha="left", va="bottom")
    draw_heatmap(fig, 0.055, hm_y0, 0.86, hm_h, feat_cols, Xz, labels)

    # -- f: representative CT thumbnails ------------------------------------
    thumb_y0, thumb_h = 0.235, 0.175
    fig.text(0.03 - 0.012, thumb_y0 + thumb_h + 0.028, "f", fontsize=14, fontweight="bold", color=INK,
             ha="left", va="bottom")
    thumb_w = 0.205
    thumb_x0s = [0.03 + i * (thumb_w + 0.015) for i in range(K)]
    for c in range(K):
        crop, mask_crop = ct_thumbnail(reps[c])
        ax = fig.add_axes([thumb_x0s[c], thumb_y0, thumb_w, thumb_h])
        ax.imshow(crop, cmap="gray", interpolation="bilinear")
        ax.contour(mask_crop, levels=[0.5], colors=[CLUSTER_COLORS[c]], linewidths=2.2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(CLUSTER_COLORS[c])
            spine.set_linewidth(2.2)
        ax.set_title(f"{reps[c]} (phenotype {c + 1})", fontsize=7.6)

    # -- g: PCA embedding -----------------------------------------------------
    pca_y0, pca_h = 0.03, 0.155
    fig.text(0.03 - 0.012, pca_y0 + pca_h + 0.028, "g", fontsize=14, fontweight="bold", color=INK,
             ha="left", va="bottom")
    draw_pca(fig, 0.055, pca_y0, 0.86, pca_h, Xz, labels, merged)

    fig.add_artist(plt.Line2D([0.025, 0.975], [thumb_y0 - 0.022, thumb_y0 - 0.022], transform=fig.transFigure,
                               color=RULE, linewidth=1.0))

    out_png = FIG_DIR / "figure5_phenotype_clusters.png"
    out_pdf = FIG_DIR / "figure5_phenotype_clusters.pdf"
    fig.savefig(out_png, dpi=300, facecolor="white")
    try:
        fig.savefig(out_pdf, facecolor="white")
        print(f"Saved {out_pdf}")
    except PermissionError as exc:
        print(f"Skipped PDF (locked by another process): {exc}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
