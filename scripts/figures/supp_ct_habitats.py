"""Supplementary Fig. S8: exploratory 3D intratumoral habitats.

Run: python scripts/figures/supp_ct_habitats.py
Output: outputs/figures/supplementary/_parts/supp_fig_s12_ct_habitats.png, .pdf
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import umap
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from scripts.figures import palette  # noqa: E402

palette.apply_style()  # Arial, compact type (the style of the desktop viewer)
from scripts.discovery.run_ct_habitat_supervoxels_3d import (  # noqa: E402
    full_volume_crop, target_n_segments_3d, LUNG_WINDOW,
)
from skimage.segmentation import slic  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures" / "supplementary" / "_parts"  # full render; render_supplementary_figures.py trims it
FIG_DIR.mkdir(parents=True, exist_ok=True)

HABITAT_COLORS = [palette.CATEGORICAL[0], palette.CATEGORICAL[1], palette.CATEGORICAL[2], palette.CATEGORICAL[3], palette.CATEGORICAL[4], "#7a5cc0"]
EXAMPLE_PATIENTS = [("R01-020", "Recurrence"), ("R01-093", "Recurrence"),
                     ("R01-067", "No recurrence"), ("R01-121", "No recurrence")]


def _save(fig, stem: str):
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    except PermissionError as exc:
        print(f"  (skipped PDF, locked: {exc})")
    print(f"Saved {png}")


def assign_habitat(mean_hu, std_hu, centers):
    z = np.stack([
        (mean_hu - centers["scaler_mean_hu"].iloc[0]) / centers["scaler_scale_hu"].iloc[0],
        (std_hu - centers["scaler_std_hu_mean"].iloc[0]) / centers["scaler_scale_std_hu"].iloc[0],
    ], axis=-1)
    centroids = centers[["mean_hu_z", "std_hu_z"]].to_numpy()
    d = np.linalg.norm(z[:, None, :] - centroids[None, :, :], axis=-1)
    return d.argmin(axis=1)


def panel_umap(ax, sv):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=0)
    emb = reducer.fit_transform(sv[["mean_hu", "std_hu"]].to_numpy())
    for h in sorted(sv["habitat"].unique()):
        m = sv["habitat"] == h
        ax.scatter(emb[m, 0], emb[m, 1], s=6, color=HABITAT_COLORS[h], alpha=0.65,
                   label=f"Habitat {h + 1} (n = {m.sum():,})", linewidths=0)
    ax.set_xlabel("UMAP 1", fontsize=8.5, color=palette.MUTED)
    ax.set_ylabel("UMAP 2", fontsize=8.5, color=palette.MUTED)
    ax.set_title("a  Pooled 3D supervoxels, all 144 patients", fontsize=9.5, fontweight="bold",
                 loc="left", color=palette.INK)
    legend = ax.legend(fontsize=6.6, frameon=True, facecolor="white", edgecolor=palette.GRID,
                        framealpha=0.92, loc="best", markerscale=1.8)
    legend.get_frame().set_linewidth(0.6)
    palette.style_axes(ax)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _rect(x, y, w, h, color):
    from matplotlib.patches import FancyBboxPatch
    return FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01",
                           facecolor=color, edgecolor="none")


def panel_characterization_table(ax, sv):
    char = sv.groupby("habitat")[["mean_hu", "std_hu"]].mean().sort_index()
    hu_med, std_med = char["mean_hu"].median(), char["std_hu"].median()
    ax.set_title("b  Habitat characterization", fontsize=9.5, fontweight="bold", loc="left", color=palette.INK)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    col_labels = ["Habitat", "Mean HU", "Local heterogeneity\n(s.d. HU)"]
    col_w = [0.20, 0.26, 0.30]
    x0 = 0.06
    row_h = 0.135
    y0 = 0.90
    for j, lbl in enumerate(col_labels):
        ax.text(x0 + sum(col_w[:j]) + col_w[j] / 2, y0 + row_h * 0.55, lbl, ha="center", va="center",
                fontsize=8, fontweight="bold", color=palette.INK)
    for i, (h, row) in enumerate(char.iterrows()):
        y = y0 - (i + 1) * row_h
        ax.add_patch(_rect(x0, y, col_w[0], row_h * 0.85, HABITAT_COLORS[h]))
        ax.text(x0 + col_w[0] / 2, y + row_h * 0.42, f"H{h + 1}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")
        hu_label = "High" if row["mean_hu"] > hu_med else "Low"
        std_label = "High" if row["std_hu"] > std_med else "Low"
        ax.text(x0 + col_w[0] + col_w[1] / 2, y + row_h * 0.42, f"{hu_label} ({row['mean_hu']:.0f})".replace("-", "−"),
                ha="center", va="center", fontsize=8, color=palette.INK2)
        ax.text(x0 + col_w[0] + col_w[1] + col_w[2] / 2, y + row_h * 0.42, f"{std_label} ({row['std_hu']:.0f})",
                ha="center", va="center", fontsize=8, color=palette.INK2)


def render_slices(case_id, centers):
    row = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    ct_uid = row.loc[row["case_id"] == case_id, "ct_series_uid"].iloc[0]
    hu_vol, mask_vol, spacing_zyx = full_volume_crop(case_id, ct_uid)

    n_vox = int(mask_vol.sum())
    n_segments = target_n_segments_3d(n_vox)
    windowed = np.clip(hu_vol, LUNG_WINDOW[0] - LUNG_WINDOW[1] / 2, LUNG_WINDOW[0] + LUNG_WINDOW[1] / 2)
    labels_3d = slic(windowed, n_segments=n_segments, mask=mask_vol, compactness=0.05,
                      sigma=0.6, channel_axis=None, start_label=1, spacing=spacing_zyx)

    habitat_3d = np.full(labels_3d.shape, -1, dtype=int)
    for lbl in np.unique(labels_3d):
        if lbl == 0:
            continue
        region_mask = labels_3d == lbl
        mean_hu = hu_vol[region_mask].mean()
        std_hu = hu_vol[region_mask].std()
        h = assign_habitat(np.array([mean_hu]), np.array([std_hu]), centers)[0]
        habitat_3d[region_mask] = h

    counts = mask_vol.reshape(mask_vol.shape[0], -1).sum(axis=1)
    z = int(np.argmax(counts))

    gray = np.clip((hu_vol[z] - (LUNG_WINDOW[0] - LUNG_WINDOW[1] / 2)) /
                    LUNG_WINDOW[1] * 255, 0, 255).astype(np.uint8)
    return gray, mask_vol[z], labels_3d[z], habitat_3d[z]


def main():
    sv = pd.read_csv(DATA_DIR / "ct_habitat3d_supervoxels.csv")
    centers = pd.read_csv(DATA_DIR / "ct_habitat3d_cluster_centers.csv")
    habitat_cmap = ListedColormap(HABITAT_COLORS)

    fig = Figure(figsize=(12.5, 14.0), dpi=100, layout="constrained")
    gs = fig.add_gridspec(6, 3, height_ratios=[1.3, 1.3, 1, 1, 1, 1], hspace=0.35, wspace=0.15)

    ax_umap = fig.add_subplot(gs[0, 0])
    ax_char = fig.add_subplot(gs[0, 1:])
    panel_umap(ax_umap, sv)
    panel_characterization_table(ax_char, sv)

    col_titles = ["CT", "Supervoxels (3D, this slice)", "Habitats (3D, this slice)"]
    for row_i, (case_id, group_label) in enumerate(EXAMPLE_PATIENTS):
        gray, mask_slice, sp_labels, habitat_map = render_slices(case_id, centers)
        masked_sp = np.ma.masked_where(sp_labels == 0, sp_labels)
        masked_hab = np.ma.masked_where(habitat_map < 0, habitat_map)

        axes_row = [fig.add_subplot(gs[row_i + 1, c]) for c in range(3)]
        axes_row[0].imshow(gray, cmap="gray", interpolation="bilinear")
        axes_row[0].contour(mask_slice, levels=[0.5], colors=["#eda100"], linewidths=1.6, zorder=5)

        axes_row[1].imshow(gray, cmap="gray", interpolation="bilinear")
        axes_row[1].imshow(masked_sp, cmap="nipy_spectral", alpha=0.65, interpolation="nearest")
        axes_row[1].contour(mask_slice, levels=[0.5], colors=["white"], linewidths=1.2, zorder=5)

        axes_row[2].imshow(gray, cmap="gray", interpolation="bilinear")
        axes_row[2].imshow(masked_hab, cmap=habitat_cmap, vmin=0, vmax=5, alpha=0.6, interpolation="nearest")
        axes_row[2].contour(mask_slice, levels=[0.5], colors=["white"], linewidths=1.2, zorder=5)

        for ax in axes_row:
            ax.axis("off")
        axes_row[0].text(-0.08, 0.5, f"{case_id}\n{group_label}", transform=axes_row[0].transAxes,
                          ha="right", va="center", fontsize=8.5, fontweight="bold", color=palette.INK, rotation=0)
        if row_i == 0:
            for ax, title in zip(axes_row, col_titles):
                ax.set_title(title, fontsize=9.5, fontweight="bold", color=palette.INK)

    fig.text(0.01, 1 - (1.3 / 6.6) - 0.015, "c", fontsize=13, fontweight="bold", color=palette.INK)
    fig.suptitle(
        "Supplementary Figure S12. CT-only intratumoral habitat mapping (3D supervoxels, no PET)",
        fontsize=12.5, fontweight="bold", x=0.02, y=1.015, ha="left",
    )
    _save(fig, "supp_fig_s12_ct_habitats")


if __name__ == "__main__":
    main()
