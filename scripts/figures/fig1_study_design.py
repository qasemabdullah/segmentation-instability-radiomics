"""Fig. 1: study design and patient cohorts.

Paper:    Figs 1-7, Suppl. Fig. S5
Pipeline: python pipeline.py run main_figures   (stage 'figures')

Six stacked panels on a single page, read top to bottom:

  a  Representative patients: two real, de-identified cases (low- vs.
     high-risk), each with a real CT crop + tumor segmentation, the
     clinicopathologic facts, and the observed overall-survival outcome.
  b  Cohort and endpoints: data available per patient of the discovery
     collection, the external cohorts, and events per endpoint.
  c  Image processing and feature extraction: CT and expert segmentation,
     the 26-feature panel, and a real CT-only 3D-supervoxel habitat map
     (the renderer of Supplementary Fig. S8).
  d  Feature sets and nested cross-validation (drawn as the actual
     outer/inner fold structure).
  e  Analyses: classification AUCs, phenotypes, overall survival by median
     out-of-fold risk (with live log-rank P) and gene-set enrichment.
  f  Segmentation variation and external cohort: controlled adjustment,
     sensitivity curve, radiation oncologists, AI segmentation and Lung1.

Every image, chart and number is computed live from this repository's own
saved pipeline output; nothing is hard-coded except the case IDs chosen
for display and the patients' clinicopathologic facts (Supplementary Fig. S13).

Layout is done in figure-fraction coordinates on one full-page overlay
axes (schematic elements), with real matplotlib axes placed on top for
the CT images and charts, so those keep proper aspect ratios and ticks.

Run: python scripts/figures/fig1_study_design.py
Output: outputs/figures/main/figure1_study_workflow.png, .pdf
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "outputs" / "figures" / "main"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

# palette -- matches scripts/figures/palette.py CATEGORICAL so Fig. 1 reads as one system
# with the rest of the paper's figures
INK, SUBINK, MUTED = "#15181e", "#4a5361", "#8a929e"
RULE, PANEL_BG, WHITE = "#e3e7ed", "#f4f6f9", "#ffffff"
BLUE, ORANGE, GREEN, AMBER, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
PURPLE, TEAL, NAVY = "#7a5cc0", "#0e9aa0", "#10306b"
BLUE_LT, ORANGE_LT = "#e6f0fc", "#fdece4"
ARROW = "#6b7482"

FIG_W, FIG_H0 = 9.6, 12.6
# The overlay axes runs y in [-EXTRA, TOP], one unit = FIG_H0 inches, so the
# page can grow downward without rescaling every hand-placed offset.
EXTRA = 0.346
TOP = 0.962
FIG_H = FIG_H0 * (TOP + EXTRA)
AY = FIG_W / FIG_H0  # multiply a horizontal length by AY to get the same physical vertical length


def fax(fig, rect, **kw):
    """fig.add_axes, but with the rect given in overlay (x, y) units."""
    x, y, w, h = rect
    return fig.add_axes([x, (y + EXTRA) / (TOP + EXTRA), w, h / (TOP + EXTRA)], **kw)


# primitives (all in figure-fraction coordinates on the overlay axes)

def rbox(ax, x, y, w, h, fc=WHITE, ec="none", lw=0.0, r=0.006, z=2, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, **kw))


def dot(ax, cx, cy, r, z=5, **kw):
    ax.add_patch(Ellipse((cx, cy), 2 * r, 2 * r * AY, zorder=z, **kw))


def arrow(ax, p0, p1, color=ARROW, lw=1.5, ms=11, z=4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                                 color=color, zorder=z, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", capstyle="round"))


def panel(ax, x, y, w, h, letter, title):
    rbox(ax, x, y, w, h, fc=PANEL_BG, r=0.010, z=0)
    lx, ly = x + 0.012, y + h - 0.012
    rbox(ax, lx, ly - 0.020, 0.020, 0.020, fc=NAVY, r=0.004, z=3)
    ax.text(lx + 0.010, ly - 0.010, letter, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=WHITE, zorder=4)
    ax.text(lx + 0.030, ly - 0.010, title, ha="left", va="center", fontsize=11,
            fontweight="bold", color=INK, zorder=4)


def chip(ax, x, y, w, h, title, sub, color, fs=8.6, sub_fs=7.6):
    """White card with a colored left accent bar: a title line + a sub line."""
    rbox(ax, x, y, w, h, fc=WHITE, ec=RULE, lw=0.6, r=0.005, z=2)
    ax.add_patch(Rectangle((x, y + 0.004), 0.0045, h - 0.008, facecolor=color, edgecolor="none", zorder=3))
    ax.text(x + 0.013, y + h * 0.63, title, ha="left", va="center", fontsize=fs, fontweight="bold",
            color=INK, zorder=4)
    ax.text(x + 0.013, y + h * 0.29, sub, ha="left", va="center", fontsize=sub_fs, color=SUBINK, zorder=4)


# icon glyphs: flat white pictograms on a colored disc. `s` is the glyph
# half-size in figure-x units; every vertical offset is scaled by AY.

def badge(ax, cx, cy, r, color):
    dot(ax, cx, cy - 0.0012, r, facecolor="#0f172a", alpha=0.12, z=4)
    dot(ax, cx, cy, r, facecolor=color, edgecolor=WHITE, linewidth=1.2, z=5)


def icon_person(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    dot(ax, cx, cy + 0.34 * s * AY, 0.24 * s, facecolor=WHITE, z=6)
    ax.add_patch(Ellipse((cx, cy - 0.36 * s * AY), 0.80 * s, 0.62 * s * AY, facecolor=WHITE, zorder=6))


def icon_people(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    for dx, sc, z in [(-0.40, 0.8, 6), (0.40, 0.8, 6), (0.0, 1.0, 7)]:
        dot(ax, cx + dx * s, cy + 0.28 * s * AY * sc, 0.17 * s * sc, facecolor=WHITE,
            edgecolor=color, linewidth=0.6, z=z)
        ax.add_patch(Ellipse((cx + dx * s, cy - 0.30 * s * AY), 0.52 * s * sc, 0.50 * s * AY * sc,
                             facecolor=WHITE, edgecolor=color, linewidth=0.6, zorder=z))


def icon_target(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    for rr in (0.72, 0.46):
        dot(ax, cx, cy, rr * s, facecolor="none", edgecolor=WHITE, linewidth=1.3, z=6)
    dot(ax, cx, cy, 0.18 * s, facecolor=WHITE, z=6)


def icon_dna(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    t = np.linspace(-0.7, 0.7, 50)
    for sign in (1, -1):
        ax.plot(cx + sign * 0.36 * s * np.sin(t * 4.5), cy + t * s * AY, color=WHITE, lw=1.3,
                zorder=6, solid_capstyle="round")
    for tt in np.linspace(-0.55, 0.55, 5):
        xa = 0.36 * s * np.sin(tt * 4.5)
        ax.plot([cx - xa, cx + xa], [cy + tt * s * AY] * 2, color=WHITE, lw=0.8, zorder=6)


def icon_grid(ax, cx, cy, s, color):
    """Image-embedding glyph: a 3x3 grid of feature-map tiles."""
    badge(ax, cx, cy, s * 1.5, color)
    step = 0.40 * s
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            a = 1.0 if (i + j) % 2 == 0 else 0.55
            ax.add_patch(Rectangle((cx + i * step - 0.15 * s, cy + j * step * AY - 0.15 * s * AY),
                                   0.30 * s, 0.30 * s * AY, facecolor=WHITE, alpha=a,
                                   edgecolor="none", zorder=6))


def icon_survival(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    xs = np.array([-0.62, -0.25, -0.25, 0.10, 0.10, 0.62])
    ys = np.array([0.50, 0.50, 0.08, 0.08, -0.42, -0.42])
    ax.plot(cx + xs * s, cy + ys * s * AY, color=WHITE, lw=1.6, zorder=6,
            solid_capstyle="round", solid_joinstyle="round")


def icon_cluster(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    rng = np.random.RandomState(3)
    for gx, gy in [(-0.34, 0.26), (0.34, 0.22), (0.0, -0.36)]:
        for px, py in rng.normal(0, 0.10, size=(4, 2)):
            dot(ax, cx + (gx + px) * s, cy + (gy + py) * s * AY, 0.075 * s, facecolor=WHITE, z=6)


def icon_gsea(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    for x, h in zip(np.linspace(-0.48, 0.48, 6), [0.55, 0.38, 0.20, -0.18, -0.34, -0.52]):
        ax.add_patch(Rectangle((cx + x * s - 0.06 * s, cy), 0.12 * s, h * s * AY,
                               facecolor=WHITE, alpha=1.0 if h > 0 else 0.6, edgecolor="none", zorder=6))
    ax.plot([cx - 0.62 * s, cx + 0.62 * s], [cy, cy], color=WHITE, lw=0.8, zorder=7)


def icon_boundary(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    t = np.linspace(0, 2 * np.pi, 80)
    for rr, ls, lw, a in [(0.70, (0, (1.5, 1.5)), 0.9, 0.8), (0.48, "-", 1.5, 1.0), (0.26, (0, (1.5, 1.5)), 0.9, 0.8)]:
        rad = rr + 0.05 * np.sin(4 * t)
        ax.plot(cx + rad * s * np.cos(t), cy + rad * s * AY * np.sin(t), color=WHITE, lw=lw,
                ls=ls, alpha=a, zorder=6)


def icon_check(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    ax.plot(cx + np.array([-0.45, -0.10, 0.50]) * s, cy + np.array([0.0, -0.35, 0.40]) * s * AY,
            color=WHITE, lw=1.8, zorder=6, solid_capstyle="round", solid_joinstyle="round")


# real data -- computed live from this repository's saved pipeline output

def load_real_ct_crop(case_id, pad_px=8):
    from nsclc_pipeline import config, viz
    import pandas as pd
    feats = pd.read_csv(config.PROCESSED_DIR / "radiomics_features.csv")
    row = feats.loc[feats["case_id"] == case_id].iloc[0]
    ct_dir = ROOT / "data" / "interim" / "dicom" / row["ct_series_uid"]
    mask_path = config.PROCESSED_DIR / "masks" / f"{case_id}_seg.nii.gz"
    ct = viz.load_ct_volume(ct_dir)
    mask = viz.load_mask_on_ct(mask_path, ct)
    ct_arr = sitk.GetArrayFromImage(ct)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    z = int(np.argmax(mask_arr.reshape(mask_arr.shape[0], -1).sum(axis=1)))
    ys, xs = np.where(mask_arr[z])
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.6) + pad_px
    y0, y1 = max(cy - half, 0), min(cy + half, ct_arr.shape[1])
    x0, x1 = max(cx - half, 0), min(cx + half, ct_arr.shape[2])
    gray = viz.window_slice(ct_arr[z], *viz.WINDOW_PRESETS["Lung"])
    return gray[y0:y1, x0:x1], mask_arr[z, y0:y1, x0:x1]


def compute_real_shap_ranking(target="recurrence", feature_set="radiomics", top_n=5):
    from nsclc_pipeline import interpretation
    import pandas as pd
    sv, _case_ids = interpretation.compute_shap(target, feature_set)
    rank = pd.DataFrame({"feature": sv.feature_names, "v": np.abs(sv.values).mean(axis=0)}) \
        .sort_values("v", ascending=False).head(top_n)
    nice = {
        "shape_flatness": "Flatness", "texture_contrast": "GLCM contrast",
        "firstorder_mean_hu": "Mean attenuation", "shape_sphericity": "Sphericity",
        "shape_roundness": "Roundness", "texture_dissimilarity": "GLCM dissimilarity",
    }
    rank["label"] = rank["feature"].map(lambda f: nice.get(f, f.replace("_", " ").capitalize()))
    return rank


def compute_real_km(pred_csv="survival_predictions.csv"):
    from nsclc_pipeline import config
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    import pandas as pd
    oof = pd.read_csv(config.PROCESSED_DIR / pred_csv)
    oof["months"] = oof["duration_days"] / 30.4375
    med = oof["risk_score"].median()
    groups = {"high": oof[oof["risk_score"] >= med], "low": oof[oof["risk_score"] < med]}
    curves = {}
    for label, grp in groups.items():
        kmf = KaplanMeierFitter().fit(grp["months"], grp["event"])
        curves[label] = (kmf.survival_function_.index.to_numpy(), kmf.survival_function_.iloc[:, 0].to_numpy(),
                         len(grp))
    p = logrank_test(groups["high"]["months"], groups["low"]["months"],
                     groups["high"]["event"], groups["low"]["event"]).p_value
    return curves, p


# real-data sub-axes

def img_axes(fig, cx, cy, w):
    """A square axes of width `w` (figure fraction) centered on (cx, cy)."""
    h = w * AY
    a = fax(fig, [cx - w / 2, cy - h / 2, w, h])
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_color(INK)
        sp.set_linewidth(0.8)
    return a


def draw_ct(a, gray, mask, color):
    a.imshow(gray, cmap="gray", interpolation="bilinear")
    a.contour(mask, levels=[0.5], colors=[color], linewidths=1.6)


def draw_km(a, curves, p):
    for label, color, name in [("low", GREEN, "Low risk"), ("high", ORANGE, "High risk")]:
        t, s, n = curves[label]
        a.step(t, s, where="post", color=color, lw=1.8, label=f"{name} (n = {n})")
    a.set_ylim(0, 1.03)
    a.set_xlim(0, None)
    a.set_yticks([0, 0.5, 1.0])
    a.set_xlabel("Months from CT", fontsize=7.6, color=SUBINK, labelpad=2)
    a.set_ylabel("Overall survival", fontsize=7.6, color=SUBINK, labelpad=2)
    a.tick_params(labelsize=7, colors=SUBINK)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        a.spines[sp].set_color(MUTED)
    a.legend(fontsize=6.9, frameon=False, loc="lower left", handlelength=1.3, borderaxespad=0.2)
    p_txt = "P < 0.001" if p < 0.001 else f"P = {p:.3f}"
    a.text(0.97, 0.97, f"log-rank {p_txt}", transform=a.transAxes, ha="right", va="top",
           fontsize=7.2, color=INK)


# panels

def patient_card(fig, ax, x, y, w, h, color, light, title, case_id, ct, facts, outcome):
    rbox(ax, x, y, w, h, fc=WHITE, ec=RULE, lw=0.7, r=0.008, z=1)
    hh = 0.030
    rbox(ax, x, y + h - hh, w, hh, fc=light, r=0.008, z=2)
    ax.add_patch(Rectangle((x + 0.001, y + h - hh), w - 0.002, 0.008, facecolor=light, edgecolor="none", zorder=2))
    icon_person(ax, x + 0.022, y + h - hh / 2, 0.0085, color)
    ax.text(x + 0.042, y + h - hh / 2, title, ha="left", va="center", fontsize=10, fontweight="bold",
            color=color, zorder=4)
    ax.text(x + w - 0.012, y + h - hh / 2, case_id, ha="right", va="center", fontsize=8.6,
            color=SUBINK, zorder=4, family="monospace")

    # real CT crop, left
    body_top, body_bot = y + h - hh - 0.008, y + 0.010
    img_w = (body_top - body_bot) / AY
    a = img_axes(fig, x + 0.010 + img_w / 2, (body_top + body_bot) / 2, img_w)
    draw_ct(a, *ct, color)

    # fact table, right
    tx0 = x + 0.022 + img_w
    tx1 = x + w - 0.012
    row_h = 0.0145
    ty = body_top - 0.004
    for label, value in facts:
        ax.text(tx0, ty - row_h / 2, label, ha="left", va="center", fontsize=7.8, color=MUTED, zorder=4)
        ax.text(tx1, ty - row_h / 2, value, ha="right", va="center", fontsize=8.3, fontweight="bold",
                color=INK, zorder=4)
        ty -= row_h
        ax.plot([tx0, tx1], [ty, ty], color=RULE, lw=0.6, zorder=3)

    # outcome pill
    ph = 0.024
    rbox(ax, tx0, body_bot, tx1 - tx0, ph, fc=color, r=0.006, z=3)
    ax.text((tx0 + tx1) / 2, body_bot + ph / 2, outcome, ha="center", va="center", fontsize=8.6,
            fontweight="bold", color=WHITE, zorder=4)


def nested_cv_diagram(ax, x, y, w):
    """The actual fold structure: outer 5-fold split (one test fold), and
    the training folds re-split 5-fold for the inner grid search."""
    bw, gap = w / 5, 0.0025
    bh = 0.020
    ax.text(x, y + bh + 0.008, "Outer 5-fold CV (performance)", ha="left", va="bottom",
            fontsize=7.8, fontweight="bold", color=INK)
    for i in range(5):
        test = i == 4
        rbox(ax, x + i * bw + gap / 2, y, bw - gap, bh, fc=NAVY if test else "#b9c7de", r=0.003, z=3)
        ax.text(x + i * bw + bw / 2, y + bh / 2, "test" if test else "train", ha="center", va="center",
                fontsize=6.6, color=WHITE if test else NAVY, zorder=4)
    iy = y - 0.046
    iw = 4 * bw
    ax.plot([x + 0.002, x + 0.002], [y - 0.002, iy + 0.018], color=MUTED, lw=0.8, zorder=2)
    ax.plot([x + iw - 0.002, x + iw - 0.002], [y - 0.002, iy + 0.018], color=MUTED, lw=0.8, zorder=2)
    ibw = iw / 5
    for i in range(5):
        val = i == 4
        rbox(ax, x + i * ibw + gap / 2, iy, ibw - gap, 0.016, fc=BLUE if val else "#d7e5f8", r=0.003, z=3)
    ax.text(x + iw + 0.006, iy + 0.008, "val", ha="left", va="center", fontsize=6.6, color=BLUE)
    ax.text(x, iy - 0.006, "Inner 5-fold CV\n(hyperparameter tuning)", ha="left", va="top",
            fontsize=7.4, color=SUBINK, linespacing=1.2)


# imaging-pipeline panel (b): one real patient, segmentation -> habitats
# -> features, in the numbered-block style of a graphical abstract

PIPE_CASE = "R01-020"
SEG_RED = "#e5484d"


def load_pipeline_case(case_id=PIPE_CASE):
    """Everything panel b needs for one patient: full-field CT slices for
    the segmentation block, the 3D supervoxel habitat volume (same
    algorithm and cluster centers as Supplementary Fig. S12), and the
    pipeline's own GLCM on the tumor's largest slice."""
    import pandas as pd
    from scipy import ndimage
    from skimage.feature import graycomatrix
    from skimage.segmentation import slic
    from nsclc_pipeline import config, viz
    from nsclc_pipeline import radiomics as rad
    from scripts.figures.supp_ct_habitats import HABITAT_COLORS, assign_habitat
    from scripts.discovery.run_ct_habitat_supervoxels_3d import LUNG_WINDOW, full_volume_crop, target_n_segments_3d

    feats = pd.read_csv(config.PROCESSED_DIR / "radiomics_features.csv")
    ct_uid = feats.loc[feats["case_id"] == case_id, "ct_series_uid"].iloc[0]

    # full-field slices (segmentation block)
    ct = viz.load_ct_volume(ROOT / "data" / "interim" / "dicom" / ct_uid)
    mask = viz.load_mask_on_ct(config.PROCESSED_DIR / "masks" / f"{case_id}_seg.nii.gz", ct)
    ct_arr = sitk.GetArrayFromImage(ct)
    m_arr = sitk.GetArrayFromImage(mask) > 0
    z = int(np.argmax(m_arr.reshape(m_arr.shape[0], -1).sum(axis=1)))
    body = ndimage.binary_fill_holes(ct_arr[z] > -400)
    lab, n = ndimage.label(body)
    if n > 1:  # keep the largest component (drops the scanner table)
        body = lab == (np.argmax(ndimage.sum(body, lab, range(1, n + 1))) + 1)
    ys, xs = np.where(body)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + 6
    sl = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
    nz = ct_arr.shape[0] - 1
    stack = [viz.window_slice(ct_arr[zz], *viz.WINDOW_PRESETS["Lung"])[sl]
             for zz in (min(z + 16, nz), min(z + 8, nz), z)]
    seg_mask = m_arr[z][sl]

    # 3D habitats
    hu_vol, mask_vol, spacing = full_volume_crop(case_id, ct_uid)
    windowed = np.clip(hu_vol, LUNG_WINDOW[0] - LUNG_WINDOW[1] / 2, LUNG_WINDOW[0] + LUNG_WINDOW[1] / 2)
    labels = slic(windowed, n_segments=target_n_segments_3d(int(mask_vol.sum())), mask=mask_vol,
                  compactness=0.05, sigma=0.6, channel_axis=None, start_label=1, spacing=spacing)
    centers = pd.read_csv(config.PROCESSED_DIR / "ct_habitat3d_cluster_centers.csv")
    hab = np.full(labels.shape, -1, dtype=int)
    for lbl in np.unique(labels[labels > 0]):
        rm = labels == lbl
        hab[rm] = assign_habitat(np.array([hu_vol[rm].mean()]), np.array([hu_vol[rm].std()]), centers)[0]
    zc = int(np.argmax(mask_vol.reshape(mask_vol.shape[0], -1).sum(axis=1)))
    hab_gray = np.clip((hu_vol[zc] - (LUNG_WINDOW[0] - LUNG_WINDOW[1] / 2)) / LUNG_WINDOW[1] * 255, 0, 255)

    # habitat descriptors, pooled over all 144 patients' supervoxels (as S12 panel b)
    sv = pd.read_csv(config.PROCESSED_DIR / "ct_habitat3d_supervoxels.csv")
    char = sv.groupby("habitat")[["mean_hu", "std_hu"]].mean()
    hu_med, sd_med = char["mean_hu"].median(), char["std_hu"].median()
    present = sorted(np.unique(hab[hab >= 0]).tolist())
    frac = {h: float((hab == h).sum()) / float((hab >= 0).sum()) for h in present}
    desc = {h: ("dense" if char.loc[h, "mean_hu"] > hu_med else "low-density") + ", " +
               ("heterogeneous" if char.loc[h, "std_hu"] > sd_med else "uniform") for h in present}

    # GLCM exactly as radiomics.py computes it for one slice
    tv = hu_vol[mask_vol > 0]
    vmin, vmax = float(tv.min()), float(tv.max())
    m2 = mask_vol[zc] > 0
    yy, xx = np.where(m2)
    roi = hu_vol[zc, yy.min():yy.max() + 1, xx.min():xx.max() + 1].astype(np.float64)
    rmask = m2[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    q = np.clip((roi - vmin) / (vmax - vmin) * (rad.GLCM_LEVELS - 1), 0, rad.GLCM_LEVELS - 1).astype(np.uint8)
    q[~rmask] = 0
    glcm = graycomatrix(q, distances=rad.GLCM_DISTANCES, angles=rad.GLCM_ANGLES, levels=rad.GLCM_LEVELS,
                        symmetric=True, normed=True)[:, :, 0, :].mean(axis=-1)

    return dict(stack=stack, seg_mask=seg_mask, hab_gray=hab_gray, hab_slice=hab[zc], hab_mask=mask_vol[zc],
                hab_vol=hab, mask_vol=mask_vol, spacing=spacing, colors=HABITAT_COLORS, present=present,
                frac=frac, desc=desc, glcm=glcm)


def draw_mesh(a3, mask_vol, spacing, color):
    """Real 3D tumor surface: marching cubes on the expert SEG mask -- the
    same whole-tumor region the 26 radiomics features are computed from."""
    from matplotlib.colors import LightSource
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy import ndimage
    from skimage.measure import marching_cubes

    vol = np.pad(ndimage.gaussian_filter(mask_vol.astype(float), 0.8), 2)
    step = 1 if mask_vol.sum() < 60000 else 2
    verts, faces, _, _ = marching_cubes(vol, 0.5, spacing=spacing, step_size=step)
    # (z, y, x) voxel order -> plot as (x, -y, z) so the patient's head is up
    tri = verts[faces][:, :, [2, 1, 0]]
    tri[:, :, 1] *= -1
    pc = Poly3DCollection(tri, facecolors=color, edgecolor="none", shade=True,
                          lightsource=LightSource(azdeg=315, altdeg=35))
    a3.add_collection3d(pc)
    lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
    a3.set_xlim(lo[0], hi[0])
    a3.set_ylim(lo[1], hi[1])
    a3.set_zlim(lo[2], hi[2])
    a3.set_box_aspect(hi - lo)
    a3.view_init(elev=18, azim=-58)
    a3.set_axis_off()
    a3.set_facecolor("none")


def pipe_block(ax, x, y, w, h, roman, title, color, light, ls=(0, (6, 2, 1.2, 2))):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006", facecolor=WHITE,
                                edgecolor=color, linewidth=1.3, linestyle=ls, zorder=1))
    label = f"{roman}. {title}" if roman else title
    tw = 0.020 + 0.0068 * len(label)
    rbox(ax, x + (w - tw) / 2, y + h - 0.013, tw, 0.026, fc=light, ec=color, lw=0.8, r=0.004, z=3)
    ax.text(x + w / 2, y + h, label, ha="center", va="center", fontsize=9.4, fontweight="bold",
            color=INK, zorder=4)


def pipeline_panel(fig, ax, x0, y, w, h, P):
    """Main path I -> II (mask -> 26 radiomics features, the model input);
    habitats hang off I as a dashed side branch, because they are a
    descriptive analysis that feeds clustering only, never a model."""
    top, bot = y + h - 0.052, y + 0.036
    widths = [0.280, 0.347, 0.285]
    gap = (w - 0.024 - sum(widths)) / 2
    xs = [x0 + 0.012]
    for bw in widths[:-1]:
        xs.append(xs[-1] + bw + gap)
    pipe_block(ax, xs[0], bot, widths[0], top - bot, "I", "Tumor segmentation", ORANGE, ORANGE_LT)
    pipe_block(ax, xs[1], bot, widths[1], top - bot, "II", "Feature extraction", GREEN, "#e0f5ec")
    pipe_block(ax, xs[2], bot, widths[2], top - bot, "", "Intratumoral habitats", AMBER, "#fdf3d6",
               ls=(0, (2, 2)))
    mid = (top + bot) / 2 - 0.004
    icy = mid + 0.006
    xa = xs[0] + widths[0]
    arrow(ax, (xa - 0.004, icy), (xa + gap + 0.006, icy), color=INK, lw=1.6, ms=11, z=6)

    # dashed side branch: from I, under II, up into the habitat block
    by_ = y + 0.017
    x_from, x_to = xs[0] + widths[0] * 0.5, xs[2] + widths[2] * 0.5
    ax.plot([x_from, x_from, x_to], [bot - 0.001, by_, by_], color=AMBER, lw=1.2, ls=(0, (3, 2)), zorder=2)
    ax.add_patch(FancyArrowPatch((x_to, by_), (x_to, bot - 0.001), arrowstyle="-|>", mutation_scale=9,
                                 color=AMBER, lw=1.2, linestyle=(0, (3, 2)), zorder=2, shrinkA=0, shrinkB=0))
    ax.text((xs[1] + xs[1] + widths[1]) / 2, by_,
            "  habitat analysis (descriptive; not used for prediction)  ",
            ha="center", va="center", fontsize=6.9, color="#9a6b00", style="italic",
            backgroundcolor=PANEL_BG, zorder=3)

    # ---- I. segmentation: stacked CT slices -> expert mask overlaid
    bx = xs[0]
    ih = 0.080
    iw = ih / AY
    off = 0.0075
    sx = bx + 0.014
    for k, g in enumerate(P["stack"]):
        a = img_axes(fig, sx + iw / 2 + (2 - k) * off, icy + (2 - k) * off * AY, iw)
        a.imshow(g, cmap="gray", interpolation="bilinear")
    ax.text(sx + iw / 2 + off, icy - ih / 2 - 0.006, "CT volume", ha="center", va="top", fontsize=7.6,
            color=SUBINK)
    ox = bx + widths[0] - 0.014 - iw / 2
    arrow(ax, (sx + iw + 2 * off + 0.005, icy), (ox - iw / 2 - 0.005, icy), color=ORANGE, lw=1.6, ms=10, z=6)
    a = img_axes(fig, ox, icy, iw)
    a.imshow(P["stack"][-1], cmap="gray", interpolation="bilinear")
    rgba = np.zeros(P["seg_mask"].shape + (4,))
    rgba[P["seg_mask"]] = matplotlib.colors.to_rgba(SEG_RED, 0.6)
    a.imshow(rgba, interpolation="nearest")
    a.contour(P["seg_mask"], levels=[0.5], colors=[SEG_RED], linewidths=0.9)
    ax.text(ox, icy - ih / 2 - 0.006, "Expert segmentation", ha="center", va="top", fontsize=7.6, color=SUBINK)

    # ---- II. features: 3D surface of the same mask, GLCM, feature families
    bx = xs[1]
    mw = 0.134
    a3 = fax(fig, [bx + 0.002, icy - mw * AY / 2, mw, mw * AY], projection="3d")
    draw_mesh(a3, P["mask_vol"], P["spacing"], SEG_RED)
    ax.text(bx + 0.002 + mw / 2, icy - ih / 2 - 0.006, "3D tumor surface", ha="center", va="top",
            fontsize=7.6, color=SUBINK)
    gw = 0.060
    gcx = bx + 0.002 + mw + 0.006 + gw / 2
    ga = img_axes(fig, gcx, icy + 0.006, gw)
    g = P["glcm"].copy()
    g[0, :] = np.nan  # level-0 row/col = bounding-box background (radiomics.py); hidden so the
    g[:, 0] = np.nan  # tumor's own gray-level co-occurrence is what's visible
    ga.imshow(np.ma.masked_invalid(np.log10(g + 1e-5)), cmap="Greens", interpolation="nearest")
    for sp in ga.spines.values():
        sp.set_color(GREEN)
    ax.text(gcx, icy + 0.006 - gw * AY / 2 - 0.005, "GLCM\n(32 gray levels)", ha="center", va="top", fontsize=7.2,
            color=SUBINK, linespacing=1.15)
    fx = gcx + gw / 2 + 0.012
    fw = bx + widths[1] - 0.010 - fx
    ax.text(fx + fw / 2, icy + 0.040, "26 features", ha="center", va="center", fontsize=8.8, fontweight="bold",
            color=INK, zorder=4)
    for i, (n, lab) in enumerate([("8", "shape"), ("13", "first-order"), ("5", "GLCM texture")]):
        cy_ = icy + 0.016 - i * 0.024
        rbox(ax, fx, cy_ - 0.010, fw, 0.020, fc="#e0f5ec", r=0.004, z=3)
        ax.text(fx + 0.007, cy_, n, ha="left", va="center", fontsize=8.4, fontweight="bold", color=GREEN, zorder=4)
        ax.text(fx + 0.024, cy_, lab, ha="left", va="center", fontsize=7.6, color=INK, zorder=4)

    # ---- side analysis: supervoxel habitat map + legend
    bx = xs[2]
    hw = 0.118
    hcx = bx + 0.014 + hw / 2
    a = img_axes(fig, hcx, icy, hw)
    a.imshow(P["hab_gray"], cmap="gray", interpolation="bilinear")
    a.imshow(np.ma.masked_where(P["hab_slice"] < 0, P["hab_slice"]), cmap=ListedColormap(P["colors"]), vmin=0,
             vmax=len(P["colors"]) - 1, alpha=0.72, interpolation="nearest")
    a.contour(P["hab_mask"], levels=[0.5], colors=[WHITE], linewidths=0.9)
    ax.text(hcx, icy - hw * AY / 2 - 0.006, "3D supervoxel habitats", ha="center", va="top",
            fontsize=7.6, color=SUBINK)
    lx = bx + 0.014 + hw + 0.014
    pres = sorted(P["present"], key=lambda hh: -P["frac"][hh])[:4]
    rh = 0.027
    ly = icy + len(pres) * rh / 2
    for hh in pres:
        ly -= rh
        rbox(ax, lx, ly + 0.004, 0.011, rh - 0.008, fc=P["colors"][hh], r=0.002, z=3)
        ax.text(lx + 0.017, ly + rh * 0.66, f"Habitat {hh + 1}  ({P['frac'][hh] * 100:.0f}%)", ha="left",
                va="center", fontsize=7.8, fontweight="bold", color=INK, zorder=4)
        ax.text(lx + 0.017, ly + rh * 0.26, P["desc"][hh], ha="left", va="center", fontsize=7.0, color=SUBINK,
                zorder=4)


# model comparison heatmap (panel d): nested-CV AUC, target x feature set

AUC_SETS = [("", "Radiomics"), ("__radiomics_clinical", "+Clin."),
            ("__radiomics_clinical_rnaseq", "+Clin.+RNA"), ("__embeddings", "CNN only"),
            ("__radiomics_embeddings", "Rad. + CNN")]
AUC_TARGETS = [("recurrence", "Recurrence"), ("egfr", "EGFR"), ("kras", "KRAS")]


def compute_auc_grid():
    """Mean outer-fold AUC -- the same statistic the paper reports
    (Results, Fig. 3) -- from the saved out-of-fold predictions."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from nsclc_pipeline import config
    grid = np.full((len(AUC_TARGETS), len(AUC_SETS)), np.nan)
    for i, (t, _) in enumerate(AUC_TARGETS):
        for j, (suffix, _) in enumerate(AUC_SETS):
            f = config.PROCESSED_DIR / f"modeling_predictions_{t}{suffix}.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            grid[i, j] = np.mean([roc_auc_score(g["y_true"], g["y_proba"]) for _, g in d.groupby("fold")])
    return grid


def draw_auc_heatmap(a, grid):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("auc", [ORANGE, "#fbe0d2", "#f7f8fa", "#cfe0f7", BLUE])
    cmap.set_bad("#e9ecf1")
    a.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0.3, vmax=0.7, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            txt, col = ("–", MUTED) if np.isnan(v) else (f"{v:.2f}", WHITE if abs(v - 0.5) > 0.14 else INK)
            a.text(j, i, txt, ha="center", va="center", fontsize=7.0, color=col,
                   fontweight="bold" if not np.isnan(v) and v >= 0.6 else "normal")
    a.set_xticks(range(len(AUC_SETS)), [lab for _, lab in AUC_SETS], rotation=45, ha="right", fontsize=6.6,
                 rotation_mode="anchor")
    a.set_yticks(range(len(AUC_TARGETS)), [lab for _, lab in AUC_TARGETS], fontsize=7.2)
    a.tick_params(length=0, pad=2, colors=INK)
    a.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
    a.set_yticks(np.arange(-0.5, grid.shape[0]), minor=True)
    a.grid(which="minor", color=WHITE, linewidth=1.4)
    a.tick_params(which="minor", length=0)
    for sp in a.spines.values():
        sp.set_visible(False)



# tiles: the shared card used by panels d and e -- icon + title + subtitle
# header, a colored top rule, and a content box handed back to the caller

TILE_HEAD = 0.044


def tile(ax, x, y, w, h, icon, color, title, sub):
    rbox(ax, x, y, w, h, fc=WHITE, ec=RULE, lw=0.6, r=0.006, z=1)
    ax.add_patch(Rectangle((x + 0.003, y + h - 0.0035), w - 0.006, 0.0035, facecolor=color, edgecolor="none",
                           zorder=2))
    icon(ax, x + 0.019, y + h - 0.022, 0.0075, color)
    ax.text(x + 0.036, y + h - 0.016, title, ha="left", va="center", fontsize=8.5, fontweight="bold",
            color=INK, zorder=4)
    ax.text(x + 0.036, y + h - 0.031, sub, ha="left", va="center", fontsize=6.7, color=SUBINK, zorder=4)
    return x, y, w, h - TILE_HEAD


def icon_curve(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    xs = np.linspace(-0.6, 0.6, 30)
    ax.plot(cx + xs * s, cy + (0.45 - 0.9 * np.exp(-((xs + 0.1) ** 2) / 0.12)) * s * AY, color=WHITE, lw=1.4,
            zorder=6, solid_capstyle="round")


def icon_bars(ax, cx, cy, s, color):
    badge(ax, cx, cy, s * 1.5, color)
    for i, wv in enumerate([0.95, 0.70, 0.45]):
        ax.add_patch(Rectangle((cx - 0.50 * s, cy + (0.30 - i * 0.34) * s * AY - 0.10 * s * AY), wv * s,
                               0.20 * s * AY, facecolor=WHITE, edgecolor="none", zorder=6))


def style_mini(a, xlabel=None, ylabel=None):
    a.tick_params(labelsize=6.4, colors=SUBINK, length=2, pad=1.5)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    for sp in ("bottom", "left"):
        a.spines[sp].set_color(MUTED)
    if xlabel:
        a.set_xlabel(xlabel, fontsize=6.8, color=SUBINK, labelpad=1.5)
    if ylabel:
        a.set_ylabel(ylabel, fontsize=6.8, color=SUBINK, labelpad=1.5)


# panel c data: cohort coverage + the five feature sets

def compute_coverage():
    import pandas as pd
    from nsclc_pipeline import config
    allp = set(pd.read_csv(config.PROCESSED_DIR / "clinical.csv")["case_id"])
    seg = set(pd.read_csv(config.PROCESSED_DIR / "radiomics_features.csv")["case_id"])
    rna = set(pd.read_csv(config.PROCESSED_DIR / "rna_seq_expression.csv.gz", nrows=1).columns[1:])
    n_inter = len({f.name.split("_")[0] for f in (ROOT / "data" / "interim" / "interobserver_nifti").glob("*_ct.nii.gz")})
    return dict(total=len(allp), both=len(seg & rna & allp), seg_only=len((seg - rna) & allp),
                rna_only=len((rna - seg) & allp), neither=len(allp - seg - rna), seg=len(seg & allp),
                rna=len(rna & allp), inter=n_inter)


FEATURE_SET_ROWS = [  # (suffix of modeling_predictions file, label, blocks used)
    ("", "Radiomics alone", (1, 0, 0, 0)),
    ("__radiomics_clinical", "+ Clinical", (1, 1, 0, 0)),
    ("__radiomics_clinical_rnaseq", "+ Clinical + RNA-seq", (1, 1, 1, 0)),
    ("__embeddings", "CNN embedding alone", (0, 0, 0, 1)),
    ("__radiomics_embeddings", "Radiomics + CNN", (1, 0, 0, 1)),
]


def feature_set_ns():
    import pandas as pd
    from nsclc_pipeline import config
    return [len(pd.read_csv(config.PROCESSED_DIR / f"modeling_predictions_recurrence{sfx}.csv"))
            for sfx, _, _ in FEATURE_SET_ROWS]


def draw_waffle(ax, x, ytop, groups, cols, pitch, r):
    k = 0
    for n, color in groups:
        for _ in range(n):
            rr, cc = divmod(k, cols)
            dot(ax, x + cc * pitch, ytop - rr * pitch * AY, r, facecolor=color, edgecolor="none", z=4)
            k += 1
    return (k - 1) // cols + 1


def compute_targets():
    """Event / labeled-n for every endpoint, read from the very
    prediction files the models wrote -- so these are the n actually
    modeled, not the collection's raw label counts."""
    import pandas as pd
    from nsclc_pipeline import config
    rows = []
    for t, lab in [("recurrence", "Recurrence"), ("egfr", "EGFR mutant"), ("kras", "KRAS mutant")]:
        d = pd.read_csv(config.PROCESSED_DIR / f"modeling_predictions_{t}.csv")
        rows.append((lab, "classification", int(d["y_true"].sum()), len(d)))
    for f, lab in [("survival_predictions", "Death (OS)"), ("survival_rfs_predictions", "Recurrence (RFS)")]:
        d = pd.read_csv(config.PROCESSED_DIR / f"{f}.csv")
        rows.append((lab, "survival", int(d["event"].sum()), len(d)))
    import json
    l1 = json.loads((ROOT / "data" / "external" / "lung1" / "lung1_results.json").read_text())
    rows.append(("Death (OS), Lung1", "external", l1["events"], l1["n"]))
    lidc = json.loads((ROOT / "data" / "external" / "lidc" / "lidc_model_summary.json").read_text())
    rows.append(("Malignant nodule, LIDC", "external", lidc["n_malignant"], lidc["n_nodules"]))
    crlm = json.loads((ROOT / "data" / "external" / "crlm" / "crlm_results.json").read_text())
    rows.append(("Death (OS), liver mets", "external", crlm["os"]["events"], crlm["n_patients"]))
    return rows


def cohort_block(ax, x, y_top, cov):
    ax.text(x, y_top, "DATA AVAILABILITY", fontsize=7.2, fontweight="bold", color=MUTED, va="bottom")
    groups = [(cov["both"], NAVY), (cov["seg_only"], BLUE), (cov["rna_only"], GREEN), (cov["neither"], "#cfd5de")]
    pitch, cols = 0.0100, 27
    wy = y_top - 0.013
    nrows = draw_waffle(ax, x + 0.004, wy, groups, cols, pitch, 0.0036)
    ax.text(x + 0.004 - 0.004, wy - nrows * pitch * AY - 0.001,
            f"NSCLC Radiogenomics, n = {cov['total']} (1 dot = 1 patient)", ha="left", va="top",
            fontsize=6.6, color=SUBINK)
    lx = x + 0.004 + cols * pitch + 0.006
    items = [(NAVY, "CT + SEG + RNA-seq", cov["both"]), (BLUE, "CT + SEG only", cov["seg_only"]),
             (GREEN, "RNA-seq only", cov["rna_only"]), ("#cfd5de", "Clinical only", cov["neither"])]
    for i, (c, t, n) in enumerate(items):
        yy = wy - i * 0.0165
        dot(ax, lx, yy, 0.0036, facecolor=c, z=4)
        ax.text(lx + 0.008, yy, t, ha="left", va="center", fontsize=6.9, color=INK)
        ax.text(lx + 0.130, yy, str(n), ha="right", va="center", fontsize=6.9, fontweight="bold", color=INK)
    ax.text(lx, wy - 4 * 0.0165 + 0.002, f"Segmented CT, n = {cov['seg']}\nRNA-seq, n = {cov['rna']}",
            ha="left", va="top", fontsize=6.4, color=MUTED, linespacing=1.2)

    ix = lx + 0.150
    draw_waffle(ax, ix, wy, [(cov["inter"], TEAL)], 7, pitch, 0.0036)
    ax.text(ix + 3 * pitch, wy - 3 * pitch * AY - 0.001, f"Interobserver1\nn = {cov['inter']}, 5 readers", ha="center", va="top", fontsize=6.6, color=SUBINK, linespacing=1.15)
    import json
    l1 = json.loads((ROOT / "data" / "external" / "lung1" / "lung1_results.json").read_text())
    ey = wy - 3 * pitch * AY - 0.036
    lidc = json.loads((ROOT / "data" / "external" / "lidc" / "lidc_model_summary.json").read_text())
    crlm = json.loads((ROOT / "data" / "external" / "crlm" / "crlm_results.json").read_text())
    ey += 0.013
    rbox(ax, ix - 0.010, ey - 0.042, 7 * pitch + 0.020, 0.042, fc=WHITE, ec=TEAL, lw=0.8, r=0.004, z=3)
    ax.text(ix + 3 * pitch, ey - 0.021,
            f"External cohorts\nLung1, n = {l1['n']}\nLIDC, {lidc['n_nodules']} nodules\nLiver mets, n = {crlm['n_patients']}",
            ha="center", va="center", fontsize=5.8, color=INK, linespacing=1.15, zorder=4)
    return ix + 7 * pitch


def targets_block(ax, x0, x1, y_top, rows):
    ax.text(x0, y_top, "ENDPOINTS  (events / n)", fontsize=7.2, fontweight="bold",
            color=MUTED, va="bottom")
    lab_w, txt_w = 0.118, 0.085
    bx0, bx1 = x0 + lab_w, x1 - txt_w
    rh = 0.0120
    for i, (lab, kind, ev, n) in enumerate(rows):
        yy = y_top - 0.012 - i * rh
        c = {"classification": BLUE, "survival": GREEN, "external": TEAL}[kind]
        ax.text(x0, yy, lab, ha="left", va="center", fontsize=6.8, color=INK)
        rbox(ax, bx0, yy - 0.0045, bx1 - bx0, 0.009, fc="#e3e7ed", r=0.002, z=2)
        rbox(ax, bx0, yy - 0.0045, (bx1 - bx0) * ev / n, 0.009, fc=c, r=0.002, z=3)
        ax.text(x1, yy, f"{ev} / {n}  ({ev / n * 100:.0f}%)", ha="right", va="center", fontsize=6.9,
                color=INK, fontweight="bold")
    ly = y_top + 0.004  # legend beside the heading, clear of the panel edge
    for j, (c, t) in enumerate([(BLUE, "Classification"), (GREEN, "Survival"), (TEAL, "External cohorts")]):
        lx = x0 + 0.150 + j * 0.085
        rbox(ax, lx, ly - 0.004, 0.012, 0.008, fc=c, r=0.002, z=3)
        ax.text(lx + 0.016, ly, t, ha="left", va="center", fontsize=6.4, color=SUBINK)


def feature_matrix(ax, x0, x1, y_top, ns):
    ax.text(x0, y_top, "FEATURE SETS", fontsize=7.2, fontweight="bold", color=MUTED,
            va="bottom")
    lab_w = 0.150
    cols = [("Radiomics", "26 features", BLUE, icon_target), ("Clinical", "11 variables", PURPLE, icon_people),
            ("RNA-seq", "10 PCs", GREEN, icon_dna), ("CNN", "10 PCs", TEAL, icon_grid)]
    n_w = 0.060
    cw = (x1 - x0 - lab_w - n_w) / len(cols)
    cxs = [x0 + lab_w + cw * (j + 0.5) for j in range(len(cols))]
    hy = y_top - 0.020
    for (t, s, c, ic), cx in zip(cols, cxs):
        ic(ax, cx - 0.030, hy, 0.0068, c)
        ax.text(cx - 0.017, hy + 0.004, t, ha="left", va="center", fontsize=7.4, fontweight="bold", color=INK)
        ax.text(cx - 0.017, hy - 0.008, s, ha="left", va="center", fontsize=6.5, color=SUBINK)
    ax.text(x1 - n_w / 2, hy - 0.002, "n\n(recur.)", ha="center", va="center", fontsize=6.6, color=SUBINK,
            linespacing=1.1)
    rh = 0.0185
    ry = hy - 0.024
    for i, ((_, lab, use), n) in enumerate(zip(FEATURE_SET_ROWS, ns)):
        yy = ry - i * rh
        if i % 2 == 0:
            rbox(ax, x0, yy - rh / 2, x1 - x0, rh, fc="#eef1f5", r=0.003, z=1)
        ax.text(x0 + 0.008, yy, lab, ha="left", va="center", fontsize=7.4, color=INK, zorder=4)
        on = [j for j, u in enumerate(use) if u]
        if len(on) > 1:
            ax.plot([cxs[on[0]], cxs[on[-1]]], [yy, yy], color=INK, lw=1.4, zorder=3)
        for j, u in enumerate(use):
            if u:
                dot(ax, cxs[j], yy, 0.0050, facecolor=cols[j][2], edgecolor=WHITE, linewidth=0.6, z=5)
            else:
                dot(ax, cxs[j], yy, 0.0034, facecolor="#d5dae2", z=4)
        ax.text(x1 - n_w / 2, yy, str(n), ha="center", va="center", fontsize=7.4, fontweight="bold", color=INK,
                zorder=4)
    ax.text(x0 + 0.008, ry - 5 * rh + 0.004, "CNN, ImageNet-pretrained ResNet18 embedding. KRAS modeled with radiomics only.", ha="left", va="top", fontsize=6.5, color=MUTED, style="italic")


# panel d data: clustering map + GSEA

def compute_cluster_map():
    from sklearn.decomposition import PCA
    from scripts.figures.supp_phenotypes import CLUSTER_COLORS
    from nsclc_pipeline.phenotypes import fit_reference_phenotypes as _fit_original_clustering
    _df, _fc, _sc, _imp, labels, _cent, Xz = _fit_original_clustering()
    xy = PCA(n_components=2, random_state=0).fit_transform(Xz)
    return xy, labels, CLUSTER_COLORS


def compute_gsea(n_up=3, n_down=2):
    import pandas as pd
    from nsclc_pipeline import config
    g = pd.read_csv(config.PROCESSED_DIR / "radiogenomics_gsea_top.csv").sort_values("NES", ascending=False)
    short = {"Epithelial Mesenchymal Transition": "EMT", "Interferon Alpha Response": "IFN-α response",
             "Interferon Gamma Response": "IFN-γ response", "Bile Acid Metabolism": "Bile acid metab.",
             "E2F Targets": "E2F targets"}
    sel = pd.concat([g.head(n_up), g.tail(n_down)])
    sel["label"] = sel["Term"].map(lambda t: short.get(t, t))
    return sel[["label", "NES", "FDR q-val"]]


def draw_cluster_map(a, xy, labels, colors):
    for c in sorted(np.unique(labels)):
        m = labels == c
        a.scatter(xy[m, 0], xy[m, 1], s=7, color=colors[c], edgecolors=WHITE, linewidths=0.3,
                  label=f"P{c + 1} (n = {m.sum()})", zorder=3)
    a.set_xticks([])
    a.set_yticks([])
    lo, hi = xy[:, 1].min(), xy[:, 1].max()
    a.set_ylim(lo - 0.05 * (hi - lo), hi + 0.42 * (hi - lo))
    style_mini(a, "PC1", "PC2")
    a.legend(fontsize=5.8, frameon=False, loc="upper right", handletextpad=0.1, borderaxespad=0.1,
             labelspacing=0.2, markerscale=1.1, ncol=2, columnspacing=0.5)


def draw_gsea(a, sel):
    r = sel.iloc[::-1]
    y = np.arange(len(r))
    a.barh(y, r["NES"], color=[TEAL if v > 0 else "#9aa3af" for v in r["NES"]], height=0.62)
    a.axvline(0, color=INK, lw=0.7)
    a.set_yticks(y, r["label"], fontsize=6.6)
    a.tick_params(axis="y", length=0, pad=2)
    style_mini(a, "NES (high vs low risk)")
    a.spines["left"].set_visible(False)


# panel e data: perturbation contours, sensitivity curve, real readers,
# PyRadiomics cross-check

MM_COLORS = {-2: "#b3421a", -1: ORANGE, 0: WHITE, 1: "#7fb3ee", 2: BLUE}


def compute_perturbation_contours(case_id=PIPE_CASE):
    import pandas as pd
    from nsclc_pipeline import config, perturbation, viz
    feats = pd.read_csv(config.PROCESSED_DIR / "radiomics_features.csv")
    ct_uid = feats.loc[feats["case_id"] == case_id, "ct_series_uid"].iloc[0]
    ct = viz.load_ct_volume(ROOT / "data" / "interim" / "dicom" / ct_uid)
    base = viz.load_mask_on_ct(config.PROCESSED_DIR / "masks" / f"{case_id}_seg.nii.gz", ct)
    ct_arr = sitk.GetArrayFromImage(ct)
    b = sitk.GetArrayFromImage(base) > 0
    z = int(np.argmax(b.reshape(b.shape[0], -1).sum(axis=1)))
    ys, xs = np.where(b[z])
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.62) + 6
    sl = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
    masks = {mm: (sitk.GetArrayFromImage(perturbation.perturb_mask(base, mm))[z] > 0)[sl]
             for mm in (-2, -1, 0, 1, 2)}
    gray = viz.window_slice(ct_arr[z], *viz.WINDOW_PRESETS["Lung"])[sl]
    return gray, masks


def compute_sensitivity():
    import pandas as pd
    from nsclc_pipeline import config
    from nsclc_pipeline.phenotypes import reassignment_by_shift as _cluster_instability_by_mm
    auc = pd.read_csv(config.PROCESSED_DIR / "perturbation_auc_comparison.csv")
    surv = pd.read_csv(config.PROCESSED_DIR / "perturbation_survival_comparison.csv")
    out = {}
    for t in ("recurrence", "egfr"):
        sub = auc[auc["target"] == t]
        s = dict(zip(sub["perturbation_mm"], sub["mean_auc"]))
        s[0] = float(sub["original_auc"].iloc[0])
        out[t] = s
    s = dict(zip(surv["perturbation_mm"], surv["mean_concordance"]))
    s[0] = float(surv["original_concordance"].iloc[0])
    out["os"] = s
    ci = _cluster_instability_by_mm()
    out["ari"] = dict(zip(ci["mm"], ci["ari"]))
    out["reassigned"] = dict(zip(ci["mm"], ci["pct_reassigned"]))
    return out


def compute_interobserver(pid="interobs21"):
    import pandas as pd
    d = ROOT / "data" / "interim" / "interobserver_nifti"
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(d / f"{pid}_ct.nii.gz")))
    obs = [sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0 for f in sorted(d.glob(f"{pid}_obs*.nii.gz"))]
    union = np.any(obs, axis=0)
    z = int(np.argmax(union.reshape(union.shape[0], -1).sum(axis=1)))
    ys, xs = np.where(union[z])
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.62) + 6
    sl = (slice(max(cy - half, 0), cy + half), slice(max(cx - half, 0), cx + half))
    from nsclc_pipeline import viz
    gray = viz.window_slice(ct[z], *viz.WINDOW_PRESETS["Lung"])[sl]
    sd = pd.read_csv(ROOT / "data" / "processed" / "interobserver_surface_distance.csv")
    return gray, [o[z][sl] for o in obs], float(sd["assd_mm"].median()), sd["patient"].nunique()


def compute_crosscheck():
    import pandas as pd
    from scipy.stats import spearmanr
    d = pd.read_csv(ROOT / "data" / "processed" / "pyradiomics_crosscheck.csv")
    rho = spearmanr(d["custom_extractor_instability"], d["pyradiomics_instability"]).correlation
    return d, rho


def draw_perturbation(fig, ax, box, gray, masks):
    x, y, w, h = box
    iw = min(w - 0.052, (h - 0.012) / AY)
    cx, cy = x + 0.008 + iw / 2, y + h / 2
    a = img_axes(fig, cx, cy, iw)
    a.imshow(gray, cmap="gray", interpolation="bilinear")
    for mm in (-2, -1, 1, 2, 0):
        a.contour(masks[mm], levels=[0.5], colors=[MM_COLORS[mm]], linewidths=1.6 if mm == 0 else 1.1,
                  linestyles=["solid"] if mm == 0 else [(0, (3, 1.2))])
    lx = x + 0.016 + iw
    for i, mm in enumerate((2, 1, 0, -1, -2)):
        yy = cy + (2 - i) * 0.016
        c = INK if mm == 0 else MM_COLORS[mm]
        ax.plot([lx, lx + 0.010], [yy, yy], color=c, lw=1.6 if mm == 0 else 1.2,
                ls="-" if mm == 0 else (0, (3, 1.2)), zorder=4)
        ax.text(lx + 0.013, yy, f"{mm:+d}" if mm else "0", ha="left", va="center", fontsize=6.6, color=INK,
                fontweight="bold" if mm == 0 else "normal")
    ax.text(lx + 0.010, cy - 0.047, "mm", ha="center", va="center", fontsize=6.4, color=SUBINK)


def draw_sensitivity(a, S):
    mm = [-2, -1, 0, 1, 2]
    series = [("recurrence", "Recurrence AUC", BLUE), ("egfr", "EGFR AUC", PURPLE),
              ("os", "OS C-index", GREEN), ("ari", "Cluster ARI", AMBER)]
    a.axvspan(-0.5, 0.5, color="#eef1f5", zorder=0)
    a.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    for key, lab, c in series:
        a.plot(mm, [S[key][m] for m in mm], color=c, lw=1.5, marker="o", ms=2.6, label=lab, zorder=3)
    a.set_xticks(mm, ["−2", "−1", "0", "+1", "+2"])
    a.set_ylim(0, 1.0)
    a.set_yticks([0, 0.5, 1.0])
    style_mini(a, "Boundary shift (mm)")
    a.legend(fontsize=5.6, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
             handlelength=1.0, labelspacing=0.1, borderaxespad=0.0, handletextpad=0.3, columnspacing=0.6)
    pct = S["reassigned"][-1]
    a.annotate(f"{pct:.1f}%\nreassigned", xy=(-1, S["ari"][-1]), xytext=(0.15, 0.24),
               fontsize=5.8, color=AMBER, fontweight="bold", va="top",
               arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.7))


def draw_readers(fig, ax, box, gray, obs, assd):
    x, y, w, h = box
    iw = min(w - 0.052, (h - 0.012) / AY)
    cx, cy = x + 0.008 + iw / 2, y + h / 2
    a = img_axes(fig, cx, cy, iw)
    a.imshow(gray, cmap="gray", interpolation="bilinear")
    rc = [BLUE, ORANGE, GREEN, PINK, AMBER]
    for i, o in enumerate(obs):
        a.contour(o, levels=[0.5], colors=[rc[i % len(rc)]], linewidths=1.1)
    lx = x + 0.016 + iw
    for i in range(len(obs)):
        yy = cy + (2 - i) * 0.016
        ax.plot([lx, lx + 0.010], [yy, yy], color=rc[i % len(rc)], lw=1.4, zorder=4)
        ax.text(lx + 0.013, yy, f"R{i + 1}", ha="left", va="center", fontsize=6.6, color=INK)


def draw_crosscheck(a, d, rho):
    xv, yv = d["custom_extractor_instability"], d["pyradiomics_instability"]
    lo, hi = min(xv.min(), yv.min()) * 0.7, max(xv.max(), yv.max()) * 1.4
    a.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    a.scatter(xv, yv, s=10, color=AMBER, edgecolors=WHITE, linewidths=0.4, zorder=3)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(lo, hi)
    a.set_ylim(lo, hi)
    a.minorticks_off()
    style_mini(a, "Study extractor", "PyRadiomics")
    a.text(0.04, 0.96, f"ρ = {rho:.2f}\n{len(d)} features", transform=a.transAxes, ha="left", va="top",
           fontsize=6.6, color=INK)


def draw_shap_mini(a, rank):
    r = rank.iloc[::-1]
    y = np.arange(len(r))
    a.barh(y, r["v"], color=PINK, height=0.62, edgecolor="none")
    a.set_yticks(y, r["label"], fontsize=6.6)
    a.tick_params(axis="y", length=0, pad=2)
    a.set_xlim(0, r["v"].max() * 1.12)
    style_mini(a, "mean |SHAP|")
    a.spines["left"].set_visible(False)


RED_AI = "#c8392b"


def draw_ai_tile(fig, ax, box):
    """Expert vs public nnU-Net contour for the patient with median Dice, plus
    the headline agreement and decision-flip numbers."""
    import pandas as pd
    from scripts.figures.main_figures import _ai_example
    ag = pd.read_csv(ROOT / "data" / "processed" / "bamf_agreement.csv")
    det = ag[(ag.source == "ai") & ag.detected]
    case = det.iloc[(det.dice - det.dice.median()).abs().argmin()].case_id
    g, e, a = _ai_example(case)
    x, y, w, h = box
    iw = min(w - 0.058, (h - 0.012) / AY)
    cx, cy = x + 0.008 + iw / 2, y + h / 2
    im = img_axes(fig, cx, cy, iw)
    im.imshow(g, cmap="gray", interpolation="bilinear")
    im.contour(e, levels=[0.5], colors=[WHITE], linewidths=1.3)
    im.contour(a, levels=[0.5], colors=[RED_AI], linewidths=1.2, linestyles=[(0, (3, 1.2))])
    # AI reclassification averaged over the 20 repetitions (scripts/discovery/run_same_patient_comparison.py)
    dc = pd.read_csv(ROOT / "data" / "processed" / "decomposition_summary.csv")
    fl = dc[dc.comparison == "AI"].segmentation_rate * 100
    lx = x + 0.016 + iw
    ax.plot([lx, lx + 0.010], [cy + 0.030] * 2, color=INK, lw=1.3, zorder=4)
    ax.text(lx + 0.013, cy + 0.030, "Expert", ha="left", va="center", fontsize=6.3, color=INK)
    ax.plot([lx, lx + 0.010], [cy + 0.014] * 2, color=RED_AI, lw=1.2, ls=(0, (3, 1.2)), zorder=4)
    ax.text(lx + 0.013, cy + 0.014, "AI", ha="left", va="center", fontsize=6.3, color=INK)
    ax.text(lx, cy - 0.010, f"Dice\n{det.dice.median():.2f}", ha="left", va="top", fontsize=6.3,
            color=SUBINK, linespacing=1.1)
    ax.text(lx, cy - 0.036, f"Reclassified\n{fl.min():.0f}–{fl.max():.0f}%", ha="left", va="top", fontsize=6.3,
            color=RED_AI, fontweight="bold", linespacing=1.1)


def draw_external(a):
    """Lung1: concordance is flat across boundary shifts, while the risk
    ordering of patients changes (rank correlation with the original)."""
    import json
    import pandas as pd
    L1 = ROOT / "data" / "external" / "lung1"
    res = json.loads((L1 / "lung1_results.json").read_text())
    fd = pd.read_csv(ROOT / "data" / "processed" / "flip_decomposition.csv")
    fd = fd[fd.cohort == "Lung1"].set_index("comparison")
    mm = [-2, -1, 0, 1, 2]
    a.plot(mm, [res[f"c_mean_{m:+d}"] for m in mm], color=GREEN, lw=1.5, marker="o", ms=2.6, label="C-index")
    rho = [fd.loc["expert vs -1 mm", "spearman"], 1.0, fd.loc["expert vs +1 mm", "spearman"]]
    a.plot([-1, 0, 1], rho, color=ORANGE, lw=1.5, marker="s", ms=2.6, label="Risk-rank ρ")
    a.set_xticks(mm, ["−2", "−1", "0", "+1", "+2"])
    a.set_ylim(0, 1.05)
    a.set_yticks([0, 0.5, 1.0])
    style_mini(a, "Boundary shift (mm)")
    a.legend(fontsize=5.6, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
             handlelength=1.0, borderaxespad=0.0, handletextpad=0.3, columnspacing=0.6)


# main figure

def main():
    print("Loading real CT crops for the two representative patients...")
    ct_low = load_real_ct_crop("R01-049")
    ct_high = load_real_ct_crop("R01-037")
    print(f"Building imaging-pipeline panel for {PIPE_CASE} (mesh, GLCM, 3D habitats)...")
    pipe = load_pipeline_case(PIPE_CASE)
    print("Cohort coverage, targets and feature-set sizes...")
    cov = compute_coverage()
    targets = compute_targets()
    fs_ns = feature_set_ns()
    print("Nested-CV AUC grid, OS Kaplan-Meier, clustering map, GSEA...")
    auc_grid = compute_auc_grid()
    km_curves, km_p = compute_real_km()
    cl_xy, cl_labels, cl_colors = compute_cluster_map()
    gsea = compute_gsea()
    print("Perturbation contours, sensitivity curve, interobserver readers, PyRadiomics cross-check, SHAP...")
    pert_gray, pert_masks = compute_perturbation_contours(PIPE_CASE)
    sens = compute_sensitivity()
    io_gray, io_obs, io_assd, io_n = compute_interobserver("interobs21")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(-EXTRA, TOP)
    ax.axis("off")

    L, R = 0.020, 0.980
    PW = R - L
    G = 0.012  # vertical gap between panels

    # a: the question
    a_y, a_h = 0.737, 0.213
    panel(ax, L, a_y, PW, a_h, "a", "Representative patients")
    cw = (PW - 0.036) / 2
    card_y, card_h = a_y + 0.012, a_h - 0.050
    patient_card(fig, ax, L + 0.012, card_y, cw, card_h, BLUE, BLUE_LT, "Low-risk patient", "R01-049", ct_low,
                 [("Age", "70"), ("Sex", "Female"), ("Histology", "Squamous cell"),
                  ("T stage", "T1a"), ("Chemotherapy", "No"), ("Tumor volume", "3 cm³")],
                 "Alive at 65 months")
    patient_card(fig, ax, L + 0.024 + cw, card_y, cw, card_h, ORANGE, ORANGE_LT, "High-risk patient", "R01-037",
                 ct_high,
                 [("Age", "84"), ("Sex", "Male"), ("Histology", "Adenocarcinoma"),
                  ("T stage", "T2a"), ("Chemotherapy", "No"), ("Tumor volume", "8 cm³")],
                 "Died at 13 months")

    # b: cohort & targets
    b_h = 0.150
    b_y = a_y - G - b_h
    panel(ax, L, b_y, PW, b_h, "b", "Cohort and endpoints")
    b_top = b_y + b_h - 0.046
    cx_end = cohort_block(ax, L + 0.014, b_top, cov)
    targets_block(ax, cx_end + 0.040, R - 0.014, b_top, targets)

    # c: imaging pipeline
    c_h = 0.232
    c_y = b_y - G - c_h
    panel(ax, L, c_y, PW, c_h, "c", "Image processing and feature extraction")
    pipeline_panel(fig, ax, L, c_y, PW, c_h, pipe)

    # d: feature sets -> nested CV
    d_h = 0.190
    d_y = c_y - G - d_h
    panel(ax, L, d_y, PW, d_h, "d", "Feature sets and nested cross-validation")
    d_top = d_y + d_h - 0.046
    fm_x1 = 0.640
    feature_matrix(ax, L + 0.014, fm_x1, d_top, fs_ns)
    cv_x, cv_w = fm_x1 + 0.075, 0.200
    mid_d = d_top - 0.058
    arrow(ax, (fm_x1 + 0.014, mid_d), (cv_x - 0.016, mid_d))
    nested_cv_diagram(ax, cv_x, mid_d + 0.010, cv_w)

    # e: four analysis paradigms
    e_h = 0.205
    e_y = d_y - G - e_h
    panel(ax, L, e_y, PW, e_h, "e", "Analyses")
    t_bot, t_top = e_y + 0.022, e_y + e_h - 0.042
    gaps = [0.012, 0.012, 0.036]
    tw = (PW - 0.024 - sum(gaps)) / 4
    th = t_top - t_bot
    txs = [L + 0.012]
    for gp in gaps:
        txs.append(txs[-1] + tw + gp)
    specs = [
        (icon_target, BLUE, "Classification", "AUC by endpoint and feature set"),
        (icon_cluster, PURPLE, "Clustering", "Ward linkage, k = 4 (n = 144)"),
        (icon_survival, GREEN, "Survival", "Cox PH, overall survival"),
        (icon_dna, TEAL, "Radiogenomics*", "Hallmark GSEA, high vs low risk"),
    ]
    boxes = [tile(ax, txs[i], t_bot, tw, th, ic, c, t, s) for i, (ic, c, t, s) in enumerate(specs)]
    # radiogenomics consumes the survival model's out-of-fold risk score
    ya = t_bot + th * 0.46
    arrow(ax, (txs[2] + tw + 0.003, ya), (txs[3] - 0.003, ya), color=GREEN, lw=1.6, ms=10, z=6)
    ax.text(txs[2] + tw + gaps[2] / 2, ya + 0.010, "Risk\nscore", ha="center", va="bottom", fontsize=6.2,
            color=GREEN, fontweight="bold", linespacing=1.05)

    bx, by, bw, bh = boxes[0]
    hm = fax(fig, [bx + 0.046, by + 0.040, bw - 0.054, bh - 0.046])
    draw_auc_heatmap(hm, auc_grid)
    hm.set_yticks(range(len(AUC_TARGETS)), ["Recur.", "EGFR", "KRAS"], fontsize=6.8)
    hm.set_xticks(range(len(AUC_SETS)), ["Rad.", "+Clin.", "+Clin.+RNA", "CNN", "Rad.+CNN"], rotation=40,
                  ha="right", rotation_mode="anchor", fontsize=6.4)
    for t in hm.texts:
        t.set_fontsize(6.8)
    bx, by, bw, bh = boxes[1]
    ca = fax(fig, [bx + 0.018, by + 0.020, bw - 0.026, bh - 0.026])
    draw_cluster_map(ca, cl_xy, cl_labels, cl_colors)
    bx, by, bw, bh = boxes[2]
    ka = fax(fig, [bx + 0.034, by + 0.030, bw - 0.044, bh - 0.036])
    draw_km(ka, km_curves, km_p)
    ka.tick_params(labelsize=6.4)
    ka.xaxis.label.set_fontsize(6.8)
    ka.set_ylabel("Overall survival", fontsize=6.8, color=SUBINK, labelpad=1.5)
    ka.legend(fontsize=6.0, frameon=False, loc="lower left", handlelength=1.1, borderaxespad=0.1)
    for t in ka.texts:
        t.set_fontsize(6.6)
    bx, by, bw, bh = boxes[3]
    ga = fax(fig, [bx + 0.066, by + 0.030, bw - 0.074, bh - 0.036])
    draw_gsea(ga, gsea)
    ax.text(L + 0.012, e_y + 0.007, "*Descriptive; not included in the segmentation-adjustment analysis.", ha="left", va="bottom",
            fontsize=6.6, color=MUTED, style="italic")

    # f: robustness & interpretation
    assert d_y - 2 * G - 0.205 - 0.232 >= -EXTRA - 1e-6, "layout overflows page; raise EXTRA"
    f_h = 0.232
    f_y = e_y - G - f_h
    panel(ax, L, f_y, PW, f_h, "f", "Segmentation variation and external cohort")
    t_bot, t_top = f_y + 0.012, f_y + f_h - 0.056
    gap = 0.010
    tw = (PW - 0.024 - 4 * gap) / 5
    th = t_top - t_bot
    xs = [L + 0.012 + i * (tw + gap) for i in range(5)]
    gy = t_top + 0.011
    ax.plot([xs[0], xs[3] + tw], [gy, gy], color=MUTED, lw=0.8, zorder=2)
    ax.text((xs[0] + xs[3] + tw) / 2, gy, "  SAME PATIENTS, DIFFERENT SEGMENTATIONS  ",
            ha="center", va="center", fontsize=7.0, fontweight="bold", color=MUTED, backgroundcolor=PANEL_BG,
            zorder=3)
    ax.plot([xs[4], xs[4] + tw], [gy, gy], color=MUTED, lw=0.8, zorder=2)
    ax.text(xs[4] + tw / 2, gy, "  EXTERNAL COHORT  ", ha="center", va="center", fontsize=7.0, fontweight="bold",
            color=MUTED, backgroundcolor=PANEL_BG, zorder=3)
    especs = [
        (icon_boundary, ORANGE, "Segmentation adjustment", "Tighter or looser by 1–2 mm"),
        (icon_curve, NAVY, "Sensitivity curve", "Performance vs boundary shift"),
        (icon_people, TEAL, "Radiation oncologists", f"n = {io_n}, median ASSD {io_assd:.1f} mm"),
        (icon_grid, RED_AI, "AI segmentation", "Public nnU-Net vs expert"),
        (icon_curve, GREEN, "External cohort", "Lung1, n = 419"),
    ]
    eb = [tile(ax, xs[i], t_bot, tw, th, ic, c, t, s) for i, (ic, c, t, s) in enumerate(especs)]
    arrow(ax, (xs[0] + tw + 0.001, t_bot + th * 0.40), (xs[1] - 0.001, t_bot + th * 0.40), color=ORANGE,
          lw=1.4, ms=9, z=6)
    draw_perturbation(fig, ax, eb[0], pert_gray, pert_masks)
    bx, by, bw, bh = eb[1]
    sa = fax(fig, [bx + 0.026, by + 0.030, bw - 0.032, bh - 0.056])
    draw_sensitivity(sa, sens)
    draw_readers(fig, ax, eb[2], io_gray, io_obs, io_assd)
    draw_ai_tile(fig, ax, eb[3])
    bx, by, bw, bh = eb[4]
    la = fax(fig, [bx + 0.030, by + 0.030, bw - 0.036, bh - 0.056])
    draw_external(la)

    out_png = FIG_DIR / "figure1_study_workflow.png"
    out_pdf = FIG_DIR / "figure1_study_workflow.pdf"
    fig.savefig(out_png, dpi=300, facecolor=WHITE)
    try:
        fig.savefig(out_pdf, facecolor=WHITE)
        print(f"Saved {out_pdf}")
    except PermissionError as exc:
        print(f"Skipped PDF (locked by another process): {exc}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
