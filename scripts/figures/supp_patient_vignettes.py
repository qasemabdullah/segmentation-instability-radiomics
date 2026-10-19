"""Supplementary Fig. S13: one patient from each overall-survival risk tertile.

Run: python scripts/figures/supp_patient_vignettes.py
Output: outputs/figures/supplementary/_parts/supp_fig_s11_vignette.png, .pdf
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.figure import Figure

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import viz  # noqa: E402
from scripts.figures import palette  # noqa: E402

palette.apply_style()  # Arial, compact type (the style of the desktop viewer)

DATA_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures" / "supplementary" / "_parts"  # full render; render_supplementary_figures.py trims it
FIG_DIR.mkdir(parents=True, exist_ok=True)

TERTILE_COLOR = {"Low": palette.CATEGORICAL[0], "Intermediate": palette.CATEGORICAL[3], "High": palette.CATEGORICAL[1]}


def _save(fig, stem: str):
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    except PermissionError as exc:
        print(f"  (skipped PDF, locked: {exc})")
    print(f"Saved {png}")


def pick_representatives():
    tert = pd.read_csv(DATA_DIR / "km_tertile_os.csv")
    reps = {}
    for grp in ["Low", "Intermediate", "High"]:
        sub = tert[tert["risk_tertile"] == grp].copy()
        med = sub["risk_score"].median()
        sub["dist"] = (sub["risk_score"] - med).abs()
        reps[grp] = sub.sort_values("dist").iloc[0]["case_id"]
    return reps


def load_crop(case_id, pad_frac=1.9):
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
    half = max(int(max(ys.max() - ys.min(), xs.max() - xs.min()) * pad_frac), 110)
    y0, y1 = max(cy - half, 0), min(cy + half, ct_arr.shape[1])
    x0, x1 = max(cx - half, 0), min(cx + half, ct_arr.shape[2])

    gray = viz.window_slice(ct_arr[z], *viz.WINDOW_PRESETS["Lung"])
    crop = gray[y0:y1, x0:x1]
    mask_crop = mask_arr[z, y0:y1, x0:x1] > 0
    return crop, mask_crop


def patient_caption(case_id):
    clin = pd.read_csv(DATA_DIR / "clinical.csv")
    tert = pd.read_csv(DATA_DIR / "km_tertile_os.csv")
    rad = pd.read_csv(DATA_DIR / "radiomics_features.csv")
    row = clin[clin["case_id"] == case_id].iloc[0]
    t_row = tert[tert["case_id"] == case_id].iloc[0]
    volume_cc = rad.loc[rad["case_id"] == case_id, "shape_volume_mm3"].iloc[0] / 1000.0

    months = t_row["duration_days"] / 30.44
    outcome = f"Died at {months:.0f} months" if t_row["event"] else f"Alive at last follow-up, {months:.0f} months"
    lines = [
        f"{case_id}",
        f"Age {int(row['age_at_histological_diagnosis'])}, {row['gender']}",
        f"{row['histology']}, {row['pathological_t_stage']}",
        f"Tumor volume: {volume_cc:.0f} cm³",
        f"Chemotherapy: {str(row['chemotherapy']).capitalize()}",
        f"{outcome}",
    ]
    return lines


def main():
    reps = pick_representatives()
    print("Representative patients (closest to tertile median risk score):", reps)

    fig = Figure(figsize=(10.5, 5.4), dpi=100, layout="constrained")
    axes = fig.subplots(1, 3)

    for ax, grp in zip(axes, ["Low", "Intermediate", "High"]):
        case_id = reps[grp]
        crop, mask_crop = load_crop(case_id)
        ax.imshow(crop, cmap="gray", interpolation="bilinear")
        ax.contour(mask_crop, levels=[0.5], colors=[TERTILE_COLOR[grp]], linewidths=2.4, zorder=5)
        ax.set_xlim(0, crop.shape[1])
        ax.set_ylim(crop.shape[0], 0)
        ax.axis("off")
        ax.set_title(f"{grp}-risk", fontsize=11, fontweight="bold", color=TERTILE_COLOR[grp], pad=14)

        caption = "\n".join(patient_caption(case_id))
        ax.text(0.5, -0.04, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color=palette.INK2, linespacing=1.5)

    fig.suptitle("Supplementary Figure S11. Representative patients across OS risk tertiles",
                 fontsize=12, fontweight="bold", x=0.02, y=1.04, ha="left")
    _save(fig, "supp_fig_s11_vignette")


if __name__ == "__main__":
    main()
