"""Multi-reader NSCLC cohort: disagreement between five radiation oncologists and feature instability between them.

Paper:    Fig. 4a-b
Pipeline: python pipeline.py run interobserver   (stage 'readers')

This directly answers the methodological question raised in review: does
a uniform +/-1mm morphological erosion/dilation actually resemble the
magnitude of real inter-observer disagreement, or is it an arbitrary
choice? We compute, per patient, all 10 pairwise combinations of the 5
manual observer masks, and for each pair:
  - the real geometric boundary disagreement (mean symmetric surface
    distance, mm) -- directly comparable to our +/-1/+/-2 mm perturbation
    levels;
  - the real per-feature relative change in the same 26 radiomics
    features used throughout this project, for direct comparison against
    perturbation.feature_stability_summary()'s synthetic -1mm-vs-+1mm
    numbers already reported in the paper.

Run: python scripts/readers/analyze_interobserver.py
Output: data/processed/interobserver_surface_distance.csv
        data/processed/interobserver_feature_stability.csv
        outputs/figures/exploratory/figure8_interobserver_validation.png/.pdf (exploratory, not in the paper)
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import pydicom_seg
import SimpleITK as sitk

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import radiomics, seg_to_mask, viz  # noqa: E402
from nsclc_pipeline.perturbation import feature_stability_summary  # noqa: E402

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

BLUE, ORANGE, GREEN, AMBER, PINK, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c9960c", "#e87ba4", "#9aa3ad"
INK, SUBINK = "#14161c", "#5b6470"

DATA_DIR = ROOT / "data" / "processed"
DICOM_DIR = ROOT / "data" / "interim" / "dicom"
FIG_DIR = ROOT / "outputs" / "figures" / "exploratory"  # exploratory figure, not in the paper
FIG_DIR.mkdir(parents=True, exist_ok=True)
SERIES_JSON = ROOT / "data" / "raw" / "interobs_series.json"


def mean_surface_distance(mask_a: sitk.Image, mask_b: sitk.Image) -> float:
    """Average symmetric surface distance (mm) between two binary masks
    sharing the same grid -- the standard metric for inter-observer
    contouring studies."""
    a_contour = sitk.GetArrayFromImage(sitk.LabelContour(mask_a, fullyConnected=True)) > 0
    b_contour = sitk.GetArrayFromImage(sitk.LabelContour(mask_b, fullyConnected=True)) > 0
    if not a_contour.any() or not b_contour.any():
        return float("nan")
    dist_a = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(mask_a, squaredDistance=False, useImageSpacing=True))
    dist_b = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(mask_b, squaredDistance=False, useImageSpacing=True))
    d_a_to_b = np.abs(dist_b[a_contour])
    d_b_to_a = np.abs(dist_a[b_contour])
    return float(np.concatenate([d_a_to_b, d_b_to_a]).mean())


def load_manual_segments(patient_id: str, seg_series_uid: str, ct_series_uid: str):
    seg_dir = DICOM_DIR / seg_series_uid
    ct_dir = DICOM_DIR / ct_series_uid
    seg_file = seg_to_mask.find_seg_dicom(seg_dir)
    dcm = pydicom.dcmread(seg_file)

    observer_nums = {}
    for item in dcm.SegmentSequence:
        desc = item.get("SegmentDescription", "") or ""
        if "vis" in desc:
            # "GTV-1vis-3" -> observer 3
            observer_nums[int(item.SegmentNumber)] = int(desc.rsplit("-", 1)[-1])

    result = pydicom_seg.SegmentReader().read(dcm)
    ct_image = viz.load_ct_volume(ct_dir)
    ct_array = sitk.GetArrayFromImage(ct_image)

    masks = {}
    for seg_num, observer in observer_nums.items():
        seg_image = result.segment_image(seg_num)
        # Resample onto the CT grid (same acquisition, but sizes can differ
        # by a slice or two at the volume edges) -- same approach as
        # viz.load_mask_on_ct for our primary cohort.
        seg_on_ct = sitk.Resample(seg_image, ct_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg_image.GetPixelID())
        masks[observer] = seg_on_ct

    return ct_image, ct_array, masks


def extract_features(ct_array: np.ndarray, mask_image: sitk.Image) -> dict:
    mask_array = sitk.GetArrayFromImage(mask_image) > 0
    if mask_array.sum() == 0:
        return None
    feats = {}
    feats.update(radiomics.shape_features(mask_image))
    feats.update(radiomics.first_order_features(ct_array[mask_array]))
    feats.update(radiomics.texture_features(ct_array, mask_array))
    return feats


def main():
    with open(SERIES_JSON) as f:
        series = json.load(f)
    by_patient = {}
    for d in series:
        by_patient.setdefault(d["PatientID"], {})[d["Modality"]] = d["SeriesInstanceUID"]

    distance_rows = []
    feature_rows = []  # long format: patient, observer, feature, value
    n_ok, n_skip = 0, 0

    for patient_id, mods in sorted(by_patient.items()):
        if "SEG" not in mods or "CT" not in mods:
            n_skip += 1
            continue
        print(f"{patient_id} ...")
        try:
            ct_image, ct_array, masks = load_manual_segments(patient_id, mods["SEG"], mods["CT"])
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            n_skip += 1
            continue
        if len(masks) != 5:
            print(f"  expected 5 manual observers, got {len(masks)} -- skipping")
            n_skip += 1
            continue

        feats_by_observer = {}
        for obs, mask_img in masks.items():
            f = extract_features(ct_array, mask_img)
            if f is None:
                continue
            feats_by_observer[obs] = f
            for feat_name, val in f.items():
                feature_rows.append({"patient": patient_id, "observer": obs, "feature": feat_name, "value": val})

        for obs_a, obs_b in itertools.combinations(sorted(masks), 2):
            dist = mean_surface_distance(masks[obs_a], masks[obs_b])
            distance_rows.append({"patient": patient_id, "observer_a": obs_a, "observer_b": obs_b, "assd_mm": dist})
        n_ok += 1

    dist_df = pd.DataFrame(distance_rows)
    feat_df = pd.DataFrame(feature_rows)
    dist_df.to_csv(DATA_DIR / "interobserver_surface_distance.csv", index=False)
    feat_df.to_csv(DATA_DIR / "interobserver_features_long.csv", index=False)
    print(f"\n{n_ok} patients processed, {n_skip} skipped")
    print(f"Mean inter-observer ASSD: {dist_df['assd_mm'].mean():.2f} +/- {dist_df['assd_mm'].std():.2f} mm "
          f"(n={len(dist_df)} pairs across {n_ok} patients)")

    # --- real per-feature relative change, all observer pairs, all patients
    feat_wide = feat_df.pivot_table(index=["patient", "observer"], columns="feature", values="value")
    real_rows = []
    for patient_id, sub in feat_wide.groupby(level=0):
        obs_ids = sub.index.get_level_values(1)
        for obs_a, obs_b in itertools.combinations(sorted(obs_ids), 2):
            va = sub.loc[(patient_id, obs_a)]
            vb = sub.loc[(patient_id, obs_b)]
            denom = np.maximum(np.abs(va), np.abs(vb)).clip(lower=1e-6)
            rel_change = (np.abs(va - vb) / denom)
            for feat_name, rc in rel_change.items():
                real_rows.append({"feature": feat_name, "rel_change": rc})
    real_feat_df = pd.DataFrame(real_rows)
    real_summary = real_feat_df.groupby("feature")["rel_change"].median().rename("real_interobserver_median_rel_change")
    real_summary.to_csv(DATA_DIR / "interobserver_feature_stability.csv")

    # --- compare against our synthetic -1mm-vs-+1mm perturbation numbers
    pert_df = pd.read_csv(DATA_DIR / "perturbation_features.csv")
    synth_summary = feature_stability_summary(pert_df, -1, 1).set_index("feature")["median_abs_rel_change"]
    synth_summary.name = "synthetic_perturbation_median_rel_change"

    compare = pd.concat([real_summary, synth_summary], axis=1).dropna()
    # Same exclusion as modeling.NON_FEATURE_COLS elsewhere in this project:
    # slice count is extraction-pipeline bookkeeping, not a radiological
    # feature -- see interpretation.py's docstring for the bug it caused
    # when this wasn't excluded from a model's feature matrix.
    compare = compare.drop(index="texture_n_slices", errors="ignore")
    from scipy.stats import spearmanr
    rho, p = spearmanr(compare["real_interobserver_median_rel_change"], compare["synthetic_perturbation_median_rel_change"])
    print(f"\nSpearman correlation (real inter-observer instability rank vs. synthetic perturbation instability rank): "
          f"rho={rho:.2f}, p={p:.4f}, n_features={len(compare)}")
    compare.to_csv(DATA_DIR / "interobserver_vs_synthetic_comparison.csv")
    print(compare.sort_values("real_interobserver_median_rel_change", ascending=False).to_string())

    build_figure(dist_df, compare, rho, p)


def build_figure(dist_df, compare, rho, p):
    fig = plt.figure(figsize=(11.5, 4.3))
    fig.patch.set_facecolor("white")
    fig.text(0.01, 0.98, "Fig. 8 | Synthetic perturbation validated against real multi-observer contours",
              fontsize=13, fontweight="bold", color=INK, ha="left", va="top")

    ax1 = fig.add_axes([0.07, 0.16, 0.38, 0.62])
    ax1.hist(dist_df["assd_mm"].dropna(), bins=18, color=BLUE, alpha=0.8, edgecolor="black", linewidth=0.8)
    for mm, ls in [(1, (0, (4, 2))), (2, (0, (1, 1)))]:
        ax1.axvline(mm, color=AMBER, linewidth=1.6, linestyle=ls)
        ax1.text(mm, ax1.get_ylim()[1] * 0.9, f" {mm}mm", fontsize=7, color=AMBER, rotation=90, va="top")
    mean_d = dist_df["assd_mm"].mean()
    ax1.axvline(mean_d, color=INK, linewidth=1.8)
    ax1.text(mean_d, ax1.get_ylim()[1] * 0.98, f" mean={mean_d:.2f}mm", fontsize=7.2, color=INK, va="top")
    ax1.set_xlabel("Real inter-observer boundary\ndisagreement (mm, ASSD)")
    ax1.set_ylabel("Pairwise comparisons")
    fig.text(0.07 - 0.02, 0.86, "a", fontsize=14, fontweight="bold", color=INK)

    # Rank, not raw value -- one extreme feature (firstorder_energy) on a
    # raw scale compresses all 26 other points into the corner and hides
    # exactly the relationship the correlation is about. Spearman rho is a
    # rank statistic, so ranks are what it measures.
    synth_rank = compare["synthetic_perturbation_median_rel_change"].rank(ascending=False)
    real_rank = compare["real_interobserver_median_rel_change"].rank(ascending=False)
    n_feat = len(compare)

    ax2 = fig.add_axes([0.58, 0.16, 0.38, 0.62])
    ax2.scatter(synth_rank, real_rank, s=32, color=GREEN, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
    lims = [0, n_feat + 1]
    ax2.plot(lims, lims, color="#c7cdd4", linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Synthetic perturbation instability rank\n(−1mm vs +1mm; 1 = least stable)")
    ax2.set_ylabel("Real inter-observer instability rank\n(5 observers; 1 = least stable)")
    ax2.text(0.05, 0.95, f"Spearman ρ={rho:.2f}\np={p:.4f}, n={n_feat} features", transform=ax2.transAxes,
              fontsize=7.5, color=INK, va="top", ha="left")
    for feat in ("firstorder_energy", "firstorder_mean_hu"):
        if feat in compare.index:
            ax2.annotate(feat.replace("firstorder_", ""), (synth_rank[feat], real_rank[feat]), fontsize=6.4,
                         color=SUBINK, textcoords="offset points", xytext=(5, -8))
    fig.text(0.58 - 0.02, 0.86, "b", fontsize=14, fontweight="bold", color=INK)

    out_png = FIG_DIR / "figure8_interobserver_validation.png"
    out_pdf = FIG_DIR / "figure8_interobserver_validation.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    except PermissionError as exc:
        print(f"Skipped PDF: {exc}")
    print(f"\nSaved {out_png}")


if __name__ == "__main__":
    main()
