"""Segmentation-perturbation robustness study.

Measures how much the radiomic features and the model AUC change when the
tumor segmentation is moved inwards or outwards by 1-2 mm, a difference of
the size seen between two experts contouring the same tumor.

Reuses radiomics.py's feature-computation functions unchanged; the only
new piece here is generating the perturbed masks themselves.
"""

from __future__ import annotations

import SimpleITK as sitk
import numpy as np
import pandas as pd

from . import config, radiomics, viz

DEFAULT_MM_LEVELS = (-2, -1, 0, 1, 2)


from .mask_shift import FINE_MM, perturb_mask  # noqa: E402,F401  (single implementation)


def extract_perturbed_features(case_id: str, ct_array, ct_image, base_mask, mm: float) -> dict | None:
    pmask = perturb_mask(base_mask, mm)
    mask_array = sitk.GetArrayFromImage(pmask)
    if mask_array.sum() == 0:
        print(f"{case_id} mm={mm}: mask is empty after perturbation, skipping")
        return None

    tumor_vals = ct_array[mask_array > 0]
    feats = {"case_id": case_id, "perturbation_mm": mm}
    feats.update(radiomics.shape_features(pmask))
    feats.update(radiomics.first_order_features(tumor_vals))
    feats.update(radiomics.texture_features(ct_array, mask_array))
    return feats


def run_perturbation_study(
    case_ids=None, mm_levels=DEFAULT_MM_LEVELS, progress_cb=None, out_path=None
) -> pd.DataFrame:
    """For each patient x each perturbation level, recompute the full
    radiomics feature set from a boundary-shifted mask. Uses the CT/mask
    already cached locally by radiomics.py -- no new downloads needed for
    patients whose features have already been extracted once."""
    out_path = out_path or config.PROCESSED_DIR / "perturbation_features.csv"
    series_index = pd.read_csv(config.SERIES_INDEX_OUT)
    seg_case_ids = sorted(series_index[series_index["modality"] == "SEG"]["case_id"].unique())
    targets = case_ids or seg_case_ids

    rows = []
    for i, case_id in enumerate(targets, start=1):
        if progress_cb:
            progress_cb(i, len(targets), case_id)
        print(f"[{i}/{len(targets)}] {case_id}")
        try:
            ct_dir, mask_path, _, _ = radiomics.resolve_ct_and_mask(case_id, series_index)
            ct_image = viz.load_ct_volume(ct_dir)
            base_mask = viz.load_mask_on_ct(mask_path, ct_image)
            ct_array = sitk.GetArrayFromImage(ct_image)
            for mm in mm_levels:
                feats = extract_perturbed_features(case_id, ct_array, ct_image, base_mask, mm)
                if feats is not None:
                    rows.append(feats)
        except Exception as e:  # noqa: BLE001 -- keep going across the batch
            print(f"FAILED {case_id}: {type(e).__name__}: {e}")

        if rows:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


def feature_stability_summary(perturbation_df: pd.DataFrame, mm_a: float, mm_b: float) -> pd.DataFrame:
    """For each feature, the median absolute relative change between two
    perturbation levels (e.g. -1mm vs +1mm) across all patients that have
    both -- a per-feature stability score, not tied to any one target."""
    a = perturbation_df[perturbation_df["perturbation_mm"] == mm_a].set_index("case_id")
    b = perturbation_df[perturbation_df["perturbation_mm"] == mm_b].set_index("case_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    feature_cols = [c for c in a.columns if c != "perturbation_mm"]
    rows = []
    for col in feature_cols:
        va, vb = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
        denom = np.maximum(np.abs(va), 1e-6)
        rel_change = np.abs(va - vb) / denom
        rows.append(
            {
                "feature": col,
                "median_abs_rel_change": float(np.median(rel_change)),
                "mean_abs_rel_change": float(np.mean(rel_change)),
                "n_patients": len(common),
            }
        )
    return pd.DataFrame(rows).sort_values("median_abs_rel_change", ascending=False).reset_index(drop=True)


def run_auc_comparison(
    targets=("recurrence", "egfr"),
    mm_levels=(-1, 1),
    outer_folds: int = 5,
    inner_folds: int = 5,
    random_state: int = 0,
    perturbation_path=None,
) -> pd.DataFrame:
    """Re-run the same nested-CV pipeline used for the real radiomics
    table, but feeding in ONLY the eroded (or ONLY the dilated) feature
    table in its place -- same patients, same model, same everything
    except which tumor boundary the features came from. Compares the
    resulting AUC to the original (unperturbed) result already on disk,
    if one exists -- run modeling.run() for that target/feature_set
    first if it doesn't."""
    from . import modeling

    perturbation_df = pd.read_csv(perturbation_path or config.PROCESSED_DIR / "perturbation_features.csv")

    rows = []
    for target in targets:
        original_path = config.PROCESSED_DIR / f"modeling_results_{target}.csv"
        original_auc = float(pd.read_csv(original_path)["auc"].mean()) if original_path.exists() else None

        for mm in mm_levels:
            sub = perturbation_df[perturbation_df["perturbation_mm"] == mm].drop(columns=["perturbation_mm"])
            print(f"\n--- {target} @ perturbation {mm:+g}mm (n={len(sub)} patients) ---")
            aucs, _oof = modeling.run(
                target=target,
                feature_set="radiomics",
                radiomics_path=sub,
                outer_folds=outer_folds,
                inner_folds=inner_folds,
                random_state=random_state,
                result_suffix=f"{target}__perturb_{mm:+g}mm".replace("+", "p").replace("-", "m"),
                save=True,
            )
            rows.append(
                {
                    "target": target,
                    "perturbation_mm": mm,
                    "mean_auc": float(aucs.mean()),
                    "std_auc": float(aucs.std()),
                    "original_auc": original_auc,
                    "delta_from_original": (float(aucs.mean()) - original_auc) if original_auc is not None else None,
                }
            )

    summary = pd.DataFrame(rows)
    out_path = config.PROCESSED_DIR / "perturbation_auc_comparison.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nSaved -> {out_path}")
    return summary


def run_survival_perturbation_comparison(
    mm_levels=(-1, 1),
    outer_folds: int = 5,
    inner_folds: int = 5,
    random_state: int = 0,
    perturbation_path=None,
) -> pd.DataFrame:
    """Same idea as run_auc_comparison, for the Cox survival model: does
    concordance also swing under a boundary shift that's smaller than
    routine inter-observer contouring variation, the same way binary
    classification AUC already does?"""
    from . import survival

    perturbation_df = pd.read_csv(perturbation_path or config.PROCESSED_DIR / "perturbation_features.csv")

    original_path = config.PROCESSED_DIR / "survival_results.csv"
    original_conc = float(pd.read_csv(original_path)["concordance"].mean()) if original_path.exists() else None

    rows = []
    for mm in mm_levels:
        sub = perturbation_df[perturbation_df["perturbation_mm"] == mm].drop(columns=["perturbation_mm"])
        print(f"\n--- survival @ perturbation {mm:+g}mm (n={len(sub)} patients) ---")
        suffix = f"survival__perturb_{mm:+g}mm".replace("+", "p").replace("-", "m")
        scores = survival.run(
            outer_folds=outer_folds, inner_folds=inner_folds, random_state=random_state, radiomics_path=sub,
            result_suffix=suffix, save=True,
        )
        rows.append(
            {
                "perturbation_mm": mm,
                "mean_concordance": float(scores.mean()),
                "std_concordance": float(scores.std()),
                "original_concordance": original_conc,
                "delta_from_original": (float(scores.mean()) - original_conc) if original_conc is not None else None,
            }
        )

    summary = pd.DataFrame(rows)
    out_path = config.PROCESSED_DIR / "perturbation_survival_comparison.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n{summary.to_string(index=False)}")
    print(f"\nSaved -> {out_path}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mm-levels", type=float, nargs="+", default=list(DEFAULT_MM_LEVELS))
    args = parser.parse_args()

    df = run_perturbation_study(mm_levels=tuple(args.mm_levels))
    print(f"\nExtracted {len(df)} (patient, perturbation) feature rows -> {config.PROCESSED_DIR / 'perturbation_features.csv'}")
