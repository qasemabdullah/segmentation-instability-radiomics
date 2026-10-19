"""Mitigation: models restricted to the most or least stable features.

Paper:    Fig. 6d, Suppl. Fig. S11, Suppl. Table S21
Pipeline: python pipeline.py run stability_filtering   (stage 'ai')

Method: rank all 26 radiomics features by the same perturbation-instability
score already used for Fig. 3b (median |relative change| between -1mm and
+1mm), split into the 13 most stable and 13 most unstable, and re-run the
identical nested-CV pipeline -- for recurrence, EGFR, and the Cox survival
model -- on EACH half separately, at every perturbation level (-2, -1, 0,
+1, +2 mm). If stability-filtering works as a remedy, the "stable" half's
reported performance should range much less across perturbation levels
than the "unstable" half's, and less than the full 26-feature set already
measured (Fig. 3c-e).

Also cross-checks the synthetic-derived stable/unstable split against the
real interobserver instability ranking (already computed for Fig. 4c), to
see whether "stable by our synthetic perturbation" and "stable by real
inter-observer disagreement" pick out the same features.

Run: python scripts/mitigation/run_stability_filtering_experiment.py
Output: data/processed/stability_filtering_results.csv
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from nsclc_pipeline import config, modeling, perturbation, survival  # noqa: E402

MM_LEVELS = (-2, -1, 0, 1, 2)
CLASSIFICATION_TARGETS = ("recurrence", "egfr")


def build_feature_splits():
    pert = pd.read_csv(config.PROCESSED_DIR / "perturbation_features.csv")
    stability = perturbation.feature_stability_summary(pert, -1, 1)
    # Same exclusion as modeling.NON_FEATURE_COLS elsewhere in this project
    # -- texture_n_slices is extraction-pipeline bookkeeping, not a
    # radiological feature (this is exactly the column the paper's own
    # SHAP analysis flagged; it must not silently end up in a model input
    # here either).
    stability = stability[~stability["feature"].isin(modeling.NON_FEATURE_COLS)].reset_index(drop=True)
    n = len(stability)
    half = n // 2
    unstable = stability.iloc[:half]["feature"].tolist()   # highest instability first
    stable = stability.iloc[half:]["feature"].tolist()      # lowest instability

    # Cross-check against the real interobserver ranking already computed
    # for Fig. 4c, if available.
    iob_path = config.PROCESSED_DIR / "interobserver_feature_stability.csv"
    if iob_path.exists():
        iob = pd.read_csv(iob_path).set_index("feature")["real_interobserver_median_rel_change"]
        common = [f for f in stability["feature"] if f in iob.index]
        real_rank = iob[common].rank()
        stable_real_median_rank = real_rank[[f for f in stable if f in real_rank.index]].median()
        unstable_real_median_rank = real_rank[[f for f in unstable if f in real_rank.index]].median()
        print(f"Median REAL interobserver instability rank (1=most unstable of {len(common)}):")
        print(f"  synthetic-'stable' half:   {stable_real_median_rank:.1f}")
        print(f"  synthetic-'unstable' half: {unstable_real_median_rank:.1f}")
        print("  (if stability transfers, the synthetic-stable half should rank as LESS unstable in real data too)\n")

    print("Stable half (13 features, lowest synthetic instability):")
    print("  " + ", ".join(stable))
    print("\nUnstable half (13 features, highest synthetic instability):")
    print("  " + ", ".join(unstable))
    return stable, unstable


def _feature_subset_table(pert_df, mm, features):
    sub = pert_df[pert_df["perturbation_mm"] == mm]
    return sub[["case_id"] + features].reset_index(drop=True)


def run_classification_arm(pert_df, features, condition_name):
    rows = []
    for target in CLASSIFICATION_TARGETS:
        for mm in MM_LEVELS:
            table = _feature_subset_table(pert_df, mm, features)
            suffix = f"{target}__stabfilt_{condition_name}_{mm:+g}mm".replace("+", "p").replace("-", "m")
            print(f"\n=== {target} | {condition_name} half | {mm:+g} mm ===")
            aucs, _oof = modeling.run(
                target=target, feature_set="radiomics", radiomics_path=table,
                result_suffix=suffix, save=True,
            )
            rows.append({
                "paradigm": "classification", "target": target, "condition": condition_name,
                "perturbation_mm": mm, "mean_score": float(aucs.mean()), "std_score": float(aucs.std()),
            })
    return rows


def run_survival_arm(pert_df, features, condition_name):
    rows = []
    for mm in MM_LEVELS:
        table = _feature_subset_table(pert_df, mm, features)
        suffix = f"survival__stabfilt_{condition_name}_{mm:+g}mm".replace("+", "p").replace("-", "m")
        print(f"\n=== survival | {condition_name} half | {mm:+g} mm ===")
        scores = survival.run(radiomics_path=table, result_suffix=suffix, save=True)
        rows.append({
            "paradigm": "survival", "target": "recurrence_survival", "condition": condition_name,
            "perturbation_mm": mm, "mean_score": float(scores.mean()), "std_score": float(scores.std()),
        })
    return rows


def main():
    pert_df = pd.read_csv(config.PROCESSED_DIR / "perturbation_features.csv")
    stable, unstable = build_feature_splits()

    all_rows = []
    for features, cond in [(stable, "stable"), (unstable, "unstable")]:
        all_rows += run_classification_arm(pert_df, features, cond)
        all_rows += run_survival_arm(pert_df, features, cond)

    summary = pd.DataFrame(all_rows)
    out_path = config.PROCESSED_DIR / "stability_filtering_results.csv"
    summary.to_csv(out_path, index=False)

    print("\n\n================ SWING SUMMARY (max - min across -2..+2mm) ================")
    swing = (
        summary.groupby(["paradigm", "target", "condition"])["mean_score"]
        .agg(lambda s: s.max() - s.min())
        .reset_index()
        .rename(columns={"mean_score": "swing"})
    )
    print(swing.to_string(index=False))
    swing_path = config.PROCESSED_DIR / "stability_filtering_swing_summary.csv"
    swing.to_csv(swing_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {swing_path}")


if __name__ == "__main__":
    main()
