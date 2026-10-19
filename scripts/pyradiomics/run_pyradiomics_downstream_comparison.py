"""PyRadiomics: model performance at every segmentation adjustment.

Paper:    Suppl. Tables S13-S14, Suppl. Fig. S6
Pipeline: python pipeline.py run pyradiomics_analyses   (stage 'pyradiomics')

Caveat stated up front (and in the write-up): the PyRadiomics table only
has the 22 features with a clean 1:1 conceptual mapping to this study's
own 26-feature panel (Methods, "PyRadiomics cross-check"), so this is a
comparison of "the same classifier, cohort, and folds, on two different
feature sets" as well as "two different extractors" -- both differences
are real and reported as such, not collapsed into one.

Run: python scripts/pyradiomics/run_pyradiomics_downstream_comparison.py
Output: data/processed/pyradiomics_downstream_comparison.csv
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import modeling, survival  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
MM_LEVELS = (-2, -1, 0, 1, 2)
CLASSIFICATION_TARGETS = ("recurrence", "egfr")


def main():
    pr = pd.read_csv(DATA_DIR / "pyradiomics_perturbation_features.csv")
    rows = []

    for target in CLASSIFICATION_TARGETS:
        for mm in MM_LEVELS:
            table = pr[pr["perturbation_mm"] == mm].drop(columns=["perturbation_mm"]).reset_index(drop=True)
            suffix = f"{target}__pyrad_{mm:+g}mm".replace("+", "p").replace("-", "m")
            print(f"\n=== {target} | PyRadiomics | {mm:+g} mm ===")
            aucs, _oof = modeling.run(
                target=target, feature_set="radiomics", radiomics_path=table,
                result_suffix=suffix, save=True,
            )
            rows.append({
                "paradigm": "classification", "target": target, "extractor": "pyradiomics",
                "perturbation_mm": mm, "mean_score": float(aucs.mean()), "std_score": float(aucs.std()),
            })

    for mm in MM_LEVELS:
        table = pr[pr["perturbation_mm"] == mm].drop(columns=["perturbation_mm"]).reset_index(drop=True)
        suffix = f"survival__pyrad_{mm:+g}mm".replace("+", "p").replace("-", "m")
        print(f"\n=== survival | PyRadiomics | {mm:+g} mm ===")
        scores = survival.run(radiomics_path=table, result_suffix=suffix, save=True)
        rows.append({
            "paradigm": "survival", "target": "recurrence_survival", "extractor": "pyradiomics",
            "perturbation_mm": mm, "mean_score": float(scores.mean()), "std_score": float(scores.std()),
        })

    pyrad_df = pd.DataFrame(rows)

    # side by side with the original custom-extractor numbers already saved
    custom_rows = []
    for target in CLASSIFICATION_TARGETS:
        for mm in MM_LEVELS:
            if mm == 0:
                fname = f"modeling_results_{target}"
            else:
                tag = {-2: "perturb_m2mm", -1: "perturb_m1mm", 1: "perturb_p1mm", 2: "perturb_p2mm"}[mm]
                fname = f"modeling_results_{target}__{tag}"
            path = DATA_DIR / f"{fname}.csv"
            if path.exists():
                aucs = pd.read_csv(path)["auc"]
                custom_rows.append({"paradigm": "classification", "target": target, "extractor": "custom",
                                     "perturbation_mm": mm, "mean_score": float(aucs.mean()), "std_score": float(aucs.std())})
    for mm in MM_LEVELS:
        if mm == 0:
            fname = "survival_results"
        else:
            tag = {-2: "perturb_m2mm", -1: "perturb_m1mm", 1: "perturb_p1mm", 2: "perturb_p2mm"}[mm]
            fname = f"survival__{tag}_results"
        path = DATA_DIR / f"{fname}.csv"
        if path.exists():
            scores = pd.read_csv(path)["concordance"]
            custom_rows.append({"paradigm": "survival", "target": "recurrence_survival", "extractor": "custom",
                                 "perturbation_mm": mm, "mean_score": float(scores.mean()), "std_score": float(scores.std())})

    custom_df = pd.DataFrame(custom_rows)
    combined = pd.concat([custom_df, pyrad_df], ignore_index=True)
    out_path = DATA_DIR / "pyradiomics_downstream_comparison.csv"
    combined.to_csv(out_path, index=False)

    print("\n\n================ SWING COMPARISON: custom extractor vs. PyRadiomics ================")
    swing = combined.groupby(["paradigm", "target", "extractor"])["mean_score"].agg(lambda s: s.max() - s.min()).reset_index()
    swing = swing.rename(columns={"mean_score": "swing"})
    print(swing.to_string(index=False))
    swing_path = DATA_DIR / "pyradiomics_downstream_swing_summary.csv"
    swing.to_csv(swing_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {swing_path}")


if __name__ == "__main__":
    main()
