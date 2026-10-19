"""Incremental value of radiomics over clinicopathological variables, and exploratory covariates.

Paper:    Fig. 2d, Fig. 7a-b, Suppl. Tables S5, S23, Suppl. Fig. S12
Pipeline: python pipeline.py run survival_tables   (stage 'discovery')

1. Incremental concordance staircase (their Fig. 6c): nested-CV
   concordance for clinical-alone -> +tumor volume -> +volume+radiomics,
   same protocol as every other nested-CV number in this paper, showing
   exactly how much the 26-feature radiomics panel adds once age, sex,
   smoking, histology, T stage, and chemotherapy are already accounted
   for.
2. Exploratory univariate Cox covariates (their "primary tumor location"
   and "oncogene mutation status" checks): right-upper-lobe location,
   EGFR-mutant, and KRAS-mutant status, each tested alone against OS and
   RFS. ALK is excluded (2 translocated patients in this cohort -- too
   few for a stable HR, the same reasoning modeling.py uses to exclude it
   as a classification target).
3. Net Reclassification Improvement (NRI) of adding the radiomics panel
   to the clinical+volume model, continuous NRI (van Calster/Pencina
   convention: sum of the fraction of events with an increased predicted
   risk minus the fraction with a decreased one, plus the mirror-image
   term for non-events), computed on the same out-of-fold predicted risk
   used for the staircase above -- not available in lifelines/scikit, so
   implemented directly here.

Run in the MAIN environment:
  python scripts/survival/run_survival_incremental_value.py

Output: data/processed/survival_staircase_{os,rfs}.csv
        data/processed/survival_exploratory_covariates.csv
        data/processed/survival_nri_{os,rfs}.csv
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import config, survival  # noqa: E402
from scripts.survival.run_survival_cox_tables import build_covariates  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
CLINICAL_COLS = ["age", "male", "ever_smoker", "adenocarcinoma", "t_stage", "chemo"]


def _merged_frame(endpoint):
    X_rad, y, case_ids, durations = survival.load_survival_dataset(endpoint=endpoint)
    rad_cols = list(X_rad.columns)
    rad_df = X_rad.copy()
    rad_df["case_id"] = case_ids
    rad_df["duration_days"] = durations["duration_days"].to_numpy()
    rad_df["event"] = durations["event"].to_numpy()

    cov = build_covariates()
    df = rad_df.merge(cov, on="case_id", how="inner").sort_values("case_id").reset_index(drop=True)
    return df, rad_cols


def staircase(endpoint):
    df, rad_cols = _merged_frame(endpoint)
    case_ids = df["case_id"].to_numpy()
    from sksurv.util import Surv
    y = Surv.from_arrays(event=df["event"].to_numpy().astype(bool), time=df["duration_days"].to_numpy())

    steps = {
        "Clinical alone": CLINICAL_COLS,
        "+ tumor volume": CLINICAL_COLS + ["tumor_volume_cc"],
        "+ radiomics (26 features)": CLINICAL_COLS + ["tumor_volume_cc"] + rad_cols,
    }
    rows = []
    print(f"\n=== {endpoint.upper()} concordance staircase (n={len(df)}, events={int(df['event'].sum())}) ===")
    for label, cols in steps.items():
        X = df[cols]
        scores, _alpha, _oof = survival.nested_cv_concordance(X, y, case_ids)
        print(f"  {label}: {scores.mean():.3f} +/- {scores.std():.3f}")
        rows.append({"endpoint": endpoint, "step": label, "n_features": len(cols),
                      "mean_concordance": float(scores.mean()), "std_concordance": float(scores.std())})
    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / f"survival_staircase_{endpoint}.csv", index=False)
    print(f"Saved -> {DATA_DIR / f'survival_staircase_{endpoint}.csv'}")
    return out


def exploratory_covariates():
    clin = pd.read_csv(config.CLINICAL_OUT)
    clin["rul"] = (clin["tumor_location"] == "RUL").astype(int)
    clin["egfr_mutant"] = clin["egfr_mutation_status"].map({"Mutant": 1, "Wildtype": 0})
    clin["kras_mutant"] = clin["kras_mutation_status"].map({"Mutant": 1, "Wildtype": 0})

    rows = []
    for endpoint, pred_file in [("os", "survival_predictions.csv"), ("rfs", "survival_rfs_predictions.csv")]:
        oof = pd.read_csv(DATA_DIR / pred_file)
        df = oof.merge(clin[["case_id", "rul", "egfr_mutant", "kras_mutant"]], on="case_id", how="left")
        for label, col in [("Right upper lobe vs. other", "rul"),
                            ("EGFR-mutant vs. wildtype", "egfr_mutant"),
                            ("KRAS-mutant vs. wildtype", "kras_mutant")]:
            sub = df.dropna(subset=[col])
            cph = CoxPHFitter()
            cph.fit(sub[[col, "duration_days", "event"]], duration_col="duration_days", event_col="event")
            s = cph.summary.loc[col]
            rows.append({
                "endpoint": endpoint, "variable": label, "n": len(sub), "n_positive": int(sub[col].sum()),
                "HR": float(np.exp(s["coef"])), "HR_lower": float(np.exp(s["coef lower 95%"])),
                "HR_upper": float(np.exp(s["coef upper 95%"])), "p": float(s["p"]),
            })
    out = pd.DataFrame(rows)
    print("\n=== Exploratory univariate covariates (OS and RFS) ===")
    print(out.to_string(index=False))
    out.to_csv(DATA_DIR / "survival_exploratory_covariates.csv", index=False)
    print(f"Saved -> {DATA_DIR / 'survival_exploratory_covariates.csv'}")
    return out


def nri(endpoint):
    """Continuous NRI, adding the 26-feature radiomics panel to a
    clinical+volume Cox model, computed on out-of-fold predicted risk
    from two nested-CV models fit under the identical protocol used
    throughout this paper (no new pipeline code, just two calls to the
    function already used for the staircase above)."""
    df, rad_cols = _merged_frame(endpoint)
    case_ids = df["case_id"].to_numpy()
    from sksurv.util import Surv
    y = Surv.from_arrays(event=df["event"].to_numpy().astype(bool), time=df["duration_days"].to_numpy())

    base_cols = CLINICAL_COLS + ["tumor_volume_cc"]
    full_cols = base_cols + rad_cols
    _scores_base, _a, oof_base = survival.nested_cv_concordance(df[base_cols], y, case_ids)
    _scores_full, _a, oof_full = survival.nested_cv_concordance(df[full_cols], y, case_ids)

    merged = oof_base.merge(oof_full, on="case_id", suffixes=("_base", "_full"))
    events = merged[merged["event_base"]]
    nonevents = merged[~merged["event_base"]]

    up_events = (events["risk_score_full"] > events["risk_score_base"]).mean()
    down_events = (events["risk_score_full"] < events["risk_score_base"]).mean()
    up_nonevents = (nonevents["risk_score_full"] > nonevents["risk_score_base"]).mean()
    down_nonevents = (nonevents["risk_score_full"] < nonevents["risk_score_base"]).mean()

    nri_events = up_events - down_events
    nri_nonevents = down_nonevents - up_nonevents
    total_nri = nri_events + nri_nonevents

    # Two-sided z-test approximation (Pencina et al. 2008) for the total NRI.
    n_e, n_ne = len(events), len(nonevents)
    se = np.sqrt((up_events + down_events) / n_e + (up_nonevents + down_nonevents) / n_ne)
    z = total_nri / se if se > 0 else np.nan
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan

    row = {
        "endpoint": endpoint, "n_events": n_e, "n_nonevents": n_ne,
        "nri_events": nri_events, "nri_nonevents": nri_nonevents, "total_nri": total_nri,
        "z": z, "p": p,
    }
    print(f"\n=== {endpoint.upper()} continuous NRI, +radiomics over clinical+volume ===")
    print(f"  events NRI={nri_events:.3f}, non-events NRI={nri_nonevents:.3f}, total NRI={total_nri:.3f}, p={p:.4f}")
    out = pd.DataFrame([row])
    out.to_csv(DATA_DIR / f"survival_nri_{endpoint}.csv", index=False)
    print(f"Saved -> {DATA_DIR / f'survival_nri_{endpoint}.csv'}")
    return out


def main():
    for endpoint in ["os", "rfs"]:
        staircase(endpoint)
    exploratory_covariates()
    for endpoint in ["os", "rfs"]:
        nri(endpoint)


if __name__ == "__main__":
    main()
