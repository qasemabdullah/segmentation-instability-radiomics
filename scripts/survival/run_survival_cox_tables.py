"""Univariable and multivariable Cox models and survival by risk tertile.

Paper:    Fig. 2d, Fig. 7a-b, Suppl. Tables S5, S23, Suppl. Fig. S12
Pipeline: python pipeline.py run survival_tables   (stage 'discovery')

Reuses the out-of-fold risk scores already saved by
survival.py (survival_predictions.csv, survival_rfs_predictions.csv) as
the radiomics risk score -- every patient's score comes from a fold that
never saw that patient during training, so this is not circular.

Run in the MAIN environment:
  python scripts/survival/run_survival_cox_tables.py

Output: data/processed/cox_univariate_{os,rfs}.csv
        data/processed/cox_multivariate_{os,rfs}.csv
        data/processed/km_tertile_{os,rfs}.csv (group assignment + logrank)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import config  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
ENDPOINTS = {"os": "survival_predictions.csv", "rfs": "survival_rfs_predictions.csv"}
T_STAGE_MAP = {"Tis": 0, "T1a": 1, "T1b": 1, "T2a": 2, "T2b": 2, "T3": 3, "T4": 4}


def build_covariates():
    clin = pd.read_csv(config.CLINICAL_OUT)
    rad = pd.read_csv(config.RADIOMICS_OUT)
    df = clin.merge(rad[["case_id", "shape_volume_mm3"]], on="case_id", how="inner")

    df["age"] = df["age_at_histological_diagnosis"]
    df["male"] = (df["gender"] == "Male").astype(int)
    df["ever_smoker"] = (df["smoking_status"] != "Nonsmoker").astype(int)
    df["adenocarcinoma"] = (df["histology"] == "Adenocarcinoma").astype(int)
    df["t_stage"] = df["pathological_t_stage"].map(T_STAGE_MAP)
    df["chemo"] = (df["chemotherapy"] == "Yes").astype(int)
    df["tumor_volume_cc"] = df["shape_volume_mm3"] / 1000.0
    return df[["case_id", "age", "male", "ever_smoker", "adenocarcinoma", "t_stage", "chemo", "tumor_volume_cc"]]


def run_endpoint(name, pred_file):
    print(f"\n{'=' * 70}\n{name.upper()}\n{'=' * 70}")
    oof = pd.read_csv(DATA_DIR / pred_file)
    cov = build_covariates()
    df = oof.merge(cov, on="case_id", how="inner")
    df = df.dropna(subset=["t_stage"])  # 0 missing within the 144-patient cohort; defensive only
    df["risk_score_z"] = (df["risk_score"] - df["risk_score"].mean()) / df["risk_score"].std()
    print(f"n={len(df)}, events={int(df['event'].sum())}")

    covariates = {
        "Radiomics risk score (per s.d.)": "risk_score_z",
        "Age (per year)": "age",
        "Male vs. female": "male",
        "Ever- vs. never-smoker": "ever_smoker",
        "Adenocarcinoma vs. other histology": "adenocarcinoma",
        "Pathological T stage (ordinal)": "t_stage",
        "Chemotherapy vs. none": "chemo",
        "Tumor volume (per cm³)": "tumor_volume_cc",
    }

    uni_rows = []
    for label, col in covariates.items():
        cph = CoxPHFitter()
        cph.fit(df[[col, "duration_days", "event"]], duration_col="duration_days", event_col="event")
        s = cph.summary.loc[col]
        uni_rows.append({
            "variable": label, "HR": np.exp(s["coef"]),
            "HR_lower": np.exp(s["coef lower 95%"]), "HR_upper": np.exp(s["coef upper 95%"]),
            "p": s["p"],
        })
    uni = pd.DataFrame(uni_rows)
    print("\nUnivariate:")
    print(uni.to_string(index=False))
    uni.to_csv(DATA_DIR / f"cox_univariate_{name}.csv", index=False)

    multi_cols = list(covariates.values())
    cph_multi = CoxPHFitter()
    cph_multi.fit(df[multi_cols + ["duration_days", "event"]], duration_col="duration_days", event_col="event")
    multi_rows = []
    for label, col in covariates.items():
        s = cph_multi.summary.loc[col]
        multi_rows.append({
            "variable": label, "HR": np.exp(s["coef"]),
            "HR_lower": np.exp(s["coef lower 95%"]), "HR_upper": np.exp(s["coef upper 95%"]),
            "p": s["p"],
        })
    multi = pd.DataFrame(multi_rows)
    print("\nMultivariate (mutually adjusted):")
    print(multi.to_string(index=False))
    multi.to_csv(DATA_DIR / f"cox_multivariate_{name}.csv", index=False)

    # sensitivity analysis: chemotherapy follows from prognosis, so refit without it
    nc_cols = [c for c in multi_cols if c != "chemo"]
    cph_nc = CoxPHFitter()
    cph_nc.fit(df[nc_cols + ["duration_days", "event"]], duration_col="duration_days", event_col="event")
    nc = pd.DataFrame([{"variable": label, "HR": np.exp(cph_nc.summary.loc[col, "coef"]),
                        "HR_lower": np.exp(cph_nc.summary.loc[col, "coef lower 95%"]),
                        "HR_upper": np.exp(cph_nc.summary.loc[col, "coef upper 95%"]), "p": cph_nc.summary.loc[col, "p"]}
                       for label, col in covariates.items() if col != "chemo"])
    print("\nMultivariate without chemotherapy:")
    print(nc.to_string(index=False))
    nc.to_csv(DATA_DIR / f"cox_multivariate_{name}_without_chemotherapy.csv", index=False)

    # Tertile risk-group stratification (Low/Intermediate/High), mirroring
    # a 3-group KM design rather than this paper's own usual median split.
    df["risk_tertile"] = pd.qcut(df["risk_score"], 3, labels=["Low", "Intermediate", "High"])
    group_counts = df["risk_tertile"].value_counts().to_dict()
    event_counts = df.groupby("risk_tertile", observed=True)["event"].sum().to_dict()
    result = multivariate_logrank_test(df["duration_days"], df["risk_tertile"], df["event"])
    print(f"\nTertile risk groups: n={group_counts}, events={event_counts}")
    print(f"Multivariate log-rank p={result.p_value:.4g}")

    df[["case_id", "duration_days", "event", "risk_score", "risk_tertile"]].to_csv(
        DATA_DIR / f"km_tertile_{name}.csv", index=False
    )
    with open(DATA_DIR / f"km_tertile_{name}_logrank.txt", "w") as f:
        f.write(f"n={group_counts}\nevents={event_counts}\np={result.p_value:.6g}\n")

    return uni, multi, result.p_value


def main():
    for name, pred_file in ENDPOINTS.items():
        run_endpoint(name, pred_file)


if __name__ == "__main__":
    main()
