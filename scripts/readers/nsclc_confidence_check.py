"""NSCLC: does segmentation sensitivity predict reclassification by another
segmentation beyond model confidence?

Two settings, both without a reference label (the outcome is instability):
  readers  multi-reader NSCLC cohort: a reader's classification differs from
           at least one other radiation oncologist's (fixed models; 21 patients
           x 5 readers x 3 endpoints)
  ai       discovery cohort: the classification from the AI segmentation
           differs from the expert segmentation's (131 patients x 3 endpoints)

Per segmentation: confidence C = |score − threshold| and sensitivity
S = |score(+1 mm) − score(−1 mm)|, both log-transformed and standardized
within endpoint (scores of different endpoints are on different scales).
Pooled logistic model: outcome ~ endpoint + C + S. Odds ratio for S per s.d.
adjusted for C, with 95% intervals from 2,000 bootstrap resamples of patients.

Paper:    Results, Suppl. Tables S15, S18
Pipeline: python pipeline.py run confidence_nsclc   (stage 'ai')
Output:   data/processed/nsclc_confidence_check.json
Run:      python scripts/readers/nsclc_confidence_check.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[2]
P = ROOT / "data" / "processed"
EPS = 1e-6


def zlog(x):
    v = np.log(np.asarray(x, float) + EPS)
    return (v - v.mean()) / v.std()


def readers() -> pd.DataFrame:
    d = pd.read_csv(P / "reader_decisions.csv")
    w = d.pivot_table(index=["endpoint", "case_id", "reader"], columns="perturbation_mm", values="score").reset_index()
    rows = []
    for ep, g in w.groupby("endpoint"):
        thr = g[0].median()  # the same threshold used for the readers' classifications
        g = g.assign(cls=(g[0] >= thr).astype(int))
        n_pos = g.groupby("case_id").cls.transform("sum")
        n = g.groupby("case_id").cls.transform("size")
        g["outcome"] = (((g.cls == 1) & (n_pos < n)) | ((g.cls == 0) & (n_pos > 0))).astype(int)
        g["zC"], g["zS"] = zlog((g[0] - thr).abs()), zlog((g[1] - g[-1]).abs())
        rows.append(g)
    return pd.concat(rows)


def ai() -> pd.DataFrame:
    d = pd.read_csv(P / "bamf_paired_predictions.csv").dropna(subset=["p_expert", "p_ai", "p_-1mm", "p_+1mm"])
    rows = []
    for ep, g in d.groupby("endpoint"):
        g = g.copy()
        g["outcome"] = ((g.p_ai >= g.threshold) != (g.p_expert >= g.threshold)).astype(int)
        g["zC"], g["zS"] = zlog((g.p_expert - g.threshold).abs()), zlog((g["p_+1mm"] - g["p_-1mm"]).abs())
        rows.append(g)
    return pd.concat(rows)


def or_s(df):
    X = np.c_[pd.get_dummies(df.endpoint, drop_first=True, dtype=float).to_numpy(), df.zC, df.zS]
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, df.outcome)
    return float(np.exp(m.coef_[0][-1])), float(np.exp(m.coef_[0][-2]))


def analyse(df, n_boot=2000):
    s, c = or_s(df)
    groups = dict(tuple(df.groupby("case_id")))
    ids = np.array(list(groups))
    rng = np.random.default_rng(0)
    boot = [or_s(pd.concat([groups[i] for i in rng.choice(ids, len(ids))])) for _ in range(n_boot)]
    b = np.array(boot)
    return {"n_segmentations": int(len(df)), "n_patients": int(len(ids)), "outcome_rate": float(df.outcome.mean()),
            "or_S_per_sd_adj_C": s, "or_S_ci95": [float(np.percentile(b[:, 0], 2.5)), float(np.percentile(b[:, 0], 97.5))],
            "or_C_per_sd_adj_S": c, "or_C_ci95": [float(np.percentile(b[:, 1], 2.5)), float(np.percentile(b[:, 1], 97.5))]}


def main():
    res = {"readers": analyse(readers()), "ai": analyse(ai())}
    (P / "nsclc_confidence_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
