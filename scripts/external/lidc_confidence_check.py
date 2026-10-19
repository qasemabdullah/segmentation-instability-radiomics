"""LIDC-IDRI: does segmentation sensitivity add information beyond model confidence?

A prediction close to the decision threshold is expected to change with a
small change in the segmentation, so the ±1-mm check might only rediscover
borderline predictions. For every reader's segmentation (threshold 0.5, as in
the main analysis) this script relates disagreement with the consensus
reference label ("error") to:

  confidence   C = |p(0 mm) − 0.5|                       distance from the threshold
  sensitivity  S = |p(+1 mm) − p(−1 mm)|                 continuous segmentation sensitivity
  flag         classification at −1, 0, +1 mm not all equal (binary ±1-mm check)

1. Logistic models error ~ C, error ~ C + S and error ~ C + flag (predictors
   standardized; S and C log-transformed because both are right-skewed):
   odds ratio for S or the flag adjusted for C, and the AUC for identifying
   errors with C alone and with C + S.
2. Within confidence strata (quintiles of C), error rate with and without the
   flag, to show the association is not explained by proximity to the threshold.
3. Selective prediction (risk-coverage): withhold the 10-50% of predictions
   ranked least trustworthy by (a) low confidence, (b) high sensitivity,
   (c) both (sum of the two ranks; nothing is fitted, so no optimism), and
   report the error rate among the predictions retained.
4. Flag sensitivity, specificity and positive predictive value for errors.

All 95% intervals come from 1,000 bootstrap resamples of patients, which keeps
a patient's correlated readers, nodules and cross-validation repetitions
together. "Error" means disagreement with the prespecified consensus rating
of the radiologists (median rating ≥ 4 malignant, ≤ 2 benign), not a
pathological diagnosis.

Paper:    Fig. 5a-f, Suppl. Table S17, Suppl. Fig. S9
Pipeline: python pipeline.py run lidc_models   (stage 'external')
Output:   data/external/lidc/lidc_confidence_check.json, lidc_risk_coverage.csv
Run:      python scripts/external/lidc_confidence_check.py [n_boot]
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "data" / "external" / "lidc"
COVERAGE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
EPS = 1e-4


def outlines() -> pd.DataFrame:
    """One row per seed x nodule x reader: confidence, sensitivity, flag and error."""
    o = pd.read_csv(D / "lidc_oof.csv.gz")
    w = (o.pivot_table(index=["seed", "PatientID", "nodule", "reader", "y"], columns="perturbation_mm", values="p")
         .dropna().reset_index())
    c = {k: (w[k] >= 0.5).astype(int) for k in (-1, 0, 1)}
    w["C"] = (w[0] - 0.5).abs()
    w["S"] = (w[1] - w[-1]).abs()
    w["flag"] = ((c[-1] != c[0]) | (c[1] != c[0])).astype(int)
    w["error"] = (c[0] != w.y).astype(int)
    # classification differs from at least one other reader of the same nodule (no reference label needed)
    grp = w.groupby(["seed", "PatientID", "nodule"])["reader"].transform("size")
    pos = c[0].groupby([w.seed, w.PatientID, w.nodule]).transform("sum")
    w["disagree"] = (((c[0] == 1) & (pos < grp)) | ((c[0] == 0) & (pos > 0))).astype(int)
    return w


def zlog(x):
    v = np.log(x + EPS)
    return (v - v.mean()) / v.std()


def design(w, use_s):
    """Confidence enters flexibly (natural spline of log C), so the test of S
    is not driven by a misspecified confidence term."""
    from sklearn.preprocessing import SplineTransformer
    xc = SplineTransformer(n_knots=5, degree=3).fit_transform(zlog(w.C.to_numpy())[:, None])
    return np.c_[xc, zlog(w.S.to_numpy())] if use_s else xc


def fit(X, y):
    return LogisticRegression(C=1e6, max_iter=5000).fit(X, y)


def oof_auc(w, outcome, use_s, folds=5):
    """AUC for identifying the outcome, on held-out patients (grouped 5-fold)."""
    from sklearn.model_selection import GroupKFold
    X, y, g = design(w, use_s), w[outcome].to_numpy(), w.PatientID.to_numpy()
    p = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(X, y, g):
        p[te] = fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, p))


def models(w):
    out = {}
    for outcome in ("error", "disagree"):
        y = w[outcome].to_numpy()
        m_cs = fit(design(w, True), y)
        m_cf = fit(np.c_[design(w, False), w.flag.to_numpy()], y)
        a_c, a_cs = oof_auc(w, outcome, False), oof_auc(w, outcome, True)
        out.update({f"{outcome}_or_S_per_sd_adj_C": float(np.exp(m_cs.coef_[0][-1])),
                    f"{outcome}_or_flag_adj_C": float(np.exp(m_cf.coef_[0][-1])),
                    f"{outcome}_auc_C": a_c, f"{outcome}_auc_C_plus_S": a_cs, f"{outcome}_auc_gain": a_cs - a_c})
    return out


def risk_coverage(w):
    rank_low_conf = w.C.rank(ascending=True)        # low confidence -> withhold first
    rank_high_s = w.S.rank(ascending=False)         # high sensitivity -> withhold first
    scores = {"confidence": -w.C.to_numpy(), "sensitivity": w.S.to_numpy(),
              "combined": -(rank_low_conf + rank_high_s).to_numpy()}
    rows = []
    for name, s in scores.items():
        order = np.argsort(-s, kind="stable")      # most suspect first
        err = w.error.to_numpy()[order]
        for cov in COVERAGE:
            keep = err[int(round((1 - cov) * len(err))):]
            rows.append({"strategy": name, "coverage": cov, "error_retained": float(keep.mean())})
    return pd.DataFrame(rows)


def flag_accuracy(w):
    f, e = w.flag.astype(bool), w.error.astype(bool)
    return {"sensitivity": float((f & e).sum() / e.sum()), "specificity": float((~f & ~e).sum() / (~e).sum()),
            "ppv": float((f & e).sum() / f.sum()), "npv": float((~f & ~e).sum() / (~f).sum()),
            "flag_rate": float(f.mean()), "error_rate": float(e.mean())}


def strata(w):
    q = pd.qcut(w.C, 5, labels=False, duplicates="drop")
    out = []
    for k, g in w.groupby(q):
        out.append({"confidence_quintile": int(k) + 1, "n": int(len(g)),
                    "error_if_flagged": float(g.error[g.flag == 1].mean()) if (g.flag == 1).any() else None,
                    "error_if_not_flagged": float(g.error[g.flag == 0].mean()) if (g.flag == 0).any() else None,
                    "flag_rate": float(g.flag.mean())})
    return out


def summary(w):
    rc = risk_coverage(w)
    out = {**models(w), **{f"flag_{k}": v for k, v in flag_accuracy(w).items()}}
    for _, r in rc.iterrows():
        out[f"rc_{r.strategy}_{int(round(r.coverage * 100))}"] = r.error_retained
    return out, rc


def main(n_boot=1000):
    w = outlines()
    point, rc = summary(w)
    groups = dict(tuple(w.groupby("PatientID")))
    pts = np.array(list(groups))
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        wb = pd.concat([groups[p] for p in rng.choice(pts, len(pts), replace=True)], ignore_index=True)
        boots.append(summary(wb)[0])
    B = pd.DataFrame(boots)
    ci = {k: [float(B[k].quantile(0.025)), float(B[k].quantile(0.975))] for k in point}
    rc["ci_low"] = [ci[f"rc_{s}_{int(round(c * 100))}"][0] for s, c in zip(rc.strategy, rc.coverage)]
    rc["ci_high"] = [ci[f"rc_{s}_{int(round(c * 100))}"][1] for s, c in zip(rc.strategy, rc.coverage)]
    rc.to_csv(D / "lidc_risk_coverage.csv", index=False)
    res = {"n_outlines": int(len(w)), "n_patients": int(w.PatientID.nunique()), "bootstrap": f"{n_boot} patient resamples",
           "point": point, "ci95": ci, "confidence_strata": strata(w)}
    (D / "lidc_confidence_check.json").write_text(json.dumps(res, indent=2))
    for k in (*[f"{o}_{m}" for o in ("error", "disagree") for m in ("or_S_per_sd_adj_C", "or_flag_adj_C", "auc_C", "auc_C_plus_S", "auc_gain")],
              "flag_sensitivity", "flag_specificity", "flag_ppv", "flag_npv"):
        print(f"{k:22s} {point[k]:.3f}  ({ci[k][0]:.3f}-{ci[k][1]:.3f})")
    print(pd.DataFrame(res["confidence_strata"]).round(3).to_string(index=False))
    print(rc.round(3).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
