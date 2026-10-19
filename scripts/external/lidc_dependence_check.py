"""LIDC-IDRI: are reader-dependent classifications more often discordant with
the reader-consensus label beyond their proximity to the threshold?

Uses the stored held-out predictions of lidc_models.py (10 seeds, readers'
own segmentations, threshold 0.5). Per seed and nodule:

  reader-dependent   the readers' segmentations do not all give the same class
  majority class     class given by most readers' segmentations; a 2-2 tie is
                     counted as malignant in the main analysis, and here also
                     as benign or excluded
  confidence         |mean predicted probability over readers - 0.5|

Reported (all intervals from 1,000 patient-level bootstrap resamples):
  * discordance of the majority class with the consensus label, reader-
    dependent vs reader-independent nodules, under each tie rule;
  * the same within quintiles of confidence, and the odds ratio for reader
    dependence adjusted for confidence (natural cubic spline of log confidence);
  * reader-to-reader disagreement vs model-development variability (same
    segmentation, two seeds), with the 95% interval of their difference.

Paper:    Fig. 5a-f, Suppl. Table S17, Suppl. Fig. S9
Pipeline: python pipeline.py run lidc_models   (stage 'external')
Needs:    data/external/lidc/lidc_oof.csv.gz (scripts/external/lidc_models.py)
Output:   data/external/lidc/lidc_dependence_check.json, lidc_dependence_strata.csv
Run:      python scripts/external/lidc_dependence_check.py
"""
from __future__ import annotations

import itertools
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = ROOT / "data" / "external" / "lidc"
N_BOOT = 1000
EPS = 1e-4


def nodules(o):
    o = o[o.perturbation_mm == 0].assign(d=lambda x: (x.p >= 0.5).astype(int))
    g = o.groupby(["seed", "PatientID", "nodule"])
    n = g.agg(k=("d", "size"), pos=("d", "sum"), pbar=("p", "mean"), y=("y", "first")).reset_index()
    n["dependent"] = ((n.pos > 0) & (n.pos < n.k)).astype(int)
    n["conf"] = (n.pbar - 0.5).abs()
    tie = 2 * n.pos == n.k
    n["err_tie_malignant"] = ((2 * n.pos >= n.k).astype(int) != n.y).astype(float)
    n["err_tie_benign"] = ((2 * n.pos > n.k).astype(int) != n.y).astype(float)
    n["err_tie_excluded"] = n.err_tie_malignant.where(~tie)
    return n


def rate_by(n, col):
    a = n[n.dependent == 1][col].mean()
    b = n[n.dependent == 0][col].mean()
    return a, b


def adj_or(n, col):
    m = n.dropna(subset=[col])
    z = np.log(m.conf.to_numpy() + EPS)
    z = (z - z.mean()) / z.std()
    X = np.c_[SplineTransformer(n_knots=5, degree=3).fit_transform(z[:, None]), m.dependent.to_numpy()]
    fit = LogisticRegression(C=1e6, max_iter=5000).fit(X, m[col].to_numpy().astype(int))
    return float(np.exp(fit.coef_[0][-1]))


def disagreement_units(o):
    """Per patient: reader-pair disagreement and model-development (seed-pair) disagreement counts."""
    o = o[o.perturbation_mm == 0].assign(d=lambda x: (x.p >= 0.5).astype(int))
    rp = []
    for (s, pid, nod), g in o.groupby(["seed", "PatientID", "nodule"]):
        d = g.d.to_numpy()
        for i, j in itertools.combinations(range(len(d)), 2):
            rp.append((pid, int(d[i] != d[j])))
    rp = pd.DataFrame(rp, columns=["PatientID", "dis"]).groupby("PatientID").dis.agg(["sum", "size"])
    w = o.pivot_table(index=["PatientID", "nodule", "reader"], columns="seed", values="d", aggfunc="first")
    sp = list(itertools.combinations(w.columns, 2))
    md = pd.DataFrame({"sum": sum((w[a] != w[b]).astype(int) for a, b in sp), "size": len(sp)})
    md = md.groupby(level=0).sum()
    return rp, md


def main():
    rng = np.random.default_rng(20260930)
    o = pd.read_csv(D / "lidc_oof.csv.gz")
    n = nodules(o)
    pts = n.PatientID.unique()
    rp, md = disagreement_units(o)
    rules = ("err_tie_malignant", "err_tie_benign", "err_tie_excluded")

    def stats(nn, rpp, mdd):
        out = {}
        for r in rules:
            a, b = rate_by(nn, r)
            out[f"{r}_dependent"], out[f"{r}_independent"] = a, b
        out["or_dependent_adj_conf"] = adj_or(nn, "err_tie_malignant")
        out["reader_pair_disagreement"] = rpp["sum"].sum() / rpp["size"].sum()
        out["model_development"] = mdd["sum"].sum() / mdd["size"].sum()
        out["difference"] = out["reader_pair_disagreement"] - out["model_development"]
        return out

    point = stats(n, rp, md)
    by_pt = {p: g for p, g in n.groupby("PatientID")}
    boots = []
    for _ in range(N_BOOT):
        s = rng.choice(pts, len(pts), replace=True)
        nn = pd.concat([by_pt[p] for p in s], ignore_index=True)
        boots.append(stats(nn, rp.loc[s], md.loc[s]))
    b = pd.DataFrame(boots)
    ci = {k: [float(np.nanpercentile(b[k], 2.5)), float(np.nanpercentile(b[k], 97.5))] for k in b.columns}

    # within quintiles of confidence
    n["conf_q"] = pd.qcut(n.conf, 5, labels=[f"Q{i}" for i in range(1, 6)])
    strata = (n.groupby(["conf_q", "dependent"], observed=True)
              .agg(nodule_predictions=("y", "size"), discordant=("err_tie_malignant", "mean")).reset_index())
    strata.to_csv(D / "lidc_dependence_strata.csv", index=False)
    res = {"n_nodule_predictions": int(len(n)), "n_patients": int(len(pts)), "n_ties": int((2 * n.pos == n.k).sum()),
           "share_dependent": float(n.dependent.mean()), "bootstrap": f"{N_BOOT} patient resamples",
           "point": point, "ci95": ci}
    (D / "lidc_dependence_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(strata.to_string(index=False))


if __name__ == "__main__":
    main()
