"""Liver-metastasis cohort: survival models, reclassification by AI segmentation and model variability.

Paper:    Fig. 5b, c, g, Suppl. Table S19
Pipeline: python pipeline.py run crlm   (stage 'external')

For each of N seeds: a ridge Cox model (same grid as the main cohort) is fitted
under 5-fold CV on expert-segmentation features of the largest lesion; every
held-out patient is scored from the expert segmentation, its -1/+1 mm versions and
the AI segmentation (and the radiologist-corrected AI segmentation where available).
Decision = risk above the training median. Refitting noise = the same patient
and segmentation classified differently by two seeds.

Endpoints: overall survival and disease-free survival (clinical table from IDC).
Output: data/external/crlm/crlm_results.json, crlm_oof.csv.gz
Run:    python scripts/external/crlm_models.py [n_seeds]
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config  # noqa: E402
from scripts.external.lung1_models import fit_cox  # noqa: E402

CRLM = ROOT / "data" / "external" / "crlm"
CLIN = str(config.idc_clinical_table("colorectal_liver_metastases_clinical"))
NON_FEATURES = {"case_id", "perturbation_mm", "PatientID", "contour", "texture_n_slices"}
ENDPOINTS = {"os": ("overall_survival_months", "vital_status"), "dfs": ("months_to_dfs_progression", "vital_status_dfs")}


def load():
    f = pd.read_csv(CRLM / "crlm_features.csv")
    f["variant"] = np.where(f.contour == "expert", "expert" + f.perturbation_mm.map({-1: "-1", 0: "", 1: "+1"}), f.contour)
    feats = [c for c in f.columns if c not in NON_FEATURES | {"variant"}]
    c = pd.read_parquet(CLIN).rename(columns={"dicom_patient_id": "PatientID"})
    return f, feats, c


def one_seed(f, feats, clin, ep, seed):
    t, e = ENDPOINTS[ep]
    pts = f[f.variant == "expert"][["PatientID"]].merge(clin[["PatientID", t, e]], on="PatientID")
    pts = pts[pts[t] > 0].reset_index(drop=True)
    X = f.set_index(["PatientID", "variant"])[feats]
    out = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(pts, pts[e]):
        a, b = pts.iloc[tr], pts.iloc[te]
        y = Surv.from_arrays(a[e].astype(bool), a[t])
        m = fit_cox(X.loc[[(p, "expert") for p in a.PatientID]].to_numpy(), y)
        thr = float(np.median(m.predict(X.loc[[(p, "expert") for p in a.PatientID]].to_numpy())))
        rows = X.reset_index()
        rows = rows[rows.PatientID.isin(b.PatientID)]
        out.append(rows[["PatientID", "variant"]].assign(risk=m.predict(rows[feats].to_numpy()), thr=thr, seed=seed))
    return pd.concat(out).merge(pts, on="PatientID")


def cindex(g, t, e):
    return concordance_index_censored(g[e].astype(bool), g[t], g.risk)[0]


def summarize(oof, ep):
    t, e = ENDPOINTS[ep]
    oof = oof.assign(d=oof.risk >= oof.thr)
    res = {}
    w = oof.pivot_table(index=["seed", "PatientID"], columns="variant", values="d", aggfunc="first")
    r = oof.pivot_table(index=["seed", "PatientID"], columns="variant", values="risk", aggfunc="first")
    for v in [c for c in w.columns if c != "expert"]:
        ok = w[v].notna()
        res[f"flips_expert_vs_{v}"] = float((w.loc[ok, v] != w.loc[ok, "expert"]).mean())
        res[f"rank_rho_expert_vs_{v}"] = float(np.mean([spearmanr(g["expert"], g[v]).correlation
                                                        for _, g in r[ok].groupby(level=0)]))
        res[f"n_{v}"] = int(ok.groupby(level=0).sum().iloc[0])
    # refit noise on the expert contour
    ex = oof[oof.variant == "expert"].pivot_table(index="PatientID", columns="seed", values="d", aggfunc="first")
    res["refit_noise"] = float(np.mean([(ex[a] != ex[b]).mean() for a, b in itertools.combinations(ex.columns, 2)]))
    # per-patient contour sensitivity over the plausible contours (expert, -1, +1 mm, AI, corrected)
    per = w.astype(float).mean(axis=1)
    sens = np.minimum(per, 1 - per)
    res["patients_stable_all_contours"] = float((sens == 0).groupby(level=0).mean().mean())
    res["patients_sensitivity_ge_25pct"] = float((sens >= 0.25).groupby(level=0).mean().mean())
    avg = oof.groupby(["seed", "PatientID"]).agg(risk=("risk", "mean"), **{t: (t, "first"), e: (e, "first")}).reset_index()
    res["c_contour_averaged"] = float(np.mean([cindex(x, t, e) for _, x in avg.groupby("seed")]))
    # concordance per contour, among patients with an AI contour (same patients for both)
    ai_pts = set(oof[oof.variant == "ai"].PatientID)
    for v in ("expert", "ai", "expert-1", "expert+1"):
        g = oof[(oof.variant == v) & oof.PatientID.isin(ai_pts)]
        res[f"c_{v}"] = float(np.mean([cindex(x, t, e) for _, x in g.groupby("seed")]))
    return res


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    f, feats, clin = load()
    agree = pd.read_csv(CRLM / "crlm_agreement.csv")
    results = {"n_patients": int(f.PatientID.nunique()),
               "ai_detected": float(agree.ai_detected.mean()),
               "ai_dice_median": float(agree.ai_dice.median()),
               "ai_dice_iqr": [float(agree.ai_dice.quantile(q)) for q in (0.25, 0.75)],
               "ai_assd_median_mm": float(agree.ai_assd_mm.median()),
               "ai_volume_ratio_median": float(agree.ai_volume_ratio.median())}
    if "corrected_dice" in agree:
        results["corrected_n"] = int(agree.corrected_dice.notna().sum())
        results["corrected_dice_median"] = float(agree.corrected_dice.median())
    oofs = []
    for ep in ENDPOINTS:
        oof = pd.concat(Parallel(n_jobs=min(8, n_seeds))(delayed(one_seed)(f, feats, clin, ep, s) for s in range(n_seeds)))
        t, e = ENDPOINTS[ep]
        results[ep] = {"events": int(clin[clin.PatientID.isin(f.PatientID)][e].sum()), **summarize(oof, ep)}
        oofs.append(oof.assign(endpoint=ep))
    pd.concat(oofs).to_csv(CRLM / "crlm_oof.csv.gz", index=False)
    (CRLM / "crlm_results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
