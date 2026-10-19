"""External validation cohort (Lung1): survival model, reclassification and phenotypes under adjustment.

Paper:    Fig. 7d-g, Suppl. Table S26
Pipeline: python pipeline.py run lung1_models   (stage 'external')

  1. OS Cox model under nested CV at -2, -1, 0, +1, +2 mm (same folds at every
     shift), with the range and a fold-stratified bootstrap 95% CI.
  2. Kaplan-Meier by median out-of-fold risk at the original boundary.
  3. Decision flips: models trained on original segmentations, held-out patients
     scored from their -1 and +1 mm segmentations; and whether the segmentation
     ensemble (mean prediction over the segmentation and its +-1 mm neighbours)
     reduces flips between the -1 mm and +1 mm "readers".
  4. Ward clustering (k = 4) and reassignment under each shift, with a
     100-replicate bootstrap reclustering baseline.
  5. Feature instability ranking compared with the discovery cohort.
  6. Transportability: the discovery-cohort OS model applied unchanged to Lung1.

Outputs: data/external/lung1/lung1_results.json and lung1_*.csv
Run:     python scripts/external/lung1_models.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config  # noqa: E402
from nsclc_pipeline.modeling import NON_FEATURE_COLS  # noqa: E402

L1 = ROOT / "data" / "external" / "lung1"
MM = [-2, -1, 0, 1, 2]


def cox():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", CoxPHSurvivalAnalysis())])


def fit_cox(X, y):
    s = GridSearchCV(cox(), {"clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, cv=KFold(5, shuffle=True, random_state=0), n_jobs=1)
    s.fit(X, y)
    return s


def load():
    f = pd.read_csv(L1 / "lung1_features.csv")
    feat = [c for c in f.columns if c not in NON_FEATURE_COLS | {"PatientID", "perturbation_mm"}]
    cl = pd.read_csv(L1 / "clinical.csv")
    cl = cl[cl["Survival.time"] > 0]
    complete = f.groupby("PatientID").perturbation_mm.nunique()
    ids = sorted(set(complete[complete == len(MM)].index) & set(cl.PatientID))
    V = {mm: f[f.perturbation_mm == mm].set_index("PatientID").loc[ids, feat] for mm in MM}
    c = cl.set_index("PatientID").loc[ids]
    y = Surv.from_arrays(event=c["deadstatus.event"].astype(bool).to_numpy(), time=c["Survival.time"].to_numpy())
    return V, y, ids, feat, c


def nested(V, y):
    """Train and test at the same shift, same folds for every shift."""
    oof = {mm: np.zeros(len(y)) for mm in MM}
    fold_of = np.zeros(len(y), int)
    per_fold = {mm: [] for mm in MM}
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(V[0]), 1):
        fold_of[te] = k
        for mm in MM:
            m = fit_cox(V[mm].iloc[tr], y[tr])
            oof[mm][te] = m.predict(V[mm].iloc[te])
            per_fold[mm].append(concordance_index_censored(y[te]["event"], y[te]["time"], oof[mm][te])[0])
    return oof, fold_of, per_fold


def boot_swing(oof, fold_of, y, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    folds = [np.where(fold_of == k)[0] for k in range(1, 6)]
    sw = []
    for _ in range(n):
        idx = [rng.choice(f, len(f), replace=True) for f in folds]
        means = []
        for mm in MM:
            cs = []
            for ix in idx:
                e, t = y[ix]["event"], y[ix]["time"]
                cs.append(concordance_index_censored(e, t, oof[mm][ix])[0] if e.any() else np.nan)
            means.append(np.nanmean(cs))
        sw.append(max(means) - min(means))
    return np.percentile(sw, [2.5, 97.5])


def deployment(V, y):
    """Train on original contours; score held-out patients from shifted contours."""
    rows = []
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(V[0]), 1):
        m = fit_cox(V[0].iloc[tr], y[tr])
        thr = np.median(m.predict(V[0].iloc[tr]))
        p = {mm: m.predict(V[mm].iloc[te]) for mm in MM}
        tte_m1 = np.mean([p[-2], p[-1], p[0]], axis=0)
        tte_p1 = np.mean([p[0], p[1], p[2]], axis=0)
        for j in range(len(te)):
            rows.append({"fold": k, "thr": thr, **{f"p{mm:+d}": p[mm][j] for mm in MM},
                         "tte_m1": tte_m1[j], "tte_p1": tte_p1[j]})
    d = pd.DataFrame(rows)
    hi = lambda col: d[col] >= d.thr  # noqa: E731
    return {"flip_0_vs_-1mm": float((hi("p+0") != hi("p-1")).mean()),
            "flip_0_vs_+1mm": float((hi("p+0") != hi("p+1")).mean()),
            "flip_-1_vs_+1mm_single": float((hi("p-1") != hi("p+1")).mean()),
            "flip_-1_vs_+1mm_ensemble": float((hi("tte_m1") != hi("tte_p1")).mean())}


def clustering(V):
    X0 = V[0].to_numpy()
    imp = SimpleImputer(strategy="median").fit(X0)
    sc = StandardScaler().fit(imp.transform(X0))
    Z = sc.transform(imp.transform(X0))
    sil = {k: silhouette_score(Z, AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)) for k in range(3, 8)}
    lab = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Z)
    cen = np.array([Z[lab == c].mean(0) for c in range(4)])
    out = {"silhouette_k4": sil[4], "sizes": np.bincount(lab).tolist()}
    for mm in MM:
        Zp = sc.transform(imp.transform(V[mm].to_numpy()))
        new = np.linalg.norm(Zp[:, None] - cen[None], axis=2).argmin(1)
        out[f"reassigned_{mm:+d}"] = float((new != lab).mean())
        out[f"ari_{mm:+d}"] = float(adjusted_rand_score(lab, new))
    rng = np.random.RandomState(0)
    aris = []
    for _ in range(100):
        idx = rng.choice(len(Z), len(Z), replace=True)
        aris.append(adjusted_rand_score(lab[idx], AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Z[idx])))
    out["bootstrap_ari_mean"], out["bootstrap_ari_sd"] = float(np.mean(aris)), float(np.std(aris))
    return out


def instability(V, feat):
    a, b = V[-1], V[1]
    rel = ((b - a).abs() / a.abs().replace(0, np.nan)).median()
    st = pd.read_csv(config.PROCESSED_DIR / "interobserver_vs_synthetic_comparison.csv").set_index("feature")
    common = rel.index.intersection(st.index)
    rho, p = spearmanr(rel[common], st.loc[common, "synthetic_perturbation_median_rel_change"])
    rel.rename("lung1_median_rel_change").to_csv(L1 / "lung1_feature_instability.csv")
    rho_r, p_r = spearmanr(rel[common], st.loc[common, "real_interobserver_median_rel_change"])
    return {"rho_vs_stanford_synthetic": rho, "p": p, "rho_vs_real_readers": rho_r, "p_real": p_r, "n_features": len(common)}


def transport(V, y, feat):
    """Stanford OS model, trained on all 144 original contours, applied to Lung1 without refitting."""
    from nsclc_pipeline import survival
    Xs, ys, _, _ = survival.load_survival_dataset(endpoint="os")
    m = fit_cox(Xs[feat], ys)
    out = {}
    for mm in (-1, 0, 1):
        r = m.predict(V[mm][feat])
        out[f"external_c_{mm:+d}"] = concordance_index_censored(y["event"], y["time"], r)[0]
    return out


def main():
    V, y, ids, feat, clin = load()
    print(f"Lung1: {len(ids)} patients with features at every shift, {int(y['event'].sum())} deaths")
    oof, fold_of, per_fold = nested(V, y)
    res = {"n": len(ids), "events": int(y["event"].sum()),
           "median_followup_months": float(np.median(y["time"]) / 30.4375)}
    for mm in MM:
        res[f"c_mean_{mm:+d}"] = float(np.mean(per_fold[mm]))
        res[f"c_sd_{mm:+d}"] = float(np.std(per_fold[mm]))
    means = [res[f"c_mean_{mm:+d}"] for mm in MM]
    res["swing"] = max(means) - min(means)
    res["swing_ci"] = boot_swing(oof, fold_of, y).tolist()
    hi = oof[0] >= np.median(oof[0])
    res["logrank_p_median_split"] = float(logrank_test(y["time"][hi], y["time"][~hi], y["event"][hi], y["event"][~hi]).p_value)
    res["events_high_low"] = [int(y["event"][hi].sum()), int(y["event"][~hi].sum())]
    res.update(deployment(V, y))
    res["clustering"] = clustering(V)
    res["instability"] = instability(V, feat)
    res.update(transport(V, y, feat))
    pd.DataFrame({"PatientID": ids, "fold": fold_of, "event": y["event"], "time": y["time"],
                  **{f"risk_{mm:+d}": oof[mm] for mm in MM}}).to_csv(L1 / "lung1_oof_predictions.csv", index=False)
    pd.DataFrame([{"mm": mm, "fold": k + 1, "c": c} for mm in MM for k, c in enumerate(per_fold[mm])]).to_csv(
        L1 / "lung1_fold_concordance.csv", index=False)
    (L1 / "lung1_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
