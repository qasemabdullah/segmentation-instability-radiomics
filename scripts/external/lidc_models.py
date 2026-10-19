"""LIDC-IDRI step 3: reader-dependent classifications, their errors and model variability.

Paper:    Fig. 5a-f, Suppl. Table S17, Suppl. Fig. S9
Pipeline: python pipeline.py run lidc_models   (stage 'external')

Design (mirrors the discovery-cohort deployment analysis):
  * label: nodule-level consensus malignancy (median of the 3-4 readers'
    ratings; >= 4 malignant, <= 2 benign, 3 excluded as indeterminate). As a
    check against circularity (the readers who rated malignancy also drew the
    segmentations), each reader's segmentation is also scored against the leave-that-
    reader-out consensus;
  * development: in every outer fold (grouped by patient, stratified), an L2
    logistic model with an inner grouped grid search is fitted on ONE reader's
    segmentation per training nodule, chosen at random (a model developed on one
    reader's segmentations);
  * deployment: each held-out nodule is scored from every reader's segmentation;
    decision = predicted probability >= 0.5 (balanced class weights) and, as a
    check, >= the training median;
  * model variability: the same segmentation scored by models refitted with a different
    fold split and a different random choice of training reader;
  * the +-1 mm self-check flag, as in the main cohort.
Repeated over N seeds; 95% intervals by patient-level bootstrap.

Output: data/external/lidc/lidc_model_summary.json, lidc_oof.csv.gz
Run:    python scripts/external/lidc_models.py [n_seeds]
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIDC = ROOT / "data" / "external" / "lidc"
NON_FEATURES = {"case_id", "perturbation_mm", "PatientID", "nodule", "reader", "texture_n_slices"}


def load(source="radiomics"):
    a = pd.read_csv(LIDC / "annotations.csv")
    if source == "fm":  # seed-based foundation-model features: one row per reader contour (seed = its centroid)
        f = pd.read_csv(LIDC / "lidc_fm_features.csv").drop(columns=["key", "seed_x", "seed_y", "seed_z"])
        f["perturbation_mm"] = 0
        rad = pd.read_csv(LIDC / "lidc_features.csv", usecols=["PatientID", "nodule", "reader", "perturbation_mm"])
        # restrict to the nodules analysed with radiomics, for a like-for-like comparison
        keep = rad.groupby(["PatientID", "nodule", "reader"]).size().reset_index()[["PatientID", "nodule", "reader"]]
        f = f.merge(keep, on=["PatientID", "nodule", "reader"])
    else:
        f = pd.read_csv(LIDC / "lidc_features.csv")
    feats = [c for c in f.columns if c not in NON_FEATURES]
    mal = a.groupby(["PatientID", "nodule"]).malignancy
    lab = pd.DataFrame({"median_rating": mal.median(), "n_readers": mal.size()}).reset_index()
    lab["y"] = np.where(lab.median_rating >= 4, 1, np.where(lab.median_rating <= 2, 0, np.nan))
    # leave-one-reader-out consensus for each reader's contour
    a = a.merge(mal.sum().rename("s").reset_index(), on=["PatientID", "nodule"]).merge(
        mal.size().rename("n").reset_index(), on=["PatientID", "nodule"])
    a["loo_mean"] = (a.s - a.malignancy) / (a.n - 1)
    a["y_loo"] = np.where(a.loo_mean >= 3.5, 1, np.where(a.loo_mean <= 2.5, 0, np.nan))
    f = f.merge(a[["PatientID", "nodule", "reader", "y_loo", "malignancy"]], on=["PatientID", "nodule", "reader"])
    f = f.merge(lab, on=["PatientID", "nodule"])
    f = f[f.y.notna()].copy()
    # every reader's own contour must be usable; +-1 mm versions are used where they exist (a 1-mm
    # erosion can erase the smallest nodules, and requiring it would bias the sample towards large ones)
    complete = f[f.perturbation_mm == 0].groupby(["PatientID", "nodule"]).apply(
        lambda g: g.reader.nunique() == g.n_readers.iloc[0], include_groups=False)
    f = f.set_index(["PatientID", "nodule"]).loc[complete[complete].index].reset_index()
    return f, feats


C_GRID = {"radiomics": [0.01, 0.03, 0.1, 0.3, 1, 3], "fm": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]}


def fit(X, y, groups, seed, grid=C_GRID["radiomics"]):
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=5000))
    gs = GridSearchCV(pipe, {"logisticregression__C": grid}, scoring="roc_auc", cv=inner)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_


def one_seed(f, feats, seed, grid=C_GRID["radiomics"]):
    rng = np.random.default_rng(seed)
    base = f[f.perturbation_mm == 0]
    nod = base.drop_duplicates(["PatientID", "nodule"])[["PatientID", "nodule", "y"]].reset_index(drop=True)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    rows = []
    for tr, te in outer.split(nod, nod.y, nod.PatientID):
        tr_n, te_n = nod.iloc[tr], nod.iloc[te]
        # one randomly chosen reader per training nodule
        cand = base.merge(tr_n[["PatientID", "nodule"]], on=["PatientID", "nodule"])
        pick = cand.groupby(["PatientID", "nodule"]).sample(1, random_state=int(rng.integers(1e9)))
        model = fit(pick[feats].to_numpy(), pick.y.to_numpy(), pick.PatientID.to_numpy(), seed, grid)
        thr_median = float(np.median(model.predict_proba(pick[feats].to_numpy())[:, 1]))
        te_rows = f.merge(te_n[["PatientID", "nodule"]], on=["PatientID", "nodule"])
        p = model.predict_proba(te_rows[feats].to_numpy())[:, 1]
        rows.append(te_rows[["PatientID", "nodule", "reader", "perturbation_mm", "y", "y_loo"]].assign(
            p=p, thr_median=thr_median, seed=seed))
    return pd.concat(rows, ignore_index=True)


def summarize(oof, thr="p05"):
    """Reader-to-reader decision disagreement, correctness lottery, refit noise."""
    o = oof.copy()
    o["d"] = (o.p >= 0.5) if thr == "p05" else (o.p >= o.thr_median)
    o0 = o[o.perturbation_mm == 0]
    out = {}
    # per-seed metrics, then averaged
    per = []
    for s, g in o0.groupby("seed"):
        w = g.pivot_table(index=["PatientID", "nodule"], columns="reader", values="d", aggfunc="first")
        pw = g.pivot_table(index=["PatientID", "nodule"], columns="reader", values="p", aggfunc="first")
        y = g.drop_duplicates(["PatientID", "nodule"]).set_index(["PatientID", "nodule"]).y.loc[w.index]
        pairs = [(i, j) for i, j in itertools.combinations(w.columns, 2)]
        dis = np.nanmean([np.nanmean((w[i] != w[j])[w[i].notna() & w[j].notna()]) for i, j in pairs])
        any_dis = ((w.max(axis=1) != w.min(axis=1))).mean()
        correct = w.eq(y, axis=0).where(w.notna())
        lottery = ((correct.max(axis=1) == 1) & (correct.min(axis=1) == 0)).mean()
        rho = np.nanmean([spearmanr(pw[i], pw[j], nan_policy="omit").correlation for i, j in pairs])
        aucs = [roc_auc_score(y[pw[r].notna()], pw[r].dropna()) for r in pw.columns if pw[r].notna().sum() > 20]
        # correctness against the leave-that-reader-out label
        loo = g.dropna(subset=["y_loo"])
        acc_loo = float(((loo.p >= 0.5) == loo.y_loo).mean()) if thr == "p05" else float("nan")
        per.append({"pairwise_disagreement": dis, "nodules_any_disagreement": any_dis,
                    "nodules_correctness_depends_on_reader": lottery, "mean_rank_rho": rho,
                    "auc_by_reader_min": min(aucs), "auc_by_reader_max": max(aucs),
                    "auc_by_reader_mean": float(np.mean(aucs)), "accuracy_loo_label": acc_loo})
    out.update(pd.DataFrame(per).mean().to_dict())
    # refitting noise: same nodule, same reader contour, two different seeds
    w = o0.pivot_table(index=["PatientID", "nodule", "reader"], columns="seed", values="d", aggfunc="first")
    out["refit_noise"] = float(np.mean([(w[a] != w[b]).mean() for a, b in itertools.combinations(w.columns, 2)]))
    out.update(contour_sensitivity(o0))  # readers' own contours only: comparable across feature types
    if o.perturbation_mm.nunique() < 3:
        return out
    # +-1 mm self-check flag vs disagreement with the other readers (per seed, pooled)
    fl = []
    for s, g in o.groupby("seed"):
        d = g.pivot_table(index=["PatientID", "nodule", "reader"], columns="perturbation_mm", values="d",
                          aggfunc="first").dropna()
        flag = (d.nunique(axis=1) > 1).rename("flag").reset_index()
        d0 = d[0].rename("d0").reset_index()
        m = d0.merge(flag, on=["PatientID", "nodule", "reader"])
        m = m.merge(m[["PatientID", "nodule", "reader", "d0"]], on=["PatientID", "nodule"], suffixes=("", "_o"))
        m = m[m.reader != m.reader_o]
        m["disagree"] = m.d0 != m.d0_o
        fl.append(m)
    m = pd.concat(fl)
    a, b = m[m.flag].disagree.mean(), m[~m.flag].disagree.mean()
    out.update({"flag_rate": float(m.flag.mean()), "disagree_if_flagged": float(a), "disagree_if_not_flagged": float(b),
                "flag_odds_ratio": float((a / (1 - a)) / (b / (1 - b)))})
    out.update({f"{k}_all_contours": v for k, v in contour_sensitivity(o).items()})
    return out


def contour_sensitivity(o):
    """Per-nodule decision stability over a set of plausible contours (the readers'
    own contours, or every reader's contour at -1, 0 and +1 mm): the share of contours whose decision differs
    from the nodule's majority decision, and the probability of a 'malignant'
    call across contours. Asks whether contour-unstable nodules are the ones the
    model gets wrong, and whether averaging over contours helps."""
    res = []
    for s, g in o.groupby("seed"):
        per = g.groupby(["PatientID", "nodule"]).agg(p_call=("d", "mean"), p_mean=("p", "mean"), y=("y", "first"),
                                                     n=("d", "size"))
        per["sensitivity"] = np.minimum(per.p_call, 1 - per.p_call)  # share of contours against the majority
        single = g[g.perturbation_mm == 0].groupby(["PatientID", "nodule"]).first()  # one reader's contour
        stable = per.sensitivity == 0
        wrong = (per.p_call >= 0.5) != per.y
        res.append({"nodules_stable_all_contours": stable.mean(),
                    "nodules_sensitivity_ge_25pct": (per.sensitivity >= 0.25).mean(),
                    "error_rate_stable": wrong[stable].mean(), "error_rate_unstable": wrong[~stable].mean(),
                    "auc_single_reader_contour": roc_auc_score(single.y, single.p),
                    "auc_contour_averaged": roc_auc_score(per.y, per.p_mean)})
    return {k: float(v) for k, v in pd.DataFrame(res).mean().items()}


def main():
    n_seeds = int(next((a for a in sys.argv[1:] if a.isdigit()), 10))
    source = "fm" if "--fm" in sys.argv else "radiomics"
    f, feats = load(source)
    nod = f.drop_duplicates(["PatientID", "nodule"])
    print(f"{len(nod)} nodules ({int(nod.y.sum())} malignant, {int((nod.y == 0).sum())} benign) in "
          f"{nod.PatientID.nunique()} patients; {len(feats)} features")
    oof = pd.concat(Parallel(n_jobs=min(n_seeds, 4))(delayed(one_seed)(f, feats, s, C_GRID[source])
                                                      for s in range(n_seeds)))
    tag = "" if source == "radiomics" else "_fm"
    oof.to_csv(LIDC / f"lidc_oof{tag}.csv.gz", index=False)
    res = {"n_nodules": int(len(nod)), "n_malignant": int(nod.y.sum()), "n_patients": int(nod.PatientID.nunique()),
           "n_seeds": n_seeds, "threshold_0.5": summarize(oof, "p05"), "threshold_train_median": summarize(oof, "median")}
    pairs = pd.read_csv(LIDC / "lidc_pairs.csv")
    res["reader_agreement"] = {"median_dice": float(pairs.dice.median()), "median_assd_mm": float(pairs.assd_mm.median()),
                               "iqr_assd_mm": [float(pairs.assd_mm.quantile(q)) for q in (0.25, 0.75)]}
    res["features"] = source
    (LIDC / f"lidc_model_summary{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
