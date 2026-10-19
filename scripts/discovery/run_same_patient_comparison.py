"""Model-development versus segmentation instability on the same patients.

Every model is developed exactly as in the main analysis (nested
cross-validation, 20 fold seeds x 5 outer folds; logistic regression for
recurrence and EGFR, ridge Cox for OS) on the discovery cohort's expert
segmentations. Each fitted model then scores, with its own training-median
threshold:

  discovery  its held-out patients from the expert, -2/-1/+1/+2-mm and AI
             segmentations;
  readers    all 21 patients of the multi-reader NSCLC cohort (never used in
             training) from every radiation oncologist's segmentation and its
             -1/+1-mm versions.

Because the same patients, models and threshold rule are used throughout,
the two sources of instability are directly comparable:

  model-development   same segmentation, two different models
  segmentation        same model, two different segmentations
  joint               different model and different segmentation

Uncertainty: 2,000 patient-level bootstrap resamples (95% intervals for each
rate and for the difference segmentation - model development). Variance
decomposition: each score is expressed as its percentile within the scoring
model's training scores (50 = threshold), and for every patient the
models x segmentations table is split by two-way ANOVA into model,
segmentation and interaction sums of squares.

Paper:    Fig. 1f, Fig. 3g, Fig. 5c, Suppl. Table S9
Pipeline: python pipeline.py run same_patient_comparison   (stage 'ai')
Needs:    radiomics_features.csv, perturbation_features.csv,
          radiomics_features_bamf_ai.csv, reader_features_pm1.csv
Outputs:  data/processed/decomposition_predictions{_25}.csv.gz
          data/processed/decomposition_summary{_25}.csv
Run:      python scripts/discovery/run_same_patient_comparison.py [--drop-roundness] [n_seeds] [n_jobs]
          --drop-roundness removes shape_roundness (identical to sphericity in
          SimpleITK), giving the 25-feature sensitivity analysis.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, modeling, survival  # noqa: E402

P = config.PROCESSED_DIR
MM = [-2, -1, 1, 2]
N_BOOT = 2000


def tables(drop):
    exp = pd.read_csv(config.RADIOMICS_OUT)
    pert = pd.read_csv(P / "perturbation_features.csv")
    ai = pd.read_csv(P / "radiomics_features_bamf_ai.csv")
    rd = pd.read_csv(P / "reader_features_pm1.csv")
    segs = {"expert": exp.set_index("case_id")}
    for mm in MM:
        segs[f"{mm:+d}mm"] = pert[pert.perturbation_mm == mm].set_index("case_id")
    segs["ai"] = ai.set_index("case_id")
    return exp, segs, rd


def pct(train_scores, x):
    t = np.sort(train_scores)
    return 100 * np.searchsorted(t, x, side="right") / len(t)


def one(ep, seed, drop):
    warnings.filterwarnings("ignore")
    exp, segs, rd = tables(drop)
    from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
    if ep == "os":
        X, y, ids, _ = survival.load_survival_dataset(exp, endpoint="os")
        outer = KFold(5, shuffle=True, random_state=seed)
        splits = outer.split(X)
    else:
        X, y, ids, num, cat, gene, emb = modeling.load_dataset(ep, "radiomics", exp)
        outer = StratifiedKFold(5, shuffle=True, random_state=seed)
        splits = outer.split(X, y)
    feats = [c for c in X.columns if not (drop and c == "shape_roundness")]
    X = X[feats]
    rows = []
    for fold, (tr, te) in enumerate(splits, start=1):
        if ep == "os":
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sksurv.linear_model import CoxPHSurvivalAnalysis
            pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                             ("clf", CoxPHSurvivalAnalysis())])
            search = GridSearchCV(pipe, {"clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
                                  cv=KFold(5, shuffle=True, random_state=seed), n_jobs=1)
            search.fit(X.iloc[tr], y[tr])
            score = search.predict
        else:
            pipe = modeling._build_pipeline(feats, [], [], [])
            search = GridSearchCV(pipe, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, scoring="roc_auc",
                                  cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=1)
            search.fit(X.iloc[tr], y[tr])
            score = lambda Z: search.predict_proba(Z)[:, 1]  # noqa: E731
        s_train = score(X.iloc[tr])
        thr = float(np.median(s_train))
        if ep == "os":
            perf = float(search.score(X.iloc[te], y[te]))
        else:
            from sklearn.metrics import roc_auc_score
            perf = float(roc_auc_score(y[te], score(X.iloc[te])))
        test_ids = ids[te]
        for name, t in segs.items():
            have = [c for c in test_ids if c in t.index]
            if not have:
                continue
            s = score(t.loc[have, feats])
            rows += [{"endpoint": ep, "seed": seed, "fold": fold, "cohort": "discovery", "case_id": c, "reader": "",
                      "segmentation": name, "score": float(v), "pct": float(q), "high": bool(v >= thr),
                      "fold_performance": perf}
                     for c, v, q in zip(have, s, pct(s_train, s))]
        s = score(rd[feats])
        rows += [{"endpoint": ep, "seed": seed, "fold": fold, "cohort": "readers", "case_id": c, "reader": str(r),
                  "segmentation": {0: "original", -1: "-1mm", 1: "+1mm"}[m], "score": float(v), "pct": float(q),
                  "high": bool(v >= thr)}
                 for c, r, m, v, q in zip(rd.case_id, rd.reader, rd.perturbation_mm, s, pct(s_train, s))]
    return rows


def boot(per, cols, rng):
    """per: one row per patient with per-patient rates; returns point, lo, hi for each column and differences."""
    idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
    out = {}
    for c in cols:
        v = per[c].to_numpy()
        b = v[idx].mean(axis=1)
        out[c] = (v.mean(), *np.percentile(b, [2.5, 97.5]))
    return out


def anova_shares(tab):
    """tab: dict patient -> 2-D array (models x segmentations); pooled sums of squares."""
    ss = np.zeros(3)
    for a in tab.values():
        a = a[~np.isnan(a).any(axis=1)]
        if a.shape[0] < 2 or a.shape[1] < 2:
            continue
        g = a.mean()
        ss += [a.shape[1] * ((a.mean(axis=1) - g) ** 2).sum(), a.shape[0] * ((a.mean(axis=0) - g) ** 2).sum(),
               ((a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + g) ** 2).sum()]
    return ss / ss.sum()


def summarize(pr):
    rng = np.random.default_rng(20260930)
    res = []
    for ep, g in pr.groupby("endpoint"):
        fp = g[(g.cohort == "discovery") & (g.segmentation == "expert")].groupby(["seed", "fold"]).fold_performance.first()
        res.append({"cohort": "discovery", "endpoint": ep, "comparison": "performance (mean of fold scores over seeds)",
                    "patients": g[g.cohort == "discovery"].case_id.nunique(), "performance": fp.groupby("seed").mean().mean(),
                    "performance_sd_over_seeds": fp.groupby("seed").mean().std()})
        # discovery: held-out patients, one prediction per seed and segmentation
        d = g[g.cohort == "discovery"].pivot_table(index="case_id", columns=["segmentation", "seed"], values="high",
                                                   aggfunc="first")
        q = g[g.cohort == "discovery"].pivot_table(index="case_id", columns=["segmentation", "seed"], values="pct",
                                                   aggfunc="first")
        seeds = sorted(g.seed.unique())
        pairs = list(itertools.combinations(seeds, 2))
        for comp, segs in (("±1 mm", ["-1mm", "+1mm"]), ("±2 mm", ["-2mm", "+2mm"]), ("AI", ["ai"])):
            pts = d.index[d[segs[0]].notna().all(axis=1)]
            e = d.loc[pts, "expert"].astype(float)
            per = pd.DataFrame(index=pts)
            per["model"] = np.mean([(e[a] != e[b]) for a, b in pairs], axis=0)
            per["segmentation"] = np.mean([(e[s] != d.loc[pts, sg][s].astype(float)) for sg in segs for s in seeds], axis=0)
            per["joint"] = np.mean([(e[a] != d.loc[pts, sg][b].astype(float)) for sg in segs for a, b in pairs], axis=0)
            per["diff"] = per.segmentation - per.model
            bt = boot(per, ["model", "segmentation", "joint", "diff"], rng)
            tab = {c: np.column_stack([q.loc[c, "expert"].to_numpy()] + [q.loc[c, sg].to_numpy() for sg in segs])
                   for c in pts}
            sh = anova_shares(tab)
            res.append({"cohort": "discovery", "endpoint": ep, "comparison": comp, "patients": len(pts),
                        **{f"{k}_{s}": v for k, t in bt.items() for s, v in zip(("rate", "lo", "hi"), t)},
                        "share_model": sh[0], "share_segmentation": sh[1], "share_interaction": sh[2]})
        # readers: 100 models x 5 readers (x3 positions) on 21 unseen patients
        rr = g[g.cohort == "readers"].copy()
        rr["model"] = rr.seed.astype(str) + "_" + rr.fold.astype(str)
        # threshold rules: each model's training median (as in the discovery cohort), or the median of that
        # model's scores over the readers' original segmentations of these 21 patients (cohort median)
        cmed = rr[rr.segmentation == "original"].groupby("model").score.median()
        for rule in ("training median", "cohort median"):
            r = rr.copy()
            if rule == "cohort median":
                r["high"] = r.score >= r.model.map(cmed)
            res.append(reader_block(r, ep, rule, rng))
    return pd.DataFrame(res)


def reader_block(r, ep, rule, rng):
    if True:
        orig = r[r.segmentation == "original"]
        h = orig.pivot_table(index=["case_id", "reader"], columns="model", values="high", aggfunc="first").astype(float)
        m = h.shape[1]
        k = h.sum(axis=1)
        model_pair = (2 * k * (m - k) / (m * (m - 1))).groupby(level=0).mean()  # discordant model pairs, per patient
        hr = orig.pivot_table(index=["case_id", "model"], columns="reader", values="high", aggfunc="first").astype(float)
        rp = list(itertools.combinations(hr.columns, 2))
        reader_pair = pd.concat([(hr[a] != hr[b]) for a, b in rp], axis=1).mean(axis=1).groupby(level=0).mean()
        anyd = (hr.nunique(axis=1) > 1).groupby(level=0).mean()
        pm = r.pivot_table(index=["case_id", "reader", "model"], columns="segmentation", values="high", aggfunc="first")
        pm1 = (((pm["-1mm"] != pm["original"]).astype(float) + (pm["+1mm"] != pm["original"]).astype(float)) / 2
               ).groupby(level=0).mean()
        per = pd.DataFrame({"model": model_pair, "segmentation": reader_pair, "any_reader": anyd, "pm1": pm1})
        per["diff"] = per.segmentation - per.model
        per["pm1_minus_reader"] = per.pm1 - per.segmentation
        bt = boot(per, ["model", "segmentation", "any_reader", "pm1", "diff", "pm1_minus_reader"], rng)
        qq = orig.pivot_table(index=["case_id", "model"], columns="reader", values="pct", aggfunc="first")
        tab = {c: qq.loc[c].to_numpy() for c in qq.index.get_level_values(0).unique()}
        sh = anova_shares(tab)
        return {"cohort": "readers", "endpoint": ep, "comparison": f"another radiation oncologist ({rule})",
                "patients": per.shape[0], "models": m,
                **{f"{k}_{s}": v for k, t in bt.items() for s, v in zip(("rate", "lo", "hi"), t)},
                "share_model": sh[0], "share_segmentation": sh[1], "share_interaction": sh[2]}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    drop = "--drop-roundness" in sys.argv
    n_seeds = int(args[0]) if args else 20
    n_jobs = int(args[1]) if len(args) > 1 else 6
    sfx = "_25" if drop else ""
    jobs = [(ep, s) for ep in ("recurrence", "egfr", "os") for s in range(n_seeds)]
    with contextlib.redirect_stdout(io.StringIO()):
        out = Parallel(n_jobs=n_jobs)(delayed(one)(ep, s, drop) for ep, s in jobs)
    pr = pd.DataFrame([row for rows in out for row in rows])
    pr.to_csv(P / f"decomposition_predictions{sfx}.csv.gz", index=False)
    s = summarize(pr)
    s.to_csv(P / f"decomposition_summary{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).T.to_string())


if __name__ == "__main__":
    main()
