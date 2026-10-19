"""Same-patient comparison with model development repeated in every bootstrap replicate.

The intervals of run_same_patient_comparison.py resample patients from
stored held-out predictions and are therefore conditional on the fitted
models. Here the whole procedure is repeated in each replicate: the 144
discovery-cohort patients are resampled with replacement, the nested
cross-validation (20 fold seeds x 5 outer folds, inner grid search) is rerun
on the resample with a patient's copies kept in one fold (outer and inner),
and every fitted model scores its held-out patients from the expert, ±1-mm and
AI segmentations and the 21 patients of the multi-reader NSCLC cohort (also
resampled) from every radiation oncologist's segmentation. The
segmentation-minus-model-development differences are recomputed in each
replicate, so their spread includes variability from training patients, fold
allocation and the threshold. Because a resample contains duplicated patients
(about 63% unique), refitted models are noisier in the resample than in the
data, which shifts the percentile interval; the reported interval is therefore
the main estimate ± 1.96 bootstrap standard errors, with the percentile
interval alongside.

Paper:    Methods, Results, Suppl. Table S10
Pipeline: python pipeline.py run refit_bootstrap   (stage 'ai')
Needs:    the same inputs as run_same_patient_comparison.py
Outputs:  data/processed/refit_bootstrap_replicates.csv, data/processed/refit_bootstrap_summary.csv
Run:      python scripts/discovery/run_refit_bootstrap.py [n_boot] [n_seeds] [n_jobs]
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
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "discovery"))
from nsclc_pipeline import config, modeling, survival  # noqa: E402
from run_same_patient_comparison import tables  # noqa: E402

P = config.PROCESSED_DIR
SEGS = ["expert", "-1mm", "+1mm", "ai"]


def fit_score(ep, Xtr, ytr, gtr, seed, feats):
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if ep == "os":
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("clf", CoxPHSurvivalAnalysis())])
        search = GridSearchCV(pipe, {"clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
                              cv=GroupKFold(5, shuffle=True, random_state=seed), n_jobs=1)
        search.fit(Xtr, ytr, groups=gtr)
        return search.predict
    pipe = modeling._build_pipeline(feats, [], [], [])
    search = GridSearchCV(pipe, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, scoring="roc_auc",
                          cv=StratifiedGroupKFold(5, shuffle=True, random_state=seed), n_jobs=1)
    search.fit(Xtr, ytr, groups=gtr)
    return lambda Z: search.predict_proba(Z)[:, 1]


def one_replicate(ep, b, n_seeds):
    """Rates for one bootstrap replicate (b = 0 is the original sample, no resampling)."""
    warnings.filterwarnings("ignore")
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    exp, segs, rd = tables(False)
    if ep == "os":
        X, y, ids, _ = survival.load_survival_dataset(exp, endpoint="os")
    else:
        X, y, ids, *_ = modeling.load_dataset(ep, "radiomics", exp)
    feats = list(X.columns)
    rng = np.random.default_rng(1000 + b)
    idx = np.arange(len(ids)) if b == 0 else rng.integers(0, len(ids), len(ids))
    Xb, yb, gb = X.iloc[idx].reset_index(drop=True), y[idx], ids[idx]
    rpat = np.array(sorted(rd.case_id.unique()))
    rsel = rpat if b == 0 else rng.choice(rpat, len(rpat), replace=True)
    # discovery rows: (row in resample, seed) -> class per segmentation; readers: (patient copy, reader, model)
    disc = {s: np.full((len(idx), n_seeds), np.nan) for s in SEGS}
    rscore = []
    for seed in range(n_seeds):
        outer = (GroupKFold(5, shuffle=True, random_state=seed).split(Xb, groups=gb) if ep == "os" else
                 StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(Xb, yb, groups=gb))
        for fold, (tr, te) in enumerate(outer):
            score = fit_score(ep, Xb.iloc[tr], yb[tr], gb[tr], seed, feats)
            thr = float(np.median(score(Xb.iloc[tr])))
            for s in SEGS:
                t = segs[s]
                ok = np.array([c in t.index for c in gb[te]])
                if ok.any():
                    disc[s][te[ok], seed] = score(t.loc[gb[te][ok], feats]) >= thr
            sc = score(rd[feats])
            rscore.append(pd.DataFrame({"case_id": rd.case_id, "reader": rd.reader, "mm": rd.perturbation_mm,
                                        "score": sc, "model": f"{seed}_{fold}"}))
    out = {"endpoint": ep, "replicate": b}
    pairs = list(itertools.combinations(range(n_seeds), 2))
    e = disc["expert"]
    out["model"] = np.nanmean([np.nanmean(e[:, a] != e[:, c]) for a, c in pairs])
    for name, ss in (("pm1", ["-1mm", "+1mm"]), ("ai", ["ai"])):
        ok = ~np.isnan(disc[ss[0]][:, 0])
        out[f"model_{name}"] = np.mean([np.mean(e[ok, a] != e[ok, c]) for a, c in pairs])
        out[f"seg_{name}"] = np.mean([np.mean(e[ok] != disc[s][ok]) for s in ss])
        out[f"diff_{name}"] = out[f"seg_{name}"] - out[f"model_{name}"]
    # readers, thresholds at each model's median over the readers' original segmentations (main analysis)
    r = pd.concat(rscore)
    r = r[r.mm == 0]
    r["high"] = r.score >= r.model.map(r.groupby("model").score.median())
    h = r.pivot_table(index=["case_id", "reader"], columns="model", values="high", aggfunc="first").astype(float)
    m = h.shape[1]
    k = h.sum(axis=1)
    model_pp = (2 * k * (m - k) / (m * (m - 1))).groupby(level=0).mean()
    hr = r.pivot_table(index=["case_id", "model"], columns="reader", values="high", aggfunc="first").astype(float)
    reader_pp = pd.concat([(hr[a] != hr[c]) for a, c in itertools.combinations(hr.columns, 2)],
                          axis=1).mean(axis=1).groupby(level=0).mean()
    out["model_readers"] = model_pp.loc[rsel].mean()
    out["seg_readers"] = reader_pp.loc[rsel].mean()
    out["diff_readers"] = out["seg_readers"] - out["model_readers"]
    return out


def main():
    args = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    n_boot = args[0] if args else 200
    n_seeds = args[1] if len(args) > 1 else 20
    n_jobs = args[2] if len(args) > 2 else 20
    if "--summarize-only" in sys.argv:
        df = pd.read_csv(P / "refit_bootstrap_replicates.csv")
    else:
        jobs = [(ep, b) for b in range(n_boot + 1) for ep in ("recurrence", "egfr", "os")]
        with contextlib.redirect_stdout(io.StringIO()):
            res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one_replicate)(ep, b, n_seeds) for ep, b in jobs)
        df = pd.DataFrame(res)
        df.to_csv(P / "refit_bootstrap_replicates.csv", index=False)
    cond = pd.read_csv(P / "decomposition_summary.csv")
    rows = []
    for ep, g in df.groupby("endpoint"):
        base, bt = g[g.replicate == 0].iloc[0], g[g.replicate > 0]
        for comp, col, cname in (("±1 mm", "pm1", "±1 mm"), ("AI", "ai", "AI"),
                                 ("readers", "readers", "another radiation oncologist (cohort median)")):
            c = cond[(cond.endpoint == ep) & (cond.comparison == cname)].iloc[0]
            lo, hi = np.percentile(bt[f"diff_{col}"], [2.5, 97.5])
            se = float(bt[f"diff_{col}"].std(ddof=1))
            rows.append({"endpoint": ep, "comparison": comp,
                         "difference": c.diff_rate, "refit_se": se,
                         "refit_lo": c.diff_rate - 1.96 * se, "refit_hi": c.diff_rate + 1.96 * se,
                         "refit_p": float(2 * stats.norm.sf(abs(c.diff_rate) / se)),
                         "conditional_lo": c.diff_lo, "conditional_hi": c.diff_hi,
                         "conditional_se": (c.diff_hi - c.diff_lo) / (2 * 1.96),
                         "grouped_fold_difference": base[f"diff_{col}"],
                         "percentile_lo": lo, "percentile_hi": hi,
                         "bootstrap_mean_model_rate": bt[f"model_{col}"].mean(),
                         "bootstrap_mean_segmentation_rate": bt[f"seg_{col}"].mean(), "n_boot": len(bt)})
    s = pd.DataFrame(rows)
    s.to_csv(P / "refit_bootstrap_summary.csv", index=False)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
