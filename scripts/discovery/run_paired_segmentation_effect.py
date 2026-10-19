"""Paired analysis of the segmentation effect on performance across the 20 repetitions.

The re-allocation null of run_repeated_cv.py compares the range across five
segmentations with the range across five randomly drawn fold seeds, which
treats variation across segmentation settings and across training
allocations as interchangeable. Here the segmentation effect is estimated
within each repetition, where folds, patients and model settings are
identical and only the segmentation differs:

  1. seed-paired: for each of the 20 repetitions, the fold-averaged score at
     each adjusted segmentation minus that at the expert segmentation;
     summarized by its mean, the number of repetitions with a positive
     difference, a Wilcoxon signed-rank test, and a Friedman test of any
     difference among the five segmentations (repetitions as blocks);
  2. patient-paired: because repetitions reuse the same patients, they are
     not independent samples, so the same differences (averaged over the 20
     repetitions) are also given 95% intervals from 2,000 patient-level
     bootstrap resamples of the stored held-out predictions, with the same
     patients drawn at every segmentation and repetition (fold-stratified).

Paper:    Table 3, Suppl. Table S7
Pipeline: python pipeline.py run paired_segmentation_effect   (stage 'discovery')
Needs:    repeated_cv_scores.csv, repeated_cv_oof.csv.gz (run_repeated_cv.py)
Outputs:  data/processed/paired_segmentation_effect.csv
Run:      python scripts/discovery/run_paired_segmentation_effect.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, modeling, survival  # noqa: E402

P = config.PROCESSED_DIR
MM = [-2, -1, 0, 1, 2]
N_BOOT = 2000


def folds(ep, seed):
    """Outer-fold membership, reconstructed exactly as in nested CV."""
    from sklearn.model_selection import KFold, StratifiedKFold
    exp = pd.read_csv(config.RADIOMICS_OUT)
    if ep == "os":
        X, y, ids, _ = survival.load_survival_dataset(exp, endpoint="os")
        sp = KFold(5, shuffle=True, random_state=seed).split(X)
    else:
        X, y, ids, *_ = modeling.load_dataset(ep, "radiomics", exp)
        sp = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    f = np.empty(len(ids), int)
    for k, (_, te) in enumerate(sp):
        f[te] = k
    return pd.Series(f, index=ids), (y if ep != "os" else y)


def auc(y, p):
    r = stats.rankdata(p)
    n1 = y.sum()
    if n1 == 0 or n1 == len(y):
        return np.nan
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def cindex(t, e, r):
    """Harrell's concordance (ties in risk count 1/2), as in sksurv."""
    ti, tj = t[:, None], t[None, :]
    comp = (ti < tj) & e[:, None].astype(bool)
    ri, rj = r[:, None], r[None, :]
    num = ((ri > rj) & comp).sum() + 0.5 * ((ri == rj) & comp).sum()
    return num / comp.sum() if comp.any() else np.nan


def main():
    sc = pd.read_csv(P / "repeated_cv_scores.csv")
    oof = pd.read_csv(P / "repeated_cv_oof.csv.gz")
    rng = np.random.default_rng(20261005)
    rows = []
    for ep, g in sc.groupby("endpoint"):
        wide = g.pivot(index="seed", columns="mm", values="score")
        seeds = list(wide.index)
        fr = stats.friedmanchisquare(*[wide[m].to_numpy() for m in MM])
        # stored held-out predictions as arrays: [seed, mm, patient]
        o = oof[oof.endpoint == ep].pivot_table(index="case_id", columns=["seed", "mm"], values="pred")
        ids = o.index.to_numpy()
        pred = np.stack([[o[(s, m)].to_numpy() for m in MM] for s in seeds])
        fold = np.stack([folds(ep, s)[0].loc[ids].to_numpy() for s in seeds])
        if ep == "os":
            _, y, sid, dd = survival.load_survival_dataset(pd.read_csv(config.RADIOMICS_OUT), endpoint="os")
            dd.index = sid
            t, e = dd.loc[ids, "duration_days"].to_numpy(), dd.loc[ids, "event"].to_numpy()
            metric = lambda idx, p: cindex(t[idx], e[idx], p[idx])  # noqa: E731
        else:
            _, y, sid, *_ = modeling.load_dataset(ep, "radiomics", pd.read_csv(config.RADIOMICS_OUT))
            yy = pd.Series(y, index=sid).loc[ids].to_numpy()
            metric = lambda idx, p: auc(yy[idx], p[idx])  # noqa: E731

        def fold_avg(si, mi, idx):
            return np.nanmean([metric(idx[fold[si][idx] == k], pred[si, mi]) for k in range(5)])

        # check: the fold-averaged score from stored predictions reproduces the recorded score
        full = np.arange(len(ids))
        assert abs(fold_avg(0, MM.index(0), full) - wide.loc[seeds[0], 0]) < 1e-6, 'fold reconstruction failed'
        # bootstrap: the same patient draw is used at every segmentation and repetition
        boot = np.empty((N_BOOT, len(MM)))
        for b in range(N_BOOT):
            idx = rng.integers(0, len(ids), len(ids))
            boot[b] = [np.mean([fold_avg(si, mi, idx) for si in range(len(seeds))]) for mi in range(len(MM))]
        base_b = boot[:, MM.index(0)]
        for mm in MM:
            if mm == 0:
                continue
            d = (wide[mm] - wide[0]).to_numpy()
            w = stats.wilcoxon(d) if np.any(d != 0) else None
            db = boot[:, MM.index(mm)] - base_b
            rows.append({"endpoint": ep, "segmentation": f"{mm:+d} mm", "mean_difference": d.mean(),
                         "sd_over_repetitions": d.std(ddof=1), "repetitions_positive": int((d > 0).sum()),
                         "repetitions": len(d), "sign_test_p": stats.binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue,
                         "wilcoxon_p": w.pvalue if w else 1.0,
                         "patient_bootstrap_lo": np.percentile(db, 2.5), "patient_bootstrap_hi": np.percentile(db, 97.5),
                         "patient_bootstrap_p": 2 * min((db <= 0).mean(), (db >= 0).mean()),
                         "friedman_chi2": fr.statistic, "friedman_p": fr.pvalue})
        rng_b = boot.max(axis=1) - boot.min(axis=1)
        rows.append({"endpoint": ep, "segmentation": "range (5 segmentations)",
                     "mean_difference": (wide.max(axis=1) - wide.min(axis=1)).mean(),
                     "patient_bootstrap_lo": np.percentile(rng_b, 2.5), "patient_bootstrap_hi": np.percentile(rng_b, 97.5),
                     "friedman_chi2": fr.statistic, "friedman_p": fr.pvalue})
        print(ep, "done")
    out = pd.DataFrame(rows)
    out.to_csv(P / "paired_segmentation_effect.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
