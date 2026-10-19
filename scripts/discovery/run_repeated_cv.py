"""Repeated nested cross-validation (20 fold seeds): model variability and the re-allocation null.

Paper:    Fig. 3c-e, g, Table 3, Suppl. Table S6
Pipeline: python pipeline.py run repeated_cv   (stage 'discovery')

  * seed-averaged sensitivity curves (no dependence on one fold split);
  * a null distribution for the range statistic: the max - min over five runs
    that all use the ORIGINAL segmentations and differ only in fold seed, to compare
    with the observed range across the five boundary positions (same seed);
  * a training-noise reference for reclassification: how often out-of-fold
    decisions for the same patient differ between two seeds on the same
    segmentations, against how often they differ between the original and a
    shifted segmentation under the same seed.

Outputs: data/processed/repeated_cv_scores.csv, repeated_cv_oof.csv.gz,
         repeated_cv_summary.csv
Run:     python scripts/discovery/run_repeated_cv.py [n_seeds] [n_jobs]
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, modeling, survival  # noqa: E402

P = config.PROCESSED_DIR
MM = [-2, -1, 0, 1, 2]


def table(mm):
    if mm == 0:
        return pd.read_csv(config.RADIOMICS_OUT)
    pert = pd.read_csv(P / "perturbation_features.csv")
    exp = pd.read_csv(config.RADIOMICS_OUT)[["case_id", "ct_series_uid", "seg_series_uid"]]
    return pert[pert.perturbation_mm == mm].drop(columns="perturbation_mm").merge(exp, on="case_id")


def one(ep, mm, seed):
    import warnings
    warnings.filterwarnings("ignore")
    t = table(mm)
    if ep == "os":
        X, y, ids, _ = survival.load_survival_dataset(t, endpoint="os")
        scores, _, oof = survival.nested_cv_concordance(X, y, ids, random_state=seed)
        oof = oof.rename(columns={"risk_score": "pred"})
    else:
        X, y, ids, num, cat, gene, emb = modeling.load_dataset(ep, "radiomics", t)
        scores, _, oof = modeling.nested_cv_auc(X, y, ids, num, cat, gene, emb, random_state=seed)
        oof = oof.rename(columns={"y_proba": "pred"})
    oof = oof[["case_id", "pred"]].assign(endpoint=ep, mm=mm, seed=seed)
    return {"endpoint": ep, "mm": mm, "seed": seed, "score": float(np.mean(scores))}, oof


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    jobs = [(ep, mm, s) for ep in ("recurrence", "egfr", "os") for mm in MM for s in range(n_seeds)]
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(*j) for j in jobs)
    sc = pd.DataFrame([r[0] for r in res])
    oof = pd.concat([r[1] for r in res], ignore_index=True)
    sc.to_csv(P / "repeated_cv_scores.csv", index=False)
    oof.to_csv(P / "repeated_cv_oof.csv.gz", index=False)

    rng = np.random.RandomState(0)
    rows = []
    for ep, g in sc.groupby("endpoint"):
        wide = g.pivot(index="seed", columns="mm", values="score")
        obs = (wide.max(axis=1) - wide.min(axis=1)).to_numpy()  # per-seed swing across boundaries
        base = wide[0].to_numpy()
        null = np.array([np.ptp(rng.choice(base, 5, replace=False)) for _ in range(5000)])
        o = oof[oof.endpoint == ep]
        hi = o.assign(high=o.groupby(["mm", "seed"]).pred.transform(lambda v: v >= v.median()))
        h = hi.pivot_table(index="case_id", columns=["mm", "seed"], values="high")
        seeds = sorted(g.seed.unique())
        noise = np.mean([(h[(0, a)] != h[(0, b)]).mean() for a, b in itertools.combinations(seeds, 2)])
        cont = {mm: np.mean([(h[(0, s)] != h[(mm, s)]).mean() for s in seeds]) for mm in MM if mm != 0}
        rows.append({"endpoint": ep, "n_seeds": len(seeds), "score_original_mean": base.mean(),
                     "score_original_sd_across_seeds": base.std(),
                     **{f"score_{mm:+d}": wide[mm].mean() for mm in MM},
                     "swing_observed_median": np.median(obs), "swing_null_median": np.median(null),
                     "swing_null_95pct": np.percentile(null, 95),
                     "p_null_ge_observed": float((null[:, None] >= obs[None, :]).mean()),
                     "flips_training_noise": noise, **{f"flips_contour_{mm:+d}": v for mm, v in cont.items()}})
    s = pd.DataFrame(rows)
    s.to_csv(P / "repeated_cv_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).T.to_string())


if __name__ == "__main__":
    main()
