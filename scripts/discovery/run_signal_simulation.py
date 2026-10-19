"""Does the effect of the outline depend on how much signal a model carries?

Paper:    Suppl. Fig. S10, Suppl. Table S20
Pipeline: python pipeline.py run signal_experiment   (stage 'discovery')

Controlled experiment in the discovery cohort (144 patients, real CT, real
expert and ±1-mm outlines). A synthetic binary outcome is generated from the
expert-outline features at a chosen signal strength:

    latent = standardized(features at 0 mm) · w,  w ~ N(0, I), rescaled to s.d. 1
    y      = 1 for the 40 patients (28%, as for recurrence) with the highest
             latent + sigma · noise

sigma = inf gives an outcome unrelated to the images; smaller sigma gives more
signal. For every level, 5 random directions w and 20 fold seeds are run
through the same nested cross-validation as the paper (L2 logistic regression,
balanced class weights, inner grid over C). Each outer-fold model scores its
held-out patients from the expert outline and from the −1-mm and +1-mm
outlines, and each patient is classified against the median training score.

Per level we report the nested cross-validated AUC, model variability (share
of patients classified differently by two seeds, expert outline), outline
reclassification (share classified differently from a ±1-mm outline by the
same model), and error rates of outline-dependent and outline-stable
classifications. Real recurrence and EGFR outcomes are run the same way as a
reference.

With --latent-average the latent score is instead computed from the mean of
the standardized features of the -1-mm, expert and +1-mm outlines, so that
the expert outline is not, by construction, the one that measures the
outcome-generating property (sensitivity analysis).

With --signal=<structure> the latent score has a different structure
(sensitivity analyses of the signal model; all use the expert segmentation):
  linear      a random linear combination of all 26 panel features (default)
  sparse      a random linear combination of 3 randomly chosen panel features
  stable      a random linear combination of the 13 most stable panel features
              (the half used in the stability-filtering analysis)
  nonlinear   5 random pairwise products and 3 squared terms of the
              standardized panel features, with no linear term, so that the
              logistic model can approximate the outcome only partly
  external    a random linear combination of the first 10 principal
              components of the ImageNet CNN embedding of the tumor's largest
              slice: an image property outside the radiomic panel, which the
              model can learn only through its correlation with the panel

Outputs: data/processed/signal_simulation_runs{_avg|_<structure>}.csv, signal_simulation_summary{...}.csv
Run:     python scripts/discovery/run_signal_simulation.py [n_seeds] [n_draws] [n_jobs] [--latent-average] [--signal=<structure>]
"""
from __future__ import annotations

import itertools
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, modeling  # noqa: E402

P = config.PROCESSED_DIR
SIGMAS = [np.inf, 3.0, 1.5, 1.0, 0.6, 0.3]
GRID = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}


def features():
    base = pd.read_csv(config.RADIOMICS_OUT)
    X0, y_rec, ids, num, *_ = modeling.load_dataset("recurrence", "radiomics", base)
    pert = pd.read_csv(P / "perturbation_features.csv")
    Xs = {0: X0[num].reset_index(drop=True)}
    for mm in (-1, 1):
        t = pert[pert.perturbation_mm == mm].set_index("case_id").loc[ids, num]
        Xs[mm] = t.reset_index(drop=True)
    Xe, y_egfr, ids_e, num_e, *_ = modeling.load_dataset("egfr", "radiomics", base)
    return Xs, np.asarray(y_rec), np.asarray(ids), num, np.asarray(y_egfr), np.asarray(ids_e)


def stable_half(num):
    from nsclc_pipeline import perturbation
    st = perturbation.feature_stability_summary(pd.read_csv(P / "perturbation_features.csv"), -1, 1)
    st = st[st.feature.isin(num)].reset_index(drop=True)
    return st.iloc[len(st) // 2:].feature.tolist()


def embedding_pcs(ids, k=10):
    from sklearn.decomposition import PCA
    e = pd.read_csv(P / "embeddings.csv").set_index("case_id").loc[ids].filter(like="embed_")
    z = (e - e.mean()) / e.std(ddof=0).replace(0, 1)
    return PCA(k, random_state=0).fit_transform(z.to_numpy())


def outcome(Xs, sigma, draw, n_pos, average=False, signal="linear", extra=None):
    rng = np.random.default_rng(1000 + draw)
    X0 = Xs[0]
    mu, sd = X0.median(), X0.std(ddof=0)
    std = lambda X: ((X.fillna(mu) - X0.fillna(mu).mean()) / sd).to_numpy()  # noqa: E731  expert-scale z-scores
    Z = np.mean([std(Xs[m]) for m in (-1, 0, 1)], axis=0) if average else std(X0)
    if signal == "linear":
        latent = Z @ rng.standard_normal(Z.shape[1])
    elif signal == "sparse":
        cols = rng.choice(Z.shape[1], 3, replace=False)
        latent = Z[:, cols] @ rng.standard_normal(3)
    elif signal == "stable":
        cols = [list(X0.columns).index(c) for c in extra["stable"]]
        latent = Z[:, cols] @ rng.standard_normal(len(cols))
    elif signal == "nonlinear":
        Zc = np.clip(Z, -3, 3)
        pairs = [rng.choice(Z.shape[1], 2, replace=False) for _ in range(5)]
        sq = rng.choice(Z.shape[1], 3, replace=False)
        terms = [Zc[:, a] * Zc[:, b] for a, b in pairs] + [Zc[:, j] ** 2 - 1 for j in sq]
        latent = np.column_stack(terms) @ rng.standard_normal(len(terms))
    elif signal == "external":
        latent = extra["pcs"] @ rng.standard_normal(extra["pcs"].shape[1])
    else:
        raise SystemExit(f"unknown signal structure {signal}")
    latent = (latent - latent.mean()) / latent.std()
    noisy = rng.standard_normal(len(latent)) if np.isinf(sigma) else latent + sigma * rng.standard_normal(len(latent))
    y = np.zeros(len(latent), int)
    y[np.argsort(-noisy)[:n_pos]] = 1
    return y


def run_seed(Xs, y, num, seed):
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    cls = {mm: np.zeros(len(y), int) for mm in Xs}
    aucs = []
    for tr, te in outer.split(Xs[0], y):
        search = GridSearchCV(modeling._build_pipeline(num, [], []), GRID, scoring="roc_auc",
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=1)
        search.fit(Xs[0].iloc[tr], y[tr])
        thr = np.median(search.predict_proba(Xs[0].iloc[tr])[:, 1])
        for mm, X in Xs.items():
            p = search.predict_proba(X.iloc[te])[:, 1]
            cls[mm][te] = (p >= thr).astype(int)
            if mm == 0:
                aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), cls


def one(Xs, y, num, label, sigma, draw, n_seeds):
    warnings.filterwarnings("ignore")
    runs = [run_seed(Xs, y, num, s) for s in range(n_seeds)]
    auc = np.mean([r[0] for r in runs])
    c0 = [r[1][0] for r in runs]
    variability = np.mean([(a != b).mean() for a, b in itertools.combinations(c0, 2)])
    outline = np.mean([((r[1][0] != r[1][-1]).mean() + (r[1][0] != r[1][1]).mean()) / 2 for r in runs])
    flag = np.concatenate([((r[1][0] != r[1][-1]) | (r[1][0] != r[1][1])) for r in runs])
    err = np.concatenate([(r[1][0] != y) for r in runs])
    return {"outcome": label, "sigma": sigma, "draw": draw, "n": len(y), "events": int(y.sum()), "auc": auc,
            "model_variability": variability, "outline_reclassification": outline,
            "ratio_outline_to_variability": outline / variability if variability > 0 else np.nan,
            "flag_rate": flag.mean(), "error_if_outline_dependent": err[flag].mean(),
            "error_if_outline_stable": err[~flag].mean()}


def main():
    argv = [a for a in sys.argv if not a.startswith("--")]
    n_seeds = int(argv[1]) if len(argv) > 1 else 20
    n_draws = int(argv[2]) if len(argv) > 2 else 5
    n_jobs = int(argv[3]) if len(argv) > 3 else 6
    average = "--latent-average" in sys.argv
    signal = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--signal=")), "linear")
    sfx = "_avg" if average else ("" if signal == "linear" else f"_{signal}")
    Xs, y_rec, ids, num, y_egfr, ids_e = features()
    n_pos = int(y_rec.sum())
    extra = {"stable": stable_half(num) if signal == "stable" else None,
             "pcs": embedding_pcs(ids) if signal == "external" else None}
    jobs = [(Xs, outcome(Xs, s, d, n_pos, average, signal, extra), num, "synthetic", s, d, n_seeds)
            for s in SIGMAS for d in range(n_draws)]
    jobs.append((Xs, y_rec, num, "recurrence (real)", np.nan, 0, n_seeds))
    keep = np.isin(ids, ids_e)
    order = pd.Series(range(len(ids_e)), index=ids_e).loc[ids[keep]].to_numpy()
    Xe = {mm: X[keep].reset_index(drop=True) for mm, X in Xs.items()}
    jobs.append((Xe, y_egfr[order], num, "EGFR (real)", np.nan, 0, n_seeds))
    res = pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(*j) for j in jobs))
    res.to_csv(P / f"signal_simulation_runs{sfx}.csv", index=False)
    cols = ["auc", "model_variability", "outline_reclassification", "ratio_outline_to_variability",
            "flag_rate", "error_if_outline_dependent", "error_if_outline_stable"]
    summ = res.groupby(["outcome", "sigma"], dropna=False)[cols].mean().reset_index().sort_values("auc")
    summ.to_csv(P / f"signal_simulation_summary{sfx}.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
