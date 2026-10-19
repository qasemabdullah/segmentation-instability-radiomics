"""Mitigation: averaging over segmentations and training on deliberately varied segmentations.

Paper:    Fig. 6e, Suppl. Table S22
Pipeline: python pipeline.py run averaging   (stage 'ai')

Four strategies are compared on the patients with both an expert and an AI
(BAMF nnU-Net) segmentation. Models are always trained on expert segmentations only;
the AI segmentation is used only at test time, as in a deployment where the
segmentation tool differs from the one used in development.

  baseline   train on expert segmentations; predict from one segmentation
  tte        test-time segmentation ensemble: average the prediction over the
             segmentation and its -1 mm and +1 mm versions
  aug        training-time segmentation augmentation: train on each training
             patient's expert segmentation at -2, -1, 0, +1 and +2 mm
             (grouped by patient in the inner cross-validation)
  aug+tte    both

For each strategy and endpoint: discrimination from the expert segmentation and
from the AI segmentation, and the share of held-out patients whose decision
(above/below the training median) differs between the two segmentations. A
paired bootstrap over patients gives a 95% CI for the change in flip rate
relative to baseline.

Outputs: data/processed/contour_robust_predictions.csv
         data/processed/contour_robust_summary.csv
Run:     python scripts/mitigation/run_segmentation_averaging.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, modeling, survival  # noqa: E402
from nsclc_pipeline.modeling import NON_FEATURE_COLS  # noqa: E402

P = config.PROCESSED_DIR
STRATS = ("baseline", "tte", "aug", "aug+tte")


def variant_tables():
    """{(contour, mm): features indexed by case_id} for expert and AI contours."""
    exp = pd.read_csv(config.RADIOMICS_OUT)
    pert = pd.read_csv(P / "perturbation_features.csv")
    ai = pd.read_csv(P / "radiomics_features_bamf_ai.csv")
    aip = pd.read_csv(P / "radiomics_features_bamf_ai_perturb.csv")
    feat = [c for c in exp.columns if c not in NON_FEATURE_COLS]
    V = {("expert", 0): exp.set_index("case_id")[feat], ("ai", 0): ai.set_index("case_id")[feat]}
    for mm in (-2, -1, 1, 2):
        V[("expert", mm)] = pert[pert.perturbation_mm == mm].set_index("case_id")[feat]
    for mm in (-1, 1):
        V[("ai", mm)] = aip[aip.perturbation_mm == mm].set_index("case_id")[feat]
    ids = sorted(set.intersection(*[set(v.index) for v in V.values()]))
    return V, ids, feat


def stack(V, ids, mms):
    X = pd.concat([V[("expert", m)].loc[ids] for m in mms])
    groups = np.concatenate([np.arange(len(ids))] * len(mms))
    return X, groups


def fit(kind, X, y, groups):
    if kind == "clf":
        pipe = modeling._build_pipeline(list(X.columns), [], [], [])
        grid = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=1)
    else:
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("clf", CoxPHSurvivalAnalysis())])
        search = GridSearchCV(pipe, {"clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, cv=GroupKFold(5), n_jobs=1)
    search.fit(X, y, groups=groups)
    return search


def score(kind, model, X):
    return model.predict_proba(X)[:, 1] if kind == "clf" else model.predict(X)


def run_endpoint(name, kind, V, ids):
    if kind == "clf":
        base = V[("expert", 0)].loc[ids].reset_index()
        X0, y, cid, *_ = modeling.load_dataset(name, "radiomics", base)
        splitter = StratifiedKFold(5, shuffle=True, random_state=0).split(X0, y)
    else:
        base = V[("expert", 0)].loc[ids].reset_index()
        X0, y, cid, _ = survival.load_survival_dataset(base, endpoint="os")
        splitter = KFold(5, shuffle=True, random_state=0).split(X0)
    cid = list(cid)
    rows = []
    for fold, (tr, te) in enumerate(splitter, 1):
        tr_ids, te_ids = [cid[i] for i in tr], [cid[i] for i in te]
        ytr = y[tr]
        models = {}
        Xa, ga = stack(V, tr_ids, [0])
        models["plain"] = fit(kind, Xa, ytr, ga)
        Xb, gb = stack(V, tr_ids, [-2, -1, 0, 1, 2])
        models["aug"] = fit(kind, Xb, np.concatenate([ytr] * 5), gb)
        for strat in STRATS:
            m = models["aug" if strat.startswith("aug") else "plain"]
            thr = np.median(score(kind, m, V[("expert", 0)].loc[tr_ids]))
            pred = {}
            for contour in ("expert", "ai"):
                mms = (-1, 0, 1) if strat.endswith("tte") else (0,)
                pred[contour] = np.mean([score(kind, m, V[(contour, mm)].loc[te_ids]) for mm in mms], axis=0)
            for j, i in enumerate(te):
                r = {"endpoint": name, "strategy": strat, "fold": fold, "case_id": cid[i], "threshold": thr,
                     "p_expert": pred["expert"][j], "p_ai": pred["ai"][j]}
                if kind == "clf":
                    r["y"] = int(y[i])
                else:
                    r["event"], r["time"] = bool(y[i][0]), float(y[i][1])
                rows.append(r)
        print(f"  {name} fold {fold} done", flush=True)
    return pd.DataFrame(rows)


def perf(d, col):
    out = []
    for _, f in d.groupby("fold"):
        if "y" in f and f["y"].notna().all():
            out.append(roc_auc_score(f.y, f[col]))
        else:
            out.append(concordance_index_censored(f.event.astype(bool), f.time, f[col])[0])
    return np.mean(out), np.std(out)


def flips(d):
    return ((d.p_expert >= d.threshold) != (d.p_ai >= d.threshold)).to_numpy()


def summarize(pred, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for ep, d in pred.groupby("endpoint"):
        base = d[d.strategy == "baseline"].set_index("case_id")
        fb = flips(base)
        for strat in STRATS:
            s = d[d.strategy == strat].set_index("case_id").loc[base.index]
            fs = flips(s)
            diffs = []
            for _ in range(n_boot):
                idx = rng.randint(0, len(fs), len(fs))
                diffs.append(fs[idx].mean() - fb[idx].mean())
            me, se = perf(s.reset_index(), "p_expert")
            ma, sa = perf(s.reset_index(), "p_ai")
            rows.append({"endpoint": ep, "strategy": strat, "n": len(s), "score_expert": me, "sd_expert": se,
                         "score_ai": ma, "sd_ai": sa, "flip_rate": fs.mean(),
                         "flip_change_vs_baseline": fs.mean() - fb.mean(),
                         "ci_lo": np.percentile(diffs, 2.5), "ci_hi": np.percentile(diffs, 97.5),
                         "spearman_expert_ai": pd.Series(s.p_expert.values).corr(pd.Series(s.p_ai.values), method="spearman")})
    return pd.DataFrame(rows)


def main():
    V, ids, _ = variant_tables()
    print(f"{len(ids)} patients with expert and AI contours at every shift")
    pred = pd.concat([run_endpoint("recurrence", "clf", V, ids), run_endpoint("egfr", "clf", V, ids),
                      run_endpoint("os", "cox", V, ids)], ignore_index=True)
    pred.to_csv(P / "contour_robust_predictions.csv", index=False)
    summ = summarize(pred)
    summ.to_csv(P / "contour_robust_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
