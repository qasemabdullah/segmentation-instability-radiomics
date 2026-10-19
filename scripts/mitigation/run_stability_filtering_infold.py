"""Stability filtering with the stable and unstable halves chosen inside each outer fold.

run_stability_filtering_experiment.py ranks feature instability (median
|relative change| between the -1-mm and +1-mm segmentations) once over all
144 patients. The ranking uses no outcome, but the held-out patients still
contribute to it. Here the ranking is recomputed in every outer fold from the
training patients only, the 13 most and 13 least stable features are taken
from that ranking, and the models (inner grid search as before) are refitted
at every boundary position; the held-out patients are scored with the
features selected without them. Folds are those of the primary allocation, so
the result is directly comparable with the whole-cohort split, and the same
fold-stratified paired bootstrap (run_range_inference.py) tests the
stable-minus-unstable difference in range.

Paper:    Fig. 6d, Suppl. Fig. S11, Suppl. Table S21
Pipeline: python pipeline.py run stability_filtering   (stage 'ai')
Needs:    perturbation_features.csv
Outputs:  data/processed/{modeling_predictions_<target>,survival}__stabfilt_<half>_infold_<mm>.csv,
          data/processed/stability_filtering_infold.csv, stability_filtering_infold_splits.csv
Run:      python scripts/mitigation/run_stability_filtering_infold.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import config, modeling, perturbation, survival  # noqa: E402
from scripts.discovery import run_range_inference as sw  # noqa: E402

P = config.PROCESSED_DIR
MM = [-2, -1, 0, 1, 2]
TAG = {-2: "m2mm", -1: "m1mm", 0: "p0mm", 1: "p1mm", 2: "p2mm"}


def split_on(pert, train_ids):
    st = perturbation.feature_stability_summary(pert[pert.case_id.isin(train_ids)], -1, 1)
    st = st[~st.feature.isin(modeling.NON_FEATURE_COLS)].reset_index(drop=True)
    half = len(st) // 2
    return st.iloc[half:].feature.tolist(), st.iloc[:half].feature.tolist()  # stable, unstable


def run_endpoint(ep, pert, tables):
    if ep == "os":
        X0, y, ids, dur = survival.load_survival_dataset(tables[0], endpoint="os")
        folds = KFold(5, shuffle=True, random_state=0).split(X0)
    else:
        X0, y, ids, *_ = modeling.load_dataset(ep, "radiomics", tables[0])
        folds = StratifiedKFold(5, shuffle=True, random_state=0).split(X0, y)
    preds = {(h, mm): [] for h in ("stable", "unstable") for mm in MM}
    splits = []
    for fold, (tr, te) in enumerate(folds, start=1):
        stable, unstable = split_on(pert, ids[tr])
        splits.append({"endpoint": ep, "fold": fold, "stable": ";".join(sorted(stable))})
        for half, feats in (("stable", stable), ("unstable", unstable)):
            for mm in MM:
                t = tables[mm].set_index("case_id")
                Xtr, Xte = t.loc[ids[tr], feats], t.loc[ids[te], feats]
                if ep == "os":
                    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                                     ("clf", CoxPHSurvivalAnalysis())])
                    s = GridSearchCV(pipe, {"clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
                                     cv=KFold(5, shuffle=True, random_state=0), n_jobs=1).fit(Xtr, y[tr])
                    preds[(half, mm)].append(pd.DataFrame({"case_id": ids[te], "fold": fold,
                                                           "event": y[te]["event"], "duration_days": y[te]["time"],
                                                           "risk_score": s.predict(Xte)}))
                else:
                    pipe = modeling._build_pipeline(feats, [], [], [])
                    s = GridSearchCV(pipe, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, scoring="roc_auc",
                                     cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1).fit(Xtr, y[tr])
                    preds[(half, mm)].append(pd.DataFrame({"case_id": ids[te], "fold": fold, "y_true": y[te],
                                                           "y_proba": s.predict_proba(Xte)[:, 1]}))
    for (half, mm), parts in preds.items():
        name = (f"survival__stabfilt_{half}_infold_{TAG[mm]}_predictions" if ep == "os"
                else f"modeling_predictions_{ep}__stabfilt_{half}_infold_{TAG[mm]}")
        pd.concat(parts).to_csv(P / f"{name}.csv", index=False)
    return splits


def loader(ep, cond):
    if cond == "full":
        return sw._load_survival_condition("full") if ep == "os" else sw._load_classification_condition(ep, "full")
    if ep == "os":
        return {mm: pd.read_csv(P / f"survival__stabfilt_{cond}_{TAG[mm]}_predictions.csv")
                .sort_values("case_id").reset_index(drop=True) for mm in MM}
    return {mm: pd.read_csv(P / f"modeling_predictions_{ep}__stabfilt_{cond}_{TAG[mm]}.csv")
            .sort_values("case_id").reset_index(drop=True) for mm in MM}


def main():
    warnings.filterwarnings("ignore")
    pert = pd.read_csv(P / "perturbation_features.csv")
    exp = pd.read_csv(config.RADIOMICS_OUT)
    tables = {0: exp}
    meta = exp[["case_id", "ct_series_uid", "seg_series_uid"]]
    for mm in MM:
        if mm:
            tables[mm] = pert[pert.perturbation_mm == mm].drop(columns="perturbation_mm").merge(meta, on="case_id")
    whole_stable, _ = split_on(pert, pert.case_id.unique())
    if "--skip-fit" in sys.argv:  # reuse stored predictions and splits
        sp = pd.read_csv(P / "stability_filtering_infold_splits.csv")
    else:
        sp = pd.DataFrame([r for ep in ("egfr", "recurrence", "os") for r in run_endpoint(ep, pert, tables)])
    sp["jaccard_with_whole_cohort_split"] = [len(set(s.split(";")) & set(whole_stable)) /
                                             len(set(s.split(";")) | set(whole_stable)) for s in sp.stable]
    sp.to_csv(P / "stability_filtering_infold_splits.csv", index=False)

    rows = []
    for i, (ep, metric) in enumerate((("egfr", sw._auc_metric), ("recurrence", sw._auc_metric),
                                      ("os", sw._concordance_metric))):
        rng = np.random.RandomState(i)
        boot, obs = {}, {}
        for cond in ("full", "stable_infold", "unstable_infold"):
            lv = loader(ep, cond)
            boot[cond] = sw._bootstrap_scores(lv, metric, rng)
            means = {mm: np.mean([metric(g) for _, g in lv[mm].groupby("fold")]) for mm in MM}
            obs[cond] = means
            rows.append({"endpoint": ep, "condition": cond, **{f"score_{mm:+d}": v for mm, v in means.items()},
                         "range": max(means.values()) - min(means.values()),
                         "range_lo": np.nanpercentile(boot[cond].max(axis=1) - boot[cond].min(axis=1), 2.5),
                         "range_hi": np.nanpercentile(boot[cond].max(axis=1) - boot[cond].min(axis=1), 97.5)})
        d = ((boot["stable_infold"].max(axis=1) - boot["stable_infold"].min(axis=1))
             - (boot["unstable_infold"].max(axis=1) - boot["unstable_infold"].min(axis=1))).to_numpy()
        rows.append({"endpoint": ep, "condition": "stable minus unstable range",
                     "range": rows[-2]["range"] - rows[-1]["range"],
                     "range_lo": np.nanpercentile(d, 2.5), "range_hi": np.nanpercentile(d, 97.5),
                     "p_value": sw._two_sided_p(d)})
    out = pd.DataFrame(rows)
    out.to_csv(P / "stability_filtering_infold.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string())
    print(sp.groupby("endpoint").jaccard_with_whole_cohort_split.describe().round(2))


if __name__ == "__main__":
    main()
