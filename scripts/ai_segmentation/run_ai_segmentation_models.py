"""AI segmentation step 2: expert versus AI segmentation (deployment and re-development).

Paper:    Fig. 3g, Fig. 4g, Suppl. Table S15
Pipeline: python pipeline.py run ai_models   (stage 'ai')

All comparisons use the same patient subset: those whose tumour the AI
detected (see ai_segmentation_masks.py), so expert and AI numbers are paired.

  1. Deployment: models are trained on expert-segmentation features inside each
     outer fold; every held-out patient is then scored twice, from the expert
     segmentation and from the AI segmentation. We report AUC / concordance for both and
     the share of patients whose reclassification (above vs below the training-set
     median). The same is done for the -1 mm and +1 mm expert perturbations,
     so AI-induced flips can be compared with synthetic ones.
  2. Re-development: the whole nested-CV pipeline refitted on AI features.
  3. Clustering: AI features assigned to the fixed expert Ward centroids.
  4. Radiologist corrections of the AI output: mask and feature change.

Outputs: data/processed/bamf_paired_predictions.csv, bamf_model_summary.csv,
         bamf_cluster_summary.csv, bamf_correction_summary.csv
Run:     python scripts/ai_segmentation/run_ai_segmentation_models.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import config, modeling, survival  # noqa: E402
from nsclc_pipeline.modeling import NON_FEATURE_COLS  # noqa: E402

P = config.PROCESSED_DIR


def tables():
    exp = pd.read_csv(config.RADIOMICS_OUT)
    ai = pd.read_csv(P / "radiomics_features_bamf_ai.csv")
    pert = pd.read_csv(P / "perturbation_features.csv")
    ids = sorted(set(ai.case_id) & set(exp.case_id))
    sub = lambda d: d[d.case_id.isin(ids)].sort_values("case_id").reset_index(drop=True)  # noqa: E731
    out = {"expert": sub(exp), "ai": sub(ai)}
    for mm in (-1, 1):
        t = pert[pert.perturbation_mm == mm].drop(columns="perturbation_mm")
        t = t.merge(exp[["case_id", "ct_series_uid", "seg_series_uid"]], on="case_id", how="left")
        out[f"{mm:+d}mm"] = sub(t)
    return out, ids


def paired_classification(target, T):
    X, y, ids, num, cat, gene, emb = modeling.load_dataset(target, "radiomics", T["expert"])
    alts = {k: modeling.load_dataset(target, "radiomics", T[k])[0].set_index(ids) for k in ("ai", "-1mm", "+1mm")}
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        pipe = modeling._build_pipeline(num, cat, gene, emb)
        search = GridSearchCV(pipe, {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}, scoring="roc_auc",
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1)
        search.fit(X.iloc[tr], y[tr])
        thr = np.median(search.predict_proba(X.iloc[tr])[:, 1])
        preds = {"expert": search.predict_proba(X.iloc[te])[:, 1]}
        for k, Xa in alts.items():
            preds[k] = search.predict_proba(Xa.loc[ids[te], X.columns])[:, 1]
        for j, i in enumerate(te):
            rows.append({"endpoint": target, "case_id": ids[i], "fold": fold, "y": int(y[i]), "threshold": thr,
                         **{f"p_{k}": v[j] for k, v in preds.items()}})
    return pd.DataFrame(rows)


def paired_survival(T, endpoint="os"):
    Xs = {k: survival.load_survival_dataset(v, endpoint=endpoint) for k, v in T.items()}
    X, y, ids, _ = Xs["expert"]
    outer = KFold(n_splits=5, shuffle=True, random_state=0)
    rows = []
    for fold, (tr, te) in enumerate(outer.split(X), 1):
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("clf", CoxPHSurvivalAnalysis())])
        search = GridSearchCV(pipe, {"clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
                              cv=KFold(5, shuffle=True, random_state=0), n_jobs=1)
        search.fit(X.iloc[tr], y[tr])
        thr = np.median(search.predict(X.iloc[tr]))
        preds = {}
        for k, (Xk, _, idk, _) in Xs.items():
            Xk = Xk.set_index(pd.Index(idk)).loc[ids[te], X.columns]
            preds[k] = search.predict(Xk)
        for j, i in enumerate(te):
            rows.append({"endpoint": endpoint, "case_id": ids[i], "fold": fold, "event": bool(y[i][0]),
                         "time": float(y[i][1]), "threshold": thr, **{f"p_{k}": v[j] for k, v in preds.items()}})
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for ep, d in df.groupby("endpoint"):
        surv = "event" in d and d["event"].notna().all()
        for k in ("expert", "ai", "-1mm", "+1mm"):
            scores = []
            for _, f in d.groupby("fold"):
                if surv:
                    scores.append(concordance_index_censored(f.event.astype(bool), f.time, f[f"p_{k}"])[0])
                else:
                    scores.append(roc_auc_score(f.y, f[f"p_{k}"]))
            flip = np.nan if k == "expert" else float(
                ((d["p_expert"] >= d.threshold) != (d[f"p_{k}"] >= d.threshold)).mean())
            out.append({"endpoint": ep, "contour": k, "n": len(d), "mean_score": np.mean(scores),
                        "sd_score": np.std(scores), "decision_flip_rate": flip})
    return pd.DataFrame(out)


def redevelop(T):
    rows = []
    for tgt in ("recurrence", "egfr"):
        for k in ("expert", "ai"):
            X, y, ids, num, cat, gene, emb = modeling.load_dataset(tgt, "radiomics", T[k])
            aucs, _, _ = modeling.nested_cv_auc(X, y, ids, num, cat, gene, emb)
            rows.append({"endpoint": tgt, "contour": k, "mean_score": aucs.mean(), "sd_score": aucs.std()})
    for k in ("expert", "ai"):
        X, y, ids, _ = survival.load_survival_dataset(T[k], endpoint="os")
        c, _, _ = survival.nested_cv_concordance(X, y, ids)
        rows.append({"endpoint": "os", "contour": k, "mean_score": c.mean(), "sd_score": c.std()})
    return pd.DataFrame(rows)


def clusters(T, ids):
    from sklearn.metrics import adjusted_rand_score
    from nsclc_pipeline.phenotypes import fit_reference_phenotypes as _fit_original_clustering
    df, feat_cols, scaler, imputer, labels, centroids, _ = _fit_original_clustering()
    orig = pd.Series(labels, index=df["case_id"])
    rows = []
    for k in ("expert", "ai", "-1mm", "+1mm"):
        t = T[k].set_index("case_id").loc[ids]
        z = scaler.transform(imputer.transform(t[feat_cols].to_numpy()))
        new = np.linalg.norm(z[:, None, :] - centroids[None], axis=2).argmin(1)
        o = orig.loc[ids].to_numpy()
        rows.append({"contour": k, "n": len(ids), "pct_reassigned": 100 * (new != o).mean(),
                     "ari": adjusted_rand_score(o, new)})
    return pd.DataFrame(rows)


def corrections():
    """Human edits to the AI output: Dice between AI and corrected masks, and
    median relative feature change, next to expert-vs-AI and synthetic +-1 mm."""
    ai = pd.read_csv(P / "radiomics_features_bamf_ai.csv").set_index("case_id")
    exp = pd.read_csv(config.RADIOMICS_OUT).set_index("case_id")
    feat = [c for c in ai.columns if c not in NON_FEATURE_COLS]
    rows = []
    for src in ("r1", "r2"):
        f = P / f"radiomics_features_bamf_{src}.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f).set_index("case_id")
        for case in r.index.intersection(ai.index):
            a = sitk.GetArrayFromImage(sitk.ReadImage(str(P / "masks_bamf" / f"{case}_ai.nii.gz"))) > 0
            b = sitk.GetArrayFromImage(sitk.ReadImage(str(P / "masks_bamf" / f"{case}_{src}.nii.gz"))) > 0
            rel_c = (r.loc[case, feat] - ai.loc[case, feat]).abs() / ai.loc[case, feat].abs().replace(0, np.nan)
            rel_e = (exp.loc[case, feat] - ai.loc[case, feat]).abs() / ai.loc[case, feat].abs().replace(0, np.nan)
            rows.append({"case_id": case, "reader": src, "dice_ai_vs_corrected": 2 * (a & b).sum() / (a.sum() + b.sum()),
                         "median_rel_change_correction": float(np.nanmedian(rel_c.astype(float))),
                         "median_rel_change_expert_vs_ai": float(np.nanmedian(rel_e.astype(float)))})
    return pd.DataFrame(rows)


def main():
    T, ids = tables()
    print(f"Paired cohort: {len(ids)} patients with an AI-detected tumour")
    paired = pd.concat([paired_classification("recurrence", T), paired_classification("egfr", T),
                        paired_survival(T, "os")], ignore_index=True)
    paired.to_csv(P / "bamf_paired_predictions.csv", index=False)
    summ = summarize(paired)
    summ["analysis"] = "deployment (trained on expert)"
    red = redevelop(T)
    red["analysis"] = "re-developed on each contour"
    allsum = pd.concat([summ, red], ignore_index=True)
    allsum.to_csv(P / "bamf_model_summary.csv", index=False)
    cl = clusters(T, ids)
    cl.to_csv(P / "bamf_cluster_summary.csv", index=False)
    cor = corrections()
    cor.to_csv(P / "bamf_correction_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(allsum.round(3).to_string())
    print(cl.round(3).to_string())
    if len(cor):
        print(cor.drop(columns="case_id").groupby("reader").median().round(3).to_string())


if __name__ == "__main__":
    main()
