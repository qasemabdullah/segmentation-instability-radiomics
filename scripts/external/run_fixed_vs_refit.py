"""Performance at each segmentation with the model kept fixed or re-developed.

Two questions are answered by the boundary-adjustment analyses:

  model fixed        a model developed on expert segmentations is applied to
                     another segmentation of the held-out patients (deployment:
                     what happens when the segmentation at use differs from the
                     one at development);
  model re-developed the whole model is developed and evaluated on the
                     adjusted segmentations (development: would a study drawn
                     differently have reached a different model and estimate).

The re-developed scores are those already reported (Table 3; Lung1 Fig. 7e).
This script adds the fixed-model scores from the same folds: in the
discovery cohort from the 20-repetition same-patient predictions
(decomposition_predictions.csv.gz), and in Lung1 by training on the original
segmentations in each outer fold (primary allocation) and scoring the
held-out patients at every boundary position.

Paper:    Table 3, Results, Suppl. Table S8
Pipeline: python pipeline.py run fixed_vs_refit   (stage 'external')
Needs:    decomposition_predictions.csv.gz, repeated_cv_scores.csv, data/external/lung1/*
Outputs:  data/processed/fixed_vs_refit.csv
Run:      python scripts/external/run_fixed_vs_refit.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from nsclc_pipeline import config, modeling, survival  # noqa: E402

P = config.PROCESSED_DIR
SEG = {-2: "-2mm", -1: "-1mm", 0: "expert", 1: "+1mm", 2: "+2mm"}


def discovery():
    dec = pd.read_csv(P / "decomposition_predictions.csv.gz")
    dec = dec[dec.cohort == "discovery"]
    exp = pd.read_csv(config.RADIOMICS_OUT)
    refit = pd.read_csv(P / "repeated_cv_scores.csv").groupby(["endpoint", "mm"]).score.mean()
    rows = []
    for ep, g in dec.groupby("endpoint"):
        if ep == "os":
            _, _, ids, dd = survival.load_survival_dataset(exp, endpoint="os")
            dd.index = ids
        else:
            _, y, ids, *_ = modeling.load_dataset(ep, "radiomics", exp)
            yy = pd.Series(y, index=ids)
        for mm, seg in SEG.items():
            s = g[g.segmentation == seg]
            per_seed = []
            for _, gs in s.groupby("seed"):
                fs = []
                for _, gf in gs.groupby("fold"):
                    if ep == "os":
                        d = dd.loc[gf.case_id]
                        fs.append(concordance_index_censored(d.event.astype(bool).to_numpy(),
                                                             d.duration_days.to_numpy(), gf.score.to_numpy())[0])
                    else:
                        fs.append(roc_auc_score(yy.loc[gf.case_id], gf.score))
                per_seed.append(np.mean(fs))
            rows.append({"cohort": "discovery", "endpoint": ep, "mm": mm, "fixed": np.mean(per_seed),
                         "fixed_sd_over_seeds": np.std(per_seed), "re_developed": refit.loc[(ep, mm)],
                         "repetitions": len(per_seed), "per_seed": per_seed})
    # the median over repetitions of the per-repetition range, the statistic of Table 3
    df = pd.DataFrame(rows)
    for ep, g in df.groupby("endpoint"):
        rg = np.ptp(np.vstack(g.per_seed.to_numpy()), axis=0)
        df.loc[g.index, "fixed_median_range"] = float(np.median(rg))
    return df.drop(columns="per_seed").to_dict("records")


def lung1():
    import scripts.external.lung1_models as L
    V, y, ids, feat, _ = L.load()
    per = {mm: [] for mm in SEG}
    for tr, te in KFold(5, shuffle=True, random_state=0).split(V[0]):
        m = L.fit_cox(V[0].iloc[tr], y[tr])
        for mm in SEG:
            per[mm].append(concordance_index_censored(y[te]["event"], y[te]["time"], m.predict(V[mm].iloc[te]))[0])
    fc = pd.read_csv(L.L1 / "lung1_fold_concordance.csv").groupby("mm").c.mean()
    return [{"cohort": "Lung1", "endpoint": "os", "mm": mm, "fixed": np.mean(per[mm]), "re_developed": fc.loc[mm],
             "repetitions": 1} for mm in SEG]


def main():
    warnings.filterwarnings("ignore")
    out = pd.DataFrame(discovery() + lung1())
    rc = pd.read_csv(P / "repeated_cv_summary.csv").set_index("endpoint").swing_observed_median
    out["re_developed_median_range"] = out.endpoint.map(rc).where(out.cohort == "discovery")
    out.to_csv(P / "fixed_vs_refit.csv", index=False)
    rng = out.groupby(["cohort", "endpoint"]).agg(fixed_range=("fixed", lambda v: v.max() - v.min()),
                                                  re_developed_range=("re_developed", lambda v: v.max() - v.min()))
    pd.set_option("display.width", 200)
    print(out.round(3).to_string())
    print(rng.round(3))


if __name__ == "__main__":
    main()
