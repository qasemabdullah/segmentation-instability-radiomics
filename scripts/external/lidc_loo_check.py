"""LIDC-IDRI: each segmentation judged only by the other readers' ratings (removes circularity).

Paper:    Fig. 5a-f, Suppl. Table S17, Suppl. Fig. S9
Pipeline: python pipeline.py run lidc_models   (stage 'external')

In the main LIDC analysis the label is the consensus (median) rating of all
readers, including the reader whose segmentation is being scored. Here every
segmentation is labelled only by the OTHER readers of that nodule (mean rating;
>= 3.5 malignant, <= 2.5 benign, otherwise excluded), both when it is used for
training and when it is scored. A reader's own opinion therefore never grades
their own segmentation. Reported per feature type (radiomics, foundation model):

  * AUC of each segmentation's score against its leave-that-reader-out label;
  * error rate of nodules with the same decision from every reader vs a
    reader-dependent decision (error judged per segmentation against its own
    leave-that-reader-out label);
  * the same comparison against the recorded diagnosis (small subset).

Decision disagreement and the +-1 mm flag need no labels and are unchanged.
Output: data/external/lidc/lidc_loo_check.json
Run:    python scripts/external/lidc_loo_check.py [n_seeds]
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config  # noqa: E402
from scripts.external.lidc_models import C_GRID, LIDC, fit, load  # noqa: E402


def one_seed(f, feats, seed, grid):
    rng = np.random.default_rng(seed)
    base = f[f.perturbation_mm == 0]
    # nodules eligible = at least one reader with a leave-out label; stratify by consensus-free majority
    nod = base.groupby(["PatientID", "nodule"]).y_loo.mean().dropna().round().rename("strat").reset_index()
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    out = []
    for tr, te in outer.split(nod, nod.strat, nod.PatientID):
        tr_n, te_n = nod.iloc[tr], nod.iloc[te]
        cand = base.merge(tr_n[["PatientID", "nodule"]], on=["PatientID", "nodule"]).dropna(subset=["y_loo"])
        pick = cand.groupby(["PatientID", "nodule"]).sample(1, random_state=int(rng.integers(1e9)))
        model = fit(pick[feats].to_numpy(), pick.y_loo.to_numpy(), pick.PatientID.to_numpy(), seed, grid)
        te_rows = base.merge(te_n[["PatientID", "nodule"]], on=["PatientID", "nodule"])
        out.append(te_rows[["PatientID", "nodule", "reader", "y_loo"]].assign(
            p=model.predict_proba(te_rows[feats].to_numpy())[:, 1], seed=seed))
    return pd.concat(out, ignore_index=True)


def summarize(o, path_lab):
    o = o.assign(d=o.p >= 0.5)
    res = []
    for s, g in o.groupby("seed"):
        lab = g.dropna(subset=["y_loo"])
        auc = roc_auc_score(lab.y_loo, lab.p)
        per = g.groupby(["PatientID", "nodule"]).d.agg(["min", "max"])
        unstable = (per["min"] != per["max"]).rename("unstable")
        lab = lab.merge(unstable.reset_index(), on=["PatientID", "nodule"])
        lab["wrong"] = lab.d != lab.y_loo.astype(bool)
        # nodule-level error = share of its contours misclassified against their own leave-out label
        nerr = lab.groupby(["PatientID", "nodule", "unstable"]).wrong.mean().reset_index()
        r = {"auc_loo": auc, "pairwise_free_nodules_unstable": float(unstable.mean()),
             "error_stable": float(nerr[~nerr.unstable].wrong.mean()),
             "error_unstable": float(nerr[nerr.unstable].wrong.mean())}
        pth = g.merge(path_lab, on="PatientID")
        if len(pth):
            pn = pth.groupby("PatientID").agg(call=("d", "mean"), y=("path_y", "first"))
            pn["unstable"] = (pn.call > 0) & (pn.call < 1)
            pn["wrong"] = (pn.call >= 0.5) != pn.y
            r.update({"path_n": int(len(pn)), "path_auc": roc_auc_score(pth.path_y, pth.p) if pth.path_y.nunique() == 2 else np.nan,
                      "path_n_unstable": int(pn.unstable.sum()),
                      "path_error_stable": float(pn[~pn.unstable].wrong.mean()),
                      "path_error_unstable": float(pn[pn.unstable].wrong.mean()) if pn.unstable.any() else np.nan})
        res.append(r)
    return {k: float(np.nanmean(v)) for k, v in pd.DataFrame(res).items()}


def main():
    n_seeds = int(next((a for a in sys.argv[1:] if a.isdigit()), 10))
    c = pd.read_parquet(config.idc_clinical_table("lidc_idri_clinical"))
    c = c[c.diagnosis_at_the_patient_level.isin([1, 2, 3])]
    path_lab = pd.DataFrame({"PatientID": c.dicom_patient_id, "path_y": (c.diagnosis_at_the_patient_level > 1).astype(int)})
    out = {}
    for source in ("radiomics", "fm"):
        f, feats = load(source)
        # load() keeps nodules with a clear consensus; the leave-out labels are per contour
        single = f[f.perturbation_mm == 0].groupby("PatientID").nodule.nunique()
        pl = path_lab[path_lab.PatientID.isin(single[single == 1].index)]
        oof = pd.concat(Parallel(n_jobs=min(n_seeds, 4))(delayed(one_seed)(f, feats, s, C_GRID[source])
                                                         for s in range(n_seeds)))
        out[source] = {"n_contours_labelled": int(oof[oof.seed == 0].y_loo.notna().sum()), **summarize(oof, pl)}
        print(source, json.dumps(out[source], indent=1), flush=True)
    (LIDC / "lidc_loo_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
