"""Multi-reader NSCLC cohort: reclassification between radiation oncologists and the ±1-mm flag.

Paper:    Fig. 5e, Suppl. Table S18
Pipeline: python pipeline.py run reader_classifications   (stage 'readers')

NSCLC-Radiomics-Interobserver1: 21 patients, each contoured manually by five
radiation oncologists. For every reader segmentation we extract the 26 features at
the original boundary and at -1 and +1 mm. Fixed models, never fitted on these
patients' segmentations, then score every segmentation:
  OS          Cox model trained on all Lung1 original segmentations (same
              institution and CT protocol as the reader cohort)
  recurrence, EGFR   logistic models trained on all 144 Stanford expert segmentations
Decisions use the cohort median of the models' scores on the readers'
original segmentations (a within-cohort high/low split, because absolute scores do
not transfer between institutions).

Reported: pairwise reader-to-reader decision disagreement, share of patients on
whom at least two readers' segmentations disagree, risk-rank correlation between
readers, and whether a reader's own +-1 mm flag predicts disagreement with the
other readers (odds ratio over reader-patient rows, with a 95% interval and P value from
2,000 bootstrap resamples of patients, because a patient's five readers are correlated).

Outputs: data/processed/reader_features_pm1.csv, reader_decisions.csv,
         reader_decision_summary.csv
Run:     python scripts/readers/run_reader_decisions.py
"""
from __future__ import annotations

import itertools
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import clustered, config, modeling, perturbation  # noqa: E402
from nsclc_pipeline.modeling import NON_FEATURE_COLS  # noqa: E402

NIFTI = ROOT / "data" / "interim" / "interobserver_nifti"
P = config.PROCESSED_DIR


def extract(pid):
    from scripts.data.lung1_features import crop_to_tumour
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(2)
    ct_full = sitk.ReadImage(str(NIFTI / f"{pid}_ct.nii.gz"))
    rows = []
    for f in sorted(NIFTI.glob(f"{pid}_obs*.nii.gz")):
        reader = int(f.name.split("_obs")[1].split(".")[0])
        m = sitk.ReadImage(str(f))
        m = sitk.Cast(sitk.Resample(m, ct_full, sitk.Transform(), sitk.sitkNearestNeighbor, 0) > 0, sitk.sitkUInt8)
        ct, mask = crop_to_tumour(ct_full, m)
        arr = sitk.GetArrayFromImage(ct)
        for mm in (-1, 0, 1):
            r = perturbation.extract_perturbed_features(pid, arr, ct, mask, mm)
            if r is not None:
                r["reader"] = reader
                rows.append(r)
    return rows


def features():
    out = P / "reader_features_pm1.csv"
    if out.exists():
        return pd.read_csv(out)
    pids = sorted({f.name.split("_")[0] for f in NIFTI.glob("*_ct.nii.gz")})
    with Pool(4) as pool:
        rows = [r for rs in pool.map(extract, pids) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def fitted_models(feat):
    import scripts.external.lung1_models as L
    V, y, ids, _, _ = L.load()
    models = {"os": L.fit_cox(V[0][feat], y)}
    for tgt in ("recurrence", "egfr"):
        X, yy, cid, num, cat, gene, emb = modeling.load_dataset(tgt, "radiomics")
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        s = GridSearchCV(modeling._build_pipeline(num, cat, gene, emb), {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
                         scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=1)
        s.fit(X, yy)
        models[tgt] = s
    return models


def score(model, ep, X):
    return model.predict(X) if ep == "os" else model.predict_proba(X)[:, 1]


def main():
    df = features()
    feat = [c for c in pd.read_csv(config.RADIOMICS_OUT, nrows=1).columns if c not in NON_FEATURE_COLS]
    models = fitted_models(feat)
    dec_rows, summ = [], []
    for ep, m in models.items():
        d = df.copy()
        d["score"] = score(m, ep, d[feat])
        thr = d[d.perturbation_mm == 0].score.median()
        d["high"] = d.score >= thr
        base = d[d.perturbation_mm == 0].pivot(index="case_id", columns="reader", values="high")
        sc = d[d.perturbation_mm == 0].pivot(index="case_id", columns="reader", values="score")
        pairs = list(itertools.combinations(base.columns, 2))
        pair_dis = np.mean([(base[a] != base[b]).mean() for a, b in pairs])
        any_dis = float((base.nunique(axis=1) > 1).mean())
        rho = np.mean([spearmanr(sc[a], sc[b]).correlation for a, b in pairs])
        # flag: a reader's own -1/0/+1 decisions disagree; outcome: that reader disagrees with >=1 other reader
        piv = d.pivot_table(index=["case_id", "reader"], columns="perturbation_mm", values="high")
        flag = (piv.nunique(axis=1) > 1)
        disagree = pd.Series({(c, r): bool((base.loc[c].drop(r) != base.loc[c, r]).any())
                              for c in base.index for r in base.columns})
        both = pd.concat([flag.rename("flag"), disagree.rename("dis")], axis=1).dropna()
        tab = [[int((both.flag & both.dis).sum()), int((both.flag & ~both.dis).sum())],
               [int((~both.flag & both.dis).sum()), int((~both.flag & ~both.dis).sum())]]
        cb = clustered.cluster_bootstrap_or(both.flag.to_numpy(), both.dis.to_numpy(),
                                            both.index.get_level_values(0), seed=11)
        summ.append({"endpoint": ep, "patients": base.shape[0], "pairwise_disagreement": pair_dis,
                     "patients_with_any_disagreement": any_dis, "mean_reader_rank_rho": rho,
                     "flag_rate": float(both.flag.mean()),
                     "disagree_if_flagged": tab[0][0] / max(sum(tab[0]), 1),
                     "disagree_if_not_flagged": tab[1][0] / max(sum(tab[1]), 1), **cb})
        d["endpoint"] = ep
        dec_rows.append(d[["endpoint", "case_id", "reader", "perturbation_mm", "score", "high"]])
    pd.concat(dec_rows).to_csv(P / "reader_decisions.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(P / "reader_decision_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
