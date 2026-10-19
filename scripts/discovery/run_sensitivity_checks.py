"""Robustness checks: borderline patients, AI design range, Lung1 reordering and calibration.

Paper:    Results, Suppl. Tables S15, S24, S26
Pipeline: python pipeline.py run sensitivity_checks   (stage 'external')

M4  Flips driven by patients near the threshold? Decision flips after excluding
    patients whose original score lies between the 40th and 60th percentile
    of their fold, and the share whose risk percentile moves by >= 25 points.
M5  AI comparison within the tool's design range (expert tumour volume up to
    that of a 30-mm sphere, 14.1 cm3).
M7  Why a 1-mm erosion reorders Lung1 patients while dilation does not:
    feature change under erosion vs dilation, each feature's contribution to
    the change in risk, and the relation of the risk change to tumour size.
M8  Calibration (intercept, slope, Brier score) of the classification models
    at every boundary position.

Output: data/processed/reviewer_checks.json (+ printed summary)
Run:    python scripts/discovery/run_sensitivity_checks.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config  # noqa: E402

P = config.PROCESSED_DIR
OUT = {}


def pct_in_fold(df, ref_col, col):
    out = np.empty(len(df))
    for _, idx in df.groupby("fold").indices.items():
        ref = np.sort(df[ref_col].to_numpy()[idx])
        out[idx] = np.searchsorted(ref, df[col].to_numpy()[idx], side="right") / len(ref) * 100
    return out


def margin_metrics(df, a, b):
    pa, pb = pct_in_fold(df, a, a), pct_in_fold(df, a, b)
    ha, hb = df[a] >= df.threshold, df[b] >= df.threshold
    far = (pa < 40) | (pa > 60)
    return {"flips_all": float((ha != hb).mean()), "flips_outside_40_60pct": float((ha != hb)[far].mean()),
            "n_outside": int(far.sum()), "moved_ge25_percentile_points": float((np.abs(pb - pa) >= 25).mean())}


def m4_m5():
    d = pd.read_csv(P / "bamf_paired_predictions.csv")
    vol = pd.read_csv(config.RADIOMICS_OUT).set_index("case_id").shape_volume_mm3 / 1000
    ag = pd.read_csv(P / "bamf_agreement.csv")
    ag = ag[ag.source == "ai"].assign(vol=lambda x: x.case_id.map(vol))
    res4, res5 = {}, {}
    for ep, g in d.groupby("endpoint"):
        g = g.reset_index(drop=True)
        res4[f"stanford_{ep}_ai"] = margin_metrics(g, "p_expert", "p_ai")
        res4[f"stanford_{ep}_-1mm"] = margin_metrics(g, "p_expert", "p_-1mm")
        small = g.case_id.map(vol) <= 14.137
        gs = g[small]
        res5[ep] = {"n": int(small.sum()),
                    "flip_ai_small": float(((gs.p_expert >= gs.threshold) != (gs.p_ai >= gs.threshold)).mean()),
                    "flip_ai_large": float(((g[~small].p_expert >= g[~small].threshold) != (g[~small].p_ai >= g[~small].threshold)).mean())}
    s = ag[ag.vol <= 14.137]
    res5["agreement_small"] = {"n": len(s), "detected": int(s.detected.sum()),
                               "dice_median": float(s[s.detected].dice.median()),
                               "dice_iqr": s[s.detected].dice.quantile([.25, .75]).round(2).tolist(),
                               "volume_ratio_median": float(s[s.detected].volume_ratio.median())}
    OUT["M4_margin"], OUT["M5_in_scope"] = res4, res5


def lung1_deploy():
    import scripts.external.lung1_models as L
    V, y, ids, feat, _ = L.load()
    rows, contrib = [], []
    for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(V[0]), 1):
        m = L.fit_cox(V[0].iloc[tr], y[tr])
        pipe = m.best_estimator_
        thr = np.median(m.predict(V[0].iloc[tr]))
        pred = {mm: m.predict(V[mm].iloc[te]) for mm in (-1, 0, 1)}
        z = {mm: pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(V[mm].iloc[te])) for mm in (-1, 0, 1)}
        coef = pipe.named_steps["clf"].coef_
        for mm in (-1, 1):
            c = (z[mm] - z[0]) * coef
            contrib.append(pd.DataFrame(np.abs(c), columns=feat).assign(mm=mm))
        for j, i in enumerate(te):
            rows.append({"fold": k, "PatientID": ids[i], "threshold": thr, "p_expert": pred[0][j],
                         "p_-1mm": pred[-1][j], "p_+1mm": pred[1][j]})
    return pd.DataFrame(rows), pd.concat(contrib), V, feat


def m4_lung1_m7():
    d, contrib, V, feat = lung1_deploy()
    OUT["M4_margin"]["lung1_os_-1mm"] = margin_metrics(d, "p_expert", "p_-1mm")
    OUT["M4_margin"]["lung1_os_+1mm"] = margin_metrics(d, "p_expert", "p_+1mm")
    vol = V[0].set_index(V[0].index)["shape_volume_mm3"]
    d = d.set_index("PatientID")
    dr = (d["p_-1mm"] - d["p_expert"]).abs()
    removed = 1 - V[-1].loc[d.index, "shape_volume_mm3"] / V[0].loc[d.index, "shape_volume_mm3"]
    rel = lambda a, b: ((V[b][feat] - V[a][feat]).abs() / V[a][feat].abs().replace(0, np.nan)).median()  # noqa: E731
    top = contrib.groupby("mm").median().T
    OUT["M7_mechanism"] = {
        "rho_abs_risk_change_vs_volume": float(spearmanr(dr, vol.loc[d.index]).correlation),
        "rho_abs_risk_change_vs_fraction_removed": float(spearmanr(dr, removed).correlation),
        "median_fraction_volume_removed_-1mm": float(removed.median()),
        "median_fraction_volume_added_+1mm": float((V[1].loc[d.index, "shape_volume_mm3"] / V[0].loc[d.index, "shape_volume_mm3"] - 1).median()),
        "top_features_rel_change_erosion": rel(0, -1).sort_values(ascending=False).head(6).round(3).to_dict(),
        "top_features_rel_change_dilation": rel(0, 1).sort_values(ascending=False).head(6).round(3).to_dict(),
        "top_risk_contributions_erosion": top[-1].sort_values(ascending=False).head(6).round(3).to_dict(),
        "top_risk_contributions_dilation": top[1].sort_values(ascending=False).head(6).round(3).to_dict(),
        "median_abs_risk_contribution_total": {"-1": float(contrib[contrib.mm == -1][feat].sum(1).median()),
                                               "+1": float(contrib[contrib.mm == 1][feat].sum(1).median())},
    }


def m8():
    res = {}
    for tgt in ("recurrence", "egfr"):
        for mm, sfx in [(-2, "__perturb_m2mm"), (-1, "__perturb_m1mm"), (0, ""), (1, "__perturb_p1mm"), (2, "__perturb_p2mm")]:
            d = pd.read_csv(P / f"modeling_predictions_{tgt}{sfx}.csv")
            p = d.y_proba.clip(1e-4, 1 - 1e-4)
            lp = np.log(p / (1 - p)).to_numpy().reshape(-1, 1)
            slope = LogisticRegression(C=1e6).fit(lp, d.y_true).coef_[0][0]
            obs, exp = d.y_true.mean(), p.mean()
            res[f"{tgt}_{mm:+d}"] = {"observed_rate": float(obs), "mean_predicted": float(exp),
                                     "calibration_in_the_large": float(obs - exp), "calibration_slope": float(slope),
                                     "brier": float(((p - d.y_true) ** 2).mean())}
    OUT["M8_calibration"] = res


def main():
    m4_m5()
    m4_lung1_m7()
    m8()
    (P / "reviewer_checks.json").write_text(json.dumps(OUT, indent=1))
    print(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    main()
