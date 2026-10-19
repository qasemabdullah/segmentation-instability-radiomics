"""Deaths without recurrence: censoring dates, disease-free survival and competing risks.

In the RFS endpoint, deaths without recurrence are censored at the
collection's last-known-alive date. This script
  1. reports how those deaths were censored (date of death or earlier);
  2. models disease-free survival (DFS; recurrence or death, whichever first)
     with the same nested cross-validation as OS and RFS, and tests the
     out-of-fold DFS risk score (median split, log-rank; univariable Cox);
  3. treats death without recurrence as a competing event for the RFS risk
     score: cause-specific Cox models for recurrence and for death without
     recurrence, a Fine-Gray subdistribution hazard model for recurrence
     (Geskus weighted counting-process form; 95% interval from 1,000
     patient-level bootstrap resamples), and Aalen-Johansen cumulative
     incidence of recurrence at 3 and 5 years above and below the median
     risk score.

Paper:    Methods, Results, Suppl. Table S4
Pipeline: python pipeline.py run competing_risks   (stage 'discovery')
Needs:    clinical.csv, radiomics_features.csv, survival_rfs_predictions.csv
Outputs:  data/processed/survival_dfs_{predictions,results}.csv, competing_risks.csv, competing_risks.json
Run:      python scripts/survival/run_competing_risks.py
"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nsclc_pipeline import config, survival  # noqa: E402

P = config.PROCESSED_DIR
N_BOOT = 1000


def outcomes():
    c = pd.read_csv(config.CLINICAL_OUT, parse_dates=["ct_date", "date_of_death", "date_of_last_known_alive",
                                                      "date_of_recurrence"])
    rec, dead = c.recurrence_bool.astype(bool), c.deceased.astype(bool)
    # status: 1 recurrence, 2 death without recurrence, 0 censored
    c["status"] = np.where(rec, 1, np.where(dead, 2, 0))
    end = c.date_of_recurrence.where(rec, c.date_of_death.where(dead, c.date_of_last_known_alive))
    c["time"] = (end - c.ct_date).dt.days
    return c[["case_id", "status", "time", "date_of_death", "date_of_last_known_alive", "ct_date"]]


def censoring_audit(o):
    d = o[o.status == 2]
    gap = (d.date_of_death - d.date_of_last_known_alive).dt.days
    return {"deaths_without_recurrence": int(len(d)), "censored_at_death_date": int((gap == 0).sum()),
            "censored_before_death": int((gap > 0).sum()), "gap_days_median_if_before": float(gap[gap > 0].median()),
            "gap_days_max": float(gap.max())}


def geskus(df):
    """Fine-Gray data: patients with a competing event stay in the risk set after it, with weight G(t)/G(t_i)."""
    km = KaplanMeierFitter().fit(df.time, (df.status == 0).astype(int))  # censoring distribution
    G = lambda t: float(km.survival_function_at_times(np.maximum(t - 1e-9, 0)).iloc[0])  # noqa: E731
    cut = np.sort(df.loc[df.status == 1, "time"].unique())
    tmax = df.time.max()
    rows = []
    for i, r in df.iterrows():
        if r.status != 2:
            rows.append({"id": i, "start": 0, "stop": r.time, "event": int(r.status == 1), "w": 1.0, "x": r.x})
            continue
        rows.append({"id": i, "start": 0, "stop": r.time, "event": 0, "w": 1.0, "x": r.x})
        g0, prev = G(r.time), r.time
        for t in np.append(cut[cut > r.time], tmax + 1):
            w = G(t) / g0 if g0 > 0 else 0.0
            if w > 0 and t > prev:
                rows.append({"id": i, "start": prev, "stop": t, "event": 0, "w": w, "x": r.x})
            prev = t
    return pd.DataFrame(rows)


def fine_gray(df):
    d = geskus(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = CoxTimeVaryingFitter().fit(d, event_col="event", start_col="start", stop_col="stop",
                                       weights_col="w", id_col="id", formula="x")
    return float(np.exp(m.params_["x"]))


def cox_hr(df, event):
    m = CoxPHFitter().fit(pd.DataFrame({"t": df.time, "e": event.astype(int), "x": df.x}), "t", "e")
    s = m.summary.loc["x"]
    return float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"])


def main():
    warnings.filterwarnings("ignore")
    o = outcomes()
    audit = censoring_audit(o[o.case_id.isin(pd.read_csv(config.RADIOMICS_OUT).case_id)])

    # ---- DFS: nested cross-validation as for OS and RFS
    with contextlib.redirect_stdout(io.StringIO()):
        scores = survival.run(endpoint="dfs", save=True)
    dfs = pd.read_csv(P / "survival_dfs_predictions.csv")
    hi = dfs.risk_score >= dfs.risk_score.median()
    lr = logrank_test(dfs.duration_days[hi], dfs.duration_days[~hi], dfs.event[hi], dfs.event[~hi])
    dfs["x"] = (dfs.risk_score - dfs.risk_score.mean()) / dfs.risk_score.std()
    dfs["time"] = dfs.duration_days
    hr_dfs = cox_hr(dfs, dfs.event)
    res = {"censoring": audit,
           "dfs": {"n": int(len(dfs)), "events": int(dfs.event.sum()), "concordance_mean": float(np.mean(scores)),
                   "concordance_sd": float(np.std(scores)), "logrank_p_median_split": float(lr.p_value),
                   "events_high": int(dfs.event[hi].sum()), "events_low": int(dfs.event[~hi].sum()),
                   "hr_per_sd": hr_dfs}}

    # ---- competing risks for the out-of-fold RFS risk score
    rfs = pd.read_csv(P / "survival_rfs_predictions.csv")[["case_id", "risk_score"]].merge(o, on="case_id")
    rfs = rfs[rfs.time > 0].reset_index(drop=True)
    rfs["x"] = (rfs.risk_score - rfs.risk_score.mean()) / rfs.risk_score.std()
    rows = [{"model": "cause-specific Cox, recurrence", **dict(zip(("hr", "lo", "hi", "p"), cox_hr(rfs, rfs.status == 1)))},
            {"model": "cause-specific Cox, death without recurrence",
             **dict(zip(("hr", "lo", "hi", "p"), cox_hr(rfs, rfs.status == 2)))}]
    shr = fine_gray(rfs)
    rng = np.random.default_rng(20261005)
    bs = []
    for _ in range(N_BOOT):
        b = rfs.iloc[rng.integers(0, len(rfs), len(rfs))].reset_index(drop=True)
        try:
            bs.append(fine_gray(b))
        except Exception:  # noqa: BLE001  (rare non-convergence in a resample)
            continue
    bs = np.log(bs)
    rows.append({"model": "Fine-Gray, recurrence", "hr": shr, "lo": float(np.exp(np.percentile(bs, 2.5))),
                 "hi": float(np.exp(np.percentile(bs, 97.5))),
                 "p": float(2 * min((bs <= 0).mean(), (bs >= 0).mean())), "n_boot": len(bs)})
    # cumulative incidence of recurrence by median RFS risk score
    hi = rfs.risk_score >= rfs.risk_score.median()
    for name, g in (("high", rfs[hi]), ("low", rfs[~hi])):
        aj = AalenJohansenFitter(calculate_variance=False).fit(g.time, g.status, event_of_interest=1)
        km = KaplanMeierFitter().fit(g.time, g.status == 1)
        for yrs in (3, 5):
            t = 365.25 * yrs
            rows.append({"model": f"cumulative incidence of recurrence, {name} risk, {yrs} y",
                         "aalen_johansen": float(aj.cumulative_density_.loc[:t].iloc[-1, 0]),
                         "one_minus_km": float(1 - km.survival_function_at_times(t).iloc[0])})
    cr = pd.DataFrame(rows)
    cr.to_csv(P / "competing_risks.csv", index=False)
    res["competing_risks"] = {"n": int(len(rfs)), "recurrences": int((rfs.status == 1).sum()),
                              "deaths_without_recurrence": int((rfs.status == 2).sum())}
    (P / "competing_risks.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(cr.round(3).to_string())


if __name__ == "__main__":
    main()
