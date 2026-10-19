"""Survival (time-to-event) modeling: radiomics features -> overall
survival (OS) or recurrence-free survival (RFS), via Cox proportional
hazards, with the same nested cross-validation as modeling.py's binary
classifiers.

Binary classification throws away information: two patients who both
"didn't recur" by last follow-up aren't equally good evidence of a good
outcome if one was followed for 3 years and the other for 3 months. Cox
regression uses the actual follow-up duration and correctly treats
patients still alive (OS) or still recurrence-free (RFS) at last contact
as censored (not as "no event" -- they may still have one later) instead
of discarding or mislabeling them.

OS duration is computed from CT date, not the dataset's own
time_to_death_days column, which differs from (date_of_death - ct_date)
by up to several hundred days and appears to be measured from surgery.
The time from the scan is also the natural reference for an imaging model. RFS duration is computed the same way, from
CT date to date_of_recurrence (event) or date_of_last_known_alive
(censored), using the same case-verified date fields.

Evaluated with Harrell's concordance index (C-index): the probability
that, for a random pair of patients, the one predicted higher-risk really
did have the worse outcome. 0.5 = no better than chance, 1.0 = perfect,
same interpretation scale as AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from . import config
from .modeling import NON_FEATURE_COLS  # single source of truth -- see modeling.py for why


def load_survival_dataset(radiomics_path=None, endpoint: str = "os"):
    if endpoint not in ("os", "rfs", "dfs"):
        raise ValueError(f"endpoint must be 'os', 'rfs' or 'dfs', got {endpoint!r}")

    if isinstance(radiomics_path, pd.DataFrame):
        rad = radiomics_path
    else:
        rad = pd.read_csv(radiomics_path or config.RADIOMICS_OUT)
    feature_cols = [c for c in rad.columns if c not in NON_FEATURE_COLS]

    date_cols = ["ct_date", "date_of_death", "date_of_last_known_alive", "date_of_recurrence"]
    clin = pd.read_csv(config.CLINICAL_OUT, parse_dates=date_cols)
    keep_cols = ["case_id", "deceased", "recurrence_bool"] + date_cols
    df = rad.merge(clin[keep_cols], on="case_id", how="left")

    if endpoint == "os":
        end_date = df["date_of_death"].where(df["deceased"], df["date_of_last_known_alive"])
        df["event"] = df["deceased"].astype(bool)
    elif endpoint == "rfs":  # event = recurrence; censored at last known alive if recurrence-free
        # (deaths without recurrence are censored: 19 of 26 at the date of death, which the collection
        # records as the last known alive date, and 7 at an earlier last-contact date)
        end_date = df["date_of_recurrence"].where(df["recurrence_bool"], df["date_of_last_known_alive"])
        df["event"] = df["recurrence_bool"].astype(bool)
    else:  # dfs: event = recurrence or death, whichever came first
        rec, dead = df["recurrence_bool"].astype(bool), df["deceased"].astype(bool)
        end_date = df["date_of_recurrence"].where(rec, df["date_of_death"].where(dead, df["date_of_last_known_alive"]))
        df["event"] = rec | dead
    df["duration_days"] = (end_date - df["ct_date"]).dt.days

    df = df.dropna(subset=["duration_days"])
    df = df[df["duration_days"] > 0]
    # Same reasoning as modeling.py: fix row order so results depend on
    # --seed and only --seed, not on incidental radiomics-extraction order.
    df = df.sort_values("case_id").reset_index(drop=True)

    y = Surv.from_arrays(event=df["event"].to_numpy(), time=df["duration_days"].to_numpy())
    X = df[feature_cols]
    return X, y, df["case_id"].to_numpy(), df[["duration_days", "event"]]


def nested_cv_concordance(X, y, case_ids, outer_folds=5, inner_folds=5, random_state=0):
    """One concordance index per outer fold, computed on a held-out fold
    the model (including its ridge penalty, tuned only on that fold's
    training split) never saw during selection. Also returns out-of-fold
    risk scores for every patient, for a Kaplan-Meier plot stratified by
    predicted risk."""
    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    param_grid = {"clf__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

    fold_scores = []
    chosen_alpha = []
    oof_rows = []
    for fold_i, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", CoxPHSurvivalAnalysis()),
            ]
        )
        inner = KFold(n_splits=inner_folds, shuffle=True, random_state=random_state)
        search = GridSearchCV(pipe, param_grid, cv=inner, n_jobs=1)
        search.fit(X_train, y_train)

        score = search.score(X_test, y_test)  # sksurv default .score() = concordance index
        fold_scores.append(score)
        chosen_alpha.append(search.best_params_["clf__alpha"])

        risk = search.predict(X_test)
        for cid, yt, r in zip(case_ids[test_idx], y_test, risk):
            oof_rows.append(
                {
                    "case_id": cid,
                    "fold": fold_i,
                    "event": bool(yt[0]),
                    "duration_days": float(yt[1]),
                    "risk_score": float(r),
                }
            )
        print(
            f"  outer fold {fold_i}: C-index={score:.3f} "
            f"(n_test={len(test_idx)}, events={int(y_test['event'].sum())}, alpha={search.best_params_['clf__alpha']})"
        )

    return np.array(fold_scores), chosen_alpha, pd.DataFrame(oof_rows)


def run(
    outer_folds: int = 5,
    inner_folds: int = 5,
    random_state: int = 0,
    radiomics_path=None,
    endpoint: str = "os",
    result_suffix: str | None = None,
    save: bool = True,
):
    X, y, case_ids, durations = load_survival_dataset(radiomics_path, endpoint=endpoint)
    n_events = int(y["event"].sum())
    label = {"os": "overall survival", "rfs": "recurrence-free survival", "dfs": "disease-free survival"}[endpoint]
    print(
        f"Target: {label} (Cox PH) -- n={len(y)}, events={n_events} "
        f"({n_events / len(y):.1%}), median follow-up={durations['duration_days'].median():.0f} days, "
        f"features={X.shape[1]}"
    )

    if min(n_events, len(y) - n_events) < outer_folds:
        raise ValueError(
            f"Only {n_events} events -- too few for {outer_folds}-fold CV. Reduce --outer-folds."
        )

    scores, chosen_alpha, oof = nested_cv_concordance(
        X, y, case_ids, outer_folds=outer_folds, inner_folds=inner_folds, random_state=random_state
    )
    print(f"\nNested CV concordance index: {scores.mean():.3f} +/- {scores.std():.3f}  (across {outer_folds} outer folds)")
    print("Per-fold:", [round(s, 3) for s in scores])
    print(
        "\nCaveat: same small-n caveat as modeling.py's classifiers -- treat "
        "as exploratory, not a validated prognostic model."
    )

    if save:
        suffix = result_suffix or {"os": "survival", "rfs": "survival_rfs", "dfs": "survival_dfs"}[endpoint]
        out_path = config.PROCESSED_DIR / f"{suffix}_results.csv"
        pd.DataFrame({"fold": range(1, len(scores) + 1), "concordance": scores, "alpha": chosen_alpha}).to_csv(
            out_path, index=False
        )
        oof_path = config.PROCESSED_DIR / f"{suffix}_predictions.csv"
        oof.to_csv(oof_path, index=False)
        print(f"\nSaved fold-level results -> {out_path}")
        print(f"Saved out-of-fold risk scores -> {oof_path}")
    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--endpoint", choices=["os", "rfs", "dfs"], default="os")
    args = parser.parse_args()
    run(outer_folds=args.outer_folds, inner_folds=args.inner_folds, random_state=args.seed, endpoint=args.endpoint)
