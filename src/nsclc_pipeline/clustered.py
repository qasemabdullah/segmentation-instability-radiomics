"""Patient-clustered inference for a binary flag against a binary outcome.

Rows from the same patient (several readers, reader pairs or nodules) are
correlated, so tests that treat them as independent (e.g. Fisher's exact
test over reader pairs) understate uncertainty. Here the odds ratio is
computed over all rows, and its 95% interval and two-sided P value come
from a bootstrap that resamples whole patients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _odds(flag, out):
    a = np.sum(flag & out) + 0.5
    b = np.sum(flag & ~out) + 0.5
    c = np.sum(~flag & out) + 0.5
    d = np.sum(~flag & ~out) + 0.5
    return (a * d) / (b * c)


def cluster_bootstrap_or(flag, outcome, cluster, n_boot=2000, seed=0):
    """Odds ratio (Haldane-corrected), patient-bootstrap 95% interval and two-sided P (log OR = 0)."""
    flag, outcome = np.asarray(flag, bool), np.asarray(outcome, bool)
    cl = pd.factorize(np.asarray(cluster))[0]
    rows = [np.flatnonzero(cl == k) for k in range(cl.max() + 1)]
    rng = np.random.default_rng(seed)
    logs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rows[k] for k in rng.integers(0, len(rows), len(rows))])
        logs[b] = np.log(_odds(flag[idx], outcome[idx]))
    lo, hi = np.exp(np.percentile(logs, [2.5, 97.5]))
    p = min(1.0, 2 * min((logs <= 0).mean(), (logs >= 0).mean()))
    return {"odds_ratio": float(_odds(flag, outcome)), "or_lo": float(lo), "or_hi": float(hi),
            "p_cluster_bootstrap": float(p), "clusters": len(rows), "rows": len(flag)}
