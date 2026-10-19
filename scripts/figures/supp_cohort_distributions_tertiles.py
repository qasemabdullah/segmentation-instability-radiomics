"""Supplementary figures drawn directly from the data, in the style of the main
figures: cohort characteristics (cohort.png), radiomic feature distributions
(dist.png) and survival by risk tertile (tertile.png).

Paper:    Suppl. Figs S1-S13
Pipeline: python pipeline.py run supplementary_figures   (stage 'figures')
Output: outputs/figures/supplementary/{cohort,dist,tertile}.png
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import mannwhitneyu

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.figures.figure_style import BLUE, DAYS_PER_MONTH, GREEN, INK, MUTED, ORANGE, AMBER, SUB, letter, p_fmt  # noqa: E402

P = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "figures" / "supplementary"
YES_NO = {"yes": "Recurrence", "no": "No recurrence"}


def hbar(ax, counts, title, colors=None):
    n = counts.sum()
    y = np.arange(len(counts))[::-1]
    cols = colors or [BLUE, ORANGE, GREEN, AMBER, MUTED][: len(counts)]
    ax.barh(y, counts.values, color=cols, height=0.62)
    for yy, v in zip(y, counts.values):
        ax.text(v + n * 0.012, yy, f"{v} (<1%)" if 0 < 100 * v / n < 0.5 else f"{v} ({100 * v / n:.0f}%)", va="center", fontsize=6.2, color=INK)
    ax.set_yticks(y, counts.index)
    ax.set_xlim(0, counts.max() * 1.35)
    ax.set_xlabel("Patients")
    ax.set_title(title, color=INK)
    ax.tick_params(axis="y", length=0)


def cohort():
    c = pd.read_csv(P / "clinical.csv")
    unk = "Not recorded"
    panels = [
        ("Site", c.site.map({"Stanford": "Stanford", "VA": "VA Palo Alto"})),
        ("Sex", c.gender),
        ("Histology", c.histology.replace({"NSCLC NOS (not otherwise specified)": "NSCLC, NOS"})),
        ("Smoking status", c.smoking_status.replace({"Nonsmoker": "Never"})),
        ("EGFR mutation", c.egfr_mutation_status.replace({"Wildtype": "Wild type"}).fillna(unk)),
        ("KRAS mutation", c.kras_mutation_status.replace({"Wildtype": "Wild type"}).fillna(unk)),
        ("ALK rearrangement", c.alk_translocation_status.replace({"Wildtype": "Absent", "Translocated": "Present"}).fillna(unk)),
        ("Recurrence", c.recurrence.map({"yes": "Yes", "no": "No"}).fillna(unk)),
        ("Vital status", c.deceased.map({True: "Died", False: "Alive"})),
    ]
    order = {"Smoking status": ["Former", "Current", "Never"], "EGFR mutation": ["Wild type", "Mutant", unk],
             "KRAS mutation": ["Wild type", "Mutant", unk], "ALK rearrangement": ["Absent", "Present", unk],
             "Recurrence": ["No", "Yes", unk]}
    fig, axes = plt.subplots(5, 2, figsize=(7.2, 8.6), gridspec_kw={"hspace": 0.75, "wspace": 0.55, "left": 0.16, "right": 0.97, "top": 0.97, "bottom": 0.06})
    for ax, (title, s), l in zip(axes.flat, panels, "abcdefghi"):
        vc = s.value_counts()
        if title in order:
            vc = vc.reindex([k for k in order[title] if k in vc.index])
        hbar(ax, vc, title)
        letter(fig, ax, l, dx=-0.13)
    ax = axes.flat[9]
    age = c.age_at_histological_diagnosis.dropna()
    ax.hist(age, bins=np.arange(20, 95, 5), color=BLUE, edgecolor="white")
    ax.axvline(age.median(), color=INK, ls="--", lw=0.8)
    ax.text(age.median() + 1, ax.get_ylim()[1] * 0.92, f"median {age.median():.0f}", fontsize=6.2, color=INK)
    ax.set_xlabel("Age at diagnosis (years)")
    ax.set_ylabel("Patients")
    ax.set_title("Age", color=INK)
    letter(fig, ax, "j", dx=-0.13)
    fig.savefig(OUT / "cohort.png", dpi=300, facecolor="white")
    plt.close(fig)


def dist():
    rad = pd.read_csv(P / "radiomics_features.csv")
    clin = pd.read_csv(P / "clinical.csv")[["case_id", "recurrence"]]
    d = rad.merge(clin, on="case_id", how="left")
    d = d[d.recurrence.isin(["yes", "no"])].copy()
    d["vol_cm3"] = d.shape_volume_mm3 / 1000
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45, left=0.08, right=0.98, top=0.95, bottom=0.08)
    groups = [("no", BLUE), ("yes", ORANGE)]
    for k, (col, lab, logy) in enumerate([("vol_cm3", "Tumor volume (cm³)", True), ("firstorder_mean_hu", "Mean attenuation (HU)", False)]):
        ax = fig.add_subplot(gs[0, k])
        vals = [d.loc[d.recurrence == g, col].to_numpy() for g, _ in groups]
        v = ax.violinplot([np.log10(x) if logy else x for x in vals], showextrema=False, widths=0.75)
        for body, (_, c) in zip(v["bodies"], groups):
            body.set_facecolor(c)
            body.set_alpha(0.35)
            body.set_edgecolor("none")
        rng = np.random.default_rng(0)
        for i, (x, (_, c)) in enumerate(zip(vals, groups), start=1):
            yy = np.log10(x) if logy else x
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(x)), yy, s=5, color=c, alpha=0.7, lw=0)
            ax.plot([i - 0.2, i + 0.2], [np.median(yy)] * 2, color=INK, lw=1.2)
        p = mannwhitneyu(*vals).pvalue
        ax.set_xticks([1, 2], [f"{YES_NO[g]}\n(n = {len(x)})" for (g, _), x in zip(groups, vals)])
        if logy:
            ticks = [0.1, 1, 10, 100]
            ax.set_yticks(np.log10(ticks), [str(t) for t in ticks])
        ax.set_ylabel(lab)
        ax.set_title(lab.split(" (")[0], color=INK)
        ax.set_xlabel(f"Mann–Whitney {p_fmt(p)}", color=SUB, labelpad=6)
        if logy:
            ax.set_ylim(np.log10(0.5), None)
        letter(fig, ax, "ab"[k])
    for k, (col, lab) in enumerate([("shape_sphericity", "Sphericity"), ("texture_contrast", "GLCM contrast")]):
        ax = fig.add_subplot(gs[1, k])
        ax.hist(rad[col], bins=15, color=BLUE, edgecolor="white")
        ax.axvline(rad[col].median(), color=INK, ls="--", lw=0.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("Patients")
        ax.set_title(lab, color=INK)
        letter(fig, ax, "cd"[k])
    ax = fig.add_subplot(gs[:, 2])
    for g, c in groups:
        m = d.recurrence == g
        ax.scatter(d.loc[m, "vol_cm3"], d.loc[m, "texture_contrast"], s=10, color=c, alpha=0.8, lw=0, label=YES_NO[g])
    ax.set_xscale("log")
    ax.set_xlabel("Tumor volume (cm³, log scale)")
    ax.set_ylabel("GLCM contrast")
    ax.set_title("Volume and texture", color=INK)
    ax.legend(frameon=False, loc="upper right")
    letter(fig, ax, "e")
    fig.savefig(OUT / "dist.png", dpi=300, facecolor="white")
    plt.close(fig)


def tertile():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.28, "left": 0.08, "right": 0.98, "bottom": 0.16, "top": 0.88})
    cols = {"Low": BLUE, "Intermediate": AMBER, "High": ORANGE}
    for ax, (ep, lab), l in zip(axes, [("os", "Overall survival"), ("rfs", "Recurrence-free survival")], "ab"):
        t = pd.read_csv(P / f"km_tertile_{ep}.csv")
        t["months"] = t.duration_days / DAYS_PER_MONTH
        for g, c in cols.items():
            m = t.risk_tertile == g
            KaplanMeierFitter().fit(t.months[m], t.event[m], label=f"{g} risk (n = {m.sum()}, {int(t.event[m].sum())} events)").plot_survival_function(
                ax=ax, color=c, ci_show=False, show_censors=True, censor_styles={"ms": 3, "marker": "|"}, lw=1.4)
        p = multivariate_logrank_test(t.months, t.risk_tertile, t.event).p_value
        ax.text(0.97, 0.05, f"Log-rank {p_fmt(p)}", transform=ax.transAxes, ha="right", fontsize=6.4, color=INK)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Months from CT")
        ax.set_ylabel(f"{lab} probability" if ep == "rfs" else "Overall survival probability")
        ax.set_title(lab, color=INK)
        ax.legend(frameon=False, loc="lower left", fontsize=5.9)
        letter(fig, ax, l)
    fig.savefig(OUT / "tertile.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cohort()
    dist()
    tertile()
    print("Saved cohort.png, dist.png, tertile.png")
