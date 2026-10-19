"""Render the legacy-style Supplementary Figures (S6, S7, S8, S11, S13) into outputs/figures/supplementary/.

Paper:    Suppl. Figs S1-S13
Pipeline: python pipeline.py run supplementary_figures   (stage 'figures')

It reuses each figure's original generating script unchanged: those
scripts draw their own "Supplementary Fig. Sx | ..." headline, so here
Figure.suptitle is disabled and fig.text() calls that start with a
title/sub-title prefix are skipped, then the saved PNG is copied under
its new number and trimmed of any leftover white margin.

Run: python scripts/figures/render_supplementary_figures.py
Output: outputs/figures/supplementary/<key>.png (the manuscript build, kept outside this repository, assigns the final S-numbers)
"""
from __future__ import annotations

import pathlib
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "figures" / "supplementary"
FIG = OUT / "_parts"  # full renders by the original scripts, before trimming
FIG.mkdir(parents=True, exist_ok=True)

TITLE_PREFIXES = ("Supplementary Fig", "Supplementary Figure", "Figure S", "Fig. S", "Fig. 5 |",
                  "Every AUC, concordance", "Every count below", "Exploratory, descriptive analysis")

Figure.suptitle = lambda self, *a, **k: None
_orig_text = Figure.text


def _text(self, x, y, s, *a, **k):
    if isinstance(s, str) and s.lstrip().startswith(TITLE_PREFIXES):
        return None
    return _orig_text(self, x, y, s, *a, **k)


Figure.text = _text

# (manuscript key, source png(s) written by the original script). Final S-numbers are
# assigned by the manuscript build (outside this repository) from the order of first citation in the text.
# Drawn directly into outputs/figures/supplementary/ by their own scripts (not handled here):
#   flow, cv                 scripts/figures/supp_patient_flow_validation.py
#   cohort, dist, tertile    scripts/figures/supp_cohort_distributions_tertiles.py
#   cvdetail                 scripts/figures/main_figures.py roc
#   signal                   scripts/figures/supp_signal_experiment.py
ORDER = [
    ("vignette", ["supp_fig_s11_vignette"]),
    ("clusters", ["figure5_phenotype_clusters"]),
    ("habitats", ["supp_fig_s12_ct_habitats"]),
    ("implem", ["supp_fig_s7_implementation_robustness"]),
    ("stabfilt", ["supp_fig_s5_stability_filtering"]),
]


def trim(path, pad=24):
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height)))
    im.save(path, dpi=(300, 300))


def render_all():
    for mod in ("supp_patient_vignettes", "supp_phenotypes", "supp_ct_habitats",
                "supp_extractor_classifier", "supp_stability_filtering"):
        print(f"-- {mod}")
        __import__(f"scripts.figures.{mod}", fromlist=["main"]).main()


def collect():
    for key, sources in ORDER:
        for i, src in enumerate(sources):
            suffix = "abc"[i] if len(sources) > 1 else ""
            dst = OUT / f"{key}{suffix}.png"
            shutil.copyfile(FIG / f"{src}.png", dst)
            trim(dst)
            print(f"{key}{suffix}: {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    if "--collect-only" not in sys.argv:
        render_all()
    collect()
