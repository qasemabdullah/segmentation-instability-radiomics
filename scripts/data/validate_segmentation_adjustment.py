"""Geometric validation of the ±1/±2-mm segmentation adjustment on synthetic spheres.

Paper:    Fig. 3a-b
Pipeline: python pipeline.py run segmentation_adjustment   (stage 'data')

Run: python scripts/data/validate_segmentation_adjustment.py
"""
import pathlib
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from nsclc_pipeline.perturbation import perturb_mask  # noqa: E402

for sp, r in [((1.0, 1.0, 1.0), 20.0), ((0.98, 0.98, 3.0), 20.0), ((0.98, 0.98, 3.0), 8.0), ((0.7, 0.7, 1.25), 12.0)]:
    shape = (int(2.8 * r / sp[2]) + 8, int(2.8 * r / sp[1]) + 8, int(2.8 * r / sp[0]) + 8)
    z, y, x = np.indices(shape).astype(float)
    c = [s / 2 for s in shape]
    dist = np.sqrt(((x - c[2]) * sp[0]) ** 2 + ((y - c[1]) * sp[1]) ** 2 + ((z - c[0]) * sp[2]) ** 2)
    arr = (dist <= r).astype(np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(sp)
    v0 = arr.sum()
    assert (sitk.GetArrayFromImage(perturb_mask(img, 0)) == arr).all()
    line = []
    for mm in (-2, -1, 1, 2):
        v = sitk.GetArrayFromImage(perturb_mask(img, mm)).sum()
        # effective radius from volume (voxelised sphere of the same grid)
        r_eff = r * (v / v0) ** (1 / 3)
        line.append(f"{mm:+d}mm: {100 * (v / v0 - 1):+5.1f}% (ideal {100 * (((r + mm) / r) ** 3 - 1):+5.1f}%, shift {r_eff - r:+.2f} mm)")
    print(f"spacing {sp}, r={r:>4} mm | " + " | ".join(line))
