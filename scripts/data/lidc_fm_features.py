"""LIDC-IDRI: foundation-model features (Pai et al. 2024) seeded at each reader's segmentation.

Paper:    Table 2
Pipeline: python pipeline.py run lidc_data   (stage 'external')

Preprocessing reproduces fmcib.preprocessing.get_transforms: resample to 1 mm
isotropic (linear), clip/scale HU -1024..3072 to 0..1, 50-mm cube centred on
the seed in LPS voxel order (z, y, x), zero padding. The network is MONAI's
ResNet50 (widen_factor=2, conv1_t_stride=2, no head), 4096 outputs. The fmcib
package itself pins torch 2.0, so the few lines needed are reproduced here.

Needs data/external/lidc/crops/*.npz with margin >= 30 mm (lidc_features.py).
Output: data/external/lidc/lidc_fm_features.csv
Run:    python scripts/data/lidc_fm_features.py [threads]
"""
from __future__ import annotations

import glob
import pathlib
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.networks.nets import resnet50

ROOT = pathlib.Path(__file__).resolve().parents[2]
LIDC = ROOT / "data" / "external" / "lidc"
WEIGHTS = ROOT / "models" / "fmcib.torch"
SIZE = 50


def load_model():
    m = resnet50(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2, feed_forward=False)
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    sd = ck["trunk_state_dict"] if "trunk_state_dict" in ck else {
        k.replace("model.backbone.", ""): v for k, v in ck["state_dict"].items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if not k.startswith("fc")], f"weights missing: {missing[:5]}"
    return m.eval()


def patch(ct: sitk.Image, seed_phys) -> np.ndarray:
    iso = sitk.Resample(ct, [int(round(n * s)) for n, s in zip(ct.GetSize(), ct.GetSpacing())], sitk.Transform(),
                        sitk.sitkLinear, ct.GetOrigin(), (1.0, 1.0, 1.0), ct.GetDirection(), -1024.0, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(iso)  # (z, y, x)
    c = [int(v) for v in iso.TransformPhysicalPointToContinuousIndex(seed_phys)][::-1]  # -> (z, y, x)
    out = np.zeros((SIZE, SIZE, SIZE), np.float32)  # 0 = -1024 HU after scaling, as fmcib's zero padding
    sl_src, sl_dst = [], []
    for ax in range(3):
        lo, hi = c[ax] - SIZE // 2, c[ax] + SIZE // 2
        a, b = max(lo, 0), min(hi, arr.shape[ax])
        sl_src.append(slice(a, b))
        sl_dst.append(slice(a - lo, a - lo + (b - a)))
    out[tuple(sl_dst)] = np.clip((arr[tuple(sl_src)] + 1024) / 4096, 0, 1)
    return out


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    model = load_model()
    out_path = LIDC / "lidc_fm_features.csv"
    done = set(pd.read_csv(out_path, usecols=["key"]).key) if out_path.exists() else set()
    files = sorted(glob.glob(str(LIDC / "crops" / "*.npz")))
    rows = []
    for i, f in enumerate(files, 1):
        d = np.load(f)
        if float(d["margin"]) < 30 if "margin" in d else True:
            continue
        pid, k = pathlib.Path(f).stem.rsplit("_n", 1)
        ct = sitk.GetImageFromArray(d["ct"].astype(np.float32))
        ct.SetSpacing(tuple(d["spacing"])), ct.SetOrigin(tuple(d["origin"])), ct.SetDirection(tuple(d["direction"]))
        batch, meta = [], []
        for r, m in zip(d["readers"], d["masks"]):
            key = f"{pid}_n{k}_r{r}"
            if key in done or m.sum() == 0:
                continue
            zz, yy, xx = np.nonzero(m)
            seed = ct.TransformContinuousIndexToPhysicalPoint((xx.mean(), yy.mean(), zz.mean()))
            batch.append(patch(ct, seed))
            meta.append({"key": key, "PatientID": pid, "nodule": int(k), "reader": int(r),
                         "seed_x": seed[0], "seed_y": seed[1], "seed_z": seed[2]})
        if not batch:
            continue
        with torch.no_grad():
            feats = model(torch.from_numpy(np.stack(batch)[:, None])).numpy()
        rows += [{**mt, **{f"fm_{j}": v for j, v in enumerate(fe)}} for mt, fe in zip(meta, feats)]
        if i % 50 == 0 or i == len(files):
            new = pd.DataFrame(rows)
            new.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
            done |= set(new.key)
            rows = []
            print(f"[{i}/{len(files)}] {len(done)} contours", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
    print("done")


if __name__ == "__main__":
    main()
