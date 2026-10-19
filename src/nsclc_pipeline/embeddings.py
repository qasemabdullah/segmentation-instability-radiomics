"""CNN image embeddings as an alternative to handcrafted radiomic features.

ResNet18 pretrained on ImageNet, applied to a single 2D slice through the
tumor centre; the 512-dim activation before the classifier is the
embedding. This is a transfer-learning baseline that runs on a CPU, not a
3D medical-imaging foundation model (e.g. Med3D, CT-FM). The embeddings go
through the same nested cross-validation as the radiomic features
(see docs/data_and_gui.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torchvision
from torchvision import transforms

from . import config, radiomics, viz

EMBEDDING_DIM = 512
_MODEL = None
_PREPROCESS = None


def _get_model():
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()  # strip the 1000-class classifier head
        model.eval()
        _MODEL = model
        _PREPROCESS = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return _MODEL, _PREPROCESS


def _central_tumor_crop(ct_array, mask_array, pad_frac=0.3, out_size=224):
    """Find the axial slice with the most tumor pixels, crop a padded
    square around the tumor's bounding box on that slice, window to lung
    HU range, resize to out_size, and replicate to 3 channels -- the
    input format ResNet18 expects."""
    per_slice_count = mask_array.sum(axis=(1, 2))
    z = int(np.argmax(per_slice_count))

    m = mask_array[z] > 0
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    side = int(max(h, w) * (1 + pad_frac))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2

    H, W = ct_array.shape[1], ct_array.shape[2]
    ry0, ry1 = max(cy - side // 2, 0), min(cy + side // 2, H)
    rx0, rx1 = max(cx - side // 2, 0), min(cx + side // 2, W)

    crop = ct_array[z, ry0:ry1, rx0:rx1]
    level, width = viz.WINDOW_PRESETS["Lung"]
    gray = viz.window_slice(crop, level, width)  # uint8, 0-255

    img = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0  # 1x1xHxW
    img = torch.nn.functional.interpolate(img, size=(out_size, out_size), mode="bilinear", align_corners=False)
    img = img.repeat(1, 3, 1, 1)  # 1x3xHxW, replicate grayscale to RGB channels
    return img, z


def extract_embedding(case_id: str, series_index: pd.DataFrame | None = None) -> dict:
    series_index = series_index if series_index is not None else pd.read_csv(config.SERIES_INDEX_OUT)
    ct_dir, mask_path, ct_series_uid, seg_series_uid = radiomics.resolve_ct_and_mask(case_id, series_index)

    ct_image = viz.load_ct_volume(ct_dir)
    mask_image = viz.load_mask_on_ct(mask_path, ct_image)
    ct_array = sitk.GetArrayFromImage(ct_image)
    mask_array = sitk.GetArrayFromImage(mask_image)

    if mask_array.sum() == 0:
        raise ValueError(f"{case_id}: mask has zero voxels after resampling onto the CT grid")

    img, z = _central_tumor_crop(ct_array, mask_array)
    model, normalize = _get_model()
    with torch.no_grad():
        x = normalize(img)
        vec = model(x).squeeze(0).numpy()

    row = {"case_id": case_id, "central_slice": int(z)}
    row.update({f"embed_{i:03d}": float(v) for i, v in enumerate(vec)})
    return row


def extract_all(case_ids=None, progress_cb=None, out_path=None) -> pd.DataFrame:
    out_path = out_path or config.PROCESSED_DIR / "embeddings.csv"
    series_index = pd.read_csv(config.SERIES_INDEX_OUT)
    seg_case_ids = sorted(series_index[series_index["modality"] == "SEG"]["case_id"].unique())
    targets = case_ids or seg_case_ids

    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    done_ids = set(done["case_id"]) if len(done) else set()
    rows = done.to_dict("records")

    for i, case_id in enumerate(targets, start=1):
        if progress_cb:
            progress_cb(i, len(targets), case_id)
        if case_id in done_ids:
            print(f"[{i}/{len(targets)}] {case_id}: already done, skipping")
            continue
        print(f"[{i}/{len(targets)}] {case_id}")
        try:
            rows.append(extract_embedding(case_id, series_index))
        except Exception as e:  # noqa: BLE001 -- keep going across the batch
            print(f"FAILED {case_id}: {type(e).__name__}: {e}")
            continue
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = extract_all()
    print(f"\nExtracted embeddings for {len(df)} patients -> {config.PROCESSED_DIR / 'embeddings.csv'}")
