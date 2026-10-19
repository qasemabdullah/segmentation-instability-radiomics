"""RadImageNet-pretrained image embeddings, the medical-imaging counterpart
of the ImageNet ResNet18 baseline in embeddings.py. RadImageNet (Mei et
al., Radiol. Artif. Intell. 2022) is a ResNet50 pretrained on about 1.35M
radiological images (CT, MR, ultrasound). It is still a 2D, single-slice
model like the ImageNet baseline; only the pretraining data differ.

Weights: Lab-Rasool/RadImageNet on Hugging Face Hub (ResNet50.pt), a
community-hosted re-release of the original RadImageNet weights,
verified to load cleanly (zero missing/unexpected keys) into a standard
torchvision ResNet50 backbone with 'backbone.' key-prefix stripped and
the avgpool+fc head removed (the released checkpoint is backbone-only,
no classification head).

Everything else (tumor localization, cropping, windowing, resizing) is
identical to embeddings.py, so the two embedding sources differ only in
which weights produced them, not in what image region they saw.
"""

from __future__ import annotations

import pandas as pd
import SimpleITK as sitk
import torch
import torchvision
from torchvision import transforms

from . import config, radiomics, viz
from .embeddings import _central_tumor_crop

EMBEDDING_DIM = 2048
_MODEL = None
_PREPROCESS = None


def _get_model():
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(repo_id="Lab-Rasool/RadImageNet", filename="ResNet50.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = {k.replace("backbone.", "", 1): v for k, v in state_dict.items()}

        resnet = torchvision.models.resnet50(weights=None)
        backbone = torch.nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc
        missing, unexpected = backbone.load_state_dict(state_dict, strict=True)
        assert not missing and not unexpected, f"RadImageNet weight loading mismatch: {missing}, {unexpected}"
        backbone.eval()

        _MODEL = backbone
        _PREPROCESS = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return _MODEL, _PREPROCESS


def extract_embedding(case_id: str, series_index: pd.DataFrame | None = None) -> dict:
    series_index = series_index if series_index is not None else pd.read_csv(config.SERIES_INDEX_OUT)
    ct_dir, mask_path, ct_series_uid, seg_series_uid = radiomics.resolve_ct_and_mask(case_id, series_index)

    ct_image = viz.load_ct_volume(ct_dir)
    mask_image = viz.load_mask_on_ct(mask_path, ct_image)
    ct_array = sitk.GetArrayFromImage(ct_image)
    mask_array = sitk.GetArrayFromImage(mask_image)

    if mask_array.sum() == 0:
        raise ValueError(f"{case_id}: mask has zero voxels after resampling onto the CT grid")

    img, z = _central_tumor_crop(ct_array, mask_array)  # identical crop/window/resize as the ImageNet baseline
    model, normalize = _get_model()
    with torch.no_grad():
        x = normalize(img)
        feat_map = model(x)
        vec = torch.nn.functional.adaptive_avg_pool2d(feat_map, 1).flatten(1).squeeze(0).numpy()

    row = {"case_id": case_id, "central_slice": int(z)}
    row.update({f"radimg_{i:04d}": float(v) for i, v in enumerate(vec)})
    return row


def extract_all(case_ids=None, progress_cb=None, out_path=None) -> pd.DataFrame:
    out_path = out_path or config.PROCESSED_DIR / "radimagenet_embeddings.csv"
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
    print(f"\nExtracted RadImageNet embeddings for {len(df)} patients -> {config.PROCESSED_DIR / 'radimagenet_embeddings.csv'}")
