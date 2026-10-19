"""Convert a downloaded DICOM-SEG series into a NIfTI mask volume.

The 144 SEG series in this collection are the actual volumetric tumor
segmentations (distinct from the single-slice AIM locator circles in
aim_annotations.py). Each SEG object is self-contained: it carries its own
per-frame geometry (ImagePositionPatient/PixelSpacing), so the mask can be
reconstructed without needing the source CT series alongside it, though the
mask's geometry should match the CT it was drawn on and can be resampled
onto it if needed for radiomics.
"""

from __future__ import annotations

from pathlib import Path

import pydicom
import pydicom_seg
import SimpleITK as sitk

from . import config


def find_seg_dicom(seg_dir: Path) -> Path:
    """A SEG series is a single DICOM file; find it in a download dir."""
    candidates = [p for p in seg_dir.glob("*.dcm")]
    if not candidates:
        raise FileNotFoundError(f"No .dcm file found in {seg_dir}")
    if len(candidates) > 1:
        raise ValueError(f"Expected exactly one SEG file in {seg_dir}, found {len(candidates)}")
    return candidates[0]


def seg_to_nifti(seg_dicom_path: Path, out_path: Path | None = None) -> sitk.Image:
    """Read a DICOM-SEG file and write it out as a NIfTI label mask.

    Returns the SimpleITK image (voxel values are the DICOM SEG segment
    numbers, background = 0) for programmatic use even if out_path is None.
    """
    dcm = pydicom.dcmread(seg_dicom_path)
    reader = pydicom_seg.MultiClassReader()
    result = reader.read(dcm)
    image = result.image  # sitk.Image, geometry-correct label map

    segments = {
        seg_num: info.SegmentLabel for seg_num, info in result.segment_infos.items()
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(out_path), useCompression=True)

    return image, segments


def referenced_ct_series_uid(seg_dicom_path: Path) -> str:
    """The exact CT series a SEG object was drawn on (from the SEG itself,
    not guessed from series_description -- a patient can have several CT
    series, e.g. R01-019 has both "THINS" and "CT_SLICES", and only one is
    the one the mask actually corresponds to)."""
    dcm = pydicom.dcmread(seg_dicom_path)
    result = pydicom_seg.MultiClassReader().read(dcm)
    return result.referenced_series_uid


def build_one(case_id: str, seg_series_dir: Path, out_dir=None) -> Path:
    out_dir = out_dir or config.MASKS_DIR
    seg_dicom = find_seg_dicom(seg_series_dir)
    out_path = out_dir / f"{case_id}_seg.nii.gz"
    image, segments = seg_to_nifti(seg_dicom, out_path)
    size = image.GetSize()
    spacing = image.GetSpacing()
    print(
        f"{case_id}: mask {size[0]}x{size[1]}x{size[2]} voxels, "
        f"spacing {spacing}, segments={segments} -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("seg_series_dir", type=Path, help="Dir containing the extracted SEG .dcm")
    args = parser.parse_args()

    build_one(args.case_id, args.seg_series_dir)
