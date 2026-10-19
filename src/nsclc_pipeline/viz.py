"""CT slice rendering: HU windowing and mask overlay, as plain numpy/PIL
so it's usable both from the GUI and from a notebook.

A DICOM-SEG mask often covers fewer slices than its source CT (only the
frames spanning the tumor, not the whole chest) -- see seg_to_mask.py.
So a mask is always resampled onto the CT's own grid before slicing;
overlaying it directly without resampling would silently misalign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

# (level, width) in Hounsfield units.
WINDOW_PRESETS = {
    "Lung": (-600, 1500),
    "Mediastinal": (40, 400),
    "Bone": (400, 1800),
}


def load_ct_volume(ct_dir: Path) -> sitk.Image:
    """Read a downloaded CT series directory (many .dcm files, one per
    slice) into a single 3D SimpleITK image, correctly ordered/spaced."""
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(ct_dir))
    if not files:
        raise FileNotFoundError(f"No DICOM series found in {ct_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def load_mask_on_ct(mask_path: Path, ct_image: sitk.Image) -> sitk.Image:
    """Load a mask NIfTI and resample it onto the CT's voxel grid (nearest
    neighbor -- it's a label map) so slice indices line up 1:1 with the CT."""
    mask = sitk.ReadImage(str(mask_path))
    return sitk.Resample(mask, ct_image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())


def window_slice(hu_slice: np.ndarray, level: float, width: float) -> np.ndarray:
    """Map a Hounsfield-unit slice to an 8-bit grayscale array."""
    lo, hi = level - width / 2, level + width / 2
    clipped = np.clip(hu_slice, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def render_slice(
    ct_array: np.ndarray,
    z: int,
    level: float,
    width: float,
    mask_array: np.ndarray | None = None,
    mask_alpha: float = 0.45,
) -> Image.Image:
    """Render one axial slice (ct_array is z,y,x from sitk.GetArrayFromImage)
    as a PIL RGB image, with an optional red mask overlay."""
    gray = window_slice(ct_array[z], level, width)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    if mask_array is not None:
        m = mask_array[z] > 0
        if m.any():
            rgb[m, 0] = rgb[m, 0] * (1 - mask_alpha) + 255 * mask_alpha
            rgb[m, 1] = rgb[m, 1] * (1 - mask_alpha)
            rgb[m, 2] = rgb[m, 2] * (1 - mask_alpha)

    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
