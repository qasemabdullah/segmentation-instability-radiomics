"""Physical boundary shift of a binary mask (see perturb_mask). Depends only on
numpy, scipy and SimpleITK so the isolated PyRadiomics environment can import it
directly from this file."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk


FINE_MM = 0.5  # isotropic working resolution for the boundary shift


def perturb_mask(mask: sitk.Image, mm: float) -> sitk.Image:
    """Move the tumour surface outwards (mm > 0) or inwards (mm < 0) by a
    physical distance, in every direction, and return the result on the
    original voxel grid.

    The mask is treated as the union of its voxels (so a 3-mm slice is a
    3-mm slab), resampled to a FINE_MM isotropic grid, and each fine voxel's
    signed Euclidean distance to that surface is computed (the surface lies
    half a fine voxel beyond the last object voxel). The shifted object is
    {distance < mm}, sampled back at the original voxel centres.

    An earlier version thresholded ITK's SignedMaurerDistanceMap on the
    original grid. ITK assigns distance 0 to the object's own boundary
    voxels, so "-1 mm" removed ~2 in-plane voxel layers plus whole end
    slices (-41% volume for a 20-mm-radius sphere at 0.98 x 0.98 x 3 mm,
    against -14% for a true 1-mm erosion) while "+1 mm" added only one
    layer. See scripts/validate_segmentation_adjustment.py for the validation."""
    if mm == 0:
        return mask


    m = sitk.Cast(mask > 0, sitk.sitkUInt8)
    ls = sitk.LabelShapeStatisticsImageFilter()
    ls.Execute(m)
    if not ls.GetLabels():
        return sitk.Cast(m, mask.GetPixelID())
    sp = np.array(m.GetSpacing())
    # crop to the tumour plus enough margin for the shift
    x0, y0, z0, sx, sy, sz = ls.GetBoundingBox(1)
    pad = np.ceil((abs(mm) + 2 * FINE_MM) / sp).astype(int) + 1
    lo = np.maximum(np.array([x0, y0, z0]) - pad, 0)
    hi = np.minimum(np.array([x0 + sx, y0 + sy, z0 + sz]) + pad, np.array(m.GetSize()))
    crop = sitk.RegionOfInterest(m, (hi - lo).tolist(), lo.tolist())

    # fine isotropic grid covering the crop's physical extent (voxel edges, not centres)
    origin = np.array(crop.GetOrigin()) - 0.5 * sp + 0.5 * FINE_MM  # assumes axis-aligned direction
    size = np.ceil(np.array(crop.GetSize()) * sp / FINE_MM).astype(int)
    fine = sitk.Resample(crop, size.tolist(), sitk.Transform(), sitk.sitkNearestNeighbor, origin.tolist(),
                         [FINE_MM] * 3, crop.GetDirection(), 0, sitk.sitkUInt8)
    # ITK's signed Maurer map (float32, memory-light) gives 0 on the object's boundary voxels and
    # +FINE_MM on the first outside layer; subtracting half a fine voxel turns it into the signed
    # distance from each voxel centre to the object surface (the surface lies midway between them)
    maurer = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        fine, insideIsPositive=False, squaredDistance=False, useImageSpacing=True))
    shifted = sitk.GetImageFromArray(((maurer - FINE_MM / 2) < mm).astype(np.uint8))
    del maurer
    shifted.CopyInformation(fine)
    back = sitk.Resample(shifted, crop, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    out = sitk.Image(m.GetSize(), sitk.sitkUInt8)
    out.CopyInformation(m)
    out = sitk.Paste(out, back, back.GetSize(), [0, 0, 0], lo.tolist())
    return sitk.Cast(out, mask.GetPixelID())
