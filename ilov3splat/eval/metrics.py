from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ilov3splat.eval.utils import mask_to_boundary


def calculate_iou(mask1: npt.NDArray[np.bool_], mask2: npt.NDArray[np.bool_]) -> float:
    """Calculate IoU between two boolean masks."""
    assert mask1.dtype == bool and mask2.dtype == bool, "Masks must be boolean"
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def calculate_biou(
    gt: npt.NDArray[np.bool_], dt: npt.NDArray[np.bool_], dilation_ratio: float = 0.02
) -> float:
    """Boundary IoU for binary masks."""
    dt_u8 = dt.astype("uint8")
    gt_u8 = gt.astype("uint8")
    gt_boundary = mask_to_boundary(gt_u8, dilation_ratio)
    dt_boundary = mask_to_boundary(dt_u8, dilation_ratio)
    intersection = ((gt_boundary * dt_boundary) > 0).sum()
    union = ((gt_boundary + dt_boundary) > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def calculate_loc(bboxes: npt.NDArray[np.float32], mask: npt.NDArray[np.bool_]) -> bool:
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox.astype(int)
        assert x_min < x_max and y_min < y_max, f"Invalid bbox: {bbox}"
        mask_cropped = mask[y_min:y_max, x_min:x_max]
        return bool(mask_cropped.any())
    return False
