from __future__ import annotations

import numpy as np
import torch


def sam_stack_to_label_map(
    sam_masks: torch.Tensor,
    height: int,
    width: int,
    has_background_row: bool = True,
) -> torch.Tensor:
    """Convert a SAM stack (K,L) into a dense label map (H,W)."""
    assert sam_masks.ndim == 2, f"Expected (K,L), got {tuple(sam_masks.shape)}"
    assert sam_masks.dtype == torch.bool, f"Expected bool mask stack, got {sam_masks.dtype}"
    assert sam_masks.shape[1] == height * width, (
        f"Mask length mismatch: got {sam_masks.shape[1]} expected {height*width}"
    )

    fg_masks = sam_masks[:-1] if (has_background_row and sam_masks.shape[0] > 1) else sam_masks
    if fg_masks.numel() == 0 or fg_masks.shape[0] == 0:
        return torch.full((height, width), -1, dtype=torch.long, device=sam_masks.device)

    areas = fg_masks.sum(dim=1)
    order = torch.argsort(areas, descending=True)
    fg_masks = fg_masks[order]

    label_flat = torch.full((height * width,), -1, dtype=torch.long, device=sam_masks.device)
    assigned = torch.zeros((height * width,), dtype=torch.bool, device=sam_masks.device)
    next_label = 0
    for i in range(fg_masks.shape[0]):
        mask = fg_masks[i]
        write_mask = mask & (~assigned)
        if write_mask.any():
            label_flat[write_mask] = next_label
            assigned |= write_mask
            next_label += 1

    return label_flat.view(height, width)


def load_instance_labels_from_npz(mask_npz_path: str, array_key: str = "default") -> np.ndarray:
    """Load a per-frame dense label map from a .npz (common layout: one key per mask level)."""
    with np.load(mask_npz_path) as level_masks:
        if array_key not in level_masks:
            raise KeyError(f"Array key '{array_key}' not found in {mask_npz_path}")
        labels = level_masks[array_key]
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D label map in {mask_npz_path}, got {labels.shape}")
    return labels


def remap_instance_labels(labels: torch.Tensor, ignore_label: int = -1) -> torch.Tensor:
    """Remap arbitrary non-ignore instance IDs to contiguous [0, I-1]."""
    assert labels.ndim == 2, f"Expected (H,W), got {tuple(labels.shape)}"
    assert labels.dtype == torch.long, f"Expected long labels, got {labels.dtype}"
    out = torch.full_like(labels, ignore_label)
    unique_ids = torch.unique(labels)
    unique_ids = unique_ids[unique_ids != ignore_label]
    for new_id, old_id in enumerate(unique_ids.tolist()):
        out[labels == old_id] = int(new_id)
    return out
