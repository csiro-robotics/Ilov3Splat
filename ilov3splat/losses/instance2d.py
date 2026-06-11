from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def masks_encode_binary(masks: torch.Tensor) -> torch.BoolTensor:
    """Dense label map to per-instance binary masks (reference: mask_utils.masks_encode_binary)."""
    assert masks.ndim == 2, "masks must be 2D"
    assert masks.dtype == torch.long, "only long tensors are supported"
    num_instances = int(masks.max().item() + 1)
    binary_masks = masks + 1
    binary_masks = torch.nn.functional.one_hot(
        binary_masks,
        num_classes=num_instances + 1,
    ).bool()
    return binary_masks.permute(2, 0, 1).contiguous()


def sample_pixels(
    network_output: torch.Tensor,
    gt: torch.Tensor,
    sample_size: int = 4096,
    ignore_label: int | None = None,
):
    """Sample pixels from feature map and 2D label map (reference: loss_utils.sample_pixels)."""
    assert gt.dtype == torch.long, f"Expected long masks tensor, got {gt.dtype}"

    gt_flat = gt.flatten(-2, -1)
    gt_ignore = None
    if ignore_label is not None:
        gt_ignore = gt_flat == ignore_label
        if gt_ignore.ndim == 2:
            gt_ignore = gt_ignore.sum(dim=0).bool()
        gt_flat = gt_flat[..., ~gt_ignore]

    sample_indices = None
    sample_targets = gt_flat
    if sample_size > 0:
        sample_indices = torch.randperm(gt_flat.size(-1), device=gt_flat.device)[:sample_size]
        sample_targets = gt_flat.gather(-1, sample_indices)

    network_output_flat = network_output.flatten(1).permute(1, 0).contiguous()
    if gt_ignore is not None:
        network_output_flat = network_output_flat[~gt_ignore]

    sample_features = network_output_flat
    if sample_indices is not None:
        sample_features = network_output_flat.gather(
            0, sample_indices.unsqueeze(-1).expand(-1, network_output_flat.shape[1])
        )

    return sample_features, sample_targets, sample_indices


def get_mean_prototypes(network_output: torch.Tensor, gt: torch.Tensor):
    """Mean feature prototype per instance (reference: loss_utils.get_mean_prototypes)."""
    with torch.no_grad():
        binary_gt = masks_encode_binary(gt)
    binary_gt = binary_gt[1:]
    binary_gt = binary_gt.flatten(1)
    network_output = network_output.flatten(1)

    npixels = binary_gt.sum(dim=-1)
    assert npixels.min() > 0, "Some instances have no pixels"
    cluster_features = network_output.unsqueeze(0) * binary_gt.unsqueeze(1)
    mean_prototypes = cluster_features.sum(dim=-1) / npixels.unsqueeze(-1)
    return mean_prototypes, binary_gt, npixels


def instance_2d_mean_contrastive_loss(
    network_output: torch.Tensor,
    gt_masks: torch.Tensor,
    mask_dim: int,
    sample_size: int = -1,
    gamma: float = 1.0,
    normalize: bool = False,
    weights: tuple[float, float] = (1.0, 1.0),
) -> dict[str, torch.Tensor]:
    """2D prototype contrastive instance loss (aligned with reference losses/instances.py)."""
    assert gt_masks.ndim == 2, f"Expected 2D masks tensor, got {gt_masks.ndim - 1}D"
    assert gt_masks.dtype == torch.long, f"Expected long masks tensor, got {gt_masks.dtype}"
    if normalize:
        network_output = F.normalize(network_output, dim=0)

    assert network_output.shape[0] == mask_dim, "Mask level dimensions and feature dimensions mismatch"

    labels: torch.Tensor = gt_masks.unique()
    if len(labels) == 1 and labels.item() == -1:
        return {"total": torch.tensor(0.0, device=network_output.device)}

    sample_features, sample_targets, _ = sample_pixels(
        network_output[:mask_dim], gt_masks, sample_size, ignore_label=-1
    )

    mean_prototypes = get_mean_prototypes(network_output[:mask_dim], gt_masks)[0]

    mean_p = mean_prototypes.gather(
        0, sample_targets.unsqueeze(1).expand(-1, mean_prototypes.shape[1])
    )
    loss_pos = (sample_features - mean_p).pow(2).sum(dim=-1)

    if mean_prototypes.size(0) > 1:
        loss_neg = torch.cdist(mean_prototypes, mean_prototypes)
        n = loss_neg.size(0)
        triu_indices = torch.triu_indices(
            n,
            n,
            offset=1,
            device=loss_neg.device,
        )
        triu_indices = triu_indices[0] * n + triu_indices[1]
        loss_neg = loss_neg.flatten().gather(0, triu_indices)
        loss_neg = F.relu(gamma - loss_neg)
    else:
        loss_neg = None

    loss = weights[0] * loss_pos.mean()
    if loss_neg is not None:
        loss += weights[1] * loss_neg.mean()

    loss_output = {"positive": loss_pos.mean()}
    if loss_neg is not None:
        loss_output["negative"] = loss_neg.mean()
    else:
        loss_output["negative"] = torch.tensor(0.0, device=network_output.device)
    loss_output["total"] = loss
    return loss_output


def instance_2d_loss(
    network_output: torch.Tensor,
    gt_masks: torch.Tensor,
    mask_dim: int,
    sample_size: int = -1,
    gamma: float = 1.0,
    weights: list[float] = [1.0, 1.0],
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:
    assert len(weights) == 2, "Expected two weights"
    return instance_2d_mean_contrastive_loss(
        network_output,
        gt_masks,
        mask_dim,
        sample_size,
        gamma,
        normalize,
        (weights[0], weights[1]),
    )
