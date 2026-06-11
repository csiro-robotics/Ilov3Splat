"""Hybrid language query: pixel CLIP relevancy + per-cluster aggregation."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch


def aggregate_relevancy_by_cluster(
    pixel_relevancy: torch.Tensor,
    cluster_id_map: torch.Tensor,
    *,
    ignore_label: int = -1,
    reduction: Literal["mean", "max"] = "mean",
) -> Dict[int, float]:
    """Aggregate per-pixel relevancy scores by cluster id.

    Args:
        pixel_relevancy: ``(H, W)`` or ``(H*W,)`` relevancy in [0, 1].
        cluster_id_map: ``(H, W)`` integer cluster label per pixel.
        ignore_label: Label to skip (typically noise ``-1``).
        reduction: ``mean`` or ``max`` over pixels in each cluster.

    Returns:
        Mapping from cluster id to aggregated score.
    """
    rel = pixel_relevancy.reshape(-1).to(torch.float32)
    ids = cluster_id_map.reshape(-1).to(torch.long)
    if rel.shape[0] != ids.shape[0]:
        raise ValueError(
            f"pixel/cluster length mismatch: relevancy={rel.shape[0]} cluster={ids.shape[0]}"
        )

    scores: Dict[int, float] = {}
    unique = ids.unique()
    for cid in unique.tolist():
        if int(cid) == ignore_label:
            continue
        mask = ids == cid
        if not bool(mask.any()):
            continue
        vals = rel[mask]
        if reduction == "max":
            scores[int(cid)] = float(vals.max().item())
        else:
            scores[int(cid)] = float(vals.mean().item())
    return scores


def aggregate_gaussian_relevancy_by_cluster(
    gaussian_relevancy: torch.Tensor,
    cluster_labels: torch.Tensor,
    *,
    ignore_label: int = -1,
    reduction: Literal["mean", "max"] = "mean",
) -> Dict[int, float]:
    """Aggregate per-Gaussian relevancy by 3D cluster label."""
    rel = gaussian_relevancy.reshape(-1).to(torch.float32)
    labels = cluster_labels.reshape(-1).to(torch.long)
    if rel.shape[0] != labels.shape[0]:
        raise ValueError(
            f"gaussian/cluster length mismatch: relevancy={rel.shape[0]} labels={labels.shape[0]}"
        )

    scores: Dict[int, float] = {}
    unique = labels.unique()
    for cid in unique.tolist():
        if int(cid) == ignore_label:
            continue
        mask = labels == cid
        if not bool(mask.any()):
            continue
        vals = rel[mask]
        if reduction == "max":
            scores[int(cid)] = float(vals.max().item())
        else:
            scores[int(cid)] = float(vals.mean().item())
    return scores


def select_clusters_from_scores(
    scores: Dict[int, float],
    *,
    threshold: float = 0.85,
    mode: Literal["relative_max", "absolute"] = "relative_max",
) -> List[int]:
    """Select cluster ids from aggregated scores."""
    if not scores:
        return []
    vals = torch.tensor(list(scores.values()), dtype=torch.float32)
    ids = list(scores.keys())
    if mode == "absolute":
        return [cid for cid, s in scores.items() if s >= threshold]
    max_val = float(vals.max().item())
    if max_val <= 0.0:
        return []
    cutoff = max_val * threshold
    return [cid for cid, s in scores.items() if s >= cutoff]


def build_gaussian_highlight_mask(
    cluster_labels: torch.Tensor,
    selected_cluster_ids: List[int],
    *,
    ignore_label: int = -1,
) -> torch.Tensor:
    """Boolean mask over Gaussians for selected clusters."""
    labels = cluster_labels.reshape(-1).to(torch.long)
    if not selected_cluster_ids:
        return torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    selected = torch.tensor(selected_cluster_ids, device=labels.device, dtype=torch.long)
    mask = torch.isin(labels, selected)
    if ignore_label >= 0:
        mask = mask & (labels != ignore_label)
    return mask
