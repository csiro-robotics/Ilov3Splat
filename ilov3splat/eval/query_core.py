"""Hybrid cluster language query helpers for LERF-OVS evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import get_viewmat

from ilov3splat.eval.hybrid_language_query import (
    aggregate_gaussian_relevancy_by_cluster,
    build_gaussian_highlight_mask,
    select_clusters_from_scores,
)
from ilov3splat.eval.utils import render_alpha, render_frames, render_scene_depth

try:
    from gsplat.rendering import rasterization
except ImportError as exc:  # pragma: no cover
    raise ImportError("gsplat>=1.0.0 is required for ilov3splat eval") from exc


@dataclass
class LanguageViewContext:
    means_crop: torch.Tensor
    quats_crop: torch.Tensor
    scales_crop: torch.Tensor
    opacities_crop: torch.Tensor
    crop_ids: Optional[torch.Tensor]
    viewmat: torch.Tensor
    clip_field_flat: torch.Tensor
    clip_h: int
    clip_w: int
    full_h: int
    full_w: int
    depth_im: torch.Tensor
    clip_fy: float


def _unwrap_model(model: Any) -> Any:
    if hasattr(model, "module"):
        return model.module
    return model


def load_cluster_labels(run_dir: Path, num_gaussians: int, device: torch.device) -> torch.Tensor:
    """Load per-Gaussian cluster labels from a run directory."""
    cluster_dir = run_dir / "clustering"
    for name in ("labels.npy", "labels_lift.npy"):
        path = cluster_dir / name
        if not path.is_file():
            continue
        arr = np.load(path)
        labels = torch.from_numpy(arr).to(device=device, dtype=torch.long)
        if int(labels.shape[0]) != int(num_gaussians):
            raise ValueError(
                f"{path.name} length {labels.shape[0]} != num Gaussians {num_gaussians}"
            )
        return labels
    raise FileNotFoundError(
        f"Missing cluster labels under {cluster_dir}. "
        "Run `ilov3splat-cluster-gaussians` first."
    )


@torch.no_grad()
def prepare_language_view_context(model: Any, camera: Cameras) -> LanguageViewContext:
    """Build per-view tensors needed for hybrid cluster language query."""
    model = _unwrap_model(model)
    depth_im = render_scene_depth(camera, model)
    optimized_camera_to_world = camera.camera_to_worlds

    if model.crop_box is not None:
        crop_ids = model.crop_box.within(model.means).squeeze()
        if crop_ids.sum() == 0:
            raise RuntimeError("Empty crop box for camera.")
    else:
        crop_ids = None

    if crop_ids is not None:
        means_crop = model.means[crop_ids]
        quats_crop = model.quats[crop_ids]
        scales_crop = model.scales[crop_ids]
        opacities_crop = model.opacities[crop_ids]
    else:
        means_crop = model.means
        quats_crop = model.quats
        scales_crop = model.scales
        opacities_crop = model.opacities

    cam_clip = copy.deepcopy(camera)
    if cam_clip.metadata is not None:
        gt_h = cam_clip.metadata.get("gt_image_height")
        gt_w = cam_clip.metadata.get("gt_image_width")
        if gt_h is not None and gt_w is not None:
            model._snap_camera_to_gt_image_resolution(cam_clip, int(gt_h), int(gt_w))

    viewmat = get_viewmat(optimized_camera_to_world.detach())
    clip_downscale = int(
        getattr(model.datamanager.config, "clip_downscale_factor", model.config.feature_downscale_factor)
    )
    if cam_clip.metadata is not None and "clip_downscale_factor" in cam_clip.metadata:
        clip_downscale = int(cam_clip.metadata["clip_downscale_factor"])
    clip_downscale = max(1, clip_downscale)

    cam_clip.rescale_output_resolution(1 / clip_downscale)
    clip_w = int(cam_clip.width.item())
    clip_h = int(cam_clip.height.item())
    clip_k = cam_clip.get_intrinsics_matrices().to(means_crop.device)

    full_h = int(camera.height.item())
    full_w = int(camera.width.item())
    if camera.metadata is not None:
        gt_h = int(camera.metadata.get("gt_image_height", full_h))
        gt_w = int(camera.metadata.get("gt_image_width", full_w))
    else:
        gt_h, gt_w = full_h, full_w

    clip_hash = model.clip_field.get_clip_hash(means_crop)
    clip_render, _, _ = rasterization(
        means=means_crop,
        quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
        scales=torch.exp(scales_crop),
        opacities=torch.sigmoid(opacities_crop).squeeze(-1),
        colors=clip_hash,
        viewmats=viewmat,
        Ks=clip_k,
        width=clip_w,
        height=clip_h,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode="RGB",
        sparse_grad=False,
        absgrad=False,
        rasterize_mode=model.config.rasterize_mode,
    )
    clip_fy = float(camera.get_intrinsics_matrices().reshape(-1, 3, 3)[0, 1, 1].item())
    return LanguageViewContext(
        means_crop=means_crop,
        quats_crop=quats_crop,
        scales_crop=scales_crop,
        opacities_crop=opacities_crop,
        crop_ids=crop_ids,
        viewmat=viewmat,
        clip_field_flat=clip_render.reshape(clip_h * clip_w, -1),
        clip_h=clip_h,
        clip_w=clip_w,
        full_h=full_h,
        full_w=full_w,
        depth_im=depth_im,
        clip_fy=clip_fy,
    )


@torch.no_grad()
def select_hybrid_clusters_for_phrase(
    model: Any,
    ctx: LanguageViewContext,
    phrase_idx: int,
    cluster_labels: torch.Tensor,
    best_scale: torch.Tensor,
    cluster_threshold: float,
) -> List[int]:
    """Select 3D cluster ids for one language phrase using hybrid aggregation."""
    model = _unwrap_model(model)
    labels = cluster_labels
    if ctx.crop_ids is not None:
        labels = labels[ctx.crop_ids]

    scale_ratio = best_scale.to(device=ctx.means_crop.device, dtype=torch.float32)
    gaussian_depth = model._gaussian_camera_depths(ctx.means_crop, ctx.viewmat)
    scale_tensor = model._depth_adjusted_clip_scale(
        scale_ratio, ctx.clip_h, gaussian_depth, ctx.clip_fy
    )
    clip_feats = model.clip_field.decode_clip_from_features(
        model.clip_field.get_clip_hash(ctx.means_crop),
        scale_tensor,
    )
    relevancy = model.image_encoder.get_relevancy(clip_feats, phrase_idx)[..., 0]
    scores = aggregate_gaussian_relevancy_by_cluster(relevancy, labels)
    return select_clusters_from_scores(
        scores, threshold=cluster_threshold, mode="relative_max"
    )


@torch.no_grad()
def _render_binary_mask_for_gaussians(
    model: Any,
    camera: Cameras,
    gaussian_mask: torch.Tensor,
    mask_threshold: float,
    render_mode: str | None,
) -> torch.Tensor:
    """Render one binary mask for selected Gaussians."""
    model = _unwrap_model(model)
    if not bool(gaussian_mask.any()):
        width = int(camera.width.item())
        height = int(camera.height.item())
        return torch.zeros(height, width, dtype=torch.bool)

    if render_mode == "alpha":
        op = model.opacities.clone()
        op[~gaussian_mask] = -100.0
        alpha = render_alpha([camera], model, override_opacity=op)[0]
        return alpha > mask_threshold

    mask_color = torch.zeros((gaussian_mask.shape[0], 3), device=gaussian_mask.device, dtype=torch.float32)
    mask_color[gaussian_mask] = 1.0
    if render_mode == "isolated":
        op = model.opacities.clone()
        op[~gaussian_mask] = -100.0
    else:
        op = None
    renders, _ = render_frames(
        [camera],
        model,
        override_color=mask_color,
        override_opacity=op,
    )
    return renders[0].mean(dim=0) > mask_threshold


@torch.no_grad()
def compute_masks_hybrid_cluster_3d(
    model: Any,
    cameras: list[Cameras],
    cluster_labels: torch.Tensor,
    query_prompts: list[str],
    cluster_threshold: float,
    mask_threshold: float = 0.3,
    render_mode: str | None = None,
) -> torch.Tensor:
    """Compute inst-3D masks with hybrid cluster language query.

    Returns:
        Boolean tensor of shape ``(num_cameras, num_prompts, H, W)``.
    """
    assert render_mode in {None, "isolated", "alpha"}
    model = _unwrap_model(model)
    if model.image_encoder is None:
        raise RuntimeError("OpenCLIP image encoder is required for hybrid cluster eval.")

    model.image_encoder.set_positives(query_prompts)
    if len(query_prompts) == 0:
        raise RuntimeError("At least one query prompt is required.")

    per_camera: list[torch.Tensor] = []
    for camera in cameras:
        ctx = prepare_language_view_context(model, camera)
        _, best_scales = model.get_max_across(
            ctx.clip_field_flat,
            ctx.clip_h,
            ctx.clip_w,
            depth=ctx.depth_im,
            fy=ctx.clip_fy,
        )
        per_prompt: list[torch.Tensor] = []
        for phrase_idx in range(len(query_prompts)):
            selected = select_hybrid_clusters_for_phrase(
                model,
                ctx,
                phrase_idx,
                cluster_labels,
                best_scales[phrase_idx],
                cluster_threshold,
            )
            gaussian_mask = build_gaussian_highlight_mask(cluster_labels, selected)
            per_prompt.append(
                _render_binary_mask_for_gaussians(
                    model, camera, gaussian_mask, mask_threshold, render_mode
                )
            )
        per_camera.append(torch.stack(per_prompt, dim=0))
    return torch.stack(per_camera, dim=0)


@torch.no_grad()
def render_hybrid_cluster_rgb(
    model: Any,
    cameras: list[Cameras],
    cluster_labels: torch.Tensor,
    query_prompts: list[str],
    cluster_threshold: float,
) -> torch.Tensor:
    """Render isolated RGB views for each prompt/cluster selection.

    Returns:
        Tensor of shape ``(num_cameras, num_prompts, 3, H, W)``.
    """
    model = _unwrap_model(model)
    if model.image_encoder is None:
        raise RuntimeError("OpenCLIP image encoder is required for hybrid cluster viz.")

    model.image_encoder.set_positives(query_prompts)
    white_bg = torch.tensor([1.0, 1.0, 1.0], device=model.device, dtype=torch.float32)
    per_camera: list[torch.Tensor] = []

    for camera in cameras:
        ctx = prepare_language_view_context(model, camera)
        _, best_scales = model.get_max_across(
            ctx.clip_field_flat,
            ctx.clip_h,
            ctx.clip_w,
            depth=ctx.depth_im,
            fy=ctx.clip_fy,
        )
        per_prompt: list[torch.Tensor] = []
        for phrase_idx in range(len(query_prompts)):
            selected = select_hybrid_clusters_for_phrase(
                model,
                ctx,
                phrase_idx,
                cluster_labels,
                best_scales[phrase_idx],
                cluster_threshold,
            )
            gaussian_mask = build_gaussian_highlight_mask(cluster_labels, selected)
            op = model.opacities.clone()
            op[~gaussian_mask] = -100.0
            renders, _ = render_frames(
                [camera],
                model,
                override_opacity=op,
                override_background=white_bg,
            )
            per_prompt.append(renders[0])
        per_camera.append(torch.stack(per_prompt, dim=0))
    return torch.stack(per_camera, dim=0)
