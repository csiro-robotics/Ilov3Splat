from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerCheckbox,
    ViewerControl,
    ViewerSlider,
)

from ilov3splat.cluster import ClusterConfig, run_hdbscan_clustering, save_cluster_artifacts
from ilov3splat.eval.hybrid_language_query import (
    aggregate_gaussian_relevancy_by_cluster,
    build_gaussian_highlight_mask,
    select_clusters_from_scores,
)
from ilov3splat.fields.clip_field import ClipLanguageField
from ilov3splat.losses import instance_2d_loss
from ilov3splat.utils.dino_dataloader import MAX_DINO_SIZE, get_img_resolution
from ilov3splat.utils.image_encoder import BaseImageEncoder
from ilov3splat.viewer_controls import Ilov3SplatViewerControls

from gsplat.strategy import DefaultStrategy

try:
    from gsplat.rendering import rasterization
except ImportError as exc:  # pragma: no cover
    raise ImportError("gsplat>=1.0.0 is required for ilov3splat") from exc


@dataclass
class Ilov3SplatModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: Ilov3SplatModel)

    instance_feature_dim: int = 16
    instance_embedding_dim: int = 16
    instance_decode_mode: Literal["identity", "linear", "mlp"] = "identity"
    feature_downscale_factor: int = 2

    inst2d_lambda: float = 1.0
    inst2d_sample_size: int = -1
    inst2d_gamma: float = 1.0
    inst2d_weights: Tuple[float, float] = (1.0, 1.0)
    inst2d_normalize: bool = False
    inst2d_from_iter: int = 0
    inst2d_interval: int = 1
    inst2d_ramp_steps: int = 0
    instance_warmup_steps: int = 0
    instance_lr_reset_after_warmup: bool = True
    var_lambda: float = 0.0

    cluster_on_train_end: bool = False
    cluster_output_dir: str = "clustering"
    cluster_overwrite_output: bool = False
    cluster_min_cluster_size: int = 20
    cluster_min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    cluster_with_position: float = 0.0
    cluster_with_color: float = 0.0
    cluster_normalize_instance_feats: bool = False
    cluster_normalize_position: bool = False
    cluster_normalize_color: bool = False
    cluster_use_dbscan_denoising: bool = False
    cluster_dbscan_min_samples: int = 5
    cluster_dbscan_eps: float = 0.05
    cluster_dbscan_min_cluster_size: int = 0
    cluster_prefer_cuml: bool = True
    cluster_assign_noise: bool = True
    cluster_assign_noise_alpha: float = 0.5
    cluster_assign_noise_k: int = 5
    cluster_assign_noise_max_dist: Optional[float] = None
    cluster_viewer_enable: bool = True
    cluster_viewer_hide_noise: bool = True

    output_depth_during_training: bool = True
    """Required for CLIP scale computation from rendered depth."""

    lang_loss_weight: float = 0.1
    lang_step: int = 4000
    """Must match datamanager lang_step."""
    dino_feat_dim: int = 16
    """Per-Gaussian DINO embedding dimension before decoder."""
    dino_dim: int = 16
    """Rendered DINO feature dimension (must match datamanager dino_pca_dim)."""
    dino_rescale_factor: int = 5
    dino_loss_weight: float = 0.1
    """DINO MSE regularization weight during CLIP training; 0 disables."""
    n_scales: int = 20
    max_scale: float = 1.0

    lang_viewer_enable: bool = True
    lang_relevancy_thresh: float = 0.0
    lang_relevancy_score: float = 0.55
    lang_cluster_threshold: float = 0.85
    lang_use_hybrid_cluster_query: bool = True


class Ilov3SplatModel(Ilov3SplatViewerControls, SplatfactoModel):
    config: Ilov3SplatModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points

        self.cluster_button: Optional[ViewerButton] = None
        self.cluster_toggle_button: Optional[ViewerButton] = None
        self.cluster_eps_slider: Optional[ViewerSlider] = None
        self.cluster_min_cluster_slider: Optional[ViewerSlider] = None
        self.cluster_min_samples_manual_checkbox: Optional[ViewerCheckbox] = None
        self.cluster_min_samples_slider: Optional[ViewerSlider] = None
        self.load_saved_features_button: Optional[ViewerButton] = None
        self._var_warning_printed = False

        super().__init__(*args, **kwargs)

        self._instance_map_for_loss: Optional[torch.Tensor] = None
        self._instance_first_moment_for_var_loss: Optional[torch.Tensor] = None
        self._instance_second_moment_for_loss: Optional[torch.Tensor] = None
        self.cluster_labels: Optional[torch.Tensor] = None
        self.cluster_overlay_enabled: bool = False
        self.cluster_hide_noise: bool = bool(self.config.cluster_viewer_hide_noise)
        self.cluster_colormap = torch.rand((8192, 3), dtype=torch.float32, device=self.device)
        self.viewer_control = ViewerControl()
        self._cluster_after_train_ran = False
        self._last_cluster_write_dir: Optional[Path] = None
        self.loaded_ckpt = False
        self.best_scales: Optional[torch.Tensor] = None
        self._lang_highlight_mask: Optional[torch.Tensor] = None
        self._pending_language_cluster_query: bool = False
        self.lang_highlight_overlay_enabled: bool = False

    def populate_modules(self):
        super().populate_modules()
        self.clip_field = ClipLanguageField()
        self.datamanager = self.kwargs.get("datamanager")
        self.image_encoder: Optional[BaseImageEncoder] = self.kwargs.get("image_encoder")
        self.gauss_params["instance_feats"] = nn.Parameter(
            torch.randn((self.num_points, self.config.instance_feature_dim))
        )
        if float(self.config.dino_loss_weight) > 0.0:
            self.gauss_params["dino_feats"] = nn.Parameter(
                torch.randn((self.num_points, self.config.dino_feat_dim))
            )
            hidden_dim = max(16, self.config.dino_feat_dim)
            self.dino_decoder = nn.Sequential(
                nn.Linear(self.config.dino_feat_dim, hidden_dim, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.config.dino_dim, bias=False),
            )
        else:
            self.dino_decoder = None

        in_dim = self.config.instance_feature_dim
        out_dim = self.config.instance_embedding_dim
        mode = self.config.instance_decode_mode
        if mode == "identity":
            if in_dim != out_dim:
                raise ValueError(
                    "instance_decode_mode='identity' requires instance_feature_dim == instance_embedding_dim"
                )
            self.instance_decoder = nn.Identity()
        elif mode == "linear":
            self.instance_decoder = nn.Linear(in_dim, out_dim, bias=False)
        elif mode == "mlp":
            hidden = max(16, in_dim)
            self.instance_decoder = nn.Sequential(
                nn.Linear(in_dim, hidden, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, out_dim, bias=False),
            )
        else:  # pragma: no cover
            raise ValueError(f"Unsupported instance_decode_mode: {mode}")

        if self.config.var_lambda > 0.0 and not self._var_warning_printed:
            CONSOLE.print(
                "[green]var_lambda > 0 enabled: using gsplat rasterization "
                "for second-moment variance regularization.[/green]"
            )
            self._var_warning_printed = True
        self._setup_viewer_controls()
        self._setup_language_viewer_controls()
        self._wrap_image_encoder_gui_cb()

    def _wrap_image_encoder_gui_cb(self) -> None:
        if self.image_encoder is None or not hasattr(self.image_encoder, "gui_cb"):
            return
        original_cb = self.image_encoder.gui_cb

        def wrapped_cb(element):
            original_cb(element)
            viewer = getattr(self.viewer_control, "viewer", None)
            if viewer is not None:
                viewer._trigger_rerender()

        self.image_encoder.gui_cb = wrapped_cb

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }
        groups["instance_feats"] = [self.gauss_params["instance_feats"]]
        groups["lang"] = self.clip_field.get_clip_parameters()
        if "dino_feats" in self.gauss_params:
            groups["dino_feats"] = [self.gauss_params["dino_feats"]]
        decoder_params = list(self.instance_decoder.parameters())
        if len(decoder_params) > 0:
            groups["instance_decoder"] = decoder_params
        return groups

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        gps = self.get_gaussian_param_groups()
        if self.dino_decoder is not None:
            gps["dino_decoder"] = list(self.dino_decoder.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _make_cluster_config(self) -> ClusterConfig:
        min_size = (
            int(self.cluster_min_cluster_slider.value)
            if self.cluster_min_cluster_slider is not None
            else int(self.config.cluster_min_cluster_size)
        )
        if self.cluster_min_samples_manual_checkbox is not None:
            if not bool(self.cluster_min_samples_manual_checkbox.value):
                min_samples = None
            elif self.cluster_min_samples_slider is not None:
                min_samples = int(self.cluster_min_samples_slider.value)
            else:
                min_samples = None
        else:
            min_samples = (
                int(self.cluster_min_samples_slider.value)
                if self.cluster_min_samples_slider is not None
                else (
                    int(self.config.cluster_min_samples)
                    if self.config.cluster_min_samples is not None
                    else None
                )
            )
        eps = (
            float(self.cluster_eps_slider.value)
            if self.cluster_eps_slider is not None
            else float(self.config.cluster_selection_epsilon)
        )
        return ClusterConfig(
            min_cluster_size=max(2, min_size),
            min_samples=min_samples,
            cluster_selection_epsilon=max(0.0, eps),
            with_position=float(self.config.cluster_with_position),
            with_color=float(self.config.cluster_with_color),
            normalize_instance_feats=bool(self.config.cluster_normalize_instance_feats),
            normalize_position=bool(self.config.cluster_normalize_position),
            normalize_color=bool(self.config.cluster_normalize_color),
            use_dbscan_denoising=bool(self.config.cluster_use_dbscan_denoising),
            dbscan_min_samples=int(self.config.cluster_dbscan_min_samples),
            dbscan_eps=float(self.config.cluster_dbscan_eps),
            dbscan_min_cluster_size=int(self.config.cluster_dbscan_min_cluster_size),
            prefer_cuml=bool(self.config.cluster_prefer_cuml),
            assign_noise=bool(self.config.cluster_assign_noise),
            assign_noise_alpha=float(self.config.cluster_assign_noise_alpha),
            assign_noise_k=int(self.config.cluster_assign_noise_k),
            assign_noise_max_dist=self.config.cluster_assign_noise_max_dist,
        )

    def _cluster_artifact_parent(self) -> Optional[Path]:
        raw = self.kwargs.get("cluster_experiment_root")
        if raw is None:
            return None
        return Path(raw).expanduser().resolve()

    def _base_cluster_artifact_dir(self) -> Optional[Path]:
        raw = self.config.cluster_output_dir.strip()
        if not raw:
            return None
        cod = Path(raw).expanduser()
        if cod.is_absolute():
            return cod.resolve()
        parent = self._cluster_artifact_parent()
        if parent is not None:
            return (parent / cod).resolve()
        return cod.resolve()

    def run_gaussian_clustering(
        self, *, save_artifacts: bool = False, output_dir: Optional[Path] = None
    ) -> Dict[str, Union[int, float, List[int]]]:
        cfg = self._make_cluster_config()
        labels_np, stats = run_hdbscan_clustering(
            instance_feats=self.gauss_params["instance_feats"],
            means=self.means,
            features_dc=self.features_dc,
            cfg=cfg,
        )
        self.cluster_labels = torch.from_numpy(labels_np).to(
            self.means.device, dtype=torch.long
        )
        self._last_cluster_write_dir = None
        if save_artifacts:
            target_dir = (
                Path(output_dir).expanduser().resolve()
                if output_dir is not None
                else self._base_cluster_artifact_dir()
            )
            if target_dir is not None:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if self.config.cluster_overwrite_output:
                    write_dir = target_dir
                elif target_dir.exists():
                    write_dir = target_dir.parent / (
                        f"{target_dir.name}_viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    CONSOLE.print(
                        f"[yellow]Cluster dir exists ({target_dir}); saving under {write_dir} "
                        f"(set cluster_overwrite_output=True to overwrite default path).[/yellow]"
                    )
                else:
                    write_dir = target_dir
                save_cluster_artifacts(write_dir, labels_np, stats, cfg)
                self._last_cluster_write_dir = write_dir
        return stats

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = super().get_training_callbacks(training_callback_attributes)
        if self.config.cluster_on_train_end:
            cbs.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN],
                    func=self._cluster_after_train_callback,
                    kwargs={"tc_attrs": training_callback_attributes},
                )
            )
        return cbs

    def step_cb(self, optimizers: Optimizers, step: int) -> None:
        super().step_cb(optimizers, step)
        if not bool(self.config.instance_lr_reset_after_warmup):
            return
        self._apply_instance_lr_schedule(optimizers=optimizers, step=step)

    def _apply_instance_lr_schedule(self, optimizers: Optimizers, step: int) -> None:
        warmup = max(0, int(self.config.instance_warmup_steps))
        for group_name in ("instance_feats", "instance_decoder"):
            opt = optimizers.optimizers.get(group_name)
            if opt is None:
                continue
            group_cfg = optimizers.config.get(group_name, {})
            opt_cfg = group_cfg.get("optimizer", None)
            if opt_cfg is None:
                continue
            lr_init = float(getattr(opt_cfg, "lr", 0.0))
            if lr_init <= 0.0:
                target_lr = 0.0
            elif step < warmup:
                target_lr = 0.0
            else:
                sch_cfg = group_cfg.get("scheduler", None)
                lr_final = float(getattr(sch_cfg, "lr_final", lr_init)) if sch_cfg is not None else lr_init
                max_steps = max(1, int(getattr(sch_cfg, "max_steps", 1))) if sch_cfg is not None else 1
                eff_step = min(max(step - warmup, 0), max_steps)
                if lr_final <= 0.0 or lr_init <= 0.0:
                    target_lr = lr_init
                else:
                    target_lr = lr_init * ((lr_final / lr_init) ** (eff_step / max_steps))
            for pg in opt.param_groups:
                pg["lr"] = target_lr

    def _cluster_after_train_callback(
        self,
        step: int,
        tc_attrs: Optional[TrainingCallbackAttributes] = None,
    ) -> None:
        del step
        if self._cluster_after_train_ran:
            return
        self._cluster_after_train_ran = True
        stats = self.run_gaussian_clustering(save_artifacts=True)
        CONSOLE.print(
            f"[green]Post-train HDBSCAN saved: clusters={stats['num_clusters']} noise_ratio={stats['noise_ratio']:.2%}[/green]"
        )

    def _decode_instance(self, features: torch.Tensor) -> torch.Tensor:
        return self.instance_decoder(features).to(torch.float32)

    @staticmethod
    def _instance_embedding_rows_to_rgb_pca(
        decoded: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        x = decoded.float()
        n, d = x.shape
        if n != height * width:
            raise RuntimeError(
                f"[instance PCA] length mismatch: rows={n} expected={height * width}"
            )
        q = min(3, d)
        if d > 3:
            _, _, v = torch.pca_lowrank(x, q=q, center=True, niter=4)
            xc = x - x.mean(dim=0, keepdim=True)
            projected = xc @ v[:, :q]
        else:
            projected = x
        if q < 3:
            projected = F.pad(projected, (0, 3 - q))
        lo = projected.amin()
        hi = projected.amax()
        rgb = (projected - lo) / (hi - lo + 1e-8)
        return rgb.view(height, width, 3)

    @staticmethod
    def _snap_camera_to_gt_image_resolution(camera: Cameras, gt_h: int, gt_w: int) -> None:
        cur_w = int(camera.width.flatten()[0].item())
        cur_h = int(camera.height.flatten()[0].item())
        if cur_w == gt_w and cur_h == gt_h:
            return
        sx = gt_w / float(cur_w)
        sy = gt_h / float(cur_h)
        camera.fx = camera.fx * sx
        camera.fy = camera.fy * sy
        camera.cx = camera.cx * sx
        camera.cy = camera.cy * sy
        camera.width = torch.full(
            camera.width.shape, gt_w, dtype=camera.width.dtype, device=camera.width.device
        )
        camera.height = torch.full(
            camera.height.shape, gt_h, dtype=camera.height.dtype, device=camera.height.device
        )

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        outputs = super().get_outputs(camera)
        # Keep full-res scene RGB for language masking; cluster overlay may replace outputs["rgb"].
        if "rgb" in outputs:
            outputs["_scene_rgb"] = outputs["rgb"]
        if self.training:
            self._instance_map_for_loss = None
            self._instance_first_moment_for_var_loss = None
            self._instance_second_moment_for_loss = None

        if not isinstance(camera, Cameras):
            return outputs

        if self.training and self.step < int(self.config.instance_warmup_steps):
            return outputs

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return outputs
        else:
            crop_ids = None

        overlay_lbl = self._cluster_overlay_label_source()

        if crop_ids is not None:
            means_crop = self.means[crop_ids]
            quats_crop = self.quats[crop_ids]
            scales_crop = self.scales[crop_ids]
            opacities_crop = self.opacities[crop_ids]
            instance_feats_crop = self.gauss_params["instance_feats"][crop_ids]
            dino_feats_crop = (
                self.gauss_params["dino_feats"][crop_ids]
                if "dino_feats" in self.gauss_params
                else None
            )
            cluster_labels_crop = (
                overlay_lbl[crop_ids]
                if overlay_lbl is not None and overlay_lbl.shape[0] == self.means.shape[0]
                else None
            )
        else:
            means_crop = self.means
            quats_crop = self.quats
            scales_crop = self.scales
            opacities_crop = self.opacities
            instance_feats_crop = self.gauss_params["instance_feats"]
            dino_feats_crop = self.gauss_params.get("dino_feats")
            cluster_labels_crop = overlay_lbl

        cam_feat = copy.deepcopy(camera)
        if cam_feat.metadata is not None:
            gt_h = cam_feat.metadata.get("gt_image_height")
            gt_w = cam_feat.metadata.get("gt_image_width")
            if gt_h is not None and gt_w is not None:
                self._snap_camera_to_gt_image_resolution(cam_feat, int(gt_h), int(gt_w))

        viewmat = get_viewmat(optimized_camera_to_world.detach())
        downscale_factor = self.config.feature_downscale_factor
        if cam_feat.metadata is not None and "feature_downscale_factor" in cam_feat.metadata:
            downscale_factor = int(cam_feat.metadata["feature_downscale_factor"])
        downscale_factor = max(1, int(downscale_factor))

        cam_feat.rescale_output_resolution(1 / downscale_factor)
        feat_w, feat_h = int(cam_feat.width.item()), int(cam_feat.height.item())
        feat_k = cam_feat.get_intrinsics_matrices().to(means_crop.device)

        instance_render, _, _ = rasterization(
            means=means_crop.detach(),
            quats=quats_crop.detach(),
            scales=torch.exp(scales_crop.detach()),
            opacities=torch.sigmoid(opacities_crop.detach()).squeeze(-1),
            colors=instance_feats_crop,
            viewmats=viewmat,
            Ks=feat_k,
            width=feat_w,
            height=feat_h,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )
        instance_second_moment_render = None
        if self.training and float(self.config.var_lambda) > 0.0:
            instance_second_moment_render, _, _ = rasterization(
                means=means_crop.detach(),
                quats=quats_crop.detach(),
                scales=torch.exp(scales_crop.detach()),
                opacities=torch.sigmoid(opacities_crop.detach()).squeeze(-1),
                colors=instance_feats_crop.pow(2),
                viewmats=viewmat,
                Ks=feat_k,
                width=feat_w,
                height=feat_h,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode="RGB",
                sparse_grad=False,
                absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
                rasterize_mode=self.config.rasterize_mode,
            )

        if not torch.isfinite(instance_render).all():
            raise RuntimeError(f"[non-finite instance_render] step={self.step}")
        if instance_second_moment_render is not None and not torch.isfinite(instance_second_moment_render).all():
            raise RuntimeError(f"[non-finite instance_second_moment_render] step={self.step}")

        instance_flat = instance_render.view(feat_h * feat_w, -1)
        instance_decoded = self._decode_instance(instance_flat)
        if self.training:
            self._instance_map_for_loss = (
                instance_decoded.view(feat_h, feat_w, -1).permute(2, 0, 1).contiguous()
            )
            if float(self.config.var_lambda) > 0.0 and instance_second_moment_render is not None:
                self._instance_first_moment_for_var_loss = (
                    instance_render.view(feat_h, feat_w, -1)
                    .permute(2, 0, 1)
                    .contiguous()
                    .to(torch.float32)
                )
                self._instance_second_moment_for_loss = (
                    instance_second_moment_render.view(feat_h, feat_w, -1)
                    .permute(2, 0, 1)
                    .contiguous()
                    .to(torch.float32)
                )
        outputs["instance"] = self._instance_embedding_rows_to_rgb_pca(
            instance_decoded, feat_h, feat_w
        )

        if (
            self.cluster_overlay_enabled
            and cluster_labels_crop is not None
            and cluster_labels_crop.shape[0] == means_crop.shape[0]
        ):
            with torch.no_grad():
                cl = cluster_labels_crop.to(device=means_crop.device, dtype=torch.long)
                cluster_opacity = opacities_crop
                colors_raster = self._build_cluster_gaussian_rgb(cl)
                if self.cluster_hide_noise:
                    noise_mask = cl < 0
                    if noise_mask.any():
                        cluster_opacity = cluster_opacity.clone()
                        cluster_opacity[noise_mask] = -100.0
                cluster_render, _, _ = rasterization(
                    means=means_crop,
                    quats=quats_crop,
                    scales=torch.exp(scales_crop),
                    opacities=torch.sigmoid(cluster_opacity).squeeze(-1),
                    colors=colors_raster,
                    viewmats=viewmat,
                    Ks=feat_k,
                    width=feat_w,
                    height=feat_h,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="RGB",
                    sparse_grad=False,
                    absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
                    rasterize_mode=self.config.rasterize_mode,
                )
                if cluster_render.ndim == 4 and cluster_render.shape[0] == 1:
                    cluster_render = cluster_render.squeeze(0)
                outputs["cluster"] = cluster_render
                outputs["rgb"] = cluster_render

        outputs = self._add_language_outputs(
            camera=camera,
            outputs=outputs,
            means_crop=means_crop,
            quats_crop=quats_crop,
            scales_crop=scales_crop,
            opacities_crop=opacities_crop,
            optimized_camera_to_world=optimized_camera_to_world,
            crop_ids=crop_ids,
        )

        if self._dino_active() and dino_feats_crop is not None:
            cam_render = copy.deepcopy(camera)
            if cam_render.metadata is not None:
                gt_h = cam_render.metadata.get("gt_image_height")
                gt_w = cam_render.metadata.get("gt_image_width")
                if gt_h is not None and gt_w is not None:
                    self._snap_camera_to_gt_image_resolution(cam_render, int(gt_h), int(gt_w))
            render_scale = self._get_downscale_factor()
            cam_render.rescale_output_resolution(1 / render_scale)
            render_h = int(cam_render.height.item())
            render_w = int(cam_render.width.item())
            render_k = cam_render.get_intrinsics_matrices().to(means_crop.device)
            outputs = self._add_dino_outputs(
                outputs=outputs,
                means_crop=means_crop,
                quats_crop=quats_crop,
                scales_crop=scales_crop,
                opacities_crop=opacities_crop,
                dino_feats_crop=dino_feats_crop,
                viewmat=get_viewmat(optimized_camera_to_world.detach()),
                k=render_k,
                render_h=render_h,
                render_w=render_w,
                camera=camera,
            )

        if not self.training and self.lang_highlight_overlay_enabled:
            background = outputs.get("background")
            if background is None:
                background = torch.ones(
                    int(camera.height.item()),
                    int(camera.width.item()),
                    3,
                    device=self.device,
                    dtype=torch.float32,
                )
            lang_rgb = self._render_language_highlight_3d(
                camera=camera,
                optimized_camera_to_world=optimized_camera_to_world,
                means_crop=means_crop,
                quats_crop=quats_crop,
                scales_crop=scales_crop,
                opacities_crop=opacities_crop,
                crop_ids=crop_ids,
                background=background,
            )
            if lang_rgb is not None:
                outputs["lang_highlight"] = lang_rgb
                outputs["rgb"] = lang_rgb

        return outputs

    def _lang_active(self) -> bool:
        if self.datamanager is None or self.image_encoder is None:
            return False
        if not getattr(self.datamanager, "use_clip", False):
            return False
        lang_step = int(getattr(self.datamanager, "lang_step", self.config.lang_step))
        return self.step >= lang_step or self.loaded_ckpt

    def _dino_active(self) -> bool:
        if not self._lang_active():
            return False
        if float(self.config.dino_loss_weight) <= 0.0:
            return False
        if self.dino_decoder is None or "dino_feats" not in self.gauss_params:
            return False
        dino_step = int(getattr(self.datamanager, "dino_step", self.config.lang_step))
        return self.step >= dino_step or self.loaded_ckpt

    def _clip_scale_ratio_sweep(self) -> torch.Tensor:
        if self.datamanager is not None:
            tile_min, tile_max = self.datamanager.config.patch_tile_size_range
        else:
            tile_min, tile_max = 0.05, float(self.config.max_scale)
        return torch.linspace(tile_min, tile_max, self.config.n_scales, device=self.device)

    def _depth_adjusted_clip_scale(
        self,
        scale_ratio: Union[float, torch.Tensor],
        render_h: int,
        depth: torch.Tensor,
        fy: float,
    ) -> torch.Tensor:
        depth_flat = depth.reshape(-1, 1).to(device=self.device, dtype=torch.float32)
        ratio = (
            scale_ratio
            if isinstance(scale_ratio, torch.Tensor)
            else torch.tensor(scale_ratio, device=self.device, dtype=torch.float32)
        )
        return ratio * float(render_h) * (depth_flat / float(fy)).clamp_min(1e-6)

    def _gaussian_camera_depths(self, means: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
        vm = viewmat.reshape(-1, 4, 4)[0]
        homog = torch.cat(
            [means, torch.ones(means.shape[0], 1, device=means.device, dtype=means.dtype)],
            dim=-1,
        )
        cam = homog @ vm.T
        return cam[:, 2:3].abs().clamp_min(1e-6)

    def _normalize_depth_map(self, depth_im: torch.Tensor) -> torch.Tensor:
        while depth_im.ndim > 2:
            if depth_im.shape[-1] == 1:
                depth_im = depth_im.squeeze(-1)
            else:
                depth_im = depth_im.squeeze(0)
        return depth_im

    def _sample_depth_for_clip_pixels(
        self,
        depth_im: torch.Tensor,
        random_pixels: torch.Tensor,
        clip_w: int,
        clip_downscale: int,
        gt_h: int,
        gt_w: int,
    ) -> torch.Tensor:
        """Map CLIP-grid pixel indices to the rendered depth map (may be downscaled)."""
        depth_im = self._normalize_depth_map(depth_im)
        depth_h, depth_w = int(depth_im.shape[0]), int(depth_im.shape[1])
        y_ds = random_pixels // clip_w
        x_ds = random_pixels % clip_w
        y_gt = y_ds.to(torch.float32) * float(clip_downscale)
        x_gt = x_ds.to(torch.float32) * float(clip_downscale)
        y_idx = (y_gt * depth_h / max(gt_h, 1)).long().clamp(0, depth_h - 1)
        x_idx = (x_gt * depth_w / max(gt_w, 1)).long().clamp(0, depth_w - 1)
        return depth_im[y_idx, x_idx].to(torch.float32).unsqueeze(1)

    def _depth_at_clip_resolution(
        self, depth_im: torch.Tensor, clip_h: int, clip_w: int
    ) -> torch.Tensor:
        depth_im = self._normalize_depth_map(depth_im)
        if depth_im.shape == (clip_h, clip_w):
            return depth_im
        return F.interpolate(
            depth_im.unsqueeze(0).unsqueeze(0),
            size=(clip_h, clip_w),
            mode="nearest",
        ).squeeze(0).squeeze(0)

    def _upsample_clip_map(
        self, clip_map: torch.Tensor, clip_h: int, clip_w: int, full_h: int, full_w: int
    ) -> torch.Tensor:
        """Nearest-neighbor upsample from CLIP supervision resolution to display resolution."""
        if clip_h == full_h and clip_w == full_w:
            return clip_map
        if clip_map.ndim == 1:
            clip_map = clip_map.view(clip_h, clip_w)
        elif clip_map.ndim == 3 and clip_map.shape[-1] == 1:
            clip_map = clip_map.view(clip_h, clip_w)
        return F.interpolate(
            clip_map.unsqueeze(0).unsqueeze(0),
            size=(full_h, full_w),
            mode="nearest",
        ).squeeze(0).squeeze(0)

    def _add_language_outputs(
        self,
        *,
        camera: Cameras,
        outputs: Dict[str, Union[torch.Tensor, List]],
        means_crop: torch.Tensor,
        quats_crop: torch.Tensor,
        scales_crop: torch.Tensor,
        opacities_crop: torch.Tensor,
        optimized_camera_to_world,
        crop_ids: Optional[torch.Tensor],
    ) -> Dict[str, Union[torch.Tensor, List]]:
        if not self._lang_active():
            return outputs

        cam_clip = copy.deepcopy(camera)
        if cam_clip.metadata is not None:
            gt_h = cam_clip.metadata.get("gt_image_height")
            gt_w = cam_clip.metadata.get("gt_image_width")
            if gt_h is not None and gt_w is not None:
                self._snap_camera_to_gt_image_resolution(cam_clip, int(gt_h), int(gt_w))

        viewmat = get_viewmat(optimized_camera_to_world.detach())
        clip_downscale = int(
            getattr(self.datamanager.config, "clip_downscale_factor", self.config.feature_downscale_factor)
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

        clip_hash = self.clip_field.get_clip_hash(means_crop)
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
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )
        cam_clip.rescale_output_resolution(clip_downscale)

        if self.training:
            depth_im = outputs.get("depth")
            if depth_im is None:
                raise RuntimeError("[lang] depth required for CLIP scale; set output_depth_during_training=True")
            random_pixels = self.datamanager.random_pixels.to(self.device).to(torch.long)
            clip_grid = clip_h * clip_w
            if random_pixels.max().item() >= clip_grid:
                raise RuntimeError(
                    f"[random_pixels out of range] max={random_pixels.max().item()} "
                    f"grid={clip_grid} clip=({clip_h},{clip_w}) gt=({gt_h},{gt_w})"
                )
            depth_samples = self._sample_depth_for_clip_pixels(
                depth_im=depth_im,
                random_pixels=random_pixels,
                clip_w=clip_w,
                clip_downscale=clip_downscale,
                gt_h=gt_h,
                gt_w=gt_w,
            ).to(self.device)
            clip_fy = float(cam_clip.fy.reshape(-1)[0].item())
            clip_scale = self.datamanager.curr_scale * clip_h * (depth_samples / clip_fy)
            clip_field_flat = clip_render.view(clip_h * clip_w, -1)
            outputs["clip"] = self.clip_field.decode_clip_from_features(
                features=clip_field_flat,
                clip_scale=clip_scale,
                random_pixels=random_pixels,
            ).to(dtype=torch.float32)
            outputs["clip_scale"] = clip_scale
        elif self.image_encoder is not None and len(self._active_language_positives()) > 0:
            depth_im = outputs.get("depth")
            if depth_im is None:
                raise RuntimeError(
                    "[lang inference] rendered depth is required for CLIP relevancy."
                )
            clip_fy = float(camera.get_intrinsics_matrices().reshape(-1, 3, 3)[0, 1, 1].item())
            outputs = self._add_language_inference_outputs(
                outputs=outputs,
                means_crop=means_crop,
                clip_field_flat=clip_render.view(clip_h * clip_w, -1),
                clip_h=clip_h,
                clip_w=clip_w,
                full_h=full_h,
                full_w=full_w,
                depth_im=depth_im,
                clip_fy=clip_fy,
                viewmat=viewmat,
                crop_ids=crop_ids,
            )

        return outputs

    def _add_dino_outputs(
        self,
        *,
        outputs: Dict,
        means_crop: torch.Tensor,
        quats_crop: torch.Tensor,
        scales_crop: torch.Tensor,
        opacities_crop: torch.Tensor,
        dino_feats_crop: torch.Tensor,
        viewmat: torch.Tensor,
        k: torch.Tensor,
        render_h: int,
        render_w: int,
        camera: Cameras,
    ) -> Dict:
        """Rasterize per-Gaussian DINO embeddings and decode to PCA DINO dim."""
        p_size = 14
        downscale = (self.config.dino_rescale_factor * MAX_DINO_SIZE / max(render_h, render_w)) / p_size
        if camera.metadata is not None and "clip_downscale_factor" not in camera.metadata:
            downscale = 1.0
        h, w = get_img_resolution(render_h, render_w, p=p_size)
        dino_k = k.clone()
        dino_k[:, :2, :] *= downscale
        dino_h = self.config.dino_rescale_factor * (h // p_size)
        dino_w = self.config.dino_rescale_factor * (w // p_size)
        if camera.metadata is not None and "clip_downscale_factor" not in camera.metadata:
            dino_h, dino_w = render_h, render_w

        detach_geo = self.training
        dino_render, dino_alpha, _ = rasterization(
            means=means_crop.detach() if detach_geo else means_crop,
            quats=quats_crop.detach() if detach_geo else quats_crop,
            scales=torch.exp(scales_crop.detach() if detach_geo else scales_crop),
            opacities=torch.sigmoid(opacities_crop.detach() if detach_geo else opacities_crop).squeeze(-1),
            colors=dino_feats_crop,
            viewmats=viewmat,
            Ks=dino_k,
            width=dino_w,
            height=dino_h,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
        )
        if dino_render.ndim == 4 and dino_render.shape[0] == 1:
            dino_render = dino_render.squeeze(0)
        if dino_alpha.ndim == 4 and dino_alpha.shape[0] == 1:
            dino_alpha = dino_alpha.squeeze(0)

        alpha = dino_alpha.clamp_min(1e-6)
        if alpha.ndim == 3 and alpha.shape[-1] == 1:
            alpha_mask = alpha.squeeze(-1)
        else:
            alpha_mask = alpha
        dino_accum = torch.where(
            alpha_mask.unsqueeze(-1) > 0,
            dino_render / alpha.detach(),
            torch.zeros_like(dino_render),
        )
        dino_flat = self.dino_decoder(dino_accum.reshape(-1, self.config.dino_feat_dim))
        outputs["dino"] = dino_flat.view(dino_h, dino_w, self.config.dino_dim)
        if not self.training:
            outputs["dino"][alpha_mask < 0.8] = 0
        return outputs

    @torch.no_grad()
    def get_max_across(
        self,
        clip_field_flat: torch.Tensor,
        clip_h: int,
        clip_w: int,
        depth: torch.Tensor,
        fy: float,
        preset_scales: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-phrase pixel relevancy across scale sweep at CLIP resolution.

        Uses the same depth-adjusted scale as training:
        ``scale_ratio * clip_h * (depth / fy)``.
        """
        n_phrases = len(self.image_encoder.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        n_phrases_maxs = [None for _ in range(n_phrases)]
        num_pixels = clip_h * clip_w

        if preset_scales is not None:
            scales_list = torch.tensor(preset_scales, device=self.device, dtype=torch.float32)
        else:
            scales_list = self._clip_scale_ratio_sweep()

        flat = clip_field_flat.reshape(num_pixels, -1)
        depth_clip = self._depth_at_clip_resolution(depth, clip_h, clip_w)

        for i, scale_ratio in enumerate(scales_list):
            clip_scale = self._depth_adjusted_clip_scale(scale_ratio, clip_h, depth_clip, fy)
            clip_output_im = self.clip_field.decode_clip_from_features(
                features=flat,
                clip_scale=clip_scale,
            ).to(dtype=torch.float32).view(clip_h, clip_w, -1)

            for j in range(n_phrases):
                if preset_scales is None or j == i:
                    probs = self.image_encoder.get_relevancy(
                        clip_output_im.reshape(-1, self.image_encoder.embedding_dim), j
                    )
                    pos_prob = probs[..., 0:1]
                    if n_phrases_sims[j] is None or pos_prob.max() > n_phrases_sims[j].max():
                        n_phrases_maxs[j] = scale_ratio
                        n_phrases_sims[j] = pos_prob

        return torch.stack(n_phrases_sims), torch.stack(n_phrases_maxs)

    def _add_language_inference_outputs(
        self,
        *,
        outputs: Dict,
        means_crop: torch.Tensor,
        clip_field_flat: torch.Tensor,
        clip_h: int,
        clip_w: int,
        full_h: int,
        full_w: int,
        depth_im: torch.Tensor,
        clip_fy: float,
        viewmat: torch.Tensor,
        crop_ids: Optional[torch.Tensor],
    ) -> Dict:
        max_across, self.best_scales = self.get_max_across(
            clip_field_flat, clip_h, clip_w, depth=depth_im, fy=clip_fy
        )
        thresh = float(
            self.lang_relevancy_thresh_slider.value
            if self.lang_relevancy_thresh_slider is not None
            else self.config.lang_relevancy_thresh
        )
        score_thresh = float(
            self.lang_relevancy_score_slider.value
            if self.lang_relevancy_score_slider is not None
            else self.config.lang_relevancy_score
        )

        for i, obj in enumerate(self.image_encoder.positives):
            sim = max_across[i].clone()
            sim[sim < thresh] = 0
            relevance_map = self._upsample_clip_map(
                sim.view(clip_h, clip_w), clip_h, clip_w, full_h, full_w
            )
            outputs[f"relevancy_{obj}"] = relevance_map.unsqueeze(-1)

            binary_mask = (relevance_map > score_thresh).float()
            binary_mask_rgb = binary_mask.unsqueeze(-1).expand(-1, -1, 3)
            outputs[f"mask_relevancy_{obj}"] = binary_mask_rgb

            rgb_image = outputs.get("_scene_rgb", outputs["rgb"])
            if rgb_image.shape[:2] != relevance_map.shape[:2]:
                rgb_image = F.interpolate(
                    rgb_image.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(relevance_map.shape[0], relevance_map.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0).to(dtype=binary_mask_rgb.dtype)
            white_background = torch.ones_like(rgb_image)
            masked_rgb = binary_mask_rgb * rgb_image + (1 - binary_mask_rgb) * white_background
            outputs[f"rgb_relevancy_{obj}"] = masked_rgb

        if (
            bool(self.config.lang_use_hybrid_cluster_query)
            and self._pending_language_cluster_query
            and self.cluster_labels is not None
            and len(self._active_language_positives()) > 0
        ):
            outputs = self._apply_hybrid_cluster_language_highlight(
                outputs=outputs,
                means_crop=means_crop,
                phrase_idx=0,
                crop_ids=crop_ids,
                viewmat=viewmat,
                clip_h=clip_h,
                clip_fy=clip_fy,
            )
            self._pending_language_cluster_query = False
        return outputs

    @torch.no_grad()
    def _apply_hybrid_cluster_language_highlight(
        self,
        *,
        outputs: Dict,
        means_crop: torch.Tensor,
        phrase_idx: int,
        crop_ids: Optional[torch.Tensor],
        viewmat: torch.Tensor,
        clip_h: int,
        clip_fy: float,
    ) -> Dict:
        if self.cluster_labels is None or self.image_encoder is None:
            return outputs
        if phrase_idx >= len(self.image_encoder.positives):
            return outputs

        scale_ratio = (
            self.best_scales[phrase_idx]
            if self.best_scales is not None
            else torch.tensor(0.05, device=self.device)
        )
        gaussian_depth = self._gaussian_camera_depths(means_crop, viewmat)
        scale_tensor = self._depth_adjusted_clip_scale(
            scale_ratio, clip_h, gaussian_depth, clip_fy
        )
        clip_hash = self.clip_field.get_clip_hash(means_crop)
        clip_feats = self.clip_field.decode_clip_from_features(clip_hash, scale_tensor)
        relevancy = self.image_encoder.get_relevancy(clip_feats, phrase_idx)[..., 0]

        labels = self.cluster_labels
        if crop_ids is not None:
            labels = labels[crop_ids]

        scores = aggregate_gaussian_relevancy_by_cluster(relevancy, labels)
        cluster_thresh = float(
            self.lang_cluster_threshold_slider.value
            if self.lang_cluster_threshold_slider is not None
            else self.config.lang_cluster_threshold
        )
        selected = select_clusters_from_scores(scores, threshold=cluster_thresh, mode="relative_max")
        highlight = build_gaussian_highlight_mask(self.cluster_labels, selected)
        self._lang_highlight_mask = highlight

        if selected and bool(highlight.any()):
            self.lang_highlight_overlay_enabled = True
            CONSOLE.print(f"[green]Language query matched clusters: {selected}[/green]")
        else:
            self.lang_highlight_overlay_enabled = False
            CONSOLE.print("[yellow]No clusters matched language query.[/yellow]")
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if (
            self.training
            and self._instance_map_for_loss is not None
            and "instance_labels" in batch
            and self.step >= int(self.config.inst2d_from_iter)
            and (
                int(self.config.inst2d_interval) <= 1
                or (self.step % int(self.config.inst2d_interval) == 0)
            )
        ):
            instance_map = self._instance_map_for_loss.to(torch.float32)
            instance_labels = batch["instance_labels"].to(self.device).to(torch.long)
            if tuple(instance_map.shape[1:]) != tuple(instance_labels.shape):
                raise RuntimeError(
                    f"[instance label-map resolution mismatch] feat={tuple(instance_map.shape[1:])} labels={tuple(instance_labels.shape)}"
                )
            if not torch.isfinite(instance_map).all():
                raise RuntimeError(f"[non-finite instance_map] step={self.step}")

            valid_ids = instance_labels[instance_labels >= 0]
            if valid_ids.numel() > 0 and valid_ids.min().item() != 0:
                raise RuntimeError(
                    f"[instance_labels must be contiguous from 0] min_valid={valid_ids.min().item()} step={self.step}"
                )

            inst2d = instance_2d_loss(
                network_output=instance_map,
                gt_masks=instance_labels,
                mask_dim=int(instance_map.shape[0]),
                sample_size=int(self.config.inst2d_sample_size),
                gamma=float(self.config.inst2d_gamma),
                weights=list(self.config.inst2d_weights),
                normalize=bool(self.config.inst2d_normalize),
            )
            ramp_steps = int(self.config.inst2d_ramp_steps)
            if ramp_steps > 0:
                ramp = min(1.0, max(0.0, (self.step - int(self.config.inst2d_from_iter) + 1) / float(ramp_steps)))
            else:
                ramp = 1.0
            loss_dict["instance_loss"] = float(self.config.inst2d_lambda) * ramp * inst2d["total"]
            loss_dict["instance_loss_pos"] = inst2d["positive"]
            loss_dict["instance_loss_neg"] = inst2d["negative"]

        if (
            self.training
            and float(self.config.var_lambda) > 0.0
            and self.step >= int(self.config.instance_warmup_steps)
        ):
            first_moment = self._instance_first_moment_for_var_loss
            second_moment = self._instance_second_moment_for_loss
            if first_moment is None or second_moment is None:
                raise RuntimeError(
                    "[variance regularization] missing moment maps while var_lambda > 0."
                )
            variance = second_moment - first_moment.pow(2)
            var_loss = variance.pow(2).mean()
            loss_dict["variance_loss"] = float(self.config.var_lambda) * var_loss

        if self.training and "clip" in outputs and "clip" in batch:
            lang_weight = float(self.config.lang_loss_weight)
            if lang_weight == 0.0:
                loss_dict["lang_loss"] = outputs["clip"].new_zeros(())
            else:
                unreduced_clip = lang_weight * F.huber_loss(
                    outputs["clip"],
                    batch["clip"].to(self.device).to(torch.float32),
                    delta=1.25,
                    reduction="none",
                )
                loss_dict["lang_loss"] = unreduced_clip.sum(dim=-1).nanmean()
            loss_dict["clip_loss"] = loss_dict["lang_loss"]

        if self.training and self._dino_active() and "dino" in outputs and "dino" in batch:
            gt = batch["dino"].to(self.device).to(torch.float32)
            pred = outputs["dino"]
            gt_chw = gt.permute(2, 0, 1).unsqueeze(0)
            gt_resized = F.interpolate(
                gt_chw,
                size=(pred.shape[0], pred.shape[1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            dino_weight = float(self.config.dino_loss_weight)
            loss_dict["dino_loss"] = dino_weight * F.mse_loss(pred, gt_resized)

        return loss_dict
