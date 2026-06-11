from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerCheckbox,
    ViewerSlider,
    ViewerText,
)


class Ilov3SplatViewerControls:
    def _setup_viewer_controls(self) -> None:
        if not self.config.cluster_viewer_enable:
            return

        self.cluster_button = ViewerButton(
            name="Run HDBSCAN",
            cb_hook=self._cluster_scene_cb,
            disabled=False,
        )
        self.cluster_toggle_button = ViewerButton(
            name="Toggle RGB/Cluster",
            cb_hook=self._toggle_cluster_overlay_cb,
            disabled=False,
        )
        self.load_saved_features_button = ViewerButton(
            name="Load saved features",
            cb_hook=self._load_saved_features_cb,
            disabled=False,
        )
        self.cluster_min_cluster_slider = ViewerSlider(
            name="Cluster Min Size",
            default_value=float(self.config.cluster_min_cluster_size),
            min_value=2.0,
            max_value=500.0,
            step=1.0,
        )
        self.cluster_min_samples_manual_checkbox = ViewerCheckbox(
            name="Set min_samples",
            default_value=False,
            cb_hook=self._cluster_min_samples_manual_cb,
            hint="Off: min_samples=None (library default). On: use the slider value.",
        )
        default_min_samples = (
            float(self.config.cluster_min_samples)
            if self.config.cluster_min_samples is not None
            else 5.0
        )
        self.cluster_min_samples_slider = ViewerSlider(
            name="Cluster Min Samples",
            default_value=default_min_samples,
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            disabled=True,
            visible=False,
        )
        self.cluster_eps_slider = ViewerSlider(
            name="Cluster Epsilon",
            default_value=float(self.config.cluster_selection_epsilon),
            min_value=0.0,
            max_value=1.0,
            step=0.001,
        )

    def _cluster_min_samples_manual_cb(self, checkbox: ViewerCheckbox) -> None:
        del checkbox
        self._sync_cluster_min_samples_slider_ui()

    def _sync_cluster_min_samples_slider_ui(self) -> None:
        if self.cluster_min_samples_manual_checkbox is None or self.cluster_min_samples_slider is None:
            return
        manual = bool(self.cluster_min_samples_manual_checkbox.value)
        self.cluster_min_samples_slider.visible = manual
        self.cluster_min_samples_slider.disabled = not manual
        if self.cluster_min_samples_slider.gui_handle is not None:
            self.cluster_min_samples_slider.set_visible(manual)
            self.cluster_min_samples_slider.set_disabled(not manual)

    def _cluster_scene_cb(self, button: ViewerButton) -> None:
        del button
        try:
            stats = self.run_gaussian_clustering(save_artifacts=True)
            self.cluster_overlay_enabled = True
            CONSOLE.print(
                f"[green]HDBSCAN complete: clusters={stats['num_clusters']} noise_ratio={stats['noise_ratio']:.2%}[/green]"
            )
            if self.viewer_control.viewer is not None:
                self.viewer_control.viewer._trigger_rerender()
        except Exception as exc:
            CONSOLE.print(f"[red]HDBSCAN failed: {exc}[/red]")

    def _toggle_cluster_overlay_cb(self, button: ViewerButton) -> None:
        del button
        self.cluster_overlay_enabled = not self.cluster_overlay_enabled
        if self.viewer_control.viewer is not None:
            self.viewer_control.viewer._trigger_rerender()

    def _load_saved_features_cb(self, button: ViewerButton) -> None:
        del button
        loaded = self._load_saved_features_from_run()
        if loaded:
            self.cluster_overlay_enabled = True
            CONSOLE.print(f"[green]Loaded saved features: {', '.join(loaded)}[/green]")
        else:
            CONSOLE.print("[yellow]No saved features loaded (check run directory paths).[/yellow]")
        if self.viewer_control.viewer is not None:
            self.viewer_control.viewer._trigger_rerender()

    def _load_saved_features_from_run(self) -> List[str]:
        """Load cluster labels from the experiment run root."""
        loaded: List[str] = []
        run_root = self._cluster_artifact_parent()
        if run_root is None:
            CONSOLE.print(
                "[yellow]Load saved features: no run root (use --load-config so cluster_experiment_root is set).[/yellow]"
            )
            return loaded

        cluster_dir = self._base_cluster_artifact_dir()
        if cluster_dir is not None:

            def _try_load_labels(path: Path, attr: str) -> bool:
                if not path.is_file():
                    return False
                try:
                    arr = np.load(path)
                    t = torch.from_numpy(arr).to(device=self.device, dtype=torch.long)
                    if t.shape[0] != self.means.shape[0]:
                        CONSOLE.print(
                            f"[yellow]{path.name} length {t.shape[0]} != "
                            f"gaussians {self.means.shape[0]}[/yellow]"
                        )
                        return False
                    setattr(self, attr, t)
                    loaded.append(f"{attr} ({path.name})")
                    return True
                except Exception as exc:
                    CONSOLE.print(f"[yellow]Failed to load {path}: {exc}[/yellow]")
                    return False

            labels_path = cluster_dir / "labels.npy"
            legacy_path = cluster_dir / "labels_lift.npy"
            if not _try_load_labels(labels_path, "cluster_labels"):
                _try_load_labels(legacy_path, "cluster_labels")

        return loaded

    def _cluster_overlay_label_source(self) -> Optional[torch.Tensor]:
        return self.cluster_labels

    def _setup_language_viewer_controls(self) -> None:
        if not bool(self.config.lang_viewer_enable):
            return
        self.language_query_box: Optional[ViewerText] = None
        self.lang_relevancy_thresh_slider: Optional[ViewerSlider] = None
        self.lang_relevancy_score_slider: Optional[ViewerSlider] = None
        self.lang_cluster_threshold_slider: Optional[ViewerSlider] = None
        self.language_query_button: Optional[ViewerButton] = None
        self.lang_highlight_toggle_button: Optional[ViewerButton] = None

        self.language_query_box = ViewerText(
            name="Language query",
            default_value="",
            hint="Semicolon-separated phrases (uses OpenCLIP relevancy + cluster aggregation).",
        )
        self.language_query_button = ViewerButton(
            name="Run language query",
            cb_hook=self._language_query_cb,
            disabled=False,
        )
        self.lang_relevancy_thresh_slider = ViewerSlider(
            name="Lang relevancy thresh",
            default_value=float(self.config.lang_relevancy_thresh),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )
        self.lang_relevancy_score_slider = ViewerSlider(
            name="Lang mask score",
            default_value=float(self.config.lang_relevancy_score),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )
        self.lang_cluster_threshold_slider = ViewerSlider(
            name="Lang cluster thresh",
            default_value=float(self.config.lang_cluster_threshold),
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            hint="Relative threshold vs max cluster relevancy score.",
        )
        self.lang_highlight_toggle_button = ViewerButton(
            name="Toggle lang 3D highlight",
            cb_hook=self._toggle_lang_highlight_cb,
            disabled=False,
        )

    def _toggle_lang_highlight_cb(self, button: ViewerButton) -> None:
        del button
        if self._lang_highlight_mask is None or not bool(self._lang_highlight_mask.any()):
            CONSOLE.print(
                "[yellow]No language highlight available. Run a language query first.[/yellow]"
            )
            return
        self.lang_highlight_overlay_enabled = not self.lang_highlight_overlay_enabled
        state = "on" if self.lang_highlight_overlay_enabled else "off"
        CONSOLE.print(f"[green]Language 3D highlight {state}.[/green]")
        if self.viewer_control.viewer is not None:
            self.viewer_control.viewer._trigger_rerender()

    def _active_language_positives(self) -> list[str]:
        if self.image_encoder is None:
            return []
        return [p.strip() for p in self.image_encoder.positives if p.strip()]

    def _language_query_cb(self, button: ViewerButton) -> None:
        del button
        if self.image_encoder is None or self.language_query_box is None:
            CONSOLE.print("[yellow]Language encoder not available.[/yellow]")
            return
        raw = self.language_query_box.value
        phrases = [p.strip() for p in raw.split(";") if p.strip()]
        if not phrases:
            CONSOLE.print("[yellow]Enter at least one phrase in Language query.[/yellow]")
            return
        self.image_encoder.set_positives(phrases)
        self._pending_language_cluster_query = True
        if self.cluster_labels is None:
            loaded = self._load_saved_features_from_run()
            if loaded:
                CONSOLE.print(f"[green]Auto-loaded clusters for language query: {', '.join(loaded)}[/green]")
        if self.viewer_control.viewer is not None:
            self.viewer_control.viewer._trigger_rerender()
        CONSOLE.print(f"[green]Language query set: {phrases}[/green]")

    def _viewer_sh_coefficients_for_raster(
        self, features_dc_crop: torch.Tensor, features_rest_crop: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[int]]:
        colors = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        if int(self.config.sh_degree) > 0:
            sh_deg = min(
                int(self.step) // int(self.config.sh_degree_interval),
                int(self.config.sh_degree),
            )
            return colors, sh_deg
        rgb = torch.sigmoid(colors).squeeze(1)
        return rgb, None

    def _build_cluster_gaussian_rgb(self, labels: torch.Tensor) -> torch.Tensor:
        rgb = torch.sigmoid(self.features_dc.detach().clone())
        dev = rgb.device
        labels = labels.to(device=dev, dtype=torch.long)
        cmap = self.cluster_colormap.to(dev)
        valid = labels >= 0
        if valid.any():
            valid_ids = labels[valid]
            colors = cmap[valid_ids % cmap.shape[0]]
            rgb[valid] = colors.to(rgb.dtype)
        return rgb

    def _build_lang_highlight_gaussian_rgb(
        self, highlight_crop: torch.Tensor, crop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        rgb = torch.sigmoid(self.features_dc.detach().clone())
        if crop_ids is not None:
            rgb = rgb[crop_ids]
        if highlight_crop.shape[0] != rgb.shape[0]:
            raise ValueError(
                f"highlight mask length {highlight_crop.shape[0]} != gaussians {rgb.shape[0]}"
            )
        highlight_color = torch.tensor([0.15, 0.95, 0.25], device=rgb.device, dtype=rgb.dtype)
        colors = rgb.clone()
        if highlight_crop.any():
            colors[highlight_crop] = 0.35 * colors[highlight_crop] + 0.65 * highlight_color
        return colors

    @torch.no_grad()
    def _render_language_highlight_3d(
        self,
        *,
        camera: Cameras,
        optimized_camera_to_world: torch.Tensor,
        means_crop: torch.Tensor,
        quats_crop: torch.Tensor,
        scales_crop: torch.Tensor,
        opacities_crop: torch.Tensor,
        crop_ids: Optional[torch.Tensor],
        background: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        from gsplat.rendering import rasterization
        from gsplat.strategy import DefaultStrategy
        from nerfstudio.models.splatfacto import get_viewmat

        if not self.lang_highlight_overlay_enabled:
            return None
        if self._lang_highlight_mask is None or not bool(self._lang_highlight_mask.any()):
            return None

        if crop_ids is not None:
            highlight_crop = self._lang_highlight_mask[crop_ids]
        else:
            highlight_crop = self._lang_highlight_mask

        colors_raster = self._build_lang_highlight_gaussian_rgb(highlight_crop, crop_ids=crop_ids)
        lang_opacity = opacities_crop
        non_highlight = ~highlight_crop
        if non_highlight.any():
            lang_opacity = opacities_crop.clone()
            lang_opacity[non_highlight] = -100.0

        viewmat = get_viewmat(optimized_camera_to_world.detach())
        K = camera.get_intrinsics_matrices().to(means_crop.device)
        width = int(camera.width.item())
        height = int(camera.height.item())

        lang_render, alpha, _ = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(lang_opacity).squeeze(-1),
            colors=colors_raster,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )
        if lang_render.ndim == 4 and lang_render.shape[0] == 1:
            lang_render = lang_render.squeeze(0)
        if alpha.ndim == 4 and alpha.shape[0] == 1:
            alpha = alpha.squeeze(0)

        bg = background
        if bg.ndim == 1:
            bg = bg.view(1, 1, 3).expand(height, width, 3)
        elif bg.ndim == 3 and bg.shape[0] == 1:
            bg = bg.squeeze(0)
        rgb_out = lang_render[..., :3] + (1 - alpha[..., :1]) * bg[..., :3]
        return torch.clamp(rgb_out, 0.0, 1.0)
