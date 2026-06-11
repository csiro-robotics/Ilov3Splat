from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModel, get_viewmat
from nerfstudio.utils.eval_utils import eval_load_checkpoint
from nerfstudio.utils.rich_utils import CONSOLE

try:
    from gsplat.rendering import rasterization
except ImportError as exc:  # pragma: no cover
    raise ImportError("gsplat>=1.0.0 is required for ilov3splat eval") from exc


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Convert binary mask to boundary mask."""
    h, w = mask.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = int(round(dilation_ratio * img_diag))
    dilation = max(1, dilation)
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # type: ignore[arg-type]
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    return mask - mask_erode


def generate_eval_output_name() -> str:
    uid = str(uuid.uuid4())[:8]
    dtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{dtime}-{uid}"


def _unwrap_model(model: Any) -> Any:
    if hasattr(model, "module"):
        return model.module
    return model


def _model_background_color(model: Any, device: torch.device) -> torch.Tensor:
    bg_name = getattr(model.config, "background_color", "black")
    if bg_name == "white":
        bg_color = [1.0, 1.0, 1.0]
    else:
        bg_color = [0.0, 0.0, 0.0]
    return torch.tensor(bg_color, dtype=torch.float32, device=device)


def _render_colors_and_sh_degree(
    model: Any,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
) -> tuple[torch.Tensor, int | None]:
    if int(getattr(model.config, "sh_degree", 0)) > 0:
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
        sh_degree = int(getattr(model.config, "sh_degree", 0))
        return colors, sh_degree
    return torch.sigmoid(features_dc), None


def render_frames(
    cameras: list[Cameras],
    model: Any,
    override_color: torch.Tensor | None = None,
    override_opacity: torch.Tensor | None = None,
    override_background: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Render RGB maps for a camera list. Returns (B, 3, H, W)."""
    model = _unwrap_model(model)
    device = model.means.device
    if override_background is None:
        bg = _model_background_color(model, device)
    else:
        bg = torch.as_tensor(override_background, dtype=torch.float32, device=device).view(3)
    rendered_images: list[torch.Tensor] = []

    if override_color is None:
        color_arg, sh_degree = _render_colors_and_sh_degree(
            model, model.features_dc, model.features_rest
        )
    else:
        color_arg, sh_degree = override_color, None

    for cam in cameras:
        cam_dev = cam.to(device)
        width = int(cam_dev.width.flatten()[0].item())
        height = int(cam_dev.height.flatten()[0].item())
        viewmat = get_viewmat(cam_dev.camera_to_worlds)
        k = cam_dev.get_intrinsics_matrices().to(device)
        rgb, _, _ = rasterization(
            means=model.means,
            quats=model.quats,
            scales=torch.exp(model.scales),
            opacities=torch.sigmoid(
                override_opacity if override_opacity is not None else model.opacities
            ).squeeze(-1),
            colors=color_arg,
            viewmats=viewmat,
            Ks=k,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=getattr(model.config, "rasterize_mode", "classic"),
            sh_degree=sh_degree,
            backgrounds=bg.view(1, 3),
        )
        img = rgb.clamp(0, 1)
        if img.ndim == 4:
            img = img[0]
        rendered_images.append(img.detach().cpu().permute(2, 0, 1).contiguous())

    return torch.stack(rendered_images), None


def render_alpha(
    cameras: list[Cameras],
    model: Any,
    override_opacity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render alpha maps and return (B, H, W)."""
    model = _unwrap_model(model)
    device = model.means.device
    bg = _model_background_color(model, device)
    rendered_alpha: list[torch.Tensor] = []
    colors = torch.zeros((model.means.shape[0], 3), device=device, dtype=torch.float32)
    opacities = torch.sigmoid(
        override_opacity if override_opacity is not None else model.opacities
    ).squeeze(-1)

    for cam in cameras:
        cam_dev = cam.to(device)
        width = int(cam_dev.width.flatten()[0].item())
        height = int(cam_dev.height.flatten()[0].item())
        viewmat = get_viewmat(cam_dev.camera_to_worlds)
        k = cam_dev.get_intrinsics_matrices().to(device)
        _, alpha, _ = rasterization(
            means=model.means,
            quats=model.quats,
            scales=torch.exp(model.scales),
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=k,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=getattr(model.config, "rasterize_mode", "classic"),
            backgrounds=bg.view(1, 3),
        )
        a = alpha
        if a.ndim == 4:
            a = a[0]
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        rendered_alpha.append(a.detach().cpu())

    return torch.stack(rendered_alpha)


def get_model_paths(model_dir: Path) -> list[Path]:
    """Resolve one or many model run directories containing config.yml."""
    model_dir = model_dir.expanduser().resolve()
    if (model_dir / "config.yml").is_file():
        return [model_dir]
    scenes = model_dir / "scenes"
    if scenes.is_dir():
        return sorted([p for p in scenes.iterdir() if (p / "config.yml").is_file()])
    candidates = sorted(model_dir.glob("**/config.yml"))
    return [c.parent for c in candidates]


def load_ilov3splat_pipeline(
    model_path: Path,
    test_mode: str = "test",
    checkpoint: Path | None = None,
) -> tuple[TrainerConfig, Any]:
    """Load Nerfstudio pipeline + weights from an Ilov3Splat run directory."""
    run_dir = model_path.expanduser().resolve()
    config_path = run_dir / "config.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")

    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)
    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    pipeline.eval()

    if checkpoint is not None:
        ckpt = checkpoint.expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        pipeline.load_pipeline(loaded["pipeline"], loaded.get("step", 0))
        CONSOLE.print(f"[green]Loaded checkpoint {ckpt}[/green]")
    else:
        config.load_dir = run_dir / "nerfstudio_models"
        eval_load_checkpoint(config, pipeline)

    model = _unwrap_model(pipeline.model)
    model.loaded_ckpt = True
    return config, pipeline


def _camera_name_from_idx(path_list: list[Path], idx: int) -> str:
    return path_list[idx].stem


def _dataparser_outputs_for_split(dm: Any, split: str) -> Any | None:
    if split == "train":
        dpo = getattr(dm, "train_dataparser_outputs", None)
        if dpo is not None:
            return dpo
        return getattr(dm.train_dataset, "_dataparser_outputs", None)
    dpo = getattr(dm, "eval_dataparser_outputs", None)
    if dpo is not None:
        return dpo
    return getattr(dm.eval_dataset, "_dataparser_outputs", None)


def _image_filenames_for_split(dm: Any, split: str) -> list[Path]:
    dpo = _dataparser_outputs_for_split(dm, split)
    if dpo is not None and hasattr(dpo, "image_filenames"):
        return [Path(x) for x in dpo.image_filenames]
    dataset = dm.train_dataset if split == "train" else dm.eval_dataset
    if hasattr(dataset, "image_filenames"):
        return [Path(x) for x in dataset.image_filenames]
    return []


def get_split_cameras(pipeline: Any, split: str) -> tuple[list[Cameras], list[str]]:
    dm = pipeline.datamanager
    if split == "train":
        cams_all = dm.train_dataset.cameras
    else:
        cams_all = dm.eval_dataset.cameras
    filenames = _image_filenames_for_split(dm, split)

    cams: list[Cameras] = []
    names: list[str] = []
    for i in range(len(cams_all)):
        cams.append(cams_all[i : i + 1].to(pipeline.model.device))
        if filenames:
            names.append(_camera_name_from_idx(filenames, i))
        else:
            names.append(str(i))
    return cams, names


def render_scene_depth(camera: Cameras, model: Any) -> torch.Tensor:
    """Render depth for one camera using the base Splatfacto path."""
    model = _unwrap_model(model)
    outputs = SplatfactoModel.get_outputs(model, camera)
    depth = outputs.get("depth")
    if depth is None:
        raise RuntimeError(
            "Rendered depth is required for hybrid cluster language eval. "
            "Set output_depth_during_training=True in the model config."
        )
    return depth
