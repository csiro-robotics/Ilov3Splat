from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from ilov3splat.utils.image_encoder import BaseImageEncoder, BaseImageEncoderConfig
from ilov3splat.utils.instance_label_utils import (
    load_instance_labels_from_npz,
    remap_instance_labels,
)
from ilov3splat.utils.dino_dataloader import DinoDataloader
from ilov3splat.utils.openclip_encoder import OpenCLIPNetworkConfig
from ilov3splat.utils.pyramid_embedding_dataloader import PyramidEmbeddingDataloader


@dataclass
class Ilov3SplatDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: Ilov3SplatDataManager)

    dataparser: NerfstudioDataParserConfig = field(
        default_factory=lambda: NerfstudioDataParserConfig(load_3D_points=True)
    )
    feature_downscale_factor: int = 2
    instance_label_source: Literal["npz", "converter"] = "npz"
    instance_mask_subdir: str = "sam"
    instance_mask_npz_key: str = "whole"
    instance_mask_ext: str = ".npz"
    instance_warmup_steps: int = 0
    """When ``step < instance_warmup_steps``, do not load ``instance_labels``."""

    # Language / CLIP supervision
    network: BaseImageEncoderConfig = field(default_factory=OpenCLIPNetworkConfig)
    patch_tile_size_range: Tuple[float, float] = (0.05, 0.5)
    patch_tile_size_res: int = 10
    patch_stride_scaler: float = 0.5
    clip_downscale_factor: int = 2
    lang_step: int = 4000
    """Step at which CLIP supervision begins. Must be < trainer max_num_iterations."""
    lang_cache_subdir: str = "language_cache"
    """Subdirectory under scene outputs for pyramid CLIP cache."""
    dino_step: int = 4000
    """Step at which DINO regularization begins (defaults to lang_step)."""
    dino_pca_dim: int = 16
    """PCA dimension for cached DINO features."""


class Ilov3SplatDataManager(FullImageDatamanager):
    config: Ilov3SplatDataManagerConfig

    def __init__(
        self,
        config: Ilov3SplatDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )
        self.device = device
        self.curr_scale: Optional[torch.Tensor] = None
        self.random_pixels: Optional[torch.Tensor] = None
        self.use_clip = True
        self.lang_step = int(self.config.lang_step)
        self.dino_step = int(self.config.dino_step)

        self.image_encoder: BaseImageEncoder = self.config.network.setup()
        images = [self.cached_train[i]["image"].permute(2, 0, 1)[None, ...] for i in range(len(self.cached_train))]
        images = torch.cat(images)
        if images.shape[1] == 4:
            images = images[:, :3, ...]

        scene_name = Path(self.config.dataparser.data).name
        cache_dir = Path("outputs") / scene_name / self.config.lang_cache_subdir
        clip_cache_path = cache_dir / f"clip_{self.image_encoder.name}"
        dino_cache_path = cache_dir / "dino.npy"
        self.dino_dataloader = DinoDataloader(
            image_list=images,
            device=device,
            cfg={"image_shape": list(images.shape[2:4])},
            cache_path=dino_cache_path,
            pca_dim=int(self.config.dino_pca_dim),
        )
        self.clip_interpolator = PyramidEmbeddingDataloader(
            image_list=images,
            device=device,
            cfg={
                "tile_size_range": list(self.config.patch_tile_size_range),
                "tile_size_res": self.config.patch_tile_size_res,
                "stride_scaler": self.config.patch_stride_scaler,
                "image_shape": list(images.shape[2:4]),
                "model_name": self.image_encoder.name,
            },
            cache_path=clip_cache_path,
            model=self.image_encoder,
        )

    def _get_image_stem(self, image_idx: int) -> str:
        image_path = self.train_dataparser_outputs.image_filenames[image_idx]
        return Path(image_path).stem

    def _try_load_instance_labels_from_npz(
        self, image_idx: int, scaled_height: int, scaled_width: int
    ) -> Optional[torch.Tensor]:
        scene_root = self.config.dataparser.data
        image_stem = self._get_image_stem(image_idx)
        npz_path = (
            Path(scene_root)
            / self.config.instance_mask_subdir
            / f"{image_stem}{self.config.instance_mask_ext}"
        )
        if not npz_path.exists():
            return None

        labels_np = load_instance_labels_from_npz(
            str(npz_path), array_key=self.config.instance_mask_npz_key
        )
        labels = torch.from_numpy(labels_np).long()
        if labels.ndim != 2:
            raise RuntimeError(
                f"[invalid instance label map shape] path={npz_path} shape={tuple(labels.shape)}"
            )

        if labels.shape != (scaled_height, scaled_width):
            labels = cv2.resize(
                labels.cpu().numpy().astype("int32"),
                (scaled_width, scaled_height),
                interpolation=cv2.INTER_NEAREST,
            )
            labels = torch.from_numpy(labels).long()

        labels = remap_instance_labels(labels, ignore_label=-1)
        return labels.to(self.device)

    def _sample_clip_pixels_from_labels(self, instance_labels: torch.Tensor) -> torch.Tensor:
        """Sample ~50% of foreground pixels from SAM instance label map."""
        # Keep indices on CPU; positions grid below is also built on CPU.
        flat = instance_labels.detach().cpu().reshape(-1)
        fg_idx = torch.nonzero(flat >= 0, as_tuple=False).squeeze(1)
        l = flat.numel()
        if fg_idx.numel() == 0:
            return torch.randperm(l)[: max(1, int(l * 0.5))]
        max_samples = int(l * 0.5)
        num_samples = min(fg_idx.numel(), max_samples)
        perm_fg = torch.randperm(fg_idx.numel())
        return fg_idx[perm_fg[:num_samples]]

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Return next training camera/data with instance and optional CLIP labels."""
        image_idx = self.train_unseen_cameras.pop(0)
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        data = deepcopy(self.cached_train[image_idx])
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        camera.metadata["feature_downscale_factor"] = self.config.feature_downscale_factor

        image_h, image_w = data["image"].shape[:2]
        camera.metadata["gt_image_height"] = int(image_h)
        camera.metadata["gt_image_width"] = int(image_w)

        if step < int(self.config.instance_warmup_steps):
            return camera, data

        scaled_height = max(1, image_h // self.config.feature_downscale_factor)
        scaled_width = max(1, image_w // self.config.feature_downscale_factor)
        clip_downscale = max(1, int(self.config.clip_downscale_factor))
        if clip_downscale != int(self.config.feature_downscale_factor):
            raise ValueError(
                "clip_downscale_factor must match feature_downscale_factor for joint instance+CLIP training"
            )

        instance_labels = None
        if self.config.instance_label_source == "npz":
            instance_labels = self._try_load_instance_labels_from_npz(
                image_idx=image_idx,
                scaled_height=scaled_height,
                scaled_width=scaled_width,
            )
        elif self.config.instance_label_source == "converter":
            raise NotImplementedError(
                "instance_label_source='converter' is not implemented; use 'npz' with per-view .npz maps."
            )

        if instance_labels is None:
            image_name = self.train_dataparser_outputs.image_filenames[image_idx]
            raise FileNotFoundError(
                "Instance label file not found for image "
                f"{image_name}. Expected .npz under "
                f"{Path(self.config.dataparser.data) / self.config.instance_mask_subdir}"
            )

        if tuple(instance_labels.shape) != (scaled_height, scaled_width):
            raise RuntimeError(
                f"[instance_labels shape mismatch] got={tuple(instance_labels.shape)} "
                f"expected={(scaled_height, scaled_width)}"
            )
        data["instance_labels"] = instance_labels

        if step >= self.lang_step:
            scale = (
                torch.rand(1).to(self.device)
                * (self.config.patch_tile_size_range[1] - self.config.patch_tile_size_range[0])
                + self.config.patch_tile_size_range[0]
            )
            self.curr_scale = scale
            l = scaled_height * scaled_width
            self.random_pixels = self._sample_clip_pixels_from_labels(instance_labels)
            if self.random_pixels.numel() == 0:
                raise RuntimeError("[random_pixels is empty]")
            if self.random_pixels.max().item() >= l:
                raise RuntimeError(
                    f"[random_pixels out of range] max={self.random_pixels.max().item()} L={l}"
                )

            x = torch.arange(
                0, scaled_width * clip_downscale, clip_downscale
            ).view(1, scaled_width, 1).expand(scaled_height, scaled_width, 1)
            y = torch.arange(
                0, scaled_height * clip_downscale, clip_downscale
            ).view(scaled_height, 1, 1).expand(scaled_height, scaled_width, 1)
            image_idx_tensor = torch.ones(scaled_height, scaled_width, 1) * image_idx
            positions = torch.cat((image_idx_tensor, y, x), dim=-1).view(-1, 3).to(torch.int)
            positions = positions[self.random_pixels]

            with torch.no_grad():
                clip_feats, clip_scale = self.clip_interpolator(positions, scale)
                data["clip"] = clip_feats
                data["clip_scale"] = clip_scale
            camera.metadata["clip_downscale_factor"] = clip_downscale

            if step >= self.dino_step:
                with torch.no_grad():
                    data["dino"] = self.dino_dataloader.get_full_img_feats(image_idx)

        return camera, data
