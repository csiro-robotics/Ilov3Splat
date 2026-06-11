import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ilov3splat.utils.feature_dataloader import FeatureDataloader
from ilov3splat.utils.image_encoder import BaseImageEncoder
from ilov3splat.utils.patch_embedding_dataloader import PatchEmbeddingDataloader


class PyramidEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cache_path: str = None,
    ):
        assert "tile_size_range" in cfg
        assert "tile_size_res" in cfg
        assert "stride_scaler" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.tile_sizes = torch.linspace(*cfg["tile_size_range"], cfg["tile_size_res"]).to(device)
        self.strider_scaler_list = [self._stride_scaler(tr.item(), cfg["stride_scaler"]) for tr in self.tile_sizes]

        self.model = model
        self.embed_size = self.model.embedding_dim
        self.data_dict = {}
        super().__init__(cfg, device, image_list, cache_path)

    def __call__(self, img_points, scale=None):
        if scale is None:
            return self._random_scales(img_points)
        return self._uniform_scales(img_points, scale)

    def _stride_scaler(self, tile_ratio, stride_scaler):
        return np.interp(tile_ratio, [0.05, 0.15], [1.0, stride_scaler])

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            for f_name in os.listdir(self.cache_path):
                os.remove(os.path.join(self.cache_path, f_name))
            raise ValueError("Config mismatch")
        raise FileNotFoundError

    def create(self, image_list):
        os.makedirs(self.cache_path, exist_ok=True)
        for i, tr in enumerate(tqdm(self.tile_sizes, desc="Scales")):
            stride_scaler = self.strider_scaler_list[i]
            self.data_dict[i] = PatchEmbeddingDataloader(
                cfg={
                    "tile_ratio": tr.item(),
                    "stride_ratio": stride_scaler,
                    "image_shape": self.cfg["image_shape"],
                    "model_name": self.cfg["model_name"],
                },
                device=self.device,
                model=self.model,
                image_list=image_list,
                cache_path=Path(f"{self.cache_path}/level_{i}.npy"),
            )

    def save(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        with open(cache_info_path, "w") as f:
            f.write(json.dumps(self.cfg))

    def _random_scales(self, img_points):
        img_points = img_points.to(self.device)
        random_scale_bin = torch.randint(self.tile_sizes.shape[0] - 1, size=(img_points.shape[0],), device=self.device)
        random_scale_weight = torch.rand(img_points.shape[0], dtype=torch.float16, device=self.device)

        stepsize = (self.tile_sizes[1] - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0])

        bottom_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)
        top_interp = torch.zeros((img_points.shape[0], self.embed_size), dtype=torch.float16, device=self.device)

        for i in range(len(self.tile_sizes) - 1):
            bottom_interp[random_scale_bin == i] = self.data_dict[i](img_points[random_scale_bin == i])
            top_interp[random_scale_bin == i] = self.data_dict[i + 1](img_points[random_scale_bin == i])

        return (
            torch.lerp(bottom_interp, top_interp, random_scale_weight[..., None]),
            (random_scale_bin * stepsize + random_scale_weight * stepsize)[..., None],
        )

    def _uniform_scales(self, img_points, scale):
        scale_bin = torch.floor(
            (scale - self.tile_sizes[0]) / (self.tile_sizes[-1] - self.tile_sizes[0]) * (self.tile_sizes.shape[0] - 1)
        ).to(torch.int64)
        scale_weight = (scale - self.tile_sizes[scale_bin]) / (
            self.tile_sizes[scale_bin + 1] - self.tile_sizes[scale_bin]
        )
        interp_lst = torch.stack([interp(img_points) for interp in self.data_dict.values()])
        point_inds = torch.arange(img_points.shape[0])
        interp = torch.lerp(
            interp_lst[scale_bin, point_inds],
            interp_lst[scale_bin + 1, point_inds],
            torch.Tensor([scale_weight]).half().to(self.device)[..., None],
        )
        return interp / interp.norm(dim=-1, keepdim=True), scale
