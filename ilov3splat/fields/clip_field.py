from typing import Tuple

import numpy as np
import torch
from torch import Tensor, nn

from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn
except ImportError as exc:
    raise ImportError(
        "tinycudann is required for CLIP language field: pip install tinycudann"
    ) from exc


class ClipLanguageField(Field):
    """CLIP-only hash field for language-aligned feature maps (no instance decoder)."""

    def __init__(
        self,
        grid_layers: Tuple[int] = (12, 12),
        grid_sizes: Tuple[Tuple[int]] = (19, 19),
        grid_resolutions: Tuple[int] = ((16, 128), (128, 512)),
        n_features_level: int = 4,
        clip_n_dims: int = 512,
    ) -> None:
        super().__init__()
        self.spatial_distortion: SceneContraction = SceneContraction()

        self.clip_encs = torch.nn.ModuleList(
            [
                ClipLanguageField._get_encoding(
                    grid_resolutions[i][0],
                    grid_resolutions[i][1],
                    grid_layers[i],
                    indim=3,
                    hash_size=grid_sizes[i],
                    features_per_level=n_features_level,
                )
                for i in range(len(grid_layers))
            ]
        )
        tot_out_dims = sum([e.n_output_dims for e in self.clip_encs])

        self.clip_net = tcnn.Network(
            n_input_dims=tot_out_dims + 1,
            n_output_dims=clip_n_dims,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 3,
            },
        )

    @staticmethod
    def _get_encoding(start_res, end_res, levels, indim=3, hash_size=19, features_per_level=8):
        growth = np.exp((np.log(end_res) - np.log(start_res)) / (levels - 1))
        enc = tcnn.Encoding(
            n_input_dims=indim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": hash_size,
                "base_resolution": start_res,
                "per_level_scale": growth,
            },
        )
        return enc

    def get_clip_hash(self, positions: Tensor) -> Tensor:
        positions = self.spatial_distortion(positions)
        positions = torch.nan_to_num(positions, nan=0.0, posinf=0.0, neginf=0.0)
        positions = (positions + 2.0) / 4.0
        encodings = [e(positions.view(-1, 3)) for e in self.clip_encs]
        return torch.concat(encodings, dim=-1).to(torch.float32)

    def decode_clip_from_features(
        self, features: Tensor, clip_scale: Tensor, random_pixels: Tensor | None = None
    ) -> Tensor:
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        clip_scale = torch.clamp(
            torch.nan_to_num(clip_scale, nan=1.0, posinf=1.0, neginf=1.0), min=1e-6
        )

        if random_pixels is not None:
            clip_features = features[random_pixels]
        else:
            clip_features = features

        if clip_scale.dtype != clip_features.dtype:
            clip_scale = clip_scale.to(dtype=clip_features.dtype)
        if clip_scale.device != clip_features.device:
            clip_scale = clip_scale.to(device=clip_features.device)

        clip_pass = self.clip_net(torch.cat([clip_features, clip_scale.view(-1, 1)], dim=-1))
        clip_norm = clip_pass.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return (clip_pass / clip_norm).to(torch.float32)

    def get_clip_parameters(self):
        params = list(self.clip_encs.parameters())
        params.extend(list(self.clip_net.parameters()))
        return params
