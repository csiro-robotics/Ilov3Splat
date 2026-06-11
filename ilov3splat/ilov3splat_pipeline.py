from __future__ import annotations

import typing
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig

from ilov3splat.ilov3splat_datamanager import Ilov3SplatDataManagerConfig
from ilov3splat.ilov3splat_model import Ilov3SplatModelConfig


def _argv_value(flag: str) -> Optional[str]:
    """Return ``sys.argv`` value for ``flag`` if present."""
    try:
        i = sys.argv.index(flag)
    except ValueError:
        return None
    if i + 1 >= len(sys.argv):
        return None
    return sys.argv[i + 1]


def _run_artifact_root_from_runtime(config: Ilov3SplatPipelineConfig) -> Optional[Path]:
    """Resolve experiment run directory used for checkpoint/config loading."""
    load_cfg = _argv_value("--load-config")
    if load_cfg:
        return Path(load_cfg).expanduser().resolve().parent

    load_dir = _argv_value("--load-dir")
    if load_dir:
        p = Path(load_dir).expanduser().resolve()
        if p.name == "nerfstudio_models":
            return p.parent
        return p

    return None


def _cluster_artifact_parent_from_data(config: Ilov3SplatPipelineConfig) -> Optional[Path]:
    """Legacy scene-level artifact parent: ``outputs/<scene_name>/``."""
    data = config.datamanager.data
    if data is None:
        return None
    p = Path(data).expanduser().resolve()
    if p.is_file():
        p = p.parent
    name = p.name
    if not name:
        return None
    return (Path("outputs") / name).resolve()


def _cluster_artifact_root(config: Ilov3SplatPipelineConfig) -> Optional[Path]:
    """Prefer run-specific artifact root, then fall back to legacy scene-level root."""
    run_root = _run_artifact_root_from_runtime(config)
    if run_root is not None:
        return run_root
    return _cluster_artifact_parent_from_data(config)


@dataclass
class Ilov3SplatPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory=lambda: Ilov3SplatPipeline)
    datamanager: Ilov3SplatDataManagerConfig = field(default_factory=Ilov3SplatDataManagerConfig)
    model: ModelConfig = field(default_factory=Ilov3SplatModelConfig)


class Ilov3SplatPipeline(VanillaPipeline):
    def __init__(
        self,
        config: Ilov3SplatPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode

        self.datamanager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )

        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            grad_scaler=grad_scaler,
            datamanager=self.datamanager,
            image_encoder=self.datamanager.image_encoder,
            seed_points=seed_pts,
            cluster_experiment_root=_cluster_artifact_root(config),
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                torch.nn.Module,
                DDP(self._model, device_ids=[local_rank], find_unused_parameters=True),
            )
            dist.barrier(device_ids=[local_rank])
