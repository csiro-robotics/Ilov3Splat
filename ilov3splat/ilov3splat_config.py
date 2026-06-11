from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from ilov3splat.ilov3splat_datamanager import Ilov3SplatDataManagerConfig
from ilov3splat.ilov3splat_model import Ilov3SplatModelConfig
from ilov3splat.ilov3splat_pipeline import Ilov3SplatPipelineConfig
from ilov3splat.utils.openclip_encoder import OpenCLIPNetworkConfig


ilov3splat_method = MethodSpecification(
    config=TrainerConfig(
        method_name="ilov3splat",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=1000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=Ilov3SplatPipelineConfig(
            datamanager=Ilov3SplatDataManagerConfig(
                instance_mask_npz_key="whole",
                instance_warmup_steps=10000,
                lang_step=10000,
                dino_step=10000,
                dino_pca_dim=16,
                clip_downscale_factor=2,
                network=OpenCLIPNetworkConfig(
                    clip_model_type="ViT-B-16",
                    clip_model_pretrained="laion2b_s34b_b88k",
                    clip_n_dims=512,
                    device="cuda:0",
                ),
            ),
            model=Ilov3SplatModelConfig(
                output_depth_during_training=True,
                instance_decode_mode="identity",
                instance_feature_dim=8,
                instance_embedding_dim=8,
                feature_downscale_factor=2,
                inst2d_lambda=1.0,
                inst2d_sample_size=-1,
                inst2d_gamma=1.0,
                inst2d_weights=(1.0, 1.0),
                inst2d_normalize=False,
                inst2d_from_iter=10000,
                inst2d_interval=1,
                inst2d_ramp_steps=2000,
                instance_warmup_steps=10000,
                instance_lr_reset_after_warmup=False,
                var_lambda=0.5,
                lang_loss_weight=0.1,
                dino_feat_dim=16,
                dino_dim=16,
                dino_rescale_factor=5,
                dino_loss_weight=0.1,
                lang_step=10000,
                cluster_on_train_end=False,
                cluster_output_dir="clustering",
                cluster_overwrite_output=False,
                cluster_min_cluster_size=50,
                cluster_min_samples=None,
                cluster_selection_epsilon=0.0,
                cluster_with_position=0.0,
                cluster_with_color=0.0,
                cluster_normalize_instance_feats=False,
                cluster_normalize_position=False,
                cluster_normalize_color=False,
                cluster_use_dbscan_denoising=False,
                cluster_dbscan_min_samples=5,
                cluster_dbscan_eps=0.05,
                cluster_dbscan_min_cluster_size=0,
                cluster_prefer_cuml=True,
                cluster_assign_noise=False,
                cluster_assign_noise_alpha=0.5,
                cluster_assign_noise_k=5,
                cluster_assign_noise_max_dist=None,
                cluster_viewer_enable=True,
                cluster_viewer_hide_noise=True,
                lang_viewer_enable=True,
                lang_use_hybrid_cluster_query=True,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=30000,
                ),
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
            "instance_feats": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=15000,
                ),
            },
            "instance_decoder": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=15000,
                ),
            },
            "lang": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=15000,
                ),
            },
            "dino_feats": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=15000,
                ),
            },
            "dino_decoder": {
                "optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-3,
                    max_steps=15000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Ilov3Splat: Instance-lavel open-vocabulary Gaussian Splatting plugin for Nerfstudio.",
)
