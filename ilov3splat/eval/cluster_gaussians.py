from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch

from ilov3splat.cluster import ClusterConfig, run_hdbscan_clustering, save_cluster_artifacts


def _resolve_checkpoint(run_dir: Path, checkpoint: Optional[Path]) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    ckpt_dir = run_dir / "nerfstudio_models"
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Could not find nerfstudio checkpoint dir at: {ckpt_dir}. "
            "Pass --checkpoint explicitly."
        )
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in: {ckpt_dir}")
    return ckpts[-1]


def _extract_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    for key in ("pipeline", "state_dict", "model"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return value
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise KeyError(
        "Unsupported checkpoint format. Could not find a tensor state dict in keys: "
        f"{list(ckpt.keys())[:10]}"
    )


def _find_tensor(state_dict: Dict[str, torch.Tensor], suffix: str) -> torch.Tensor:
    candidates = [k for k in state_dict.keys() if k.endswith(suffix)]
    if not candidates:
        raise KeyError(f"Missing tensor ending with `{suffix}` in checkpoint state dict")
    return state_dict[candidates[0]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HDBSCAN clustering on Ilov3Splat Gaussian instance features."
    )
    parser.add_argument("run_dir", type=Path, help="Ilov3Splat run directory (contains config.yml).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit .ckpt path. Defaults to latest in run_dir/nerfstudio_models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for labels/stats/config. Defaults to run_dir/clustering.",
    )
    parser.add_argument("--with-position", type=float, default=0.0)
    parser.add_argument("--with-color", type=float, default=0.0)
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--normalize-instance-feats", action="store_true")
    parser.add_argument("--normalize-position", action="store_true")
    parser.add_argument("--normalize-color", action="store_true")
    parser.add_argument("--dbscan-denoising", action="store_true")
    parser.add_argument("--dbscan-min-samples", type=int, default=5)
    parser.add_argument("--dbscan-eps", type=float, default=0.05)
    parser.add_argument("--dbscan-min-cluster-size", type=int, default=0)
    parser.add_argument(
        "--no-cuml",
        action="store_true",
        help="Disable cuML preference and use CPU implementations when available.",
    )
    parser.add_argument(
        "--no-assign-noise",
        action="store_true",
        help="Disable kNN reassignment of HDBSCAN noise points.",
    )
    parser.add_argument("--assign-noise-alpha", type=float, default=0.5)
    parser.add_argument("--assign-noise-k", type=int, default=5)
    parser.add_argument(
        "--assign-noise-max-dist",
        type=float,
        default=None,
        help="Max blended distance for noise assignment; omit to assign all noise.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_path = _resolve_checkpoint(run_dir, args.checkpoint)
    output_dir = (args.output_dir or (run_dir / "clustering")).expanduser().resolve()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(ckpt)

    instance_feats = _find_tensor(state_dict, "gauss_params.instance_feats")
    means = _find_tensor(state_dict, "gauss_params.means")
    features_dc = _find_tensor(state_dict, "gauss_params.features_dc")

    cfg = ClusterConfig(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.epsilon,
        with_position=args.with_position,
        with_color=args.with_color,
        normalize_instance_feats=args.normalize_instance_feats,
        normalize_position=args.normalize_position,
        normalize_color=args.normalize_color,
        use_dbscan_denoising=args.dbscan_denoising,
        dbscan_min_samples=args.dbscan_min_samples,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_cluster_size=args.dbscan_min_cluster_size,
        prefer_cuml=not args.no_cuml,
        assign_noise=not args.no_assign_noise,
        assign_noise_alpha=args.assign_noise_alpha,
        assign_noise_k=args.assign_noise_k,
        assign_noise_max_dist=args.assign_noise_max_dist,
    )

    labels, stats = run_hdbscan_clustering(
        instance_feats=instance_feats,
        means=means,
        features_dc=features_dc,
        cfg=cfg,
    )
    stats["checkpoint"] = str(checkpoint_path)
    stats["run_dir"] = str(run_dir)
    save_cluster_artifacts(output_dir, labels, stats, cfg)

    print(f"Saved clustering artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
