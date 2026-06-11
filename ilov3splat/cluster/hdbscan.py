from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from nerfstudio.models.splatfacto import SH2RGB

try:  # pragma: no cover - optional dependency
    from cuml.cluster import DBSCAN as CuDBSCAN
    from cuml.cluster import HDBSCAN as CuHDBSCAN
except ImportError:  # pragma: no cover - optional dependency
    CuDBSCAN = None
    CuHDBSCAN = None

try:  # pragma: no cover - optional dependency
    import hdbscan as cpu_hdbscan
except ImportError:  # pragma: no cover - optional dependency
    cpu_hdbscan = None

from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.neighbors import NearestNeighbors


@dataclass
class ClusterConfig:
    min_cluster_size: int = 20
    min_samples: Optional[int] = None
    cluster_selection_epsilon: float = 0.0
    with_position: float = 0.0
    with_color: float = 0.0
    normalize_instance_feats: bool = False
    normalize_position: bool = False
    normalize_color: bool = False
    use_dbscan_denoising: bool = False
    dbscan_min_samples: int = 5
    dbscan_eps: float = 0.05
    dbscan_min_cluster_size: int = 0
    prefer_cuml: bool = True
    assign_noise: bool = True
    """Assign HDBSCAN noise (-1) via kNN to nearest seed clusters."""
    assign_noise_alpha: float = 0.5
    """Weight on feature distance; spatial weight is ``1 - assign_noise_alpha``."""
    assign_noise_k: int = 5
    """Number of labeled seed neighbors for majority-vote assignment."""
    assign_noise_max_dist: Optional[float] = None
    """If set, leave as noise when blended distance to the k-th neighbor exceeds this."""


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").float().numpy()


def _normalize_block(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps)


def _build_cluster_features(
    instance_feats: torch.Tensor,
    means: Optional[torch.Tensor] = None,
    features_dc: Optional[torch.Tensor] = None,
    *,
    with_position: float = 0.0,
    with_color: float = 0.0,
    normalize_instance_feats: bool = False,
    normalize_position: bool = False,
    normalize_color: bool = False,
) -> np.ndarray:
    blocks: list[np.ndarray] = []

    inst = _to_numpy(instance_feats)
    if normalize_instance_feats:
        inst = _normalize_block(inst)
    blocks.append(inst)

    if with_color > 0.0 and features_dc is not None:
        base_color = _to_numpy(SH2RGB(features_dc)) * float(with_color)
        if normalize_color:
            base_color = _normalize_block(base_color)
        blocks.insert(0, base_color)

    if with_position > 0.0 and means is not None:
        xyz = _to_numpy(means) * float(with_position)
        if normalize_position:
            xyz = _normalize_block(xyz)
        blocks.insert(0, xyz)

    return np.concatenate(blocks, axis=1)


def _run_hdbscan(features: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    if cfg.prefer_cuml and CuHDBSCAN is not None:
        try:
            clusterer = CuHDBSCAN(
                min_cluster_size=cfg.min_cluster_size,
                min_samples=cfg.min_samples,
                cluster_selection_epsilon=cfg.cluster_selection_epsilon,
            )
            labels = clusterer.fit_predict(features)
            return np.asarray(labels, dtype=np.int32)
        except Exception as exc:  # pragma: no cover - backend dependent
            msg = str(exc)
            precision_related = (
                "Number of edges found by MST is invalid" in msg
                or "Try increasing precision of weights" in msg
                or "raft::logic_error" in msg
            )
            if not precision_related or cpu_hdbscan is None:
                raise
            warnings.warn(
                "cuML HDBSCAN failed with RAFT precision error; falling back to CPU hdbscan.",
                RuntimeWarning,
                stacklevel=2,
            )
            clusterer_cpu = cpu_hdbscan.HDBSCAN(
                min_cluster_size=cfg.min_cluster_size,
                min_samples=cfg.min_samples,
                cluster_selection_epsilon=cfg.cluster_selection_epsilon,
            )
            return np.asarray(
                clusterer_cpu.fit_predict(features.astype(np.float64, copy=False)),
                dtype=np.int32,
            )

    if cpu_hdbscan is None:
        raise ImportError(
            "HDBSCAN dependency missing. Install RAPIDS cuML or `hdbscan`."
        )
    clusterer = cpu_hdbscan.HDBSCAN(
        min_cluster_size=cfg.min_cluster_size,
        min_samples=cfg.min_samples,
        cluster_selection_epsilon=cfg.cluster_selection_epsilon,
    )
    return np.asarray(clusterer.fit_predict(features), dtype=np.int32)


def _dbscan_denoising(
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    min_samples: int,
    eps: float,
    min_cluster_size: int,
    prefer_cuml: bool = True,
) -> np.ndarray:
    labels_out = labels.copy()
    unique_labels = np.unique(labels_out)
    max_label = int(np.max(labels_out)) + 1 if labels_out.size > 0 else 0

    dbscan_cls: Any
    if prefer_cuml and CuDBSCAN is not None:
        dbscan_cls = CuDBSCAN
    else:
        dbscan_cls = SkDBSCAN

    for label in unique_labels:
        if int(label) == -1:
            continue
        mask = labels_out == label
        cluster_xyz = xyz[mask]
        new_labels = np.asarray(
            dbscan_cls(min_samples=min_samples, eps=eps).fit_predict(cluster_xyz),
            dtype=np.int32,
        )
        labels_out[mask] = -1

        for new_label in np.unique(new_labels):
            if int(new_label) == -1:
                continue
            if np.sum(new_labels == new_label) < min_cluster_size:
                continue
            new_mask = mask.copy()
            new_mask[mask] = new_labels == new_label
            labels_out[new_mask] = int(new_label) + max_label

        if new_labels.size > 0:
            max_label = int(np.max(new_labels)) + max_label + 1

    return _remap_non_noise_labels(labels_out)


def _assign_noise_knn(
    labels: np.ndarray,
    instance_feats: np.ndarray,
    xyz: np.ndarray,
    *,
    alpha: float = 0.5,
    k: int = 5,
    max_dist: Optional[float] = None,
) -> np.ndarray:
    labels_out = np.asarray(labels, dtype=np.int32).copy()
    noise_mask = labels_out == -1
    if not np.any(noise_mask):
        return labels_out

    seed_mask = labels_out >= 0
    n_seeds = int(np.sum(seed_mask))
    if n_seeds == 0:
        return labels_out

    feats = np.asarray(instance_feats, dtype=np.float64)
    pos = np.asarray(xyz, dtype=np.float64)
    if feats.shape[0] != labels_out.shape[0] or pos.shape[0] != labels_out.shape[0]:
        raise ValueError("labels, instance_feats, and xyz must have the same length")

    feats = _normalize_block(feats)
    pos = _normalize_block(pos)

    alpha = float(np.clip(alpha, 0.0, 1.0))
    k_eff = max(1, min(int(k), n_seeds))

    seed_feats = feats[seed_mask]
    seed_pos = pos[seed_mask]
    seed_labels = labels_out[seed_mask]

    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nn.fit(np.concatenate([alpha * seed_feats, (1.0 - alpha) * seed_pos], axis=1))

    query = np.concatenate([alpha * feats[noise_mask], (1.0 - alpha) * pos[noise_mask]], axis=1)
    dists, indices = nn.kneighbors(query)

    if k_eff == 1:
        assigned = seed_labels[indices[:, 0]]
        if max_dist is not None:
            assigned = np.where(dists[:, 0] <= float(max_dist), assigned, -1)
    else:
        assigned = np.empty(indices.shape[0], dtype=np.int32)
        for i in range(indices.shape[0]):
            neighbor_labels = seed_labels[indices[i]]
            if max_dist is not None and dists[i, -1] > float(max_dist):
                assigned[i] = -1
                continue
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            assigned[i] = vals[int(np.argmax(counts))]

    labels_out[noise_mask] = assigned
    return labels_out


def _remap_non_noise_labels(labels: np.ndarray) -> np.ndarray:
    labels_out = np.asarray(labels, dtype=np.int32).copy()
    non_noise = np.unique(labels_out[labels_out >= 0])
    for new_id, old_id in enumerate(non_noise):
        labels_out[labels_out == old_id] = new_id
    return labels_out


def _compute_cluster_stats(labels: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int32)
    total = int(labels.size)
    noise_count = int(np.sum(labels == -1))
    non_noise = labels[labels >= 0]
    unique_non_noise, counts = np.unique(non_noise, return_counts=True)
    order = np.argsort(counts)[::-1] if counts.size > 0 else np.array([], dtype=int)

    return {
        "num_gaussians": total,
        "num_clusters": int(unique_non_noise.size),
        "noise_count": noise_count,
        "noise_ratio": float(noise_count / total) if total > 0 else 0.0,
        "largest_cluster_ids": unique_non_noise[order][:20].tolist() if counts.size > 0 else [],
        "largest_cluster_sizes": counts[order][:20].tolist() if counts.size > 0 else [],
    }


def run_hdbscan_clustering(
    instance_feats: torch.Tensor,
    means: torch.Tensor,
    features_dc: torch.Tensor,
    cfg: ClusterConfig,
) -> tuple[np.ndarray, Dict[str, Any]]:
    features = _build_cluster_features(
        instance_feats,
        means=means,
        features_dc=features_dc,
        with_position=cfg.with_position,
        with_color=cfg.with_color,
        normalize_instance_feats=cfg.normalize_instance_feats,
        normalize_position=cfg.normalize_position,
        normalize_color=cfg.normalize_color,
    )
    if not np.isfinite(features).all():
        raise ValueError(
            "Non-finite values detected in clustering features (NaN/Inf). "
            "Check training stability or feature normalization settings."
        )

    labels = _remap_non_noise_labels(_run_hdbscan(features, cfg))
    hdbscan_stats = _compute_cluster_stats(labels)

    assign_stats: Optional[Dict[str, Any]] = None
    if cfg.assign_noise:
        labels = _assign_noise_knn(
            labels,
            _to_numpy(instance_feats),
            _to_numpy(means),
            alpha=cfg.assign_noise_alpha,
            k=cfg.assign_noise_k,
            max_dist=cfg.assign_noise_max_dist,
        )
        assign_stats = _compute_cluster_stats(labels)

    denoise_stats: Optional[Dict[str, Any]] = None
    if cfg.use_dbscan_denoising:
        labels = _dbscan_denoising(
            xyz=_to_numpy(means),
            labels=labels,
            min_samples=cfg.dbscan_min_samples,
            eps=cfg.dbscan_eps,
            min_cluster_size=cfg.dbscan_min_cluster_size,
            prefer_cuml=cfg.prefer_cuml,
        )
        denoise_stats = _compute_cluster_stats(labels)

    stats = _compute_cluster_stats(labels)
    stats["hdbscan_seed"] = hdbscan_stats
    if assign_stats is not None:
        stats["post_assign"] = assign_stats
    if denoise_stats is not None:
        stats["post_denoise"] = denoise_stats
    return labels, stats


def save_cluster_artifacts(
    output_dir: Path,
    labels: np.ndarray,
    stats: Dict[str, Any],
    cfg: ClusterConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, dtype=np.int32)
    np.save(output_dir / "labels.npy", labels)
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    cfg_dict = asdict(cfg)
    try:  # pragma: no cover - optional dependency
        import yaml

        with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)
    except ImportError:
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
            for key, value in cfg_dict.items():
                f.write(f"{key}: {value}\n")
