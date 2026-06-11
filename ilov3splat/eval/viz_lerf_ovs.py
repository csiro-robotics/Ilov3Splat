"""Standalone visualization for hybrid cluster language masks (LERF-OVS-style inst-3D)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch
from omegaconf import OmegaConf

from ilov3splat.eval.query_core import (
    compute_masks_hybrid_cluster_3d,
    load_cluster_labels,
    render_hybrid_cluster_rgb,
)
from ilov3splat.eval.eval_lerf_ovs import save_lerf_ovs_prediction_visualizations
from ilov3splat.eval.utils import (
    generate_eval_output_name,
    get_split_cameras,
    load_ilov3splat_pipeline,
    render_frames,
)

SCENE_TEXTS: dict[str, list[str]] = {
    "waldo_kitchen": ["Stainless steel pots", "dark cup", "refrigerator"],
    "ramen": ["nori", "sake cup", "kamaboko"],
    "figurines": ["jake", "pirate hat", "pikachu"],
    "teatime": ["sheep", "yellow pouf", "stuffed bear"],
}


def _read_nonempty_lines(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            out.append(s)
    return out


def resolve_cameras(
    train_cams: list[torch.Tensor],
    train_names: list[str],
    test_cams: list[torch.Tensor],
    test_names: list[str],
    split: str | None,
    explicit_frames: list[str] | None,
) -> tuple[list[torch.Tensor], list[str]]:
    if split is not None:
        if split == "train":
            return train_cams, train_names
        return test_cams, test_names
    assert explicit_frames is not None and len(explicit_frames) > 0
    by_name = {n: c for n, c in zip(train_names + test_names, train_cams + test_cams)}
    missing = [n for n in explicit_frames if n not in by_name]
    if missing:
        raise ValueError(f"Unknown camera name(s): {missing}")
    return [by_name[n] for n in explicit_frames], explicit_frames


@torch.no_grad()
def save_rgb_renders(
    cameras: list[torch.Tensor],
    model: torch.nn.Module,
    frame_stems: list[str],
    output_dir: Path,
) -> bool:
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    rendered, _ = render_frames(cameras, model, override_color=None, override_opacity=None)
    for i, stem in enumerate(frame_stems):
        img = rendered[i].numpy()
        img_hwc = np.transpose(np.clip(img, 0.0, 1.0), (1, 2, 0))
        bgr = cv2.cvtColor((img_hwc * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_dir / f"{stem}.png"), bgr)
    return True


@torch.no_grad()
def run_viz(
    model_path: Path,
    query_prompts: list[str],
    cameras: list[torch.Tensor],
    frame_stems: list[str],
    cluster_threshold: float,
    mask_threshold: float,
    render_mode: str | None,
    output_dir: Path,
    rgb_only: bool = False,
    output_rgb: bool = False,
) -> bool:
    _, pipeline = load_ilov3splat_pipeline(model_path, test_mode="test")
    model = pipeline.model.module if hasattr(pipeline.model, "module") else pipeline.model
    if rgb_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        return save_rgb_renders(cameras, model, frame_stems, output_dir)

    cluster_labels = load_cluster_labels(model_path, model.means.shape[0], model.device)
    pred_masks: npt.NDArray = compute_masks_hybrid_cluster_3d(
        model,
        cameras,
        cluster_labels,
        query_prompts,
        cluster_threshold,
        mask_threshold,
        render_mode,
    ).numpy()
    cluster_rgb = render_hybrid_cluster_rgb(
        model,
        cameras,
        cluster_labels,
        query_prompts,
        cluster_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_lerf_ovs_prediction_visualizations(
        pred_masks.astype(np.bool_),
        cluster_rgb,
        frame_stems,
        query_prompts,
        output_dir,
    )
    if output_rgb:
        save_rgb_renders(cameras, model, frame_stems, output_dir)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Ilov3Splat hybrid cluster language masks (LERF-OVS style)."
    )
    parser.add_argument("model_dir", type=Path)
    cam = parser.add_mutually_exclusive_group(required=True)
    cam.add_argument("--split", choices=["train", "test"])
    cam.add_argument("--frames", nargs="+", metavar="STEM")
    cam.add_argument("--frame-list", type=Path, metavar="PATH")
    parser.add_argument("--queries", nargs="+", default=None, metavar="TEXT")
    parser.add_argument("--queries-file", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.85,
        help="Relative cluster relevancy threshold (default matches lang_cluster_threshold).",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.3)
    parser.add_argument("--render-mode", type=str, default=None, choices=["alpha", "isolated"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rgb-only", action="store_true")
    parser.add_argument("--rgb", action="store_true")
    args = parser.parse_args()

    model_path = args.model_dir.expanduser().resolve()
    _, pipeline = load_ilov3splat_pipeline(model_path, test_mode="test")
    train_cams, train_names = get_split_cameras(pipeline, "train")
    test_cams, test_names = get_split_cameras(pipeline, "eval")

    scene_stem = Path(getattr(pipeline.datamanager.config.dataparser, "data")).stem
    if args.queries is not None and args.queries_file is not None:
        raise SystemExit("Use only one of --queries and --queries-file")
    if args.queries_file is not None:
        query_prompts = _read_nonempty_lines(args.queries_file.resolve())
    elif args.queries is not None:
        query_prompts = list(args.queries)
    elif args.rgb_only:
        query_prompts = []
    elif scene_stem in SCENE_TEXTS:
        query_prompts = SCENE_TEXTS[scene_stem]
    else:
        raise SystemExit("No query list provided and no scene preset found.")

    if args.frame_list is not None:
        split, frames_arg = None, _read_nonempty_lines(args.frame_list.resolve())
    elif args.frames is not None:
        split, frames_arg = None, list(args.frames)
    else:
        split, frames_arg = args.split, None
    cameras, frame_stems = resolve_cameras(
        train_cams, train_names, test_cams, test_names, split, frames_arg
    )
    out = args.output_dir.resolve() if args.output_dir is not None else model_path / "viz_lerf_ovs" / generate_eval_output_name()
    run_viz(
        model_path,
        query_prompts,
        cameras,
        frame_stems,
        args.cluster_threshold,
        args.mask_threshold,
        args.render_mode,
        out,
        rgb_only=args.rgb_only,
        output_rgb=args.rgb and not args.rgb_only,
    )

    cfg = OmegaConf.create(
        {
            "model_dir": str(model_path),
            "scene_stem": scene_stem,
            "split": split,
            "frames": frames_arg,
            "queries": query_prompts,
            "rgb_only": args.rgb_only,
            "output_rgb": args.rgb,
            "query_mode": "hybrid_cluster",
            "lang_model": "clip",
            "cluster_threshold": args.cluster_threshold,
            "mask_threshold": args.mask_threshold,
            "render_mode": args.render_mode,
            "camera_names": frame_stems,
        }
    )
    OmegaConf.save(cfg, out / "viz_config.yaml")
    print(f"Wrote visualization outputs under {out}")


if __name__ == "__main__":
    main()
