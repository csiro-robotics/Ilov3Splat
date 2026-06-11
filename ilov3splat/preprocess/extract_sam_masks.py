"""Extract per-image SAM masks as NPZ."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import cv2
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from ilov3splat.preprocess.sam_levels_model import SamLevelsAutomaticMaskGenerator
except ImportError:  # optional dependency: pip install -e ".[preprocess-sam]"
    SamAutomaticMaskGenerator = None  # type: ignore[misc, assignment]
    sam_model_registry = None  # type: ignore[misc, assignment]
    SamLevelsAutomaticMaskGenerator = None  # type: ignore[misc, assignment]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def process_mask_level(
    image: npt.NDArray,
    masks: list[dict[str, Any]],
    binary_mask: bool = False,
    sort_key: str | None = None,
    pre_kernel_size: tuple[int, int] | None = None,
    post_kernel_size: tuple[int, int] | None = None,
):
    if sort_key is not None:
        assert sort_key in {
            "predicted_iou",
            "stability_score",
            "score",
            "area",
            "area+score",
        }
        if sort_key == "score":
            masks = sorted(
                masks, key=lambda x: x["predicted_iou"] * x["stability_score"]
            )
        elif sort_key == "area":
            masks = sorted(
                masks, key=lambda x: x["area"], reverse=True
            )
        elif sort_key == "area+score":
            masks = sorted(
                masks, key=lambda x: x["predicted_iou"] * x["stability_score"]
            )
            masks = sorted(
                masks, key=lambda x: x["area"], reverse=True
            )
        else:
            masks = sorted(masks, key=lambda x: x[sort_key])
    if binary_mask:
        if pre_kernel_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, pre_kernel_size)
            for m in masks:
                mask: npt.NDArray = m["segmentation"]
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = mask.astype(bool)
                m["segmentation"] = mask
        return np.stack([m["segmentation"] for m in masks])
    else:
        fullmask = -1 * np.ones(image.shape[:2], dtype=np.int32)

        kernel = None
        if pre_kernel_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, pre_kernel_size)

        for i, m in enumerate(masks):
            mask: npt.NDArray = m["segmentation"]
            if kernel is not None:
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = mask.astype(bool)
            fullmask[mask] = i

        if post_kernel_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, post_kernel_size)
            for mask_id in np.unique(fullmask):
                mask = fullmask == mask_id
                modified_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                modified_mask = cv2.dilate(modified_mask, kernel, iterations=1)
                modified_mask = np.asarray(modified_mask, dtype=bool)
                fullmask[mask ^ modified_mask] = -1

        idx = np.unique(fullmask)
        idx = idx[idx >= 0]
        for i, j in enumerate(idx):
            fullmask[fullmask == j] = i

        return fullmask


def compute_masks(
    mask_generator: Any,
    image_path: Path,
    binary_mask: bool = False,
    sort_key: str | None = None,
    pre_kernel_size: tuple[int, int] | None = None,
    post_kernel_size: tuple[int, int] | None = None,
):
    image = iio.imread(image_path)
    if SamLevelsAutomaticMaskGenerator is not None and isinstance(
        mask_generator, SamLevelsAutomaticMaskGenerator
    ):
        mask_levels = mask_generator.generate(image)
        levels = ["default", "subpart", "part", "whole"]
        assert len(mask_levels) == len(levels), f"Expected 4 levels of masks {levels}"
        output: dict[str, npt.NDArray] = {}
        for masks, level in zip(mask_levels, levels):
            output[level] = process_mask_level(
                image,
                masks,
                binary_mask,
                sort_key,
                pre_kernel_size,
                post_kernel_size,
            )
        return output
    else:
        masks = mask_generator.generate(image)
        return {
            "default": process_mask_level(
                image,
                masks,
                binary_mask,
                sort_key,
                pre_kernel_size,
                post_kernel_size,
            )
        }


def run_extract_sam_masks(
    scene_dir: Path,
    output_dir: Path,
    image_subdir: str = "images",
    levels: bool = True,
    binary_mask: bool = False,
    sort_key: str | None = "score",
    pre_kernel_size: tuple[int, int] | None = None,
    post_kernel_size: tuple[int, int] | None = None,
    min_mask_region_area: int = 0,
    compress: bool = False,
    sam_checkpoint: Path | None = None,
    sam_model_type: str = "vit_h",
    device: str | None = None,
):
    if not scene_dir.exists():
        print(f"Scene {scene_dir} does not exist")
        return
    if not scene_dir.is_dir():
        print(f"Scene {scene_dir} is not a directory")
        return

    image_dir = scene_dir / image_subdir
    if not image_dir.exists():
        print(f"Image directory {image_dir} does not exist")
        return
    if not image_dir.is_dir():
        print(f"Image directory {image_dir} is not a directory")
        return

    ckpt = sam_checkpoint
    if ckpt is None:
        ckpt = Path(os.environ.get("SAM_CHECKPOINT", "ckpts/sam_vit_h_4b8939.pth"))
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {ckpt}. "
            "Pass --sam-checkpoint or set SAM_CHECKPOINT (see README)."
        )

    if sam_model_registry is None or SamAutomaticMaskGenerator is None:
        raise ImportError(
            "segment_anything is not installed. Install with: pip install -e \".[preprocess-sam]\""
        )

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAM model ({sam_model_type}) on {dev}")
    registry = cast(dict[str, Any], sam_model_registry)
    if sam_model_type not in registry:
        raise ValueError(f"Unknown SAM model type {sam_model_type!r}; choose from {list(registry)}")
    sam = registry[sam_model_type](checkpoint=str(ckpt))
    sam.to(torch.device(dev))

    if levels:
        mask_generator = SamLevelsAutomaticMaskGenerator(
            sam, min_mask_region_area=min_mask_region_area
        )
    else:
        mask_generator = SamAutomaticMaskGenerator(
            sam, min_mask_region_area=min_mask_region_area
        )

    output_dir.mkdir(exist_ok=True, parents=True)
    config = OmegaConf.create(
        {
            "scene_dir": str(scene_dir.resolve()),
            "image_subdir": image_subdir,
            "levels": levels,
            "binary_mask": binary_mask,
            "sort_key": sort_key,
            "pre_kernel_size": pre_kernel_size,
            "post_kernel_size": post_kernel_size,
            "min_mask_region_area": min_mask_region_area,
            "compress": compress,
            "sam_checkpoint": str(ckpt.resolve()),
            "sam_model_type": sam_model_type,
            "device": dev,
        }
    )
    OmegaConf.save(config, output_dir / "config.yaml", resolve=True)

    imgs = sorted(
        [
            x
            for x in image_dir.iterdir()
            if x.is_file() and x.suffix.lower() in IMAGE_SUFFIXES
        ]
    )
    for image_path in tqdm(imgs, total=len(imgs)):
        mask_path = output_dir / f"{image_path.stem}.npz"
        if mask_path.exists():
            continue
        mask_levels = compute_masks(
            mask_generator,
            image_path,
            binary_mask,
            sort_key,
            pre_kernel_size,
            post_kernel_size,
        )
        for level, masks in mask_levels.items():
            assert not binary_mask or masks.ndim == 3, (
                f"Expected masks of shape (N, H, W), got {masks.ndim} for level {level}"
            )

        if compress:
            np.savez_compressed(mask_path, **mask_levels)
        else:
            np.savez(mask_path, **mask_levels)


def main() -> None:
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser(description="Extract SAM masks for a scene")
    parser.add_argument(
        "source_dir", type=str, help="Scene directory to extract SAM masks for"
    )
    parser.add_argument(
        "--output-subdir", type=str, default="sam", help="Output subdirectory for masks"
    )
    parser.add_argument(
        "--img-subdir",
        type=str,
        default="images",
        help="Subdirectory of scene directory containing images",
    )
    parser.add_argument(
        "--levels",
        action=BooleanOptionalAction,
        default=True,
        help="Extract all SAM levels (default, subpart, part, whole). Use --no-levels for default-only.",
    )
    parser.add_argument(
        "--binary-mask",
        action="store_true",
        help="Output binary masks instead of merged masks",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="score",
        choices=["predicted_iou", "stability_score", "score", "area", "area+score"],
        help="Sort masks by the given key (default: score)",
    )
    parser.add_argument(
        "--pre-kernel-size",
        type=int,
        help="Size of morphological kernel for preprocessing",
    )
    parser.add_argument(
        "--post-kernel-size",
        type=int,
        help="Size of morphological kernel for postprocessing",
    )
    parser.add_argument(
        "--min-mask-region-area",
        type=int,
        default=0,
        help="Minimum area of mask region to consider",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress the masks",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="Path to SAM .pth weights (default: $SAM_CHECKPOINT or ckpts/sam_vit_h_4b8939.pth)",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="vit_h",
        help="SAM backbone key for sam_model_registry (e.g. vit_h, vit_l, vit_b)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )

    args = parser.parse_args()

    source_path = Path(args.source_dir)
    output_path: Path = source_path / args.output_subdir

    run_extract_sam_masks(
        source_path,
        output_path,
        args.img_subdir,
        args.levels,
        args.binary_mask,
        args.sort,
        (args.pre_kernel_size, args.pre_kernel_size) if args.pre_kernel_size else None,
        (args.post_kernel_size, args.post_kernel_size)
        if args.post_kernel_size
        else None,
        args.min_mask_region_area,
        args.compress,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
