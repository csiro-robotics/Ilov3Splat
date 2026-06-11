from __future__ import annotations

import json
import os
import re
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import tabulate
import torch
from omegaconf import OmegaConf

from ilov3splat.eval.metrics import calculate_iou, calculate_loc
from ilov3splat.eval.query_core import (
    compute_masks_hybrid_cluster_3d,
    load_cluster_labels,
    render_hybrid_cluster_rgb,
)
from ilov3splat.eval.utils import (
    generate_eval_output_name,
    get_model_paths,
    get_split_cameras,
    load_ilov3splat_pipeline,
)

LERF_OVS_LABEL_PATH = (
    Path(os.environ["LERF_OVS_LABEL_PATH"])
    if "LERF_OVS_LABEL_PATH" in os.environ
    else None
)


@dataclass
class ObjectAnnotation:
    label: str
    bboxes: npt.NDArray[np.float32]
    masks: npt.NDArray[np.bool_]


@dataclass
class FrameAnnotation:
    frame: Path
    annotations: dict[str, ObjectAnnotation]


def polygon_to_mask(img_shape: tuple[int, int], points_list: list[list[float]]) -> npt.NDArray[np.bool_]:
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)  # type: ignore[arg-type]
    return mask.astype(bool)


def get_annotations(label_scene_path: Path) -> tuple[list[FrameAnnotation], list[str]]:
    annotations: list[FrameAnnotation] = []
    frames = sorted(label_scene_path.glob("*.jpg"))
    for frame in frames:
        anno: dict[str, dict[str, list[npt.NDArray[Any]]]] = {}
        with open(frame.with_suffix(".json"), "r", encoding="utf-8") as f:
            json_data = json.load(f)
        h = int(json_data["info"]["height"])
        w = int(json_data["info"]["width"])
        for obj in json_data["objects"]:
            label = obj["category"]
            bbox = np.asarray(obj["bbox"], dtype=np.float32)
            mask = polygon_to_mask((h, w), obj["segmentation"])
            anno.setdefault(label, {"bboxes": [], "masks": []})
            anno[label]["bboxes"].append(bbox)
            anno[label]["masks"].append(mask)
        annotation = {
            k: ObjectAnnotation(k, np.stack(v["bboxes"]).astype(np.float32), np.stack(v["masks"]).astype(bool))
            for k, v in anno.items()
        }
        annotations.append(FrameAnnotation(frame, annotation))

    scene_labels = sorted(set([k for f in annotations for k in f.annotations.keys()]))
    return annotations, scene_labels


def _resize_mask_nearest(
    mask: npt.NDArray[np.bool_], out_hw: tuple[int, int]
) -> npt.NDArray[np.bool_]:
    oh, ow = out_hw
    if mask.shape == (oh, ow):
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def lerf_ovs_metrics_table_str(metrics: dict[str, Any]) -> str:
    table = [("Category", "IoU", "Acc", "Loc")]
    table.extend(
        [
            (
                cat_id,
                f"{metrics['classes']['average_iou'][cat_id] * 100:.2f}",
                f"{metrics['classes']['acc_025'][cat_id] * 100:.2f}",
                f"{metrics['classes']['average_loc'][cat_id] * 100:.2f}",
            )
            for cat_id in metrics["classes"]["average_iou"]
        ]
    )
    table.append((tabulate.SEPARATING_LINE,) * 4)
    table.append(
        (
            "Mean",
            f"{metrics['all']['average_iou'] * 100:.2f}",
            f"{metrics['all']['acc_025'] * 100:.2f}",
            f"{metrics['all']['average_loc'] * 100:.2f}",
        )
    )
    return tabulate.tabulate(table, headers="firstrow", floatfmt=".2f")


def save_lerf_ovs_results(
    results: list[tuple[Path, dict]],
    output_path: Path,
    save_txt: bool = True,
    save_json: bool = True,
) -> None:
    output_json = {"scenes": {}, "all": {}}
    log_lines: list[str] = []
    all_metrics = {"miou": 0.0, "macc": 0.0, "mloc": 0.0}
    table = [("Scene", "mIoU", "mAcc", "mLoc")]

    for model_path, metrics in results:
        log_lines.extend(["\n", f"{model_path.name}\n", lerf_ovs_metrics_table_str(metrics), "\n"])
        scene_name = model_path.name
        all_metrics["miou"] += float(metrics["all"]["average_iou"])
        all_metrics["macc"] += float(metrics["all"]["acc_025"])
        all_metrics["mloc"] += float(metrics["all"]["average_loc"])
        table.append(
            (
                str(scene_name),
                f"{metrics['all']['average_iou'] * 100:.2f}",
                f"{metrics['all']['acc_025'] * 100:.2f}",
                f"{metrics['all']['average_loc'] * 100:.2f}",
            )
        )
        output_json["scenes"][scene_name] = metrics

    if results:
        all_metrics["miou"] /= len(results)
        all_metrics["macc"] /= len(results)
        all_metrics["mloc"] /= len(results)

    table.append((tabulate.SEPARATING_LINE,) * 4)
    table.append(("Mean", f"{all_metrics['miou'] * 100:.2f}", f"{all_metrics['macc'] * 100:.2f}", f"{all_metrics['mloc'] * 100:.2f}"))
    output_json["all"] = all_metrics
    log_lines.extend(["\n\n--- Summary ---\n", tabulate.tabulate(table, headers="firstrow", floatfmt=".2f")])

    if save_txt:
        with open(output_path / "results.txt", "w", encoding="utf-8") as f:
            f.writelines(log_lines)
    if save_json:
        with open(output_path / "results.json", "w", encoding="utf-8") as f:
            json.dump(output_json, f)


def evaluate_lerf_ovs_metrics(
    gt_annotations: list[FrameAnnotation],
    predictions: dict[str, dict[str, npt.NDArray[np.bool_]]],
) -> dict[str, Any]:
    ious: list[float] = []
    locs: list[bool] = []
    class_iou: dict[str, list[float]] = {}
    class_loc: dict[str, list[bool]] = {}
    for frame_anno in gt_annotations:
        frame_preds = predictions[frame_anno.frame.stem]
        for label, anno in frame_anno.annotations.items():
            mask_gt = anno.masks.any(axis=0)
            mask_pred = frame_preds[label]
            if mask_pred.shape != mask_gt.shape:
                mask_pred = _resize_mask_nearest(mask_pred, mask_gt.shape)
            iou = calculate_iou(mask_gt, mask_pred)
            loc = calculate_loc(anno.bboxes, mask_pred)
            class_iou.setdefault(label, []).append(iou)
            class_loc.setdefault(label, []).append(loc)
            ious.append(iou)
            locs.append(loc)

    total_count = max(1, len(ious))
    arr = np.array(ious)
    class_iou_avg = {label: float(np.mean(iou)) for label, iou in class_iou.items()}
    class_acc_025 = {label: float((np.array(iou) > 0.25).sum() / len(iou)) for label, iou in class_iou.items()}
    class_acc_05 = {label: float((np.array(iou) > 0.5).sum() / len(iou)) for label, iou in class_iou.items()}
    class_loc_avg = {label: float(np.mean(loc)) for label, loc in class_loc.items()}
    return {
        "classes": {
            "average_iou": class_iou_avg,
            "acc_025": class_acc_025,
            "acc_05": class_acc_05,
            "average_loc": class_loc_avg,
        },
        "all": {
            "average_iou": float(np.mean(ious) if ious else 0.0),
            "average_loc": float(np.mean(locs) if locs else 0.0),
            "acc_025": float((arr > 0.25).sum() / total_count),
            "acc_05": float((arr > 0.5).sum() / total_count),
        },
    }


def _sanitize_filename_component(name: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    s = s.strip("_") or "query"
    return s[:max_len]


def save_lerf_ovs_prediction_visualizations(
    pred_masks: npt.NDArray[np.bool_],
    cluster_rgb: npt.NDArray[np.floating] | torch.Tensor,
    frame_stems: list[str],
    query_prompts: list[str],
    output_dir: Path,
) -> None:
    mask_dir = output_dir / "mask"
    cluster_dir = output_dir / "cluster"
    mask_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(cluster_rgb, torch.Tensor):
        cluster_rgb = cluster_rgb.detach().cpu().numpy()

    for bi, stem in enumerate(frame_stems):
        for pi, prompt in enumerate(query_prompts):
            safe = _sanitize_filename_component(prompt)
            base = f"{stem}__{safe}"
            m = pred_masks[bi, pi]
            cv2.imwrite(str(mask_dir / f"{base}.png"), (m.astype(np.uint8) * 255))

            rgb = cluster_rgb[bi, pi]
            img = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
            img_hwc = np.transpose(img, (1, 2, 0))
            bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(cluster_dir / f"{base}.png"), bgr)


@torch.no_grad()
def eval_lerf_ovs_hybrid_cluster(
    model_paths: list[Path],
    cluster_threshold: float,
    mask_threshold: float = 0.3,
    render_mode: str | None = None,
    viz_output_rel: Path | None = None,
):
    assert LERF_OVS_LABEL_PATH is not None, "LERF_OVS_LABEL_PATH is not set."

    for model_path in model_paths:
        _, pipeline = load_ilov3splat_pipeline(model_path, test_mode="test")
        model = pipeline.model.module if hasattr(pipeline.model, "module") else pipeline.model
        test_cameras, test_names = get_split_cameras(pipeline, split="eval")
        train_cameras, train_names = get_split_cameras(pipeline, split="train")

        scene_name = Path(getattr(pipeline.datamanager.config.dataparser, "data")).stem
        scene_label_path = LERF_OVS_LABEL_PATH / scene_name
        if not scene_label_path.exists():
            raise RuntimeError(
                f"LERF-OVS label path does not exist for scene '{scene_name}': {scene_label_path}. "
                "Check LERF_OVS_LABEL_PATH and scene naming."
            )
        gt_annotations, query_prompts = get_annotations(scene_label_path)
        if len(gt_annotations) == 0:
            raise RuntimeError(
                f"No annotation frames found in {scene_label_path} (expected *.jpg + *.json)."
            )
        if len(query_prompts) == 0:
            raise RuntimeError(
                f"No query prompts (object categories) found in annotations under {scene_label_path}."
            )

        cluster_labels = load_cluster_labels(model_path, model.means.shape[0], model.device)
        wanted_names = [x.frame.stem for x in gt_annotations]
        by_name_eval = {n: c for n, c in zip(test_names, test_cameras)}
        by_name_all = {n: c for n, c in zip(train_names + test_names, train_cameras + test_cameras)}
        common_names = [n for n in wanted_names if n in by_name_all]
        missing = sorted(set(wanted_names) - set(by_name_all.keys()))
        if missing:
            print(
                f"[LERF-OVS] Warning: {len(missing)} annotation frames missing from train+eval cameras "
                f"for scene '{scene_name}'. Ignoring missing stems (first 10): {missing[:10]}"
            )
        if len(common_names) == 0:
            raise RuntimeError(
                f"No overlapping frames between annotations ({len(wanted_names)}) and available cameras "
                f"(train={len(train_names)}, eval={len(test_names)}) for scene '{scene_name}'."
            )
        n_eval = sum(1 for n in common_names if n in by_name_eval)
        n_train = len(common_names) - n_eval
        if n_train > 0:
            print(
                f"[LERF-OVS] Info: using {n_eval} eval + {n_train} train cameras to match annotation frames "
                f"for scene '{scene_name}'."
            )
        test_image_cameras = [by_name_all[n] for n in common_names]
        gt_annotations = [anno for anno in gt_annotations if anno.frame.stem in by_name_all]
        wanted_names = common_names

        pred_masks = compute_masks_hybrid_cluster_3d(
            model=model,
            cameras=test_image_cameras,
            cluster_labels=cluster_labels,
            query_prompts=query_prompts,
            cluster_threshold=cluster_threshold,
            mask_threshold=mask_threshold,
            render_mode=render_mode,
        ).numpy()

        predictions = {
            cam_name: {
                prompt: obj_mask for prompt, obj_mask in zip(query_prompts, frame_masks)
            }
            for cam_name, frame_masks in zip(wanted_names, pred_masks)
        }
        metrics = evaluate_lerf_ovs_metrics(gt_annotations, predictions)

        if viz_output_rel is not None:
            viz_dir = model_path / viz_output_rel
            viz_dir.mkdir(parents=True, exist_ok=True)
            cluster_rgb = render_hybrid_cluster_rgb(
                model=model,
                cameras=test_image_cameras,
                cluster_labels=cluster_labels,
                query_prompts=query_prompts,
                cluster_threshold=cluster_threshold,
            )
            save_lerf_ovs_prediction_visualizations(
                pred_masks.astype(np.bool_),
                cluster_rgb,
                wanted_names,
                query_prompts,
                viz_dir,
            )

        yield model_path, predictions, metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="LERF-OVS evaluation for Ilov3Splat using hybrid cluster CLIP language query."
    )
    parser.add_argument("model_dir", type=Path, help="Path to model or run directory")
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.85,
        help="Relative cluster relevancy threshold (default matches lang_cluster_threshold).",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.3)
    parser.add_argument("--render-mode", type=str, default=None, choices=["alpha", "isolated"])
    parser.add_argument("--per-scene", action="store_true")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    exp_dir = Path(args.model_dir).expanduser().resolve()
    model_dirs = get_model_paths(exp_dir)
    if not model_dirs:
        raise SystemExit(f"No model directories with config.yml found under {exp_dir}")

    is_eval_dir = (exp_dir / "scenes").exists() or len(model_dirs) > 1
    eval_subdir = f"eval_results/hybrid_cluster{f'_{args.render_mode}' if args.render_mode is not None else ''}"
    output_name = generate_eval_output_name()
    viz_output_rel = None if args.no_viz else Path(eval_subdir) / output_name
    eval_cfg = OmegaConf.create(
        {
            "query_mode": "hybrid_cluster",
            "lang_model": "clip",
            "cluster_threshold": args.cluster_threshold,
            "mask_threshold": args.mask_threshold,
            "render_mode": args.render_mode,
        }
    )

    results: list[tuple[Path, dict[str, typing.Any]]] = []
    for model_path, _, metrics in eval_lerf_ovs_hybrid_cluster(
        model_dirs,
        args.cluster_threshold,
        args.mask_threshold,
        args.render_mode,
        viz_output_rel=viz_output_rel,
    ):
        if is_eval_dir:
            results.append((model_path, metrics))
            if not args.per_scene:
                continue
        output_path = model_path / eval_subdir / output_name
        output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(eval_cfg, output_path / "eval_config.yaml")
        save_lerf_ovs_results([(model_path, metrics)], output_path)
        print(f"[LERF-OVS] Outputs written to: {output_path}")

    if is_eval_dir and results:
        run_output_path = exp_dir / eval_subdir / output_name
        run_output_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(eval_cfg, run_output_path / "eval_config.yaml")
        save_lerf_ovs_results(results, run_output_path)
        print(f"[LERF-OVS] Aggregated outputs written to: {run_output_path}")


if __name__ == "__main__":
    main()
