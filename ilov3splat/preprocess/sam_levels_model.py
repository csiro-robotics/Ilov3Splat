"""Multi-level SAM automatic mask generator."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
from torchvision.ops.boxes import batched_nms, box_area


class SamLevelsAutomaticMaskGenerator(SamAutomaticMaskGenerator):
    @torch.no_grad()
    def generate(self, image: np.ndarray) -> tuple[list[dict[str, Any]], ...]:
        mask_levels = self._generate_masks(image)
        curr_anns = tuple(
            self._generate_current_anns(mask_data) for mask_data in mask_levels
        )
        return curr_anns

    def _generate_current_anns(self, mask_data: MaskData) -> list[dict[str, Any]]:
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [
                coco_encode_rle(rle) for rle in mask_data["rles"]
            ]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> tuple[MaskData, ...]:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        data_levels = (MaskData(), MaskData(), MaskData(), MaskData())
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_levels = self._process_crop(image, crop_box, layer_idx, orig_size)
            assert len(crop_levels) == len(data_levels)
            for u, v in zip(data_levels, crop_levels):
                u.cat(v)

        data_levels = tuple(
            self._generate_masks_data(data, crop_boxes) for data in data_levels
        )
        return data_levels

    def _generate_masks_data(
        self, data: MaskData, crop_boxes: list[list[int]]
    ) -> MaskData:
        if len(crop_boxes) > 1:
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: list[int],
        crop_layer_idx: int,
        orig_size: tuple[int, ...],
    ) -> tuple[MaskData, ...]:
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        data_levels = (MaskData(), MaskData(), MaskData(), MaskData())
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size
            )
            assert len(batch_data) == len(data_levels)
            for u, v in zip(data_levels, batch_data):
                u.cat(v)
            del batch_data
        self.predictor.reset_image()

        data_levels = tuple(
            self._process_crop_data(data, crop_box) for data in data_levels
        )

        return data_levels

    def _process_crop_data(self, data: MaskData, crop_box: list[int]) -> MaskData:
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])
        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: tuple[int, ...],
        crop_box: list[int],
        orig_size: tuple[int, ...],
    ) -> tuple[MaskData, ...]:
        orig_h, orig_w = orig_size

        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        data_default = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )

        points_tensor = torch.as_tensor(points)
        assert masks.shape[1] == 3
        data_levels = tuple(
            MaskData(
                masks=masks[:, i, :, :],
                iou_preds=iou_preds[:, i],
                points=points_tensor,
            )
            for i in range(masks.shape[1])
        )
        del masks

        data_default = self._process_batch_data(data_default, crop_box, orig_h, orig_w)
        data_levels = tuple(
            self._process_batch_data(data, crop_box, orig_h, orig_w)
            for data in data_levels
        )
        return data_default, *data_levels

    def _process_batch_data(
        self, data: MaskData, crop_box: list[int], orig_h: int, orig_w: int
    ) -> MaskData:
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        data["stability_score"] = calculate_stability_score(
            data["masks"],
            self.predictor.model.mask_threshold,
            self.stability_score_offset,
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data
