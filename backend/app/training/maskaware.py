"""Managed Ultralytics trainers and validators for canonical mask snapshots."""
from __future__ import annotations

import sys
from copy import copy
import torch
import torch.nn.functional as functional
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.nn.tasks import DetectionModel, SegmentationModel, guess_model_task
from ultralytics.utils import ops
from ultralytics.utils.metrics import mask_iou

from app.training.maskaware_data import CanonicalDataset


def valid_box_iou(first: torch.Tensor, second: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Compute rectangle intersections using only valid pixel centres."""
    height, width = valid.shape
    integral = functional.pad(valid.float().cumsum(0).cumsum(1), (1, 0, 1, 0))

    def area(boxes: torch.Tensor) -> torch.Tensor:
        low = (boxes[..., :2] - 0.5).ceil().long()
        high = (boxes[..., 2:] - 0.5).ceil().long()
        x1, y1 = low[..., 0].clamp(0, width), low[..., 1].clamp(0, height)
        x2, y2 = high[..., 0].clamp(0, width), high[..., 1].clamp(0, height)
        result = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
        return result * ((x2 > x1) & (y2 > y1))

    intersections = torch.cat((torch.maximum(first[:, None, :2], second[None, :, :2]),
                               torch.minimum(first[:, None, 2:], second[None, :, 2:])), dim=-1)
    intersection = area(intersections)
    return intersection / (area(first)[:, None] + area(second)[None, :] - intersection + 1e-7)


class CanonicalValidation:
    def init_metrics(self, model):
        from ultralytics.nn.modules.head import Detect, Segment

        current = model
        head = None
        for _ in range(4):
            current = getattr(current, "model", None)
            if isinstance(current, (torch.nn.Sequential, torch.nn.ModuleList)):
                head = current[-1]
                break
        if type(head) not in {Detect, Segment} or getattr(head, "end2end", False):
            raise ValueError("Canonical validation requires a non-end2end v8/v11 Detect or Segment head.")
        self._canonical_strides = head.stride.detach().cpu().tolist()
        super().init_metrics(model)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        valid = batch.get("valid_mask")
        if valid is None or valid.shape != (batch["img"].shape[0], *batch["img"].shape[-2:]):
            raise ValueError("Canonical validation requires full-resolution valid_mask.")
        self._canonical_valid = valid.to(device=self.device, dtype=torch.bool)
        return batch

    def postprocess(self, preds):
        valid = getattr(self, "_canonical_valid", None)
        strides = getattr(self, "_canonical_strides", None)
        if valid is None or not strides or getattr(self, "end2end", False):
            raise ValueError("Canonical validation needs an initialized valid-anchor profile before NMS.")
        height, width = valid.shape[-2:]
        anchor_validity = []
        for stride in strides:
            stride = int(stride)
            if stride <= 0 or height % stride or width % stride:
                raise ValueError("Validation dimensions must match the supported head strides.")
            ys = ((torch.arange(height // stride, device=valid.device) + .5) * stride).long()
            xs = ((torch.arange(width // stride, device=valid.device) + .5) * stride).long()
            anchor_validity.append(valid[:, ys[:, None], xs[None, :]].flatten(1))
        valid_anchors = torch.cat(anchor_validity, dim=1)
        prototypes = None
        if self.args.task == "segment":
            prototypes = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
            if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 4:
                raise ValueError("Canonical segmentation requires the supported prototype output.")
            excluded = functional.adaptive_max_pool2d((~valid).float().unsqueeze(1), prototypes.shape[-2:]).bool()
            prototypes = prototypes.masked_fill(excluded, 0)

        def masked_output(value):
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                if value.shape[0] != len(valid) or value.shape[-1] != valid_anchors.shape[1] or value.shape[1] < 4 + self.nc:
                    raise ValueError("Prediction output does not match the declared canonical anchor grid.")
                output = value.clone()
                boxes = ops.xywh2xyxy(output[:, :4].transpose(1, 2))
                eligible = valid_anchors.clone()
                for image_index in range(len(valid)):
                    full = boxes.new_tensor([[0, 0, width, height]])
                    eligible[image_index] &= valid_box_iou(boxes[image_index], full, valid[image_index])[:, 0] > 0
                    if prototypes is not None:
                        if output.shape[1] != 4 + self.nc + prototypes.shape[1]:
                            raise ValueError("Canonical segmentation coefficients do not match its prototype channels.")
                        selected = torch.where(eligible[image_index] & (output[image_index, 4:4 + self.nc].amax(0) > self.args.conf))[0]
                        # Mask-only ignore filtering must precede NMS too. Bound temporary mask memory.
                        for chunk in selected.split(64):
                            masks = ops.process_mask_native(prototypes[image_index], output[image_index, 4 + self.nc:, chunk].T,
                                                            boxes[image_index, chunk], (height, width)).bool()
                            positive = masks.flatten(1).any(1)
                            visible = (masks & valid[image_index]).flatten(1).any(1)
                            eligible[image_index, chunk] &= ~positive | visible
                output[:, 4:4 + self.nc] = output[:, 4:4 + self.nc].masked_fill(~eligible[:, None], 0)
                return output
            if isinstance(value, torch.Tensor) and value.ndim == 4 and self.args.task == "segment":
                return prototypes
            if isinstance(value, tuple):
                return tuple(masked_output(item) for item in value)
            if isinstance(value, list):
                return [masked_output(item) for item in value]
            return value

        return super().postprocess(masked_output(preds))

    def build_dataset(self, img_path, mode="val", batch=None):
        return CanonicalDataset(self.data, self.args.split, self.args.imgsz)

    def _prepare_batch(self, si, batch):
        prepared = super()._prepare_batch(si, batch)
        prepared["valid_mask"] = batch["valid_mask"][si].to(self.device)
        return prepared

    def update_metrics(self, preds, batch):
        filtered = []
        for index, prediction in enumerate(preds):
            valid = batch["valid_mask"][index].to(prediction["bboxes"].device)
            prediction = dict(prediction)
            if "masks" in prediction:
                masks = prediction["masks"]
                if masks.shape[-2:] != valid.shape:
                    masks = functional.interpolate(masks[:, None].float(), size=valid.shape, mode="nearest")[:, 0]
                positive = masks.bool().flatten(1).any(1)
                prediction["masks"] = masks.bool() & valid
                full = prediction["bboxes"].new_tensor([[0, 0, valid.shape[1], valid.shape[0]]])
                box_visible = valid_box_iou(prediction["bboxes"], full, valid)[:, 0] > 0
                keep = prediction["masks"].flatten(1).any(1) | (~positive & box_visible)
            else:
                boxes = prediction["bboxes"]
                full = boxes.new_tensor([[0, 0, valid.shape[1], valid.shape[0]]])
                keep = valid_box_iou(boxes, full, valid)[:, 0] > 0
            filtered.append({key: value[keep] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(keep) else value
                             for key, value in prediction.items()})
        super().update_metrics(filtered, batch)

    def _process_batch(self, preds, batch):
        iou = valid_box_iou(batch["bboxes"], preds["bboxes"], batch["valid_mask"])
        result = {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}
        if "masks" in preds:
            valid = batch["valid_mask"]
            ground_truth = batch["masks"]
            if ground_truth.shape[-2:] != valid.shape:
                ground_truth = functional.interpolate(ground_truth[:, None].float(), size=valid.shape, mode="nearest")[:, 0]
            truth = ground_truth.bool() & valid
            mask_overlap = mask_iou(truth.flatten(1).float(), preds["masks"].flatten(1).float())
            result["tp_m"] = self.match_predictions(preds["cls"], batch["cls"], mask_overlap).cpu().numpy()
        return result


class CanonicalDetectionValidator(CanonicalValidation, DetectionValidator):
    pass


class CanonicalSegmentationValidator(CanonicalValidation, SegmentationValidator):
    def init_metrics(self, model):
        # Canonical mask metrics use full resolution; the generic COCO exporter is bypassed.
        self.args.save_json = False
        super().init_metrics(model)
        self.process = ops.process_mask_native


class CanonicalDetectionModel(DetectionModel):
    def init_criterion(self):
        from app.training.maskaware_loss import MaskAwareDetectionLoss

        if getattr(self, "end2end", False):
            raise ValueError("Canonical supervision currently requires a non-end2end YOLO detection head.")
        return MaskAwareDetectionLoss(self)


class CanonicalSegmentationModel(SegmentationModel):
    def init_criterion(self):
        from app.training.maskaware_loss import MaskAwareSegmentationLoss

        if getattr(self, "end2end", False):
            raise ValueError("Canonical supervision currently requires a non-end2end YOLO segmentation head.")
        return MaskAwareSegmentationLoss(self)


class CanonicalTraining:
    model_class = CanonicalDetectionModel
    validator_class = CanonicalDetectionValidator

    def build_dataset(self, img_path, mode="train", batch=None):
        return CanonicalDataset(self.data, "train" if mode == "train" else "val", self.args.imgsz, augment=mode == "train")

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(self.model_class(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose))
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        return self.validator_class(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)


class CanonicalDetectionTrainer(CanonicalTraining, DetectionTrainer):
    pass


class CanonicalSegmentationTrainer(CanonicalTraining, SegmentationTrainer):
    model_class = CanonicalSegmentationModel
    validator_class = CanonicalSegmentationValidator


def main(arguments: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if len(arguments) < 2 or arguments[0] not in {"detect", "segment"} or arguments[1] not in {"train", "val"}:
        raise ValueError("Expected a managed detect/segment train/val invocation.")
    task, mode = arguments[:2]
    from ultralytics.cfg import parse_key_value_pair

    overrides = dict(parse_key_value_pair(value) for value in arguments[2:])
    model_reference = overrides.pop("model")
    overrides.update(overlap_mask=False, mask_ratio=1, rect=False, multi_scale=0.0, close_mosaic=0,
                     mosaic=0.0, mixup=0.0, copy_paste=0.0, amp=False)
    model = YOLO(model_reference, task=task)
    actual_task = guess_model_task(model.model)
    if actual_task != task:
        raise ValueError(f"The selected model supports {actual_task}; this workflow requires {task}.")
    if getattr(model.model, "end2end", False):
        raise ValueError("This canonical adapter supports non-end2end YOLO v8/v11 heads.")
    if mode == "train":
        trainer = CanonicalSegmentationTrainer if task == "segment" else CanonicalDetectionTrainer
        model.train(trainer=trainer, **overrides)
    else:
        validator = CanonicalSegmentationValidator if task == "segment" else CanonicalDetectionValidator
        model.val(validator=validator, **overrides)


if __name__ == "__main__":
    # Persist checkpoint classes under their importable package module.
    from app.training.maskaware import main as run_managed

    run_managed()
