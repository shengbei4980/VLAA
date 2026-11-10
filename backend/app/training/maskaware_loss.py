"""Valid-pixel supervision for pinned Ultralytics 8.4 v8/v11 heads.

The assignment mask follows TaskAlignedAssigner.get_pos_mask in Ultralytics
8.4.115. Invalid anchors are removed before top-k selection and normalization.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.utils.loss import v8DetectionLoss, v8SegmentationLoss
from ultralytics.utils.ops import crop_mask
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors


class _ValidAssigner(TaskAlignedAssigner):
    valid_anchors: torch.Tensor | None = None

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        if self.valid_anchors is None:
            raise ValueError("Mask-aware assignment requires a valid-anchor mask.")
        valid = self.valid_anchors.to(pd_scores.device).unsqueeze(1)
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt) * valid
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        return mask_topk * mask_in_gts * mask_gt, align_metric, overlaps


class _ValidBCE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.valid_anchors: torch.Tensor | None = None

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.valid_anchors is None:
            raise ValueError("Mask-aware classification requires a valid-anchor mask.")
        return F.binary_cross_entropy_with_logits(prediction, target, reduction="none") * self.valid_anchors.unsqueeze(-1)


def canonical_batch(batch: dict, device: torch.device) -> dict:
    """Remove ignored target pixels before object boxes and assignments are built."""
    if "valid_mask" not in batch:
        raise ValueError("Mask-aware supervision requires batch['valid_mask'].")
    valid = batch["valid_mask"].to(device=device, dtype=torch.bool)
    if valid.ndim != 3 or valid.shape != (batch["img"].shape[0], *batch["img"].shape[-2:]):
        raise ValueError("valid_mask must have shape [batch, image_height, image_width].")
    height, width = valid.shape[-2:]
    indices = batch["batch_idx"].to(device=device).view(-1).long()
    classes = batch["cls"].to(device=device)
    boxes = batch["bboxes"].to(device=device)
    if len(indices) != len(classes) or len(indices) != len(boxes):
        raise ValueError("Target classes, boxes and image indices must have matching lengths.")
    masks = batch.get("masks")
    if masks is not None:
        masks = masks.to(device=device)
        if masks.shape != (len(indices), height, width) or not torch.isfinite(masks).all():
            raise ValueError("Instance masks must be finite full-resolution [instances, height, width].")
    keep, visible_boxes, visible_masks = [], [], []
    for index, image_index in enumerate(indices.tolist()):
        if not 0 <= image_index < len(valid):
            raise ValueError("Target image index is out of bounds.")
        if masks is not None:
            mask = (masks[index] > 0.5) & valid[image_index]
            ys, xs = torch.where(mask)
        else:
            cx, cy, box_width, box_height = boxes[index]
            x1 = max(0, int(torch.floor((cx - box_width / 2) * width).item()))
            y1 = max(0, int(torch.floor((cy - box_height / 2) * height).item()))
            x2 = min(width, int(torch.ceil((cx + box_width / 2) * width).item()))
            y2 = min(height, int(torch.ceil((cy + box_height / 2) * height).item()))
            if x2 <= x1 or y2 <= y1:
                continue
            ys, xs = torch.where(valid[image_index, y1:y2, x1:x2])
            ys, xs = ys + y1, xs + x1
        if not len(xs):
            continue
        left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        visible_boxes.append(torch.stack(((left + right) / (2 * width), (top + bottom) / (2 * height),
                                         (right - left) / width, (bottom - top) / height)))
        keep.append(index)
        if masks is not None:
            visible_masks.append(mask)
    selected = torch.tensor(keep, device=device, dtype=torch.long)
    order = indices[selected].argsort(stable=True)
    result = {**batch, "valid_mask": valid, "batch_idx": indices[selected][order], "cls": classes[selected][order],
              "bboxes": torch.stack(visible_boxes)[order].to(boxes.dtype) if keep else boxes.new_empty((0, 4))}
    if masks is not None:
        result["masks"] = torch.stack(visible_masks)[order].float() if keep else masks.new_empty((0, height, width)).float()
    return result


class _MaskAwareTargets:
    def __init__(self, model, *args, **kwargs) -> None:
        if getattr(model, "end2end", False):
            raise ValueError("The mask-aware adapter supports non-end2end v8/v11 heads.")
        super().__init__(model, *args, **kwargs)
        if getattr(self, "overlap", False):
            raise ValueError("Mask-aware instance supervision requires overlap_mask=False.")
        original = self.assigner
        self.assigner = _ValidAssigner(topk=original.topk, num_classes=original.num_classes, alpha=original.alpha,
                                      beta=original.beta, stride=original.stride, eps=original.eps, topk2=original.topk2)
        self.bce = _ValidBCE()
        self._valid_pixels: torch.Tensor | None = None

    def loss(self, preds: dict, batch: dict):
        batch = canonical_batch(batch, self.device)
        self._valid_pixels = batch["valid_mask"]
        if "proto" in preds:
            if isinstance(preds["proto"], tuple):
                raise ValueError("Semantic auxiliary heads require a separate mask-aware adapter.")
            proto = preds["proto"]
            # A prototype cell touching ignore has no supervised output or gradient.
            excluded = F.adaptive_max_pool2d((~self._valid_pixels).float().unsqueeze(1), proto.shape[-2:]).bool()
            preds = {**preds, "proto": proto.masked_fill(excluded, 0)}
        return super().loss(preds, batch)

    def get_assigned_targets_and_loss(self, preds: dict, batch: dict):
        valid = batch["valid_mask"].to(self.device).bool()
        anchors, strides = make_anchors(preds["feats"], self.stride, 0.5)
        centers = anchors * strides
        x, y = centers[:, 0].long(), centers[:, 1].long()
        if (x >= valid.shape[2]).any() or (y >= valid.shape[1]).any():
            raise ValueError("Prediction anchor grid does not fit valid_mask dimensions.")
        valid_anchors = valid[:, y, x]
        self.assigner.valid_anchors = valid_anchors
        self.bce.valid_anchors = valid_anchors
        return super().get_assigned_targets_and_loss(preds, batch)


class MaskAwareDetectionLoss(_MaskAwareTargets, v8DetectionLoss):
    """Detection loss with valid-region targets and negative supervision."""


class MaskAwareSegmentationLoss(_MaskAwareTargets, v8SegmentationLoss):
    """Instance loss on canonical masks, retaining holes and disconnected regions."""

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz):
        if self._valid_pixels is None:
            raise ValueError("Mask-aware segmentation requires valid pixels.")
        loss = proto.sum() * 0 + pred_masks.sum() * 0
        height, width = proto.shape[-2:]
        scale = torch.tensor([width, height, width, height], device=proto.device)
        boxes = target_bboxes / imgsz[[1, 0, 1, 0]] * scale
        valid = self._valid_pixels
        if valid.shape[-2:] != (height, width):
            raise ValueError("Mask loss must use original-resolution canonical masks.")
        for image_index in range(len(fg_mask)):
            foreground = fg_mask[image_index]
            if not foreground.any():
                continue
            target_indices = target_gt_idx[image_index, foreground]
            truth = masks[batch_idx.view(-1) == image_index][target_indices]
            logits = torch.einsum("in,nhw->ihw", pred_masks[image_index, foreground], proto[image_index])
            pixel_loss = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
            selected_boxes = boxes[image_index, foreground]
            supervision = crop_mask(valid[image_index].expand_as(pixel_loss).float(), selected_boxes)
            loss += ((pixel_loss * supervision).sum((1, 2)) / supervision.sum((1, 2)).clamp_min(1)).sum()
        return loss / fg_mask.sum().clamp_min(1)
