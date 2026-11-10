"""Canonical instance masks and supervision regions share every image transform."""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from app.core.errors import ValidationError
from app.core.storage import Storage
from app.dataset.annotation.masks import MaskService, polygon_mask


class CanonicalDataset(Dataset):
    rect = False
    channels = 3

    def __init__(self, data: dict, split: str, imgsz: int, *, augment: bool = False):
        self.manifest_path = Path(data["workflow_manifest"]).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.root = self.manifest_path.parent
        self.images = [image for image in self.manifest["images"] if image["split"] == split]
        self.class_indexes = {item["id"]: item["index"] for item in self.manifest["classes"]}
        self.task = self.manifest["task_type"]
        if self.task not in {"detect", "segment"} or not self.images:
            raise ValidationError("canonical_dataset_unsupported", "Canonical training needs a nonempty detect or segment split.")
        data_dir = Path(data["workflow_data_dir"]).resolve()
        if not self.root.is_relative_to(data_dir / "snapshots"):
            raise ValidationError("canonical_manifest_unsafe", "Canonical manifests must belong to managed snapshots.")
        self.masks = MaskService(Storage(data_dir, data_dir / "imports"))
        self.imgsz = imgsz
        self.augment = augment
        self.im_files = [str(self._image_path(image)) for image in self.images]
        self.labels = []
        for image in self.images:
            objects, _ = self._objects(image)
            boxes, classes = [], []
            for class_index, mask in objects:
                ys, xs = np.nonzero(mask)
                boxes.append([(xs.min() + xs.max() + 1) / (2 * image["width"]),
                              (ys.min() + ys.max() + 1) / (2 * image["height"]),
                              (xs.max() - xs.min() + 1) / image["width"], (ys.max() - ys.min() + 1) / image["height"]])
                classes.append([class_index])
            self.labels.append({"im_file": str(self._image_path(image)), "shape": (image["height"], image["width"]),
                                "cls": np.asarray(classes, dtype=np.float32).reshape(-1, 1),
                                "bboxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                                "segments": [], "normalized": True, "bbox_format": "xywh"})

    def _image_path(self, image: dict) -> Path:
        path = (self.root / image["path"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValidationError("canonical_image_missing", "A canonical image is missing or outside its snapshot.")
        return path

    def _objects(self, image: dict) -> tuple[Iterator[tuple[int, np.ndarray]], np.ndarray]:
        shape = (image["height"], image["width"])
        valid = self.masks.decode(image["roi_mask_ref"]) if image.get("roi_mask_ref") else np.ones(shape, dtype=bool)
        if image.get("ignore_mask_ref"):
            valid = valid & ~self.masks.decode(image["ignore_mask_ref"])
        if valid.shape != shape:
            raise ValidationError("canonical_region_size", "Supervision masks must match their original image.")
        return self._object_masks(image, valid), valid

    def _object_masks(self, image: dict, valid: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
        shape = valid.shape
        for annotation in image["annotations"]:
            if self.task == "segment":
                mask = self.masks.decode(annotation["mask_ref"]) if annotation.get("mask_ref") else polygon_mask(annotation["polygon"], image["width"], image["height"])
            else:
                box = annotation["bbox"]
                mask = np.zeros(shape, dtype=bool)
                x1, y1 = int(np.floor(box["x"])), int(np.floor(box["y"]))
                x2, y2 = int(np.ceil(box["x"] + box["width"])), int(np.ceil(box["y"] + box["height"]))
                mask[max(0, y1):min(shape[0], y2), max(0, x1):min(shape[1], x2)] = True
            if mask.shape != shape:
                raise ValidationError("canonical_object_size", "Object masks must match their original image.")
            mask = mask & valid
            if mask.any():
                yield self.class_indexes[annotation["class_id"]], mask

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image = self.images[index]
        path = self._image_path(image)
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != image["image_hash"]:
                raise ValidationError("canonical_image_changed", "A frozen image failed its integrity check.")
        with Image.open(path) as opened:
            pixels = np.asarray(opened.convert("RGB"))
        objects, valid = self._objects(image)
        height, width = pixels.shape[:2]
        ratio = self.imgsz / max(height, width)
        resized_width, resized_height = max(1, round(width * ratio)), max(1, round(height * ratio))
        left, top = (self.imgsz - resized_width) // 2, (self.imgsz - resized_height) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(pixels, (resized_width, resized_height))

        def transform(mask: np.ndarray) -> np.ndarray:
            result = np.zeros((self.imgsz, self.imgsz), dtype=bool)
            result[top:top + resized_height, left:left + resized_width] = cv2.resize(mask.astype(np.uint8), (resized_width, resized_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            return result

        valid = transform(valid)
        transformed = [(class_index, transform(mask)) for class_index, mask in objects]
        if self.augment and random.random() < 0.5:
            canvas = canvas[:, ::-1]
            valid = valid[:, ::-1]
            transformed = [(class_index, mask[:, ::-1]) for class_index, mask in transformed]
        boxes, classes, masks = [], [], []
        for class_index, mask in transformed:
            ys, xs = np.nonzero(mask)
            if not len(xs):
                raise ValidationError("canonical_target_resize_loss", "A target disappears at this image size. Increase image size or use tiles.")
            boxes.append([(xs.min() + xs.max() + 1) / (2 * self.imgsz), (ys.min() + ys.max() + 1) / (2 * self.imgsz),
                          (xs.max() - xs.min() + 1) / self.imgsz, (ys.max() - ys.min() + 1) / self.imgsz])
            classes.append([class_index])
            masks.append(mask)
        count = len(classes)
        return {"img": torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))),
                "valid_mask": torch.from_numpy(np.ascontiguousarray(valid)),
                "masks": torch.from_numpy(np.asarray(masks, dtype=np.float32).reshape(count, self.imgsz, self.imgsz)),
                "cls": torch.tensor(classes, dtype=torch.float32).reshape(-1, 1),
                "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), "batch_idx": torch.zeros(count),
                "im_file": str(path), "ori_shape": (height, width), "resized_shape": (self.imgsz, self.imgsz),
                "ratio_pad": ((ratio, ratio), (left, top))}

    @staticmethod
    def collate_fn(samples: list[dict]) -> dict:
        batch = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key in {"img", "valid_mask"}:
                batch[key] = torch.stack(values)
            elif key in {"masks", "cls", "bboxes"}:
                batch[key] = torch.cat(values)
            elif key == "batch_idx":
                batch[key] = torch.cat([value + index for index, value in enumerate(values)])
            else:
                batch[key] = values
        return batch

    def close_mosaic(self, hyp=None) -> None:
        pass
