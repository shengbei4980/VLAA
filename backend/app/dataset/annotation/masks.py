from __future__ import annotations

import json
import os
import re
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel
from pycocotools import mask as coco_mask

from app.core.errors import ValidationError
from app.core.schemas import BBox
from app.core.storage import Storage
from app.dataset.annotation.geometry import validate_polygon

MAX_MASK_PIXELS = 100_000_000
_MASK_REF = re.compile(r"masks/([a-f0-9]{64})\.json\Z")


class MaskArtifact(BaseModel):
    mask_ref: str
    sha256: str
    width: int
    height: int
    area: int
    bbox: BBox | None = None
    coordinate_space: Literal["image_pixel"] = "image_pixel"


def _dimensions(size: object) -> tuple[int, int]:
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValidationError("invalid_mask_size", "RLE size must be [height, width].")
    if any(type(value) is not int or value <= 0 for value in size):
        raise ValidationError("invalid_mask_size", "Mask dimensions must be positive integers.")
    height, width = size
    if height * width > MAX_MASK_PIXELS:
        raise ValidationError("mask_too_large", "Mask exceeds the supported pixel limit.")
    return height, width


def _runs(counts: object, pixels: int) -> list[int]:
    if isinstance(counts, list):
        runs = counts
    elif isinstance(counts, str):
        if len(counts) > min(pixels * 8 + 16, 32_000_000):
            raise ValidationError("invalid_mask_rle", "RLE counts exceed the supported limit.")
        runs = []
        position = 0
        while position < len(counts):
            value = 0
            shift = 0
            while True:
                if position >= len(counts) or shift > 60:
                    raise ValidationError("invalid_mask_rle", "Truncated or excessive RLE count.")
                code = ord(counts[position]) - 48
                position += 1
                if not 0 <= code <= 63:
                    raise ValidationError("invalid_mask_rle", "RLE contains an invalid character.")
                value |= (code & 31) << shift
                shift += 5
                if not code & 32:
                    if code & 16:
                        value |= -1 << shift
                    break
            if len(runs) > 2:
                value += runs[-2]
            runs.append(value)
    else:
        raise ValidationError("invalid_mask_rle", "RLE counts must be a string or integer array.")
    if not runs or len(runs) > pixels + 1 or any(type(run) is not int or run < 0 for run in runs):
        raise ValidationError("invalid_mask_rle", "RLE counts must be nonnegative integer runs.")
    if sum(runs) != pixels:
        raise ValidationError("invalid_mask_rle", "RLE runs do not cover the declared image.")
    return runs


def decode_mask(rle: dict) -> np.ndarray:
    if not isinstance(rle, dict):
        raise ValidationError("invalid_mask_rle", "Mask must be a COCO RLE object.")
    height, width = _dimensions(rle.get("size"))
    runs = _runs(rle.get("counts"), height * width)
    flat = np.zeros(height * width, dtype=np.bool_)
    position = 0
    for index, length in enumerate(runs):
        if index % 2:
            flat[position:position + length] = True
        position += length
    return flat.reshape((height, width), order="F")


def encode_mask(mask: np.ndarray) -> dict:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValidationError("invalid_mask_size", "A mask must be a two-dimensional array.")
    _dimensions(list(array.shape))
    if not np.isin(array, [0, 1]).all():
        raise ValidationError("invalid_mask_values", "Masks must contain only binary values.")
    encoded = coco_mask.encode(np.asfortranarray(array, dtype=np.uint8))
    return {"size": [int(value) for value in encoded["size"]], "counts": encoded["counts"].decode("ascii")}


def polygon_mask(points: list[tuple[float, float]], width: int, height: int) -> np.ndarray:
    _dimensions([height, width])
    validate_polygon(points, width, height)
    rles = coco_mask.frPyObjects([[coordinate for point in points for coordinate in point]], height, width)
    return coco_mask.decode(coco_mask.merge(rles)).astype(np.bool_)


def mask_bbox(mask: np.ndarray) -> BBox | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return BBox(x=int(xs.min()), y=int(ys.min()), width=int(xs.max() - xs.min() + 1), height=int(ys.max() - ys.min() + 1))


class MaskService:
    """Content-addressed, lossless mask IO inside the managed data directory."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage

    def path(self, mask_ref: str) -> Path:
        if not _MASK_REF.fullmatch(mask_ref):
            raise ValidationError("unsafe_mask_ref", "Mask reference must identify a managed mask artifact.")
        path = (self.storage.data_dir / mask_ref).resolve()
        if not path.is_relative_to(self.storage.data_dir):
            raise ValidationError("unsafe_mask_ref", "Mask path escapes managed storage.")
        return path

    def write(self, mask: np.ndarray | dict) -> MaskArtifact:
        array = decode_mask(mask) if isinstance(mask, dict) else np.asarray(mask)
        rle = encode_mask(array)
        content = json.dumps(rle, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = sha256(content).hexdigest()
        reference = f"masks/{digest}.json"
        path = self.path(reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            if path.is_file():
                if path.read_bytes() != content:
                    raise ValidationError("mask_integrity_failed", "Existing mask artifact failed its content hash.")
            else:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as target:
                    temporary = target.name
                    target.write(content)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, path)
        except OSError as exc:
            raise ValidationError("mask_write_failed", "Could not store the managed mask.") from exc
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        height, width = array.shape
        return MaskArtifact(mask_ref=reference, sha256=digest, width=width, height=height, area=int(array.sum()), bbox=mask_bbox(array))

    def read(self, mask_ref: str) -> dict:
        path = self.path(mask_ref)
        try:
            if path.stat().st_size > 32_000_000:
                raise ValidationError("mask_too_large", "Stored mask exceeds the supported file limit.")
            content = path.read_bytes()
            if sha256(content).hexdigest() != _MASK_REF.fullmatch(mask_ref).group(1):
                raise ValidationError("mask_integrity_failed", "Stored mask failed its content hash.")
            rle = json.loads(content)
        except (OSError, ValueError) as exc:
            raise ValidationError("mask_read_failed", "Could not read the managed mask.") from exc
        if not isinstance(rle, dict):
            raise ValidationError("invalid_mask_rle", "Stored mask must be a COCO RLE object.")
        height, width = _dimensions(rle.get("size"))
        _runs(rle.get("counts"), height * width)
        return rle

    def decode(self, mask_ref: str) -> np.ndarray:
        return decode_mask(self.read(mask_ref))
