"""Pixel-aligned Sentinel-2 classification and priority fusion for draft LabelMe labels."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.errors import ValidationError
from app.dataset.annotation.masks import decode_mask, encode_mask, polygon_mask
from app.dataset.exchange.labelme import _native_mask, _shape_polygon

CLASSES = ("water", "bare", "grass", "wood", "crop", "built")
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
MODEL_PATH = Path(__file__).with_name("weights") / "panan_s2_2025.xml.gz"
MODEL_SHA256 = "9825ceb2058f19f341723cba34325e451e84aab68f92b8297bdead5497463f2f"
MODEL_NAME = "panan-s2-rf-2025"
CONFLICT = 254
IGNORE = 255
MAX_WINDOW_PIXELS = 4_194_304
TIFF_HEADERS = {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}


def _invalid(message: str) -> ValidationError:
    return ValidationError("invalid_hybrid_input", message)


def model_ready() -> bool:
    return MODEL_PATH.is_file()


@lru_cache(maxsize=1)
def load_forest():
    if not model_ready() or hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise _invalid("The bundled ten-band random forest is missing or differs from its published checksum.")
    forest = cv2.ml.RTrees_load(str(MODEL_PATH))
    if not forest.isTrained() or not forest.isClassifier() or forest.getVarCount() != len(BANDS):
        raise _invalid("The bundled model is not the expected ten-band classifier.")
    return forest


def _band_name(description):
    token = re.split(r"[^A-Za-z0-9]+", (description or "").strip().upper())[0]
    if re.fullmatch(r"B[2-8]", token):
        return "B0" + token[1]
    return token


def _reflectance_encoding(dataset, indexes):
    """Use an explicit study-wide scale tag or GDAL per-band calibration."""
    scales = np.array([dataset.scales[i - 1] for i in indexes], dtype=np.float32)
    offsets = np.array([dataset.offsets[i - 1] for i in indexes], dtype=np.float32)
    tag = dataset.tags().get("reflectance_scale_factor")
    if tag is not None:
        try:
            scale = float(tag)
        except (ValueError, TypeError) as exc:
            raise _invalid("Invalid reflectance_scale_factor metadata.") from exc
        if np.any(scales != 1) and not np.allclose(scales, scale, rtol=1e-6, atol=0):
            raise _invalid("The global reflectance scale and GDAL band scales disagree.")
        scales[:] = scale
    elif np.all(scales == 1) and not all(np.issubdtype(np.dtype(dataset.dtypes[i-1]), np.floating) for i in indexes):
        raise _invalid("Integer spectral data must declare reflectance_scale_factor or GDAL band scales.")
    if not np.isfinite(scales).all() or np.any(scales <= 0) or not np.isfinite(offsets).all():
        raise _invalid("Spectral calibration must contain finite positive scales and finite offsets.")
    return scales, offsets


def _tiff_source(source):
    if isinstance(source, bytes):
        header = source[:4]
        source = io.BytesIO(source)
    elif isinstance(source, (Path, str)):
        if "://" in str(source) or str(source).lower().startswith("/vsi"):
            raise _invalid("Supply a local GeoTIFF file or uploaded bytes.")
        with Path(source).open("rb") as handle:
            header = handle.read(4)
    else:
        source.seek(0)
        header = source.read(4)
        source.seek(0)
    if header not in TIFF_HEADERS:
        raise _invalid("The uploaded file is not a TIFF/BigTIFF raster.")
    return source


def predict_geotiff(source, *, width: int, height: int, x: int = 0, y: int = 0,
                    roi=None, expected_crs=None, expected_transform=None):
    """Read exactly one tile on its original pixel grid and score valid reflectance.

    Integer coordinates are zero-based offsets in the supplied GeoTIFF. expected_transform,
    when present, is the tile's six affine coefficients (a,b,c,d,e,f). A floating
    GeoTIFF with identity calibration is interpreted as physical reflectance.
    """
    import rasterio
    from rasterio.windows import Window
    from rasterio.errors import RasterioError

    if any(type(v) is not int for v in (x, y, width, height)) or min(x, y) < 0 or min(width, height) < 1:
        raise _invalid("The raster window requires nonnegative integer offsets and positive dimensions.")
    if width * height > MAX_WINDOW_PIXELS:
        raise _invalid("The requested tile exceeds the 4,194,304 pixel spectral window limit.")
    if roi is not None and (np.shape(roi) != (height, width) or np.asarray(roi).dtype != np.bool_):
        raise _invalid("The ROI must be a boolean mask matching the tile dimensions.")
    file_source = _tiff_source(source)
    window = Window(x, y, width, height)
    try:
        with rasterio.open(file_source, driver="GTiff") as dataset:
            if dataset.driver != "GTiff":
                raise _invalid("Only GeoTIFF rasters are supported.")
            descriptions = [_band_name(d) for d in dataset.descriptions]
            if any(descriptions.count(band) != 1 for band in BANDS):
                raise _invalid("Expected one labelled band for each of B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12.")
            if not dataset.crs or (expected_crs is not None and dataset.crs != rasterio.crs.CRS.from_user_input(expected_crs)):
                raise _invalid("The GeoTIFF must have a CRS matching the declared tile grid.")
            if x + width > dataset.width or y + height > dataset.height:
                raise _invalid("The requested tile lies outside the spectral raster.")
            if expected_transform is not None:
                expected = np.asarray(expected_transform, dtype=float)
                if expected.shape != (6,) or not np.allclose(tuple(dataset.window_transform(window))[:6], expected, rtol=0, atol=1e-6):
                    raise _invalid("The spectral pixel grid differs from the declared tile grid.")
            indexes = [descriptions.index(band) + 1 for band in BANDS]
            scales, offsets = _reflectance_encoding(dataset, indexes)
            raw = dataset.read(indexes=indexes, window=window, masked=True).astype(np.float32).filled(np.nan)
    except RasterioError as exc:
        raise _invalid("Could not read the supplied GeoTIFF or its spatial reference.") from exc
    reflectance = np.ascontiguousarray(raw.transpose(1, 2, 0) * scales + offsets)
    eligible = np.isfinite(reflectance).all(axis=2)
    if roi is not None:
        eligible &= roi
    prediction = np.full((height, width), IGNORE, dtype=np.uint8)
    # Keep feature matrix and OpenCV result allocations bounded for full-size tiles.
    indexes = np.flatnonzero(eligible)
    matrix = reflectance.reshape(-1, len(BANDS))
    for start in range(0, len(indexes), 65536):
        selected = indexes[start:start + 65536]
        _, values = load_forest().predict(matrix[selected])
        prediction.flat[selected] = values.ravel().astype(np.uint8)
    return prediction, eligible


def _rasterize(shape: dict, width: int, height: int):
    if shape.get("shape_type", "polygon") != "mask":
        return polygon_mask(_shape_polygon(shape, width, height), width, height)
    if isinstance(shape.get("mask"), str):
        return _native_mask(shape, width, height)
    rle = shape.get("mask") or shape.get("starter_rle") or (shape.get("starter_annotation") or {}).get("mask")
    mask = decode_mask(rle)
    if mask.shape != (height, width):
        raise _invalid("A draft mask has a different size from its RGB image.")
    return mask


def _geometry(document: dict, required_type=None):
    width, height = document["imageWidth"], document["imageHeight"]
    layers = np.zeros((len(CLASSES), height, width), dtype=bool)
    for item in document["shapes"]:
        if item.get("label") not in CLASSES:
            raise _invalid("A draft contains a label outside the six supported classes.")
        if required_type and item.get("shape_type", "polygon") != required_type:
            raise _invalid(f"The input stream must contain only {required_type} shapes.")
        layers[CLASSES.index(item["label"])] |= _rasterize(item, width, height)
    return layers


def class_map(document: dict, eligible=None):
    """Return class IDs 0–5, conflict 254 and unclassified/outside 255."""
    if eligible is None and document.get("starter_roi") is not None:
        eligible = decode_mask(document["starter_roi"])
    layers = _geometry(document)
    count = layers.sum(axis=0)
    codes = layers.argmax(axis=0).astype(np.uint8)
    codes[count == 0] = IGNORE
    codes[count > 1] = CONFLICT
    if eligible is not None:
        codes[~eligible] = IGNORE
    return codes


def _mask_shape(label, mask, group, flag, source):
    yy, xx = np.nonzero(mask)
    x0, x1 = int(xx.min()), int(xx.max())
    y0, y1 = int(yy.min()), int(yy.max())
    crop = np.ascontiguousarray(mask[y0:y1+1, x0:x1+1] * np.uint8(255))
    png = io.BytesIO()
    Image.fromarray(crop).save(png, format="PNG")
    return {
        "label": label, "shape_type": "mask", "group_id": group, "flags": {flag: True},
        "points": [[x0, y0], [x1, y1]], "mask": base64.b64encode(png.getvalue()).decode("ascii"),
        "starter_rle": encode_mask(mask), "hybrid_source": source,
    }


def fuse_drafts(primary: dict, sam: dict, prediction: np.ndarray, eligible: np.ndarray):
    """Preserve GPT shapes, fill eligible gaps with unique SAM classes, then with RF.

    A SAM region contested by multiple classes remains empty in stage two. As in
    the measured pipeline, that remaining blank is eligible for the RF. A GPT
    overlap remains a conflict throughout. Equal-class overlaps are one class.
    """
    width, height = primary["imageWidth"], primary["imageHeight"]
    if (sam["imageWidth"], sam["imageHeight"]) != (width, height):
        raise _invalid("The GPT and DeepSeek/SAM drafts use different tile dimensions.")
    if prediction.shape != (height, width) or prediction.dtype != np.uint8 or not np.isin(prediction, [0,1,2,3,4,5,IGNORE]).all():
        raise _invalid("Expected six-class RF predictions or nodata in a tile-sized uint8 grid.")
    if eligible.dtype != np.bool_ or eligible.shape != (height, width):
        raise _invalid("The eligibility mask must be boolean and tile-sized.")
    eligible = eligible.copy()
    for draft in (primary, sam):
        for field in ("starter_roi", "starter_ignore"):
            if draft.get(field) is None:
                continue
            mask = decode_mask(draft[field])
            if mask.shape != eligible.shape:
                raise _invalid("A source ROI or ignore mask differs from the RGB tile dimensions.")
            eligible &= mask if field == "starter_roi" else ~mask
    first = _geometry(primary, "polygon").sum(axis=0)
    secondary = _geometry(sam, "mask").sum(axis=0)
    sam_fill = eligible & (first == 0) & (secondary == 1)
    rf_fill = eligible & (first == 0) & ~sam_fill & (prediction < len(CLASSES))
    output = {
        "version": "5.8.0", "imagePath": primary["imagePath"], "imageData": primary.get("imageData"),
        "imageWidth": width, "imageHeight": height, "flags": {"draft": True}, "shapes": copy.deepcopy(primary["shapes"]),
    }
    if not eligible.all():
        output["starter_roi"] = encode_mask(eligible)
        output["starter_ignore"] = encode_mask(~eligible)
    groups = []
    for i, item in enumerate(output["shapes"]):
        item["flags"] = {**(item.get("flags") or {}), "direct_v6": True}
        item["hybrid_source"] = {"stage": "gpt", "input_shape_index": i}
        if type(item.get("group_id")) is int:
            groups.append(item["group_id"])
    next_group = max(groups, default=0) + 1
    assigned_sam = np.zeros_like(eligible)
    for i, item in enumerate(sam["shapes"]):
        mask = _rasterize(item, width, height) & sam_fill & ~assigned_sam
        if mask.any():
            source = {"stage": "deepseek_sam", "input_shape_index": i}
            element = _mask_shape(item["label"], mask, next_group, "sam_added", source)
            output["shapes"].append(element)
            next_group += 1
            assigned_sam |= mask
    for code, label in enumerate(CLASSES):
        mask = rf_fill & (prediction == code)
        if mask.any():
            source = {"stage": "rf", "model": MODEL_NAME, "sha256_gzip": MODEL_SHA256}
            output["shapes"].append(_mask_shape(label, mask, next_group, "rf_added", source))
            next_group += 1
    stats = {
        "eligible_pixels": int(eligible.sum()), "gpt_pixels": int(((first == 1) & eligible).sum()),
        "sam_pixels": int(sam_fill.sum()), "rf_pixels": int(rf_fill.sum()),
        "conflict_pixels": int(((first > 1) & eligible).sum()),
        "unclassified_pixels": int(((first == 0) & ~sam_fill & ~rf_fill & eligible).sum()),
    }
    return output, stats
