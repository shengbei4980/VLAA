"""Local file boundary for the interactive hybrid land-cover workflow."""
from __future__ import annotations

import base64
import binascii
import io
import json
import shutil
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.errors import ValidationError
from app.sam.configuration import freeze_sam_configuration
from app.settings.service import SamSettingsService

from . import core, online

CLASS_COLORS = [(37, 99, 235), (180, 83, 9), (132, 204, 22), (22, 101, 52), (234, 179, 8), (239, 68, 68)]
SOURCE_COLORS = [(37, 99, 235), (245, 158, 11), (22, 163, 74)]
CONFLICT_COLOR = (217, 70, 239)
MIB = 1024 * 1024


def status(settings):
    sam = SamSettingsService(settings).get()
    path = Path(sam.model or "")
    if not path.is_absolute():
        path = settings.project_root / path
    sam_ready = bool(sam.enabled and sam.model and path.is_file())
    rf_ready = core.model_ready() and find_spec("rasterio") is not None
    return {
        "sam_ready": sam_ready, "rf_ready": rf_ready,
        "sam_detail": "Local SAM weights available." if sam_ready else "Select and enable local SAM weights in Settings.",
        "rf_detail": "Panan 2025 ten-band RF available." if rf_ready else "Install backend requirements and the bundled Panan RF weights.",
    }


def read_upload(upload, limit):
    if upload is None:
        raise ValidationError("hybrid_file_required", "A required input file is missing.")
    raw = upload.file.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValidationError("hybrid_file_size", "An input file is empty or exceeds its size limit.")
    return raw


def tile_pixels(raw, *, roi=False):
    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in (("PNG",) if roi else ("PNG", "JPEG")) or image.width * image.height > core.MAX_WINDOW_PIXELS:
                raise ValueError()
            if roi and image.mode not in ("1", "L", "I", "I;16"):
                raise ValueError()
            return np.asarray(image).copy() if roi else np.asarray(image.convert("RGB")).copy()
    except (UnidentifiedImageError, ValueError, OSError, Image.DecompressionBombError):
        raise ValidationError("hybrid_tile_invalid", "Use an RGB PNG/JPEG tile up to 4,194,304 pixels at its native dimensions, and a matching single-channel PNG for the optional ROI.") from None


def checked_draft(upload, source, pixels):
    try:
        doc = json.loads(read_upload(upload, 64 * MIB))
        if (not isinstance(doc, dict) or not isinstance(doc.get("shapes"), list) or len(doc["shapes"]) > 4096
                or not all(isinstance(shape, dict) for shape in doc["shapes"])):
            raise ValueError()
        name = Path(str(doc.get("imagePath", "")).replace("\\", "/")).name
        if name != source["imagePath"] or (doc.get("imageWidth"), doc.get("imageHeight")) != (source["imageWidth"], source["imageHeight"]):
            raise ValueError()
        embedded = doc.get("imageData")
        if embedded and not np.array_equal(tile_pixels(base64.b64decode(embedded, validate=True)), pixels):
            raise ValueError()
    except (ValueError, TypeError, binascii.Error, UnicodeError):
        raise ValidationError("hybrid_draft_mismatch", "The LabelMe draft must match the RGB tile filename, dimensions and embedded pixels.") from None
    return doc


def png_bytes(pixels):
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


def overlay(pixels, codes, palette):
    result = pixels.copy()
    for number, color in [*enumerate(palette), (254, CONFLICT_COLOR)]:
        mask = codes == number
        result[mask] = 0.4 * result[mask] + 0.6 * np.asarray(color)
    return base64.b64encode(png_bytes(result)).decode("ascii")


def render(result, pixels):
    classes = core.class_map(result)
    sources = np.full(classes.shape, 255, dtype=np.uint8)
    for code, stage in enumerate(("gpt", "deepseek_sam", "rf")):
        subset = {**result, "shapes": [s for s in result["shapes"] if s["hybrid_source"]["stage"] == stage]}
        group = core.class_map(subset)
        sources[group < 6] = code
    sources[classes == 254] = 254
    return overlay(pixels, classes, CLASS_COLORS), overlay(pixels, sources, SOURCE_COLORS)


def run(settings, storage, *, mode, rgb, raster, roi, x, y, gpt_draft, sam_draft,
        gpt_endpoint, deepseek_endpoint, approved):
    if mode not in ("online", "drafts") or x < 0 or y < 0:
        raise ValidationError("hybrid_options_invalid", "Choose a workflow mode and non-negative raster offsets.")
    setup = None
    if mode == "online":
        if not approved:
            raise ValidationError("hybrid_remote_approval", "Confirm sending this RGB tile to both chosen endpoints.")
        setup = (online.resolve_endpoint(gpt_endpoint, settings), online.resolve_endpoint(deepseek_endpoint, settings))
        config = freeze_sam_configuration(settings)
        if not config["enabled"] or not config["weights_available"]:
            raise ValidationError("hybrid_sam_unavailable", "Choose a local SAM weight file and enable SAM before running the online recipe.")
    raw = read_upload(rgb, 32 * MIB)
    pixels = tile_pixels(raw)
    height, width = pixels.shape[:2]
    filename = Path((rgb.filename or "").replace("\\", "/")).name
    if not filename or Path(filename).suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValidationError("hybrid_filename", "Use a PNG or JPEG RGB tile filename.")
    source = {"imagePath": filename, "imageWidth": width, "imageHeight": height}
    eligible_roi = tile_pixels(read_upload(roi, 16 * MIB), roi=True) != 0 if roi else None
    if eligible_roi is not None and eligible_roi.shape != (height, width):
        raise ValidationError("hybrid_roi_mismatch", "ROI dimensions must match the RGB tile exactly.")
    if mode == "drafts":
        primary = checked_draft(gpt_draft, source, pixels)
        secondary = checked_draft(sam_draft, source, pixels)
    raster.file.seek(0, 2)
    raster_length = raster.file.tell()
    raster.file.seek(0)
    if not 0 < raster_length <= 512 * MIB:
        raise ValidationError("hybrid_raster_size", "Choose a GeoTIFF of at most 512 MiB.")
    storage.tmp_dir.mkdir(exist_ok=True, parents=True)
    warnings = []
    with TemporaryDirectory(dir=storage.tmp_dir, prefix="land-cover-") as work:
        path = Path(work) / "reflectance.tif"
        with path.open("wb") as target:
            shutil.copyfileobj(raster.file, target, 1024 * 1024)
        # Complete local checks before making any chargeable requests.
        prediction, eligible = core.predict_geotiff(path, x=x, y=y, width=width, height=height, roi=eligible_roi)
        if not eligible.any():
            raise ValidationError("hybrid_no_valid_pixels", "The spectral window and ROI contain no valid pixels.")
        if setup:
            normalized = png_bytes(pixels)
            png = Path(work) / "rgb.png"
            png.write_bytes(normalized)
            primary, secondary, warnings = online.online_drafts(source, png, normalized, *setup, config)
        try:
            result, stats = core.fuse_drafts(primary, secondary, prediction, eligible)
            preview, source_preview = render(result, pixels)
        except (KeyError, ValueError, IndexError, OverflowError, TypeError, OSError):
            raise ValidationError("hybrid_invalid_geometry", "A draft has invalid geometry or an unreadable mask.") from None
    result.update(**source, imageData=base64.b64encode(raw).decode("ascii"))
    result["hybrid_run"] = {"mode": mode, "raster_offset_xy": [x, y], "rgb_size": [width, height],
                            "spectral_bands": list(core.BANDS), "model": core.MODEL_NAME, "human_confirmed": False}
    if setup:
        result["hybrid_run"]["vision_models"] = [endpoint.model for endpoint, protocol in setup]
        result["hybrid_run"]["sam_weights_sha256"] = config["weights_sha256"]
    return {"labelme": result, "stats": stats, "preview_png": preview, "source_png": source_preview, "warnings": warnings}
