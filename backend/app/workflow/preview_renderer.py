"""Render persisted review candidates as original-resolution PNG previews."""
from __future__ import annotations

from hashlib import sha256
import io
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from pydantic import ValidationError as SchemaError

from app.core.errors import ConflictError, ValidationError
from app.core.schemas import BBox
from app.core.storage import Storage
from app.dataset.annotation.geometry import validate_bbox, validate_polygon
from app.dataset.annotation.masks import MAX_MASK_PIXELS, MaskService, decode_mask


def _font(size: int):
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def render_preview(storage: Storage, image_path: Path, payload: dict, class_rules: list[dict]) -> bytes:
    width, height = payload["width"], payload["height"]
    if type(width) is not int or type(height) is not int or min(width, height) < 1 or width * height > MAX_MASK_PIXELS:
        raise ValidationError("workflow_preview_dimensions_invalid", "预览图尺寸超出支持范围。")
    try:
        content = image_path.read_bytes()
        if payload.get("image_hash") and sha256(content).hexdigest() != payload["image_hash"]:
            raise ConflictError("workflow_preview_image_changed", "原图已改变，请重新检查当前图片后导出。")
        with Image.open(io.BytesIO(content)) as original:
            if original.size != (width, height):
                raise ValidationError("workflow_preview_dimensions_mismatch", "原图尺寸与当前复核记录不一致。")
            image = original.convert("RGBA")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValidationError("workflow_preview_image_unavailable", "原图无法完整读取，请检查原图后重试。") from error

    classes = {rule.get("class_id", rule.get("id")): rule for rule in class_rules}
    masks = MaskService(storage)
    font_size = max(14, min(96, round(max(width, height) / 75)))
    font = _font(font_size)
    line_width = max(2, round(max(width, height) / 500))
    objects = payload.get("objects", payload.get("annotations", []))
    if not isinstance(objects, list) or len(objects) > 10000 or any(not isinstance(item, dict) for item in objects):
        raise ValidationError("workflow_invalid_annotations", "标注候选必须为不超过 10000 项的对象列表。")

    for index, candidate in enumerate(objects, 1):
        if candidate.get("decision") == "rejected":
            continue
        rule = classes.get(candidate.get("class_id"))
        color = rule.get("color", "") if rule else "#f59e0b"
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            color = "#22c55e"
        rgb = ImageColor.getrgb(color)
        anchor = (8, 8 + index * (font_size + 8))
        mask = None
        if candidate.get("mask") is not None:
            mask = decode_mask(candidate["mask"])
        elif candidate.get("mask_ref"):
            mask = masks.decode(candidate["mask_ref"])
        draw = ImageDraw.Draw(image)
        if mask is not None:
            if mask.shape != (height, width) or not mask.any():
                raise ValidationError("mask_geometry_invalid", "对象掩膜必须与原图尺寸一致并包含目标像素。")
            alpha = Image.fromarray(mask.astype(np.uint8) * 102)
            overlay = Image.new("RGBA", image.size, rgb + (0,))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
            first = int(mask.argmax())
            anchor = (first % width, first // width)
            draw = ImageDraw.Draw(image)
        elif candidate.get("type") == "polygon" and candidate.get("polygon"):
            points = [tuple(point) for point in candidate["polygon"]]
            validate_polygon(points, width, height)
            overlay = Image.new("RGBA", image.size)
            ImageDraw.Draw(overlay).polygon(points, fill=rgb + (64,), outline=rgb + (255,), width=line_width)
            image = Image.alpha_composite(image, overlay)
            anchor = (min(point[0] for point in points), min(point[1] for point in points))
            draw = ImageDraw.Draw(image)
        elif candidate.get("bbox"):
            try:
                box = BBox.model_validate(candidate["bbox"])
                validate_bbox(box, width, height)
            except SchemaError as error:
                raise ValidationError("workflow_preview_geometry_invalid", "候选框格式无效，请修正对应对象后导出。") from error
            draw.rectangle((box.x, box.y, box.x + box.width, box.y + box.height), outline=rgb, width=line_width)
            anchor = (box.x, box.y)
        else:
            raise ValidationError("workflow_preview_geometry_invalid", "候选缺少可绘制的目标框或分割轮廓。")
        label = f"#{index} {rule['name'] if rule else '待确定类别'}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0] + 8
        text_height = text_box[3] - text_box[1] + 8
        x = max(0, min(anchor[0], width - text_width))
        y = max(0, min(anchor[1] - text_height, height - text_height))
        draw.rectangle((x, y, x + text_width, y + text_height), fill=(15, 23, 42, 230))
        draw.text((x + 4 - text_box[0], y + 4 - text_box[1]), label, fill=rgb, font=font)

    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
