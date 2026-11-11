from __future__ import annotations

import io
import math
from dataclasses import dataclass
from hashlib import sha256

from PIL import Image

from app.core.errors import ValidationError
from app.core.schemas import BBox
from app.dataset.annotation.geometry import validate_polygon
from app.vision.schemas import CoordinateFormat, ImageStrategy, MAX_VISION_POLYGON_POINTS, VisionObject


def _finite_coordinate(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return ((ab_c <= 0 <= ab_d or ab_d <= 0 <= ab_c)
            and (cd_a <= 0 <= cd_b or cd_b <= 0 <= cd_a))


def _simple_polygon(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    if len(set(points)) != count:
        return False
    for index, b in enumerate(points):
        a, c = points[index - 1], points[(index + 1) % count]
        if _cross(a, b, c) == 0 and (a[0] - b[0]) * (c[0] - b[0]) + (a[1] - b[1]) * (c[1] - b[1]) > 0:
            return False
    for index in range(count):
        a, b = points[index], points[(index + 1) % count]
        for other in range(index + 2, count):
            if index == 0 and other == count - 1:
                continue
            if _intersects(a, b, points[other], points[(other + 1) % count]):
                return False
    return True


@dataclass(frozen=True)
class ImageView:
    view_id: str
    content: bytes
    original_width: int
    original_height: int
    crop: tuple[int, int, int, int]
    sent_width: int
    sent_height: int
    is_overview: bool

    def __post_init__(self) -> None:
        sizes = (self.original_width, self.original_height, self.sent_width, self.sent_height)
        if any(type(value) is not int or value <= 0 for value in sizes):
            raise ValidationError("vision_image_frame_invalid", "Image coordinate frames require positive integer dimensions.")
        if (len(self.crop) != 4 or any(type(value) is not int for value in self.crop)
                or not 0 <= self.crop[0] < self.crop[2] <= self.original_width
                or not 0 <= self.crop[1] < self.crop[3] <= self.original_height):
            raise ValidationError("vision_image_frame_invalid", "Image crops must be positive regions inside the original image.")

    def metadata(self) -> dict:
        x1, y1, x2, y2 = self.crop
        return {
            "view_id": self.view_id, "crop_xyxy": list(self.crop),
            "original_size": [self.original_width, self.original_height],
            "sent_size": [self.sent_width, self.sent_height],
            "scale_xy": [self.sent_width / (x2 - x1), self.sent_height / (y2 - y1)],
            "padding_ltrb": [0, 0, 0, 0], "is_overview": self.is_overview,
        }

    def _box_error(self, box, coordinate_format: CoordinateFormat, source_frame: str = "sent_image") -> ValidationError:
        native = [value if _finite_coordinate(value) else str(value)[:256]
                  for value in box] if isinstance(box, (list, tuple)) else str(box)[:1024]
        limits = {"pixel": [self.sent_width, self.sent_height], "normalized_1": [1, 1], "normalized_1000": [1000, 1000]}
        expected_bounds = ([self.original_width, self.original_height] if source_frame == "original_image"
                           else limits.get(coordinate_format))
        return ValidationError("vision_box_out_of_bounds", "Vision bounding box does not fit the declared sent-image coordinates.",
            details={"native_bbox": native, "view_id": self.view_id, "coordinate_format": coordinate_format,
                     "sent_size": [self.sent_width, self.sent_height],
                     "original_size": [self.original_width, self.original_height], "crop": list(self.crop),
                     "expected_bounds": expected_bounds, "source_frame": source_frame})

    def original_box(self, box: tuple[float, float, float, float], coordinate_format: CoordinateFormat) -> BBox:
        if (coordinate_format not in ("pixel", "normalized_1", "normalized_1000")
                or not isinstance(box, (list, tuple)) or len(box) != 4
                or any(not _finite_coordinate(value) for value in box)):
            raise self._box_error(box, coordinate_format)
        x1, y1, x2, y2 = box
        if coordinate_format != "pixel":
            divisor = 1 if coordinate_format == "normalized_1" else 1000
            x1, x2 = x1 * self.sent_width / divisor, x2 * self.sent_width / divisor
            y1, y2 = y1 * self.sent_height / divisor, y2 * self.sent_height / divisor
        tolerance = 1e-6
        if (not all(math.isfinite(value) for value in (x1, y1, x2, y2))
                or x1 < -tolerance or y1 < -tolerance
                or x2 > self.sent_width + tolerance or y2 > self.sent_height + tolerance
                or x2 <= x1 or y2 <= y1):
            raise self._box_error(box, coordinate_format)
        # Only absorb sub-pixel floating-point drift at known image edges.
        x1, x2 = max(0.0, min(x1, self.sent_width)), max(0.0, min(x2, self.sent_width))
        y1, y2 = max(0.0, min(y1, self.sent_height)), max(0.0, min(y2, self.sent_height))
        if x2 <= x1 or y2 <= y1:
            raise self._box_error(box, coordinate_format)
        left, top, right, bottom = self.crop
        scale_x = (right - left) / self.sent_width
        scale_y = (bottom - top) / self.sent_height
        return BBox(x=left + x1 * scale_x, y=top + y1 * scale_y, width=(x2 - x1) * scale_x, height=(y2 - y1) * scale_y)

    def sent_box(self, box: BBox, coordinate_format: CoordinateFormat) -> tuple[float, float, float, float] | None:
        x1, y1, x2, y2 = box.x, box.y, box.x + box.width, box.y + box.height
        native = (x1, y1, x2, y2)
        left, top, right, bottom = self.crop
        scale_x, scale_y = self.sent_width / (right - left), self.sent_height / (bottom - top)
        tolerance_x, tolerance_y = 1e-6 / scale_x, 1e-6 / scale_y
        if (coordinate_format not in ("pixel", "normalized_1", "normalized_1000")
                or not all(math.isfinite(value) for value in native)
                or x1 < -tolerance_x or y1 < -tolerance_y
                or x2 > self.original_width + tolerance_x or y2 > self.original_height + tolerance_y
                or x2 <= x1 or y2 <= y1):
            raise self._box_error(native, coordinate_format, "original_image")
        x1, y1, x2, y2 = max(left, x1), max(top, y1), min(right, x2), min(bottom, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        x1, x2 = (x1 - left) * scale_x, (x2 - left) * scale_x
        y1, y2 = (y1 - top) * scale_y, (y2 - top) * scale_y
        if coordinate_format != "pixel":
            factor = 1 if coordinate_format == "normalized_1" else 1000
            x1, x2 = x1 * factor / self.sent_width, x2 * factor / self.sent_width
            y1, y2 = y1 * factor / self.sent_height, y2 * factor / self.sent_height
        return x1, y1, x2, y2

    def original_polygon(self, points, coordinate_format: CoordinateFormat) -> list[tuple[float, float]]:
        def invalid(reason: str) -> ValidationError:
            native = [
                [value if _finite_coordinate(value) else str(value)[:128] for value in point[:3]]
                if isinstance(point, (list, tuple)) else str(point)[:256]
                for point in points[:32]
            ] if isinstance(points, (list, tuple)) else str(points)[:256]
            return ValidationError("vision_polygon_invalid", "Vision polygon is missing, malformed, out of bounds or self-intersecting.",
                details={"native_polygon": native, "point_count": len(points) if isinstance(points, (list, tuple)) else 0,
                         "reason": reason, "view_id": self.view_id, "coordinate_format": coordinate_format,
                         "sent_size": [self.sent_width, self.sent_height],
                         "original_size": [self.original_width, self.original_height], "crop": list(self.crop),
                         "source_frame": "sent_image"})

        if (coordinate_format not in ("pixel", "normalized_1", "normalized_1000")
                or not isinstance(points, (list, tuple)) or not 3 <= len(points) <= MAX_VISION_POLYGON_POINTS
                or any(not isinstance(point, (list, tuple)) or len(point) != 2
                       or any(not _finite_coordinate(value) for value in point) for point in points)):
            raise invalid("invalid_vertices")
        factor_x, factor_y = 1.0, 1.0
        if coordinate_format != "pixel":
            divisor = 1 if coordinate_format == "normalized_1" else 1000
            factor_x, factor_y = self.sent_width / divisor, self.sent_height / divisor
        sent = [(x * factor_x, y * factor_y) for x, y in points]
        tolerance = 1e-6
        if any(not all(math.isfinite(value) for value in point)
               or x < -tolerance or y < -tolerance
               or x > self.sent_width + tolerance or y > self.sent_height + tolerance
               for point in sent for x, y in [point]):
            raise invalid("out_of_bounds")
        sent = [(max(0.0, min(x, self.sent_width)), max(0.0, min(y, self.sent_height))) for x, y in sent]
        if sent[0] == sent[-1]:
            sent.pop()
        if len(sent) < 3 or not _simple_polygon(sent):
            raise invalid("degenerate_or_self_intersecting")
        left, top, right, bottom = self.crop
        original = [(left + x * (right - left) / self.sent_width,
                     top + y * (bottom - top) / self.sent_height) for x, y in sent]
        try:
            validate_polygon(original, self.original_width, self.original_height)
        except ValidationError as error:
            raise invalid(error.error_code) from error
        return original

    def sent_polygon(self, points: list[tuple[float, float]], coordinate_format: CoordinateFormat) -> list[tuple[float, float]] | None:
        """Project only complete contours; a crop must not invent closing edges."""
        left, top, right, bottom = self.crop
        if any(x < left or y < top or x > right or y > bottom for x, y in points):
            return None
        factor = 1 if coordinate_format == "normalized_1" else 1000
        if coordinate_format == "pixel":
            return [((x - left) * self.sent_width / (right - left), (y - top) * self.sent_height / (bottom - top)) for x, y in points]
        return [((x - left) * factor / (right - left), (y - top) * factor / (bottom - top)) for x, y in points]


def prepare_views(content: bytes, strategy: ImageStrategy) -> tuple[str, list[ImageView]]:
    try:
        with Image.open(io.BytesIO(content)) as source:
            width, height = source.size
            if width * height > 100_000_000:
                raise ValidationError("vision_image_too_large", "Vision image exceeds the supported pixel limit.")
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValidationError("vision_image_invalid", "Vision source is not a supported decoded image.") from exc
    crops = []
    if strategy.include_overview or strategy.tile_size is None:
        crops.append(((0, 0, width, height), True))
    if strategy.tile_size is not None:
        tile = strategy.tile_size
        step = max(1, round(tile * (1 - strategy.overlap)))
        xs = sorted(set([*range(0, max(1, width - tile + 1), step), max(0, width - tile)]))
        ys = sorted(set([*range(0, max(1, height - tile + 1), step), max(0, height - tile)]))
        if len(xs) * len(ys) + len(crops) > strategy.max_views:
            raise ValidationError("vision_view_budget_exceeded", "Image tiling exceeds the approved view limit.")
        crops.extend(((x, y, min(width, x + tile), min(height, y + tile)), False) for y in ys for x in xs)
    unique = {}
    for crop, overview in crops:
        unique.setdefault(crop, overview)
    views = []
    for index, (crop, overview) in enumerate(unique.items()):
        view = image.crop(crop)
        output = io.BytesIO()
        view.save(output, format="PNG")
        views.append(ImageView(f"view_{index}", output.getvalue(), width, height, crop, view.width, view.height, overview))
    return sha256(content).hexdigest(), views


def box_iou(left: BBox, right: BBox) -> float:
    width = max(0.0, min(left.x + left.width, right.x + right.width) - max(left.x, right.x))
    height = max(0.0, min(left.y + left.height, right.y + right.height) - max(left.y, right.y))
    intersection = width * height
    return intersection / (left.width * left.height + right.width * right.height - intersection)


def merge_candidates(objects: list[VisionObject], threshold: float) -> list[VisionObject]:
    def contour_key(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
        first = min(range(len(points)), key=points.__getitem__)
        forward = tuple(points[first:] + points[:first])
        reverse = (forward[0], *reversed(forward[1:]))
        return min(forward, reverse)

    merged: list[VisionObject] = []
    for candidate in objects:
        duplicate = next((item for item in merged if item.class_id == candidate.class_id and (
            bool(item.polygon) and contour_key(item.polygon) == contour_key(candidate.polygon)
            if candidate.polygon else not item.polygon and box_iou(item.bbox, candidate.bbox) >= threshold)), None)
        if duplicate is None:
            merged.append(candidate)
        else:
            duplicate.provenance_json.setdefault("merged_sources", []).append(candidate.model_dump())
    return merged
