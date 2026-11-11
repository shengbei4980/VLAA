"""Coordinate transforms and final request contracts use offline, deterministic fixtures."""
from dataclasses import replace
from hashlib import sha256
import io
import json
import math

import httpx
from PIL import Image
import pytest

from app.core.errors import ValidationError
from app.core.schemas import BBox
from app.vision.images import ImageView, prepare_views
from app.vision.provider import OpenAICompatibleVisionProvider
from app.vision.schemas import ImageStrategy, NativeProposal, VisionEndpointInternal


FORMATS = ("pixel", "normalized_1", "normalized_1000")
FRAMES = [
    (2301, 1955, (0, 0, 2301, 1955), 1024, 870),
    (6000, 320, (0, 0, 6000, 320), 1024, 55),
    (320, 6000, (0, 0, 320, 6000), 55, 1024),
    (5000, 4000, (1800, 2100, 2800, 3100), 512, 512),
    (2000, 1300, (101, 205, 301, 505), 128, 192),
]


def frame():
    return ImageView("image", b"offline image", 2301, 1955, (0, 0, 2301, 1955), 1024, 870, True)


def encoded_box(pixel_box, coordinate_format, view):
    if coordinate_format == "pixel":
        return pixel_box
    factor = 1 if coordinate_format == "normalized_1" else 1000
    return tuple(value * factor / (view.sent_width if index % 2 == 0 else view.sent_height)
                 for index, value in enumerate(pixel_box))


@pytest.mark.parametrize("dimensions", [(2301, 1955), (5001, 193), (193, 5001)])
def test_original_size_sending_preserves_pixels_source_hash_and_coordinate_mapping(dimensions):
    width, height = dimensions
    source = Image.new("RGB", dimensions, (20, 40, 60))
    source.putpixel((width // 2, height // 2), (240, 80, 10))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    content = buffer.getvalue()
    expected_hash = sha256(content).hexdigest()

    image_hash, views = prepare_views(content, ImageStrategy())

    assert image_hash == expected_hash
    assert buffer.getvalue() == content
    assert sha256(content).hexdigest() == expected_hash
    assert len(views) == 1
    view = views[0]
    assert (view.sent_width, view.sent_height) == dimensions
    assert view.crop == (0, 0, width, height)
    assert view.metadata()["scale_xy"] == [1, 1]
    with Image.open(io.BytesIO(view.content)) as sent:
        assert sent.size == source.size
        assert sent.tobytes() == source.tobytes()
    box = BBox(x=width * .1, y=height * .1, width=width * .2, height=height * .3)
    projected = view.sent_box(box, "pixel")
    assert projected == pytest.approx((box.x, box.y, box.x + box.width, box.y + box.height))
    assert view.original_box(projected, "pixel").model_dump() == pytest.approx(box.model_dump())


@pytest.mark.parametrize("coordinate_format", FORMATS)
@pytest.mark.parametrize("dimensions", FRAMES, ids=["actual-failure-size", "wide", "tall", "tile", "repair-crop"])
def test_every_frame_roundtrips_original_boxes_without_changing_units(coordinate_format, dimensions):
    width, height, crop, sent_width, sent_height = dimensions
    view = ImageView("view", b"", width, height, crop, sent_width, sent_height, crop == (0, 0, width, height))
    left, top, right, bottom = crop
    source = BBox(x=left + (right - left) * .1, y=top + (bottom - top) * .2,
                  width=(right - left) * .6, height=(bottom - top) * .5)
    sent = view.sent_box(source, coordinate_format)
    expected = encoded_box((sent_width * .1, sent_height * .2, sent_width * .7, sent_height * .7), coordinate_format, view)
    assert sent == pytest.approx(expected)
    assert view.original_box(sent, coordinate_format).model_dump() == pytest.approx(source.model_dump())


@pytest.mark.parametrize("coordinate_format", FORMATS)
def test_polygon_vertices_map_through_normalization_and_crop_offsets(coordinate_format):
    view = ImageView("crop", b"", 1000, 1200, (200, 300, 600, 900), 100, 150, False)
    sent_pixels = [(10, 20), (80, 20), (75, 100), (20, 90)]
    if coordinate_format == "pixel":
        polygon = sent_pixels
    else:
        factor = 1 if coordinate_format == "normalized_1" else 1000
        polygon = [(x * factor / 100, y * factor / 150) for x, y in sent_pixels]
    original = view.original_polygon(polygon, coordinate_format)
    for actual, expected in zip(original, [(240, 380), (520, 380), (500, 700), (280, 660)]):
        assert actual == pytest.approx(expected)


@pytest.mark.parametrize("polygon", [
    None, [], [(0, 0), (20, 0)], [(0, 0), (20, 0), (40, 0)],
    [(0, 0), (50, 50), (0, 50), (50, 0)],
    [(0, 0), (100, 0), (20, 80), (80, 80), (0, 40)],
    [(0, 0), (100, 0), (100, 50), (50, 0), (0, 50)],
    [(0, 0), (20, 0), (20, float("inf"))],
    [(0, 0), (20, 0), (20, 900)], [(0, 0), (20, 0), (True, 20)],
    [(0, 0), (20, 0), ("20", 20)], [(0, 0), (20, 0), (20,)],
])
def test_invalid_direct_polygons_are_rejected_with_safe_evidence(polygon):
    with pytest.raises(ValidationError) as error:
        frame().original_polygon(polygon, "pixel")
    assert error.value.error_code == "vision_polygon_invalid"
    assert error.value.details["source_frame"] == "sent_image"
    json.dumps(error.value.details, allow_nan=False)


@pytest.mark.parametrize("coordinate_format", FORMATS)
def test_projection_clips_to_visible_crop_and_omits_objects_outside_it(coordinate_format):
    view = ImageView("tile", b"", 1000, 1200, (200, 300, 600, 900), 100, 150, False)
    projected = view.sent_box(BBox(x=100, y=150, width=500, height=650), coordinate_format)
    assert projected == pytest.approx(encoded_box((0, 0, 100, 125), coordinate_format, view))
    restored = view.original_box(projected, coordinate_format)
    assert restored.model_dump() == pytest.approx({"x": 200, "y": 300, "width": 400, "height": 500})
    assert view.sent_box(BBox(x=0, y=0, width=20, height=20), coordinate_format) is None
    assert view.sent_box(BBox(x=100, y=300, width=100, height=100), coordinate_format) is None


@pytest.mark.parametrize("coordinate_format", FORMATS)
def test_only_microscopic_sent_pixel_edge_drift_is_absorbed(coordinate_format):
    view = frame()
    almost = encoded_box((-5e-7, -5e-7, view.sent_width + 5e-7, view.sent_height + 5e-7), coordinate_format, view)
    assert view.original_box(almost, coordinate_format).model_dump() == {
        "x": 0, "y": 0, "width": 2301, "height": 1955,
    }
    outside = encoded_box((0, 0, view.sent_width + 2e-6, view.sent_height), coordinate_format, view)
    with pytest.raises(ValidationError):
        view.original_box(outside, coordinate_format)


@pytest.mark.parametrize("box", [
    (-1, 0, 100, 100), (0, 0, 1025, 870), (0, 0, 1024, 871),
    (10, 10, 5, 20), (5, 5, 5, 20), (0, 0, math.inf, 10),
    (math.nan, 0, 10, 10), (0, 0, 10, -math.inf),
    (0, 0, 2301, 1955), (0, 0, 10 ** 400, 10),
    (0, 0, 5), (0, 0, 5, 5, 5), (0, "5", 10, 10), None,
])
def test_invalid_boxes_raise_with_json_safe_coordinate_evidence(box):
    view = frame()
    with pytest.raises(ValidationError) as error:
        view.original_box(box, "pixel")
    assert error.value.error_code == "vision_box_out_of_bounds"
    details = error.value.details
    assert details["view_id"] == "image"
    assert details["coordinate_format"] == "pixel"
    assert details["sent_size"] == [1024, 870]
    assert details["original_size"] == [2301, 1955]
    assert details["crop"] == [0, 0, 2301, 1955]
    assert details["expected_bounds"] == [1024, 870]
    assert details["source_frame"] == "sent_image"
    json.dumps(details, allow_nan=False)


@pytest.mark.parametrize("box", [
    BBox(x=-1, y=0, width=10, height=10), BBox(x=2300, y=0, width=2, height=10),
    BBox(x=0, y=1950, width=10, height=10), BBox(x=0, y=0, width=0, height=10),
    BBox(x=0, y=0, width=-10, height=10), BBox(x=math.nan, y=0, width=10, height=10),
    BBox(x=0, y=0, width=math.inf, height=10),
])
def test_invalid_original_boxes_cannot_be_projected(box):
    with pytest.raises(ValidationError) as error:
        frame().sent_box(box, "pixel")
    assert error.value.details["source_frame"] == "original_image"
    json.dumps(error.value.details, allow_nan=False)


@pytest.mark.parametrize("changes", [
    {"sent_width": 0}, {"original_height": -1}, {"sent_height": True},
    {"crop": (0, 0, 2302, 1955)}, {"crop": (0, 0, 0, 100)}, {"crop": (0, -1, 100, 100)},
])
def test_invalid_view_geometry_is_rejected_before_division(changes):
    with pytest.raises(ValidationError) as error:
        replace(frame(), **changes)
    assert error.value.error_code == "vision_image_frame_invalid"


@pytest.mark.parametrize("coordinate_format", FORMATS)
def test_provider_ends_with_an_unambiguous_sent_image_contract(coordinate_format):
    sent = []

    def endpoint(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"objects": []}'}, "finish_reason": "stop"}]})

    config = VisionEndpointInternal(enabled=True, mode="remote", api_base="https://offline.example/v1",
                                    model="test-vision", coordinate_format=coordinate_format, multi_image=True)
    provider = OpenAICompatibleVisionProvider(config, transport=httpx.MockTransport(endpoint))
    view = frame()
    provider.complete("User task asks for original-image coordinates.", [view, replace(view, view_id="overlay")], NativeProposal)
    content = sent[0]["messages"][0]["content"]
    assert content[0]["text"].startswith("User task asks for original-image coordinates.")
    final_contract = content[-1]["text"]
    assert content[-1]["type"] == "text"
    assert "frame=sent_image" in final_contract
    assert "1024 pixels wide and 870 pixels high" in final_contract
    assert f"coordinate_format={coordinate_format}" in final_contract
    bound_x, bound_y = (1024, 870) if coordinate_format == "pixel" else (1, 1) if coordinate_format == "normalized_1" else (1000, 1000)
    assert f"0 <= left < right <= {bound_x}" in final_contract
    assert f"0 <= top < bottom <= {bound_y}" in final_contract
    assert "takes precedence" in final_contract
    assert len([item for item in content if item["type"] == "image_url"]) == 2


@pytest.mark.parametrize("changes", [{"sent_height": 500}, {"crop": (0, 0, 2000, 1900)}, {"original_width": 2400}])
def test_provider_rejects_incompatible_multiview_frames_without_sending(changes):
    def endpoint(request):
        pytest.fail("Ambiguous coordinate frames must not be sent")

    config = VisionEndpointInternal(enabled=True, mode="remote", api_base="https://offline.example/v1",
                                    model="test-vision", multi_image=True)
    provider = OpenAICompatibleVisionProvider(config, transport=httpx.MockTransport(endpoint))
    with pytest.raises(ValidationError) as error:
        provider.complete("Check the image", [frame(), replace(frame(), **changes)], NativeProposal)
    assert error.value.error_code == "vision_image_frames_mismatch"
