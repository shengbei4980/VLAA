"""Model geometry is isolated per entry and every prompt uses its sent image frame."""
import json

import httpx
import pytest

from app.vision.schemas import ImageStrategy, VisionClassRule, VisionEndpointUpdate, VisionSettingsUpdate
from app.vision.service import VisionService
from app.vision.settings import VisionSettingsService
from test_vision import approval, environment, reply


def configure(settings, coordinate_format="pixel"):
    configuration = VisionSettingsService(settings)
    current = configuration.get()
    endpoint = current.proposer.model_dump(exclude={"config_hash", "api_key_configured"})
    endpoint.update(max_image_side=64, coordinate_format=coordinate_format)
    configuration.update(VisionSettingsUpdate(proposer=VisionEndpointUpdate(**endpoint), reviewer=VisionEndpointUpdate(**endpoint)))


@pytest.mark.parametrize("coordinate_format,expected", [
    ("pixel", [8, 8, 20, 20]),
    ("normalized_1", [.125, .2, .3125, .5]),
    ("normalized_1000", [125, 200, 312.5, 500]),
])
def test_review_context_boxes_use_the_sent_frame(environment, coordinate_format, expected):
    settings, storage, path = environment
    configure(settings, coordinate_format)
    sent = []

    def endpoint(request):
        sent.append(json.loads(request.content))
        return reply({"objects": []} if len(sent) == 1 else {"issues": []})

    objects = [{"object_id": "visible", "class_id": "tower", "bbox": {"x": 40, "y": 40, "width": 60, "height": 60}}]
    VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).review(
        "image", path, [VisionClassRule(id="tower", name="tower")], "v1", 1, objects, ImageStrategy(), approval(settings, path, "reviewer"),
    )
    prompt = sent[1]["messages"][0]["content"][0]["text"]
    values, _ = json.JSONDecoder().raw_decode(prompt.split("Existing objects in SENT-image coordinates=", 1)[1])
    assert values[0]["bbox"] == pytest.approx(expected)
    assert "mask_ref" not in values[0] and "polygon" not in values[0]


@pytest.mark.parametrize("bad_box", [[100, 20, 200, 60], [40, 20, 10, 30], [0, 0, float("inf"), 20]])
def test_bad_proposal_geometry_preserves_good_candidates_and_creates_review_issue(environment, bad_box):
    settings, storage, path = environment
    configure(settings)

    def endpoint(request):
        return reply({"objects": [
            {"class_id": "tower", "bbox": [8, 8, 20, 20], "evidence": "Valid visible object"},
            {"class_id": "tower", "bbox": bad_box, "evidence": "Geometry needs human review"},
        ]})

    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).propose(
        "image", path, [VisionClassRule(id="tower", name="tower")], "v1", ImageStrategy(), approval(settings, path),
    )
    assert result.request_count == 1
    assert len(result.objects) == 1
    assert result.objects[0].bbox.model_dump() == {"x": 40, "y": 40, "width": 60, "height": 60}
    issue = next(issue for issue in result.issues if issue["type"] == "coordinate_error")
    assert issue["blocking"] is False and issue["severity"] == "warning"
    assert issue["region"] == {"x": 0, "y": 0, "width": 320, "height": 200}
    assert issue["coordinate_details"]["phase"] == "propose"
    json.dumps(result.model_dump(), allow_nan=False)


def test_bad_repair_retains_the_original_object_and_audits_the_problem(environment):
    settings, storage, path = environment
    configure(settings)
    prompts = []

    def endpoint(request):
        prompts.append(json.loads(request.content)["messages"][0]["content"][0]["text"])
        return reply({"objects": [{"class_id": "tower", "bbox": [0, 0, 9999, 9999], "evidence": "Invalid replacement"}]})

    objects = [{"object_id": "stable", "class_id": "tower", "bbox": {"x": 40, "y": 40, "width": 20, "height": 20}}]
    issues = [{"type": "boundary", "object_id": "stable", "bbox": objects[0]["bbox"], "evidence": "Check the visible boundary"}]
    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).repair(
        "image", path, [VisionClassRule(id="tower", name="tower")], "v1", 1, objects, issues, ImageStrategy(), approval(settings, path),
    )
    assert result["objects"][0]["object_id"] == "stable"
    assert result["objects"][0]["bbox"] == objects[0]["bbox"]
    assert result["repair_diff"][0]["applied"] is False
    assert result["request_count"] == 1
    assert any(issue["type"] == "coordinate_error" for issue in result["issues"])
    assert '"bbox": [8.0, 8.0, 28.0, 28.0]' in prompts[0]
