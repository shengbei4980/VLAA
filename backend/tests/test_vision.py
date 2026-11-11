from __future__ import annotations

import base64
import io
import json
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import ValidationError
from app.core.storage import Storage
from app.dataset.annotation.masks import MaskService, decode_mask, encode_mask
from app.sam.schemas import SamPredictRequest
from app.sam.service import SamService
from app.vision.images import ImageView, prepare_views
from app.vision.schemas import (
    ImageStrategy, VisionApproval, VisionClassRule, VisionEndpointUpdate,
    VisionSettingsUpdate,
)
from app.vision.service import VisionService
from app.vision.settings import VisionSettingsService


@pytest.fixture
def environment(tmp_path):
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data", import_root=tmp_path / "imports", database_url="sqlite://")
    storage = Storage(settings.data_dir, settings.import_root)
    endpoint = VisionEndpointUpdate(enabled=True, mode="remote", api_base="https://chosen.example/v1", model="explicit-vision-model", api_key="private-secret")
    VisionSettingsService(settings).update(VisionSettingsUpdate(proposer=endpoint, reviewer=endpoint))
    source = io.BytesIO()
    Image.new("RGB", (320, 200), "white").save(source, format="PNG")
    path = storage.image_path("dataset", "source.png")
    path.write_bytes(source.getvalue())
    return settings, storage, path


def approval(settings, path, role="proposer", max_requests=10):
    endpoint = VisionSettingsService(settings).internal(role)
    return VisionApproval(mode=endpoint.mode, api_base=endpoint.api_base, model=endpoint.model, config_hash=endpoint.config_hash,
                          allowed_image_hashes=[sha256(path.read_bytes()).hexdigest()], max_requests=max_requests)


def reply(value, status=200):
    return httpx.Response(status, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}], "usage": {"total_tokens": 10}})


@pytest.mark.parametrize("seed", range(5))
def test_rle_roundtrip_preserves_holes_components_and_random_masks(tmp_path, seed):
    mask = np.random.default_rng(seed).integers(0, 2, (40, 60)).astype(bool)
    mask[3:20, 3:20] = True
    mask[7:15, 7:15] = False
    mask[30:35, 50:58] = True
    service = MaskService(Storage(tmp_path / "data", tmp_path / "imports"))
    result = service.write(mask)
    assert np.array_equal(mask, decode_mask(encode_mask(mask)))
    assert np.array_equal(mask, service.decode(result.mask_ref))
    assert result.area == mask.sum()
    assert service.write(service.read(result.mask_ref)) == result
    assert not service.decode(result.mask_ref)[10, 10]


@pytest.mark.parametrize("rle", [
    {"size": [10, 10], "counts": [0, 99]},
    {"size": [10, 10], "counts": [-1, 101]},
    {"size": [10, 10], "counts": "~~~~"},
    {"size": [0, 10], "counts": [0]},
    {"size": [100000, 100000], "counts": [10000000000]},
])
def test_invalid_rle_is_rejected_without_native_decoder(rle):
    with pytest.raises(ValidationError):
        decode_mask(rle)


def test_mask_storage_rejects_paths_and_tampering(tmp_path):
    service = MaskService(Storage(tmp_path, tmp_path / "imports"))
    with pytest.raises(ValidationError):
        service.path("masks/../../settings.json")
    result = service.write(np.ones((4, 5), dtype=bool))
    service.path(result.mask_ref).write_text("{}")
    with pytest.raises(ValidationError, match="hash"):
        service.read(result.mask_ref)


@pytest.mark.parametrize("coordinate_format,box", [
    ("pixel", (10, 20, 50, 60)),
    ("normalized_1", (.1, .2, .5, .6)),
    ("normalized_1000", (100, 200, 500, 600)),
])
def test_crop_resize_coordinates_return_to_original(coordinate_format, box):
    view = ImageView("crop", b"", 1000, 2000, (200, 400, 600, 800), 100, 100, False)
    mapped = view.original_box(box, coordinate_format)
    assert mapped.model_dump() == {"x": 240, "y": 480, "width": 160, "height": 160}
    with pytest.raises(ValidationError):
        view.original_box((-1, 0, 20, 20), "pixel")


def test_long_image_tiling_has_edge_coverage_and_bounded_views():
    image = io.BytesIO()
    Image.new("RGB", (120, 3000)).save(image, format="PNG")
    _, views = prepare_views(image.getvalue(), ImageStrategy(tile_size=512, overlap=.2))
    assert (views[0].sent_width, views[0].sent_height) == (120, 3000)
    for view in views:
        left, top, right, bottom = view.crop
        assert (view.sent_width, view.sent_height) == (right - left, bottom - top)
        assert view.metadata()["scale_xy"] == [1, 1]
    assert views[-1].crop[-1] == 3000
    assert any(view.crop[1] == 0 and not view.is_overview for view in views)
    with pytest.raises(ValidationError, match="limit"):
        prepare_views(image.getvalue(), ImageStrategy(tile_size=64, max_views=2))


def test_settings_keep_secrets_out_of_public_contract_and_preserve_agent_settings(environment):
    settings, _, _ = environment
    service = VisionSettingsService(settings)
    raw = service.settings._read()
    raw["llm"] = {"model": "existing-text-model"}
    service.settings._write(raw)
    before = service.get()
    assert "private-secret" not in before.model_dump_json()
    payload = VisionSettingsUpdate(proposer=VisionEndpointUpdate(**before.proposer.model_dump(exclude={"config_hash", "api_key_configured"})),
                                   reviewer=VisionEndpointUpdate(**before.reviewer.model_dump(exclude={"config_hash", "api_key_configured"})))
    service.update(payload)
    assert service.internal("proposer").api_key.get_secret_value() == "private-secret"
    assert service.settings._read()["llm"]["model"] == "existing-text-model"


@pytest.mark.parametrize("old_limit", [64, 1024])
def test_deprecated_image_limit_is_ignored_without_losing_credentials(environment, old_limit):
    settings, _, _ = environment
    service = VisionSettingsService(settings)
    raw = service.settings._read()
    for role in ("proposer", "reviewer"):
        raw["vision"][role]["max_image_side"] = old_limit
    service.settings._write(raw)
    public = service.get()
    assert "max_image_side" not in public.model_dump_json()
    assert "max_image_side" not in VisionEndpointUpdate.model_json_schema()["properties"]
    values = public.proposer.model_dump(exclude={"api_key_configured", "config_hash"})
    values["max_image_side"] = old_limit
    service.update(VisionSettingsUpdate(proposer=VisionEndpointUpdate(**values), reviewer=VisionEndpointUpdate(**values)))
    for role in ("proposer", "reviewer"):
        assert "max_image_side" not in service.settings._read()["vision"][role]
        assert service.internal(role).api_key.get_secret_value() == "private-secret"


@pytest.mark.parametrize("mode", ["remote", "local"])
def test_default_and_legacy_endpoints_send_native_pixels_for_propose_and_review(environment, mode):
    settings, storage, path = environment
    source = Image.new("RGB", (2301, 1955), (20, 40, 60))
    source.putpixel((1200, 900), (240, 80, 10))
    source.save(path, format="PNG")
    original = path.read_bytes()
    configuration = VisionSettingsService(settings)
    raw = configuration.settings._read()
    for role in ("proposer", "reviewer"):
        raw["vision"][role].update(mode=mode, multi_image=True, max_image_side=64)
    configuration.settings._write(raw)
    calls = []

    def endpoint(request):
        data = json.loads(request.content)
        content = data["messages"][0]["content"]
        images = [part for part in content if part["type"] == "image_url"]
        for index, part in enumerate(images):
            decoded = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            with Image.open(io.BytesIO(decoded)) as sent:
                assert sent.size == source.size
                if index == 0:
                    assert sent.tobytes() == source.tobytes()
        assert "2301 pixels wide and 1955 pixels high" in content[-1]["text"]
        calls.append(data)
        if len(calls) == 3:
            return reply({"issues": [{"type": "boundary", "bbox": [120, 80, 1800, 1800], "evidence": "Review boundary."}]})
        return reply({"objects": [{"class_id": "tower", "bbox": [120, 80, 1800, 1800], "evidence": "Visible tower."}]})

    service = VisionService(storage, settings, transport=httpx.MockTransport(endpoint))
    rules = [VisionClassRule(id="tower", name="Tower")]
    proposed = service.propose("image", path, rules, "v1", ImageStrategy(), approval(settings, path))
    expected = {"x": 120, "y": 80, "width": 1680, "height": 1720}
    assert proposed.objects[0].bbox.model_dump() == expected
    reviewed = service.review("image", path, rules, "v1", 1,
        [item.model_dump() for item in proposed.objects], ImageStrategy(), approval(settings, path, "reviewer"))
    assert reviewed.issues[-1]["bbox"] == expected
    assert all(call["view"]["scale_xy"] == [1, 1] for call in reviewed.provenance_json["calls"])
    assert len(calls) == 3
    assert path.read_bytes() == original


def test_proposal_sends_image_and_maps_native_coordinates(environment):
    settings, storage, path = environment
    calls = []
    def endpoint(request):
        data = json.loads(request.content)
        calls.append(data)
        assert request.headers["authorization"] == "Bearer private-secret"
        assert data["messages"][0]["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")
        return reply({"objects": [{"class_id": "tower", "bbox": [10, 20, 100, 150], "evidence": "Visible tower."}]})
    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).propose(
        "image", path, [VisionClassRule(id="tower", name="tower")], "v1", ImageStrategy(), approval(settings, path),
    )
    assert result.request_count == len(calls) == 1
    assert result.objects[0].bbox.width == 90
    assert "private-secret" not in result.model_dump_json()


def test_approval_hash_destination_and_budget_reject_before_network(environment):
    settings, storage, path = environment
    def endpoint(request):
        pytest.fail("Unapproved request reached the network transport")
    service = VisionService(storage, settings, transport=httpx.MockTransport(endpoint))
    accepted = approval(settings, path)
    for rejected in (accepted.model_copy(update={"model": "different"}), accepted.model_copy(update={"allowed_image_hashes": ["unknown"]})):
        with pytest.raises(ValidationError):
            service.propose("image", path, [VisionClassRule(id="a", name="a")], "v1", ImageStrategy(), rejected)
    with pytest.raises(ValidationError, match="budget"):
        service.review("image", path, [VisionClassRule(id="a", name="a")], "v1", 1, [], ImageStrategy(), approval(settings, path, "reviewer", 1))


def test_empty_proposals_still_get_independent_inventory_and_comparison(environment):
    settings, storage, path = environment
    calls = []
    def endpoint(request):
        data = json.loads(request.content)
        calls.append(data)
        if len(calls) == 1:
            assert "Existing objects" not in data["messages"][0]["content"][0]["text"]
            return reply({"objects": [{"class_id": "tower", "bbox": [10, 20, 100, 150], "evidence": "An omitted tower."}]})
        return reply({"issues": []})
    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).review(
        "image", path, [VisionClassRule(id="tower", name="tower")], "v1", 4, [], ImageStrategy(), approval(settings, path, "reviewer"),
    )
    assert result.request_count == 2
    assert result.revision == 4
    assert any(issue["type"] == "missing" and issue["object_id"] is None for issue in result.issues)


@pytest.mark.parametrize("mode", ["remote", "local"])
def test_synthetic_probe_reads_real_sent_pixels_and_checks_crop_scaling(environment, mode):
    settings, storage, _ = environment
    configuration = VisionSettingsService(settings)
    current = configuration.get()
    endpoint_values = current.proposer.model_dump(exclude={"config_hash", "api_key_configured"})
    endpoint_values["mode"] = mode
    configuration.update(VisionSettingsUpdate(proposer=VisionEndpointUpdate(**endpoint_values), reviewer=VisionEndpointUpdate(**endpoint_values)))
    calls = []
    def endpoint(request):
        calls.append(request)
        data = json.loads(request.content)
        content = data["messages"][0]["content"]
        raw = base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1])
        array = np.asarray(Image.open(io.BytesIO(raw)))
        metadata = json.loads(content[1]["text"])
        left, top, right, bottom = metadata["crop_xyxy"]
        assert metadata["scale_xy"] == [1, 1]
        assert array.shape[:2] == (bottom - top, right - left)
        ys, xs = np.where((array[:, :, 0] > 200) & (array[:, :, 1] < 60))
        return reply({"objects": [{"class_id": "red_rectangle", "bbox": [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)], "evidence": "Red region read from image."}]})
    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).probe("proposer")
    assert result.ok and result.coordinate_verified
    assert len(calls) == 2
    assert not result.real_image_quality_verified
    assert result.mode == mode


def test_sam_raw_mask_retains_holes_and_is_not_a_box_fallback(environment, monkeypatch):
    settings, storage, path = environment
    image = SimpleNamespace(id="image", dataset_id="dataset", storage_name=path.name, width=320, height=200)
    class_label = SimpleNamespace(id="class", dataset_id="dataset")
    dataset = SimpleNamespace(task_type="segment")
    session = SimpleNamespace(get=lambda model, key: dataset if key == "dataset" else class_label)
    monkeypatch.setattr("app.sam.service.get_image", lambda session, image_id: image)
    mask = np.zeros((1, 200, 320), dtype=bool)
    mask[0, 20:100, 20:100] = True
    mask[0, 40:70, 40:70] = False
    mask[0, 130:140, 200:220] = True
    result = SimpleNamespace(masks=SimpleNamespace(data=mask, xy=[]), boxes=None)
    monkeypatch.setattr("app.sam.service._model_for", lambda model, device: SimpleNamespace(predict=lambda **kwargs: [result]))
    service = SamService(storage, replace(settings, sam_model="existing.pt"))
    response = service.predict_mask(session, SamPredictRequest(image_id="image", class_id="class", box={"x": 10, "y": 10, "width": 200, "height": 150}))
    assert np.array_equal(MaskService(storage).decode(response.mask_ref), mask[0])
    assert response.score is None
    with pytest.raises(ValidationError, match="real configured"):
        SamService(storage, settings).predict_mask(session, SamPredictRequest(image_id="image", class_id="class", box={"x": 10, "y": 10, "width": 20, "height": 20}))


@pytest.mark.parametrize("has_replacement", [False, True])
def test_targeted_repair_only_sends_local_crop_preserves_identity_and_keeps_diff(environment, has_replacement):
    settings, storage, path = environment
    sent = []
    def endpoint(request):
        data = json.loads(request.content)
        metadata = json.loads(data["messages"][0]["content"][1]["text"])
        sent.append(metadata)
        assert metadata["crop_xyxy"] != [0, 0, 320, 200]
        return reply({"objects": [{"class_id": "tower", "bbox": [8, 8, 28, 28], "evidence": "Corrected visible boundary."}] if has_replacement else []})
    objects = [{"object_id": "stable-object", "class_id": "tower", "type": "bbox", "bbox": {"x": 40, "y": 40, "width": 20, "height": 20}, "source": "manual"}]
    issues = [{"type": "boundary", "object_id": "stable-object", "evidence": "Boundary includes nearby content."}]
    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).repair(
        "image", path, [VisionClassRule(id="tower", name="tower")], "v1", 5, objects, issues, ImageStrategy(), approval(settings, path),
    )
    assert len(sent) == result["request_count"] == 1
    assert result["objects"][0]["object_id"] == "stable-object"
    assert result["repair_diff"][0]["applied"] is has_replacement


def test_large_repair_crop_keeps_native_resolution_and_maps_crop_offset(environment):
    settings, storage, path = environment
    source = Image.new("RGB", (2301, 1955), (20, 40, 60))
    source.save(path, format="PNG")
    configuration = VisionSettingsService(settings)
    raw = configuration.settings._read()
    raw["vision"]["proposer"]["max_image_side"] = 64
    configuration.settings._write(raw)
    sent_metadata = []

    def endpoint(request):
        content = json.loads(request.content)["messages"][0]["content"]
        metadata = json.loads(content[1]["text"])
        left, top, right, bottom = metadata["crop_xyxy"]
        assert metadata["scale_xy"] == [1, 1]
        assert right - left > 1024 and bottom - top > 1024
        with Image.open(io.BytesIO(base64.b64decode(content[2]["image_url"]["url"].split(",", 1)[1]))) as sent:
            assert sent.size == (right - left, bottom - top)
            assert sent.tobytes() == source.crop((left, top, right, bottom)).tobytes()
        sent_metadata.append(metadata)
        return reply({"objects": [{"class_id": "tower", "bbox": [200, 200, 800, 900], "evidence": "Corrected boundary."}]})

    objects = [{"object_id": "tower", "class_id": "tower", "bbox": {"x": 500, "y": 400, "width": 1100, "height": 1000}}]
    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).repair(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", 1, objects,
        [{"type": "boundary", "object_id": "tower"}], ImageStrategy(), approval(settings, path))
    left, top, _, _ = sent_metadata[0]["crop_xyxy"]
    assert result["repair_diff"][0]["applied"] is True
    assert result["objects"][0]["bbox"] == {"x": left + 200, "y": top + 200, "width": 600, "height": 700}
    assert result["repair_diff"][0]["before"][0]["bbox"] == objects[0]["bbox"]
    assert result["provenance_json"]["requires_human_review"]


@pytest.mark.parametrize("same_dataset", [False, True])
def test_managed_next_batch_inference_keeps_raw_holes_and_components(environment, monkeypatch, same_dataset):
    import torch
    from app.core.models import ImageItem
    from app.vision.workflow import build_vision_handlers

    settings, storage, path = environment
    weight = storage.model_version_dir("managed") / "best.pt"
    weight.write_bytes(b"offline test substitute")
    image = SimpleNamespace(id="image", dataset_id="dataset", storage_name=path.name, width=320, height=200,
                            annotation_revision=0, roi_mask_ref=None, ignore_mask_ref=None)
    version = SimpleNamespace(status="active", format="pt", task_type="segment", model_path=str(weight),
                              dataset_id="dataset" if same_dataset else "other", class_schema_json=[{"id": "tower", "index": 0}])
    @contextmanager
    def sessions():
        yield SimpleNamespace(get=lambda kind, identity: image if kind is ImageItem else version)
    mask = torch.zeros(1, 200, 320)
    mask[0, 20:100, 20:100] = 1
    mask[0, 40:70, 40:70] = 0
    mask[0, 130:140, 200:220] = 1
    box = SimpleNamespace(cls=torch.tensor([0.]), xyxy=torch.tensor([[20., 20., 220., 140.]]), conf=torch.tensor([.8]))
    def predict(**kwargs):
        assert kwargs["retina_masks"] is True
        result = SimpleNamespace(boxes=[box], masks=SimpleNamespace(data=mask))
        result.cpu = lambda: result
        return [result]
    monkeypatch.setattr("app.models.inference.ModelCache.get_model", lambda *args: SimpleNamespace(predict=predict))
    context = {"inputs": {"dataset_id": "dataset", "model_id": "managed", "class_mapping": {"0": "tower"}},
               "profile": {"task_type": "segment", "class_rules": [{"id": "tower", "name": "tower"}]},
               "item": {"image_id": "image", "image_hash": sha256(path.read_bytes()).hexdigest(), "base_revision": 0}}
    if same_dataset:
        context["inputs"].pop("class_mapping")
    result = build_vision_handlers(storage, settings, sessions)["vision_propose"](context)
    assert result["request_count"] == 0
    assert len(result["annotations"]) == 1
    assert np.array_equal(MaskService(storage).decode(result["annotations"][0]["mask_ref"]), mask[0].numpy())


def test_explicit_vision_source_ignores_old_managed_model_and_reserves_api_calls(environment, monkeypatch):
    from app.core.models import ImageItem
    from app.vision.workflow import build_vision_handlers

    settings, storage, path = environment
    image = SimpleNamespace(id="image", dataset_id="dataset", storage_name=path.name,
                            width=320, height=200, annotation_revision=0, roi_mask_ref=None, ignore_mask_ref=None)

    @contextmanager
    def sessions():
        def get(kind, identity):
            assert kind is ImageItem, "An explicit LLM template must not load a managed model"
            return image
        yield SimpleNamespace(get=get)

    requests = []

    def endpoint(request):
        requests.append(json.loads(request.content))
        return reply({"objects": [{"class_id": "tower", "bbox": [20, 30, 70, 90], "evidence": "Visible tower"}]})

    service = VisionService(storage, settings, transport=httpx.MockTransport(endpoint))
    monkeypatch.setattr("app.vision.workflow.VisionService", lambda *args: service)
    context = {
        "node_type": "vision_propose", "params": {"source": "vision"},
        "inputs": {"dataset_id": "dataset", "model_id": "old-model",
                   "vision_approvals": {"proposer": approval(settings, path).model_dump()}},
        "profile": {"task_type": "detect", "class_rules": [{"id": "tower", "class_id": "tower", "name": "Tower"}]},
        "item": {"image_id": "image", "image_hash": sha256(path.read_bytes()).hexdigest(), "base_revision": 0},
    }
    handlers = build_vision_handlers(storage, settings, sessions)
    assert handlers["estimate_requests"](context)["count"] == 1
    assert requests == []
    result = handlers["vision_propose"](context)
    assert result["request_count"] == 1
    assert len(requests) == 1
    assert any(item["type"] == "image_url" for item in requests[0]["messages"][0]["content"])
    assert result["annotations"][0]["class_id"] == "tower"
    assert result["annotations"][0]["bbox"] == {"x": 20, "y": 30, "width": 50, "height": 60}


def test_vision_http_settings_masks_and_disabled_probe(client):
    configuration = {"enabled": False, "mode": "local", "api_base": "http://127.0.0.1:19090/v1", "model": "chosen-later", "api_key": "do-not-return"}
    response = client.put("/api/vision/settings", json={"proposer": configuration, "reviewer": configuration})
    assert response.status_code == 200, response.text
    assert "do-not-return" not in response.text
    assert response.json()["proposer"]["api_key_configured"]
    mask = np.zeros((12, 16), dtype=bool)
    mask[2:10, 2:14] = True
    mask[4:8, 5:11] = False
    written = client.post("/api/vision/masks", json={"rle": encode_mask(mask)})
    assert written.status_code == 201, written.text
    result = client.get("/api/vision/masks/" + written.json()["sha256"])
    assert result.status_code == 200
    assert np.array_equal(decode_mask(result.json()["rle"]), mask)
    unavailable = client.post("/api/vision/probe/proposer")
    assert unavailable.status_code == 200
    assert unavailable.json()["code"] == "vision_endpoint_unavailable"
    assert not unavailable.json()["remote_test_performed"]


@pytest.mark.parametrize("mode", ["local", "remote"])
def test_vision_requests_ignore_environment_proxy_configuration(environment, monkeypatch, mode):
    settings, storage, path = environment
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"message": {"content": '{"objects": []}'}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            monkeypatch.setenv(key, "http://127.0.0.1:1")
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        endpoint = VisionEndpointUpdate(enabled=True, mode=mode, api_base=f"http://127.0.0.1:{server.server_port}/v1", model="offline-loopback-test")
        VisionSettingsService(settings).update(VisionSettingsUpdate(proposer=endpoint, reviewer=endpoint))
        result = VisionService(storage, settings).propose(
            "image", path, [VisionClassRule(id="tower", name="tower")], "v1", ImageStrategy(), approval(settings, path),
        )
        assert result.objects == []
        assert len(received) == 1
        assert received[0][0] == "/v1/chat/completions"
        assert b"data:image/png;base64," in received[0][1]
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize("status", [302, 307])
def test_vision_redirect_does_not_send_image_to_another_destination(environment, status):
    settings, storage, path = environment
    requests = []

    def endpoint(request):
        requests.append(request)
        assert str(request.url) == "https://chosen.example/v1/chat/completions"
        return httpx.Response(status, headers={"Location": "https://unapproved.example/v1/chat/completions"})

    with pytest.raises(ValidationError) as error:
        VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).propose(
            "image", path, [VisionClassRule(id="tower", name="tower")], "v1", ImageStrategy(), approval(settings, path),
        )
    assert error.value.error_code == "vision_http_error"
    assert error.value.details["http_status"] == status
    assert len(requests) == 1


def configure_polygon_test(settings, coordinate_format="pixel", multi_image=True):
    configuration = VisionSettingsService(settings)
    values = configuration.get().proposer.model_dump(exclude={"api_key_configured", "config_hash"})
    values.update(coordinate_format=coordinate_format, multi_image=multi_image)
    configuration.update(VisionSettingsUpdate(proposer=VisionEndpointUpdate(**values), reviewer=VisionEndpointUpdate(**values)))


def polygon_coordinates(points, coordinate_format, width=320, height=200):
    if coordinate_format == "pixel":
        return points
    factor = 1 if coordinate_format == "normalized_1" else 1000
    return [(x * factor / width, y * factor / height) for x, y in points]


@pytest.mark.parametrize("coordinate_format", ["pixel", "normalized_1", "normalized_1000"])
def test_direct_polygon_proposal_keeps_concavity_and_native_geometry(environment, coordinate_format):
    settings, storage, path = environment
    configure_polygon_test(settings, coordinate_format)
    expected = [(40, 40), (100, 40), (70, 65), (100, 100), (40, 100)]
    sent = []

    def endpoint(request):
        data = json.loads(request.content)
        text = data["messages"][0]["content"][0]["text"]
        assert '"NativePolygonCandidate"' in text and '"polygon"' in text
        assert "Do not output bbox as a replacement contour" in data["messages"][0]["content"][-1]["text"]
        sent.append(data)
        return reply({"objects": [{"class_id": "tower", "polygon": polygon_coordinates(expected, coordinate_format), "evidence": "Concave outline visible."}]})

    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).propose(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", ImageStrategy(), approval(settings, path), geometry="polygon")
    assert len(sent) == 1 and result.issues == []
    actual = result.objects[0]
    assert actual.type == "polygon" and actual.mask_ref is None
    for point, reference in zip(actual.polygon, expected):
        assert point == pytest.approx(reference)
    assert actual.bbox.model_dump() == pytest.approx({"x": 40, "y": 40, "width": 60, "height": 60})
    assert actual.provenance_json["geometry_source"] == "vision_llm"
    assert "segmenter" not in actual.provenance_json


@pytest.mark.parametrize("bad_polygon", [None, [[0, 0], [100, 100], [0, 100], [100, 0]], [[0, 0], [999, 0], [10, 40]], [[0, 0], ["bad", 40], [40, 0]],
                                        [[0, 0], [20, 0], [20, 10 ** 400]], [[0, 0], [20, 0], [20, float("nan")]],
                                        [[0, 0], [20, 0], [20, float("inf")]], [[0, 0], [20, 0], [20, True]]])
def test_bad_or_missing_direct_contours_are_isolated_without_box_fallback(environment, bad_polygon):
    settings, storage, path = environment

    def endpoint(request):
        return reply({"objects": [
            {"class_id": "tower", "polygon": [[10, 10], [90, 10], [50, 90]], "evidence": "Valid tower."},
            {"class_id": "tower", "polygon": bad_polygon, "bbox": [10, 10, 90, 90], "evidence": "Invalid contour cannot fall back to box."},
        ]})

    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).propose(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", ImageStrategy(), approval(settings, path), geometry="polygon")
    assert len(result.objects) == 1 and result.objects[0].type == "polygon"
    assert len(result.issues) == 1 and result.issues[0]["code"] == "vision_polygon_invalid"
    assert result.issues[0]["coordinate_details"]["phase"] == "propose"
    json.dumps(result.model_dump(), allow_nan=False)


@pytest.mark.parametrize("coordinate_format", ["pixel", "normalized_1", "normalized_1000"])
@pytest.mark.parametrize("multi_image", [False, True])
def test_direct_contour_review_receives_real_vertices_and_visible_outline(environment, coordinate_format, multi_image):
    settings, storage, path = environment
    configure_polygon_test(settings, coordinate_format, multi_image)
    points = [(40, 40), (100, 40), (70, 65), (100, 100), (40, 100)]
    objects = [{"object_id": "tower", "class_id": "tower", "type": "polygon", "polygon": points}]
    calls = []

    def endpoint(request):
        data = json.loads(request.content)
        calls.append(data)
        if len(calls) == 1:
            return reply({"objects": [{"class_id": "tower", "polygon": polygon_coordinates(points, coordinate_format), "evidence": "Target outline."}]})
        content = data["messages"][0]["content"]
        prompt = content[0]["text"]
        existing, _ = json.JSONDecoder().raw_decode(prompt.split("Existing objects in SENT-image coordinates=", 1)[1])
        for actual, expected in zip(existing[0]["polygon"], polygon_coordinates(points, coordinate_format)):
            assert actual == pytest.approx(expected)
        images = [part for part in content if part["type"] == "image_url"]
        assert len(images) == (2 if multi_image else 1)
        with Image.open(io.BytesIO(base64.b64decode(images[-1]["image_url"]["url"].split(",", 1)[1]))) as overlay:
            assert overlay.getpixel((50, 60)) != (255, 255, 255)
            assert overlay.getpixel((95, 65)) == (255, 255, 255)
        return reply({"issues": []})

    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).review(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", 1, objects, ImageStrategy(),
        approval(settings, path, "reviewer"), geometry="polygon")
    assert result.issues == []
    assert result.inventory[0].type == "polygon" and result.inventory[0].polygon is not None


@pytest.mark.parametrize("coordinate_format", ["pixel", "normalized_1", "normalized_1000"])
def test_direct_polygon_repair_maps_all_vertices_from_crop_and_preserves_identity(environment, coordinate_format):
    settings, storage, path = environment
    configure_polygon_test(settings, coordinate_format)
    objects = [{"object_id": "stable", "class_id": "tower", "type": "polygon", "source": "manual",
                "polygon": [(40, 40), (100, 40), (70, 70), (100, 100), (40, 100)],
                "bbox": {"x": 40, "y": 40, "width": 60, "height": 60}}]
    issues = [{"type": "boundary", "object_id": "stable", "bbox": objects[0]["bbox"]}]
    native = [(10, 10), (65, 10), (35, 35), (65, 65), (10, 65)]
    crops = []

    def endpoint(request):
        content = json.loads(request.content)["messages"][0]["content"]
        metadata = json.loads(content[1]["text"])
        crops.append(metadata["crop_xyxy"])
        width, height = metadata["sent_size"]
        return reply({"objects": [{"class_id": "tower", "polygon": polygon_coordinates(native, coordinate_format, width, height), "evidence": "Boundary completed."}]})

    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).repair(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", 1, objects, issues, ImageStrategy(), approval(settings, path), geometry="polygon")
    assert result["repair_diff"][0]["applied"] is True
    actual = result["objects"][0]
    assert actual["object_id"] == "stable" and actual["type"] == "polygon" and actual["mask_ref"] is None
    left, top, _, _ = crops[0]
    for point, (x, y) in zip(actual["polygon"], native):
        assert point == pytest.approx((left + x, top + y))


def test_polygon_merge_does_not_discard_different_contours_with_same_bounds(environment):
    settings, storage, path = environment
    first = [(20, 20), (80, 20), (20, 80)]
    second = [(20, 20), (80, 20), (80, 80)]

    def endpoint(request):
        return reply({"objects": [
            {"class_id": "tower", "polygon": contour, "evidence": "Visible outline."}
            for contour in [first, second, [first[1], first[0], first[2]]]
        ]})

    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).propose(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", ImageStrategy(), approval(settings, path), geometry="polygon")
    assert len(result.objects) == 2
    assert result.objects[0].polygon == first and result.objects[1].polygon == second
    assert len(result.objects[0].provenance_json["merged_sources"]) == 1
