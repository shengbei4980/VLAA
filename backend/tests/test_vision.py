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
    _, views = prepare_views(image.getvalue(), ImageStrategy(tile_size=512, overlap=.2), 256)
    assert views[0].sent_width < 256
    assert views[-1].crop[-1] == 3000
    assert any(view.crop[1] == 0 and not view.is_overview for view in views)
    with pytest.raises(ValidationError, match="limit"):
        prepare_views(image.getvalue(), ImageStrategy(tile_size=64, max_views=2), 256)


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
