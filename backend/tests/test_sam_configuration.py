from __future__ import annotations

from hashlib import sha256
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.agent.workflow_tools import build_workflow_service, normalize_workflow_payload
from app.core.config import Settings
from app.core.errors import ValidationError
from app.core.storage import Storage
from app.sam.configuration import freeze_sam_configuration, verify_sam_configuration
from app.sam.schemas import SamPredictRequest
from app.sam.service import SamService
from app.settings.schemas import SamSettingsUpdate
from app.settings.service import SettingsService
from app.workflow.schemas import RunCreate


@pytest.fixture
def configured_sam(tmp_path):
    weights = tmp_path / "selected.pt"
    weights.write_bytes(b"first-weight-bytes")
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data", import_root=tmp_path / "imports", database_url="sqlite://")
    SettingsService(settings).update_sam(SamSettingsUpdate(model=str(weights), device="cpu", fallback_mode="disabled"))
    return settings, weights


def test_frozen_sam_includes_weight_content_and_rejects_same_path_replacement(configured_sam):
    settings, weights = configured_sam
    frozen = freeze_sam_configuration(settings)
    assert frozen["weights_sha256"] == sha256(weights.read_bytes()).hexdigest()
    assert frozen["weights_path"] == str(weights.resolve())
    assert frozen["weights_available"]
    assert verify_sam_configuration(settings, frozen, require_weights=True) == frozen
    weights.write_bytes(b"other-weight-bytes")
    with pytest.raises(ValidationError) as error:
        verify_sam_configuration(settings, frozen, require_weights=True)
    assert error.value.error_code == "vision_segmenter_changed"


@pytest.mark.parametrize("field,value", [("device", "auto"), ("img_size", 2048), ("fallback_mode", "box"), ("enabled", False)])
def test_sam_configuration_changes_require_new_approved_snapshot(configured_sam, field, value):
    settings, _ = configured_sam
    frozen = freeze_sam_configuration(settings)
    values = {key: frozen[key] for key in ("enabled", "model", "device", "img_size", "fallback_mode")}
    values[field] = value
    SettingsService(settings).update_sam(SamSettingsUpdate(**values))
    with pytest.raises(ValidationError, match="approved run"):
        verify_sam_configuration(settings, frozen)


def test_unavailable_named_weights_are_described_without_loading_or_downloading(configured_sam):
    settings, _ = configured_sam
    SettingsService(settings).update_sam(SamSettingsUpdate(model="not-downloaded.pt", device="cpu"))
    frozen = freeze_sam_configuration(settings)
    assert not frozen["weights_available"] and frozen["weights_sha256"] is None
    with pytest.raises(ValidationError) as error:
        verify_sam_configuration(settings, frozen, require_weights=True)
    assert error.value.error_code == "sam_weights_unavailable"


def test_http_and_agent_preparation_share_the_same_sam_snapshot(client, tmp_path):
    weights = tmp_path / "preview-only.pt"
    weights.write_bytes(b"synthetic-weight-metadata")
    response = client.put("/api/settings/sam", json={"enabled": True, "model": str(weights), "device": "cpu", "img_size": 1024, "fallback_mode": "disabled"})
    assert response.status_code == 200
    dataset = client.post("/api/datasets", json={"name": "SAM prepare", "task_type": "segment"}).json()
    content = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(content, format="PNG")
    image = client.post(f"/api/datasets/{dataset['id']}/images/upload", files={"files": ("synthetic.png", content.getvalue(), "image/png")}).json()["items"][0]
    profile = client.post("/api/workflows/profiles", json={"key": "sam-freeze", "name": "SAM freeze", "config": {"task_type": "segment", "geometry": "mask"}}).json()
    template_response = client.post("/api/workflows/templates", json={"key": "sam-preview", "name": "SAM preview", "nodes": [{"id": "segment", "type": "segment_proposals"}, {"id": "confirm", "type": "human_review"}]})
    assert template_response.status_code == 201, template_response.text
    template = template_response.json()
    request = RunCreate(idempotency_key="sam-prepare", template_id=template["id"], profile_id=profile["id"], dataset_id=dataset["id"], image_ids=[image["id"]])
    prepared = client.post("/api/workflows/runs/prepare", json=request.model_dump())
    assert prepared.status_code == 200, prepared.text
    frozen = prepared.json()["sam_config"]
    assert frozen["weights_sha256"] == sha256(weights.read_bytes()).hexdigest()
    assert prepared.json()["request"]["inputs"]["sam_config"] == frozen
    service = build_workflow_service(client.app.state.storage, client.app.state.database.session_factory, client.app.state.settings)
    agent = service.prepare_run(request, {})
    normalized = normalize_workflow_payload("create_workflow_run", agent["request"])
    assert normalized["preview"]["sam_config"] == frozen


@pytest.mark.parametrize("change_during_inference", [False, True])
def test_sam_call_rejects_weight_replacement_before_or_during_inference(configured_sam, monkeypatch, change_during_inference):
    settings, weights = configured_sam
    storage = Storage(settings.data_dir, settings.import_root)
    storage.write_image("dataset", "image.png", _image_bytes())
    image = SimpleNamespace(id="image", dataset_id="dataset", storage_name="image.png", width=32, height=32)
    label = SimpleNamespace(id="class", dataset_id="dataset")
    dataset = SimpleNamespace(task_type="segment")
    session = SimpleNamespace(get=lambda kind, identity: dataset if identity == "dataset" else label)
    monkeypatch.setattr("app.sam.service.get_image", lambda session, image_id: image)
    frozen = freeze_sam_configuration(settings)
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        if change_during_inference:
            weights.write_bytes(b"replacement during inference")
        return [SimpleNamespace(masks=SimpleNamespace(data=np.ones((1, 32, 32), dtype=bool), xy=[]), boxes=None)]

    def model_for(path, device, digest):
        assert path == frozen["weights_path"] and digest == frozen["weights_sha256"]
        return SimpleNamespace(predict=predict)

    monkeypatch.setattr("app.sam.service._model_for", model_for)
    if not change_during_inference:
        weights.write_bytes(b"replacement before inference")
    with pytest.raises(ValidationError) as error:
        SamService(storage, settings).predict_mask(session, SamPredictRequest(image_id="image", class_id="class", box={"x": 0, "y": 0, "width": 32, "height": 32}), expected_config=frozen)
    assert error.value.error_code == "vision_segmenter_changed"
    assert len(calls) == int(change_during_inference)


def _image_bytes() -> bytes:
    image = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(image, format="PNG")
    return image.getvalue()


def test_sam_cache_distinguishes_weight_hash_at_the_same_path(monkeypatch):
    from app.sam import service

    loaded = []
    def constructor(path):
        model = object()
        loaded.append(model)
        return model
    monkeypatch.setattr("ultralytics.SAM", constructor)
    service._MODELS.clear()
    try:
        first = service._model_for("same.pt", "cpu", "first-hash")
        assert service._model_for("same.pt", "cpu", "first-hash") is first
        assert service._model_for("same.pt", "cpu", "second-hash") is not first
        assert len(loaded) == 2
    finally:
        service._MODELS.clear()
