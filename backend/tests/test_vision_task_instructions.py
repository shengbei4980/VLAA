"""Task instructions travel through frozen workflow profiles and offline image requests."""
import io
import json

import httpx
from PIL import Image
from pydantic import ValidationError as SchemaValidationError
import pytest

from app.core.config import Settings
from app.core.database import Base, Database
from app.core.models import ClassLabel, Dataset, ImageItem
from app.core.storage import Storage
from app.vision.schemas import ImageStrategy, VisionApproval, VisionClassRule, VisionEndpointUpdate, VisionSettingsUpdate
from app.vision.service import VisionService
from app.vision.settings import VisionSettingsService
from app.vision.workflow import build_vision_handlers
from app.workflow.bridges import WorkflowBindings
from app.workflow.schemas import NodeDefinition, ProfileCreate, RunCreate, TaskProfileConfig, TemplateCreate
from app.workflow.service import WorkflowService


@pytest.fixture
def environment(tmp_path):
    settings = Settings(project_root=tmp_path, data_dir=tmp_path / "data", import_root=tmp_path / "imports",
                        database_url=f"sqlite:///{tmp_path / 'instructions.db'}")
    storage = Storage(settings.data_dir, settings.import_root)
    database = Database(settings.database_url)
    Base.metadata.create_all(database.engine)
    content = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(content, format="PNG")
    storage.write_image("dataset", "image.png", content.getvalue())
    with database.session_factory() as session:
        session.add(Dataset(id="dataset", name="Unlabeled architecture", task_type="detect"))
        session.flush()
        session.add(ClassLabel(id="tower", dataset_id="dataset", class_index=0, name="Tower"))
        session.add(ImageItem(id="image", dataset_id="dataset", file_name="image.png", storage_name="image.png",
                              width=100, height=80, split="train"))
        session.commit()
    endpoint = VisionEndpointUpdate(enabled=True, mode="remote", api_base="https://offline.example/v1", model="mock-vision")
    VisionSettingsService(settings).update(VisionSettingsUpdate(proposer=endpoint, reviewer=endpoint))
    try:
        yield settings, storage, database
    finally:
        database.dispose()


def response(value):
    return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})


def test_frozen_task_instructions_reach_proposal_independent_review_comparison_and_repair(environment, monkeypatch):
    settings, storage, database = environment
    instructions = "识别建筑主体。\n参考说明另列 Trees、Road 等类别，只标出实际可见区域。"
    updated_instructions = "新方案仅识别道路。"
    requests = []

    def endpoint(request):
        requests.append(json.loads(request.content))
        if len(requests) == 3:
            return response({"issues": [{"type": "boundary", "bbox": [40, 40, 60, 60],
                                          "evidence": "Check the visible boundary."}]})
        bbox = [8, 8, 28, 28] if len(requests) == 4 else [40, 40, 60, 60]
        return response({"objects": [{"class_id": "tower", "bbox": bbox, "evidence": "A visible tower."}]})

    visual = VisionService(storage, settings, transport=httpx.MockTransport(endpoint))
    monkeypatch.setattr("app.vision.workflow.VisionService", lambda *args: visual)
    service = WorkflowService(database.session_factory, storage,
                              WorkflowBindings(handlers=build_vision_handlers(storage, settings, database.session_factory)),
                              project_root=settings.project_root)
    profile = service.create_profile(ProfileCreate(key="instructions", name="Architecture", config=TaskProfileConfig(
        class_rules=[{"class_id": "tower", "name": "Tower"}], rules={"task_instructions": instructions}, repair_limit=1,
    )))
    template = service.create_template(TemplateCreate(key="instructions", name="Instructions", nodes=[
        NodeDefinition(id="propose", type="vision_propose", params={"source": "vision"}),
        NodeDefinition(id="review", type="vision_review"),
        NodeDefinition(id="repair", type="repair_proposals"),
        NodeDefinition(id="human", type="human_review"),
    ]))
    prepared = service.prepare_run(RunCreate(idempotency_key="instructions", template_id=template["id"],
        profile_id=profile["id"], dataset_id="dataset", image_ids=["image"]),
        VisionSettingsService(settings).get().model_dump(mode="json"))
    run = service.create_run(RunCreate.model_validate(prepared["request"]))
    service.create_profile(ProfileCreate(key="instructions", name="Changed later", config=TaskProfileConfig(
        class_rules=[{"class_id": "tower", "name": "Tower"}], rules={"task_instructions": updated_instructions}, repair_limit=1,
    )))
    for _ in range(12):
        service.tick(run["id"])
        current = service.get_run(run["id"])
        if current["status"] in {"awaiting_review", "needs_attention"}:
            break
    assert current["status"] == "awaiting_review", current.get("error_message")
    assert current["profile_snapshot"]["rules"]["task_instructions"] == instructions
    assert len(requests) == 4
    for request in requests:
        prompt = request["messages"][0]["content"][0]["text"]
        assert json.dumps(instructions, ensure_ascii=False) in prompt
        assert updated_instructions not in prompt
        assert "coordinate_format=pixel" in prompt
        assert "bbox is [left,top,right,bottom]" in prompt
        assert '"id": "tower"' in prompt
        assert "Return one JSON object matching this schema" in prompt
        assert request["response_format"]["type"] == "json_object"
        scope_start = prompt.index("Authoritative target class IDs=")
        scope, _ = json.JSONDecoder().raw_decode(prompt[scope_start:].split("=", 1)[1])
        assert scope == ["tower"]
        assert scope_start > prompt.index("User task instructions=")
        assert "null class_id only for a plausible instance of these target classes" in prompt
    inventory_prompt = requests[1]["messages"][0]["content"][0]["text"]
    compare_prompt = requests[2]["messages"][0]["content"][0]["text"]
    assert "Existing objects" not in inventory_prompt
    assert "Existing objects" in compare_prompt
    assert "Repair only the localized issues" in requests[3]["messages"][0]["content"][0]["text"]


def test_omitted_task_instructions_keep_the_existing_image_request_contract(environment):
    from hashlib import sha256

    settings, storage, _ = environment
    path = storage.image_path("dataset", "image.png")
    config = VisionSettingsService(settings).internal("proposer")
    approval = VisionApproval(mode=config.mode, api_base=config.api_base, model=config.model, config_hash=config.config_hash,
                              allowed_image_hashes=[sha256(path.read_bytes()).hexdigest()], max_requests=1)
    requests = []

    def endpoint(request):
        requests.append(json.loads(request.content))
        return response({"objects": []})

    result = VisionService(storage, settings, transport=httpx.MockTransport(endpoint)).propose(
        "image", path, [VisionClassRule(id="tower", name="Tower")], "v1", ImageStrategy(), approval,
    )
    assert result.request_count == len(requests) == 1
    prompt = requests[0]["messages"][0]["content"][0]["text"]
    assert "User task instructions=" not in prompt
    assert "coordinate_format=pixel" in prompt
    assert TaskProfileConfig(rules={"existing_annotations": "preserve"}).rules == {"existing_annotations": "preserve"}
    assert TaskProfileConfig(rules={"task_instructions": ""}).rules["task_instructions"] == ""
    assert len(TaskProfileConfig(rules={"task_instructions": "字" * 20000}).rules["task_instructions"]) == 20000


@pytest.mark.parametrize("instructions", [None, 12, ["label towers"], {"scope": "all"}, "字" * 20001],
                         ids=["null", "number", "list", "object", "too-long"])
def test_task_instructions_reject_invalid_type_or_length(instructions):
    with pytest.raises(SchemaValidationError):
        TaskProfileConfig(rules={"task_instructions": instructions})
