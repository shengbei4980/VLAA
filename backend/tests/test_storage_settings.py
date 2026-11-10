from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import select

from app.api.error_handlers import register_error_handlers
from app.core.config import Settings
from app.core.database import Base, Database
from app.core.errors import ConflictError, ValidationError
from app.core.models import (
    AgentRun, AgentSession, Annotation, ClassLabel, ImageItem, Dataset,
    ModelEvaluationRecord, ModelVersion, TrainingTask,
)
from app.core.storage import Storage
from app.dataset.annotation.masks import MaskService
from app.dataset.annotation.review import confirm_image, current_content, json_fingerprint, require_confirmed
from app.dataset.snapshots import create_snapshot, get_snapshot
from app.storage_settings.bootstrap import read_config, startup_paths
from app.storage_settings.middleware import StorageWriteGuard
from app.storage_settings.router import router as storage_router
from app.storage_settings.service import StorageSettingsService, file_hash
from app.workflow.artifacts import WorkflowArtifactStore
from app.workflow.models import WorkflowArtifact, WorkflowRun
from app.workflow.schemas import ProfileCreate, ReviewDecision, RunCreate, TaskProfileConfig, TemplateCreate, NodeDefinition
from app.workflow.service import WorkflowService


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("YWA_DATA_DIR", raising=False)
    monkeypatch.delenv("YWA_DATABASE_URL", raising=False)
    project = tmp_path / "工程 空间"
    source = project / "data"
    external_db = project / "metadata" / "original.sqlite"
    external_db.parent.mkdir(parents=True)
    settings = Settings(project_root=project, data_dir=source, import_root=source / "imports",
                        database_url=f"sqlite:///{external_db}")
    settings.ensure_directories()
    database = Database(settings.database_url)
    Base.metadata.create_all(database.engine)
    storage = Storage(source, settings.import_root)
    (source / "business.bin").write_bytes(b"business asset")
    (source / "settings.json").write_text(json.dumps({"llm": {"api_key": "private-test-key"}}), encoding="utf-8")
    (source / "logs").mkdir(exist_ok=True)
    (source / "logs" / "backend.log").write_text("runtime log", encoding="utf-8")
    stream = io.BytesIO()
    Image.new("RGB", (80, 60), "white").save(stream, format="PNG")
    with database.session_factory() as session:
        session.add(Dataset(id="ds_storage", name="Storage fixture", task_type="detect"))
        session.flush()
        session.add(ClassLabel(id="cls_storage", dataset_id="ds_storage", class_index=0, name="part"))
        session.flush()
        for image_id in ("img_confirmed", "img_pending"):
            storage.write_image("ds_storage", image_id + ".png", stream.getvalue())
            session.add(ImageItem(id=image_id, dataset_id="ds_storage", file_name=image_id + ".png",
                storage_name=image_id + ".png", width=80, height=60, split="train", annotation_revision=1))
        session.flush()
        session.add(Annotation(id="ann_storage", image_id="img_confirmed", dataset_id="ds_storage", class_id="cls_storage",
            type="bbox", x=5, y=5, width=20, height=20, source="manual", object_id="object-storage",
            provenance_json={"original_path": str(source / "original_audit.txt")}))
        session.flush()
        image = session.get(ImageItem, "img_confirmed")
        confirm_image(session, storage, image.id, expected_revision=1,
                      expected_fingerprint=json_fingerprint(current_content(session, storage, image)), reason="Reviewed fixture")
        session.commit()
    workflow = WorkflowService(database.session_factory, storage, project_root=project)
    workflow.ensure_defaults()
    profile = workflow.create_profile(ProfileCreate(key="storage", name="Storage", config=TaskProfileConfig()))
    app = FastAPI()
    app.state.settings = settings
    app.state.database = database
    app.state.storage = storage
    app.state.workflow_service = workflow
    app.add_middleware(StorageWriteGuard, settings=settings)
    app.include_router(storage_router, prefix="/api")
    register_error_handlers(app)

    @app.post("/api/probe-write")
    def probe():
        (source / "mutation.txt").write_text("mutated", encoding="utf-8")
        return {"ok": True}

    @app.get("/api/training/tasks")
    def telemetry_probe():
        (source / "telemetry.txt").write_text("refresh", encoding="utf-8")
        return {"items": []}

    with TestClient(app) as client:
        yield {"settings": settings, "database": database, "storage": storage, "service": StorageSettingsService(settings, database),
               "client": client, "workflow": workflow, "profile": profile, "target": tmp_path / "新的 数据位置"}
    database.dispose()


def test_registered_destinations_are_absolute_and_credential_free(workspace):
    client = workspace["client"]
    assert client.post("/api/storage/destinations/validate", json={"path": "relative"}).status_code == 422
    target = workspace["target"]
    first = client.post("/api/storage/destinations", json={"name": "中文 输出", "path": str(target)})
    assert first.status_code == 201, first.text
    second = client.post("/api/storage/destinations", json={"name": "same", "path": str(target)})
    assert second.json()["id"] == first.json()["id"]
    assert len(client.get("/api/storage/destinations").json()) == 1
    config = (workspace["settings"].project_root / ".ywa-storage.json").read_text(encoding="utf-8")
    assert "private-test-key" not in config
    assert "private-test-key" not in client.get("/api/storage/settings").text


@pytest.mark.parametrize("kind", ["same", "child", "parent", "nonempty"])
def test_relocation_rejects_recursive_and_nonempty_targets(workspace, kind):
    source = workspace["settings"].data_dir
    target = {"same": source, "child": source / "nested", "parent": source.parent, "nonempty": workspace["target"]}[kind]
    if kind == "nonempty":
        target.mkdir()
        (target / "existing.txt").write_text("preserve", encoding="utf-8")
    response = workspace["client"].post("/api/storage/data-root/prepare", json={"path": str(target)})
    assert response.status_code in {409, 422}, response.text
    assert (source / "business.bin").read_bytes() == b"business asset"


def test_activity_and_environment_override_prevent_migration(workspace, monkeypatch):
    with workspace["database"].session_factory() as session:
        session.add(AgentSession(id="agent_storage", title="active"))
        session.flush()
        session.add(AgentRun(id="agent_run_storage", session_id="agent_storage", status="running", provider="mock", model="mock"))
        session.commit()
    response = workspace["client"].post("/api/storage/data-root/prepare", json={"path": str(workspace["target"])})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "storage_tasks_active"
    monkeypatch.setenv("YWA_DATA_DIR", str(workspace["settings"].data_dir))
    response = workspace["client"].post("/api/storage/data-root/prepare", json={"path": str(workspace["target"])})
    assert response.json()["error"]["code"] == "storage_environment_override"


def test_migration_preserves_database_assets_and_valid_review_after_old_root_renamed(workspace):
    workflow = workspace["workflow"]
    template = next(item for item in workflow.list_templates() if item["key"] == "confirmed_train_evaluate_v1")
    run = workflow.create_run(RunCreate(idempotency_key="completed-storage", template_id=template["id"], profile_id=workspace["profile"]["id"],
        dataset_id="ds_storage", image_ids=["img_confirmed"]))
    with workspace["database"].session_factory() as session:
        snapshot = create_snapshot(session, workspace["storage"], "ds_storage", image_ids=["img_confirmed"])
        workflow._artifact(session, run["id"], None, None, "dataset_snapshot", snapshot)
        session.get(WorkflowRun, run["id"]).status = "completed"
        session.commit()
    client = workspace["client"]
    prepared = client.post("/api/storage/data-root/prepare", json={"path": str(workspace["target"])})
    assert prepared.status_code == 200, prepared.text
    assert prepared.json()["file_count"] > 1
    assert "logs" in prepared.json()["excluded_runtime_directories"]
    result = client.post("/api/storage/data-root/execute", json={"plan_id": prepared.json()["plan_id"]})
    assert result.status_code == 200, result.text
    state = result.json()
    assert state["restart_required"] and state["write_locked"]
    assert state["pending"]["data_dir"] == str(workspace["target"])
    assert client.post("/api/probe-write").status_code == 423
    assert (workspace["settings"].data_dir / "business.bin").exists()
    assert (workspace["target"] / "business.bin").read_bytes() == b"business asset"
    workspace["database"].dispose()
    old = workspace["settings"].data_dir
    old.rename(old.with_name("original retained"))
    target_db = Database(f"sqlite:///{workspace['target'] / 'yolowebagent-starter.db'}")
    try:
        storage = Storage(workspace["target"], workspace["target"] / "imports")
        with target_db.session_factory() as session:
            assert require_confirmed(session, storage, "ds_storage", ["img_confirmed"])[0]["receipt"] is not None
            assert session.get(ImageItem, "img_pending").review_status == "pending"
            frozen = get_snapshot(session, storage, snapshot["id"])
            assert Path(frozen["root"]).is_relative_to(workspace["target"])
            for artifact in session.scalars(select(WorkflowArtifact)):
                WorkflowArtifactStore(storage).verify(artifact.storage_name, artifact.sha256)
                assert artifact.payload_json["root"].startswith(str(workspace["target"]))
    finally:
        target_db.dispose()
    selected, selected_db = startup_paths(workspace["settings"].project_root, {})
    assert selected == workspace["target"] and str(workspace["target"]) in selected_db
    assert "private-test-key" in (workspace["target"] / "settings.json").read_text(encoding="utf-8")


def test_copy_and_pending_activation_hold_write_gate_until_cancel(workspace, monkeypatch):
    from app.storage_settings import rebase
    entered, release = threading.Event(), threading.Event()
    original = rebase.transform_copied_data

    def paused(*args):
        entered.set()
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(rebase, "transform_copied_data", paused)
    client = workspace["client"]
    prepared = client.post("/api/storage/data-root/prepare", json={"path": str(workspace["target"])}).json()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.post, "/api/storage/data-root/execute", json={"plan_id": prepared["plan_id"]})
        assert entered.wait(10)
        assert client.get("/api/storage/settings").json()["write_locked"]
        assert client.post("/api/probe-write").status_code == 423
        assert client.get("/api/training/tasks").status_code == 423
        assert client.post("/api/storage/data-root/cancel", json={}).status_code == 409
        release.set()
        response = future.result(timeout=15)
    assert response.status_code == 200, response.text
    assert client.post("/api/probe-write").status_code == 423
    assert client.get("/api/training/tasks").status_code == 423
    cancelled = client.post("/api/storage/data-root/cancel", json={})
    assert cancelled.status_code == 200, cancelled.text
    assert not cancelled.json()["restart_required"]
    assert client.post("/api/probe-write").status_code == 200
    assert (workspace["target"] / "business.bin").exists()


def test_failed_rebase_never_switches_pointer_or_deletes_source(workspace, monkeypatch):
    def fail(*args):
        raise ValidationError("fixture_rebase_failed", "fixture failure")

    monkeypatch.setattr("app.storage_settings.rebase.transform_copied_data", fail)
    service = workspace["service"]
    prepared = service.prepare(str(workspace["target"]))
    before = (workspace["settings"].data_dir / "business.bin").read_bytes()
    with pytest.raises(ValidationError, match="fixture failure"):
        service.execute(prepared["plan_id"])
    assert not read_config(workspace["settings"].project_root).get("data_dir")
    assert (workspace["settings"].data_dir / "business.bin").read_bytes() == before
    assert not service.get()["restart_required"]


def test_insufficient_space_and_environment_precedence(workspace, monkeypatch):
    from collections import namedtuple
    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("app.storage_settings.registry.shutil.disk_usage", lambda _: usage(100, 99, 1))
    with pytest.raises(ValidationError, match="space"):
        workspace["service"].prepare(str(workspace["target"]))
    root, url = startup_paths(workspace["settings"].project_root, {"YWA_DATA_DIR": str(workspace["target"]), "YWA_DATABASE_URL": "sqlite:///override.db"})
    assert root == workspace["target"] and url == "sqlite:///override.db"


def test_workflow_output_destination_is_frozen_and_candidates_are_self_contained(workspace, tmp_path):
    import numpy as np
    registry = workspace["service"].registry
    destination = registry.register("标注 输出", str(tmp_path / "输出 结果"))
    workflow = workspace["workflow"]
    profile = workflow.create_profile(ProfileCreate(key="external", name="External", config=TaskProfileConfig(
        outputs={"destination_id": destination["id"]})))
    template = workflow.create_template(TemplateCreate(key="review", name="Review", nodes=[NodeDefinition(id="review", type="human_review")]))
    with workspace["database"].session_factory() as session:
        image = session.get(ImageItem, "img_confirmed")
        roi = np.ones((60, 80), dtype=np.bool_)
        roi[:5] = False
        image.roi_mask_ref = MaskService(workspace["storage"]).write(roi).mask_ref
        session.commit()
    request = RunCreate(idempotency_key="external", template_id=template["id"], profile_id=profile["id"],
                        dataset_id="ds_storage", image_ids=["img_confirmed"])
    prepared = workflow.prepare_run(request, {})
    assert prepared["outputs"]["annotations"]["path"] == destination["path"]
    run = workflow.create_run(RunCreate.model_validate(prepared["request"]))
    assert workflow.tick(run["id"])
    run = workflow.get_run(run["id"])
    assert run["status"] == "awaiting_review", run["error_message"]
    review = run["reviews"][0]
    workflow.decide_review(review["id"], ReviewDecision(expected_revision=1, decision="approved"))
    output = Path(run["context"]["output_destinations"]["annotations"]["run_path"])
    receipts = list(output.glob("*/receipt.json"))
    assert len(receipts) == 2
    kinds = set()
    for receipt_file in receipts:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        kinds.add(receipt["kind"])
        for item in receipt["files"]:
            copied = receipt_file.parent / item["path"]
            assert copied.is_file() and file_hash(copied) == item["sha256"]
        assert any(item["path"].startswith("masks/") for item in receipt["files"])
        assert any(item["path"].startswith("images/") for item in receipt["files"])
    assert kinds == {"review_proposals", "confirmed_annotations"}
    confirmed = next(receipt.parent for receipt in receipts if json.loads(receipt.read_text(encoding="utf-8"))["kind"] == "confirmed_annotations")
    assert json.loads((confirmed / "artifact.json").read_text(encoding="utf-8"))["receipt"]["receipt_id"]
    workspace["database"].dispose()
    source = workspace["settings"].data_dir
    source.rename(source.with_name("original retained"))
    for receipt_file in receipts:
        for item in json.loads(receipt_file.read_text(encoding="utf-8"))["files"]:
            assert file_hash(receipt_file.parent / item["path"]) == item["sha256"]


def test_model_output_has_best_last_class_schema_and_evaluation_report(workspace, tmp_path):
    registry = workspace["service"].registry
    destination = registry.register("Models and reports", str(tmp_path / "模型 输出"))
    workflow = workspace["workflow"]
    profile = workflow.create_profile(ProfileCreate(key="models", name="Models", config=TaskProfileConfig(outputs={"destination_id": destination["id"]})))
    template = next(item for item in workflow.list_templates() if item["key"] == "confirmed_train_evaluate_v1")
    run = workflow.create_run(RunCreate(idempotency_key="model-output", template_id=template["id"], profile_id=profile["id"],
        dataset_id="ds_storage", image_ids=["img_confirmed"]))
    root = workspace["settings"].data_dir
    with workspace["database"].session_factory() as session:
        session.add(TrainingTask(id="train_output", dataset_id="ds_storage", name="fixture", status="completed", task_type="detect",
            model_name="yolo11n.pt", model_path=str(root / "models" / "base.pt"), epochs=1, img_size=64, batch_size=1,
            device="cpu", workers=0, val_ratio=0.2, seed=42))
        session.flush()
        schema = [{"id": "cls_storage", "index": 0, "name": "part"}]
        for kind in ("best", "last"):
            path = workspace["storage"].model_version_dir("model_" + kind) / (kind + ".pt")
            path.write_bytes((kind + "-fixture-copy").encode())
            session.add(ModelVersion(id="model_" + kind, name=kind, training_task_id="train_output", dataset_id="ds_storage",
                artifact_type=kind, format="pt", task_type="detect", model_path=str(path), class_schema_json=schema,
                metrics_json={"map50": 0.9}))
        session.flush()
        evaluation_root = workspace["storage"].evaluation_task_dir("evaluation_output")
        log = evaluation_root / "evaluation.log"
        log.write_text("completed evaluation fixture", encoding="utf-8")
        plot = evaluation_root / "result.png"
        Image.new("RGB", (20, 20), "white").save(plot)
        session.add(ModelEvaluationRecord(id="evaluation_output", model_id="model_best", dataset_id="ds_storage", split="val",
            status="completed", confidence=0.25, iou=0.5, logs_path=str(log), result_json={"metrics": {"map50": 0.9}, "artifacts": {"plot": str(plot)}}))
        session.flush()
        workflow._artifact(session, run["id"], None, None, "model", {"model_id": "model_best"})
        workflow._artifact(session, run["id"], None, None, "evaluation", {"evaluation_id": "evaluation_output"})
        session.get(WorkflowRun, run["id"]).status = "completed"
        session.commit()
    assert workflow._publish_outputs(run["id"])
    model_root = Path(workflow.get_run(run["id"])["context"]["output_destinations"]["models"]["run_path"])
    package_dir = next(path.parent for path in model_root.glob("*/package.json"))
    assert (package_dir / "best.pt").read_bytes().startswith(b"best")
    assert (package_dir / "last.pt").read_bytes().startswith(b"last")
    package = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
    assert all(model["class_schema_json"] == schema for model in package["models"])
    assert workflow._publish_outputs(run["id"])
    assert len(list(model_root.glob("*/package.json"))) == 1
    evaluation_output = Path(workflow.get_run(run["id"])["context"]["output_destinations"]["evaluations"]["run_path"])
    report_file = next(evaluation_output.glob("*/report.json"))
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["artifacts"]["plot"] == "figures/result.png"
    assert (report_file.parent / report["artifacts"]["plot"]).is_file()
