"""Run history is recoverable while workflow assets and execution stay consistent."""
from pathlib import Path

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine, func, inspect, select, text

from app.api.error_handlers import register_error_handlers
from app.core.models import Annotation
from app.workflow.models import WorkflowArtifact, WorkflowAttempt, WorkflowEvent, WorkflowItem, WorkflowReview, WorkflowRun
from app.workflow.router import get_workflow_service, router
from app.workflow.schemas import NodeDefinition, ReviewDecision, ReviewDraft, RunCreate, TemplateCreate
from test_workflow import drain, workflow


@pytest.fixture
def history(workflow, tmp_path):
    service, database, dataset_id, images, profile, _, calls = workflow
    service.project_root = tmp_path
    template = service.create_template(TemplateCreate(key="trash-history", name="运行记录", nodes=[
        NodeDefinition(id="propose", type="vision_propose"),
        NodeDefinition(id="human", type="human_review"),
    ]))
    request = RunCreate(idempotency_key="trash-history", template_id=template["id"], profile_id=profile["id"],
                        dataset_id=dataset_id, image_ids=images)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_workflow_service] = lambda: service
    with TestClient(app) as client:
        yield client, service, database, request, calls


def completed(service, request):
    run = drain(service, service.create_run(request)["id"])
    for review in run["reviews"]:
        service.decide_review(review["id"], ReviewDecision(expected_revision=1, decision="approved"))
    run = drain(service, run["id"])
    assert run["status"] == "completed"
    return run


def test_http_delete_restore_are_idempotent_and_preserve_review_labels_and_files(history):
    client, service, database, request, calls = history
    run = completed(service, request)
    review = run["reviews"][0]
    exported = service.save_review_labelme(review["id"], ReviewDraft(expected_revision=1, payload={}))
    export_path = Path(exported["saved_path"])
    export_bytes = export_path.read_bytes()
    with database.session_factory() as session:
        artifacts = list(session.scalars(select(WorkflowArtifact).where(WorkflowArtifact.run_id == run["id"])))
        files = {service.artifacts.path(row.storage_name): service.artifacts.path(row.storage_name).read_bytes() for row in artifacts}
        before = {model: session.scalar(select(func.count()).select_from(model))
                  for model in (WorkflowRun, WorkflowItem, WorkflowAttempt, WorkflowArtifact, WorkflowReview, Annotation)}
    assert client.get("/api/workflows/runs").json()[0]["can_delete"] is True
    assert client.get("/api/workflows/runs?deleted=true").json() == []
    first = client.delete(f"/api/workflows/runs/{run['id']}")
    assert first.status_code == 200, first.text
    assert first.json()["deleted_at"]
    assert first.json()["reviews"] == client.get(f"/api/workflows/runs/{run['id']}").json()["reviews"]
    assert client.delete(f"/api/workflows/runs/{run['id']}").json()["deleted_at"] == first.json()["deleted_at"]
    assert client.get("/api/workflows/runs").json() == []
    assert client.get("/api/workflows/runs?deleted=true").json()[0]["id"] == run["id"]
    assert service.create_run(request)["id"] == run["id"]
    assert run["id"] not in service.runnable_ids()
    restored = client.post(f"/api/workflows/runs/{run['id']}/restore")
    assert restored.status_code == 200 and restored.json()["deleted_at"] is None
    assert restored.json()["status"] == "completed"
    assert client.post(f"/api/workflows/runs/{run['id']}/restore").status_code == 200
    assert client.get("/api/workflows/runs?deleted=true").json() == []
    assert client.get("/api/workflows/runs").json()[0]["id"] == run["id"]
    assert not service.tick(run["id"])
    assert calls == {"train": [], "evaluate": []}
    assert export_path.read_bytes() == export_bytes
    assert client.get(exported["download_url"]).content == export_bytes
    assert all(path.read_bytes() == content for path, content in files.items())
    with database.session_factory() as session:
        assert {model: session.scalar(select(func.count()).select_from(model)) for model in before} == before
        management = list(session.scalars(select(WorkflowEvent.kind).where(
            WorkflowEvent.run_id == run["id"], WorkflowEvent.kind.in_(["run_deleted", "run_restored"]))))
        assert sorted(management) == ["run_deleted", "run_restored"]


@pytest.mark.parametrize("status", ["pending", "running", "waiting_task", "awaiting_review", "needs_attention", "cancelling"])
def test_active_history_requires_cancel_before_delete(history, status):
    client, service, database, request, _ = history
    run = service.create_run(request)
    with database.session_factory() as session:
        session.get(WorkflowRun, run["id"]).status = status
        session.commit()
    listed = client.get("/api/workflows/runs").json()[0]
    assert listed["can_delete"] is False
    response = client.delete(f"/api/workflows/runs/{run['id']}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "workflow_run_active"
    assert service.get_run(run["id"])["deleted_at"] is None


def test_cancel_then_delete_filters_before_limit_and_preserves_parent_child(history):
    client, service, database, request, _ = history
    older = service.cancel_run(service.create_run(request)["id"])
    newer = service.cancel_run(service.create_run(request.model_copy(update={"idempotency_key": "newer"}))["id"])
    with database.session_factory() as session:
        session.get(WorkflowRun, newer["id"]).parent_run_id = older["id"]
        session.commit()
    assert client.delete(f"/api/workflows/runs/{newer['id']}").status_code == 200
    assert client.get("/api/workflows/runs?limit=1").json()[0]["id"] == older["id"]
    assert client.get("/api/workflows/runs?deleted=true&limit=1").json()[0]["id"] == newer["id"]
    assert service.get_run(newer["id"])["parent_run_id"] == older["id"]
    assert client.delete(f"/api/workflows/runs/{older['id']}").status_code == 200
    assert client.post(f"/api/workflows/runs/{newer['id']}/restore").json()["parent_run_id"] == older["id"]
    assert service.get_run(older["id"])["deleted_at"] is not None


@pytest.mark.parametrize("active_status", ["running", "waiting_task", "cancelling"])
def test_terminal_run_with_active_attempt_cannot_be_deleted(history, active_status):
    client, service, database, request, _ = history
    run = drain(service, service.create_run(request)["id"])
    with database.session_factory() as session:
        session.get(WorkflowRun, run["id"]).status = "completed"
        session.get(WorkflowAttempt, run["attempts"][0]["id"]).status = active_status
        session.commit()
    assert client.delete(f"/api/workflows/runs/{run['id']}").status_code == 409
    assert service.get_run(run["id"])["can_delete"] is False


def test_terminal_parent_waits_for_active_child_before_delete(history):
    client, service, database, request, _ = history
    parent = service.cancel_run(service.create_run(request)["id"])
    child = service.create_run(request.model_copy(update={"idempotency_key": "child"}))
    with database.session_factory() as session:
        session.get(WorkflowRun, child["id"]).parent_run_id = parent["id"]
        session.commit()
    assert client.delete(f"/api/workflows/runs/{parent['id']}").status_code == 409
    service.cancel_run(child["id"])
    assert client.delete(f"/api/workflows/runs/{parent['id']}").status_code == 200


def test_terminal_record_waits_until_all_pending_outputs_are_published(history):
    client, service, database, request, _ = history
    run = completed(service, request)
    with database.session_factory() as session:
        session.add(WorkflowEvent(id="pending-output", run_id=run["id"], kind="output_pending",
                                  payload_json={"artifact_id": run["artifacts"][0]["id"], "category": "annotations"}))
        session.add(WorkflowEvent(id="unrelated-publication", run_id=run["id"], kind="output_published",
                                  payload_json={"intent_id": "another-output"}))
        session.commit()
    assert service.get_run(run["id"])["can_delete"] is False
    response = client.delete(f"/api/workflows/runs/{run['id']}")
    assert response.status_code == 409
    assert "仍在处理中" in response.json()["error"]["message"]
    assert service.get_run(run["id"])["deleted_at"] is None
    with database.session_factory() as session:
        session.add(WorkflowEvent(id="matching-publication", run_id=run["id"], kind="output_published",
                                  payload_json={"intent_id": "pending-output"}))
        session.commit()
    assert service.get_run(run["id"])["can_delete"] is True
    assert client.delete(f"/api/workflows/runs/{run['id']}").status_code == 200


def test_late_publisher_failure_cannot_reactivate_deleted_completed_record(history, monkeypatch):
    _, service, _, request, _ = history
    run = completed(service, request)

    def late_failure(_publisher, run_id):
        service.delete_run(run_id)
        raise OSError("late publication error")

    monkeypatch.setattr("app.storage_settings.outputs.WorkflowOutputPublisher.publish", late_failure)
    assert not service._publish_outputs(run["id"])
    current = service.get_run(run["id"])
    assert current["status"] == "completed" and current["deleted_at"] is not None
    assert current["error_message"] is None
    assert not any(event["kind"] == "output_failed" for event in current["events"])
    monkeypatch.setattr("app.storage_settings.outputs.WorkflowOutputPublisher.publish",
                        lambda *_: pytest.fail("Deleted records must not publish again"))
    assert not service.tick(run["id"])
    assert not service._tick_once(run["id"])


def test_unknown_record_returns_not_found(history):
    client, *_ = history
    assert client.delete("/api/workflows/runs/missing").status_code == 404
    assert client.post("/api/workflows/runs/missing/restore").status_code == 404


def test_legacy_run_migration_keeps_history_and_round_trips(tmp_path):
    backend = Path(__file__).resolve().parents[1]
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "alembic"))
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "0020_visual_workflows")
    engine = create_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(text("""INSERT INTO workflow_templates
                (id,key,version,name,nodes_json,created_at,updated_at)
                VALUES ('template','test',1,'Legacy template','[]','2026-01-01','2026-01-01')"""))
            connection.execute(text("""INSERT INTO workflow_task_profiles
                (id,key,version,name,config_json,created_at,updated_at)
                VALUES ('profile','test',1,'Legacy profile','{}','2026-01-01','2026-01-01')"""))
            connection.execute(text("""INSERT INTO workflow_runs
                (id,idempotency_key,fingerprint,template_id,profile_id,template_snapshot,profile_snapshot,
                 context_json,status,cursor,generation,request_count,cancel_requested,created_at,updated_at)
                VALUES ('legacy','legacy-key','hash','template','profile','{}','{}','{}',
                        'cancelled',0,0,0,1,'2026-01-01','2026-01-01')"""))
        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.execute(text("SELECT status,deleted_at FROM workflow_runs WHERE id='legacy'")).one()
            assert row.status == "cancelled" and row.deleted_at is None
        assert "ix_workflow_runs_deleted_at" in {index["name"] for index in inspect(engine).get_indexes("workflow_runs")}
        command.downgrade(config, "0020_visual_workflows")
        assert "deleted_at" not in {column["name"] for column in inspect(engine).get_columns("workflow_runs")}
        command.upgrade(config, "head")
        with engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM workflow_runs WHERE id='legacy'")).scalar_one() == 1
    finally:
        engine.dispose()
