"""PNG preview files are revision-bound exports, separate from reviewed labels."""
import io
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import func, select

from app.api.error_handlers import register_error_handlers
from app.core.models import Annotation
from app.workflow.models import WorkflowArtifact
from app.workflow.router import get_workflow_service, router
from app.workflow.schemas import ReviewDecision
from test_workflow import drain, start, workflow


def image_bytes(size=(80, 400), format_name="PNG"):
    content = io.BytesIO()
    Image.new("RGB", size, "#22c55e").save(content, format=format_name)
    return content.getvalue()


def damaged_png_checksum():
    content = bytearray(image_bytes())
    chunk = content.index(b"IDAT")
    length = int.from_bytes(content[chunk - 4:chunk], "big")
    content[chunk + 4 + length] ^= 1
    return bytes(content)


@pytest.fixture
def preview(workflow):
    service, database, *_ = workflow
    run = drain(service, start(workflow)["id"])
    assert run["status"] == "awaiting_review"
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_workflow_service] = lambda: service
    with TestClient(app) as client:
        yield client, service, database, run


def save(client, review, content=None, *, revision=None, dirty=False):
    return client.post(f"/api/workflows/reviews/{review['id']}/preview",
        data={"expected_revision": revision if revision is not None else review["revision"],
              "has_unsaved_changes": str(dirty).lower()},
        files={"file": ("preview.png", image_bytes() if content is None else content, "image/png")})


def test_preview_is_saved_downloadable_and_does_not_change_labels_or_review(preview):
    client, service, database, run = preview
    review = run["reviews"][0]
    with database.session_factory() as session:
        artifacts_before = session.scalar(select(func.count()).select_from(WorkflowArtifact))
        labels_before = session.scalar(select(func.count()).select_from(Annotation))
    content = image_bytes()
    response = save(client, review, content)
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["preview_state"] == "draft"
    assert result["review_revision"] == review["revision"]
    assert (result["width"], result["height"]) == (80, 400)
    path = Path(result["saved_path"])
    assert path.parent == service.storage.exports_dir
    assert path.read_bytes() == content
    assert list(path.parent.glob("*.tmp")) == []
    downloaded = client.get(result["download_url"])
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "image/png"
    assert downloaded.content == content
    with Image.open(io.BytesIO(downloaded.content)) as image:
        assert image.size == (80, 400)
    # Retrying an identical export reuses its complete content-addressed file.
    assert save(client, review, content).json()["saved_path"] == result["saved_path"]
    assert service.get_review(review["id"]) == review
    with database.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(WorkflowArtifact)) == artifacts_before
        assert session.scalar(select(func.count()).select_from(Annotation)) == labels_before


@pytest.mark.parametrize("content,code", [
    (b"not a PNG", "workflow_preview_invalid_png"),
    (image_bytes(format_name="JPEG"), "workflow_preview_invalid_png"),
    (image_bytes(size=(81, 400)), "workflow_preview_dimensions_mismatch"),
    (image_bytes()[:40], "workflow_preview_invalid_png"),
    (damaged_png_checksum(), "workflow_preview_invalid_png"),
], ids=["text", "jpeg", "dimensions", "truncated", "checksum"])
def test_invalid_preview_content_is_rejected_without_an_output_file(preview, content, code):
    client, service, _, run = preview
    response = save(client, run["reviews"][0], content)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code
    assert list(service.storage.exports_dir.glob("*.png")) == []


def test_stale_and_oversized_previews_are_rejected(preview, monkeypatch):
    client, service, _, run = preview
    review = run["reviews"][0]
    response = save(client, review, revision=review["revision"] + 1)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "workflow_preview_stale"
    monkeypatch.setattr("app.workflow.previews.MAX_PREVIEW_BYTES", 10)
    response = save(client, review)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "workflow_preview_too_large"
    assert list(service.storage.exports_dir.glob("*.png")) == []


def test_download_is_bound_to_its_review_and_validates_file_integrity(preview):
    client, _, _, run = preview
    review, other_review = run["reviews"]
    result = save(client, review).json()
    response = client.get(f"/api/workflows/reviews/{other_review['id']}/preview/{result['file_name']}")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "workflow_preview_unsafe"
    response = client.get(f"/api/workflows/reviews/{review['id']}/preview/another-export.zip")
    assert response.status_code == 422
    missing_name = f"{review['id']}-r1-draft-{'0' * 64}.png"
    assert client.get(f"/api/workflows/reviews/{review['id']}/preview/{missing_name}").status_code == 404
    Path(result["saved_path"]).write_bytes(b"changed")
    response = client.get(result["download_url"])
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "workflow_preview_changed"
    assert save(client, review).status_code == 409


def test_confirmed_preview_with_unsaved_edits_is_still_a_draft(preview):
    client, service, _, run = preview
    review = run["reviews"][0]
    confirmed = service.decide_review(review["id"], ReviewDecision(expected_revision=review["revision"], decision="approved"))
    assert confirmed["status"] == "decided"
    result = save(client, confirmed).json()
    assert result["preview_state"] == "confirmed"
    assert "-confirmed-" in result["file_name"]
    result = save(client, confirmed, dirty=True).json()
    assert result["preview_state"] == "draft"
    assert "-draft-" in result["file_name"]
    assert service.get_review(review["id"]) == confirmed


def test_cancelled_run_remains_available_for_readonly_preview_export(preview):
    client, service, _, run = preview
    service.cancel_run(run["id"])
    assert service.get_run(run["id"])["status"] == "cancelled"
    review = service.get_review(run["reviews"][0]["id"])
    result = save(client, review)
    assert result.status_code == 201
    assert service.get_run(run["id"])["status"] == "cancelled"
