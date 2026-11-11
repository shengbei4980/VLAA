from dataclasses import asdict

from fastapi import APIRouter, Body, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse

from app.core.errors import ValidationError
from app.workflow.contracts import CONTRACTS
from app.workflow.previews import MAX_PREVIEW_BYTES
from app.workflow.schemas import (
    BatchExportRequest, ContinueReviewEditing, ProfileCreate, ProjectCreate, ReviewDecision, ReviewDraft,
    RetryRequest, RunCreate, TemplateCreate,
)
from app.workflow.service import WorkflowService

router = APIRouter(prefix="/workflows", tags=["workflows"])


def get_workflow_service(request: Request) -> WorkflowService:
    service = getattr(request.app.state, "workflow_service", None)
    if service is None:
        raise ValidationError("workflow_runtime_unavailable", "The workflow runtime is not initialized.")
    return service


@router.get("/nodes")
def nodes():
    return [asdict(contract) for contract in CONTRACTS.values()]


@router.get("/profiles")
def profiles(service: WorkflowService = Depends(get_workflow_service)):
    return service.list_profiles()


@router.get("/projects")
def projects(service: WorkflowService = Depends(get_workflow_service)):
    return service.list_projects()


@router.post("/projects", status_code=201)
def create_project(payload: ProjectCreate, service: WorkflowService = Depends(get_workflow_service)):
    return service.create_project(payload.name, payload.root_path)


@router.post("/profiles", status_code=201)
def create_profile(payload: ProfileCreate, service: WorkflowService = Depends(get_workflow_service)):
    return service.create_profile(payload)


@router.get("/profiles/{profile_id}")
def profile(profile_id: str, service: WorkflowService = Depends(get_workflow_service)):
    return service.get_profile(profile_id)


@router.get("/templates")
def templates(service: WorkflowService = Depends(get_workflow_service)):
    return service.list_templates()


@router.post("/templates", status_code=201)
def create_template(payload: TemplateCreate, service: WorkflowService = Depends(get_workflow_service)):
    return service.create_template(payload)


@router.get("/templates/{template_id}")
def template(template_id: str, service: WorkflowService = Depends(get_workflow_service)):
    return service.get_template(template_id)


@router.get("/runs")
def runs(limit: int = Query(default=50, ge=1, le=200), deleted: bool = Query(default=False),
         service: WorkflowService = Depends(get_workflow_service)):
    return service.list_runs(limit, deleted=deleted)


@router.post("/runs", status_code=202)
def create_run(payload: RunCreate, service: WorkflowService = Depends(get_workflow_service)):
    return service.create_run(payload)


@router.post("/runs/prepare")
def prepare_run(payload: RunCreate, request: Request, service: WorkflowService = Depends(get_workflow_service)):
    from app.vision.settings import VisionSettingsService
    settings = VisionSettingsService(request.app.state.settings).get().model_dump(mode="json")
    return service.prepare_run(payload, settings)


@router.get("/runs/{run_id}")
def run(run_id: str, summary: bool = Query(default=False), service: WorkflowService = Depends(get_workflow_service)):
    return service.get_run(run_id, summary=summary)


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, service: WorkflowService = Depends(get_workflow_service)):
    return service.delete_run(run_id)


@router.post("/runs/{run_id}/restore")
def restore_run(run_id: str, service: WorkflowService = Depends(get_workflow_service)):
    return service.restore_run(run_id)


@router.get("/runs/{run_id}/export")
def download_export(run_id: str, service: WorkflowService = Depends(get_workflow_service)):
    path = service.export_download(run_id)
    return FileResponse(path, media_type="application/zip", filename=path.name)


@router.post("/runs/{run_id}/annotation-batch", status_code=201)
def export_annotation_batch(run_id: str, payload: BatchExportRequest = Body(default_factory=BatchExportRequest),
                            service: WorkflowService = Depends(get_workflow_service)):
    return service.export_annotation_batch(run_id, payload.include_previews)


@router.get("/runs/{run_id}/annotation-batch/{file_name}")
def download_annotation_batch(run_id: str, file_name: str, service: WorkflowService = Depends(get_workflow_service)):
    path = service.annotation_batch_download(run_id, file_name)
    return FileResponse(path, media_type="application/zip", filename=path.name)


@router.post("/runs/{run_id}/cancel")
def cancel(run_id: str, service: WorkflowService = Depends(get_workflow_service)):
    return service.cancel_run(run_id)


@router.post("/runs/{run_id}/retry", status_code=202)
@router.post("/runs/{run_id}/resume", status_code=202)
def retry(run_id: str, payload: RetryRequest = Body(default_factory=RetryRequest),
          service: WorkflowService = Depends(get_workflow_service)):
    return service.retry_run(run_id, payload.item_ids)


@router.get("/reviews/{review_id}")
def review(review_id: str, service: WorkflowService = Depends(get_workflow_service)):
    return service.get_review(review_id)


@router.post("/reviews/{review_id}/preview", status_code=201)
async def save_preview(review_id: str, file: UploadFile = File(...), expected_revision: int = Form(..., ge=1),
                       has_unsaved_changes: bool = Form(False), service: WorkflowService = Depends(get_workflow_service)):
    try:
        content = await file.read(MAX_PREVIEW_BYTES + 1)
    finally:
        await file.close()
    return service.save_review_preview(review_id, content, expected_revision, has_unsaved_changes)


@router.get("/reviews/{review_id}/preview/{file_name}")
def download_preview(review_id: str, file_name: str, service: WorkflowService = Depends(get_workflow_service)):
    path = service.preview_download(review_id, file_name)
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.put("/reviews/{review_id}/draft")
def save_review(review_id: str, payload: ReviewDraft, service: WorkflowService = Depends(get_workflow_service)):
    return service.save_review(review_id, payload)


@router.post("/reviews/{review_id}/continue-editing", status_code=201)
def continue_editing(review_id: str, payload: ContinueReviewEditing,
                     service: WorkflowService = Depends(get_workflow_service)):
    return service.continue_editing(review_id, payload.expected_revision)


@router.post("/reviews/{review_id}/labelme", status_code=201)
def save_labelme(review_id: str, payload: ReviewDraft, service: WorkflowService = Depends(get_workflow_service)):
    return service.save_review_labelme(review_id, payload)


@router.get("/reviews/{review_id}/labelme/{file_name}")
def download_labelme(review_id: str, file_name: str, service: WorkflowService = Depends(get_workflow_service)):
    path = service.labelme_download(review_id, file_name)
    return FileResponse(path, media_type="application/json", filename=path.name)


@router.post("/reviews/{review_id}/decide")
def decide_review(review_id: str, payload: ReviewDecision, service: WorkflowService = Depends(get_workflow_service)):
    return service.decide_review(review_id, payload)
