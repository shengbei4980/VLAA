from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from app.storage_settings.registry import validate_directory
from app.storage_settings.service import StorageSettingsService
from app.storage_settings.manual_exports import manual_export_locations, save_manual_export_locations

router = APIRouter(prefix="/storage", tags=["storage"])


class PathRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1, max_length=4096)


class DestinationRequest(PathRequest):
    name: str = Field(min_length=1, max_length=255)


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plan_id: str


class ManualExportLocationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    json_path: str = Field(min_length=1, max_length=4096)
    images_path: str = Field(min_length=1, max_length=4096)


def service(request: Request):
    return StorageSettingsService(request.app.state.settings, request.app.state.database)


@router.get("/settings")
def settings(request: Request):
    return service(request).get()


@router.get("/destinations")
def destinations(request: Request):
    return service(request).registry.list()


@router.get("/manual-exports")
def manual_exports(request: Request):
    return manual_export_locations(request.app.state.settings)


@router.put("/manual-exports")
def save_manual_exports(payload: ManualExportLocationsRequest, request: Request):
    return save_manual_export_locations(request.app.state.settings, payload.json_path, payload.images_path)


@router.post("/destinations/validate")
def validate(payload: PathRequest):
    return validate_directory(payload.path)


@router.post("/destinations", status_code=201)
def register(payload: DestinationRequest, request: Request):
    return service(request).registry.register(payload.name, payload.path)


@router.post("/data-root/prepare")
def prepare(payload: PathRequest, request: Request):
    return service(request).prepare(payload.path)


@router.post("/data-root/execute")
def execute(payload: ExecuteRequest, request: Request):
    return service(request).execute(payload.plan_id)


@router.post("/data-root/cancel")
def cancel(request: Request):
    return service(request).cancel()
