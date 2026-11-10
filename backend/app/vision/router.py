from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, get_settings, get_storage
from app.core.config import Settings
from app.core.storage import Storage
from app.dataset.annotation.masks import MaskArtifact, MaskService
from app.sam.schemas import SamMaskPredictResponse, SamPredictRequest
from app.sam.service import SamService
from app.vision.schemas import VisionProbeResult, VisionRole, VisionSettingsOut, VisionSettingsUpdate
from app.vision.service import VisionService
from app.vision.settings import VisionSettingsService

router = APIRouter(prefix="/vision", tags=["vision"])


class MaskWriteRequest(BaseModel):
    rle: dict


@router.get("/settings", response_model=VisionSettingsOut)
def settings_get(settings: Settings = Depends(get_settings)) -> VisionSettingsOut:
    return VisionSettingsService(settings).get()


@router.put("/settings", response_model=VisionSettingsOut)
def settings_update(payload: VisionSettingsUpdate, settings: Settings = Depends(get_settings)) -> VisionSettingsOut:
    return VisionSettingsService(settings).update(payload)


@router.post("/probe/{role}", response_model=VisionProbeResult)
def probe(role: VisionRole, storage: Storage = Depends(get_storage), settings: Settings = Depends(get_settings)) -> VisionProbeResult:
    return VisionService(storage, settings).probe(role)


@router.post("/masks", response_model=MaskArtifact, status_code=201)
def mask_write(payload: MaskWriteRequest, storage: Storage = Depends(get_storage)) -> MaskArtifact:
    return MaskService(storage).write(payload.rle)


@router.get("/masks/{digest}")
def mask_read(digest: str, storage: Storage = Depends(get_storage)) -> dict:
    service = MaskService(storage)
    reference = f"masks/{digest}.json"
    rle = service.read(reference)
    artifact = service.write(rle)
    return {**artifact.model_dump(), "rle": rle}


@router.post("/sam/predict", response_model=SamMaskPredictResponse)
def sam_mask(payload: SamPredictRequest, session: Session = Depends(get_session), storage: Storage = Depends(get_storage), settings: Settings = Depends(get_settings)) -> SamMaskPredictResponse:
    return SamService(storage, settings).predict_mask(session, payload)
