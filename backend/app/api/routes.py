from fastapi import APIRouter

from app.api.routers import datasets, models, sam, settings, system, training
from app.agent.router import router as agent_router
from app.auto_annotation.router import dataset_router as auto_annotation_dataset_router, task_router as auto_annotation_task_router
from app.logs.router import router as logs_router
from app.video_import.router import router as video_import_router
from app.dataset.annotation.review_router import router as review_router
from app.workflow.router import router as workflow_router
from app.vision.router import router as vision_router
from app.dataset.exchange.labelme_router import router as labelme_router
from app.storage_settings.router import router as storage_router

api_router = APIRouter(prefix="/api")
api_router.include_router(system.router)
api_router.include_router(datasets.router)
api_router.include_router(datasets.file_router)
api_router.include_router(training.router)
api_router.include_router(models.router)
api_router.include_router(sam.router)
api_router.include_router(settings.router)
api_router.include_router(logs_router)
api_router.include_router(auto_annotation_dataset_router)
api_router.include_router(auto_annotation_task_router)
api_router.include_router(video_import_router)
api_router.include_router(review_router)
api_router.include_router(workflow_router)
api_router.include_router(vision_router)
api_router.include_router(labelme_router)
api_router.include_router(storage_router)
api_router.include_router(agent_router)
