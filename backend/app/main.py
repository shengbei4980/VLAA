from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.error_handlers import register_error_handlers
from app.api.routes import api_router
from app.core.config import Settings
from app.core.database import Database
from app.core.migrations import upgrade_database
from app.core.storage import Storage
from app.training.runtime.queue import training_queue
from app.models.evaluation import YoloEvaluationRunner
from app.logs.service import configure_runtime_logging
from app.auto_annotation.queue import auto_annotation_queue
from app.video_import.queue import video_import_queue
from app.agent.service import AgentService
from app.agent.runtime import AgentRunExecutor
from app.workflow.service import WorkflowService
from app.workflow.bridges import WorkflowBindings
from app.workflow.runtime import WorkflowExecutor
from app.vision.workflow import build_vision_handlers
from app.storage_settings.middleware import StorageWriteGuard
from app.sam.configuration import freeze_sam_configuration


def create_app(settings: Settings | None = None, *, run_migrations: bool = True) -> FastAPI:
    resolved = settings or Settings.from_env()
    resolved.ensure_directories()
    configure_runtime_logging(resolved)
    database = Database(resolved.database_url)
    storage = Storage(resolved.data_dir, resolved.import_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if run_migrations:
            upgrade_database(resolved.project_root / "backend", resolved.database_url)
        # Alembic's fileConfig replaces the root handlers, so install the
        # Starter runtime handler again after migrations have completed.
        configure_runtime_logging(resolved)
        training_queue.configure(database.session_factory, storage)
        auto_annotation_queue.configure(database.session_factory, storage)
        video_import_queue.configure(database.session_factory, storage, resolved)
        YoloEvaluationRunner(database.session_factory, storage).recover_orphaned()
        training_queue.recover_orphaned()
        auto_annotation_queue.recover_orphaned()
        video_import_queue.recover_orphaned()
        AgentService(database.session_factory, resolved).recover_orphaned()
        app.state.agent_executor = AgentRunExecutor()
        app.state.workflow_service = WorkflowService(database.session_factory, storage, WorkflowBindings(
            handlers=build_vision_handlers(storage, resolved, database.session_factory),
            sam_resolver=lambda: freeze_sam_configuration(resolved)), project_root=resolved.project_root)
        app.state.workflow_executor = WorkflowExecutor(app.state.workflow_service)
        app.state.workflow_executor.start()
        import logging

        logging.getLogger("ywa").info("Vision-Language Automated Annotation (VLAA) runtime initialized")
        try:
            yield
        finally:
            app.state.workflow_executor.shutdown()
            app.state.agent_executor.shutdown()
            database.dispose()

    app = FastAPI(title="Vision-Language Automated Annotation API", version="0.2.0", lifespan=lifespan)
    app.state.settings = resolved
    app.state.database = database
    app.state.storage = storage
    app.add_middleware(StorageWriteGuard, settings=resolved)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()
