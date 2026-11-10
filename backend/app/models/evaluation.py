from __future__ import annotations

import subprocess
import threading
import os
import sys
from pathlib import Path

from sqlalchemy import select, update

from app.core.models import ModelEvaluationRecord, ModelVersion
from app.core.time import utc_now
from app.core.storage import Storage
from app.training.observability.log_store import TrainingLogStore
from app.training.observability.metrics import TrainingMetricsParser
from app.training.runtime.runner import resolve_yolo_command_prefix
from app.training.runtime.process_registry import process_registry
from app.core.local_compute import local_compute_gate, release_local_models


def build_evaluation_command(record: ModelEvaluationRecord, model: ModelVersion) -> list[str]:
    if not record.data_path or not record.run_dir:
        raise ValueError("Evaluation export or run directory is missing.")
    run_path = Path(record.run_dir)
    prefix = [sys.executable, "-m", "app.training.maskaware"] if (record.result_json or {}).get("mask_aware") else resolve_yolo_command_prefix()
    return [
        *prefix,
        model.task_type,
        "val",
        f"model={model.model_path}",
        f"data={record.data_path}",
        f"split={record.split}",
        # Keep predictions below the review threshold so the upstream error
        # sample pass can identify low-confidence results after validation.
        f"conf={min(record.confidence, 0.001) if record.confidence > 0 else 0.001}",
        f"iou={record.iou}",
        f"project={run_path.parent}",
        f"name={run_path.name}",
        "plots=True",
        "save_json=True",
        "exist_ok=True",
    ]


def run_evaluation_process(record: ModelEvaluationRecord, model: ModelVersion) -> tuple[int, str, dict[str, float]]:
    command = build_evaluation_command(record, model)
    log_store = TrainingLogStore(record.logs_path or Path(record.run_dir or ".") / "evaluation.log")
    log_store.append("Command: " + " ".join(command))
    with local_compute_gate.acquire():
        release_local_models()
        if process_registry.stop_requested(record.id):
            process_registry.unregister(record.id)
            return -1, log_store.read(), {}
        process = subprocess.Popen(command,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            shell=False, bufsize=1, start_new_session=os.name == "posix",
            env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[2]), os.getenv("PYTHONPATH")]))})
        process_registry.register(record.id, process)
        lines: list[str] = []
        try:
            if process_registry.stop_requested(record.id):
                process_registry.request_stop(record.id)
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip("\n")
                lines.append(clean)
                log_store.append(clean)
            return_code = process.wait()
        finally:
            process_registry.unregister(record.id)
    return return_code, log_store.read(), parse_validation_metrics("\n".join(lines), model.task_type)


def parse_validation_metrics(text: str, task_type: str) -> dict[str, float]:
    return TrainingMetricsParser().parse_validation_text(text, task_type)


class YoloEvaluationRunner:
    """Community port of upstream app/evaluation/runner.py."""

    def __init__(self, session_factory, storage: Storage) -> None:
        self.session_factory = session_factory
        self.storage = storage

    def start_task(self, task_id: str) -> None:
        threading.Thread(target=self.run, args=(task_id,), name=f"ywa-evaluation-{task_id}", daemon=True).start()

    def run(self, task_id: str) -> None:
        from app.models.service import ModelService

        with self.session_factory() as session:
            claimed = session.execute(update(ModelEvaluationRecord).where(
                ModelEvaluationRecord.id == task_id, ModelEvaluationRecord.status == "pending",
            ).values(status="running", started_at=utc_now()))
            if claimed.rowcount != 1:
                session.rollback()
                return
            session.commit()
            ModelService(self.storage).run_evaluation(session, task_id)

    def recover_orphaned(self) -> None:
        """Fail interrupted work and restart pending tasks after a local restart."""

        with self.session_factory() as session:
            running = list(session.scalars(select(ModelEvaluationRecord).where(ModelEvaluationRecord.status == "running")))
            pending_ids = list(session.scalars(select(ModelEvaluationRecord.id).where(ModelEvaluationRecord.status == "pending")))
            for record in running:
                record.status = "failed"
                record.error_message = "Evaluation process was interrupted by a service restart."
                record.finished_at = utc_now()
            if running:
                session.commit()
        for task_id in pending_ids:
            self.start_task(task_id)
