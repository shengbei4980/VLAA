"""App-owned scheduler. Database rows survive worker and service restarts."""
from __future__ import annotations

import logging
import threading

from app.workflow.service import WorkflowService

logger = logging.getLogger("ywa.workflow")


class WorkflowExecutor:
    def __init__(self, service: WorkflowService, poll_seconds: float = 1.0):
        self.service = service
        self.poll_seconds = poll_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        service.wake = self._wake.set

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self.service.ensure_defaults()
        self.service.recover()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ywa-workflow", daemon=True)
        self._thread.start()

    def submit(self, run_id: str | None = None):
        self._wake.set()

    def _run(self):
        while not self._stop.is_set():
            self._wake.clear()
            progressed = False
            try:
                for run_id in self.service.runnable_ids():
                    if self._stop.is_set():
                        break
                    progressed = self.service.tick(run_id) or progressed
            except Exception:
                logger.exception("Workflow scheduler iteration failed")
            if not progressed:
                self._wake.wait(self.poll_seconds)

    def shutdown(self):
        self._stop.set()
        self._wake.set()
        if self._thread:
            # An in-flight request finishes its bounded timeout before the DB closes.
            self._thread.join()
            self._thread = None
