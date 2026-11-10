from __future__ import annotations

from contextlib import contextmanager
import threading
from collections.abc import Iterator


class LocalComputeGate:
    """Serializes local training, inference, SAM and evaluation on one machine."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    @contextmanager
    def acquire(self) -> Iterator[None]:
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()


local_compute_gate = LocalComputeGate()


def release_local_models() -> None:
    """Called inside the compute gate before launching a separate GPU process."""
    from app.models.inference import ModelCache
    from app.sam.service import clear_model_cache

    ModelCache.clear()
    clear_model_cache()
