from contextlib import contextmanager
from pathlib import Path
import threading

from app.core.errors import ConflictError
from app.storage_settings.bootstrap import read_config


class StorageGate:
    def __init__(self):
        self.condition = threading.Condition()
        self.mutations = 0
        self.exclusive = False

    def enter_mutation(self, pending) -> bool:
        with self.condition:
            if self.exclusive or pending():
                return False
            self.mutations += 1
            return True

    def leave_mutation(self):
        with self.condition:
            self.mutations -= 1
            self.condition.notify_all()

    @contextmanager
    def relocate(self):
        with self.condition:
            if self.exclusive:
                raise ConflictError("storage_migration_in_progress", "Another storage migration is in progress.")
            self.exclusive = True
            while self.mutations:
                self.condition.wait()
        try:
            yield
        finally:
            with self.condition:
                self.exclusive = False
                self.condition.notify_all()


_GATES = {}
_GATES_LOCK = threading.Lock()


def gate_for(project_root: Path) -> StorageGate:
    key = str(Path(project_root).resolve())
    with _GATES_LOCK:
        return _GATES.setdefault(key, StorageGate())


def pending_restart(settings) -> bool:
    config = read_config(settings.project_root)
    selected = config.get("data_dir")
    migration = config.get("migration") or {}
    return bool(selected and migration.get("status") == "ready" and Path(selected).resolve() != settings.data_dir.resolve())
