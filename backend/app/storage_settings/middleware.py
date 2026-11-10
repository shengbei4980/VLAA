from starlette.responses import JSONResponse
import re

from app.storage_settings.gate import gate_for, pending_restart


class StorageWriteGuard:
    """Drain existing mutations, then freeze writes through migration and restart."""
    def __init__(self, app, settings):
        self.app = app
        self.settings = settings
        self.gate = gate_for(settings.project_root)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        exempt = path in {"/api/storage/data-root/execute", "/api/storage/data-root/cancel"}
        telemetry_write = path == "/api/training/tasks" or re.fullmatch(r"/api/training/tasks/[^/]+(?:/summary)?", path) is not None
        mutation = (scope.get("method") not in {"GET", "HEAD", "OPTIONS"} or "/export/" in path
                    or path.endswith("/export") or telemetry_write)
        if exempt or not mutation:
            return await self.app(scope, receive, send)
        if not self.gate.enter_mutation(lambda: pending_restart(self.settings)):
            response = JSONResponse({"error": {"code": "storage_write_locked", "message": "Storage migration requires read-only access until restart or cancellation.", "details": {}}}, status_code=423)
            return await response(scope, receive, send)
        try:
            return await self.app(scope, receive, send)
        finally:
            self.gate.leave_mutation()
