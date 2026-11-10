"""Start the local Windows workbench using its existing Python and Node environments."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener
import webbrowser

from backend.app.storage_settings.bootstrap import startup_paths

ROOT = Path(__file__).resolve().parent
PRODUCT_NAME = "Vision-Language Automated Annotation"
URL = "http://127.0.0.1:5173/"
HTTP = build_opener(ProxyHandler({}))


def is_ready(port: int) -> bool:
    path = "/api/system/info" if port == 8000 else "/"
    try:
        with HTTP.open(f"http://127.0.0.1:{port}{path}", timeout=2) as response:
            content = response.read(1024 * 1024)
        if port == 5173:
            return any(f"<title>{name}</title>".encode() in content
                       for name in (PRODUCT_NAME, "YoloWebAgentStarter"))
        info = json.loads(content)
        return (
            isinstance(info, dict)
            and info.get("name") in (PRODUCT_NAME, "YoloWebAgentStarter")
            and isinstance(info.get("data_dir"), str)
            and Path(info["data_dir"]).resolve() == startup_paths(ROOT)[0]
        )
    except (OSError, ValueError):
        return False


def wait_for_service(port: int, process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service on port {port} exited. Check {log_path}")
        if is_ready(port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"Service on port {port} did not become ready. Check {log_path}")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        # The venv launcher and training workers can be child processes on Windows.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        process.wait(timeout=10)


def main() -> None:
    print(f"{PRODUCT_NAME} (VLAA)", flush=True)
    data_dir, _ = startup_paths(ROOT)
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=str(ROOT / "backend"),
        PYTHONUTF8="1",
        YOLO_CONFIG_DIR=str(data_dir),
        MPLCONFIGDIR=str(data_dir / "matplotlib"),
    )
    services = (
        (
            "backend", 8000, ROOT,
            [
                str(ROOT / ".venv/Scripts/python.exe"), "-m", "uvicorn", "app.main:app",
                "--host", "127.0.0.1", "--port", "8000",
            ],
        ),
        (
            "frontend", 5173, ROOT / "frontend",
            [
                shutil.which("node") or "node",
                str(ROOT / "frontend/node_modules/vite/bin/vite.js"),
                "--host", "127.0.0.1", "--port", "5173", "--strictPort",
            ],
        ),
    )
    started: list[tuple[str, subprocess.Popen]] = []
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name, port, cwd, command in services:
            if is_ready(port):
                print(f"{name} is already running on port {port}.", flush=True)
                continue
            log_path = log_dir / f"launcher-{name}.log"
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            started.append((name, process))
            print(f"Starting {name}...", flush=True)
            wait_for_service(port, process, log_path)
        print(f"VLAA workbench ready: {URL}", flush=True)
        webbrowser.open(URL)
        if started:
            print("Keep this window open. Press Ctrl+C to stop services started here.", flush=True)
        while started:
            for name, process in started:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited. Check {log_dir / ('launcher-' + name + '.log')}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping services started by this launcher...", flush=True)
    finally:
        for _, process in reversed(started):
            stop_process(process)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        if sys.stdin.isatty():
            input("Press Enter to close...")
        raise SystemExit(1)
