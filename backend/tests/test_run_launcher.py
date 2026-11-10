import io
import json
from unittest.mock import Mock

import pytest

import run as launcher


def test_unrelated_frontend_is_not_reused(monkeypatch):
    monkeypatch.setattr(launcher.HTTP, "open", lambda *args, **kwargs: io.BytesIO(b"<title>Other app</title>"))
    assert not launcher.is_ready(5173)


@pytest.mark.parametrize("name", ["Vision-Language Automated Annotation", "YoloWebAgentStarter"])
def test_current_and_legacy_frontend_names_are_reused(monkeypatch, name):
    content = f"<title>{name}</title>".encode()
    monkeypatch.setattr(launcher.HTTP, "open", lambda *args, **kwargs: io.BytesIO(content))
    assert launcher.is_ready(5173)


@pytest.mark.parametrize("name", ["Vision-Language Automated Annotation", "YoloWebAgentStarter"])
def test_current_and_legacy_backend_names_are_reused(monkeypatch, tmp_path, name):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    payload = {"name": name, "data_dir": str(launcher.startup_paths(tmp_path)[0])}
    monkeypatch.setattr(launcher.HTTP, "open", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()))
    assert launcher.is_ready(8000)


@pytest.mark.parametrize("name", ["Vision-Language Automated Annotation", "YoloWebAgentStarter"])
def test_backend_from_another_workspace_is_not_reused(monkeypatch, tmp_path, name):
    payload = {"name": name, "data_dir": str(tmp_path / "other-data")}
    monkeypatch.setattr(launcher.HTTP, "open", lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()))
    assert not launcher.is_ready(8000)


def test_api_exposes_the_product_name(client):
    assert client.get("/api/system/info").json()["name"] == "Vision-Language Automated Annotation"
    assert client.get("/openapi.json").json()["info"]["title"] == "Vision-Language Automated Annotation API"


def test_running_services_only_open_browser(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "is_ready", lambda port: True)
    start = Mock(side_effect=AssertionError("Must not start duplicate services"))
    browser = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", start)
    monkeypatch.setattr(launcher.webbrowser, "open", browser)
    launcher.main()
    start.assert_not_called()
    browser.assert_called_once_with(launcher.URL)


def test_startup_failure_cleans_up_its_process(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "is_ready", lambda port: False)
    process = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(launcher, "wait_for_service", Mock(side_effect=RuntimeError("Startup failed")))
    stop = Mock()
    monkeypatch.setattr(launcher, "stop_process", stop)
    with pytest.raises(RuntimeError, match="Startup failed"):
        launcher.main()
    stop.assert_called_once_with(process)
