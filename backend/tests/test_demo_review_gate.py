import importlib.util
from pathlib import Path


def test_demo_waits_for_human_receipts_and_applied_candidates(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts" / "demo_local.py"
    spec = importlib.util.spec_from_file_location("demo_review_test", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    responses = iter([
        {"total": 1, "pending": 1, "items": [{"has_draft": True}]},
        {"total": 1, "pending": 0, "items": [{"has_draft": True}]},
        {"total": 1, "pending": 0, "items": [{"has_draft": False}]},
    ])
    waits = []
    monkeypatch.setattr(demo, "api", lambda *args: next(responses))
    monkeypatch.setattr(demo.time, "sleep", waits.append)
    demo.wait_for_human_review(None, "demo-dataset")
    assert waits == [2, 2]
