"""Progress is measured work, not invented LLM token-level completion."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from app.workflow.models import WorkflowAttempt, WorkflowItem, WorkflowRun
from app.workflow.progress import run_progress
from test_workflow import workflow, start, drain  # noqa: F401

BASE = datetime(2026, 11, 11, tzinfo=timezone.utc)


def sample_run(count=3):
    run = WorkflowRun(status="running", profile_snapshot={"task_type": "detect"}, template_snapshot={"nodes": [
        {"id": "propose", "type": "vision_propose", "scope": "item"},
        {"id": "segment", "type": "segment_proposals", "scope": "item", "task_types": ["segment"]},
        {"id": "check", "type": "vision_review", "scope": "item"},
        {"id": "human", "type": "human_review", "scope": "item"},
        {"id": "train", "type": "train", "scope": "batch"},
    ]})
    items = [WorkflowItem(id=str(i), status="pending", confirmed=False, context_json={"file_name": f"{i}.png"}) for i in range(count)]
    return run, items


def attempt(item, node, begin, end=None, *, status="completed", number=1):
    return WorkflowAttempt(item_id=str(item), node_id=node, node_type="vision_propose" if node == "propose" else "vision_review",
                           attempt=number, status=status, created_at=BASE + timedelta(seconds=begin),
                           updated_at=BASE + timedelta(seconds=end if end is not None else begin),
                           finished_at=BASE + timedelta(seconds=end) if end is not None else None)


def test_eta_uses_completed_image_times_not_human_wait_or_skipped_segment():
    run, items = sample_run()
    # Image 1 costs 10s, image 2 costs 30s, with a long human/queue gap between them.
    attempts = [attempt(0, "propose", 0, 4), attempt(0, "check", 4, 10),
                attempt(0, "human", 10, 1000), attempt(1, "propose", 1000, 1010),
                attempt(1, "check", 1010, 1030), attempt(2, "propose", 1030, status="running")]
    progress = run_progress(run, items, attempts, now=BASE + timedelta(seconds=1035))
    assert progress["total_steps"] == 6
    assert progress["completed_steps"] == 4
    assert progress["ready_images"] == 2
    assert progress["percent"] == 66.7
    assert progress["processing_seconds"] == 45
    assert progress["average_image_seconds"] == 20
    assert progress["eta_seconds"] == 15
    assert progress["current"]["image_index"] == 3
    assert progress["current"]["elapsed_seconds"] == 5
    assert progress["eta_state"] == "estimated"
    slow = run_progress(run, items, attempts, now=BASE + timedelta(seconds=1100))
    assert slow["eta_state"] == "overdue"
    assert slow["eta_seconds"] is None


def test_retry_deduplicates_progress_and_includes_request_cost():
    run, items = sample_run(2)
    attempts = [attempt(0, "propose", 0, 5, status="failed"),
                attempt(0, "propose", 5, 10, number=2), attempt(0, "check", 10, 15)]
    progress = run_progress(run, items, attempts, now=BASE + timedelta(seconds=20))
    assert progress["ready_images"] == 1
    assert progress["completed_steps"] == 2
    assert progress["percent"] == 50
    assert progress["average_image_seconds"] == 15
    assert progress["eta_state"] == "learning"
    assert progress["eta_seconds"] is None


@pytest.mark.parametrize("status, state", [("needs_attention", "blocked"), ("cancelled", "cancelled"), ("cancelling", "cancelled")])
def test_stopped_states_never_offer_a_countdown(status, state):
    run, items = sample_run()
    run.status = status
    progress = run_progress(run, items, [], now=BASE)
    assert progress["eta_state"] == state
    assert progress["eta_seconds"] is None


def test_single_image_and_waiting_human_do_not_fake_eta_or_inflate_elapsed():
    run, items = sample_run(1)
    empty = run_progress(run, items, [], now=BASE)
    assert empty["eta_state"] == "learning" and empty["percent"] == 0
    run.status = "awaiting_review"
    items[0].status = "awaiting_review"
    attempts = [attempt(0, "propose", 0, 10), attempt(0, "check", 10, 20), attempt(0, "human", 20, status="awaiting_review")]
    progress = run_progress(run, items, attempts, now=BASE + timedelta(days=10))
    assert progress["eta_state"] == "complete"
    assert progress["percent"] == 100
    assert progress["waiting_review_images"] == 1
    assert progress["confirmed_images"] == 0
    assert progress["processing_seconds"] == 20
    assert progress["eta_seconds"] is None


def test_excluded_and_interrupted_images_do_not_count_as_success():
    run, items = sample_run(1)
    items[0].status = "excluded"
    progress = run_progress(run, items, [attempt(0, "propose", 0, 10000, status="interrupted")], now=BASE + timedelta(days=3))
    assert progress["ready_images"] == 0
    assert progress["excluded_images"] == 1
    assert progress["percent"] == 0
    assert progress["processing_seconds"] == 0
    assert progress["eta_state"] == "complete"


def test_training_only_template_does_not_report_annotation_percent():
    run, items = sample_run(0)
    run.template_snapshot["nodes"] = [{"id": "train", "type": "train", "scope": "batch"}]
    progress = run_progress(run, items, [], now=BASE)
    assert progress["percent"] is None
    assert progress["eta_state"] == "not_applicable"


def test_large_batch_never_rounds_incomplete_work_to_one_hundred_percent():
    run, items = sample_run(1000)
    attempts = [attempt(index, node, 0, 1) for index in range(1000) for node in ("propose", "check")]
    progress = run_progress(run, items, attempts[:-1], now=BASE + timedelta(seconds=2))
    assert progress["ready_images"] == 999
    assert progress["percent"] == 99.9


def test_custom_forward_route_removes_skipped_work_from_denominator():
    run, items = sample_run(1)
    run.template_snapshot["nodes"][0]["routes"] = {"valid": "human"}
    done = attempt(0, "propose", 0, 10)
    done.output_json = {"outcome": "valid"}
    progress = run_progress(run, items, [done], now=BASE + timedelta(seconds=20))
    assert progress["ready_images"] == 1
    assert progress["total_steps"] == 1
    assert progress["skipped_steps"] == 1
    assert progress["percent"] == 100


def test_summary_keeps_review_links_without_loading_annotation_payloads(workflow, monkeypatch):
    service, *_ = workflow
    run = start(workflow)
    full = drain(service, run["id"])
    def unexpected(*args):
        raise AssertionError("Progress polling must not hydrate review masks")
    monkeypatch.setattr(service, "_review_response", unexpected)
    summary = service.get_run(run["id"], summary=True)
    assert summary["progress"]["ready_images"] == 2
    assert summary["progress"]["waiting_review_images"] == 2
    assert summary["progress"]["confirmed_images"] == 0
    assert summary["progress"]["processing_seconds"] < 10
    assert summary["reviews"][0]["id"] == full["reviews"][0]["id"]
    assert "payload" not in summary["reviews"][0]
    assert "payload" not in summary["artifacts"][0]
    assert "existing_annotations" not in summary["items"][0]["context"]
    assert summary["items"][0]["file_name"].endswith(".png")
    assert len(json.dumps(summary, default=str)) < len(json.dumps(full, default=str))


def test_summary_http_query_is_explicit(client, monkeypatch):
    calls = []
    monkeypatch.setattr(client.app.state.workflow_service, "get_run", lambda run_id, **kwargs: calls.append((run_id, kwargs)) or {"id": run_id})
    assert client.get("/api/workflows/runs/example?summary=true").status_code == 200
    assert client.get("/api/workflows/runs/example").status_code == 200
    assert calls == [("example", {"summary": True}), ("example", {"summary": False})]
