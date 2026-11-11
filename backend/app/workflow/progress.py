"""Read-only progress from persisted attempts; human waiting is never timed as work."""
from __future__ import annotations

from datetime import datetime
from collections import defaultdict
from statistics import mean

from app.core.time import utc_now
from app.workflow.models import WorkflowAttempt, WorkflowItem, WorkflowRun


def run_progress(run: WorkflowRun, items: list[WorkflowItem], attempts: list[WorkflowAttempt],
                 *, now: datetime | None = None) -> dict:
    now = now or utc_now()
    nodes = run.template_snapshot["nodes"]
    automatic = []
    for node in nodes:
        if node["type"] == "human_review":
            break
        if node["scope"] == "item" and (
            not node.get("task_types") or run.profile_snapshot["task_type"] in node["task_types"]
        ):
            automatic.append(node)
    node_ids = {node["id"] for node in automatic}
    latest: dict[tuple[str | None, str], WorkflowAttempt] = {}
    histories: dict[str, list[WorkflowAttempt]] = defaultdict(list)
    for attempt in attempts:
        if attempt.item_id and attempt.node_id in node_ids:
            histories[attempt.item_id].append(attempt)
        key = (attempt.item_id, attempt.node_id)
        if key not in latest or attempt.attempt > latest[key].attempt:
            latest[key] = attempt

    def duration(attempt: WorkflowAttempt) -> float:
        if attempt.status == "interrupted":
            return 0.0  # The time the service was down cannot be inferred from this row.
        end = attempt.finished_at
        if end is None:
            end = now if attempt.status in {"running", "waiting_task", "cancelling"} else attempt.updated_at
        return max(0.0, (min(end, now) - attempt.created_at).total_seconds())

    intervals = []
    for attempt in attempts:
        if attempt.node_id not in node_ids or not attempt.item_id:
            continue
        seconds = duration(attempt)
        intervals.append((attempt.created_at.timestamp(), attempt.created_at.timestamp() + seconds))
    # Merge intervals to avoid double counting if execution gains parallel workers later.
    elapsed = 0.0
    previous_end = 0.0
    for start, end in sorted(intervals):
        elapsed += max(0.0, end - max(start, previous_end))
        previous_end = max(previous_end, end)

    ready = 0
    completed_steps = 0
    total_steps = 0
    skipped_steps = 0
    node_positions = {node["id"]: index for index, node in enumerate(nodes)}
    samples: list[tuple[datetime, float]] = []
    unfinished: list[WorkflowItem] = []
    for item in items:
        skipped = set()
        for node in automatic:
            previous = latest.get((item.id, node["id"]))
            if previous is not None and previous.status == "completed":
                target = node.get("routes", {}).get((previous.output_json or {}).get("outcome", "completed"))
                if target in node_positions:
                    skipped.update(value["id"] for value in nodes[node_positions[node["id"]] + 1:node_positions[target]])
        required = node_ids - skipped
        skipped_steps += len(node_ids & skipped)
        total_steps += len(required)
        completed = [latest[(item.id, node_id)] for node_id in required
                     if (item.id, node_id) in latest and latest[(item.id, node_id)].status == "completed"]
        completed_steps += len(completed)
        if automatic and len(completed) == len(required):
            ready += 1
            # All attempts count toward cost, but exclude interrupted work (unknown downtime).
            history = histories[item.id]
            if history and all(a.finished_at is not None for a in history):
                samples.append((max(a.finished_at for a in history), sum(duration(a) for a in history)))
        elif item.status != "excluded":
            unfinished.append(item)

    percent = round(100 * completed_steps / total_steps, 1) if total_steps else None
    if total_steps and completed_steps < total_steps:
        percent = min(percent, 99.9)
    active = next((a for a in reversed(attempts) if a.status in {"running", "waiting_task", "cancelling"}), None)
    current = None
    if active:
        item = next((item for item in items if item.id == active.item_id), None)
        current = {"node_id": active.node_id, "node_type": active.node_type,
                   "item_id": active.item_id, "image_index": items.index(item) + 1 if item else None,
                   "file_name": item.context_json.get("file_name") if item else None,
                   "elapsed_seconds": round(duration(active))}

    failed = sum(item.status == "failed" for item in items)
    waiting = sum(item.status == "awaiting_review" for item in items)
    excluded = sum(item.status == "excluded" for item in items)
    samples = sorted(samples)[-10:]
    average = mean(seconds for _, seconds in samples) if samples else None
    eta = None
    if not automatic:
        eta_state = "not_applicable"
    elif run.status in {"cancelled", "cancelling"}:
        eta_state = "cancelled"
    elif not unfinished:
        eta_state = "complete"
    elif failed or run.status in {"needs_attention", "interrupted", "failed"}:
        eta_state = "blocked"
    elif run.status == "awaiting_review":
        eta_state = "waiting_review"
    elif len(samples) < 2:
        eta_state = "learning"
    else:
        # Current executor is serial. Estimate only automatic work before the human gate.
        spent = {item.id: sum(duration(a) for a in histories[item.id]) for item in unfinished}
        if any(value >= average for value in spent.values()):
            eta_state = "overdue"
        else:
            eta = max(1, round(sum(average - spent[item.id] for item in unfinished)))
            eta_state = "estimated"

    return {
        "measured_at": now, "automatic_applicable": bool(automatic),
        "total_images": len(items), "ready_images": ready,
        "waiting_review_images": waiting, "failed_images": failed, "excluded_images": excluded,
        "confirmed_images": sum(item.confirmed for item in items),
        "completed_steps": completed_steps, "total_steps": total_steps, "skipped_steps": skipped_steps, "percent": percent,
        "processing_seconds": round(elapsed), "sample_count": len(samples),
        "average_image_seconds": round(average, 1) if average is not None else None,
        "eta_seconds": eta, "eta_state": eta_state, "current": current,
    }
