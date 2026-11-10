// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { WorkflowRunView } from "./WorkflowRunView";
import { workflowApi } from "../api/workflows";
import type { WorkflowRun } from "../types/workflows";
vi.mock("../api/workflows", () => ({
  workflowApi: {
    run: vi.fn(),
    controlRun: vi.fn(),
    exportBatch: vi.fn(),
    exportUrl: (id: string) => `/api/workflow/runs/${id}/export`,
  },
}));
vi.mock("./ArtifactReviewView", () => ({
  ArtifactReviewView: ({ reviewId, onSaved }: { reviewId: string; onSaved: () => void }) => (
    <p>复核 {reviewId}<button onClick={onSaved}>保存并同步</button></p>
  ),
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
describe("batch review barrier", () => {
  it("opens a dataset-linked review and notifies the dataset when it is saved", async () => {
    vi.clearAllMocks();
    vi.mocked(workflowApi.run).mockResolvedValue({
      id: "run-linked", status: "awaiting_review", profile_snapshot: {},
      template_snapshot: { name: "visual", nodes: [] }, context: {},
      items: [], attempts: [], artifacts: [], reviews: [], events: [], batch_counts: {},
    } as unknown as WorkflowRun);
    const changed = vi.fn();
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => root.render(<WorkflowRunView runId="run-linked" initialReviewId="review-linked" onDataChanged={changed} onBack={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
    expect(container.textContent).toContain("复核 review-linked");
    expect(window.location.hash).toBe("#workflow/run-linked/review/review-linked");
    await act(async () => container.querySelector<HTMLButtonElement>("button")!.click());
    expect(changed).toHaveBeenCalledTimes(1);
    expect(workflowApi.run).toHaveBeenCalledTimes(2);
    act(() => root.unmount());
  });
  it("uses the latest attempt for each image and batch while keeping old failures in collapsed history", async () => {
    vi.clearAllMocks();
    const message =
      "Vision bounding box does not fit the declared sent-image coordinates.";
    const run = {
      id: "run",
      status: "completed",
      profile_snapshot: { task_type: "detect" },
      template_snapshot: {
        name: "visual",
        nodes: [
          { id: "review", type: "vision_review", scope: "item" },
          { id: "export", type: "export", scope: "batch" },
        ],
      },
      context: {},
      items: [
        { id: "image-one", status: "completed" },
        { id: "image-two", status: "completed" },
      ],
      attempts: [
        {
          id: "review-old",
          node_id: "review",
          item_id: "image-one",
          status: "failed",
          error_message: message,
        },
        {
          id: "review-second-image",
          node_id: "review",
          item_id: "image-two",
          status: "completed",
        },
        {
          id: "review-new",
          node_id: "review",
          item_id: "image-one",
          status: "completed",
        },
        {
          id: "export-old",
          node_id: "export",
          item_id: null,
          status: "failed",
          error_message: "Old export failure",
        },
        {
          id: "export-new",
          node_id: "export",
          item_id: null,
          status: "completed",
        },
      ],
      artifacts: [],
      reviews: [],
      events: [],
      batch_counts: { total: 2, pending: 0 },
    } as unknown as WorkflowRun;
    vi.mocked(workflowApi.run).mockResolvedValue(run);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () =>
      root.render(
        <WorkflowRunView
          runId="run"
          onBack={vi.fn()}
          onOpenTraining={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      ),
    );
    const cards = container.querySelectorAll<HTMLElement>(
      ".workflow-step-list > li",
    );
    expect(cards).toHaveLength(2);
    expect(cards[0].dataset.status).toBe("completed");
    expect(cards[1].dataset.status).toBe("completed");
    expect(cards[0].querySelectorAll(":scope > span")[1].textContent).toBe(
      "已完成 2 张",
    );
    expect(cards[1].querySelectorAll(":scope > span")[1].textContent).toBe(
      "已完成",
    );
    for (const card of cards) {
      const history = Array.from(card.querySelectorAll("details")).find(
        (details) =>
          details.querySelector("summary")?.textContent === "历史尝试记录（1）",
      )!;
      expect(history.open).toBe(false);
      expect(history.textContent).toContain("执行失败");
      expect(card.querySelector('[role="alert"]')?.closest("details")).toBe(
        history,
      );
    }
    expect(cards[0].textContent).toContain("review-old");
    expect(cards[0].textContent).toContain(message);
    expect(cards[1].textContent).toContain("export-old");
    expect(cards[1].textContent).toContain("Old export failure");
    act(() => root.unmount());
    container.remove();
  });

  it("shows coordinate errors in Chinese and keeps original run, step and image errors collapsed", async () => {
    vi.clearAllMocks();
    const message =
      "Vision bounding box does not fit the declared sent-image coordinates.";
    const run = {
      id: "run",
      status: "needs_attention",
      error_message: message,
      profile_snapshot: { task_type: "detect" },
      template_snapshot: {
        name: "visual",
        nodes: [{ id: "review", type: "vision_review", scope: "item" }],
      },
      context: {},
      items: [{ id: "image-one", status: "failed", error_message: message }],
      attempts: [
        {
          id: "attempt-one",
          node_id: "review",
          status: "failed",
          error_message: message,
        },
      ],
      artifacts: [],
      reviews: [],
      events: [],
      batch_counts: { total: 1, failed: 1 },
    } as unknown as WorkflowRun;
    vi.mocked(workflowApi.run).mockResolvedValue(run);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () =>
      root.render(
        <WorkflowRunView
          runId="run"
          onBack={vi.fn()}
          onOpenTraining={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      ),
    );
    const errors = container.querySelectorAll('[role="alert"]');
    expect(errors).toHaveLength(3);
    for (const error of errors) {
      expect(error.firstChild?.textContent).toContain(
        "视觉模型返回的框超出当前图像范围",
      );
      const details = error.querySelector("details")!;
      expect(details.open).toBe(false);
      expect(details.textContent).toContain(message);
    }
    expect(workflowApi.controlRun).not.toHaveBeenCalled();
    act(() => root.unmount());
    container.remove();
  });

  it("shows snapshot verification for a template starting from confirmed data", async () => {
    vi.clearAllMocks();
    const run = {
      id: "run",
      status: "completed",
      profile_snapshot: {},
      template_snapshot: { nodes: [{ id: "snapshot", type: "snapshot" }] },
      context: {},
      items: [{ id: "one", status: "completed" }],
      attempts: [{ id: "attempt", node_id: "snapshot", status: "completed" }],
      artifacts: [],
      reviews: [],
      events: [],
      batch_counts: { total: 1, approved: 0, pending: 1 },
    } as unknown as WorkflowRun;
    vi.mocked(workflowApi.run).mockResolvedValue(run);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () =>
      root.render(
        <WorkflowRunView
          runId="run"
          onBack={vi.fn()}
          onOpenTraining={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      ),
    );
    expect(container.textContent).toContain("已核对本次数据");
    expect(container.textContent).not.toContain("已确认 / 纳入");
    expect(container.textContent).not.toContain("待审");
    act(() => root.unmount());
    container.remove();
  });
  it("shows the pending selected image and does not start training from one confirmation", async () => {
    vi.clearAllMocks();
    const run = {
      id: "run",
      status: "awaiting_review",
      profile_snapshot: {},
      template_snapshot: {
        name: "visual",
        nodes: [{ id: "confirm", type: "human_review" }],
      },
      context: {},
      items: [
        { id: "one", status: "completed" },
        { id: "two", status: "awaiting_review" },
      ],
      attempts: [],
      artifacts: [],
      reviews: [
        { id: "old-one", item_id: "one", status: "pending" },
        { id: "review-two", item_id: "two", status: "pending" },
        { id: "confirmed-one", item_id: "one", status: "decided" },
      ],
      events: [],
      batch_counts: {
        total: 2,
        approved: 1,
        pending: 1,
        background: 0,
        excluded: 0,
        failed: 0,
      },
    } as unknown as WorkflowRun;
    vi.mocked(workflowApi.run).mockResolvedValue(run);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () =>
      root.render(
        <WorkflowRunView
          runId="run"
          onBack={vi.fn()}
          onOpenTraining={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      ),
    );
    expect(container.textContent).toContain("1 / 2");
    expect(container.textContent).toContain("所有选中图片都检查完成后");
    expect(container.textContent).toContain("two");
    const nextAction = container.querySelector(".workflow-next-action")!;
    expect(nextAction.textContent).toContain("从第 2 张图片继续");
    const openReview =
      nextAction.querySelector<HTMLButtonElement>("button.primary")!;
    expect(openReview.textContent).toBe("打开图像复核");
    act(() => openReview.click());
    expect(container.textContent).toContain("复核 review-two");
    expect(workflowApi.controlRun).not.toHaveBeenCalled();
    act(() => root.unmount());
    container.remove();
  });

  it("shows the ready export at the top and keeps result jumps on the current run route", async () => {
    vi.clearAllMocks();
    window.history.replaceState(null, "", "#workflow/run");
    vi.mocked(workflowApi.run).mockResolvedValue({
      id: "run",
      status: "completed",
      profile_snapshot: {},
      template_snapshot: { name: "visual", nodes: [] },
      context: {},
      items: [],
      attempts: [],
      artifacts: [{ id: "export", kind: "export" }],
      reviews: [],
      events: [],
      batch_counts: {},
    } as unknown as WorkflowRun);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () =>
      root.render(
        <WorkflowRunView
          runId="run"
          onBack={vi.fn()}
          onOpenTraining={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      ),
    );
    const nextAction = container.querySelector(".workflow-next-action")!;
    const download = nextAction.querySelector<HTMLAnchorElement>("a.primary")!;
    expect(nextAction.textContent).toContain("下载本次已确认的标注");
    expect(download.textContent).toBe("下载已确认导出");
    expect(download.getAttribute("href")).toBe("/api/workflow/runs/run/export");
    const results = container.querySelector<HTMLElement>("#workflow-results")!;
    results.scrollIntoView = vi.fn();
    act(() => nextAction.querySelector<HTMLButtonElement>("button")!.click());
    expect(results.scrollIntoView).toHaveBeenCalledWith({ block: "start" });
    expect(window.location.hash).toBe("#workflow/run");
    expect(workflowApi.controlRun).not.toHaveBeenCalled();
    act(() => root.unmount());
    container.remove();
  });
});
