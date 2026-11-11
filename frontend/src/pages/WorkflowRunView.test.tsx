// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { WorkflowRunView } from "./WorkflowRunView";
import { workflowApi } from "../api/workflows";
import type { WorkflowRun, WorkflowReview } from "../types/workflows";
const { artifactProps } = vi.hoisted(() => ({ artifactProps: vi.fn() }));
vi.mock("../api/workflows", () => ({
  workflowApi: {
    run: vi.fn(),
    controlRun: vi.fn(),
    exportBatch: vi.fn(),
    exportUrl: (id: string) => `/api/workflow/runs/${id}/export`,
  },
}));
vi.mock("./ArtifactReviewView", () => ({
  ArtifactReviewView: (props: { reviewId: string; onSaved: (saved?: WorkflowReview) => void; onNext?: () => void; onOpenRevision?: (value: WorkflowReview) => void; onEditImage?: (datasetId: string, imageId: string) => void; batchPosition?: { index: number; total: number; remaining: number } }) => {
    artifactProps(props);
    const { reviewId, onSaved, onNext, batchPosition } = props;
    return (
    <div>复核 {reviewId}<button onClick={() => onSaved()}>保存并同步</button>
      {batchPosition && <p>第 {batchPosition.index} / {batchPosition.total} 张 · 其余 {batchPosition.remaining}</p>}
      {onNext && <button onClick={() => { onSaved({ id: reviewId, revision: 2, status: "decided" } as WorkflowReview); onNext(); }}>测试确认并继续</button>}
    </div>
    );
  },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
describe("batch review barrier", () => {
  it("hands new revisions to workflow navigation and keeps the dataset editor callback", async () => {
    vi.clearAllMocks();
    vi.mocked(workflowApi.run).mockResolvedValue({
      id: "run-old", status: "completed", profile_snapshot: {}, template_snapshot: { nodes: [] },
      context: {}, items: [], attempts: [], artifacts: [], reviews: [], events: [], batch_counts: {},
    } as unknown as WorkflowRun);
    const open = vi.fn();
    const edit = vi.fn();
    const changed = vi.fn();
    const root = createRoot(document.createElement("div"));
    try {
      await act(async () => root.render(<WorkflowRunView runId="run-old" initialReviewId="review-old" onOpenReviewRevision={open} onEditImage={edit} onDataChanged={changed} onBack={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
      const props = artifactProps.mock.lastCall![0] as { onOpenRevision: (review: WorkflowReview) => void; onEditImage: typeof edit };
      const next = { id: "review-next", run_id: "run-next" } as WorkflowReview;
      act(() => props.onOpenRevision(next));
      expect(open).toHaveBeenCalledWith(next);
      expect(changed).toHaveBeenCalledOnce();
      expect(props.onEditImage).toBe(edit);
      expect(workflowApi.run).toHaveBeenCalledOnce();
    } finally { act(() => root.unmount()); }
  });
  it("continues in image order, skips finished reviews and wraps to earlier pending images", async () => {
    vi.clearAllMocks();
    const run = {
      id: "batch-next", status: "awaiting_review", profile_snapshot: {}, template_snapshot: { nodes: [] }, context: {},
      items: [1, 2, 3, 4, 5].map((n) => ({ id: `item-${n}`, status: "awaiting_review" })),
      reviews: ["pending", "approved", "pending", "excluded", "pending"].map((status, index) => ({ id: `review-${index + 1}`, item_id: `item-${index + 1}`, status, revision: 1 })),
      attempts: [], artifacts: [], events: [], batch_counts: {},
    } as unknown as WorkflowRun;
    vi.mocked(workflowApi.run).mockResolvedValueOnce(run).mockImplementation(() => new Promise(() => {}));
    const container = document.createElement("div");
    const root = createRoot(container);
    try {
      await act(async () => root.render(<WorkflowRunView runId="batch-next" initialReviewId="review-3" onBack={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
      expect(container.textContent).toContain("第 3 / 5 张 · 其余 2");
      const next = () => Array.from(container.querySelectorAll("button")).find((button) => button.textContent === "测试确认并继续")!;
      act(() => next().click());
      expect(container.textContent).toContain("复核 review-5");
      expect(container.textContent).toContain("第 5 / 5 张 · 其余 1");
      act(() => next().click());
      expect(container.textContent).toContain("复核 review-1");
      expect(container.textContent).toContain("第 1 / 5 张 · 其余 0");
      expect(next()).toBeUndefined();
    } finally { act(() => root.unmount()); }
  });
  it("does not let an old poll overwrite a newer cancellation response", async () => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    const base = { id: "race", status: "running", profile_snapshot: {}, template_snapshot: { name: "visual", nodes: [] }, context: {}, items: [], attempts: [], artifacts: [], reviews: [], events: [], batch_counts: {} } as unknown as WorkflowRun;
    let resolvePoll!: (value: WorkflowRun) => void;
    vi.mocked(workflowApi.run).mockResolvedValueOnce(base)
      .mockImplementationOnce(() => new Promise((resolve) => { resolvePoll = resolve; }))
      .mockResolvedValueOnce({ ...base, status: "cancelled" });
    vi.mocked(workflowApi.controlRun).mockResolvedValue({ ...base, status: "cancelled" });
    const container = document.createElement("div");
    const root = createRoot(container);
    try {
      await act(async () => root.render(<WorkflowRunView runId="race" onBack={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
      await act(async () => vi.advanceTimersByTimeAsync(2000));
      const cancel = Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find((button) => button.textContent === "取消运行")!;
      await act(async () => cancel.click());
      await act(async () => resolvePoll(base));
      expect(container.querySelector(".workflow-status-badge")?.textContent).toBe("已取消");
    } finally {
      act(() => root.unmount());
      vi.useRealTimers();
    }
  });
  it("filters large batches without losing original image numbers and limits mounted rows", async () => {
    vi.clearAllMocks();
    vi.mocked(workflowApi.run).mockResolvedValue({
      id: "batch", status: "needs_attention", profile_snapshot: {},
      template_snapshot: { name: "visual", nodes: [] }, context: {},
      items: Array.from({ length: 120 }, (_, index) => ({ id: `item-${index}`, status: index === 109 ? "failed" : "pending", file_name: `tower-${index}.png` })),
      attempts: [], artifacts: [], reviews: [], events: [], batch_counts: {},
    } as unknown as WorkflowRun);
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => root.render(<WorkflowRunView runId="batch" onBack={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
    expect(container.querySelectorAll(".workflow-item")).toHaveLength(50);
    const filter = Array.from(container.querySelectorAll<HTMLButtonElement>(".workflow-image-filters button")).find((button) => button.textContent?.startsWith("失败"))!;
    act(() => filter.click());
    expect(container.querySelectorAll(".workflow-item")).toHaveLength(1);
    expect(container.querySelector(".workflow-item")?.textContent).toContain("第 110 张图片");
    expect(container.querySelector(".workflow-item")?.textContent).toContain("tower-109.png");
    act(() => root.unmount());
  });

  it("stops background polling for closed runs and while editing a review", async () => {
    vi.useFakeTimers();
    try {
      for (const status of ["completed", "awaiting_review"]) {
        vi.clearAllMocks();
        vi.mocked(workflowApi.run).mockResolvedValue({
          id: "quiet", status, profile_snapshot: {}, template_snapshot: { name: "visual", nodes: [] },
          context: {}, items: [], attempts: [], artifacts: [], reviews: [], events: [], batch_counts: {},
        } as unknown as WorkflowRun);
        const root = createRoot(document.createElement("div"));
        await act(async () => root.render(<WorkflowRunView runId="quiet" initialReviewId={status === "awaiting_review" ? "review" : undefined} onBack={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
        await act(async () => vi.advanceTimersByTimeAsync(8000));
        expect(workflowApi.run).toHaveBeenCalledTimes(1);
        act(() => root.unmount());
      }
    } finally { vi.useRealTimers(); }
  });

  it("does not label an entire batch step complete after only one image", async () => {
    vi.clearAllMocks();
    vi.mocked(workflowApi.run).mockResolvedValue({
      id: "partial", status: "running", profile_snapshot: {},
      template_snapshot: { name: "visual", nodes: [{ id: "check", type: "vision_review", scope: "item" }] }, context: {},
      items: [{ id: "one", status: "awaiting_review" }, { id: "two", status: "pending" }],
      attempts: [{ id: "attempt", item_id: "one", node_id: "check", status: "completed" }],
      artifacts: [], reviews: [], events: [], batch_counts: {},
    } as unknown as WorkflowRun);
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => root.render(<WorkflowRunView runId="partial" onBack={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
    const step = container.querySelector<HTMLElement>(".workflow-step-list > li")!;
    expect(step.dataset.status).toBe("pending");
    expect(step.textContent).toContain("已完成 1 张 · 尚未到此步 1 张");
    act(() => root.unmount());
  });
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
