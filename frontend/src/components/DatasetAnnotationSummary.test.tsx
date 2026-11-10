// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetAnnotationSummary } from "./DatasetAnnotationSummary";
import { DatasetReviewStatusPanel } from "./DatasetReviewStatusPanel";
import { reviewApi, type DatasetReviewStatus } from "../api/reviews";
import type { Dataset } from "../types";

vi.mock("../api/reviews", () => ({ reviewApi: { status: vi.fn() } }));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const dataset = {
  id: "dataset", name: "Tower", image_count: 1, annotated_image_count: 0,
  generated_image_count: 1, pending_review_image_count: 1, confirmed_image_count: 0,
  workflow_image_count: 1, latest_workflow_run_id: "run", latest_workflow_review_id: "review",
} as Dataset;
const status: DatasetReviewStatus = {
  confirmed: 0, pending: 1, confirmed_empty: 0, total: 1,
  confirmed_image_count: 0, pending_review_image_count: 1,
  items: [{ image_id: "image", file_name: "tower.jpg", revision: 0, review_status: "pending", annotation_count: 0, contains_auto: false,
    has_draft: true, workflow_run_id: "run", workflow_review_id: "review", workflow_annotation_count: 5, workflow_status: "draft" }],
};

let container: HTMLDivElement;
let root: ReturnType<typeof createRoot>;
beforeEach(() => {
  vi.clearAllMocks();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("dataset annotation synchronization", () => {
  it("shows generated LLM annotations separately from human confirmation and opens their review", async () => {
    const open = vi.fn();
    await act(async () => root.render(<DatasetAnnotationSummary dataset={dataset} onOpenWorkflow={open} />));
    expect(container.textContent).toContain("100%");
    expect(container.textContent).toContain("人工已确认 0 / 1 张");
    expect(container.textContent).toContain("待人工确认 1 张");
    act(() => container.querySelector<HTMLButtonElement>("button")!.click());
    expect(open).toHaveBeenCalledWith("run", "review");
  });

  it("shows confirmed data and supports datasets returned by older backends", async () => {
    await act(async () => root.render(<DatasetAnnotationSummary dataset={{ ...dataset, pending_review_image_count: 0, confirmed_image_count: 1 }} />));
    expect(container.textContent).toContain("人工已确认 1 / 1 张");
    expect(container.textContent).toContain("待人工确认 0 张");
    expect(container.textContent).not.toContain("自动标注已同步");
    await act(async () => root.render(<DatasetAnnotationSummary dataset={{ image_count: 2, annotated_image_count: 1 } as Dataset} />));
    expect(container.textContent).toContain("50%");
  });

  it("shows workflow candidates and refreshes an expanded review summary after edits", async () => {
    vi.mocked(reviewApi.status).mockResolvedValue(status);
    const openImage = vi.fn();
    const openWorkflow = vi.fn();
    await act(async () => root.render(<DatasetReviewStatusPanel datasetId="dataset" onOpenImage={openImage} onOpenWorkflowReview={openWorkflow} refreshToken={0} />));
    expect(reviewApi.status).not.toHaveBeenCalled();
    await act(async () => container.querySelector<HTMLButtonElement>("button")!.click());
    expect(container.textContent).toContain("tower.jpg");
    expect(container.textContent).toContain("工作流已生成标注，待人工确认");
    expect(container.textContent).toContain("正式标注 0 个 · 工作流候选 5 个");
    const reviewButton = Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find((button) => button.textContent === "打开 LLM 标注复核")!;
    act(() => reviewButton.click());
    expect(openWorkflow).toHaveBeenCalledWith("run", "review");
    expect(openImage).not.toHaveBeenCalled();
    vi.mocked(reviewApi.status).mockResolvedValue({ ...status, confirmed: 1, pending: 0, confirmed_image_count: 1, pending_review_image_count: 0,
      items: [{ ...status.items[0], annotation_count: 5, review_status: "confirmed", workflow_status: "confirmed" }] });
    await act(async () => root.render(<DatasetReviewStatusPanel datasetId="dataset" onOpenImage={openImage} onOpenWorkflowReview={openWorkflow} refreshToken={1} />));
    expect(reviewApi.status).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("已确认并同步数据集");
    expect(container.textContent).toContain("正式标注 5 个");
  });
});
