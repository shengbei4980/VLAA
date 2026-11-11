// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, it, expect } from "vitest";
import { WorkflowProgressPanel, durationText } from "./WorkflowProgressPanel";
import type { WorkflowProgress, WorkflowRun } from "../types/workflows";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const progress: WorkflowProgress = {
  measured_at: "2026-11-11T00:00:00Z", automatic_applicable: true, total_images: 10,
  ready_images: 2, waiting_review_images: 2, failed_images: 0, excluded_images: 0,
  confirmed_images: 0, completed_steps: 12, total_steps: 60, percent: 20,
  processing_seconds: 165, sample_count: 2, average_image_seconds: 80,
  eta_seconds: 625, eta_state: "estimated",
  current: { node_id: "propose", node_type: "vision_propose", item_id: "item3", image_index: 3, file_name: "tower.png", elapsed_seconds: 5 },
};

describe("measured automatic progress", () => {
  it("shows persisted percent, current image, measured time and explicitly estimated ETA", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    act(() => root.render(<WorkflowProgressPanel run={{ status: "running", progress } as WorkflowRun} disconnected={false} />));
    expect(container.querySelector('[role="progressbar"]')?.getAttribute("aria-valuenow")).toBe("20");
    expect(container.textContent).toContain("第 3 / 10 张");
    expect(container.textContent).toContain("tower.png");
    expect(container.textContent).toContain("2 分 45 秒");
    expect(container.textContent).toContain("约 10 分 25 秒");
    expect(container.textContent).toContain("不包含后续人工复核、训练和导出");
    act(() => root.unmount());
  });

  it("suppresses stale ETA after disconnect and never implies automatic human approval", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    act(() => root.render(<WorkflowProgressPanel run={{ status: "running", progress } as WorkflowRun} disconnected />));
    expect(container.textContent).toContain("连接恢复后更新");
    expect(container.textContent).not.toContain("约 10 分");
    act(() => root.render(<WorkflowProgressPanel run={{ status: "awaiting_review", progress: { ...progress, ready_images: 10, waiting_review_images: 10, percent: 100, current: null, eta_seconds: null, eta_state: "complete" } } as WorkflowRun} disconnected={false} />));
    expect(container.textContent).toContain("标注已就绪，等待你检查");
    expect(container.textContent).toContain("已人工确认 0 张");
    expect(container.querySelectorAll('li[data-done="true"]')).toHaveLength(1);
    act(() => root.unmount());
  });

  it("explains the lack of an ETA for one image, failures, and slow requests", () => {
    const container = document.createElement("div");
    const root = createRoot(container);
    for (const [state, text] of [["learning", "正在积累测速样本"], ["blocked", "处理异常，暂不能估计"], ["overdue", "本张耗时较长，重新估计中"]] as const) {
      act(() => root.render(<WorkflowProgressPanel run={{ status: "running", progress: { ...progress, eta_state: state, eta_seconds: null } } as WorkflowRun} disconnected={false} />));
      expect(container.textContent).toContain(text);
      expect(container.textContent).not.toContain("约 0 秒");
    }
    act(() => root.unmount());
    expect(durationText(0)).toBe("0 秒");
    expect(durationText(3661)).toBe("1 小时 1 分");
  });
});
