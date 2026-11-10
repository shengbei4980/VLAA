// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { workflowApi } from "../api/workflows";
import type { WorkflowRun } from "../types/workflows";
import { WorkflowRunHistory } from "./WorkflowRunHistory";

vi.mock("../api/workflows", () => ({
  workflowApi: {
    runs: vi.fn(),
    run: vi.fn(),
    deleteRun: vi.fn(),
    restoreRun: vi.fn(),
    controlRun: vi.fn(),
  },
}));

const run: WorkflowRun = {
  id: "run-1",
  template_name: "用视觉 LLM 标注并检查",
  profile_name: "塔体构件",
  status: "cancelled",
  template_snapshot: { id: "template-1", key: "visual_llm_review_export_v1", name: "标注", version: 1, nodes: [] },
  profile_snapshot: {
    task_type: "detect", geometry: "bbox", class_rules: [], models: {}, image_strategy: {}, training: {},
    quality_conditions: {}, repair_limit: 0, max_requests: 50, export_format: "labelme",
  },
  context: {}, items: [], attempts: [], artifacts: [], reviews: [], events: [], batch_counts: {},
  created_at: "2026-11-10T10:00:00Z",
  can_delete: true,
};
let root: Root;
let container: HTMLDivElement;
let onOpen: ReturnType<typeof vi.fn>;
const button = (text: string) => Array.from(container.querySelectorAll<HTMLButtonElement>("button"))
  .find((item) => item.textContent === text)!;
const render = async () => {
  await act(async () => root.render(<WorkflowRunHistory profiles={[]} templates={[]} onOpenRun={onOpen} />));
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(workflowApi.runs).mockImplementation(async (deleted) => deleted ? [{ ...run, deleted_at: "2026-11-10T10:30:00Z" }] : [run]);
  vi.mocked(workflowApi.deleteRun).mockResolvedValue({ ...run, deleted_at: "2026-11-10T10:30:00Z" });
  vi.mocked(workflowApi.restoreRun).mockResolvedValue(run);
  vi.mocked(workflowApi.controlRun).mockResolvedValue(run);
  vi.mocked(workflowApi.run).mockResolvedValue(run);
  onOpen = vi.fn();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

describe("workflow run record management", () => {
  it("confirms deletion separately from opening the run and restores from the deleted list", async () => {
    await render();
    act(() => button("删除记录").click());
    expect(onOpen).not.toHaveBeenCalled();
    expect(workflowApi.deleteRun).not.toHaveBeenCalled();
    expect(container.querySelector('[role="dialog"]')).not.toBeNull();
    act(() => button("保留记录").click());
    expect(workflowApi.deleteRun).not.toHaveBeenCalled();
    act(() => button("删除记录").click());
    await act(async () => button("确认删除记录").click());
    expect(workflowApi.deleteRun).toHaveBeenCalledWith("run-1");
    expect(workflowApi.controlRun).not.toHaveBeenCalled();
    expect(container.querySelectorAll(".workflow-run-row")).toHaveLength(0);
    expect(container.textContent).toContain("记录已移入“已删除记录”");
    await act(async () => button("已删除记录").click());
    expect(workflowApi.runs).toHaveBeenLastCalledWith(true);
    await act(async () => button("恢复记录").click());
    expect(workflowApi.restoreRun).toHaveBeenCalledWith("run-1");
    expect(container.querySelectorAll(".workflow-run-row")).toHaveLength(0);
    await act(async () => button("返回运行记录").click());
    expect(workflowApi.runs).toHaveBeenLastCalledWith(false);
    expect(container.querySelectorAll(".workflow-run-row")).toHaveLength(1);
  });

  it("cancels a pending review before deleting its record", async () => {
    vi.mocked(workflowApi.runs).mockResolvedValue([{ ...run, status: "awaiting_review", can_delete: false }]);
    await render();
    act(() => button("删除记录").click());
    expect(container.textContent).toContain("恢复记录后，任务将保留取消状态");
    await act(async () => button("取消任务并删除记录").click());
    expect(workflowApi.controlRun).toHaveBeenCalledWith("run-1", "cancel");
    expect(workflowApi.deleteRun).toHaveBeenCalledWith("run-1");
    expect(vi.mocked(workflowApi.controlRun).mock.invocationCallOrder[0]).toBeLessThan(
      vi.mocked(workflowApi.deleteRun).mock.invocationCallOrder[0],
    );
  });

  it("waits for cancellation to settle and checks status without cancelling twice", async () => {
    vi.mocked(workflowApi.runs).mockResolvedValue([{ ...run, status: "running", can_delete: false }]);
    vi.mocked(workflowApi.controlRun).mockResolvedValue({ ...run, status: "cancelling", can_delete: false });
    await render();
    act(() => button("删除记录").click());
    await act(async () => button("取消任务并删除记录").click());
    expect(workflowApi.deleteRun).not.toHaveBeenCalled();
    expect(container.querySelectorAll(".workflow-run-row")).toHaveLength(1);
    expect(container.textContent).toContain("任务正在取消");
    await act(async () => button("检查状态并删除").click());
    expect(workflowApi.run).toHaveBeenCalledWith("run-1");
    expect(workflowApi.controlRun).toHaveBeenCalledTimes(1);
    expect(workflowApi.deleteRun).toHaveBeenCalledTimes(1);
  });

  it("keeps a record and its confirmation dialog when deletion fails", async () => {
    vi.mocked(workflowApi.deleteRun).mockRejectedValueOnce(new Error("关联任务尚未结束"));
    await render();
    act(() => button("删除记录").click());
    await act(async () => button("确认删除记录").click());
    expect(container.textContent).toContain("关联任务尚未结束");
    expect(container.querySelector('[role="dialog"]')).not.toBeNull();
    expect(container.querySelectorAll(".workflow-run-row")).toHaveLength(1);
    await act(async () => button("确认删除记录").click());
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(container.querySelectorAll(".workflow-run-row")).toHaveLength(0);
  });

  it("disables deletion while terminal records still have active related tasks", async () => {
    vi.mocked(workflowApi.runs).mockResolvedValue([{ ...run, can_delete: false }]);
    await render();
    expect(button("删除记录").disabled).toBe(true);
    act(() => container.querySelector<HTMLButtonElement>(".workflow-run-row")!.click());
    expect(onOpen).toHaveBeenCalledWith("run-1");
  });
});
