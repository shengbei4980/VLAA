// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api/client";
import type { Dataset } from "./types";

vi.mock("./api/client", async (original) => {
  const actual = await original<typeof import("./api/client")>();
  return { ...actual, api: { ...actual.api, listDatasets: vi.fn(), getSystemInfo: vi.fn() } };
});
vi.mock("./pages/WorkflowView", () => ({
  WorkflowView: ({ initialDatasetId, onDataChanged }: { initialDatasetId?: string; onDataChanged?: () => void }) => (
    <section><p>LLM 目标数据集 {initialDatasetId}</p><button onClick={onDataChanged}>模拟保存复核</button></section>
  ),
}));

const dataset: Dataset = {
  id: "tower", name: "塔体", description: null, task_type: "segment",
  image_count: 1, annotated_image_count: 0, class_count: 1, created_at: "", updated_at: "",
};
let root: Root;
let container: HTMLDivElement;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  vi.resetAllMocks();
  window.history.replaceState(null, "", "/");
  vi.mocked(api.getSystemInfo).mockResolvedValue({} as Awaited<ReturnType<typeof api.getSystemInfo>>);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});
async function click(label: string, scope: ParentNode = container) {
  const button = Array.from(scope.querySelectorAll("button")).find((item) => item.textContent === label);
  expect(button, label).toBeDefined();
  await act(async () => button!.click());
}

it("opens LLM annotation with the selected dataset and refreshes counts after saving", async () => {
  const second = { ...dataset, id: "satellite", name: "卫星" };
  vi.mocked(api.listDatasets).mockResolvedValue([dataset, second]);
  await act(async () => root.render(<App />));
  await click("LLM 自动标注", container.querySelectorAll(".dataset-card")[1]);
  expect(container.textContent).toContain("LLM 目标数据集 satellite");
  const before = vi.mocked(api.listDatasets).mock.calls.length;
  vi.mocked(api.listDatasets).mockResolvedValue([dataset, {
    ...second, generated_image_count: 1, confirmed_image_count: 1, annotated_image_count: 1,
  }]);
  await click("模拟保存复核");
  expect(vi.mocked(api.listDatasets).mock.calls.length).toBeGreaterThan(before);
  await click("数据集");
  expect(container.querySelectorAll(".dataset-card")[1].textContent).toContain("100%");
});

it("continues annotation at the matching existing workflow review", async () => {
  vi.mocked(api.listDatasets).mockResolvedValue([{
    ...dataset, generated_image_count: 1, pending_review_image_count: 1,
    latest_workflow_run_id: "run-tower", latest_workflow_review_id: "review-tower",
  }]);
  await act(async () => root.render(<App />));
  await click("继续标注");
  expect(window.location.hash).toBe("#workflow/run-tower/review/review-tower");
  expect(container.textContent).toContain("LLM 目标数据集 tower");
});

it("keeps the newer dataset response when an earlier refresh finishes late", async () => {
  let finishFirst!: (value: Dataset[]) => void;
  vi.mocked(api.listDatasets)
    .mockImplementationOnce(() => new Promise((resolve) => { finishFirst = resolve; }))
    .mockResolvedValue([{ ...dataset, name: "最新状态" }]);
  await act(async () => root.render(<App />));
  await act(async () => window.dispatchEvent(new Event("focus")));
  expect(container.textContent).toContain("最新状态");
  await act(async () => finishFirst([{ ...dataset, name: "过期状态" }]));
  expect(container.textContent).toContain("最新状态");
  expect(container.textContent).not.toContain("过期状态");
});
