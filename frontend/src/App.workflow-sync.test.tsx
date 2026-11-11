// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import App from "./App";
import { api } from "./api/client";
import type { Dataset } from "./types";
import type { AnnotationDraft } from "./annotation/AnnotationCanvas";
const { editorProps } = vi.hoisted(() => ({ editorProps: vi.fn() }));

vi.mock("./api/client", async (original) => {
  const actual = await original<typeof import("./api/client")>();
  return { ...actual, api: { ...actual.api, listDatasets: vi.fn(), getSystemInfo: vi.fn(), listClasses: vi.fn(), listImages: vi.fn(), getAnnotations: vi.fn(), replaceAnnotations: vi.fn() } };
});
vi.mock("./pages/WorkflowView", () => ({
  WorkflowView: ({ initialDatasetId, onDataChanged, onDirtyChange, onEditImage }: { initialDatasetId?: string; onDataChanged?: () => void; onDirtyChange?: (dirty: boolean) => void; onEditImage?: (datasetId: string, imageId: string) => void }) => (
    <section><p>LLM 目标数据集 {initialDatasetId}</p><button onClick={onDataChanged}>模拟保存复核</button><button onClick={() => onDirtyChange?.(true)}>模拟未保存修改</button><button onClick={() => onEditImage?.("tower", "image-1")}>在数据集编辑器中继续修改</button></section>
  ),
}));
vi.mock("./annotation/StandaloneMaskEditor", () => ({
  StandaloneMaskEditor: (props: { drafts: AnnotationDraft[]; busy: boolean; onChange: (drafts: AnnotationDraft[]) => void }) => {
    editorProps(props);
    return <button disabled={props.busy} onClick={() => props.onChange(props.drafts.map((draft) => ({ ...draft, polygon: [[0, 0], [20, 0], [20, 20]] })))}>模拟补画轮廓</button>;
  },
}));
vi.mock("./components/AnnotationReviewPanel", () => ({ AnnotationReviewPanel: () => <p>整图确认</p> }));

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

it("opens current formal annotations for manual editing with unsaved-change protection", async () => {
  vi.mocked(api.listDatasets).mockResolvedValue([dataset]);
  vi.mocked(api.listClasses).mockResolvedValue([{ id: "class-1", dataset_id: "tower", class_index: 0, name: "塔体", color: "#123456", created_at: "", updated_at: "" }]);
  vi.mocked(api.listImages).mockResolvedValue({ total: 1, items: [{ id: "image-1", dataset_id: "tower", file_name: "tower.png", width: 30, height: 30, split: "train", status: "annotated", annotation_revision: 4, review_status: "approved", file_url: "/image", created_at: "", updated_at: "" }] });
  const formal = { id: "formal-current", image_id: "image-1", dataset_id: "tower", class_id: "class-1", class_index: 0, label: "塔体", color: "#123456", type: "polygon" as const, polygon: [[0, 0], [10, 0], [10, 10]] as [number, number][], bbox: null, obb: null, source: "manual" as const, created_at: "", updated_at: "" };
  vi.mocked(api.getAnnotations).mockResolvedValue([formal]);
  vi.mocked(api.replaceAnnotations).mockResolvedValue([formal]);
  window.history.replaceState(null, "", "#workflow/run-old/review/review-old");
  await act(async () => root.render(<App />));
  await click("模拟未保存修改");
  await click("在数据集编辑器中继续修改");
  expect(container.querySelector('[role="alertdialog"]')).not.toBeNull();
  expect(api.getAnnotations).not.toHaveBeenCalled();
  await click("继续编辑");
  expect(api.getAnnotations).not.toHaveBeenCalled();
  await click("在数据集编辑器中继续修改");
  await click("放弃修改");
  expect(api.getAnnotations).toHaveBeenCalledWith("tower", "image-1");
  expect((editorProps.mock.lastCall![0] as { drafts: AnnotationDraft[] }).drafts[0]).toMatchObject({ id: "formal-current", polygon: formal.polygon });
  expect(window.location.hash).toBe("");
  expect(api.replaceAnnotations).not.toHaveBeenCalled();
  await click("模拟补画轮廓");
  expect(container.textContent).toContain("未保存修改");
  await click("保存标注");
  expect(api.replaceAnnotations).toHaveBeenCalledWith("tower", "image-1", expect.arrayContaining([expect.objectContaining({ polygon: [[0, 0], [20, 0], [20, 20]] })]), 4, {});
});

it("leaves the workflow open if loading the dataset editor fails", async () => {
  vi.mocked(api.listDatasets).mockResolvedValue([dataset]);
  vi.mocked(api.listClasses).mockResolvedValue([]);
  vi.mocked(api.listImages).mockRejectedValue(new Error("图片读取失败"));
  vi.mocked(api.getAnnotations).mockResolvedValue([]);
  window.history.replaceState(null, "", "#workflow/run-old/review/review-old");
  await act(async () => root.render(<App />));
  await click("在数据集编辑器中继续修改");
  expect(container.textContent).toContain("LLM 目标数据集");
  expect(container.textContent).toContain("图片读取失败");
  expect(window.location.hash).toBe("#workflow/run-old/review/review-old");
  expect(api.replaceAnnotations).not.toHaveBeenCalled();
});

it("reads annotations after freezing the image revision and saves with that earlier revision", async () => {
  vi.mocked(api.listDatasets).mockResolvedValue([dataset]);
  vi.mocked(api.listClasses).mockResolvedValue([{ id: "class-1", dataset_id: "tower", class_index: 0, name: "塔体", color: "#123456", created_at: "", updated_at: "" }]);
  let finishPage!: (page: Awaited<ReturnType<typeof api.listImages>>) => void;
  vi.mocked(api.listImages).mockImplementation(() => new Promise((resolve) => { finishPage = resolve; }));
  vi.mocked(api.getAnnotations).mockResolvedValue([]);
  vi.mocked(api.replaceAnnotations).mockResolvedValue([]);
  window.history.replaceState(null, "", "#workflow/run-old/review/review-old");
  await act(async () => root.render(<App />));
  await click("在数据集编辑器中继续修改");
  expect(api.listImages).toHaveBeenCalledWith("tower");
  expect(api.getAnnotations).not.toHaveBeenCalled();
  await act(async () => finishPage({ total: 1, items: [{ id: "image-1", dataset_id: "tower", file_name: "tower.png", width: 30, height: 30, split: "train", status: "annotated", annotation_revision: 4, review_status: "approved", file_url: "/image", created_at: "", updated_at: "" }] }));
  expect(api.getAnnotations).toHaveBeenCalledWith("tower", "image-1");
  // If formal annotations change during or after this read, the older
  // expected revision makes the domain service reject the save with 409.
  await click("保存标注");
  expect(api.replaceAnnotations).toHaveBeenCalledWith("tower", "image-1", [], 4, {});
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
