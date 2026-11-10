// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { BatchExportPanel } from "./BatchExportPanel";
import { workflowApi, type BatchAnnotationExport } from "../api/workflows";

vi.mock("../api/workflows", () => ({ workflowApi: { exportBatch: vi.fn() } }));
vi.mock("../api/client", () => ({
  apiUrl: (path: string) => "http://127.0.0.1:8000" + path,
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root;
let container: HTMLDivElement;
const exported: BatchAnnotationExport = {
  file_name: "batch.zip",
  saved_path: "E:\\塔体项目\\workflow-run\\annotations\\batch.zip",
  download_url: "/api/workflows/runs/run/annotation-batch/batch.zip",
  total_images: 3,
  exported_images: 1,
  pending_images: 1,
  failed_images: 1,
  skipped_images: 1,
  results: [
    {
      image_id: "one",
      status: "exported",
      export_state: "draft",
      json_path: "E:\\塔体项目\\json\\one.json",
      image_path: "E:\\塔体项目\\images\\one.png",
    },
    { image_id: "two", status: "failed", error: "原图读取失败" },
    { image_id: "three", status: "skipped", reason: "图片尚未完成自动标注" },
  ],
};

beforeEach(() => {
  vi.resetAllMocks();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

it("exports saved drafts before human confirmation and reports each image outcome", async () => {
  vi.mocked(workflowApi.exportBatch).mockResolvedValue(exported);
  await act(async () =>
    root.render(<BatchExportPanel runId="run" readyImages={2} />),
  );
  await act(async () => container.querySelector("button")!.click());
  expect(workflowApi.exportBatch).toHaveBeenCalledOnce();
  expect(workflowApi.exportBatch).toHaveBeenCalledWith("run");
  expect(container.textContent).toContain("已保存 1 张，其中草稿 1 张");
  expect(container.textContent).toContain("失败 1 张，暂未导出 1 张");
  expect(container.textContent).toContain("E:\\塔体项目\\json\\one.json");
  expect(container.textContent).toContain("原图读取失败");
  expect(container.textContent).toContain("图片尚未完成自动标注");
  const link = container.querySelector("a")!;
  expect(link.href).toBe("http://127.0.0.1:8000" + exported.download_url);
  expect(link.target).toBe("_blank");
  expect(link.rel).toBe("noopener noreferrer");
  expect(container.querySelector("button")!.textContent).toBe(
    "重新批量保存 JSON 与标注图片",
  );
});

it("disables empty batches and retries after a failed export", async () => {
  vi.mocked(workflowApi.exportBatch)
    .mockRejectedValueOnce(new Error("目录暂不可写"))
    .mockResolvedValueOnce(exported);
  await act(async () =>
    root.render(<BatchExportPanel runId="run" readyImages={0} />),
  );
  expect(container.querySelector("button")!.disabled).toBe(true);
  await act(async () =>
    root.render(<BatchExportPanel runId="run" readyImages={2} />),
  );
  await act(async () => container.querySelector("button")!.click());
  expect(container.querySelector('[role="alert"]')?.textContent).toContain(
    "目录暂不可写",
  );
  expect(container.querySelector("button")!.disabled).toBe(false);
  await act(async () => container.querySelector("button")!.click());
  expect(workflowApi.exportBatch).toHaveBeenCalledTimes(2);
  expect(container.querySelector('[role="alert"]')).toBeNull();
  expect(container.querySelector("a")).not.toBeNull();
});
