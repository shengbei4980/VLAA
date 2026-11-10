// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ExportLocationsPanel } from "./ExportLocationsPanel";
import { storageApi } from "../api/storage";

vi.mock("../api/storage", () => ({
  storageApi: { manualExports: vi.fn(), saveManualExports: vi.fn() },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root;
let container: HTMLDivElement;
const paths = { json_path: "D:\\输出\\JSON", images_path: "E:\\输出\\PNG" };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(storageApi.manualExports).mockResolvedValue(paths);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

const setInput = (input: HTMLInputElement, value: string) =>
  act(() => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
const button = (name: string) =>
  Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find(
    (element) => element.textContent === name,
  )!;

it("loads independent folders and preserves entered values when saving fails", async () => {
  vi.mocked(storageApi.saveManualExports).mockRejectedValue(
    new Error("目录不可写"),
  );
  await act(async () => root.render(<ExportLocationsPanel />));
  const inputs = container.querySelectorAll<HTMLInputElement>("input");
  expect(inputs[0].value).toBe(paths.json_path);
  expect(inputs[1].value).toBe(paths.images_path);
  setInput(inputs[0], "D:\\研究\\JSON");
  setInput(inputs[1], "E:\\研究\\PNG");
  await act(async () => button("保存导出位置").click());
  expect(storageApi.saveManualExports).toHaveBeenCalledWith({
    json_path: "D:\\研究\\JSON",
    images_path: "E:\\研究\\PNG",
  });
  expect(inputs[0].value).toBe("D:\\研究\\JSON");
  expect(inputs[1].value).toBe("E:\\研究\\PNG");
  expect(container.textContent).toContain("目录不可写");
});

it("retries a failed initial read without replacing a folder the user has typed", async () => {
  vi.mocked(storageApi.manualExports)
    .mockRejectedValueOnce(new Error("后端暂不可用"))
    .mockResolvedValueOnce(paths);
  await act(async () => root.render(<ExportLocationsPanel />));
  const inputs = container.querySelectorAll<HTMLInputElement>("input");
  setInput(inputs[0], "D:\\当前填写");
  await act(async () => button("重新读取位置").click());
  expect(inputs[0].value).toBe("D:\\当前填写");
  expect(inputs[1].value).toBe(paths.images_path);
  expect(container.textContent).not.toContain("后端暂不可用");
});

it("shows the normalized saved folders and reports success", async () => {
  vi.mocked(storageApi.saveManualExports).mockResolvedValue(paths);
  const onSaved = vi.fn();
  await act(async () =>
    root.render(<ExportLocationsPanel onSaved={onSaved} />),
  );
  await act(async () => button("保存导出位置").click());
  expect(onSaved).toHaveBeenCalledOnce();
  expect(container.textContent).toContain("导出位置已保存");
});
