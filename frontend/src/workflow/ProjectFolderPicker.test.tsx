// @vitest-environment jsdom
import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ProjectFolderPicker } from "./ProjectFolderPicker";
import { workflowApi } from "../api/workflows";
import type { WorkflowProject } from "../types/workflows";

vi.mock("../api/workflows", () => ({
  workflowApi: { projects: vi.fn(), saveProject: vi.fn() },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root;
let container: HTMLDivElement;
const project: WorkflowProject = {
  id: "tower-project",
  name: "塔体构件",
  root_path: "E:\\塔体项目",
  destination_id: "dest",
  created_at: "",
};

function Picker({ onReady }: { onReady: (ready: boolean) => void }) {
  const [value, setValue] = useState<WorkflowProject>();
  return (
    <ProjectFolderPicker
      value={value}
      onChange={setValue}
      onReadyChange={onReady}
    />
  );
}

beforeEach(() => {
  vi.resetAllMocks();
  window.localStorage.clear();
  vi.mocked(workflowApi.projects).mockResolvedValue([project]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

const button = (name: string) =>
  Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find(
    (item) => item.textContent === name,
  )!;
const input = (index: number, value: string) =>
  act(() => {
    const element =
      container.querySelectorAll<HTMLInputElement>("input")[index];
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(element, value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });

it("starts with default storage and restores only a previously selected registered project", async () => {
  const onReady = vi.fn();
  await act(async () => root.render(<Picker onReady={onReady} />));
  const select = container.querySelector("select")!;
  expect(select.value).toBe("");
  expect(onReady).toHaveBeenLastCalledWith(true);
  act(() => {
    select.value = project.id;
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  expect(window.localStorage.getItem("vlaa.workflow.last-project")).toBe(
    project.id,
  );
  expect(container.textContent).toContain(
    "E:\\塔体项目\\workflow-<运行编号>\\annotations\\json",
  );
  act(() => root.unmount());
  root = createRoot(container);
  await act(async () => root.render(<Picker onReady={onReady} />));
  expect(container.querySelector("select")!.value).toBe(project.id);
  expect(workflowApi.saveProject).not.toHaveBeenCalled();
});

it("saves a new project once and keeps entered paths after a failed attempt", async () => {
  const onReady = vi.fn();
  vi.mocked(workflowApi.saveProject)
    .mockRejectedValueOnce(new Error("根目录不可写"))
    .mockResolvedValueOnce(project);
  await act(async () => root.render(<Picker onReady={onReady} />));
  act(() => button("新建项目并设置根目录").click());
  expect(onReady).toHaveBeenLastCalledWith(false);
  input(0, project.name);
  input(1, project.root_path);
  await act(async () => button("保存项目并使用").click());
  expect(container.textContent).toContain("根目录不可写");
  expect(container.querySelectorAll<HTMLInputElement>("input")[1].value).toBe(
    project.root_path,
  );
  expect(onReady).toHaveBeenLastCalledWith(false);
  await act(async () => button("保存项目并使用").click());
  expect(workflowApi.saveProject).toHaveBeenLastCalledWith({
    name: project.name,
    root_path: project.root_path,
  });
  expect(container.querySelector("select")!.value).toBe(project.id);
  expect(onReady).toHaveBeenLastCalledWith(true);
  expect(container.querySelectorAll("input")).toHaveLength(0);
});

it("does not select a deleted remembered project and allows cancellation of a new folder", async () => {
  window.localStorage.setItem("vlaa.workflow.last-project", "missing");
  const onReady = vi.fn();
  await act(async () => root.render(<Picker onReady={onReady} />));
  expect(container.querySelector("select")!.value).toBe("");
  act(() => button("新建项目并设置根目录").click());
  expect(onReady).toHaveBeenLastCalledWith(false);
  act(() => button("取消新建").click());
  expect(onReady).toHaveBeenLastCalledWith(true);
  expect(workflowApi.saveProject).not.toHaveBeenCalled();
});

it("keeps launch blocked after a failed project read and restores the chosen root on retry", async () => {
  window.localStorage.setItem("vlaa.workflow.last-project", project.id);
  vi.mocked(workflowApi.projects)
    .mockRejectedValueOnce(new Error("项目暂时读取失败"))
    .mockResolvedValueOnce([project]);
  const onReady = vi.fn();
  await act(async () => root.render(<Picker onReady={onReady} />));
  expect(onReady).toHaveBeenLastCalledWith(false);
  expect(container.querySelector("select")!.disabled).toBe(true);
  expect(window.localStorage.getItem("vlaa.workflow.last-project")).toBe(
    project.id,
  );
  await act(async () => button("重新读取项目").click());
  expect(container.querySelector("select")!.value).toBe(project.id);
  expect(onReady).toHaveBeenLastCalledWith(true);
  expect(container.querySelector('[role="alert"]')).toBeNull();
});
