// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TemplateForm } from "./TemplateForm";
import { workflowApi } from "../api/workflows";
import type { WorkflowTemplate } from "../types/workflows";

vi.mock("../api/workflows", () => ({
  workflowApi: { nodes: vi.fn(), saveTemplate: vi.fn() },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
const mounted: Array<{ root: Root; container: HTMLDivElement }> = [];

function template(): WorkflowTemplate {
  const first = {
    id: "infer",
    type: "managed_inference",
    scope: "item" as const,
    version: 1,
    params: { confidence: 0.7, custom: { preserve: true } },
    inputs: {},
    routes: { completed: "human" },
    task_types: ["detect" as const],
    max_attempts: 3,
  };
  return {
    id: "template-1",
    key: "model_review_export_v1",
    name: "模型推理、人工复核与导出",
    version: 2,
    nodes: [
      first,
      {
        id: "human",
        type: "human_review",
        scope: "item",
        version: 1,
        params: {},
        inputs: { proposals: "infer" },
        routes: {},
      },
    ],
  };
}
async function mount(source = template()) {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  mounted.push({ root, container });
  const saved = vi.fn();
  await act(async () =>
    root.render(
      <TemplateForm source={source} onSaved={saved} onClose={vi.fn()} />,
    ),
  );
  return { container, saved };
}
function button(container: HTMLElement, text: string) {
  return Array.from(container.querySelectorAll("button")).find(
    (item) => item.textContent === text,
  )!;
}
function change(
  element: HTMLSelectElement | HTMLTextAreaElement,
  value: string,
) {
  const prototype =
    element instanceof HTMLSelectElement
      ? HTMLSelectElement.prototype
      : HTMLTextAreaElement.prototype;
  act(() => {
    Object.getOwnPropertyDescriptor(prototype, "value")!.set!.call(
      element,
      value,
    );
    element.dispatchEvent(
      new Event(element instanceof HTMLSelectElement ? "change" : "input", {
        bubbles: true,
      }),
    );
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(workflowApi.nodes).mockResolvedValue([
    { type: "managed_inference", scope: "item", version: 1 },
    { type: "human_review", scope: "item", version: 1 },
    { type: "export", scope: "batch", version: 1 },
  ]);
  vi.mocked(workflowApi.saveTemplate).mockResolvedValue({
    ...template(),
    id: "template-2",
    version: 3,
  });
});
afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
});

describe("流程模板表单", () => {
  it("以中文展示步骤和用途，技术字段默认收起", async () => {
    const { container } = await mount();
    expect(document.activeElement).toBe(container.querySelector("h2"));
    expect(container.querySelector("input")!.value).toBe(
      "用已有模型标注并检查",
    );
    const list = container.querySelector("ol")!;
    expect(list.textContent).toContain("用已有模型预标注");
    expect(list.textContent).toContain("使用已登记的模型生成标注草稿");
    expect(list.textContent).toContain("人工检查并确认");
    expect(list.textContent).not.toContain("managed_inference");
    expect(list.textContent).not.toContain("human_review");
    const addSelect = container.querySelector<HTMLSelectElement>("select")!;
    expect(
      Array.from(addSelect.options).map((option) => option.textContent),
    ).toEqual(["用已有模型预标注", "人工检查并确认", "导出标注结果"]);
    const json = container.querySelector("textarea")!;
    expect(json.closest("details")!.open).toBe(false);
    expect(json.closest("details")!.textContent).toContain("内部标识（高级）");
  });

  it("保存新版本时保留所有步骤参数、输入、路由和扩展字段", async () => {
    const source = template();
    const original = structuredClone(source);
    const { container, saved } = await mount(source);
    await act(async () => button(container, "保存流程模板").click());
    expect(workflowApi.saveTemplate).toHaveBeenCalledWith({
      key: source.key,
      name: "用已有模型标注并检查",
      nodes: source.nodes,
    });
    expect(source).toEqual(original);
    expect(saved).toHaveBeenCalledWith(expect.objectContaining({ version: 3 }));
  });

  it("通过中文选项添加步骤并自动生成唯一内部标识", async () => {
    const { container } = await mount();
    change(container.querySelector<HTMLSelectElement>("select")!, "export");
    act(() => button(container, "添加步骤").click());
    act(() => button(container, "添加步骤").click());
    const nodes = JSON.parse(
      container.querySelector("textarea")!.value,
    ) as WorkflowTemplate["nodes"];
    expect(nodes).toHaveLength(4);
    expect(nodes[2]).toEqual(
      expect.objectContaining({
        type: "export",
        scope: "batch",
        version: 1,
        params: {},
        inputs: {},
        routes: {},
      }),
    );
    expect(nodes[2].id).toMatch(/^step-[a-f0-9-]+$/);
    expect(nodes[2].id).not.toBe(nodes[3].id);
  });

  it("无需编辑内部标识即可复制为独立模板", async () => {
    const source = template();
    const { container } = await mount(source);
    const copy = button(container, "复制为独立模板");
    expect(copy.closest("details")).toBeNull();
    act(() => copy.click());
    await act(async () => button(container, "保存流程模板").click());
    expect(workflowApi.saveTemplate).toHaveBeenCalledWith(
      expect.objectContaining({
        key: expect.stringMatching(/^workflow-[a-f0-9-]+$/),
        name: "用已有模型标注并检查（副本）",
        nodes: source.nodes,
      }),
    );
    expect(vi.mocked(workflowApi.saveTemplate).mock.calls[0][0].key).not.toBe(
      source.key,
    );
    expect(container.textContent).toContain("独立模板的第 1 版");
  });

  it("高级步骤格式错误时保留 JSON，禁止覆盖和保存", async () => {
    const { container } = await mount();
    const raw = container.querySelector("textarea")!;
    change(raw, '{"unfinished":');
    expect(button(container, "添加步骤").disabled).toBe(true);
    await act(async () => button(container, "保存流程模板").click());
    expect(raw.value).toBe('{"unfinished":');
    expect(workflowApi.saveTemplate).not.toHaveBeenCalled();
    expect(container.querySelector('[role="alert"]')!.textContent).toContain(
      "步骤格式有误",
    );
  });
});
