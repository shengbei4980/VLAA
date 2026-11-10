// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ProfileForm } from "./ProfileForm";
import { api } from "../api/client";
import { workflowApi } from "../api/workflows";
import { storageApi } from "../api/storage";
import type { ClassLabel, Dataset, ModelVersion } from "../types";
import type { TaskProfile, WorkflowTemplate } from "../types/workflows";

vi.mock("../api/workflows", () => ({ workflowApi: { saveProfile: vi.fn() } }));
vi.mock("../api/storage", () => ({ storageApi: { destinations: vi.fn() } }));
vi.mock("../api/client", () => ({
  api: {
    listClasses: vi.fn(),
    listModels: vi.fn(),
    createClass: vi.fn(),
    deleteClass: vi.fn(),
  },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

const model = {
  id: "model-1",
  name: "已训练的塔楼检测模型",
  version: "v2",
  task_type: "detect",
  format: "pt",
  status: "active",
} as ModelVersion;
const dataset = {
  id: "dataset-1",
  name: "建筑图片",
  task_type: "detect",
} as Dataset;
const mounted: Array<{ root: Root; container: HTMLDivElement }> = [];

function profile(): TaskProfile {
  const models = {
    training: "custom.yaml",
    inference: "model-1",
    proposer: "saved-role",
    segmenter: "saved-segmenter",
    custom_option: { threshold: 0.7 },
  };
  const rule = {
    class_id: "class-1",
    name: "塔楼",
    color: "#ff0000",
    definition: "高耸的建筑",
    aliases: ["楼塔"],
    positive_examples: ["钟塔"],
    custom_rule: "keep",
  };
  return {
    id: "profile-1",
    key: "tower",
    name: "建筑识别方案",
    version: 3,
    config: {
      task_type: "detect",
      geometry: "bbox",
      class_rules: [rule],
      models,
      image_strategy: {
        tile_size: null,
        max_views: 27,
        nested: { preserve: true },
      },
      training: {
        epochs: 12,
        batch_size: 2,
        img_size: 640,
        device: "cpu",
        workers: 0,
        optimizer: "AdamW",
      },
      rules: {
        existing_annotations: "preserve",
        nested: { instances: "separate" },
      },
      quality_conditions: { map50: 0.6 },
      repair_limit: 0,
      max_requests: 10,
      export_format: "yolo",
      outputs: {
        destination_id: "project-output",
        models_destination_id: "model-output",
        include_images: false,
      },
    },
  };
}

async function mount(
  source?: TaskProfile,
  initialDatasetId?: string,
  availableDatasets = [dataset],
  template?: WorkflowTemplate,
) {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  mounted.push({ root, container });
  const saved = vi.fn();
  await act(async () =>
    root.render(
      <ProfileForm
        source={source}
        datasets={availableDatasets}
        initialDatasetId={initialDatasetId}
        template={template}
        onSaved={saved}
        onClose={vi.fn()}
      />,
    ),
  );
  return { container, saved };
}

function control<
  T extends HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement,
>(container: HTMLElement, label: string): T {
  return Array.from(container.querySelectorAll("label"))
    .find((item) => item.textContent?.startsWith(label))!
    .querySelector("input, select, textarea") as T;
}

function change(
  element: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement,
  value: string,
) {
  const prototype =
    element instanceof HTMLSelectElement
      ? HTMLSelectElement.prototype
      : element instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype;
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

async function submit(container: HTMLElement) {
  await act(async () =>
    container
      .querySelector("form")!
      .dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(storageApi.destinations).mockResolvedValue([]);
  vi.mocked(api.listModels).mockResolvedValue({ items: [model], total: 1 });
  vi.mocked(api.listClasses).mockResolvedValue([
    { id: "class-1", name: "塔楼", color: "#ff0000" },
  ] as Awaited<ReturnType<typeof api.listClasses>>);
  vi.mocked(workflowApi.saveProfile).mockResolvedValue({
    ...profile(),
    id: "profile-2",
    version: 4,
  });
});
afterEach(() => {
  for (const { root, container } of mounted.splice(0)) {
    act(() => root.unmount());
    container.remove();
  }
  vi.restoreAllMocks();
});

describe("类别与模型方案表单", () => {
  it("单独移除类别后添加其他类别，保存仍只包含所选类别与说明", async () => {
    const source = profile();
    const retained = {
      class_id: "class-2",
      name: "people",
      definition: "保留这段识别说明",
    };
    source.config.class_rules.push(retained);
    vi.mocked(api.listClasses).mockResolvedValue(
      source.config.class_rules.map((rule) => ({
        id: rule.class_id,
        name: rule.name,
        color: "#22c55e",
      })) as ClassLabel[],
    );
    vi.mocked(api.createClass).mockResolvedValue({
      id: "class-road",
      name: "道路",
      color: "#22c55e",
    } as ClassLabel);
    const { container } = await mount(source);
    act(() =>
      container
        .querySelector<HTMLButtonElement>('[aria-label="从本方案移除 塔楼"]')!
        .click(),
    );
    expect(api.deleteClass).not.toHaveBeenCalled();
    change(control<HTMLInputElement>(container, "新类别名称"), "道路");
    await act(async () =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "添加类别")!
        .click(),
    );
    expect(
      Array.from(container.querySelectorAll(".workflow-rule strong")).map(
        (item) => item.textContent,
      ),
    ).toEqual(["people", "道路"]);
    await submit(container);
    expect(
      vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config.class_rules,
    ).toEqual([
      retained,
      { class_id: "class-road", name: "道路", color: "#22c55e" },
    ]);
    expect(source.config.class_rules).toHaveLength(2);
  });

  it("移除最后一个类别后要求重新选择，避免空方案自动纳入所有类别", async () => {
    const { container } = await mount(profile());
    act(() =>
      container
        .querySelector<HTMLButtonElement>('[aria-label="从本方案移除 塔楼"]')!
        .click(),
    );
    await submit(container);
    expect(workflowApi.saveProfile).not.toHaveBeenCalled();
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "请先手动添加类别或读取数据集类别",
    );
  });

  it("删除数据集类别先确认，成功后只移除指定类别并保留其他说明", async () => {
    const source = profile();
    const retained = {
      class_id: "class-2",
      name: "people",
      definition: "保留人物规则",
    };
    source.config.class_rules.push(retained);
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    vi.mocked(api.deleteClass).mockResolvedValue(undefined);
    const { container } = await mount(source);
    const remove = container.querySelector<HTMLButtonElement>(
      '[aria-label="删除数据集类别 塔楼"]',
    )!;
    await act(async () => remove.click());
    expect(api.deleteClass).not.toHaveBeenCalled();
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining("建筑图片"));
    confirm.mockReturnValue(true);
    await act(async () => remove.click());
    expect(api.deleteClass).toHaveBeenCalledWith(dataset.id, "class-1");
    await submit(container);
    expect(
      vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config.class_rules,
    ).toEqual([retained]);
  });

  it("已被标注引用时保留类别，重试发现类别已删除时可恢复一致状态", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(api.deleteClass).mockRejectedValue(
      new Error("Delete annotations using this class before deleting it."),
    );
    const { container } = await mount(profile());
    const remove = container.querySelector<HTMLButtonElement>(
      '[aria-label="删除数据集类别 塔楼"]',
    )!;
    await act(async () => remove.click());
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "该类别已有标注使用",
    );
    expect(container.querySelector(".workflow-rule strong")?.textContent).toBe(
      "塔楼",
    );
    vi.mocked(api.deleteClass).mockRejectedValue(
      new Error("Class label was not found."),
    );
    await act(async () => remove.click());
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(container.querySelector(".workflow-rule")).toBeNull();
    expect(container.querySelector('[role="status"]')?.textContent).toContain(
      "已没有类别",
    );
  });

  it("全局任务说明与已有标注规则同步编辑，保存和复制均保留完整内容", async () => {
    const source = profile();
    const originalRules = source.config.rules;
    source.config.rules = {
      ...source.config.rules,
      task_instructions: "先前的整批要求",
    };
    const { container } = await mount(source);
    const instructions = control<HTMLTextAreaElement>(container, "任务内容");
    expect(instructions.value).toBe("先前的整批要求");
    expect(instructions.closest("details")).toBeNull();
    change(instructions, "识别塔体和屋顶。排除水印，模糊区域交由人工复核。");
    const raw = container.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="标注规则 JSON"]',
    )!;
    const updatedRules = {
      ...source.config.rules,
      task_instructions: instructions.value,
    };
    expect(JSON.parse(raw.value)).toEqual(updatedRules);
    act(() =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "复制为独立方案")!
        .click(),
    );
    await submit(container);
    expect(
      vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config.rules,
    ).toEqual(updatedRules);
    change(
      raw,
      JSON.stringify({
        ...updatedRules,
        task_instructions: "高级配置中的更新",
      }),
    );
    expect(instructions.value).toBe("高级配置中的更新");
    change(instructions, "");
    expect(JSON.parse(raw.value)).toEqual(originalRules);
  });

  it("未标注且没有类别的数据集可手动添加，说明与真实类别编号一起保存", async () => {
    const added = {
      id: "class-new",
      dataset_id: dataset.id,
      name: "塔体",
      color: "#22c55e",
    } as ClassLabel;
    vi.mocked(api.listClasses).mockResolvedValue([]);
    vi.mocked(api.createClass).mockResolvedValue(added);
    const { container } = await mount();
    const name = control<HTMLInputElement>(container, "新类别名称");
    change(name, "  塔体  ");
    await act(async () =>
      name.dispatchEvent(
        new KeyboardEvent("keydown", {
          key: "Enter",
          bubbles: true,
          cancelable: true,
        }),
      ),
    );
    expect(api.createClass).toHaveBeenCalledWith(dataset.id, "塔体");
    expect(workflowApi.saveProfile).not.toHaveBeenCalled();
    expect(name.value).toBe("");
    change(
      control<HTMLTextAreaElement>(container, "怎样识别这个类别"),
      "建筑的主体部分，排除周围树木。",
    );
    vi.mocked(api.listClasses).mockResolvedValue([added]);
    await act(async () =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "读取数据集类别")!
        .click(),
    );
    await submit(container);
    expect(
      vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config.class_rules,
    ).toEqual([
      {
        class_id: "class-new",
        name: "塔体",
        color: "#22c55e",
        definition: "建筑的主体部分，排除周围树木。",
      },
    ]);
  });

  it("添加已有同名类别时复用编号并保留类别定义", async () => {
    const source = profile();
    const { container } = await mount(source);
    change(control<HTMLInputElement>(container, "新类别名称"), "塔楼");
    await act(async () =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "添加类别")!
        .click(),
    );
    expect(api.createClass).not.toHaveBeenCalled();
    expect(container.querySelector('[role="status"]')?.textContent).toContain(
      "已选用数据集中的类别",
    );
    await submit(container);
    expect(
      vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config.class_rules,
    ).toEqual(source.config.class_rules);
  });

  it("创建响应失败后保留名称，重试可读取已提交类别并使用所选数据集", async () => {
    const otherDataset = { ...dataset, id: "dataset-2", name: "未标注街景" };
    const added = {
      id: "class-road",
      dataset_id: otherDataset.id,
      name: "道路",
      color: "#22c55e",
    } as ClassLabel;
    vi.mocked(api.listClasses).mockResolvedValue([]);
    vi.mocked(api.createClass).mockRejectedValue(new Error("连接中断，请重试"));
    const { container } = await mount(undefined, otherDataset.id, [
      dataset,
      otherDataset,
    ]);
    const name = control<HTMLInputElement>(container, "新类别名称");
    const add = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "添加类别",
    )!;
    change(name, "道路");
    await act(async () => add.click());
    expect(name.value).toBe("道路");
    expect(container.querySelector('[role="alert"]')?.textContent).toContain(
      "连接中断",
    );
    vi.mocked(api.listClasses).mockResolvedValue([added]);
    await act(async () => add.click());
    expect(api.createClass).toHaveBeenCalledTimes(1);
    expect(api.createClass).toHaveBeenCalledWith(otherDataset.id, "道路");
    expect(api.listClasses).toHaveBeenLastCalledWith(otherDataset.id);
    expect(container.querySelector('[role="alert"]')).toBeNull();
    await submit(container);
    expect(
      vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config.class_rules[0]
        .class_id,
    ).toBe(added.id);
  });

  it("视觉 LLM 流程仅显示所需服务，保存时仍保留复用方案中的模型配置", async () => {
    const source = profile();
    const { container } = await mount(source, undefined, [dataset], {
      id: "llm-template",
      key: "vision_review_export_v1",
      name: "用视觉 LLM 标注并检查",
      version: 1,
      nodes: [
        {
          id: "propose",
          type: "vision_propose",
          version: 1,
          scope: "item",
          params: { source: "vision" },
        },
      ],
    });
    const labels = Array.from(container.querySelectorAll("label")).map(
      (item) => item.textContent,
    );
    expect(labels.some((label) => label?.startsWith("已有的预标注模型"))).toBe(
      false,
    );
    expect(
      labels.some((label) => label?.startsWith("训练时使用的基础模型")),
    ).toBe(false);
    expect(labels.some((label) => label?.startsWith("训练轮数"))).toBe(false);
    expect(container.textContent).toContain("本流程由视觉 LLM 直接看图");
    change(control<HTMLSelectElement>(container, "标注导出格式"), "labelme");
    expect(container.textContent).toContain("LabelMe 5.8");
    await submit(container);
    expect(vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config).toEqual({
      ...source.config,
      export_format: "labelme",
    });
  });

  it("优先展示中文常用参数，将内部标识和完整 JSON 折叠到高级配置", async () => {
    const { container } = await mount(profile());
    expect(container.textContent).toContain("编辑类别与模型方案");
    expect(document.activeElement).toBe(container.querySelector("h2"));
    expect(container.textContent).toContain("正在编辑第 3 版");
    expect(control<HTMLInputElement>(container, "训练轮数").value).toBe("12");
    const modelSelect = control<HTMLSelectElement>(
      container,
      "已有的预标注模型",
    );
    expect(modelSelect.selectedOptions[0].textContent).toBe(
      "已训练的塔楼检测模型 · v2",
    );
    for (const textarea of container.querySelectorAll("textarea[aria-label]")) {
      const advanced = textarea.closest("details")!;
      expect(advanced.open).toBe(false);
      expect(advanced.querySelector("summary")!.textContent).toBe("高级配置");
    }
    expect(
      control<HTMLInputElement>(container, "内部标识（高级）").closest(
        "details",
      )!.open,
    ).toBe(false);
    expect(container.textContent).toContain("实际服务由视觉模型设置决定");
    expect(container.textContent).not.toContain("视觉提出者配置");
    expect(
      container.querySelectorAll('input[value="saved-role"]'),
    ).toHaveLength(0);
  });

  it("修改常用参数后完整保留高级字段和上一版本", async () => {
    const source = profile();
    const original = structuredClone(source);
    const { container, saved } = await mount(source);
    change(control<HTMLInputElement>(container, "训练轮数"), "20");
    change(
      control<HTMLSelectElement>(container, "视觉服务查看图片的方式"),
      "tiles",
    );
    await submit(container);
    const request = vi.mocked(workflowApi.saveProfile).mock.calls[0][0];
    expect(request.key).toBe(source.key);
    expect(request.config).toEqual({
      ...original.config,
      training: { ...original.config.training, epochs: 20 },
      image_strategy: { ...original.config.image_strategy, tile_size: 1024 },
    });
    expect(source).toEqual(original);
    expect(saved).toHaveBeenCalledWith(expect.objectContaining({ version: 4 }));
  });

  it("一键复制时生成独立标识，原方案保持原样", async () => {
    const source = profile();
    const original = structuredClone(source);
    const { container } = await mount(source);
    const copyButton = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "复制为独立方案",
    )!;
    expect(copyButton.closest("details")).toBeNull();
    act(() => copyButton.click());
    await submit(container);
    expect(workflowApi.saveProfile).toHaveBeenCalledWith(
      expect.objectContaining({
        key: expect.stringMatching(/^profile-[a-f0-9-]+$/),
        name: source.name + "（副本）",
      }),
    );
    expect(vi.mocked(workflowApi.saveProfile).mock.calls[0][0].key).not.toBe(
      source.key,
    );
    expect(source).toEqual(original);
  });

  it("新建方案自动生成不同标识，读取类别后可以直接保存", async () => {
    const first = await mount();
    const second = await mount();
    const firstKey = control<HTMLInputElement>(
      first.container,
      "内部标识（高级）",
    ).value;
    expect(firstKey).toMatch(/^profile-[a-f0-9-]+$/);
    expect(
      control<HTMLInputElement>(second.container, "内部标识（高级）").value,
    ).not.toBe(firstKey);
    await act(async () =>
      Array.from(first.container.querySelectorAll("button"))
        .find((button) => button.textContent === "读取数据集类别")!
        .click(),
    );
    await submit(first.container);
    expect(
      vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config.class_rules,
    ).toEqual([{ class_id: "class-1", name: "塔楼", color: "#ff0000" }]);
  });

  it("新建方案沿用主页面已选的数据集和任务，已有方案保留原任务", async () => {
    const segmentationDataset = {
      ...dataset,
      id: "dataset-segment",
      task_type: "segment" as const,
    };
    const availableDatasets = [dataset, segmentationDataset];
    const created = await mount(
      undefined,
      segmentationDataset.id,
      availableDatasets,
    );
    expect(
      control<HTMLSelectElement>(created.container, "类别来源数据集").value,
    ).toBe(segmentationDataset.id);
    expect(
      control<HTMLSelectElement>(created.container, "想识别什么").value,
    ).toBe("segment");
    const edited = await mount(
      profile(),
      segmentationDataset.id,
      availableDatasets,
    );
    expect(
      control<HTMLSelectElement>(edited.container, "类别来源数据集").value,
    ).toBe(segmentationDataset.id);
    expect(
      control<HTMLSelectElement>(edited.container, "想识别什么").value,
    ).toBe("detect");
  });

  it("重新读取同一类别时保留定义、例子和自定义规则", async () => {
    const source = profile();
    const { container } = await mount(source);
    await act(async () =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "读取数据集类别")!
        .click(),
    );
    await submit(container);
    expect(
      vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config.class_rules,
    ).toEqual(source.config.class_rules);
  });

  it("高级 JSON 无效时保留输入并阻止常用字段覆盖它", async () => {
    const { container } = await mount(profile());
    const raw = container.querySelector<HTMLTextAreaElement>(
      'textarea[aria-label="训练参数 JSON"]',
    )!;
    change(raw, '{"custom":');
    change(control<HTMLInputElement>(container, "训练轮数"), "8");
    expect(raw.value).toBe('{"custom":');
    expect(raw.closest("details")!.open).toBe(true);
    await submit(container);
    expect(workflowApi.saveProfile).not.toHaveBeenCalled();
    expect(container.querySelector('[role="alert"]')!.textContent).toContain(
      "训练参数的格式有误",
    );
  });

  it("模型和目录列表读取失败时仍保留原有选择与配置", async () => {
    vi.mocked(api.listModels).mockRejectedValue(new Error("unavailable"));
    vi.mocked(storageApi.destinations).mockRejectedValue(
      new Error("unavailable"),
    );
    const source = profile();
    const { container } = await mount(source);
    expect(
      control<HTMLSelectElement>(container, "已有的预标注模型")
        .selectedOptions[0].textContent,
    ).toBe("已保存的模型（等待核验）");
    await submit(container);
    expect(vi.mocked(workflowApi.saveProfile).mock.calls[0][0].config).toEqual(
      source.config,
    );
  });
});
