// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { WorkflowView } from "./WorkflowView";
import { workflowApi, type PreparedWorkflowRun } from "../api/workflows";
import { api } from "../api/client";
import type { Dataset, ImageItem } from "../types";
import type { TaskProfile, WorkflowTemplate } from "../types/workflows";
vi.mock("../api/workflows", () => ({
  workflowApi: {
    profiles: vi.fn(),
    templates: vi.fn(),
    runs: vi.fn(),
    prepareRun: vi.fn(),
    createRun: vi.fn(),
    projects: vi.fn(),
    saveProject: vi.fn(),
  },
}));
vi.mock("../api/client", () => ({
  api: { listImages: vi.fn() },
  apiUrl: (path: string) => path,
}));
vi.mock("./WorkflowRunView", () => ({
  WorkflowRunView: ({ runId, initialReviewId }: { runId: string; initialReviewId?: string }) => <p>运行 {runId}{initialReviewId ? ` · 复核 ${initialReviewId}` : ""}</p>,
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
describe("reviewable launch scope", () => {
  it("opens the exact workflow review linked from a dataset", async () => {
    vi.clearAllMocks();
    window.history.replaceState(null, "", "#workflow/run-target/review/review-target");
    vi.mocked(workflowApi.profiles).mockResolvedValue([]);
    vi.mocked(workflowApi.templates).mockResolvedValue([]);
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => root.render(<WorkflowView datasets={[]} onSettings={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
    expect(container.textContent).toBe("运行 run-target · 复核 review-target");
    act(() => root.unmount());
    window.history.replaceState(null, "", "#workflow");
  });

  it("preselects the originating dataset and latest visual LLM template", async () => {
    vi.clearAllMocks();
    window.history.replaceState(null, "", "#workflow");
    const datasets: Dataset[] = [
      { id: "first", name: "First", task_type: "detect", description: null, image_count: 0, annotated_image_count: 0, class_count: 0, created_at: "", updated_at: "" },
      { id: "target", name: "Tower", task_type: "segment", description: null, image_count: 0, annotated_image_count: 0, class_count: 0, created_at: "", updated_at: "" },
    ];
    vi.mocked(workflowApi.profiles).mockResolvedValue([]);
    vi.mocked(workflowApi.templates).mockResolvedValue([
      { id: "model", key: "model", name: "Model", version: 1, nodes: [] },
      { id: "llm-old", key: "vision", name: "Visual", version: 1, nodes: [{ id: "propose", type: "vision_propose", scope: "item", version: 1, params: { source: "vision" } }] },
      { id: "llm-new", key: "vision", name: "Visual", version: 2, nodes: [{ id: "propose", type: "vision_propose", scope: "item", version: 1, params: { source: "vision" } }] },
    ]);
    vi.mocked(workflowApi.runs).mockResolvedValue([]);
    vi.mocked(api.listImages).mockResolvedValue({ total: 0, items: [] });
    const container = document.createElement("div");
    const root = createRoot(container);
    await act(async () => root.render(<WorkflowView datasets={datasets} initialDatasetId="target" onSettings={vi.fn()} onOpenTraining={vi.fn()} onOpenModel={vi.fn()} />));
    expect(container.querySelector<HTMLSelectElement>(".workflow-dataset-select select")?.value).toBe("target");
    expect(container.querySelector<HTMLSelectElement>(".workflow-advanced select")?.value).toBe("llm-new");
    expect(api.listImages).toHaveBeenCalledWith("target");
    expect(api.listImages).not.toHaveBeenCalledWith("first");
    expect(container.textContent).toContain("请核对类别与模型方案");
    act(() => root.unmount());
  });
  beforeEach(() => {
    window.localStorage.clear();
    vi.mocked(workflowApi.projects).mockResolvedValue([]);
  });
  it("offers Chinese task cards and keeps older versions in advanced settings", async () => {
    vi.clearAllMocks();
    window.history.replaceState(null, "", "#workflow");
    const templates: WorkflowTemplate[] = [
      {
        id: "training-v2",
        key: "confirmed_train_evaluate_v1",
        name: "已确认快照训练与评估",
        version: 2,
        nodes: [
          {
            id: "train",
            type: "train",
            version: 1,
            scope: "batch",
            params: {},
          },
        ],
      },
      {
        id: "training-v1",
        key: "confirmed_train_evaluate_v1",
        name: "已确认快照训练与评估",
        version: 1,
        nodes: [],
      },
      {
        id: "label-v1",
        key: "model_review_export_v1",
        name: "模型推理、人工复核与导出",
        version: 1,
        nodes: [
          {
            id: "infer",
            type: "managed_inference",
            version: 1,
            scope: "item",
            params: {},
          },
          {
            id: "confirm",
            type: "human_review",
            version: 1,
            scope: "item",
            params: {},
          },
        ],
      },
    ];
    vi.mocked(workflowApi.profiles).mockResolvedValue([]);
    vi.mocked(workflowApi.templates).mockResolvedValue(templates);
    vi.mocked(workflowApi.runs).mockResolvedValue([]);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    await act(async () =>
      root.render(
        <WorkflowView
          datasets={[]}
          onSettings={vi.fn()}
          onOpenTraining={vi.fn()}
          onOpenModel={vi.fn()}
        />,
      ),
    );
    const cards = container.querySelectorAll<HTMLButtonElement>(
      ".workflow-template-card",
    );
    expect(cards).toHaveLength(2);
    const setupSteps = container.querySelectorAll<HTMLButtonElement>(
      ".workflow-setup-guide button",
    );
    expect(setupSteps).toHaveLength(5);
    const startHeading = container.querySelector<HTMLElement>(
      "#workflow-start-title",
    )!;
    startHeading.scrollIntoView = vi.fn();
    act(() => setupSteps[4].click());
    expect(startHeading.scrollIntoView).toHaveBeenCalledWith({
      block: "start",
    });
    expect(window.location.hash).toBe("#workflow");
    expect(
      container.querySelector(".workflow-launch-summary")?.textContent,
    ).toContain("待选择方案");
    expect(cards[0].textContent).toContain("用已标注图片训练模型");
    expect(cards[0].textContent).toContain("第 2 版");
    expect(cards[0].getAttribute("aria-pressed")).toBe("true");
    expect(
      container.querySelector(".workflow-advanced")?.hasAttribute("open"),
    ).toBe(false);
    expect(
      container.querySelectorAll(".workflow-advanced option"),
    ).toHaveLength(3);
    expect(container.textContent).not.toContain("已确认快照训练与评估");
    expect(
      container.querySelector(".workflow-flow-preview")?.textContent,
    ).toContain("训练识别模型");
    act(() => cards[1].click());
    expect(cards[1].getAttribute("aria-pressed")).toBe("true");
    expect(
      container.querySelector(".workflow-flow-preview")?.textContent,
    ).toContain("用已有模型预标注");
    expect(
      container.querySelector(".workflow-flow-preview")?.textContent,
    ).toContain("人工检查并确认");
    expect(workflowApi.prepareRun).not.toHaveBeenCalled();
    expect(workflowApi.createRun).not.toHaveBeenCalled();
    act(() => root.unmount());
    container.remove();
    window.history.replaceState(null, "", "#");
  });

  it.each([false, true])(
    "freezes the chosen labeling source and starts only after approval (LLM %s)",
    async (useLlm) => {
      vi.clearAllMocks();
      window.history.replaceState(null, "", "#workflow");
      const dataset: Dataset = {
        id: "dataset",
        name: "Tower",
        task_type: "detect",
        description: null,
        image_count: 1,
        annotated_image_count: 0,
        class_count: 1,
        created_at: "",
        updated_at: "",
      };
      const profile: TaskProfile = {
        id: "profile",
        key: "tower",
        name: "Tower",
        version: 1,
        config: {
          task_type: "detect",
          geometry: "bbox",
          class_rules: [{ class_id: "class", name: "tower" }],
          models: { inference: "previously-trained-model" },
          image_strategy: {},
          training: {},
          quality_conditions: {},
          repair_limit: 0,
          max_requests: 10,
          export_format: "yolo",
        },
      };
      const template: WorkflowTemplate = {
        id: "template",
        key: "visual",
        name: "Visual",
        version: 1,
        nodes: useLlm
          ? [
              {
                id: "propose",
                type: "vision_propose",
                scope: "item",
                version: 1,
                params: { source: "vision" },
              },
            ]
          : [],
      };
      const image: ImageItem = {
        id: "image",
        dataset_id: "dataset",
        file_name: "tower.png",
        width: 10,
        height: 10,
        split: "train",
        status: "unannotated",
        file_url: "/image",
        created_at: "",
        updated_at: "",
      };
      vi.mocked(workflowApi.profiles).mockResolvedValue([profile]);
      vi.mocked(workflowApi.templates).mockResolvedValue([template]);
      vi.mocked(workflowApi.runs).mockResolvedValue([]);
      const project = {
        id: "tower-project",
        name: "塔体项目",
        root_path: "E:\\塔体项目",
        destination_id: "destination",
        created_at: "",
      };
      vi.mocked(workflowApi.projects).mockResolvedValue([project]);
      vi.mocked(api.listImages).mockResolvedValue({ total: 1, items: [image] });
      const prepared: PreparedWorkflowRun = {
        request: {
          idempotency_key: "server-frozen",
          template_id: "template",
          profile_id: "profile",
          dataset_id: "dataset",
          project_id: project.id,
          image_ids: ["image"],
          next_batches: [],
          inputs: { vision_approvals: { proposer: { config_hash: "frozen" } } },
        },
        vision_approvals: {
          proposer: {
            mode: "local",
            api_base: "http://127.0.0.1:9000/v1",
            model: "vision",
            max_requests: 10,
          },
        },
        scope: [
          {
            dataset_id: "dataset",
            image_ids: ["image"],
            image_hashes: ["hash"],
          },
        ],
        profile_snapshot: profile.config,
        template_snapshot: template,
        outputs: {
          project,
          json_path: "E:\\塔体项目\\workflow-<run-id>\\annotations\\json",
          images_path: "E:\\塔体项目\\workflow-<run-id>\\annotations\\images",
          annotations: {
            destination_id: "dest",
            path: "E:\\results",
            enabled: true,
          },
          models: { destination_id: null, path: null, enabled: true },
          evaluations: { destination_id: null, path: null, enabled: false },
          include_images: false,
        },
      };
      vi.mocked(workflowApi.prepareRun).mockResolvedValue(prepared);
      vi.mocked(workflowApi.createRun).mockResolvedValue({
        id: "new-run",
      } as never);
      const container = document.createElement("div");
      document.body.append(container);
      const root = createRoot(container);
      await act(async () =>
        root.render(
          <WorkflowView
            datasets={[dataset]}
            onSettings={vi.fn()}
            onOpenTraining={vi.fn()}
            onOpenModel={vi.fn()}
          />,
        ),
      );
      const button = (text: string) =>
        Array.from(
          container.querySelectorAll<HTMLButtonElement>("button"),
        ).find((item) => item.textContent === text)!;
      const projectSelect = container.querySelector<HTMLSelectElement>(
        "[aria-labelledby='workflow-project-title'] select",
      )!;
      act(() => {
        projectSelect.value = project.id;
        projectSelect.dispatchEvent(new Event("change", { bubbles: true }));
      });
      await act(async () => button("预览本次任务").click());
      expect(workflowApi.prepareRun).toHaveBeenCalledWith(
        expect.objectContaining({
          inputs: useLlm ? {} : { model_id: "previously-trained-model" },
          project_id: project.id,
        }),
      );
      expect(workflowApi.createRun).not.toHaveBeenCalled();
      expect(container.textContent).toContain("http://127.0.0.1:9000/v1");
      expect(container.textContent).toContain("E:\\results");
      expect(
        container.querySelector(".workflow-launch")?.textContent,
      ).toContain("E:\\塔体项目\\workflow-<run-id>\\annotations\\json");
      expect(container.textContent).toContain("标注输出仅保存标签与清单");
      expect(button("启动运行").disabled).toBe(true);
      await act(async () => button("返回修改").click());
      act(() => {
        projectSelect.value = "";
        projectSelect.dispatchEvent(new Event("change", { bubbles: true }));
      });
      expect(container.querySelector(".workflow-launch")).toBeNull();
      expect(
        window.localStorage.getItem("vlaa.workflow.last-project"),
      ).toBeNull();
      const defaultPrepared = {
        ...prepared,
        request: { ...prepared.request, project_id: undefined },
        outputs: undefined,
      };
      vi.mocked(workflowApi.prepareRun).mockResolvedValue(defaultPrepared);
      await act(async () => button("预览本次任务").click());
      expect(workflowApi.prepareRun).toHaveBeenLastCalledWith(
        expect.objectContaining({ project_id: undefined }),
      );
      act(() =>
        container
          .querySelector<HTMLInputElement>(
            '.workflow-launch input[type="checkbox"]',
          )!
          .click(),
      );
      await act(async () => button("启动运行").click());
      expect(workflowApi.createRun).toHaveBeenCalledWith(
        defaultPrepared.request,
      );
      expect(container.textContent).toContain("运行 new-run");
      act(() => root.unmount());
      container.remove();
      window.history.replaceState(null, "", "#");
    },
  );
});
