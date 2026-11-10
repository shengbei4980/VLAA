// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ArtifactReviewView } from "./ArtifactReviewView";
import { workflowApi } from "../api/workflows";
import { annotationPreview } from "../workflow/annotation-preview";
import { visionApi } from "../api/vision";
import { encodeMask } from "../workflow/masks";
import type {
  ReviewPayload,
  TaskProfileConfig,
  WorkflowReview,
} from "../types/workflows";

const { maskCanvasProps } = vi.hoisted(() => ({ maskCanvasProps: vi.fn() }));
type MaskCanvasProps = {
  payload: ReviewPayload;
  activeClassId?: string;
  selectedId: string;
  newObjectRequest?: number;
  disabled?: boolean;
  onDrawingChange?: (drawing: boolean) => void;
  onChange: (payload: ReviewPayload) => void;
  onSelect: (id: string) => void;
  onSamBox: (box: { x: number; y: number; width: number; height: number }) => void;
};

vi.mock("../api/workflows", () => ({
  workflowApi: {
    review: vi.fn(),
    saveDraft: vi.fn(),
    savePreview: vi.fn(),
    exportLabelMe: vi.fn(),
    decide: vi.fn(),
  },
}));
vi.mock("../workflow/annotation-preview", () => ({
  annotationPreview: vi.fn(),
}));
vi.mock("../api/vision", () => ({
  visionApi: { mask: vi.fn(), saveMask: vi.fn(), sam: vi.fn() },
}));
vi.mock("../annotation/AnnotationCanvas", () => ({
  AnnotationCanvas: ({
    onChange,
  }: {
    onChange: (drafts: unknown[]) => void;
  }) => (
    <button
      onClick={() =>
        onChange([
          {
            id: "obj-1",
            class_id: "class-2",
            type: "bbox",
            bbox: { x: 4, y: 5, width: 8, height: 9 },
            source: "manual",
          },
        ])
      }
    >
      修改框和类别
    </button>
  ),
}));
vi.mock("../annotation/MaskReviewCanvas", () => ({
  MaskReviewCanvas: (props: MaskCanvasProps) => {
    maskCanvasProps(props);
    return (
      <div data-testid="mask-canvas" data-selected-id={props.selectedId}>
        <button disabled={props.disabled} onClick={() => props.onDrawingChange?.(true)}>开始测试轮廓</button>
        <button disabled={props.disabled} onClick={() => {
          props.onChange({
            ...props.payload,
            objects: [...props.payload.objects, {
              object_id: "drawn-object", class_id: props.activeClassId ?? null,
              type: "polygon", source: "manual", decision: "accepted",
              mask: manualMask(),
            }],
          });
          props.onSelect("drawn-object");
          props.onDrawingChange?.(false);
        }}>完成测试轮廓</button>
        <button disabled={props.disabled} onClick={() => props.onDrawingChange?.(false)}>取消测试轮廓</button>
        <button disabled={props.disabled} onClick={() => props.onSamBox({ x: 1, y: 1, width: 8, height: 8 })}>测试 SAM 修边</button>
      </div>
    );
  },
}));
vi.mock("../components/ExportLocationsPanel", () => ({
  ExportLocationsPanel: () => null,
}));

const profile: TaskProfileConfig = {
  task_type: "detect",
  geometry: "bbox",
  class_rules: [
    { class_id: "class-1", name: "tower" },
    { class_id: "class-2", name: "window" },
  ],
  models: {},
  image_strategy: {},
  training: {},
  quality_conditions: {},
  repair_limit: 0,
  max_requests: 10,
  export_format: "yolo",
};
const payload: ReviewPayload = {
  image_id: "image-1",
  dataset_id: "dataset-1",
  image_url: "/image",
  width: 30,
  height: 30,
  base_revision: 0,
  objects: [
    {
      object_id: "obj-1",
      class_id: "class-1",
      type: "bbox",
      bbox: { x: 1, y: 2, width: 8, height: 9 },
      provenance_json: { backend: "managed" },
    },
  ],
  issues: [],
};
const review: WorkflowReview = {
  id: "review-1",
  run_id: "run-1",
  item_id: "item-1",
  revision: 1,
  base_revision: 0,
  status: "pending",
  payload,
};
function manualMask() {
  const pixels = new Uint8Array(30 * 30);
  pixels[10 * 30 + 10] = 1;
  pixels[10 * 30 + 11] = 1;
  return encodeMask(pixels, 30, 30);
}
const segmentProfile: TaskProfileConfig = { ...profile, task_type: "segment", geometry: "mask" };
const savedPreview = {
  file_name: "image-1-draft-r1-123.png",
  saved_path: "D:/managed/exports/image-1-draft-r1-123.png",
  download_url: "/api/workflows/reviews/review-1/preview/123",
  width: 30,
  height: 30,
  review_revision: 1,
  preview_state: "draft" as const,
};
const savedLabelMe = {
  file_name: "image-1-draft-r1-123.json",
  saved_path: "D:/managed/exports/image-1-draft-r1-123.json",
  download_url: "/api/workflows/reviews/review-1/labelme/123",
  review_revision: 1,
  export_state: "draft" as const,
  exported_objects: 1,
  pending_objects: 1,
};
let root: Root;
let container: HTMLDivElement;
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(workflowApi.review).mockResolvedValue(structuredClone(review));
  vi.mocked(workflowApi.saveDraft).mockImplementation(
    async (_id, revision, value) => ({
      ...review,
      revision: revision + 1,
      payload: value,
    }),
  );
  vi.mocked(workflowApi.decide).mockResolvedValue({
    ...review,
    status: "decided",
    decision: "approved",
  });
  vi.mocked(workflowApi.savePreview).mockResolvedValue(savedPreview);
  vi.mocked(workflowApi.exportLabelMe).mockResolvedValue(savedLabelMe);
  vi.mocked(visionApi.saveMask).mockResolvedValue({ mask_ref: "masks/manual.json" } as never);
  vi.mocked(visionApi.mask).mockResolvedValue({ rle: manualMask() } as never);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  act(() => root.unmount());
  container.remove();
});
const button = (label: string) =>
  Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find(
    (item) => item.textContent === label,
  )!;
const render = async (taskProfile = profile, onDirtyChange?: (dirty: boolean) => void) => {
  await act(async () =>
    root.render(
      <ArtifactReviewView
        reviewId="review-1"
        profile={taskProfile}
        onBack={vi.fn()}
        onSaved={vi.fn()}
        onDirtyChange={onDirtyChange}
      />,
    ),
  );
};

describe("human review gate", () => {
  it("starts drawing a missing object without adding an empty candidate", async () => {
    await render(segmentProfile);
    act(() => button("新增漏标对象").click());
    const canvas = maskCanvasProps.mock.lastCall![0] as MaskCanvasProps;
    expect(canvas.newObjectRequest).toBe(1);
    expect(canvas.activeClassId).toBe("class-1");
    expect(canvas.payload.objects).toHaveLength(1);
    expect(button("保存复核草稿").disabled).toBe(true);
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
  });

  it("protects an unfinished polygon while leaving the canvas available to finish or cancel", async () => {
    const onDirtyChange = vi.fn();
    await render(segmentProfile, onDirtyChange);
    act(() => button("开始测试轮廓").click());
    expect(onDirtyChange).toHaveBeenLastCalledWith(true);
    expect(container.textContent).toContain("请先完成轮廓或取消本次绘制");
    for (const label of ["保存复核草稿", "导出 LabelMe JSON", "保存标注预览 PNG", "确认整图并应用", "撤销", "重做"])
      expect(button(label).disabled).toBe(true);
    expect(container.querySelector<HTMLSelectElement>('[aria-label="对象 1 类别"]')!.disabled).toBe(true);
    expect(button("完成测试轮廓").disabled).toBe(false);
    expect(button("取消测试轮廓").disabled).toBe(false);
    expect((maskCanvasProps.mock.lastCall![0] as MaskCanvasProps).disabled).toBe(false);
    act(() => button("取消测试轮廓").click());
    expect(onDirtyChange).toHaveBeenLastCalledWith(false);
    expect(button("导出 LabelMe JSON").disabled).toBe(false);
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
    expect(workflowApi.exportLabelMe).not.toHaveBeenCalled();
  });

  it("retains completed manual masks through undo, redo, JSON export and draft save", async () => {
    const original = {
      ...payload.objects[0], type: "polygon" as const, bbox: undefined,
      mask: manualMask(), mask_ref: "masks/original.json",
    };
    const issues = [{ id: "boundary", type: "boundary", message: "需要检查轮廓", resolution: "dismissed", reason: "已查看原图" }];
    vi.mocked(workflowApi.review).mockResolvedValue({
      ...review, payload: { ...payload, objects: [original], issues },
    });
    await render(segmentProfile);
    act(() => button("新增漏标对象").click());
    act(() => button("开始测试轮廓").click());
    act(() => button("完成测试轮廓").click());
    expect((maskCanvasProps.mock.lastCall![0] as MaskCanvasProps).payload.objects).toHaveLength(2);
    act(() => button("撤销").click());
    expect((maskCanvasProps.mock.lastCall![0] as MaskCanvasProps).payload.objects).toHaveLength(1);
    expect(container.querySelector('[data-testid="mask-canvas"]')?.getAttribute("data-selected-id")).toBe("obj-1");
    act(() => button("重做").click());
    await act(async () => button("导出 LabelMe JSON").click());
    const exported = vi.mocked(workflowApi.exportLabelMe).mock.calls[0][2];
    expect(exported.objects[0]).toEqual(original);
    expect(exported.objects[1]).toMatchObject({ object_id: "drawn-object", class_id: "class-1", source: "manual", mask: manualMask() });
    expect(exported.issues).toEqual(issues);
    await act(async () => button("保存复核草稿").click());
    const stored = vi.mocked(workflowApi.saveDraft).mock.calls[0][2];
    expect(stored.objects).toHaveLength(2);
    expect(stored.objects[0].mask_ref).toBe("masks/original.json");
    expect(stored.objects[1]).toMatchObject({ object_id: "drawn-object", mask_ref: "masks/manual.json", mask: manualMask() });
    expect(stored.annotations).toEqual(expect.arrayContaining([
      expect.objectContaining({ object_id: "drawn-object", type: "polygon", mask_ref: "masks/manual.json" }),
    ]));
    expect(stored.issues).toEqual(issues);
    expect(visionApi.saveMask).toHaveBeenCalledTimes(1);
    expect(button("保存复核草稿").disabled).toBe(true);
  });

  it("uploads an edited mask instead of restoring its old managed reference", async () => {
    const original = { ...payload.objects[0], type: "polygon" as const, mask: manualMask(), mask_ref: "masks/original.json" };
    vi.mocked(workflowApi.review).mockResolvedValue({ ...review, payload: { ...payload, objects: [original] } });
    await render(segmentProfile);
    const canvas = maskCanvasProps.mock.lastCall![0] as MaskCanvasProps;
    const pixels = new Uint8Array(900);
    pixels[20 * 30 + 20] = 1;
    const edited = encodeMask(pixels, 30, 30);
    act(() => canvas.onChange({ ...canvas.payload, objects: [{ ...original, mask: edited, mask_ref: undefined, bbox: undefined }] }));
    await act(async () => button("保存复核草稿").click());
    expect(visionApi.saveMask).toHaveBeenCalledWith(edited);
    expect(vi.mocked(workflowApi.saveDraft).mock.calls[0][2].objects[0]).toMatchObject({
      object_id: "obj-1", mask: edited, mask_ref: "masks/manual.json", provenance_json: { backend: "managed" },
    });
  });

  it("refines the selected object's class and retains its original provenance", async () => {
    const original = {
      ...payload.objects[0], type: "polygon" as const, mask: manualMask(), mask_ref: "masks/original.json",
      provenance: { proposer: "vision-llm", evidence: "original proposal" },
    };
    vi.mocked(workflowApi.review).mockResolvedValue({ ...review, payload: { ...payload, objects: [original] } });
    vi.mocked(visionApi.sam).mockResolvedValue({
      image_id: "image-1", mask_ref: "masks/sam.json", backend_used: "sam2", score: 0.9, device: "cpu",
    } as never);
    await render(segmentProfile);
    const drawingClass = container.querySelector<HTMLSelectElement>("select")!;
    act(() => {
      drawingClass.value = "class-2";
      drawingClass.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => button("测试 SAM 修边").click());
    expect(visionApi.sam).toHaveBeenCalledWith(expect.objectContaining({ class_id: "class-1" }));
    const refined = (maskCanvasProps.mock.lastCall![0] as MaskCanvasProps).payload.objects[0];
    expect(refined).toMatchObject({
      object_id: "obj-1", class_id: "class-1", source: "sam", mask_ref: "masks/sam.json",
      provenance_json: { backend: "managed" },
      provenance: { proposer: "vision-llm", evidence: "original proposal", sam_refinement: { backend: "sam2" } },
    });
    expect(refined.bbox).toBeUndefined();
    expect(refined.polygon).toBeUndefined();
  });

  it("requires mapping an existing candidate before asking SAM to refine it", async () => {
    vi.mocked(workflowApi.review).mockResolvedValue({
      ...review,
      payload: { ...payload, objects: [{ ...payload.objects[0], class_id: null }] },
    });
    await render(segmentProfile);
    await act(async () => button("测试 SAM 修边").click());
    expect(visionApi.sam).not.toHaveBeenCalled();
    expect(container.textContent).toContain("请先为当前对象选择本次任务的类别");
  });

  it("shows the run project folders instead of global export settings", async () => {
    vi.mocked(workflowApi.review).mockResolvedValue({
      ...structuredClone(review),
      project: {
        id: "project-1",
        name: "塔体项目",
        root_path: "D:/塔体项目",
        destination_id: "destination-1",
        created_at: "2026-11-10T00:00:00Z",
      },
      export_locations: {
        json_path: "D:/塔体项目/workflow-run-1/annotations/json",
        images_path: "D:/塔体项目/workflow-run-1/annotations/images",
      },
    });
    await render();
    expect(container.textContent).toContain("本次运行的项目保存位置");
    expect(container.textContent).toContain("D:/塔体项目/workflow-run-1/annotations/json");
    expect(container.textContent).toContain("D:/塔体项目/workflow-run-1/annotations/images");
    expect(container.textContent).not.toContain("设置 JSON 和预览图保存位置");
  });

  it("rejects only unmapped candidates as an undoable local edit", async () => {
    vi.mocked(workflowApi.review).mockResolvedValue({
      ...structuredClone(review),
      payload: {
        ...structuredClone(payload),
        objects: [
          payload.objects[0],
          { ...payload.objects[0], object_id: "unknown", class_id: null },
          {
            ...payload.objects[0],
            object_id: "outside",
            class_id: "unconfigured",
          },
          {
            ...payload.objects[0],
            object_id: "old-rejected",
            class_id: null,
            decision: "rejected",
          },
        ],
      },
    });
    await render();
    act(() => button("拒绝这 2 个未归类候选").click());
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
    expect(workflowApi.decide).not.toHaveBeenCalled();
    expect(container.textContent).toContain(
      "已在当前草稿中拒绝 2 个未归类候选",
    );
    act(() => button("撤销").click());
    expect(button("拒绝这 2 个未归类候选").disabled).toBe(false);
    expect(
      container.querySelectorAll(".workflow-object")[1].textContent,
    ).not.toContain("已拒绝");
    act(() => button("拒绝这 2 个未归类候选").click());
    await act(async () => button("保存复核草稿").click());
    const stored = vi.mocked(workflowApi.saveDraft).mock.calls[0][2];
    expect(stored.annotations).toEqual([
      expect.objectContaining({ object_id: "obj-1", class_id: "class-1" }),
    ]);
    expect(
      stored.objects
        .filter((object) => object.decision === "rejected")
        .map((object) => object.object_id),
    ).toEqual(["unknown", "outside", "old-rejected"]);
    expect(workflowApi.decide).not.toHaveBeenCalled();
  });
  it("applies a user-written common conclusion only to unresolved semantic issues", async () => {
    vi.mocked(workflowApi.review).mockResolvedValue({
      ...structuredClone(review),
      payload: {
        ...structuredClone(payload),
        issues: [
          {
            id: "blocker",
            type: "unmapped_class",
            blocking: true,
            message: "类别待确定",
          },
          { id: "one", type: "boundary", message: "边界需要检查" },
          { id: "two", type: "uncertain", message: "对象需要判断" },
          {
            id: "error",
            type: "invalid_geometry",
            severity: "error",
            message: "格式需要修正",
          },
          {
            id: "done",
            type: "wrong_class",
            message: "类别检查完成",
            resolution: "resolved",
            reason: "原有核查理由",
          },
        ],
      },
    });
    await render();
    const form = container.querySelector<HTMLFormElement>("form")!;
    const resolution = form.querySelector<HTMLSelectElement>(
      '[name="resolution"]',
    )!;
    const explanation = form.querySelector<HTMLTextAreaElement>(
      '[name="explanation"]',
    )!;
    expect(container.textContent).toContain("批量填写检查结论（2 项）");
    resolution.value = "dismissed";
    act(() =>
      form.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true }),
      ),
    );
    expect(container.textContent).toContain("填写你核对后的理由");
    expect(button("撤销").disabled).toBe(true);
    explanation.value = "我已逐项对照原图，接受当前可见轮廓。";
    act(() =>
      form.dispatchEvent(
        new Event("submit", { bubbles: true, cancelable: true }),
      ),
    );
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
    expect(workflowApi.decide).not.toHaveBeenCalled();
    expect(button("确认整图并应用").disabled).toBe(true);
    await act(async () => button("保存复核草稿").click());
    const issues = vi.mocked(workflowApi.saveDraft).mock.calls[0][2].issues;
    expect(
      issues.filter((issue) => ["one", "two"].includes(issue.id!)),
    ).toEqual([
      expect.objectContaining({
        id: "one",
        resolution: "dismissed",
        reason: "我已逐项对照原图，接受当前可见轮廓。",
      }),
      expect.objectContaining({
        id: "two",
        resolution: "dismissed",
        reason: "我已逐项对照原图，接受当前可见轮廓。",
      }),
    ]);
    expect(issues[0].resolution).toBeUndefined();
    expect(issues[3].resolution).toBeUndefined();
    expect(issues[4].reason).toBe("原有核查理由");
    expect(workflowApi.decide).not.toHaveBeenCalled();
  });
  it("exports the current LabelMe draft despite blockers without saving or confirming it", async () => {
    vi.mocked(workflowApi.review).mockResolvedValue({
      ...structuredClone(review),
      payload: {
        ...structuredClone(payload),
        issues: [
          {
            id: "blocker",
            object_id: "obj-1",
            type: "invalid_geometry",
            blocking: true,
            message: "需要修正",
          },
        ],
      },
    });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    await render();
    act(() => button("修改框和类别").click());
    expect(button("确认整图并应用").disabled).toBe(true);
    expect(button("导出 LabelMe JSON").disabled).toBe(false);
    expect(button("直接导出 LabelMe JSON 草稿").disabled).toBe(false);
    await act(async () => button("直接导出 LabelMe JSON 草稿").click());
    const [id, revision, exported] = vi.mocked(workflowApi.exportLabelMe).mock
      .calls[0];
    expect([id, revision]).toEqual(["review-1", 1]);
    expect(exported.objects[0]).toMatchObject({
      class_id: "class-2",
      bbox: { x: 4, y: 5, width: 8, height: 9 },
    });
    expect(exported.issues[0].blocking).toBe(true);
    expect(container.textContent).toContain(savedLabelMe.saved_path);
    expect(container.textContent).toContain("草稿导出");
    expect(container.textContent).toContain("未保存修改");
    const link = Array.from(
      container.querySelectorAll<HTMLAnchorElement>("a"),
    ).find((item) => item.textContent === "下载 JSON")!;
    expect(link.href).toContain(savedLabelMe.download_url);
    expect(link.target).toBe("_blank");
    expect(link.rel).toBe("noopener noreferrer");
    expect(click).not.toHaveBeenCalled();
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
    expect(workflowApi.decide).not.toHaveBeenCalled();
  });
  it("explains coordinate and unmapped-class checks in Chinese", async () => {
    vi.mocked(workflowApi.review).mockResolvedValue({
      ...structuredClone(review),
      payload: {
        ...structuredClone(payload),
        issues: [
          {
            id: "coordinate",
            type: "coordinate_error",
            message: "坐标需要核对",
            coordinate_details: { sent_size: [1024, 870] },
          },
          {
            id: "class",
            type: "invalid_geometry",
            object_id: "obj-1",
            blocking: true,
            message: "Candidate class is not mapped to the task profile.",
          },
          { id: "boundary", type: "boundary", message: "请核对轮廓边缘" },
        ],
      },
    });
    await render();
    expect(container.textContent).toContain("坐标需要复核");
    expect(container.textContent).toContain("查看坐标诊断信息");
    expect(container.textContent).toContain("类别待确定");
    expect(container.textContent).toContain(
      "请选择已配置类别，或拒绝该候选后保存草稿",
    );
    expect(container.textContent).toContain("边界需要复核");
    expect(container.textContent).toContain("对象 #1 — 类别尚未确定");
    expect(container.textContent).toContain("系统会重新检查");
    expect(container.textContent).not.toContain("不可豁免");
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
    expect(workflowApi.decide).not.toHaveBeenCalled();
  });
  it("saves the current unsaved preview and provides a download link without navigating or confirming annotations", async () => {
    const image = new Blob(["png"], { type: "image/png" });
    vi.mocked(annotationPreview).mockResolvedValue(image);
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    await render();
    expect(button("下载本图 COCO 标注").disabled).toBe(true);
    act(() => button("修改框和类别").click());
    await act(async () => button("保存标注预览 PNG").click());
    const downloaded = vi.mocked(annotationPreview).mock.calls[0][0];
    expect(downloaded.objects[0].class_id).toBe("class-2");
    expect(click).not.toHaveBeenCalled();
    expect(workflowApi.savePreview).toHaveBeenCalledWith(
      "review-1",
      1,
      image,
      true,
    );
    expect(container.textContent).toContain(savedPreview.saved_path);
    expect(container.textContent).toContain("预览 PNG 已保存");
    const link = container.querySelector<HTMLAnchorElement>("a[download]")!;
    expect(link.href).toContain(savedPreview.download_url);
    expect(link.download).toBe(savedPreview.file_name);
    expect(link.target).toBe("_blank");
    expect(link.rel).toBe("noopener noreferrer");
    expect(container.textContent).toContain("未保存修改");
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
    expect(workflowApi.decide).not.toHaveBeenCalled();
  });
  it("does not save a PNG when the draft changes while the image is being rendered", async () => {
    let finishRendering!: (image: Blob) => void;
    vi.mocked(annotationPreview).mockImplementation(
      () =>
        new Promise((resolve) => {
          finishRendering = resolve;
        }),
    );
    await render();
    act(() => button("保存标注预览 PNG").click());
    act(() => button("修改框和类别").click());
    await act(async () =>
      finishRendering(new Blob(["png"], { type: "image/png" })),
    );
    expect(workflowApi.savePreview).not.toHaveBeenCalled();
    expect(container.textContent).toContain("未保存修改");
  });
  it("keeps edits and reports the older saved preview when the draft changes during upload", async () => {
    vi.mocked(annotationPreview).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    let finishSaving!: (saved: typeof savedPreview) => void;
    vi.mocked(workflowApi.savePreview).mockImplementation(
      () =>
        new Promise((resolve) => {
          finishSaving = resolve;
        }),
    );
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    await render();
    await act(async () => button("保存标注预览 PNG").click());
    act(() => button("修改框和类别").click());
    await act(async () => finishSaving(savedPreview));
    expect(container.textContent).toContain("此前版本的预览文件已保存");
    expect(container.textContent).toContain(savedPreview.saved_path);
    expect(container.textContent).toContain("未保存修改");
    expect(click).not.toHaveBeenCalled();
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
    expect(workflowApi.decide).not.toHaveBeenCalled();
  });
  it("shows an unmapped candidate without selecting the first class", async () => {
    vi.mocked(workflowApi.review).mockResolvedValue({
      ...structuredClone(review),
      payload: {
        ...structuredClone(payload),
        objects: [{ ...payload.objects[0], class_id: null }],
      },
    });
    await render();
    const classSelect = container.querySelector<HTMLSelectElement>(
      '[aria-label="对象 1 类别"]',
    )!;
    expect(classSelect.value).toBe("");
    expect(classSelect.selectedOptions[0].textContent).toBe("待确定类别");
    expect(workflowApi.saveDraft).not.toHaveBeenCalled();
    act(() => {
      classSelect.value = "class-2";
      classSelect.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await act(async () => button("保存复核草稿").click());
    expect(
      vi.mocked(workflowApi.saveDraft).mock.calls[0][2].objects[0].class_id,
    ).toBe("class-2");
  });
  it("saves edits as a draft, then confirms the exact saved revision", async () => {
    await render();
    act(() => button("修改框和类别").click());
    await act(async () => button("保存复核草稿").click());
    expect(workflowApi.decide).not.toHaveBeenCalled();
    const stored = vi.mocked(workflowApi.saveDraft).mock.calls[0][2];
    expect(stored.annotations).toEqual([
      expect.objectContaining({
        class_id: "class-2",
        bbox: { x: 4, y: 5, width: 8, height: 9 },
        provenance_json: { backend: "managed" },
      }),
    ]);
    act(() =>
      container
        .querySelector<HTMLInputElement>('input[type="checkbox"]')!
        .click(),
    );
    await act(async () => button("确认整图并应用").click());
    expect(workflowApi.decide).toHaveBeenCalledWith(
      "review-1",
      2,
      "approved",
      "",
      {},
    );
    expect(container.textContent).toContain("已提交人工决定");
    const cocoLink = Array.from(container.querySelectorAll("a")).find(
      (link) => link.textContent === "下载本图 COCO 标注",
    );
    expect(cocoLink?.href).toContain(
      "/api/datasets/dataset-1/export/coco?image_ids=image-1",
    );
  });
  it("retains rejected objects for review history while excluding them from formal labels", async () => {
    await render();
    act(() => button("拒绝对象").click());
    await act(async () => button("保存复核草稿").click());
    const stored = vi.mocked(workflowApi.saveDraft).mock.calls[0][2];
    expect(stored.annotations).toEqual([]);
    expect(stored.objects[0].decision).toBe("rejected");
    expect(button("确认整图并应用").disabled).toBe(true);
  });
  it("keeps local edits when a save conflicts", async () => {
    vi.mocked(workflowApi.saveDraft).mockRejectedValue(new Error("版本冲突"));
    await render();
    act(() => button("修改框和类别").click());
    await act(async () => button("保存复核草稿").click());
    expect(container.textContent).toContain("版本冲突");
    expect(container.textContent).toContain("未保存修改");
    expect(workflowApi.decide).not.toHaveBeenCalled();
  });
});
