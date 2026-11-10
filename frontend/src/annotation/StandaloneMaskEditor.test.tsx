// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { StandaloneMaskEditor } from "./StandaloneMaskEditor";
import { visionApi } from "../api/vision";
import type { ImageItem } from "../types";
import type { ReviewPayload } from "../types/workflows";
const { maskCanvasProps } = vi.hoisted(() => ({ maskCanvasProps: vi.fn() }));
vi.mock("../api/vision", () => ({ visionApi: { mask: vi.fn() } }));
vi.mock("./MaskReviewCanvas", () => ({
  MaskReviewCanvas: ({
    payload,
    onChange,
    selectedId,
    activeClassId,
    newObjectRequest,
    onDrawingChange,
  }: {
    payload: ReviewPayload;
    onChange: (payload: ReviewPayload) => void;
    selectedId: string;
    activeClassId?: string;
    newObjectRequest?: number;
    onDrawingChange?: (drawing: boolean) => void;
  }) => {
    maskCanvasProps({ payload, selectedId, activeClassId, newObjectRequest, onDrawingChange });
    return (
    <div data-testid="mask-count" data-selected-id={selectedId}>
      {payload.objects.filter((object) => object.mask).length} 个权威掩膜
      <button onClick={() => onChange({ ...payload })}>编辑已有掩膜</button>
      <button onClick={() => onDrawingChange?.(true)}>开始测试轮廓</button>
      <button onClick={() => onDrawingChange?.(false)}>取消测试轮廓</button>
    </div>
    );
  },
}));
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
describe("ordinary segmentation review", () => {
  it("starts a new polygon without creating an empty annotation and protects in-progress drawing", async () => {
    vi.clearAllMocks();
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const onChange = vi.fn();
    const onDrawingChange = vi.fn();
    await act(async () => root.render(
      <StandaloneMaskEditor
        image={{ id: "image", dataset_id: "dataset", file_name: "image.png", width: 3, height: 3,
          split: "train", status: "annotated", file_url: "/image", annotation_revision: 1,
          created_at: "", updated_at: "" }}
        classes={[]}
        drafts={[{ id: "existing", object_id: "existing", class_id: "class", type: "polygon", source: "auto",
          mask: { size: [3, 3], counts: [0, 1, 8] } }]}
        activeClassId="class" selectedId="existing" onSelect={vi.fn()} onChange={onChange}
        onRegionsChange={vi.fn()} onSamBox={vi.fn()} onSamPoints={vi.fn()} busy={false}
        onDrawingChange={onDrawingChange}
      />,
    ));
    const button = (label: string) => Array.from(container.querySelectorAll<HTMLButtonElement>("button"))
      .find((item) => item.textContent === label)!;
    act(() => button("新增分割对象").click());
    expect(maskCanvasProps.mock.lastCall![0]).toMatchObject({ newObjectRequest: 1, activeClassId: "class" });
    expect(maskCanvasProps.mock.lastCall![0].payload.objects).toHaveLength(1);
    expect(onChange).not.toHaveBeenCalled();
    act(() => button("编辑已有掩膜").click());
    expect(button("撤销掩膜编辑").disabled).toBe(false);
    act(() => button("开始测试轮廓").click());
    expect(onDrawingChange).toHaveBeenLastCalledWith(true);
    expect(button("新增分割对象").disabled).toBe(true);
    expect(button("撤销掩膜编辑").disabled).toBe(true);
    expect(button("重做掩膜编辑").disabled).toBe(true);
    expect(container.textContent).toContain("请先完成轮廓或取消本次绘制");
    act(() => button("取消测试轮廓").click());
    expect(onDrawingChange).toHaveBeenLastCalledWith(false);
    expect(button("撤销掩膜编辑").disabled).toBe(false);
    act(() => button("开始测试轮廓").click());
    expect(onDrawingChange).toHaveBeenLastCalledWith(true);
    act(() => root.unmount());
    expect(onDrawingChange).toHaveBeenLastCalledWith(false);
    container.remove();
  });
  it("renders a mask-only COCO annotation by loading its canonical RLE", async () => {
    vi.mocked(visionApi.mask).mockResolvedValue({
      mask_ref: "masks/hash.json",
      width: 3,
      height: 3,
      area: 7,
      rle: { size: [3, 3], counts: "0321" },
    });
    const image: ImageItem = {
      id: "image",
      dataset_id: "dataset",
      file_name: "image.png",
      width: 3,
      height: 3,
      split: "train",
      status: "annotated",
      file_url: "/image",
      annotation_revision: 1,
      created_at: "",
      updated_at: "",
    };
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    const onChange = vi.fn();
    await act(async () =>
      root.render(
        <StandaloneMaskEditor
          image={image}
          classes={[]}
          drafts={[
            {
              id: "annotation-row",
              object_id: "object-stable",
              class_id: "class",
              type: "polygon",
              source: "imported",
              mask_ref: "masks/hash.json",
            },
          ]}
          activeClassId="class"
          selectedId="annotation-row"
          onSelect={vi.fn()}
          onChange={onChange}
          onRegionsChange={vi.fn()}
          onSamBox={vi.fn()}
          onSamPoints={vi.fn()}
          busy={false}
        />,
      ),
    );
    expect(container.textContent).toContain("1 个权威掩膜");
    expect(visionApi.mask).toHaveBeenCalledWith("masks/hash.json");
    expect(
      container
        .querySelector('[data-testid="mask-count"]')
        ?.getAttribute("data-selected-id"),
    ).toBe("object-stable");
    act(() =>
      Array.from(container.querySelectorAll("button"))
        .find((button) => button.textContent === "编辑已有掩膜")!
        .click(),
    );
    expect(onChange).toHaveBeenCalledWith([
      expect.objectContaining({
        id: "annotation-row",
        object_id: "object-stable",
      }),
    ]);
    act(() => root.unmount());
    container.remove();
  });
});
