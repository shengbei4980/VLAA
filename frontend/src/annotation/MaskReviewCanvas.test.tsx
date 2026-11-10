// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MaskReviewCanvas } from "./MaskReviewCanvas";
import type { MaskRle, ReviewPayload } from "../types/workflows";
import { decodeMask, encodeMask } from "../workflow/masks";

const rasterize = vi.hoisted(() => vi.fn());
vi.mock("../workflow/masks", async (original) => ({
  ...(await original<typeof import("../workflow/masks")>()),
  polygonMask: rasterize,
}));

let root: Root;
let container: HTMLDivElement;
let payload: ReviewPayload;
let region: MaskRle;
let onChange: ReturnType<typeof vi.fn>;
let onSelect: ReturnType<typeof vi.fn>;
let onError: ReturnType<typeof vi.fn>;
let onDrawingChange: ReturnType<typeof vi.fn>;
const classes = [{ class_id: "person", name: "人物", color: "#22c55e" }];
const button = (text: string) =>
  Array.from(container.querySelectorAll<HTMLButtonElement>("button")).find(
    (item) => item.textContent === text,
  )!;
const canvas = () =>
  container.querySelector<HTMLCanvasElement>(
    'canvas[aria-label="掩膜复核画布"]',
  )!;
const pointer = (type: string, x: number, y: number, detail = 1) => {
  act(() =>
    canvas().dispatchEvent(
      new MouseEvent(type, {
        bubbles: true,
        button: 0,
        clientX: 10 + x * 2,
        clientY: 20 + y * 2,
        detail,
      }),
    ),
  );
};
const point = (x: number, y: number, detail = 1) => {
  pointer("pointerdown", x, y, detail);
  pointer("pointerup", x, y, detail);
};
const render = async (selectedId = "first", newObjectRequest = 0) => {
  await act(async () =>
    root.render(
      <MaskReviewCanvas
        payload={payload}
        classes={classes}
        selectedId={selectedId}
        activeClassId="person"
        newObjectRequest={newObjectRequest}
        onSelect={onSelect}
        onChange={onChange}
        onError={onError}
        onDrawingChange={onDrawingChange}
        onSamBox={vi.fn()}
        onSamPoints={vi.fn()}
      />,
    ),
  );
};

beforeEach(() => {
  const pixels = new Uint8Array(10000);
  pixels[40 * 100 + 40] = 1;
  pixels[80 * 100 + 80] = 1;
  const area = new Uint8Array(10000);
  area[20 * 100 + 20] = 1;
  area[40 * 100 + 40] = 1;
  region = encodeMask(area, 100, 100);
  rasterize.mockReset().mockImplementation(() => region);
  payload = {
    image_id: "image",
    dataset_id: "dataset",
    image_url: "/image.png",
    width: 100,
    height: 100,
    base_revision: 0,
    objects: [
      {
        object_id: "first",
        class_id: "person",
        type: "polygon",
        mask: encodeMask(pixels, 100, 100),
        mask_ref: "masks/old.json",
        source: "auto",
        provenance_json: { model: "vision" },
      },
    ],
    issues: [{ id: "issue", type: "boundary", message: "检查边界" }],
    roi: encodeMask(new Uint8Array(10000).fill(1), 100, 100),
  };
  onChange = vi.fn();
  onSelect = vi.fn();
  onError = vi.fn();
  onDrawingChange = vi.fn();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(
    () =>
      ({
        clearRect: vi.fn(),
        drawImage: vi.fn(),
        putImageData: vi.fn(),
        createImageData: (width: number, height: number) => ({
          data: new Uint8ClampedArray(width * height * 4),
        }),
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        lineTo: vi.fn(),
        closePath: vi.fn(),
        stroke: vi.fn(),
        arc: vi.fn(),
        fill: vi.fn(),
        strokeRect: vi.fn(),
        strokeText: vi.fn(),
        fillText: vi.fn(),
      }) as unknown as CanvasRenderingContext2D,
  );
  vi.spyOn(
    HTMLCanvasElement.prototype,
    "getBoundingClientRect",
  ).mockReturnValue({
    left: 10,
    top: 20,
    width: 200,
    height: 200,
    right: 210,
    bottom: 220,
    x: 10,
    y: 20,
    toJSON: () => ({}),
  });
  HTMLCanvasElement.prototype.setPointerCapture = vi.fn();
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});
(
  globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

describe("manual mask review gestures", () => {
  it("creates an object only after completing original-pixel polygon coordinates", async () => {
    await render("first", 1);
    expect(button("多边形").getAttribute("aria-pressed")).toBe("true");
    point(10, 10);
    point(70, 10);
    point(70, 70);
    expect(onChange).not.toHaveBeenCalled();
    expect(onDrawingChange).toHaveBeenLastCalledWith(true);
    act(() => button("完成轮廓").click());
    const next: ReviewPayload = onChange.mock.calls[0][0];
    expect(next.objects).toHaveLength(2);
    expect(next.objects[1]).toMatchObject({
      class_id: "person",
      type: "polygon",
      source: "manual",
      mask: region,
      polygon: [
        [10, 10],
        [70, 10],
        [70, 70],
      ],
    });
    expect(next.objects[0]).toBe(payload.objects[0]);
    expect(next.issues).toBe(payload.issues);
    expect(next.roi).toBe(payload.roi);
    expect(onDrawingChange).toHaveBeenLastCalledWith(false);
  });

  it("closes a double click at the first vertex once without leaving a new unfinished outline", async () => {
    await render("first", 1);
    point(10, 10);
    point(70, 10);
    point(70, 70);
    point(10, 10, 1);
    point(10, 10, 2);
    pointer("dblclick", 10, 10, 2);
    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onDrawingChange).toHaveBeenLastCalledWith(false);
    expect(button("完成轮廓").disabled).toBe(true);
    expect(onError.mock.calls.filter(([message]) => message)).toEqual([]);
  });

  it.each(["add", "subtract"] as const)(
    "applies polygon %s to one mask and removes its old geometry reference",
    async (mode) => {
      await render();
      act(() => button("多边形").click());
      const selector = container.querySelectorAll("select")[1];
      act(() => {
        selector.value = mode;
        selector.dispatchEvent(new Event("change", { bubbles: true }));
      });
      point(10, 10);
      point(70, 10);
      point(70, 70);
      act(() => button("完成轮廓").click());
      const next: ReviewPayload = onChange.mock.calls[0][0];
      const object = next.objects[0];
      expect(next.objects).toHaveLength(1);
      expect(object.object_id).toBe("first");
      expect(object.mask_ref).toBeUndefined();
      expect(object.polygon).toBeUndefined();
      expect(object.provenance_json).toEqual({ model: "vision" });
      const pixels = decodeMask(object.mask!, 100, 100);
      expect(pixels[80 * 100 + 80]).toBe(1);
      expect(pixels[20 * 100 + 20]).toBe(mode === "add" ? 1 : 0);
      expect(pixels[40 * 100 + 40]).toBe(mode === "add" ? 1 : 0);
      expect(pixels[50 * 100 + 50]).toBe(0);
    },
  );

  it("turns a legacy box into a manual segmentation when painted", async () => {
    payload.objects[0] = {
      ...payload.objects[0],
      type: "bbox",
      mask: undefined,
      mask_ref: undefined,
      bbox: { x: 5, y: 5, width: 10, height: 10 },
    };
    await render();
    act(() => button("画笔").click());
    point(10, 10);
    expect(onChange.mock.calls[0][0].objects[0]).toMatchObject({
      object_id: "first",
      type: "polygon",
      source: "manual",
    });
    expect(onChange.mock.calls[0][0].objects[0].bbox).toBeUndefined();
  });

  it("discards a stroke if its selected target changes before pointer up", async () => {
    payload.objects.push({ ...payload.objects[0], object_id: "second" });
    await render();
    act(() => button("画笔").click());
    pointer("pointerdown", 10, 10);
    await render("second");
    pointer("pointerup", 20, 20);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("ignores rejected and hidden targets during selection", async () => {
    payload.objects.push({
      ...payload.objects[0],
      object_id: "rejected",
      decision: "rejected",
    });
    await render();
    point(40, 40);
    expect(onSelect).toHaveBeenLastCalledWith("first");
    onSelect.mockClear();
    const classCheckbox = Array.from(
      container.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'),
    ).find((input) => input.closest("label")?.textContent === "人物")!;
    act(() => classCheckbox.click());
    point(40, 40);
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("edits vector vertices but keeps native pixel masks on pixel-edit tools", async () => {
    await render();
    expect(button("编辑顶点").disabled).toBe(true);
    payload.objects[0] = {
      ...payload.objects[0],
      polygon: [
        [10, 10],
        [70, 10],
        [70, 70],
        [10, 70],
      ],
      mask: region,
    };
    await render();
    expect(button("编辑顶点").disabled).toBe(false);
    act(() => button("编辑顶点").click());
    pointer("pointerdown", 10, 10);
    pointer("pointermove", 15, 15);
    pointer("pointerup", 15, 15);
    expect(onChange.mock.calls[0][0].objects[0].polygon[0]).toEqual([15, 15]);
    expect(onChange.mock.calls[0][0].objects[0].mask_ref).toBeUndefined();
    onChange.mockClear();
    pointer("dblclick", 40, 10, 2);
    expect(onChange.mock.calls[0][0].objects[0].polygon).toHaveLength(5);
  });
});
