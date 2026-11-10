// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { annotationPreview } from "./annotation-preview";
import { api } from "../api/client";
import { encodeMask } from "./masks";
import type { ReviewPayload } from "../types/workflows";

vi.mock("../api/client", () => ({ api: { imageBlob: vi.fn() } }));

function drawingContext() {
  return {
    drawImage: vi.fn(),
    strokeRect: vi.fn(),
    fillRect: vi.fn(),
    fillText: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    closePath: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    measureText: vi.fn(() => ({ width: 35 })),
    createImageData: vi.fn((width: number, height: number) => ({
      data: new Uint8ClampedArray(width * height * 4),
    })),
    putImageData: vi.fn(),
  };
}

const contexts: ReturnType<typeof drawingContext>[] = [];
const imageSources: string[] = [];
const revokeSource = vi.fn();
let output: { width: number; height: number; type: string } | undefined;
const payload: ReviewPayload = {
  image_id: "image",
  dataset_id: "dataset",
  image_url: "/api/images/image/file",
  width: 64,
  height: 40,
  base_revision: 1,
  objects: [],
  issues: [],
};
const classes = [{ class_id: "people", name: "人物", color: "#22c55e" }];

beforeEach(() => {
  vi.clearAllMocks();
  contexts.length = 0;
  imageSources.length = 0;
  output = undefined;
  vi.mocked(api.imageBlob).mockResolvedValue(
    new Blob(["original-image"], { type: "image/png" }),
  );
  const originalUrl = URL;
  vi.stubGlobal(
    "URL",
    class extends originalUrl {
      static createObjectURL = vi.fn(() => "blob:source-image");
      static revokeObjectURL = revokeSource;
    },
  );
  vi.stubGlobal(
    "Image",
    class {
      naturalWidth = 64;
      naturalHeight = 40;
      onload?: () => void;
      onerror?: () => void;
      set src(value: string) {
        imageSources.push(value);
        queueMicrotask(() =>
          value.includes("unavailable") ? this.onerror?.() : this.onload?.(),
        );
      }
    },
  );
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(
    function () {
      const context = drawingContext();
      contexts.push(context);
      return context as unknown as CanvasRenderingContext2D;
    },
  );
  vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (
    this: HTMLCanvasElement,
    callback,
    type,
  ) {
    output = { width: this.width, height: this.height, type: type ?? "" };
    callback(new Blob(["encoded-preview"], { type }));
  });
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("exports original-size PNG with mask, polygon, box, OBB and class labels while excluding rejected objects", async () => {
  const pixels = new Uint8Array(64 * 40);
  pixels[70] = 1;
  const blob = await annotationPreview(
    {
      ...payload,
      objects: [
        {
          object_id: "box",
          class_id: "people",
          type: "bbox",
          bbox: { x: 2, y: 3, width: 6, height: 8 },
        },
        {
          object_id: "mask",
          class_id: "people",
          type: "polygon",
          mask: encodeMask(pixels, 64, 40),
        },
        {
          object_id: "polygon",
          class_id: "people",
          type: "polygon",
          polygon: [
            [10, 10],
            [20, 10],
            [15, 20],
          ],
        },
        {
          object_id: "obb",
          class_id: "people",
          type: "obb",
          obb: { cx: 30, cy: 30, width: 10, height: 4, angle: 0 },
        },
        { object_id: "classification", class_id: "people", type: "classify" },
        {
          object_id: "rejected",
          class_id: null,
          type: "bbox",
          decision: "rejected",
          bbox: { x: 99, y: 99, width: 1, height: 1 },
        },
      ],
    },
    classes,
  );
  expect(blob.type).toBe("image/png");
  expect(api.imageBlob).toHaveBeenCalledWith("image");
  expect(imageSources).toEqual(["blob:source-image"]);
  expect(revokeSource).toHaveBeenCalledWith("blob:source-image");
  expect(output).toEqual({ width: 64, height: 40, type: "image/png" });
  expect(contexts[0].strokeRect.mock.calls).toEqual([[2, 3, 6, 8]]);
  expect(contexts[0].moveTo).toHaveBeenCalledWith(25, 28);
  expect(contexts[0].fillText.mock.calls.map((call) => call[0])).toEqual([
    "#1 人物",
    "#2 人物",
    "#3 人物",
    "#4 人物",
    "#5 人物",
  ]);
  expect(
    contexts[1].putImageData.mock.calls[0][0].data.slice(70 * 4, 70 * 4 + 4),
  ).toEqual(new Uint8ClampedArray([34, 197, 94, 102]));
});

it("exports the original image when the reviewed draft contains no objects", async () => {
  await annotationPreview(payload, classes);
  expect(output).toEqual({ width: 64, height: 40, type: "image/png" });
  expect(contexts[0].drawImage).toHaveBeenCalledTimes(1);
  expect(contexts[0].fillText).not.toHaveBeenCalled();
});

it("rejects inconsistent original image and mask dimensions instead of exporting a misleading preview", async () => {
  await expect(
    annotationPreview({ ...payload, width: 65 }, classes),
  ).rejects.toThrow("原图尺寸与当前标注不一致");
  await expect(
    annotationPreview(
      {
        ...payload,
        objects: [
          {
            object_id: "bad-mask",
            class_id: "people",
            type: "polygon",
            mask: { size: [1, 1], counts: [0, 1] },
          },
        ],
      },
      classes,
    ),
  ).rejects.toThrow("掩膜尺寸与原图不一致");
  expect(output).toBeUndefined();
});

it("reports unavailable original images", async () => {
  vi.mocked(api.imageBlob).mockRejectedValue(
    new Error("原图读取失败（HTTP 404）"),
  );
  await expect(
    annotationPreview({ ...payload, image_url: "/unavailable" }, classes),
  ).rejects.toThrow("原图读取失败");
  expect(output).toBeUndefined();
  expect(imageSources).toEqual([]);
});

it("releases the source object URL when image decoding fails", async () => {
  vi.mocked(URL.createObjectURL).mockReturnValue("blob:unavailable");
  await expect(annotationPreview(payload, classes)).rejects.toThrow(
    "原图解码失败",
  );
  expect(revokeSource).toHaveBeenCalledWith("blob:unavailable");
  expect(output).toBeUndefined();
});

it("reports browser security failures when creating the downloadable PNG", async () => {
  vi.mocked(HTMLCanvasElement.prototype.toBlob).mockImplementation(() => {
    throw new DOMException("Canvas is tainted", "SecurityError");
  });
  await expect(annotationPreview(payload, classes)).rejects.toThrow("跨域设置");
});
