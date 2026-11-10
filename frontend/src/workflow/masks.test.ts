import { describe, expect, it } from "vitest";
import { decodeMask, encodeMask, expandCounts, paintMask } from "./masks";

describe("canonical mask editing", () => {
  it("decodes compressed COCO counts from pycocotools in column-major order", () => {
    expect(expandCounts("0321")).toEqual([0, 3, 2, 4]);
    expect(
      Array.from(decodeMask({ size: [3, 3], counts: "0321" }, 3, 3)),
    ).toEqual([1, 0, 1, 1, 0, 1, 1, 1, 1]);
  });
  it("preserves a hole and disconnected components across round trips", () => {
    const pixels = new Uint8Array(7 * 7);
    for (let y = 1; y <= 4; y++)
      for (let x = 1; x <= 4; x++) pixels[y * 7 + x] = 1;
    pixels[2 * 7 + 2] = 0;
    pixels[6 * 7 + 6] = 1;
    expect(decodeMask(encodeMask(pixels, 7, 7), 7, 7)).toEqual(pixels);
  });
  it("paints and erases within image bounds", () => {
    const pixels = new Uint8Array(25);
    paintMask(pixels, 5, 5, 0, 0, 1, 1);
    expect(Array.from(pixels).reduce((sum, value) => sum + value, 0)).toBe(3);
    paintMask(pixels, 5, 5, 0, 0, 1, 0);
    expect(pixels.every((value) => value === 0)).toBe(true);
  });
  it("rejects incompatible dimensions and truncated RLE", () => {
    expect(() => decodeMask({ size: [4, 4], counts: [16] }, 5, 5)).toThrow(
      "尺寸",
    );
    expect(() => decodeMask({ size: [4, 4], counts: [15] }, 4, 4)).toThrow(
      "覆盖",
    );
    expect(() => decodeMask({ size: [4, 4], counts: [-1, 17] }, 4, 4)).toThrow(
      "长度",
    );
  });
});
