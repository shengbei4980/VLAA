import { describe, expect, it } from "vitest";
import { toOriginalPoint } from "../utils/geometry";
import { decodeMask, encodeMask } from "../workflow/masks";
import { combineMask, validatePolygon } from "./mask-editing";

function originalMask() {
  const pixels = new Uint8Array(7 * 7);
  for (let y = 1; y <= 4; y++) {
    for (let x = 1; x <= 4; x++) pixels[y * 7 + x] = 1;
  }
  pixels[2 * 7 + 2] = 0;
  pixels[6 * 7 + 6] = 1;
  return pixels;
}

describe("manual mask completion", () => {
  it("adds only the drawn pixels while retaining holes and separated components", () => {
    const base = encodeMask(originalMask(), 7, 7);
    const snapshot = structuredClone(base);
    const regionPixels = new Uint8Array(49);
    regionPixels[0] = 1;
    regionPixels[1 * 7 + 1] = 1;
    const region = encodeMask(regionPixels, 7, 7);
    const combined = decodeMask(combineMask(base, region, 7, 7, "add"), 7, 7);
    const expected = originalMask();
    expected[0] = 1;
    expect(combined).toEqual(expected);
    expect(base).toEqual(snapshot);
    expect(decodeMask(region, 7, 7)).toEqual(regionPixels);
  });

  it("subtracts a region without rebuilding the rest of the object's contour", () => {
    const base = encodeMask(originalMask(), 7, 7);
    const snapshot = structuredClone(base);
    const regionPixels = new Uint8Array(49);
    regionPixels[3 * 7 + 3] = 1;
    regionPixels[0] = 1;
    const combined = decodeMask(
      combineMask(base, encodeMask(regionPixels, 7, 7), 7, 7, "subtract"),
      7,
      7,
    );
    const expected = originalMask();
    expected[3 * 7 + 3] = 0;
    expect(combined).toEqual(expected);
    expect(base).toEqual(snapshot);
  });

  it("creates or replaces an object from the region and permits an empty subtraction", () => {
    const region = encodeMask(originalMask(), 7, 7);
    expect(
      decodeMask(combineMask(undefined, region, 7, 7, "add"), 7, 7),
    ).toEqual(originalMask());
    expect(
      decodeMask(combineMask(undefined, region, 7, 7, "subtract"), 7, 7),
    ).toEqual(new Uint8Array(49));
    expect(
      decodeMask(combineMask(region, region, 7, 7, "subtract"), 7, 7),
    ).toEqual(new Uint8Array(49));
    const replacement = encodeMask(new Uint8Array(49), 7, 7);
    expect(
      decodeMask(combineMask(region, replacement, 7, 7, "replace"), 7, 7),
    ).toEqual(new Uint8Array(49));
  });

  it("rejects masks that do not match the original image dimensions", () => {
    const base = encodeMask(originalMask(), 7, 7);
    const wrongSize = encodeMask(new Uint8Array(9), 3, 3);
    expect(() => combineMask(base, wrongSize, 7, 7, "add")).toThrow("尺寸");
    expect(() => combineMask(wrongSize, base, 7, 7, "subtract")).toThrow(
      "尺寸",
    );
  });
});

describe("editable polygon validation", () => {
  it("allows boundary vertices, either winding, concave outlines, and tiny valid polygons", () => {
    const rectangle: [number, number][] = [
      [0, 0],
      [8, 0],
      [8, 6],
      [0, 6],
    ];
    expect(() => validatePolygon(rectangle, 8, 6)).not.toThrow();
    expect(() => validatePolygon([...rectangle].reverse(), 8, 6)).not.toThrow();
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [6, 0],
          [3, 2],
          [6, 6],
          [0, 6],
        ],
        8,
        6,
      ),
    ).not.toThrow();
    expect(() =>
      validatePolygon(
        [
          [2000, 1900],
          [2000.000001, 1900],
          [2000, 1900.000001],
        ],
        2301,
        1955,
      ),
    ).not.toThrow();
  });

  it("requires three distinct finite vertices within the original image", () => {
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [2, 2],
        ],
        8,
        6,
      ),
    ).toThrow("至少");
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [2, 2],
          [0, 0],
        ],
        8,
        6,
      ),
    ).toThrow("重复");
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [Infinity, 2],
          [2, 0],
        ],
        8,
        6,
      ),
    ).toThrow("数值");
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [NaN, 2],
          [2, 0],
        ],
        8,
        6,
      ),
    ).toThrow("数值");
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [8.01, 2],
          [2, 0],
        ],
        8,
        6,
      ),
    ).toThrow("范围");
    expect(() =>
      validatePolygon(
        [
          [0, -0.01],
          [2, 2],
          [2, 0],
        ],
        8,
        6,
      ),
    ).toThrow("范围");
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [2, 2],
          [2, 0],
        ],
        0,
        6,
      ),
    ).toThrow("尺寸");
  });

  it("rejects crossed edges, touching nonadjacent edges, and zero-area outlines", () => {
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [6, 6],
          [0, 6],
          [6, 0],
        ],
        8,
        6,
      ),
    ).toThrow("交叉");
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [6, 0],
          [3, 0],
          [6, 6],
          [0, 6],
        ],
        8,
        6,
      ),
    ).toThrow("交叉");
    expect(() =>
      validatePolygon(
        [
          [0, 0],
          [2, 2],
          [4, 4],
        ],
        8,
        6,
      ),
    ).toThrow("面积");
  });

  it("validates canvas points after conversion back into original-image pixels", () => {
    const screenPoints: [number, number][] = [
      [20, 10],
      [70, 10],
      [70, 40],
      [20, 40],
    ];
    const originalPoints = screenPoints.map((point) =>
      toOriginalPoint(point, 100, 50, 2301, 1955),
    );
    const expectedPoints = [
      [460.2, 391],
      [1610.7, 391],
      [1610.7, 1564],
      [460.2, 1564],
    ];
    originalPoints.forEach(([x, y], index) => {
      expect(x).toBeCloseTo(expectedPoints[index][0]);
      expect(y).toBeCloseTo(expectedPoints[index][1]);
    });
    expect(() => validatePolygon(originalPoints, 2301, 1955)).not.toThrow();
  });
});
