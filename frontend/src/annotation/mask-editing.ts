import type { MaskRle } from "../types/workflows";
import { decodeMask, encodeMask } from "../workflow/masks";

export type MaskEditMode = "replace" | "add" | "subtract";

export function combineMask(
  base: MaskRle | undefined,
  region: MaskRle,
  width: number,
  height: number,
  mode: MaskEditMode,
): MaskRle {
  const regionPixels = decodeMask(region, width, height);
  if (mode === "replace") return encodeMask(regionPixels, width, height);

  const pixels = base
    ? decodeMask(base, width, height)
    : new Uint8Array(width * height);
  for (let index = 0; index < pixels.length; index++) {
    if (regionPixels[index]) pixels[index] = mode === "add" ? 1 : 0;
  }
  return encodeMask(pixels, width, height);
}

type Point = [number, number];

function cross(start: Point, end: Point, point: Point): number {
  return (
    (end[0] - start[0]) * (point[1] - start[1]) -
    (end[1] - start[1]) * (point[0] - start[0])
  );
}

function onSegment(start: Point, end: Point, point: Point): boolean {
  return (
    point[0] >= Math.min(start[0], end[0]) &&
    point[0] <= Math.max(start[0], end[0]) &&
    point[1] >= Math.min(start[1], end[1]) &&
    point[1] <= Math.max(start[1], end[1])
  );
}

function segmentsIntersect(a: Point, b: Point, c: Point, d: Point): boolean {
  const ac = cross(a, b, c);
  const ad = cross(a, b, d);
  const ca = cross(c, d, a);
  const cb = cross(c, d, b);
  if (ac === 0 && onSegment(a, b, c)) return true;
  if (ad === 0 && onSegment(a, b, d)) return true;
  if (ca === 0 && onSegment(c, d, a)) return true;
  if (cb === 0 && onSegment(c, d, b)) return true;
  return (
    ((ac > 0 && ad < 0) || (ac < 0 && ad > 0)) &&
    ((ca > 0 && cb < 0) || (ca < 0 && cb > 0))
  );
}

export function validatePolygon(
  points: Point[],
  width: number,
  height: number,
): void {
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width <= 0 ||
    height <= 0
  ) {
    throw new Error("原图尺寸必须为正整数。");
  }
  if (points.length < 3) throw new Error("请至少绘制三个轮廓顶点。");
  if (points.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y))) {
    throw new Error("轮廓坐标必须为有效数值。");
  }
  if (points.some(([x, y]) => x < 0 || y < 0 || x > width || y > height)) {
    throw new Error("轮廓顶点必须位于原图范围内。");
  }
  if (new Set(points.map(([x, y]) => `${x},${y}`)).size !== points.length) {
    throw new Error("轮廓顶点重复，请移除重合的顶点。");
  }

  for (let first = 0; first < points.length; first++) {
    const firstEnd = (first + 1) % points.length;
    for (let second = first + 1; second < points.length; second++) {
      const secondEnd = (second + 1) % points.length;
      if (second === firstEnd || secondEnd === first) continue;
      if (
        segmentsIntersect(
          points[first],
          points[firstEnd],
          points[second],
          points[secondEnd],
        )
      ) {
        throw new Error("轮廓边线存在交叉或重叠，请调整顶点后再完成。");
      }
    }
  }

  // Relative coordinates keep very small valid polygons precise on large images.
  let doubledArea = 0;
  for (let index = 1; index < points.length - 1; index++) {
    doubledArea += cross(points[0], points[index], points[index + 1]);
  }
  if (doubledArea === 0)
    throw new Error("轮廓不能只有一条直线，请绘制有面积的区域。");
}
