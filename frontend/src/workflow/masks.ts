import type { MaskRle } from "../types/workflows";

export function expandCounts(counts: string | number[]): number[] {
  if (Array.isArray(counts)) return counts;
  const result: number[] = [];
  let position = 0;
  while (position < counts.length) {
    let value = 0;
    let shift = 0;
    let digit: number;
    do {
      digit = counts.charCodeAt(position++) - 48;
      if (!Number.isInteger(digit) || digit < 0 || digit > 63 || shift > 30)
        throw new Error("无效的 COCO RLE 编码");
      value |= (digit & 31) << shift;
      shift += 5;
    } while (digit & 32);
    if (digit & 16) value |= -1 << shift;
    if (result.length > 2) value += result[result.length - 2];
    result.push(value);
  }
  return result;
}

export function decodeMask(
  mask: MaskRle,
  width: number,
  height: number,
): Uint8Array {
  if (mask.size[0] !== height || mask.size[1] !== width)
    throw new Error("掩膜尺寸与原图不一致");
  const pixels = new Uint8Array(width * height);
  let offset = 0;
  let foreground = false;
  for (const count of expandCounts(mask.counts)) {
    if (
      !Number.isSafeInteger(count) ||
      count < 0 ||
      offset + count > pixels.length
    )
      throw new Error("无效的 COCO RLE 长度");
    if (foreground) {
      for (let index = offset; index < offset + count; index++)
        pixels[(index % height) * width + Math.floor(index / height)] = 1;
    }
    offset += count;
    foreground = !foreground;
  }
  if (offset !== pixels.length) throw new Error("COCO RLE 未覆盖原图尺寸");
  return pixels;
}

export function encodeMask(
  pixels: Uint8Array,
  width: number,
  height: number,
): MaskRle {
  if (pixels.length !== width * height)
    throw new Error("掩膜像素数与尺寸不一致");
  const counts: number[] = [];
  let last = 0;
  let count = 0;
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      const value = pixels[y * width + x] ? 1 : 0;
      if (value === last) count++;
      else {
        counts.push(count);
        last = value;
        count = 1;
      }
    }
  }
  counts.push(count);
  return { size: [height, width], counts };
}

export function paintMask(
  pixels: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  radius: number,
  value: 0 | 1,
): void {
  for (
    let row = Math.max(0, Math.floor(y - radius));
    row <= Math.min(height - 1, Math.ceil(y + radius));
    row++
  ) {
    for (
      let column = Math.max(0, Math.floor(x - radius));
      column <= Math.min(width - 1, Math.ceil(x + radius));
      column++
    ) {
      if ((column - x) ** 2 + (row - y) ** 2 <= radius ** 2)
        pixels[row * width + column] = value;
    }
  }
}

export function polygonMask(
  points: [number, number][],
  width: number,
  height: number,
): MaskRle {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("浏览器无法创建绘图画布");
  context.beginPath();
  points.forEach(([x, y], index) =>
    index ? context.lineTo(x, y) : context.moveTo(x, y),
  );
  context.closePath();
  context.fill();
  const rgba = context.getImageData(0, 0, width, height).data;
  const pixels = Uint8Array.from({ length: width * height }, (_, index) =>
    rgba[index * 4 + 3] > 0 ? 1 : 0,
  );
  return encodeMask(pixels, width, height);
}
