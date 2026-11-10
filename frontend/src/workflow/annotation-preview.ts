import { api } from "../api/client";
import { obbCorners } from "../utils/geometry";
import type { ProfileClassRule, ReviewPayload } from "../types/workflows";
import { decodeMask } from "./masks";

export async function annotationPreview(
  payload: ReviewPayload,
  classes: ProfileClassRule[],
): Promise<Blob> {
  const { width, height } = payload;
  const image = new Image();
  const sourceUrl = URL.createObjectURL(await api.imageBlob(payload.image_id));
  try {
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve();
      image.onerror = () =>
        reject(new Error("原图解码失败，请确认图片完整后重试下载。"));
      image.src = sourceUrl;
    });
  } finally {
    URL.revokeObjectURL(sourceUrl);
  }
  if (image.naturalWidth !== width || image.naturalHeight !== height)
    throw new Error("原图尺寸与当前标注不一致，请重新打开图像复核后下载。");

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("浏览器无法创建标注预览画布。");
  context.drawImage(image, 0, 0);
  const fontSize = Math.max(14, Math.round(Math.max(width, height) / 75));
  context.font = `bold ${fontSize}px sans-serif`;
  context.lineWidth = Math.max(2, Math.max(width, height) / 500);

  for (const [index, object] of payload.objects.entries()) {
    if (object.decision === "rejected") continue;
    const rule = classes.find((item) => item.class_id === object.class_id);
    const color =
      rule?.color && /^#[0-9a-f]{6}$/i.test(rule.color)
        ? rule.color
        : rule
          ? "#22c55e"
          : "#f59e0b";
    const label = `#${index + 1} ${rule?.name ?? "待确定类别"}`;
    let x = object.bbox?.x ?? 8;
    let y = object.bbox?.y ?? 8 + index * (fontSize + 8);
    context.strokeStyle = color;
    context.fillStyle = color;

    if (object.mask) {
      const pixels = decodeMask(object.mask, width, height);
      const overlay = document.createElement("canvas");
      overlay.width = width;
      overlay.height = height;
      const overlayContext = overlay.getContext("2d");
      if (!overlayContext) throw new Error("浏览器无法创建掩膜预览画布。");
      const imageData = overlayContext.createImageData(width, height);
      const rgb = [1, 3, 5].map((position) =>
        parseInt(color.slice(position, position + 2), 16),
      );
      for (let pixel = 0; pixel < pixels.length; pixel++) {
        if (pixels[pixel]) imageData.data.set([...rgb, 102], pixel * 4);
      }
      overlayContext.putImageData(imageData, 0, 0);
      context.drawImage(overlay, 0, 0);
      const first = pixels.findIndex((value) => value > 0);
      if (first >= 0) {
        x = first % width;
        y = Math.floor(first / width);
      }
    } else if (object.mask_ref) {
      throw new Error("对象掩膜尚未读取完成，请重新打开图像复核后下载。");
    } else if (
      (object.type === "polygon" && object.polygon?.length) ||
      (object.type === "obb" && object.obb)
    ) {
      const points =
        object.type === "obb" ? obbCorners(object.obb!) : object.polygon!;
      context.beginPath();
      points.forEach(([left, top], pointIndex) =>
        pointIndex ? context.lineTo(left, top) : context.moveTo(left, top),
      );
      context.closePath();
      context.globalAlpha = 0.25;
      context.fill();
      context.globalAlpha = 1;
      context.stroke();
      x = Math.min(...points.map(([left]) => left));
      y = Math.min(...points.map(([, top]) => top));
    } else if (object.bbox) {
      const box = object.bbox;
      context.strokeRect(box.x, box.y, box.width, box.height);
    }

    const labelWidth = context.measureText(label).width + 8;
    x = Math.max(0, Math.min(x, width - labelWidth));
    y = Math.max(fontSize + 8, Math.min(y, height));
    context.fillStyle = "rgba(15, 23, 42, 0.85)";
    context.fillRect(x, y - fontSize - 8, labelWidth, fontSize + 8);
    context.fillStyle = color;
    context.fillText(label, x + 4, y - 5);
  }

  return new Promise<Blob>((resolve, reject) => {
    try {
      canvas.toBlob(
        (blob) =>
          blob
            ? resolve(blob)
            : reject(new Error("标注预览生成失败，请重试下载。")),
        "image/png",
      );
    } catch {
      reject(new Error("浏览器无法导出原图，请检查图像服务的跨域设置后重试。"));
    }
  });
}
