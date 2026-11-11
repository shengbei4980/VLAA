import { useEffect, useMemo, useRef, useState } from "react";
import { apiUrl } from "../api/client";
import { decodeMask, encodeMask, paintMask } from "../workflow/masks";
import { polygonMask } from "../workflow/masks";
import { combineMask, validatePolygon } from "./mask-editing";
import type { BBox } from "../types";
import type {
  MaskRle,
  ProfileClassRule,
  ReviewPayload,
  ReviewObject,
} from "../types/workflows";

type Tool =
  | "select"
  | "brush"
  | "erase"
  | "polygon"
  | "vertices"
  | "sam_box"
  | "sam_point";
type MaskLayer = "object" | "roi" | "ignore";
type Point = [number, number];
type PolygonMode = "new" | "add" | "subtract";
export interface PromptPoint {
  x: number;
  y: number;
  label: 0 | 1;
}

export function MaskReviewCanvas({
  payload,
  classes,
  selectedId,
  onSelect,
  onChange,
  onSamBox,
  onSamPoints,
  disabled = false,
  onError,
  focusRegion,
  activeClassId,
  newObjectRequest = 0,
  onDrawingChange,
  onUndo,
  onRedo,
  canUndo = false,
  canRedo = false,
}: {
  payload: ReviewPayload;
  classes: ProfileClassRule[];
  selectedId: string;
  onSelect: (id: string) => void;
  onChange: (payload: ReviewPayload) => void;
  onSamBox: (box: BBox) => void;
  onSamPoints: (points: PromptPoint[]) => void;
  disabled?: boolean;
  onError: (message: string) => void;
  focusRegion?: BBox;
  activeClassId?: string;
  newObjectRequest?: number;
  onDrawingChange?: (drawing: boolean) => void;
  onUndo?: () => void;
  onRedo?: () => void;
  canUndo?: boolean;
  canRedo?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const [fitSize, setFitSize] = useState({ width: 850, height: 600 });
  const painting = useRef<Uint8Array>();
  const previousPoint = useRef<[number, number]>();
  const boxStart = useRef<[number, number]>();
  const boxEnd = useRef<[number, number]>();
  const gesture = useRef<{
    payload: ReviewPayload;
    selectedId: string;
    layer: MaskLayer;
  }>();
  const decoded = useRef(new WeakMap<MaskRle, Uint8Array>());
  const polygonContext = useRef<{
    payload: ReviewPayload;
    selectedId: string;
    classId?: string;
  }>();
  const vertexDraft = useRef<Point[]>();
  const vertexIndex = useRef<number>();
  const recentCompletion = useRef<{ point: Point; timeStamp: number }>();
  const [tool, setTool] = useState<Tool>("select");
  const [layer, setLayer] = useState<MaskLayer>("object");
  const [radius, setRadius] = useState(12);
  const [opacity, setOpacity] = useState(0.4);
  const [zoom, setZoom] = useState(1);
  const [showObjects, setShowObjects] = useState(true);
  const [hiddenClasses, setHiddenClasses] = useState<string[]>([]);
  const [pointLabel, setPointLabel] = useState<0 | 1>(1);
  const [points, setPoints] = useState<PromptPoint[]>([]);
  const [frame, setFrame] = useState(0);
  const [polygonMode, setPolygonMode] = useState<PolygonMode>("new");
  const [polygonPoints, setPolygonPoints] = useState<Point[]>([]);
  const [hoverPoint, setHoverPoint] = useState<Point>();
  const [selectedVertex, setSelectedVertex] = useState<number>();
  const currentContext = useRef({ payload, selectedId, layer, disabled });
  currentContext.current = { payload, selectedId, layer, disabled };
  const { width, height } = payload;
  const selected = payload.objects.find(
    (object) => object.object_id === selectedId,
  );
  const editableSelected =
    selected?.decision !== "rejected" ? selected : undefined;
  const drawingClass = activeClassId ?? editableSelected?.class_id ?? "";
  const drawing = polygonPoints.length > 0 || !!vertexDraft.current;
  const pixelsFor = (mask: MaskRle) => {
    let pixels = decoded.current.get(mask);
    if (!pixels) {
      pixels = decodeMask(mask, width, height);
      decoded.current.set(mask, pixels);
    }
    return pixels;
  };
  const canEditVertices = useMemo(() => {
    if (!editableSelected?.polygon?.length) return false;
    try {
      validatePolygon(editableSelected.polygon, width, height);
      if (!editableSelected.mask) return !editableSelected.mask_ref;
      const vector = decodeMask(
        polygonMask(editableSelected.polygon, width, height),
        width,
        height,
      );
      const mask = decodeMask(editableSelected.mask, width, height);
      return vector.every((value, index) => value === mask[index]);
    } catch {
      return false;
    }
  }, [
    editableSelected?.polygon,
    editableSelected?.mask,
    editableSelected?.mask_ref,
    width,
    height,
  ]);
  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const maxHeight = Number.parseFloat(getComputedStyle(viewport).maxHeight);
      const next = { width: Math.max(1, viewport.clientWidth - 2), height: Math.max(1, (Number.isFinite(maxHeight) ? maxHeight : viewport.clientHeight || 600) - 2) };
      setFitSize((current) => current.width === next.width && current.height === next.height ? current : next);
    };
    const observer = new ResizeObserver(measure);
    observer.observe(viewport);
    window.addEventListener("resize", measure);
    measure();
    return () => { observer.disconnect(); window.removeEventListener("resize", measure); };
  }, []);
  const scale = Math.min(fitSize.width / width, fitSize.height / height, 1) * zoom;
  const imageUrl = payload.image_url.startsWith("http")
    ? payload.image_url
    : apiUrl(payload.image_url);

  useEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    const draw = (
      mask: MaskRle | null | undefined,
      color: string,
      activePixels?: Uint8Array,
    ) => {
      if (!mask && !activePixels) return;
      const pixels = activePixels ?? pixelsFor(mask!);
      const overlay = document.createElement("canvas");
      overlay.width = width;
      overlay.height = height;
      const overlayContext = overlay.getContext("2d");
      if (!overlayContext) return;
      const imageData = overlayContext.createImageData(width, height);
      const normalized = /^#[0-9a-f]{6}$/i.test(color) ? color : "#22c55e";
      const rgb = [1, 3, 5].map((position) =>
        parseInt(normalized.slice(position, position + 2), 16),
      );
      for (let index = 0; index < pixels.length; index++) {
        if (!pixels[index]) continue;
        imageData.data.set(
          [rgb[0], rgb[1], rgb[2], Math.round(opacity * 255)],
          index * 4,
        );
      }
      overlayContext.putImageData(imageData, 0, 0);
      context.drawImage(overlay, 0, 0);
    };
    try {
      if (showObjects)
        payload.objects.forEach((object, index) => {
          if (
            object.decision === "rejected" ||
            (object.class_id !== null &&
              hiddenClasses.includes(object.class_id))
          )
            return;
          draw(
            object.mask,
            classes.find((item) => item.class_id === object.class_id)?.color ??
              "#22c55e",
            layer === "object" && object.object_id === selectedId
              ? painting.current
              : undefined,
          );
          const pixels = object.mask ? pixelsFor(object.mask) : undefined;
          const start = pixels?.findIndex((value) => value > 0) ?? -1;
          if (start < 0 && !object.bbox) return;
          if (!object.mask && object.bbox) {
            context.strokeStyle = "#f59e0b";
            context.lineWidth = 2 / scale;
            context.strokeRect(
              object.bbox.x,
              object.bbox.y,
              object.bbox.width,
              object.bbox.height,
            );
          }
          context.font = `${14 / scale}px sans-serif`;
          context.fillStyle =
            object.object_id === selectedId ? "#fbbf24" : "#ffffff";
          context.strokeStyle = "#0f172a";
          context.lineWidth = 3 / scale;
          const text = `#${index + 1}`;
          const x = start >= 0 ? start % width : object.bbox!.x;
          const y = Math.max(
            16 / scale,
            start >= 0 ? Math.floor(start / width) : object.bbox!.y,
          );
          context.strokeText(text, x, y);
          context.fillText(text, x, y);
        });
      draw(
        payload.roi,
        "#3b82f6",
        layer === "roi" ? painting.current : undefined,
      );
      draw(
        payload.ignore,
        "#ef4444",
        layer === "ignore" ? painting.current : undefined,
      );
      context.lineWidth = Math.max(2, 2 / scale);
      context.strokeStyle = "#f59e0b";
      if (focusRegion)
        context.strokeRect(
          focusRegion.x,
          focusRegion.y,
          focusRegion.width,
          focusRegion.height,
        );
      if (boxStart.current && boxEnd.current)
        context.strokeRect(
          boxStart.current[0],
          boxStart.current[1],
          boxEnd.current[0] - boxStart.current[0],
          boxEnd.current[1] - boxStart.current[1],
        );
      for (const point of points) {
        context.fillStyle = point.label ? "#22c55e" : "#ef4444";
        context.beginPath();
        context.arc(point.x, point.y, 4 / scale, 0, Math.PI * 2);
        context.fill();
      }
      const outline =
        tool === "vertices" && canEditVertices
          ? (vertexDraft.current ?? editableSelected?.polygon ?? [])
          : polygonPoints;
      if (outline.length) {
        context.strokeStyle = "#fbbf24";
        context.fillStyle = "#fbbf24";
        context.lineWidth = 2 / scale;
        context.beginPath();
        outline.forEach(([x, y], index) =>
          index ? context.lineTo(x, y) : context.moveTo(x, y),
        );
        if (tool === "vertices") context.closePath();
        else if (hoverPoint) context.lineTo(...hoverPoint);
        context.stroke();
        outline.forEach(([x, y], index) => {
          context.fillStyle =
            index === selectedVertex
              ? "#ef4444"
              : index === 0
                ? "#22c55e"
                : "#fbbf24";
          context.beginPath();
          context.arc(x, y, 5 / scale, 0, Math.PI * 2);
          context.fill();
        });
      }
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "掩膜绘制失败");
    }
  }, [
    payload,
    classes,
    selectedId,
    layer,
    frame,
    opacity,
    showObjects,
    hiddenClasses,
    points,
    width,
    height,
    scale,
    onError,
    focusRegion,
    tool,
    polygonPoints,
    hoverPoint,
    selectedVertex,
    canEditVertices,
  ]);

  const cancelGesture = () => {
    painting.current = undefined;
    previousPoint.current = undefined;
    boxStart.current = undefined;
    boxEnd.current = undefined;
    vertexDraft.current = undefined;
    vertexIndex.current = undefined;
    gesture.current = undefined;
    setFrame((value) => value + 1);
  };
  const cancelPolygon = () => {
    setPolygonPoints([]);
    setHoverPoint(undefined);
    polygonContext.current = undefined;
  };
  useEffect(() => {
    onDrawingChange?.(drawing);
  }, [drawing, onDrawingChange]);
  useEffect(() => () => onDrawingChange?.(false), [onDrawingChange]);
  useEffect(() => {
    if (!polygonPoints.length) {
      polygonContext.current = undefined;
      setHoverPoint(undefined);
    }
  }, [polygonPoints.length]);
  useEffect(() => {
    const started = gesture.current;
    if (
      started &&
      (started.payload !== payload ||
        started.selectedId !== selectedId ||
        started.layer !== layer ||
        disabled)
    )
      cancelGesture();
    const draft = polygonContext.current;
    if (
      draft &&
      (draft.payload !== payload || draft.selectedId !== selectedId || disabled)
    )
      cancelPolygon();
  }, [payload, selectedId, layer, disabled]);
  useEffect(() => setSelectedVertex(undefined), [selectedId, layer]);
  useEffect(() => {
    if (!newObjectRequest || disabled) return;
    recentCompletion.current = undefined;
    cancelGesture();
    cancelPolygon();
    setLayer("object");
    setTool("polygon");
    setPolygonMode("new");
    canvasRef.current?.focus();
  }, [newObjectRequest]);

  const updatedObject = (
    object: ReviewObject,
    mask: MaskRle,
    polygon?: Point[],
  ): ReviewObject => ({
    ...object,
    type: "polygon",
    mask,
    polygon,
    mask_ref: undefined,
    bbox: undefined,
    obb: undefined,
    source: "manual",
    decision: "accepted",
  });
  const updateSelectedMask = (mask: MaskRle, polygon?: Point[]) => {
    if (!editableSelected) throw new Error("请先选择需要修改的对象。");
    if (!decodeMask(mask, width, height).some(Boolean))
      throw new Error(
        "对象轮廓已为空。需要删除整个对象时，请使用右侧的“拒绝对象”。",
      );
    onChange({
      ...payload,
      objects: payload.objects.map((object) =>
        object.object_id === selectedId
          ? updatedObject(object, mask, polygon)
          : object,
      ),
    });
  };
  const finishPolygon = () => {
    if (disabled) return;
    try {
      validatePolygon(polygonPoints, width, height);
      const region = polygonMask(polygonPoints, width, height);
      if (!decodeMask(region, width, height).some(Boolean))
        throw new Error("轮廓没有覆盖有效像素，请重新绘制。");
      if (polygonMode === "new") {
        const classId = polygonContext.current?.classId ?? drawingClass;
        if (!classes.some((item) => item.class_id === classId))
          throw new Error("请先选择本次要绘制的类别。");
        const object: ReviewObject = {
          object_id: crypto.randomUUID(),
          class_id: classId,
          type: "polygon",
          polygon: polygonPoints,
          mask: region,
          source: "manual",
          decision: "accepted",
        };
        onChange({ ...payload, objects: [...payload.objects, object] });
        onSelect(object.object_id);
      } else {
        if (!editableSelected)
          throw new Error("请先选择需要补全或擦除的对象。");
        if (editableSelected.mask_ref && !editableSelected.mask)
          throw new Error(
            "当前对象的掩膜尚未读取完成，请重新打开复核后再修边。",
          );
        updateSelectedMask(
          combineMask(
            editableSelected.mask,
            region,
            width,
            height,
            polygonMode,
          ),
        );
      }
      cancelPolygon();
      onError("");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "轮廓绘制失败");
    }
  };
  const commitVertices = (outline: Point[]) => {
    try {
      validatePolygon(outline, width, height);
      updateSelectedMask(polygonMask(outline, width, height), outline);
      onError("");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "顶点调整失败");
    }
  };
  const removeVertex = () => {
    if (
      !canEditVertices ||
      !editableSelected?.polygon ||
      selectedVertex === undefined
    )
      return;
    if (editableSelected.polygon.length <= 3) {
      onError("一个多边形至少需要 3 个顶点。");
      return;
    }
    commitVertices(
      editableSelected.polygon.filter((_, index) => index !== selectedVertex),
    );
    setSelectedVertex(undefined);
  };

  const position = (
    event: React.MouseEvent<HTMLCanvasElement>,
  ): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      Math.max(
        0,
        Math.min(
          tool === "polygon" || tool === "vertices" ? width : width - 1,
          ((event.clientX - rect.left) * width) / rect.width,
        ),
      ),
      Math.max(
        0,
        Math.min(
          tool === "polygon" || tool === "vertices" ? height : height - 1,
          ((event.clientY - rect.top) * height) / rect.height,
        ),
      ),
    ];
  };
  const stroke = (point: [number, number]) => {
    if (!painting.current) return;
    const previous = previousPoint.current ?? point;
    const distance = Math.hypot(point[0] - previous[0], point[1] - previous[1]);
    const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius / 2)));
    for (let step = 1; step <= steps; step++)
      paintMask(
        painting.current,
        width,
        height,
        previous[0] + ((point[0] - previous[0]) * step) / steps,
        previous[1] + ((point[1] - previous[1]) * step) / steps,
        radius,
        tool === "erase" ? 0 : 1,
      );
    previousPoint.current = point;
    setFrame((value) => value + 1);
  };
  const commit = () => {
    const pixels = painting.current;
    const started = gesture.current;
    cancelGesture();
    if (!pixels) return;
    if (
      disabled ||
      started?.payload !== payload ||
      started.selectedId !== selectedId ||
      started.layer !== layer
    )
      return;
    const mask = encodeMask(pixels, width, height);
    try {
      if (layer === "object") updateSelectedMask(mask);
      else onChange({ ...payload, [layer]: mask, [`${layer}_mask_ref`]: null });
      onError("");
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "掩膜修改失败");
    }
  };
  const importMask = async (file?: File) => {
    if (!file) return;
    const starting = currentContext.current;
    try {
      const mask = JSON.parse(await file.text()) as MaskRle;
      decodeMask(mask, width, height);
      const current = currentContext.current;
      if (
        current.disabled ||
        current.payload !== starting.payload ||
        current.selectedId !== starting.selectedId ||
        current.layer !== starting.layer
      )
        throw new Error("导入期间编辑目标已变化，请为当前对象重新导入掩膜。");
      if (layer === "object") {
        updateSelectedMask(mask);
      } else
        onChange({ ...payload, [layer]: mask, [`${layer}_mask_ref`]: null });
    } catch (reason) {
      onError(reason instanceof Error ? reason.message : "掩膜导入失败");
    }
  };
  return (
    <section className="canvas-card workflow-mask-canvas">
      <div className="workflow-actions">
        <label>
          编辑图层
          <select
            value={layer}
            disabled={disabled || drawing}
            onChange={(event) => setLayer(event.target.value as MaskLayer)}
          >
            <option value="object">对象掩膜</option>
            <option value="roi">ROI 有效区域（蓝）</option>
            <option value="ignore">Ignore 排除区域（红）</option>
          </select>
        </label>
        {onUndo && (
          <button
            className="button"
            disabled={disabled || drawing || !canUndo}
            onClick={onUndo}
            title="撤销最近一次已完成的修改"
          >
            撤销上一步
          </button>
        )}
        {onRedo && (
          <button
            className="button"
            disabled={disabled || drawing || !canRedo}
            onClick={onRedo}
            title="恢复刚刚撤销的修改"
          >
            重做上一步
          </button>
        )}
        {(
          [
            "select",
            "brush",
            "erase",
            "polygon",
            "vertices",
            "sam_box",
            "sam_point",
          ] as const
        ).map((value) => (
          <button
            key={value}
            className={`button${tool === value ? " active" : ""}`}
            disabled={
              disabled ||
              drawing ||
              (value === "vertices" &&
                (!canEditVertices || layer !== "object")) ||
              (value === "polygon" && layer !== "object") ||
              ((value === "sam_box" || value === "sam_point") &&
                layer !== "object")
            }
            title={
              value === "vertices" && !canEditVertices
                ? "像素掩膜可用多边形补全、擦除或画笔修边"
                : value === "sam_box" || value === "sam_point"
                  ? "用 SAM 重新生成选中对象的完整轮廓"
                  : undefined
            }
            aria-pressed={tool === value}
            onClick={() => {
              cancelGesture();
              cancelPolygon();
              setPoints([]);
              setTool(value);
            }}
          >
            {
              {
                select: "选择",
                brush: "画笔",
                erase: "橡皮",
                polygon: "多边形",
                vertices: "编辑顶点",
                sam_box: "SAM 框选",
                sam_point: "SAM 正负点",
              }[value]
            }
          </button>
        ))}
      </div>
      {tool === "polygon" && (
        <div className="workflow-actions">
          <label>
            多边形用途
            <select
              value={polygonMode}
              disabled={disabled || drawing}
              onChange={(event) =>
                setPolygonMode(event.target.value as PolygonMode)
              }
            >
              <option value="new">新增漏标对象</option>
              <option value="add">补全选中对象</option>
              <option value="subtract">擦除选中区域</option>
            </select>
          </label>
          <button
            className="button primary"
            disabled={disabled || polygonPoints.length < 3}
            onClick={finishPolygon}
          >
            完成轮廓
          </button>
          <button
            className="button"
            disabled={disabled || !polygonPoints.length}
            onClick={() => setPolygonPoints((values) => values.slice(0, -1))}
          >
            撤回上一个点
          </button>
          <button
            className="button"
            disabled={disabled || !polygonPoints.length}
            onClick={cancelPolygon}
          >
            取消本次绘制
          </button>
          <span>{polygonPoints.length} 个顶点</span>
        </div>
      )}
      {tool === "vertices" && (
        <div className="workflow-actions">
          <button
            className="button"
            disabled={disabled || drawing || selectedVertex === undefined}
            onClick={removeVertex}
          >
            删除选中顶点
          </button>
          <span>拖动顶点调整轮廓，双击边缘添加顶点，Delete 删除选中顶点。</span>
        </div>
      )}
      <div className="workflow-actions">
        <label>
          画笔半径 {radius}px
          <input
            type="range"
            min="1"
            max="128"
            value={radius}
            onChange={(event) => setRadius(Number(event.target.value))}
          />
        </label>
        <label>
          叠加透明度
          <input
            type="range"
            min="0"
            max="0.85"
            step="0.05"
            value={opacity}
            onChange={(event) => setOpacity(Number(event.target.value))}
          />
        </label>
        <label>
          缩放
          <input
            type="range"
            min="0.25"
            max="4"
            step="0.25"
            value={zoom}
            onChange={(event) => setZoom(Number(event.target.value))}
          />
        </label>
        <label className="workflow-check">
          <input
            type="checkbox"
            checked={showObjects}
            onChange={(event) => setShowObjects(event.target.checked)}
          />
          对象叠加
        </label>
        <label className="button">
          导入此层 RLE
          <input
            aria-label="导入掩膜 RLE"
            type="file"
            accept="application/json,.json"
            hidden
            disabled={disabled || drawing}
            onChange={(event) => {
              void importMask(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
        </label>
        {layer !== "object" && (
          <button
            className="button"
            disabled={disabled}
            onClick={() =>
              onChange({
                ...payload,
                [layer]: null,
                [`${layer}_mask_ref`]: null,
              })
            }
          >
            {layer === "roi" ? "恢复全图有效" : "清除 ignore"}
          </button>
        )}
      </div>
      {tool === "sam_point" && (
        <div className="workflow-actions">
          <select
            aria-label="SAM 点类型"
            value={pointLabel}
            onChange={(event) =>
              setPointLabel(Number(event.target.value) as 0 | 1)
            }
          >
            <option value="1">前景点</option>
            <option value="0">背景点</option>
          </select>
          <button
            className="button primary"
            disabled={disabled || !points.length}
            onClick={() => {
              onSamPoints(points);
              setPoints([]);
            }}
          >
            提交 {points.length} 个提示点
          </button>
          <button className="button" onClick={() => setPoints([])}>
            清空提示点
          </button>
        </div>
      )}
      <p className="hint">
        {layer === "roi"
          ? "未设置 ROI 时全图有效。绘制 ROI 后仅蓝色区域参与监督。"
          : layer === "ignore"
            ? "红色区域不参与监督。无法判断的区域可明确标为 ignore。"
            : tool === "polygon"
              ? "逐点勾画轮廓，点击起点、双击或按 Enter 完成。Backspace 撤回一点，Esc 取消。补全和擦除只修改选中对象。"
              : tool === "vertices"
                ? "可编辑手绘多边形的顶点。像素掩膜使用画笔、橡皮或多边形补全与擦除修边。"
                : "选择已有对象后，画笔补上漏掉的部分，橡皮擦去多余区域。新增对象可点击“新增漏标对象”并逐点勾画。"}
      </p>
      <div className="workflow-actions">
        {classes.map((item) => (
          <label key={item.class_id} className="workflow-check">
            <input
              type="checkbox"
              checked={!hiddenClasses.includes(item.class_id)}
              onChange={(event) =>
                setHiddenClasses((values) =>
                  event.target.checked
                    ? values.filter((id) => id !== item.class_id)
                    : [...values, item.class_id],
                )
              }
            />
            {item.name}
          </label>
        ))}
      </div>
      <div className="workflow-canvas-scroll" ref={viewportRef}>
        <div
          className="workflow-mask-stage"
          style={{ width: width * scale, height: height * scale }}
        >
          <img
            src={imageUrl}
            alt="原图"
            draggable={false}
            style={{ width: "100%", height: "100%" }}
          />
          <canvas
            ref={canvasRef}
            width={width}
            height={height}
            aria-label="掩膜复核画布"
            tabIndex={0}
            style={{
              width: "100%",
              height: "100%",
              touchAction: "none",
              cursor: tool === "select" ? "pointer" : "crosshair",
            }}
            onPointerDown={(event) => {
              if (disabled || event.button !== 0) return;
              event.currentTarget.focus();
              const point = position(event);
              if (tool === "select") {
                const hit =
                  showObjects &&
                  [...payload.objects].reverse().find((object) => {
                    if (
                      object.decision === "rejected" ||
                      (object.class_id &&
                        hiddenClasses.includes(object.class_id))
                    )
                      return false;
                    if (object.mask)
                      return (
                        pixelsFor(object.mask)[
                          Math.floor(point[1]) * width + Math.floor(point[0])
                        ] > 0
                      );
                    const box = object.bbox;
                    return (
                      box &&
                      point[0] >= box.x &&
                      point[0] <= box.x + box.width &&
                      point[1] >= box.y &&
                      point[1] <= box.y + box.height
                    );
                  });
                if (hit) onSelect(hit.object_id);
                return;
              }
              if (tool === "polygon") {
                const recent = recentCompletion.current;
                if (
                  recent &&
                  event.timeStamp - recent.timeStamp < 600 &&
                  Math.hypot(
                    point[0] - recent.point[0],
                    point[1] - recent.point[1],
                  ) <=
                    8 / scale
                )
                  return;
                recentCompletion.current = undefined;
                if (
                  polygonMode === "new" &&
                  !classes.some((item) => item.class_id === drawingClass)
                ) {
                  onError("请先选择要绘制的类别。");
                  return;
                }
                if (polygonMode !== "new" && !editableSelected) {
                  onError("请先选择需要补全或擦除的对象。");
                  return;
                }
                if (
                  polygonPoints.length >= 3 &&
                  Math.hypot(
                    point[0] - polygonPoints[0][0],
                    point[1] - polygonPoints[0][1],
                  ) <=
                    8 / scale
                ) {
                  recentCompletion.current = {
                    point,
                    timeStamp: event.timeStamp,
                  };
                  finishPolygon();
                  return;
                }
                const last = polygonPoints[polygonPoints.length - 1];
                if (
                  last &&
                  Math.hypot(point[0] - last[0], point[1] - last[1]) <
                    1.5 / scale
                )
                  return;
                polygonContext.current ??= {
                  payload,
                  selectedId,
                  classId: drawingClass,
                };
                setPolygonPoints((values) => [...values, point]);
                return;
              }
              if (tool === "vertices") {
                if (!canEditVertices || !editableSelected?.polygon) return;
                const index = editableSelected.polygon.findIndex(
                  ([x, y]) =>
                    Math.hypot(point[0] - x, point[1] - y) <= 8 / scale,
                );
                if (index < 0) return;
                vertexIndex.current = index;
                setSelectedVertex(index);
                vertexDraft.current = editableSelected.polygon.map(([x, y]) => [
                  x,
                  y,
                ]);
                gesture.current = { payload, selectedId, layer };
                event.currentTarget.setPointerCapture(event.pointerId);
                setFrame((value) => value + 1);
                return;
              }
              if (tool === "sam_point") {
                if (points.length < 32)
                  setPoints([
                    ...points,
                    { x: point[0], y: point[1], label: pointLabel },
                  ]);
                return;
              }
              event.currentTarget.setPointerCapture(event.pointerId);
              gesture.current = { payload, selectedId, layer };
              if (tool === "sam_box") {
                boxStart.current = point;
                boxEnd.current = point;
                return;
              }
              const mask =
                layer === "object" ? editableSelected?.mask : payload[layer];
              if (layer === "object" && !editableSelected) {
                onError("请先选择或新增对象");
                cancelGesture();
                return;
              }
              if (layer === "object" && !mask && editableSelected?.mask_ref) {
                onError("当前对象的掩膜尚未读取完成，请重新打开复核后再修边。");
                cancelGesture();
                return;
              }
              painting.current = mask
                ? decodeMask(mask, width, height)
                : new Uint8Array(width * height);
              stroke(point);
            }}
            onPointerMove={(event) => {
              if (disabled) return;
              const point = position(event);
              if (tool === "polygon") {
                if (polygonPoints.length) setHoverPoint(point);
              } else if (
                vertexDraft.current &&
                vertexIndex.current !== undefined
              ) {
                vertexDraft.current = vertexDraft.current.map((value, index) =>
                  index === vertexIndex.current ? point : value,
                );
                setFrame((value) => value + 1);
              } else if (boxStart.current) {
                boxEnd.current = point;
                setFrame((value) => value + 1);
              } else if (painting.current) stroke(point);
            }}
            onPointerUp={() => {
              if (disabled) {
                cancelGesture();
                return;
              }
              if (vertexDraft.current) {
                const outline = vertexDraft.current;
                const validContext =
                  gesture.current?.payload === payload &&
                  gesture.current.selectedId === selectedId;
                cancelGesture();
                if (
                  validContext &&
                  outline.some(
                    (point, index) =>
                      point[0] !== editableSelected?.polygon?.[index][0] ||
                      point[1] !== editableSelected?.polygon?.[index][1],
                  )
                )
                  commitVertices(outline);
                return;
              }
              if (boxStart.current && boxEnd.current) {
                const [start, end] = [boxStart.current, boxEnd.current];
                const validContext =
                  gesture.current?.payload === payload &&
                  gesture.current.selectedId === selectedId;
                cancelGesture();
                const box = {
                  x: Math.min(start[0], end[0]),
                  y: Math.min(start[1], end[1]),
                  width: Math.abs(end[0] - start[0]),
                  height: Math.abs(end[1] - start[1]),
                };
                if (validContext && box.width >= 2 && box.height >= 2)
                  onSamBox(box);
                setFrame((value) => value + 1);
              } else commit();
            }}
            onPointerCancel={cancelGesture}
            onLostPointerCapture={() => {
              if (gesture.current) cancelGesture();
            }}
            onDoubleClick={(event) => {
              if (disabled) return;
              if (tool === "polygon") {
                const recent = recentCompletion.current;
                const point = position(event);
                if (
                  recent &&
                  event.timeStamp - recent.timeStamp < 600 &&
                  Math.hypot(
                    point[0] - recent.point[0],
                    point[1] - recent.point[1],
                  ) <=
                    8 / scale
                ) {
                  recentCompletion.current = undefined;
                  return;
                }
                finishPolygon();
                return;
              }
              if (
                tool !== "vertices" ||
                !canEditVertices ||
                !editableSelected?.polygon
              )
                return;
              const point = position(event);
              const outline = editableSelected.polygon;
              if (
                outline.some(
                  ([x, y]) =>
                    Math.hypot(point[0] - x, point[1] - y) <= 8 / scale,
                )
              )
                return;
              let best:
                | { index: number; point: Point; distance: number }
                | undefined;
              outline.forEach((start, index) => {
                const end = outline[(index + 1) % outline.length];
                const dx = end[0] - start[0],
                  dy = end[1] - start[1];
                const t = Math.max(
                  0,
                  Math.min(
                    1,
                    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) /
                      (dx * dx + dy * dy),
                  ),
                );
                const projected: Point = [start[0] + dx * t, start[1] + dy * t];
                const distance = Math.hypot(
                  point[0] - projected[0],
                  point[1] - projected[1],
                );
                if (!best || distance < best.distance)
                  best = { index, point: projected, distance };
              });
              if (best && best.distance <= 8 / scale) {
                const inserted = [...outline];
                inserted.splice(best.index + 1, 0, best.point);
                commitVertices(inserted);
                setSelectedVertex(best.index + 1);
              }
            }}
            onKeyDown={(event) => {
              if (disabled) return;
              if (event.key === "Escape") {
                event.preventDefault();
                cancelGesture();
                cancelPolygon();
                setPoints([]);
              } else if (tool === "polygon" && event.key === "Enter") {
                event.preventDefault();
                finishPolygon();
              } else if (
                tool === "polygon" &&
                (event.key === "Backspace" ||
                  ((event.ctrlKey || event.metaKey) &&
                    event.key.toLowerCase() === "z"))
              ) {
                event.preventDefault();
                setPolygonPoints((values) => values.slice(0, -1));
              } else if (tool === "vertices" && event.key === "Delete") {
                event.preventDefault();
                removeVertex();
              }
            }}
          />
        </div>
      </div>
      <p className="hint">
        {width} × {height}px · 原图像素坐标 · ROI 蓝色 · Ignore 红色
      </p>
    </section>
  );
}
