import { useEffect, useState } from "react";
import { visionApi } from "../api/vision";
import type { AnnotationDraft } from "./AnnotationCanvas";
import { MaskReviewCanvas } from "./MaskReviewCanvas";
import { polygonMask } from "../workflow/masks";
import type { BBox, ClassLabel, ImageItem } from "../types";
import type { MaskRle, ReviewPayload } from "../types/workflows";

export interface AnnotationRegions {
  roi?: MaskRle | null;
  ignore?: MaskRle | null;
}

export function StandaloneMaskEditor({
  image,
  classes,
  drafts,
  activeClassId,
  selectedId,
  onSelect,
  onChange,
  onRegionsChange,
  onSamBox,
  onSamPoints,
  onDrawingChange,
  busy,
}: {
  image: ImageItem;
  classes: ClassLabel[];
  drafts: AnnotationDraft[];
  activeClassId: string;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  onChange: (drafts: AnnotationDraft[]) => void;
  onRegionsChange: (regions: AnnotationRegions) => void;
  onSamBox: (box: BBox) => void;
  onSamPoints: (points: Array<{ x: number; y: number; label: 0 | 1 }>) => void;
  onDrawingChange?: (drawing: boolean) => void;
  busy: boolean;
}) {
  const [payload, setPayload] = useState<ReviewPayload>();
  const [history, setHistory] = useState<ReviewPayload[]>([]);
  const [future, setFuture] = useState<ReviewPayload[]>([]);
  const [newObjectRequest, setNewObjectRequest] = useState(0);
  const [pendingDrawing, setPendingDrawing] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => onDrawingChange?.(pendingDrawing), [pendingDrawing, onDrawingChange]);
  useEffect(() => () => onDrawingChange?.(false), [onDrawingChange]);
  useEffect(() => {
    let active = true;
    const load = async () => {
      const objects = await Promise.all(
        drafts.map(async (draft, index) => ({
          ...draft,
          object_id: draft.object_id ?? draft.id ?? `draft-${index}`,
          mask:
            draft.mask ??
            (draft.mask_ref
              ? (await visionApi.mask(draft.mask_ref)).rle
              : draft.polygon?.length
                ? polygonMask(draft.polygon, image.width, image.height)
                : undefined),
        })),
      );
      const regions =
        payload?.image_id === image.id
          ? { roi: payload.roi, ignore: payload.ignore }
          : {
              roi: image.roi_mask_ref
                ? (await visionApi.mask(image.roi_mask_ref)).rle
                : null,
              ignore: image.ignore_mask_ref
                ? (await visionApi.mask(image.ignore_mask_ref)).rle
                : null,
            };
      if (active) {
        setPayload({
          image_id: image.id,
          dataset_id: image.dataset_id,
          image_url: image.file_url,
          width: image.width,
          height: image.height,
          base_revision: image.annotation_revision ?? 0,
          objects,
          issues: [],
          ...regions,
        });
        setError("");
      }
    };
    void load().catch((reason) => {
      if (active)
        setError(reason instanceof Error ? reason.message : "掩膜读取失败");
    });
    return () => {
      active = false;
    };
  }, [drafts, image]);
  useEffect(() => {
    setHistory([]);
    setFuture([]);
    setPendingDrawing(false);
  }, [image.id, image.annotation_revision]);
  useEffect(() => {
    if (!payload || !selectedId) return;
    const objectId = drafts.find((draft) => draft.id === selectedId)?.object_id ?? selectedId;
    if (payload.objects.some((object) =>
      object.object_id === objectId && object.decision !== "rejected")) return;
    const next = payload.objects.find((object) => object.decision !== "rejected");
    onSelect(next
      ? drafts.find((draft) => (draft.object_id ?? draft.id) === next.object_id)?.id ?? next.object_id
      : null);
  }, [payload?.objects, selectedId, drafts, onSelect]);
  const apply = (next: ReviewPayload, remember = true) => {
    if (remember && payload) {
      setHistory((items) => [...items.slice(-19), payload]);
      setFuture([]);
    }
    if (next.roi !== payload?.roi || next.ignore !== payload?.ignore)
      onRegionsChange({ roi: next.roi, ignore: next.ignore });
    setPayload(next);
    onChange(
      next.objects.map((object) => ({
        id:
          drafts.find(
            (draft) => (draft.object_id ?? draft.id) === object.object_id,
          )?.id ?? object.object_id,
        object_id: object.object_id,
        class_id: object.class_id ?? "",
        type: "polygon",
        mask: object.mask,
        mask_ref: object.mask_ref,
        polygon: object.polygon,
        source: object.source ?? "manual",
        provenance_json: object.provenance_json,
      })),
    );
  };
  const add = () => {
    if (!payload || !activeClassId || busy || pendingDrawing) return;
    setNewObjectRequest((value) => value + 1);
  };
  const undo = () => {
    if (busy || pendingDrawing || !payload || !history.length) return;
    setFuture((items) => [payload, ...items]);
    const next = history[history.length - 1];
    setHistory((items) => items.slice(0, -1));
    apply(next, false);
  };
  const redo = () => {
    if (busy || pendingDrawing || !payload || !future.length) return;
    setHistory((items) => [...items, payload]);
    const next = future[0];
    setFuture((items) => items.slice(1));
    apply(next, false);
  };
  return (
    <section>
      {error && (
        <p role="alert" className="validation invalid">
          {error}
        </p>
      )}
      {!payload && !error && <p>正在读取权威掩膜…</p>}
      <div className="workflow-actions">
        <button
          className="button"
          disabled={!payload || busy || pendingDrawing || !activeClassId}
          onClick={add}
        >
          新增分割对象
        </button>
        <button
          className="button"
          disabled={busy || pendingDrawing || !payload || !history.length}
          onClick={undo}
        >
          撤销掩膜编辑
        </button>
        <button
          className="button"
          disabled={busy || pendingDrawing || !payload || !future.length}
          onClick={redo}
        >
          重做掩膜编辑
        </button>
      </div>
      {pendingDrawing && <p className="hint">请先完成轮廓或取消本次绘制。</p>}
      {payload && (
        <MaskReviewCanvas
          activeClassId={activeClassId}
          newObjectRequest={newObjectRequest}
          onDrawingChange={setPendingDrawing}
          onUndo={undo}
          onRedo={redo}
          canUndo={history.length > 0}
          canRedo={future.length > 0}
          payload={payload}
          classes={classes.map((item) => ({
            class_id: item.id,
            name: item.name,
            color: item.color,
          }))}
          selectedId={
            drafts.find((draft) => draft.id === selectedId)?.object_id ??
            selectedId ??
            ""
          }
          onSelect={(id) =>
            onSelect(
              drafts.find((draft) => (draft.object_id ?? draft.id) === id)
                ?.id ?? id,
            )
          }
          onChange={apply}
          onSamBox={onSamBox}
          onSamPoints={onSamPoints}
          disabled={busy}
          onError={setError}
        />
      )}
    </section>
  );
}
