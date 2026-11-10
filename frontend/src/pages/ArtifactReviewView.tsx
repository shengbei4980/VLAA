import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import {
  AnnotationCanvas,
  type AnnotationDraft,
} from "../annotation/AnnotationCanvas";
import {
  MaskReviewCanvas,
  type PromptPoint,
} from "../annotation/MaskReviewCanvas";
import { api, apiUrl } from "../api/client";
import { workflowApi } from "../api/workflows";
import { visionApi } from "../api/vision";
import { useLeaveConfirm } from "../components/useLeaveConfirm";
import { ExportLocationsPanel } from "../components/ExportLocationsPanel";
import { decodeMask, polygonMask } from "../workflow/masks";
import { annotationPreview } from "../workflow/annotation-preview";
import { obbCorners } from "../utils/geometry";
import type { Annotation, BBox, ClassLabel, ImageItem } from "../types";
import type {
  ReviewObject,
  ReviewPayload,
  TaskProfileConfig,
  WorkflowReview,
} from "../types/workflows";

const issueLabels: Record<string, string> = {
  coordinate_error: "坐标需要复核",
  invalid_geometry: "标注格式待修正",
  unmapped_class: "类别待确定",
  missing: "疑似漏标",
  wrong_class: "类别需要复核",
  boundary: "边界需要复核",
  split: "需要拆分对象",
  merge: "需要合并对象",
  excluded_region: "排除区域需要复核",
  uncertain: "需要进一步判断",
};

async function hydratePayload(raw: ReviewPayload): Promise<ReviewPayload> {
  const originalObjects =
    raw.objects ?? (raw.annotations as ReviewObject[]) ?? [];
  const objects = await Promise.all(
    originalObjects.map(async (object) => {
      const normalized = {
        ...object,
        object_id:
          object.object_id ||
          (object as ReviewObject & { id?: string }).id ||
          crypto.randomUUID(),
      };
      if (object.mask_ref && !object.mask)
        normalized.mask = (await visionApi.mask(object.mask_ref)).rle;
      if (!normalized.mask && object.polygon?.length)
        normalized.mask = polygonMask(object.polygon, raw.width, raw.height);
      return normalized;
    }),
  );
  const payload = {
    ...raw,
    objects,
    issues: (raw.issues ?? []).map((issue, index) => ({
      ...issue,
      id: issue.id ?? `issue-${index}`,
      message: issue.message ?? issue.evidence ?? "需要人工核对",
      region: issue.region ?? issue.bbox,
    })),
  };
  for (const layer of ["roi", "ignore"] as const) {
    const reference = raw[`${layer}_mask_ref`];
    if (!payload[layer] && typeof reference === "string")
      payload[layer] = (await visionApi.mask(reference)).rle;
  }
  return payload;
}

function annotationsFromObjects(objects: ReviewObject[]) {
  return objects
    .filter((object) => object.decision !== "rejected")
    .map((object) => ({
      object_id: object.object_id,
      class_id: object.class_id,
      type: object.type,
      bbox: object.bbox,
      polygon: object.polygon,
      obb: object.obb,
      mask_ref: object.mask_ref,
      source: object.source ?? "manual",
      provenance_json: { ...object.provenance_json, ...object.provenance },
    }));
}

export function ArtifactReviewView({
  reviewId,
  profile,
  onBack,
  onSaved,
  onDirtyChange,
}: {
  reviewId: string;
  profile: TaskProfileConfig;
  onBack: () => void;
  onSaved: () => void;
  onDirtyChange?: (dirty: boolean) => void;
}) {
  const [review, setReview] = useState<WorkflowReview>();
  const [payload, setPayload] = useState<ReviewPayload>();
  const [selectedId, setSelectedId] = useState("");
  const [focusRegion, setFocusRegion] = useState<BBox>();
  const [classId, setClassId] = useState(
    profile.class_rules[0]?.class_id ?? "",
  );
  const [history, setHistory] = useState<ReviewPayload[]>([]);
  const [future, setFuture] = useState<ReviewPayload[]>([]);
  const [newObjectRequest, setNewObjectRequest] = useState(0);
  const [pendingDrawing, setPendingDrawing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState<ReactNode>("");
  const [coverage, setCoverage] = useState(false);
  const [reason, setReason] = useState("");
  const savedFingerprint = useRef("");
  const currentPayload = useRef(payload);
  const mounted = useRef(true);
  currentPayload.current = payload;
  const dirty =
    pendingDrawing ||
    (!!payload && JSON.stringify(payload) !== savedFingerprint.current);
  const { requestLeave, dialog } = useLeaveConfirm(dirty);
  useEffect(() => onDirtyChange?.(dirty), [dirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange]);
  const reportError = useCallback((message: string) => setError(message), []);
  useEffect(() => {
    mounted.current = true;
    workflowApi
      .review(reviewId)
      .then(async (result) => {
        const loaded = await hydratePayload(result.payload);
        if (!mounted.current) return;
        setReview(result);
        setPayload(loaded);
        savedFingerprint.current = JSON.stringify(loaded);
        setSelectedId(loaded.objects[0]?.object_id ?? "");
      })
      .catch((failure) => {
        if (mounted.current)
          setError(failure instanceof Error ? failure.message : "复核读取失败");
      });
    return () => {
      mounted.current = false;
    };
  }, [reviewId]);

  useEffect(() => {
    if (!payload) return;
    if (payload.objects.some((object) =>
      object.object_id === selectedId && object.decision !== "rejected")) return;
    const next = payload.objects.find((object) => object.decision !== "rejected");
    setSelectedId(next?.object_id ?? "");
    if (next?.class_id) setClassId(next.class_id);
  }, [payload?.objects, selectedId]);

  const change = (next: ReviewPayload) => {
    if (!payload) return;
    setHistory((items) => [...items.slice(-19), payload]);
    setFuture([]);
    setPayload(next);
    setCoverage(false);
    setNotice("当前修订已修改，机器检查结论需以保存后的版本为准。");
  };
  const undo = () => {
    if (!payload || pendingDrawing || !history.length) return;
    setFuture((items) => [payload, ...items]);
    setPayload(history[history.length - 1]);
    setHistory((items) => items.slice(0, -1));
    setCoverage(false);
  };
  const redo = () => {
    if (!payload || pendingDrawing || !future.length) return;
    setHistory((items) => [...items, payload]);
    setPayload(future[0]);
    setFuture((items) => items.slice(1));
    setCoverage(false);
  };
  const savePayload = async (): Promise<WorkflowReview> => {
    if (!payload || !review) throw new Error("复核尚未载入");
    if (pendingDrawing) throw new Error("请先完成轮廓或取消本次绘制。");
    const objects = await Promise.all(
      payload.objects.map(async (object) =>
        object.mask && !object.mask_ref
          ? {
              ...object,
              mask_ref: (await visionApi.saveMask(object.mask)).mask_ref,
            }
          : object,
      ),
    );
    const prepared: ReviewPayload = {
      ...payload,
      objects,
      annotations: annotationsFromObjects(objects),
    };
    for (const layer of ["roi", "ignore"] as const)
      prepared[`${layer}_mask_ref`] = prepared[layer]
        ? (await visionApi.saveMask(prepared[layer]!)).mask_ref
        : null;
    const saved = await workflowApi.saveDraft(
      reviewId,
      review.revision,
      prepared,
    );
    const hydrated = await hydratePayload(saved.payload);
    if (!mounted.current) return saved;
    setReview(saved);
    setPayload(hydrated);
    savedFingerprint.current = JSON.stringify(hydrated);
    setHistory([]);
    setFuture([]);
    onSaved();
    return saved;
  };
  const save = async () => {
    setBusy(true);
    setSaving(true);
    setError("");
    try {
      await savePayload();
      if (mounted.current) setNotice("复核草稿已保存，正式标签待整图确认。");
    } catch (failure) {
      if (mounted.current)
        setError(
          failure instanceof Error
            ? failure.message
            : "保存失败，草稿保留在当前页面",
        );
    } finally {
      if (mounted.current) {
        setBusy(false);
        setSaving(false);
      }
    }
  };
  const decide = async (
    decision: "approved" | "background" | "rejected" | "excluded",
  ) => {
    if (!review || !payload || pendingDrawing) return;
    setBusy(true);
    setError("");
    try {
      const saved = dirty ? await savePayload() : review;
      const semanticResolutions = Object.fromEntries(
        payload.issues.map((issue, index) => [
          issue.id ?? String(index),
          { resolution: issue.resolution, reason: issue.reason },
        ]),
      );
      const result = await workflowApi.decide(
        reviewId,
        saved.revision,
        decision,
        reason,
        semanticResolutions,
      );
      if (!mounted.current) return;
      setReview(result);
      setNotice(
        decision === "approved" || decision === "background"
          ? "当前图片已确认。批次关卡会等待其余纳入项完成。"
          : "人工决定已保存。",
      );
      onSaved();
    } catch (failure) {
      if (mounted.current)
        setError(
          failure instanceof Error ? failure.message : "确认失败，草稿保留",
        );
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  const sam = async (prompt: { box: BBox } | { points: PromptPoint[] }) => {
    if (!payload || pendingDrawing) return;
    const startingPayload = payload;
    const startingObject = payload.objects.find((object) => object.object_id === selectedId);
    const targetClassId = startingObject ? startingObject.class_id : classId;
    if (!targetClassId || !profile.class_rules.some((rule) => rule.class_id === targetClassId)) {
      setError("请先为当前对象选择本次任务的类别。");
      return;
    }
    if (startingObject?.decision === "rejected") {
      setError("请先恢复接受该对象或选择其他对象。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const suggestion = await visionApi.sam({
        image_id: payload.image_id,
        class_id: targetClassId,
        prompt_type: "box" in prompt ? "box" : "point",
        ...prompt,
      });
      const mask = (await visionApi.mask(suggestion.mask_ref)).rle;
      if (!mask) throw new Error("SAM 未返回原始掩膜，无法应用建议");
      if (!decodeMask(mask, payload.width, payload.height).some((value) => value > 0))
        throw new Error("SAM 返回了空轮廓，请调整提示后重试。");
      if (
        !mounted.current ||
        currentPayload.current !== startingPayload ||
        suggestion.image_id !== payload.image_id
      )
        return;
      const object: ReviewObject = {
        ...startingObject,
        object_id: startingObject?.object_id ?? crypto.randomUUID(),
        class_id: targetClassId,
        type: "polygon",
        mask,
        mask_ref: suggestion.mask_ref,
        bbox: undefined,
        polygon: undefined,
        obb: undefined,
        source: "sam",
        decision: "accepted",
        provenance: {
          ...startingObject?.provenance,
          sam_refinement: {
            backend: suggestion.backend_used,
            score: suggestion.score,
            device: suggestion.device,
            prompt,
          },
        },
      };
      change({
        ...payload,
        objects: payload.objects.some(
          (item) => item.object_id === object.object_id,
        )
          ? payload.objects.map((item) =>
              item.object_id === object.object_id ? object : item,
            )
          : [...payload.objects, object],
      });
      setSelectedId(object.object_id);
    } catch (failure) {
      if (mounted.current)
        setError(failure instanceof Error ? failure.message : "SAM 请求失败");
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  const readOnly =
    !!review &&
    (["approved", "background", "excluded"].includes(review.status) ||
      (review.status === "decided" && review.decision !== "rejected"));
  const editLocked = busy || readOnly || pendingDrawing;
  const confirmed =
    !!review &&
    !dirty &&
    (["approved", "background"].includes(review.status) ||
      (review.status === "decided" &&
        ["approved", "background"].includes(review.decision ?? "")));
  const supportsBoxMaskExport =
    profile.task_type === "detect" || profile.task_type === "segment";
  const accepted =
    payload?.objects.filter((object) => object.decision !== "rejected") ?? [];
  const unmappedObjects =
    payload?.objects.filter(
      (object) =>
        object.decision !== "rejected" &&
        !profile.class_rules.some((rule) => rule.class_id === object.class_id),
    ) ?? [];
  const selected = payload?.objects.find(
    (object) => object.object_id === selectedId,
  );
  const blockingIssues =
    payload?.issues.filter(
      (issue) => issue.blocking || issue.severity === "error",
    ) ?? [];
  const blocking = blockingIssues.length > 0;
  const unresolvedIssues =
    payload?.issues.filter(
      (issue) =>
        !issue.blocking &&
        issue.severity !== "error" &&
        (!issue.resolution || !issue.reason?.trim()),
    ) ?? [];
  const unresolved = unresolvedIssues.length > 0;
  const classes: ClassLabel[] = profile.class_rules.map((rule, index) => ({
    id: rule.class_id,
    name: rule.name,
    dataset_id: payload?.dataset_id ?? "",
    class_index: index,
    color: rule.color ?? "#22c55e",
    created_at: "",
    updated_at: "",
  }));
  const annotations: Annotation[] = accepted.map((object) => ({
    id: object.object_id,
    image_id: payload?.image_id ?? "",
    dataset_id: payload?.dataset_id ?? "",
    class_id: object.class_id ?? "",
    class_index: classes.findIndex((item) => item.id === object.class_id),
    label:
      classes.find((item) => item.id === object.class_id)?.name ?? "未映射类别",
    color:
      classes.find((item) => item.id === object.class_id)?.color ?? "#ef4444",
    type: object.type,
    bbox: object.bbox ?? null,
    polygon: object.polygon ?? null,
    obb: object.obb ?? null,
    source: object.source ?? "manual",
    created_at: "",
    updated_at: "",
  }));
  const changeGeometry = (drafts: AnnotationDraft[]) => {
    if (!payload) return;
    const objects = drafts.map((draft) => ({
      ...payload.objects.find((object) => object.object_id === draft.id),
      ...draft,
      object_id: draft.id ?? crypto.randomUUID(),
      decision: "accepted" as const,
    }));
    change({
      ...payload,
      objects: [
        ...objects,
        ...payload.objects.filter((object) => object.decision === "rejected"),
      ],
    });
  };
  const addMask = () => {
    if (!payload || !classId || editLocked) return;
    setFocusRegion(undefined);
    setError("");
    setNewObjectRequest((value) => value + 1);
    setNotice("请在图片中逐点勾勒漏标对象，完成轮廓后加入标注。");
  };
  const rejectUnmapped = () => {
    if (!payload || editLocked || !unmappedObjects.length) return;
    const rejectedIds = new Set(
      unmappedObjects.map((object) => object.object_id),
    );
    change({
      ...payload,
      objects: payload.objects.map((object) =>
        rejectedIds.has(object.object_id)
          ? { ...object, decision: "rejected" as const }
          : object,
      ),
    });
    setNotice(
      `已在当前草稿中拒绝 ${unmappedObjects.length} 个未归类候选。可点击“撤销”恢复；点击“保存复核草稿”后会重新检查。`,
    );
  };
  const applyIssueConclusions = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!payload || editLocked || !unresolvedIssues.length) return;
    const form = new FormData(event.currentTarget);
    const resolution = String(form.get("resolution") ?? "");
    const explanation = String(form.get("explanation") ?? "").trim();
    if (!["resolved", "dismissed"].includes(resolution) || !explanation) {
      setError("请选择批量检查结论并填写你核对后的理由。");
      return;
    }
    const targets = new Set(unresolvedIssues);
    change({
      ...payload,
      issues: payload.issues.map((issue) =>
        targets.has(issue)
          ? { ...issue, resolution, reason: explanation }
          : issue,
      ),
    });
    setError("");
    setNotice(
      `已为列出的 ${unresolvedIssues.length} 项填写检查结论。可继续逐项修改，再保存复核草稿。`,
    );
  };
  const savePreviewImage = async () => {
    if (!payload || !review || pendingDrawing) return;
    const snapshot = payload;
    setBusy(true);
    setError("");
    try {
      const blob = await annotationPreview(snapshot, profile.class_rules);
      if (!mounted.current || currentPayload.current !== snapshot) return;
      const saved = await workflowApi.savePreview(
        reviewId,
        review.revision,
        blob,
        dirty,
      );
      if (!mounted.current) return;
      const changed = currentPayload.current !== snapshot;
      const url = apiUrl(saved.download_url);
      setNotice(
        <span>
          {changed
            ? "此前版本的预览文件已保存，当前草稿仍可继续编辑。"
            : "预览 PNG 已保存。"}
          <br />
          保存位置{" "}
          <code style={{ overflowWrap: "anywhere" }}>
            {saved.saved_path}
          </code>{" "}
          <a
            className="button"
            href={url}
            download={saved.file_name}
            target="_blank"
            rel="noopener noreferrer"
          >
            下载 PNG
          </a>
        </span>,
      );
    } catch (failure) {
      if (mounted.current)
        setError(
          failure instanceof Error
            ? failure.message
            : "标注预览保存失败，请重试。",
        );
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  const exportLabelMe = async () => {
    if (!payload || !review || pendingDrawing) return;
    const snapshot = payload;
    setBusy(true);
    setError("");
    try {
      const saved = await workflowApi.exportLabelMe(
        reviewId,
        review.revision,
        snapshot,
      );
      if (!mounted.current) return;
      const changed = currentPayload.current !== snapshot;
      setNotice(
        <span>
          {changed
            ? "此前版本的 LabelMe JSON 已导出，当前草稿可继续编辑。"
            : "LabelMe JSON 已导出，可在 LabelMe 中打开并继续编辑。"}{" "}
          {saved.export_state === "confirmed" ? "已确认标注" : "草稿导出"}，导出{" "}
          {saved.exported_objects} 个对象，另有 {saved.pending_objects}{" "}
          个对象待处理。
          <br />
          保存位置{" "}
          <code style={{ overflowWrap: "anywhere" }}>
            {saved.saved_path}
          </code>{" "}
          <a
            className="button"
            href={apiUrl(saved.download_url)}
            download={saved.file_name}
            target="_blank"
            rel="noopener noreferrer"
          >
            下载 JSON
          </a>
        </span>,
      );
    } catch (failure) {
      if (mounted.current)
        setError(
          failure instanceof Error
            ? failure.message
            : "LabelMe JSON 导出失败，请重试。",
        );
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  return (
    <main className="workflow-page workflow-review-page">
      {dialog}
      <header className="workflow-heading">
        <div>
          <button className="button" onClick={() => requestLeave(onBack)}>
            ← 返回运行
          </button>
          <h1>图像复核</h1>
          <p>
            {payload?.image_id ?? reviewId} · 修订 {review?.revision ?? "…"} ·{" "}
            {readOnly ? "已提交人工决定" : dirty ? "未保存修改" : "草稿已保存"}
          </p>
        </div>
        <div className="workflow-actions">
          {supportsBoxMaskExport && (
            <button
              className="button primary"
              disabled={busy || pendingDrawing || !payload}
              onClick={() => void exportLabelMe()}
            >
              导出 LabelMe JSON
            </button>
          )}
          <button
            className="button"
            disabled={busy || pendingDrawing || !payload}
            onClick={() => void savePreviewImage()}
          >
            保存标注预览 PNG
          </button>
          {supportsBoxMaskExport &&
            (confirmed && payload ? (
              <a
                className="button"
                href={api.exportCocoUrl(payload.dataset_id, [payload.image_id])}
              >
                下载本图 COCO 标注
              </a>
            ) : (
              <button
                className="button"
                disabled
                title="先完成整图确认，再下载原图和正式标注数据。"
              >
                下载本图 COCO 标注
              </button>
            ))}
          <button
            className="button"
            disabled={editLocked || !history.length}
            onClick={undo}
          >
            撤销
          </button>
          <button
            className="button"
            disabled={editLocked || !future.length}
            onClick={redo}
          >
            重做
          </button>
          <button
            className="button primary"
            disabled={editLocked || !payload || !dirty}
            onClick={() => void save()}
          >
            保存复核草稿
          </button>
        </div>
      </header>
      {error && (
        <p className="validation invalid" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className="validation" role="status">
          {notice}
        </p>
      )}
      {pendingDrawing && (
        <p className="validation" role="status">
          请先完成轮廓或取消本次绘制，再保存、导出或确认整图。
        </p>
      )}
      {payload && review && (
        <>
          <section className="panel workflow-form">
            {supportsBoxMaskExport && (
              <p>
                继续在外部编辑时，可直接点击“导出 LabelMe JSON”。JSON
                包含原图，能够在 LabelMe 中打开。JSON
                草稿可直接导出，正式应用标注需处理剩余问题并完成整图确认。
              </p>
            )}
            <p>
              本次识别 {profile.class_rules.length} 个类别（
              {profile.class_rules.map((rule) => rule.name).join("、")}
              ）。同一类别可以有多个对象，每个人物或物体分别形成一个候选。
            </p>
            <p>
              保存带标注的图片可点击“保存标注预览 PNG”，文件保持原图{" "}
              {payload.width} × {payload.height}{" "}
              像素并包含当前草稿。保存后会显示本地文件位置，需要下载副本时点击“下载
              PNG”。
            </p>
            <p>
              保存标注数据时，先修改并“保存复核草稿”，逐项填写检查意见的裁决和理由，再勾选整图检查并“确认整图并应用”。
            </p>
            {supportsBoxMaskExport && (
              <p>
                整图确认后可点击“下载本图 COCO
                标注”，保存这张原图及标注数据，分割掩膜包含孔洞和分离区域。流程预设格式的批量导出在返回运行页面后点击“下载已确认导出”。
              </p>
            )}
            {!supportsBoxMaskExport && (
              <p>
                返回运行页面，流程完成后点击“下载已确认导出”，保存流程预设格式的标注数据。
              </p>
            )}
            {unmappedObjects.length > 0 && (
              <div>
                <p>
                  有 {unmappedObjects.length} 个候选尚未匹配本次类别（
                  {unmappedObjects
                    .map((object) => `#${payload.objects.indexOf(object) + 1}`)
                    .join("、")}
                  ）。属于目标的候选请选择类别，范围外的候选可拒绝后再保存草稿重新检查。
                </p>
                <button
                  className="button"
                  disabled={editLocked}
                  onClick={rejectUnmapped}
                >
                  拒绝这 {unmappedObjects.length} 个未归类候选
                </button>
              </div>
            )}
            {review.project && review.export_locations ? (
              <section className="workflow-form-section">
                <h3>本次运行的项目保存位置</h3>
                <p>
                  {review.project.name} · 根目录 {review.project.root_path}
                </p>
                <p>单图与批量导出会自动使用本次运行选定的项目目录。</p>
                <dl className="workflow-launch-summary">
                  <div>
                    <dt>LabelMe JSON</dt>
                    <dd><code>{review.export_locations.json_path}</code></dd>
                  </div>
                  <div>
                    <dt>标注预览图 PNG</dt>
                    <dd><code>{review.export_locations.images_path}</code></dd>
                  </div>
                </dl>
              </section>
            ) : (
              <details>
                <summary>设置 JSON 和预览图保存位置</summary>
                <ExportLocationsPanel />
              </details>
            )}
          </section>
          <div className="workflow-review-grid">
            <div className="workflow-review-main">
              <section className="workflow-review-savebar" aria-label="保存补充标注">
                <div>
                  <strong>保存补充标注</strong>
                  <p role="status" aria-live="polite">
                    {saving
                      ? "正在保存修改…"
                      : readOnly
                        ? "人工决定已保存"
                        : pendingDrawing
                          ? "请先完成轮廓或取消本次绘制"
                          : dirty
                            ? "有未保存的补画或修改"
                            : `当前修改已保存至工具 · 修订 ${review.revision}`}
                  </p>
                </div>
                <button
                  className="button primary"
                  disabled={editLocked || !dirty}
                  onClick={() => void save()}
                >
                  {saving ? "正在保存…" : "保存修改"}
                </button>
                <p className="workflow-review-save-hint">保存后，可从数据集的“继续 LLM 标注复核”重新打开。用于训练前，请完成整图确认。</p>
                {error && <p className="workflow-review-save-error">{error}</p>}
              </section>
              <div className="workflow-actions">
                <label>
                  当前绘制类别
                  <select
                    value={classId}
                    disabled={editLocked}
                    onChange={(event) => setClassId(event.target.value)}
                  >
                    <option value="" disabled>
                      请选择绘制类别
                    </option>
                    {classes.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
                {profile.task_type === "segment" && (
                  <button
                    className="button"
                    disabled={editLocked || !classId}
                    onClick={addMask}
                  >
                    新增漏标对象
                  </button>
                )}
              </div>
              {profile.task_type === "segment" && (
                <p className="hint">
                  补漏时点击“新增漏标对象”，沿边界逐点绘制，再点击“完成轮廓”。
                  修正已有对象时先选中它，再用多边形“补全选中对象”“擦除选中区域”或画笔、橡皮修边。
                  有可编辑顶点的多边形还可拖动顶点调整。SAM 会根据提示重新生成选中对象的轮廓。
                  完成后保存复核草稿，再导出结果。
                </p>
              )}
              {profile.task_type === "segment" ? (
                <MaskReviewCanvas
                  activeClassId={classId}
                  newObjectRequest={newObjectRequest}
                  onDrawingChange={setPendingDrawing}
                  onUndo={undo}
                  onRedo={redo}
                  canUndo={history.length > 0}
                  canRedo={future.length > 0}
                  focusRegion={focusRegion}
                  payload={payload}
                  classes={profile.class_rules}
                  selectedId={selectedId}
                  onSelect={(id) => {
                    setFocusRegion(undefined);
                    setSelectedId(id);
                    const object = payload.objects.find(
                      (item) => item.object_id === id,
                    );
                    if (object) setClassId(object.class_id ?? "");
                  }}
                  onChange={change}
                  onSamBox={(box) => void sam({ box })}
                  onSamPoints={(points) => void sam({ points })}
                  disabled={busy || readOnly}
                  onError={reportError}
                />
              ) : profile.task_type === "classify" ? (
                <section className="panel workflow-form">
                  <img
                    className="workflow-source-image"
                    src={
                      payload.image_url.startsWith("http")
                        ? payload.image_url
                        : apiUrl(payload.image_url)
                    }
                    alt="待分类图像"
                  />
                  <div className="workflow-actions">
                    {classes.map((item) => (
                      <button
                        key={item.id}
                        disabled={editLocked}
                        className={`button${accepted[0]?.class_id === item.id ? " active" : ""}`}
                        onClick={() =>
                          change({
                            ...payload,
                            objects: [
                              {
                                object_id:
                                  accepted[0]?.object_id ?? crypto.randomUUID(),
                                class_id: item.id,
                                type: "classify",
                                decision: "accepted",
                                source: "manual",
                              },
                            ],
                          })
                        }
                      >
                        {item.name}
                      </button>
                    ))}
                  </div>
                </section>
              ) : (
                <div
                  style={{ pointerEvents: busy || readOnly ? "none" : "auto" }}
                >
                  <AnnotationCanvas
                    focusRegion={focusRegion}
                    image={
                      {
                        id: payload.image_id,
                        dataset_id: payload.dataset_id,
                        file_name: payload.image_id,
                        width: payload.width,
                        height: payload.height,
                        file_url: payload.image_url,
                        split: "train",
                        status: "unannotated",
                        created_at: "",
                        updated_at: "",
                      } satisfies ImageItem
                    }
                    classes={classes}
                    annotations={annotations}
                    activeClassId={classId}
                    selectedAnnotationId={selectedId}
                    onSelectAnnotation={(id) => {
                      setFocusRegion(undefined);
                      setSelectedId(id ?? "");
                    }}
                    taskType={profile.task_type}
                    onChange={changeGeometry}
                  />
                </div>
              )}
              <ObjectCrop
                payload={payload}
                object={selected}
                region={focusRegion}
              />
            </div>
            <aside className="panel workflow-form workflow-review-sidebar">
              <h2>对象与来源 · {accepted.length} 个候选对象</h2>
              {payload.objects.map((object, index) => (
                <article
                  key={object.object_id}
                  className={`workflow-object${selectedId === object.object_id ? " selected" : ""}`}
                >
                  <button
                    className="button"
                    disabled={busy || pendingDrawing}
                    onClick={() => {
                      setFocusRegion(undefined);
                      setSelectedId(object.object_id);
                      setClassId(object.class_id ?? "");
                    }}
                  >
                    #{index + 1} · {object.type} ·{" "}
                    {object.decision === "rejected" ? "已拒绝" : "候选"}
                  </button>
                  <select
                    aria-label={`对象 ${index + 1} 类别`}
                    disabled={editLocked}
                    value={object.class_id ?? ""}
                    onChange={(event) =>
                      change({
                        ...payload,
                        objects: payload.objects.map((item) =>
                          item.object_id === object.object_id
                            ? { ...item, class_id: event.target.value }
                            : item,
                        ),
                      })
                    }
                  >
                    <option value="" disabled>
                      待确定类别
                    </option>
                    {classes.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                  <button
                    className="button"
                    disabled={editLocked}
                    onClick={() =>
                      change({
                        ...payload,
                        objects: payload.objects.map((item) =>
                          item.object_id === object.object_id
                            ? {
                                ...item,
                                decision:
                                  item.decision === "rejected"
                                    ? "accepted"
                                    : "rejected",
                              }
                            : item,
                        ),
                      })
                    }
                  >
                    {object.decision === "rejected" ? "恢复接受" : "拒绝对象"}
                  </button>
                  <details>
                    <summary>来源与几何</summary>
                    <pre>
                      {JSON.stringify(
                        {
                          ...object,
                          mask: object.mask
                            ? { size: object.mask.size, encoding: "COCO RLE" }
                            : undefined,
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </article>
              ))}
              {!payload.objects.length && (
                <p>没有候选仍需完整检查原图，补充漏标或明确确认无目标。</p>
              )}
              <h2>检查问题</h2>
              {!readOnly && unresolvedIssues.length > 0 && (
                <details>
                  <summary>
                    批量填写检查结论（{unresolvedIssues.length} 项）
                  </summary>
                  <p>
                    请先核对以下各项，再选择共同结论并填写你实际核对的理由。
                  </p>
                  <ol>
                    {unresolvedIssues.map((issue) => (
                      <li key={issue.id}>
                        检查项 #{payload.issues.indexOf(issue) + 1} ·{" "}
                        {issueLabels[issue.type ?? ""] ?? "需要核对"}
                        <p>{issue.message}</p>
                      </li>
                    ))}
                  </ol>
                  <form onSubmit={applyIssueConclusions}>
                    <label>
                      批量检查结论
                      <select
                        name="resolution"
                        defaultValue=""
                        required
                        disabled={busy || pendingDrawing}
                      >
                        <option value="" disabled>
                          请选择共同结论
                        </option>
                        <option value="resolved">已核对并修正</option>
                        <option value="dismissed">核对后认为可接受</option>
                      </select>
                    </label>
                    <label>
                      本次共同理由
                      <textarea
                        name="explanation"
                        required
                        disabled={busy || pendingDrawing}
                        placeholder="填写你检查这些问题后的依据与处理情况"
                      />
                    </label>
                    <button className="button" type="submit" disabled={busy || pendingDrawing}>
                      应用到这 {unresolvedIssues.length} 项
                    </button>
                  </form>
                </details>
              )}
              {payload.issues.map((issue, index) => {
                const unmappedClass =
                  issue.type === "unmapped_class" ||
                  issue.message ===
                    "Candidate class is not mapped to the task profile.";
                return (
                  <article className="workflow-issue" key={issue.id ?? index}>
                    <button
                      className="button"
                      disabled={busy || pendingDrawing || (!issue.object_id && !issue.region)}
                      onClick={() => {
                        setFocusRegion(issue.region);
                        if (issue.object_id) setSelectedId(issue.object_id);
                      }}
                    >
                      {unmappedClass
                        ? "类别待确定"
                        : (issueLabels[issue.type ?? ""] ??
                          issue.type ??
                          issue.code ??
                          "检查项")}
                      {issue.blocking ? " · 需处理后确认" : ""}
                    </button>
                    <p>
                      {unmappedClass
                        ? "该候选尚未匹配到任务类别。请选择已配置类别，或拒绝该候选后保存草稿。"
                        : issue.message}
                    </p>
                    {issue.type === "coordinate_error" && (
                      <details>
                        <summary>查看坐标诊断信息</summary>
                        <pre>{JSON.stringify(issue, null, 2)}</pre>
                      </details>
                    )}
                    {issue.region && (
                      <p>
                        区域 {Math.round(issue.region.x)},{" "}
                        {Math.round(issue.region.y)} ·{" "}
                        {Math.round(issue.region.width)} ×{" "}
                        {Math.round(issue.region.height)}
                      </p>
                    )}
                    {!issue.blocking && issue.severity !== "error" && (
                      <>
                        <select
                          aria-label={`问题 ${index + 1} 裁决`}
                          disabled={editLocked}
                          value={issue.resolution ?? ""}
                          onChange={(event) =>
                            change({
                              ...payload,
                              issues: payload.issues.map((item, itemIndex) =>
                                itemIndex === index
                                  ? { ...item, resolution: event.target.value }
                                  : item,
                              ),
                            })
                          }
                        >
                          <option value="">待人工裁决</option>
                          <option value="resolved">已核对并修正</option>
                          <option value="dismissed">核对后认为可接受</option>
                        </select>
                        <textarea
                          aria-label={`问题 ${index + 1} 理由`}
                          disabled={editLocked}
                          placeholder="当前修订的证据和理由"
                          value={issue.reason ?? ""}
                          onChange={(event) =>
                            change({
                              ...payload,
                              issues: payload.issues.map((item, itemIndex) =>
                                itemIndex === index
                                  ? { ...item, reason: event.target.value }
                                  : item,
                              ),
                            })
                          }
                        />
                      </>
                    )}
                  </article>
                );
              })}
              {!payload.issues.length && (
                <p>当前未列出问题，仍需人工检查整图覆盖。</p>
              )}
            </aside>
          </div>
          {!readOnly && (
            <section className="panel workflow-form">
              <h2>整图决定</h2>
              <label className="workflow-check">
                <input
                  type="checkbox"
                  checked={coverage}
                  disabled={busy || pendingDrawing}
                  onChange={(event) => setCoverage(event.target.checked)}
                />
                我已检查全图、对象类别与边界、漏标和有效监督范围，并完成当前修订的问题裁决
              </label>
              <label>
                复核说明或排除理由
                <textarea
                  value={reason}
                  disabled={busy || pendingDrawing}
                  onChange={(event) => setReason(event.target.value)}
                />
              </label>
              {blocking && (
                <div className="validation invalid">
                  <p>
                    以下 {blockingIssues.length}{" "}
                    项需要修正后才能确认整图。修改或拒绝对应对象，再点击“保存复核草稿”，系统会重新检查。
                  </p>
                  <ul>
                    {blockingIssues.map((issue, index) => {
                      const objectIndex = payload.objects.findIndex(
                        (object) => object.object_id === issue.object_id,
                      );
                      const unmapped =
                        issue.type === "unmapped_class" ||
                        issue.message ===
                          "Candidate class is not mapped to the task profile.";
                      return (
                        <li key={issue.id ?? index}>
                          {objectIndex >= 0
                            ? `对象 #${objectIndex + 1}`
                            : "当前标注"}{" "}
                          —{" "}
                          {unmapped
                            ? "类别尚未确定。请选择本次类别，或对范围外的候选点击“拒绝对象”后保存草稿。"
                            : issue.message ||
                              issue.evidence ||
                              issueLabels[issue.type ?? ""] ||
                              "请核对标注格式。"}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
              {unresolved && <p>请填写每个语义问题的裁决和理由。</p>}
              {supportsBoxMaskExport && (
                <div>
                  <p>也可以先导出当前 LabelMe JSON 草稿，在外部继续编辑。</p>
                  <button
                    className="button"
                    disabled={busy || pendingDrawing || !payload}
                    onClick={() => void exportLabelMe()}
                  >
                    直接导出 LabelMe JSON 草稿
                  </button>
                </div>
              )}
              <div className="workflow-actions">
                <button
                  className="button primary"
                  disabled={
                    busy || pendingDrawing ||
                    !coverage ||
                    blocking ||
                    unresolved ||
                    !accepted.length
                  }
                  onClick={() => void decide("approved")}
                >
                  确认整图并应用
                </button>
                {profile.task_type !== "classify" && (
                  <button
                    className="button"
                    disabled={
                      busy || pendingDrawing ||
                      !coverage ||
                      blocking ||
                      unresolved ||
                      !!accepted.length
                    }
                    onClick={() => void decide("background")}
                  >
                    确认无目标
                  </button>
                )}
                <button
                  className="button"
                  disabled={busy || pendingDrawing || !reason.trim()}
                  onClick={() => void decide("rejected")}
                >
                  拒绝并保留阻塞
                </button>
                <button
                  className="button danger"
                  disabled={busy || pendingDrawing || !reason.trim()}
                  onClick={() => void decide("excluded")}
                >
                  按理由排除此图
                </button>
              </div>
              <p className="hint">
                应用正式标签与审核回执在同一事务完成。单图确认后，批次仍等待其余纳入图片。
              </p>
            </section>
          )}
        </>
      )}
    </main>
  );
}

function ObjectCrop({
  payload,
  object,
  region,
}: {
  payload: ReviewPayload;
  object?: ReviewObject;
  region?: BBox;
}) {
  let box = region ?? object?.bbox;
  if (!box && object?.obb) {
    const corners = obbCorners(object.obb);
    const xs = corners.map(([x]) => x),
      ys = corners.map(([, y]) => y);
    box = {
      x: Math.min(...xs),
      y: Math.min(...ys),
      width: Math.max(...xs) - Math.min(...xs),
      height: Math.max(...ys) - Math.min(...ys),
    };
  }
  if (!box && object?.mask) {
    const pixels = decodeMask(object.mask, payload.width, payload.height);
    let minX = payload.width,
      minY = payload.height,
      maxX = -1,
      maxY = -1;
    pixels.forEach((value, index) => {
      if (value) {
        const x = index % payload.width;
        const y = Math.floor(index / payload.width);
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    });
    if (maxX >= minX)
      box = {
        x: minX,
        y: minY,
        width: maxX - minX + 1,
        height: maxY - minY + 1,
      };
  }
  if (!box || !box.width || !box.height) return null;
  const padding = Math.max(8, Math.min(box.width, box.height) * 0.15);
  const imageUrl = payload.image_url.startsWith("http")
    ? payload.image_url
    : apiUrl(payload.image_url);
  return (
    <section className="panel workflow-form">
      <h3>选中对象原图 crop</h3>
      <svg
        className="workflow-crop"
        viewBox={`${box.x - padding} ${box.y - padding} ${box.width + padding * 2} ${box.height + padding * 2}`}
        role="img"
        aria-label="选中对象裁剪"
      >
        <image href={imageUrl} width={payload.width} height={payload.height} />
        <rect
          {...box}
          fill="none"
          stroke="#f59e0b"
          strokeWidth={Math.max(1, box.width / 200)}
        />
      </svg>
    </section>
  );
}
