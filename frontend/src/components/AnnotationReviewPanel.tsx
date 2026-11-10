import { useState } from "react";
import { reviewApi, type ImageReviewDetail } from "../api/reviews";
import type { Annotation, TaskType } from "../types";

export function AnnotationReviewPanel({
  datasetId,
  imageId,
  taskType,
  dirty,
  onLoadDraft,
}: {
  datasetId: string;
  imageId: string;
  taskType: TaskType;
  dirty: boolean;
  onLoadDraft: (annotations: Annotation[]) => void;
}) {
  const [detail, setDetail] = useState<ImageReviewDetail>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [coverage, setCoverage] = useState(false);
  const [reason, setReason] = useState("");
  const [message, setMessage] = useState("");
  const load = async () => {
    setBusy(true);
    setError("");
    try {
      setDetail(await reviewApi.image(datasetId, imageId));
      setCoverage(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "审核读取失败");
    } finally {
      setBusy(false);
    }
  };
  const confirm = async () => {
    if (!detail) return;
    setBusy(true);
    setError("");
    try {
      await reviewApi.confirm(
        datasetId,
        imageId,
        detail.revision,
        detail.fingerprint,
        !detail.annotations.length,
        reason,
      );
      setMessage("当前整图标签已确认。再次保存标注会使这份确认失效。");
      await load();
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : "确认失败，需重新核对版本",
      );
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="panel workflow-form">
      <div className="workflow-heading">
        <div>
          <h2>人工审核</h2>
          <p>保存标注后核对整图，确认回执用于训练和发布导出。</p>
        </div>
        <button
          className="button"
          disabled={busy || dirty}
          onClick={() => void load()}
        >
          {busy ? "正在读取…" : "查看审核状态"}
        </button>
      </div>
      {dirty && <p>有未保存标注，请先保存再确认整图。</p>}
      {error && (
        <p role="alert" className="validation invalid">
          {error}
        </p>
      )}
      {message && <p role="status">{message}</p>}
      {detail && (
        <>
          <p>
            版本 {detail.revision} · {detail.review_status} ·{" "}
            {detail.annotations.length} 个正式标注
          </p>
          {detail.draft?.annotations && (
            <div className="validation">
              <p>
                存在 {detail.draft.annotations.length}{" "}
                个待审自动候选，载入后在画布核对并保存。
              </p>
              <button
                className="button"
                disabled={busy || dirty}
                onClick={() => {
                  onLoadDraft(detail.draft!.annotations!);
                  setDetail(undefined);
                }}
              >
                载入待审候选
              </button>
            </div>
          )}
          {detail.issues?.map((issue, index) => (
            <p
              key={index}
              className={issue.blocking ? "validation invalid" : "hint"}
            >
              {issue.message}
            </p>
          ))}
          <label className="workflow-check">
            <input
              type="checkbox"
              checked={coverage}
              disabled={dirty || busy}
              onChange={(event) => setCoverage(event.target.checked)}
            />
            我已核对当前保存版本的全部类别、对象、边界与漏标
          </label>
          <label>
            审核说明
            <textarea
              value={reason}
              onChange={(event) => setReason(event.target.value)}
            />
          </label>
          <button
            className="button primary"
            disabled={
              dirty ||
              busy ||
              !coverage ||
              !!detail.issues?.some((issue) => issue.blocking) ||
              (taskType === "classify" && detail.annotations.length !== 1)
            }
            onClick={() => void confirm()}
          >
            {detail.annotations.length ? "确认当前整图" : "确认此图无目标"}
          </button>
        </>
      )}
    </section>
  );
}
