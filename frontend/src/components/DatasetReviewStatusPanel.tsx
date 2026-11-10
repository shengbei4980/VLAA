import { useEffect, useState } from "react";
import { reviewApi, type DatasetReviewStatus } from "../api/reviews";

export function DatasetReviewStatusPanel({
  datasetId,
  onOpenImage,
  onOpenWorkflowReview,
  refreshToken,
}: {
  datasetId: string;
  onOpenImage: (id: string) => void;
  onOpenWorkflowReview?: (runId: string, reviewId?: string) => void;
  refreshToken?: number;
}) {
  const [status, setStatus] = useState<DatasetReviewStatus>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [reload, setReload] = useState(0);
  useEffect(() => {
    if (!expanded) return;
    let active = true;
    setBusy(true);
    setError("");
    void reviewApi.status(datasetId).then(
      (result) => { if (active) setStatus(result); },
      (failure) => {
        if (active) setError(failure instanceof Error ? failure.message : "审核汇总读取失败");
      },
    ).finally(() => { if (active) setBusy(false); });
    return () => { active = false; };
  }, [datasetId, expanded, refreshToken, reload]);
  return (
    <section className="workflow-dataset-review">
      <button className="button" disabled={busy} onClick={() => { setExpanded(true); setReload((value) => value + 1); }}>
        {busy ? "正在读取…" : status ? "刷新标注与审核状态" : "查看标注与审核状态"}
      </button>
      {error && (
        <p className="validation invalid" role="alert">
          {error}
        </p>
      )}
      {status && (
        <>
          <p>
            人工已确认 {status.confirmed_image_count ?? status.confirmed} · 待人工确认 {status.pending_review_image_count ?? status.pending} · 确认无目标{" "}
            {status.confirmed_empty} · 总计 {status.total}
          </p>
          <div className="workflow-image-picker">
            {status.items.map((item) => (
              <article className="workflow-item" key={item.image_id}>
                <div>
                  <strong>{item.file_name || item.image_id}</strong>
                  <small>{annotationState(item)}</small>
                  <small>
                    正式标注 {item.annotation_count} 个
                    {item.workflow_run_id ? ` · 工作流候选 ${item.workflow_annotation_count ?? 0} 个` : ""}
                    {item.contains_auto ? " · 含自动来源" : ""}
                  </small>
                </div>
                <div className="workflow-actions">
                  {item.workflow_run_id && onOpenWorkflowReview && (
                    <button className="button" onClick={() => onOpenWorkflowReview(item.workflow_run_id!, item.workflow_review_id || undefined)}>
                      {item.workflow_review_id ? "打开 LLM 标注复核" : "查看自动标注进度"}
                    </button>
                  )}
                  <button className="button" onClick={() => onOpenImage(item.image_id)}>
                    打开数据集标注
                  </button>
                </div>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function annotationState(item: DatasetReviewStatus["items"][number]): string {
  if (item.workflow_stale) return "标注版本已变化，请重新复核";
  const workflowLabels: Record<string, string> = {
    processing: "自动标注正在处理",
    draft: "工作流已生成标注，待人工确认",
    confirmed: "已确认并同步数据集",
    stale: "标注版本已变化，请重新复核",
    cancelled: "自动标注运行已取消",
    failed: "自动标注运行失败",
    excluded: "已排除此图",
  };
  if (item.workflow_status && workflowLabels[item.workflow_status]) return workflowLabels[item.workflow_status];
  if (item.review_status === "confirmed_empty") return "已确认无目标";
  if (item.review_status === "confirmed") return "已完成人工确认";
  return item.has_draft ? "已有编辑草稿，待人工确认" : "待人工确认";
}
