import type { Dataset } from "../types";

export function DatasetAnnotationSummary({ dataset, onOpenWorkflow }: {
  dataset: Dataset;
  onOpenWorkflow?: (runId: string, reviewId?: string) => void;
}) {
  const generated = dataset.generated_image_count ?? dataset.annotated_image_count;
  const confirmed = dataset.confirmed_image_count ?? 0;
  const pending = dataset.pending_review_image_count ?? Math.max(0, generated - confirmed);
  const progress = dataset.image_count ? Math.round(generated / dataset.image_count * 100) : 0;
  return (
    <div className="dataset-annotation-summary">
      <div className="dataset-progress">
        <span>标注生成进度</span>
        <strong>{dataset.image_count ? `${progress}%` : "等待导入"}</strong>
      </div>
      <div className="progress-track" role="progressbar" aria-label="标注生成进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}>
        <i style={{ width: `${progress}%` }} />
      </div>
      <p className="dataset-annotation-counts">
        <span>已生成 {generated} / {dataset.image_count} 张</span>
        <span>待人工确认 {pending} 张</span>
        <span>人工已确认 {confirmed} / {dataset.image_count} 张</span>
      </p>
      {!!dataset.workflow_image_count && pending > 0 && (
        <p className="hint dataset-annotation-hint">自动标注已同步。打开 LLM 标注复核，确认后衔接训练与导出。</p>
      )}
      {dataset.latest_workflow_run_id && onOpenWorkflow && (
        <button className="button" onClick={() => onOpenWorkflow(dataset.latest_workflow_run_id!, dataset.latest_workflow_review_id || undefined)}>
          {dataset.latest_workflow_review_id ? pending > 0 ? "继续 LLM 标注复核" : "查看 LLM 标注结果" : "查看自动标注进度"}
        </button>
      )}
    </div>
  );
}
