import type { WorkflowProgress, WorkflowRun } from "../types/workflows";
import { nodeLabel } from "./presentation";

export function durationText(seconds: number): string {
  const value = Math.max(0, Math.round(seconds));
  if (value < 60) return `${value} 秒`;
  if (value < 3600) return `${Math.floor(value / 60)} 分 ${value % 60} 秒`;
  return `${Math.floor(value / 3600)} 小时 ${Math.floor(value % 3600 / 60)} 分`;
}

const etaText: Record<WorkflowProgress["eta_state"], string> = {
  estimated: "",
  learning: "正在积累测速样本",
  blocked: "处理异常，暂不能估计",
  waiting_review: "等待人工确认",
  overdue: "本张耗时较长，重新估计中",
  complete: "自动处理已结束",
  cancelled: "运行已停止估时",
  not_applicable: "请查看下方训练进度",
};

export function WorkflowProgressPanel({ run, disconnected }: { run: WorkflowRun; disconnected: boolean }) {
  const progress = run.progress;
  if (!progress) return null;
  const active = ["pending", "running", "waiting_task"].includes(run.status);
  const current = progress.current;
  const allReady = progress.ready_images === progress.total_images && progress.total_images > 0;
  const title = current ? nodeLabel(current.node_type)
    : run.status === "completed" ? "本次流程已完成"
    : run.status === "cancelled" ? "本次运行已取消"
    : run.status === "cancelling" ? "正在等待当前请求结束"
    : run.status === "needs_attention" ? "有图片需要处理后重试"
    : progress.waiting_review_images > 0 && !active ? "标注已就绪，等待你检查"
    : "等待调度下一步";
  const stages = [
    { title: "自动标注与检查", done: allReady },
    { title: "人工复核", done: progress.confirmed_images + progress.excluded_images === progress.total_images && progress.total_images > 0 },
    { title: "训练或导出", done: run.status === "completed" },
  ];
  const estimate = disconnected ? "连接恢复后更新"
    : progress.eta_seconds !== null && progress.eta_state === "estimated"
      ? `约 ${durationText(progress.eta_seconds)}` : etaText[progress.eta_state];
  return (
    <section className="workflow-progress-panel" aria-labelledby="workflow-progress-title" data-active={active && !disconnected}>
      <div className="workflow-progress-heading">
        <div>
          <span className="eyebrow">{progress.automatic_applicable ? "自动标注进度" : "流程进度"}</span>
          <h2 id="workflow-progress-title">{title}</h2>
          <p className="workflow-progress-current">
            {current?.image_index ? `第 ${current.image_index} / ${progress.total_images} 张` : `本批 ${progress.total_images} 张图片`}
            {current?.file_name && <span title={current.file_name}> · {current.file_name}</span>}
            {current && <> · 此步骤已用 {durationText(current.elapsed_seconds)}</>}
          </p>
        </div>
        <span className="workflow-live-label" role="status">
          <i aria-hidden="true" />{disconnected ? "连接中断 · 显示上次状态" : active ? "自动更新中" : "已同步"}
        </span>
      </div>
      {progress.automatic_applicable && (
        <>
          <div className="workflow-progress-caption">
            <span><strong>{progress.ready_images}</strong> / {progress.total_images} 张已生成并检查</span>
            <strong>{progress.percent ?? 0}%</strong>
          </div>
          <div className="workflow-progress-track" role="progressbar" aria-label="自动标注与检查进度"
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress.percent ?? 0}
            aria-valuetext={`${progress.ready_images} / ${progress.total_images} 张；${progress.completed_steps} / ${progress.total_steps} 个自动步骤完成`}>
            <div style={{ width: `${progress.percent ?? 0}%` }} />
          </div>
          <p className="workflow-progress-note">按已完成的自动步骤计算；进度到 100% 后仍需人工确认。排除的图片不计为标注成功。</p>
          <dl className="workflow-progress-metrics">
            <div><dt>累计自动处理耗时</dt><dd>{durationText(progress.processing_seconds)}</dd><small>不含人工等待和步骤间隔</small></div>
            <div><dt>预计剩余自动处理时间</dt><dd>{estimate}</dd><small>{progress.sample_count >= 2 ? `参考本批最近 ${progress.sample_count} 张的实际耗时` : "至少完成 2 张后估计；单张任务暂不估时"}</small></div>
            <div><dt>待你复核 / 处理失败</dt><dd>{progress.waiting_review_images}<span> / </span>{progress.failed_images}<small> 张</small></dd><small>已人工确认 {progress.confirmed_images} 张 · 已排除 {progress.excluded_images} 张</small></div>
          </dl>
          <ol className="workflow-progress-stages" aria-label="流程阶段">
            {stages.map((stage, index) => <li key={stage.title} data-done={stage.done}><span>{stage.done ? "✓" : `0${index + 1}`}</span>{stage.title}</li>)}
          </ol>
          <p className="workflow-progress-note">剩余时间为估计值，会随图片复杂度、服务响应和重试变化；不包含后续人工复核、训练和导出。</p>
        </>
      )}
    </section>
  );
}
