import { useEffect, useState } from "react";
import { workflowApi } from "../api/workflows";
import type { WorkflowRun } from "../types/workflows";
import { runProfile } from "../types/workflows";
import { ArtifactReviewView } from "./ArtifactReviewView";
import { apiUrl } from "../api/client";
import { BatchExportPanel } from "../workflow/BatchExportPanel";
import {
  nodeDescription,
  nodeLabel,
  scopeLabel,
  templateTitle,
  workflowError,
  workflowStatus,
} from "../workflow/presentation";

const terminal = new Set(["completed", "cancelled"]);
export { workflowStatus } from "../workflow/presentation";

function WorkflowErrorMessage({ message }: { message: string }) {
  const explanation = workflowError(message);
  return (
    <div role="alert" className="validation invalid">
      {explanation}
      {explanation !== message && (
        <details>
          <summary>查看原始错误信息</summary>
          <p>{message}</p>
        </details>
      )}
    </div>
  );
}

export function WorkflowRunView({
  runId,
  onBack,
  onOpenTraining,
  onOpenModel,
  onOpenRun,
  onDirtyChange,
  onDataChanged,
  initialReviewId,
}: {
  runId: string;
  onBack: () => void;
  onOpenTraining: (datasetId: string | undefined, taskId: string) => void;
  onOpenModel: (datasetId: string | undefined, modelId: string) => void;
  onOpenRun?: (id: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
  onDataChanged?: () => void;
  initialReviewId?: string;
}) {
  const [run, setRun] = useState<WorkflowRun>();
  const [reviewId, setReviewId] = useState(initialReviewId ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const refresh = async () => setRun(await workflowApi.run(runId));
  useEffect(() => {
    setRun(undefined);
    setReviewId(initialReviewId ?? "");
    let active = true;
    let pending = false;
    const poll = async () => {
      if (pending) return;
      pending = true;
      try {
        const result = await workflowApi.run(runId);
        if (active) {
          setRun(result);
          setError("");
        }
      } catch (reason) {
        if (active)
          setError(reason instanceof Error ? reason.message : "运行读取失败");
      } finally {
        pending = false;
      }
    };
    void poll();
    const timer = window.setInterval(() => {
      if (!document.hidden) void poll();
    }, 2000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [runId]);
  useEffect(() => {
    window.history.replaceState(null, "", `#workflow/${encodeURIComponent(runId)}${reviewId ? `/review/${encodeURIComponent(reviewId)}` : ""}`);
  }, [runId, reviewId]);
  const control = async (
    action: "cancel" | "resume" | "retry",
    itemIds?: string[],
  ) => {
    setBusy(true);
    setError("");
    try {
      await workflowApi.controlRun(runId, action, itemIds);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "操作失败");
    } finally {
      setBusy(false);
    }
  };
  if (reviewId && run)
    return (
      <ArtifactReviewView
        key={reviewId}
        reviewId={reviewId}
        profile={runProfile(run)}
        onDirtyChange={onDirtyChange}
        onBack={() => {
          setReviewId("");
          void refresh().catch((reason) => setError(String(reason)));
        }}
        onSaved={() => {
          onDataChanged?.();
          void refresh().catch((reason) => setError(String(reason)));
        }}
      />
    );
  const counts = run?.batch_counts ?? {};
  const confirmed = (counts.approved ?? 0) + (counts.background ?? 0);
  const requiresReview = run?.template_snapshot.nodes.some(
    (node) => node.type === "human_review",
  );
  const snapshotNode = run?.template_snapshot.nodes.find(
    (node) => node.type === "snapshot",
  );
  const snapshotVerified = run?.attempts.some(
    (attempt) =>
      attempt.node_id === snapshotNode?.id && attempt.status === "completed",
  );
  const datasetId =
    typeof run?.context.dataset_id === "string"
      ? run.context.dataset_id
      : undefined;
  const latestReviews = new Map(
    run?.reviews.map((review) => [review.item_id, review]),
  );
  const nextReviewItem = run?.items.find((item) => {
    const review = latestReviews.get(item.id);
    return (
      review &&
      !["approved", "background", "excluded", "decided"].includes(review.status)
    );
  });
  const nextReview = nextReviewItem
    ? latestReviews.get(nextReviewItem.id)
    : undefined;
  const hasExport = run?.artifacts.some(
    (artifact) => artifact.kind === "export",
  );
  return (
    <main className="workflow-page workflow-run-page">
      <header className="workflow-heading">
        <div>
          <button className="button" onClick={onBack}>
            ← 工作流
          </button>
          <h1>
            {run
              ? templateTitle({
                  key: run.template_snapshot?.key ?? "",
                  name: run.template_snapshot?.name ?? "任务进度",
                })
              : "任务进度"}
          </h1>
          <p>
            <span className="workflow-status-badge" data-status={run?.status}>
              {run ? workflowStatus(run.status) : "正在载入…"}
            </span>
          </p>
          <details>
            <summary>查看任务编号</summary>
            <code>{runId}</code>
          </details>
        </div>
        <div className="workflow-actions">
          {run && (
            <>
              <button
                className="button"
                disabled={busy || terminal.has(run.status)}
                onClick={() => void control("resume")}
              >
                继续/恢复
              </button>
              <button
                className="button"
                disabled={busy || terminal.has(run.status)}
                onClick={() => void control("retry")}
              >
                重试失败项
              </button>
              <button
                className="button danger"
                disabled={
                  busy ||
                  terminal.has(run.status) ||
                  run.status === "cancelling"
                }
                onClick={() => void control("cancel")}
              >
                取消运行
              </button>
            </>
          )}
        </div>
      </header>
      {error && <WorkflowErrorMessage message={error} />}
      {run?.error_message && (
        <WorkflowErrorMessage message={run.error_message} />
      )}
      {run && (
        <>
          <section
            className="workflow-next-action"
            aria-labelledby="workflow-next-action-title"
          >
            <div>
              <span className="eyebrow">
                {hasExport ? "结果已就绪" : "当前待办"}
              </span>
              <h2 id="workflow-next-action-title">
                {nextReview
                  ? "检查这张图片的标注"
                  : hasExport
                    ? "下载本次已确认的标注"
                    : workflowStatus(run.status)}
              </h2>
              <p>
                {nextReview
                  ? `从第 ${run.items.indexOf(nextReviewItem!) + 1} 张图片继续，核对类别、边界与漏标，保存草稿后确认整图。`
                  : hasExport
                    ? "导出文件已生成，包含本次按流程确认的标注。保存位置和训练结果可在下方查看。"
                    : "下方按顺序显示每一步的进度。需要人工检查时，图片旁会出现复核入口。"}
              </p>
            </div>
            <div className="workflow-actions">
              {nextReview && (
                <button
                  className="button primary"
                  onClick={() => setReviewId(nextReview.id)}
                >
                  打开图像复核
                </button>
              )}
              {hasExport && (
                <a
                  className="button primary"
                  href={workflowApi.exportUrl(run.id)}
                >
                  下载已确认导出
                </a>
              )}
              <button
                className="button"
                onClick={() =>
                  document
                    .getElementById("workflow-results")
                    ?.scrollIntoView({ block: "start" })
                }
              >
                查看结果与保存位置
              </button>
            </div>
          </section>
          <section className="panel workflow-form">
            <h2>
              {requiresReview
                ? "本批图片的人工检查进度"
                : "本次使用的已确认数据"}
            </h2>
            {requiresReview ? (
              <>
                <div className="workflow-counts">
                  <span>
                    <strong>
                      {confirmed} / {counts.total ?? run.items.length}
                    </strong>
                    已确认 / 选中图片
                  </span>
                  <span>
                    <strong>{counts.pending ?? 0}</strong>待审
                  </span>
                  <span>
                    <strong>{counts.failed ?? 0}</strong>失败
                  </span>
                  <span>
                    <strong>{counts.excluded ?? 0}</strong>明确排除
                  </span>
                  <span>
                    <strong>{counts.background ?? 0}</strong>确认无目标
                  </span>
                </div>
                <p>
                  所有选中图片都检查完成后，才会继续训练或导出。请先处理待检查、失败或退回修改的图片。
                </p>
              </>
            ) : (
              <>
                <p>
                  {snapshotVerified
                    ? "已核对本次数据"
                    : "正在核对图片是否已完成整图确认"}{" "}
                  · 选中 {counts.total ?? run.items.length} 张
                </p>
                <p>
                  本流程使用已经确认的标注，并保存本次数据版本供训练和评估使用。
                </p>
              </>
            )}
          </section>
          {["detect", "segment"].includes(runProfile(run).task_type) && (
            <BatchExportPanel
              key={run.id}
              runId={run.id}
              readyImages={
                new Set(run.reviews.map((review) => review.item_id)).size
              }
            />
          )}
          <section className="panel workflow-form">
            <h2>步骤与执行记录</h2>
            <p className="hint">
              按编号顺序执行。技术信息与历史重试记录可展开查看。
            </p>
            <ol className="workflow-step-list">
              {run.template_snapshot.nodes.map((node, index) => {
                const attempts = run.attempts.filter(
                  (attempt) => attempt.node_id === node.id,
                );
                const latestByItem = new Map(
                  attempts.map((attempt) => [attempt.item_id ?? null, attempt]),
                );
                const currentAttempts = Array.from(latestByItem.values());
                const statusCounts = new Map<string, number>();
                for (const attempt of currentAttempts) {
                  statusCounts.set(
                    attempt.status,
                    (statusCounts.get(attempt.status) ?? 0) + 1,
                  );
                }
                const previousAttempts = attempts.filter(
                  (attempt) =>
                    latestByItem.get(attempt.item_id ?? null)?.id !==
                    attempt.id,
                );
                const unnecessary =
                  node.task_types?.length &&
                  !node.task_types.includes(runProfile(run).task_type);
                return (
                  <li
                    key={node.id}
                    data-status={
                      unnecessary
                        ? "skipped"
                        : (currentAttempts.find(
                            (attempt) => attempt.status !== "completed",
                          )?.status ??
                          (currentAttempts.length ? "completed" : "pending"))
                    }
                  >
                    <strong>
                      {index + 1}.{" "}
                      {node.params?.only_if_repaired
                        ? "复查修正后的标注"
                        : nodeLabel(node.type)}
                    </strong>
                    <span>{nodeDescription(node.type)}</span>
                    <span className="workflow-step-status">
                      {unnecessary
                        ? "此任务无需此步"
                        : currentAttempts.length
                          ? Array.from(
                              statusCounts,
                              ([status, count]) =>
                                `${workflowStatus(status)}${node.scope === "item" ? ` ${count} 张` : ""}`,
                            ).join(" · ")
                          : "等待前面的步骤完成"}
                    </span>
                    <details>
                      <summary>步骤技术信息</summary>
                      <code>
                        {node.id} · {node.type} · {scopeLabel(node.scope)}
                      </code>
                    </details>
                    {currentAttempts.map((attempt) => (
                      <div key={attempt.id}>
                        {attempt.error_message && (
                          <WorkflowErrorMessage
                            message={attempt.error_message}
                          />
                        )}
                      </div>
                    ))}
                    {previousAttempts.length > 0 && (
                      <details>
                        <summary>
                          历史尝试记录（{previousAttempts.length}）
                        </summary>
                        <ol>
                          {previousAttempts.map((attempt) => (
                            <li key={attempt.id}>
                              <span>{workflowStatus(attempt.status)}</span>
                              <code>
                                {attempt.id} · {attempt.item_id ?? "整批处理"}
                              </code>
                              {attempt.error_message && (
                                <WorkflowErrorMessage
                                  message={attempt.error_message}
                                />
                              )}
                            </li>
                          ))}
                        </ol>
                      </details>
                    )}
                  </li>
                );
              })}
            </ol>
          </section>
          <section className="panel workflow-form" id="workflow-images">
            <h2>逐张图片检查</h2>
            {run.items.map((item, index) => {
              const review = latestReviews.get(item.id);
              const context =
                item.context && typeof item.context === "object"
                  ? (item.context as Record<string, unknown>)
                  : {};
              return (
                <article key={item.id} className="workflow-item">
                  <div className="workflow-item-info">
                    {typeof context.image_url === "string" && (
                      <img
                        className="workflow-image-thumb"
                        src={apiUrl(context.image_url)}
                        alt=""
                        loading="lazy"
                      />
                    )}
                    <div>
                      <strong>第 {index + 1} 张图片</strong>
                      <span>{workflowStatus(item.status)}</span>
                      <details>
                        <summary>查看图片编号</summary>
                        <code>
                          {item.image_id ?? item.subject_ref ?? item.id}
                        </code>
                      </details>
                      {item.error_message && (
                        <WorkflowErrorMessage message={item.error_message} />
                      )}
                    </div>
                  </div>
                  <div className="workflow-actions">
                    {review && (
                      <button
                        className="button primary"
                        onClick={() => setReviewId(review.id)}
                      >
                        {["approved", "background", "excluded"].includes(
                          review.status,
                        ) || review.status === "decided"
                          ? "查看审核"
                          : "打开图像复核"}
                      </button>
                    )}
                    {["failed", "needs_attention", "interrupted"].includes(
                      item.status,
                    ) && (
                      <button
                        className="button"
                        disabled={busy}
                        onClick={() => void control("retry", [item.id])}
                      >
                        重试此项
                      </button>
                    )}
                  </div>
                </article>
              );
            })}
            {!run.items.length && <p>尚未产生处理单元。</p>}
          </section>
          <section className="panel workflow-form" id="workflow-results">
            <h2>查看训练与结果</h2>
            <p>
              已完成请求 {run.request_count ?? 0} · 预留请求{" "}
              {run.reserved_requests ?? 0} · 配置上限{" "}
              {runProfile(run).max_requests}。费用按端点实际账单核对。
            </p>
            <div className="workflow-actions">
              {run.attempts.map((attempt) => {
                const output =
                  attempt.output && typeof attempt.output === "object"
                    ? (attempt.output as Record<string, unknown>)
                    : {};
                return (
                  <span key={attempt.id}>
                    {attempt.task_kind === "train" &&
                      typeof attempt.task_id === "string" && (
                        <button
                          className="button"
                          onClick={() =>
                            onOpenTraining(datasetId, attempt.task_id as string)
                          }
                        >
                          查看训练进度
                        </button>
                      )}
                    {typeof output.model_id === "string" && (
                      <button
                        className="button"
                        onClick={() =>
                          onOpenModel(datasetId, output.model_id as string)
                        }
                      >
                        查看生成的模型
                      </button>
                    )}
                    {typeof output.next_run_id === "string" && onOpenRun && (
                      <button
                        className="button primary"
                        onClick={() => onOpenRun(output.next_run_id as string)}
                      >
                        打开下一批运行
                      </button>
                    )}
                    {typeof output.snapshot_id === "string" && (
                      <details>
                        <summary>本次数据版本编号</summary>
                        <code>{output.snapshot_id}</code>
                      </details>
                    )}
                  </span>
                );
              })}
              {hasExport && (
                <a
                  className="button primary"
                  href={workflowApi.exportUrl(run.id)}
                >
                  下载已确认导出
                </a>
              )}
              {run.parent_run_id && onOpenRun && (
                <button
                  className="button"
                  onClick={() => onOpenRun(run.parent_run_id!)}
                >
                  返回上批运行
                </button>
              )}
            </div>
            <div className="workflow-actions">
              {typeof run.context.training_task_id === "string" && (
                <button
                  className="button"
                  onClick={() =>
                    onOpenTraining(
                      datasetId,
                      run.context.training_task_id as string,
                    )
                  }
                >
                  打开训练任务
                </button>
              )}
              {typeof run.context.model_id === "string" && (
                <button
                  className="button"
                  onClick={() =>
                    onOpenModel(datasetId, run.context.model_id as string)
                  }
                >
                  打开模型
                </button>
              )}
            </div>
            {run.context.output_destinations && (
              <section>
                <h3>实际产物保存位置</h3>
                {run.context.output_destinations.project && (
                  <p>
                    项目 {run.context.output_destinations.project.name} · 根目录{" "}
                    <code>
                      {run.context.output_destinations.project.root_path}
                    </code>
                  </p>
                )}
                {run.context.output_destinations.json_path && (
                  <p>
                    LabelMe JSON{" "}
                    <code>{run.context.output_destinations.json_path}</code>
                  </p>
                )}
                {run.context.output_destinations.images_path && (
                  <p>
                    标注预览 PNG{" "}
                    <code>{run.context.output_destinations.images_path}</code>
                  </p>
                )}
                <ul>
                  {(["annotations", "models", "evaluations"] as const).map(
                    (kind) => {
                      const destination =
                        run.context.output_destinations![kind];
                      return (
                        <li key={kind}>
                          {
                            {
                              annotations: "标注",
                              models: "模型副本",
                              evaluations: "评估结果",
                            }[kind]
                          }{" "}
                          ·{" "}
                          {destination?.enabled === false
                            ? "本次不额外输出"
                            : (destination?.run_path ??
                              destination?.path ??
                              "工作台受管目录")}
                        </li>
                      );
                    },
                  )}
                </ul>
              </section>
            )}
            <details>
              <summary>
                高级详情 · 配置与结果文件（{run.artifacts.length}）
              </summary>
              <pre>
                {JSON.stringify(
                  {
                    context: run.context,
                    profile: run.profile_snapshot,
                    artifacts: run.artifacts,
                  },
                  null,
                  2,
                )}
              </pre>
            </details>
            <details>
              <summary>运行事件（{run.events.length}）</summary>
              <pre>{JSON.stringify(run.events, null, 2)}</pre>
            </details>
          </section>
        </>
      )}
    </main>
  );
}
