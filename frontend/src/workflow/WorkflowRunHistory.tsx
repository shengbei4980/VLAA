import { useEffect, useState } from "react";
import { workflowApi } from "../api/workflows";
import type {
  TaskProfile,
  WorkflowRun,
  WorkflowTemplate,
} from "../types/workflows";
import { templateTitle, workflowStatus } from "./presentation";

const finished = new Set(["completed", "cancelled"]);

export function WorkflowRunHistory({
  profiles,
  templates,
  onOpenRun,
}: {
  profiles: TaskProfile[];
  templates: WorkflowTemplate[];
  onOpenRun: (id: string) => void;
}) {
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [deleted, setDeleted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [reload, setReload] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [target, setTarget] = useState<WorkflowRun>();
  const [waitingForCancel, setWaitingForCancel] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    workflowApi
      .runs(deleted)
      .then(
        (items) => {
          if (active) setRuns(items);
        },
        (failure) => {
          if (active)
            setError(failure instanceof Error ? failure.message : "运行记录读取失败");
        },
      )
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [deleted, reload]);

  const title = (run: WorkflowRun) =>
    templateTitle({
      key:
        run.template_snapshot?.key ??
        templates.find((item) => item.id === run.template_id)?.key ?? "",
      name:
        run.template_name ?? run.template_snapshot?.name ??
        templates.find((item) => item.id === run.template_id)?.name ?? "自定义流程",
    });

  const remove = async () => {
    if (!target) return;
    setBusy(true);
    setError("");
    try {
      if (!finished.has(target.status)) {
        const current = waitingForCancel
          ? await workflowApi.run(target.id)
          : await workflowApi.controlRun(target.id, "cancel");
        setTarget(current);
        setWaitingForCancel(true);
        setRuns((items) =>
          items.map((item) => item.id === current.id ? current : item),
        );
        if (!finished.has(current.status)) {
          setError("任务正在取消，请稍后点击“检查状态并删除”。");
          return;
        }
      }
      await workflowApi.deleteRun(target.id);
      setRuns((items) => items.filter((item) => item.id !== target.id));
      setTarget(undefined);
      setNotice("记录已移入“已删除记录”，可在那里恢复。");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "删除记录失败，请重试。");
    } finally {
      setBusy(false);
    }
  };

  const restore = async (run: WorkflowRun) => {
    setBusy(true);
    setError("");
    try {
      await workflowApi.restoreRun(run.id);
      setRuns((items) => items.filter((item) => item.id !== run.id));
      setNotice("记录已恢复，可返回运行记录查看。");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "恢复记录失败，请重试。");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      className="panel workflow-form"
      aria-labelledby="workflow-history-title"
    >
      <div className="workflow-rule-heading">
        <h2 id="workflow-history-title">{deleted ? "已删除记录" : "运行记录"}</h2>
        <div className="workflow-actions">
          <button
            className="button"
            disabled={loading || busy}
            onClick={() => setReload((value) => value + 1)}
          >
            刷新记录
          </button>
          <button
            className="button"
            disabled={loading || busy}
            onClick={() => {
              setDeleted(!deleted);
              setRuns([]);
              setNotice("");
            }}
          >
            {deleted ? "返回运行记录" : "已删除记录"}
          </button>
        </div>
      </div>
      {notice && <p role="status" className="validation">{notice}</p>}
      {error && !target && (
        <div role="alert" className="validation invalid">
          <p>{error}</p>
          <button
            className="button"
            disabled={loading || busy}
            onClick={() => setReload((value) => value + 1)}
          >
            重新读取记录
          </button>
        </div>
      )}
      {loading && <p>正在读取运行记录…</p>}
      {!loading && !error && !runs.length && (
        <p>{deleted ? "暂无已删除记录。" : "尚无运行记录。"}</p>
      )}
      <div className="workflow-run-list">
        {runs.map((run) => (
          <div className="workflow-run-entry" key={run.id}>
            <button
              className="workflow-run-row"
              onClick={() => onOpenRun(run.id)}
              disabled={busy}
            >
              <strong>{title(run)}</strong>
              <span>
                {run.profile_name ??
                  profiles.find((item) => item.id === run.profile_id)?.name ?? ""}
              </span>
              <span className="workflow-status-badge">{workflowStatus(run.status)}</span>
              <time>{run.created_at?.replace("T", " ").slice(0, 19)}</time>
            </button>
            {deleted ? (
              <button
                className="button"
                disabled={loading || busy}
                onClick={() => void restore(run)}
                aria-label={`恢复运行记录 ${run.id}`}
              >
                恢复记录
              </button>
            ) : (
              <button
                className="button danger"
                disabled={loading || busy || (finished.has(run.status) && run.can_delete === false)}
                title={finished.has(run.status) && run.can_delete === false
                  ? "关联任务正在结束，请稍后再删除" : "移入已删除记录，可恢复"}
                aria-label={`删除运行记录 ${run.id}`}
                onClick={() => {
                  setTarget(run);
                  setWaitingForCancel(false);
                  setError("");
                  setNotice("");
                }}
              >
                删除记录
              </button>
            )}
          </div>
        ))}
      </div>
      {target && (
        <div className="modal-backdrop">
          <section
            className="dataset-dialog workflow-launch"
            role="dialog"
            aria-modal="true"
            aria-labelledby="workflow-delete-title"
          >
            <h2 id="workflow-delete-title">删除这条运行记录</h2>
            <p>{title(target)} · {workflowStatus(target.status)}</p>
            <p>记录将移入“已删除记录”，之后可以恢复。</p>
            {!finished.has(target.status) && (
              <p>继续后会先取消这项任务，结束后再删除记录。恢复记录后，任务将保留取消状态。</p>
            )}
            {error && <p className="validation invalid" role="alert">{error}</p>}
            <div className="workflow-actions">
              <button
                className="button"
                disabled={busy}
                onClick={() => {
                  setTarget(undefined);
                  setError("");
                }}
              >
                保留记录
              </button>
              <button className="button danger" disabled={busy} onClick={() => void remove()}>
                {busy
                  ? "正在处理…"
                  : waitingForCancel
                    ? "检查状态并删除"
                    : finished.has(target.status)
                      ? "确认删除记录"
                      : "取消任务并删除记录"}
              </button>
            </div>
          </section>
        </div>
      )}
    </section>
  );
}
