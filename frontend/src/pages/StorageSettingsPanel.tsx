import { useEffect, useState } from "react";
import {
  storageApi,
  type DataRootPlan,
  type DestinationValidation,
  type StorageSettings,
} from "../api/storage";
import "../workflow.css";
import { ExportLocationsPanel } from "../components/ExportLocationsPanel";

const formatBytes = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
const storageError = (reason: unknown) => {
  const message = reason instanceof Error ? reason.message : "存储请求失败";
  return /404|not found/i.test(message)
    ? "后端存储接口尚未加载，请重启新版后端。目录输入会保留。"
    : message;
};

export function StorageSettingsPanel({
  onChanged,
}: {
  onChanged?: () => void;
}) {
  const [settings, setSettings] = useState<StorageSettings>();
  const [destinationName, setDestinationName] = useState("");
  const [destinationPath, setDestinationPath] = useState("");
  const [validated, setValidated] = useState<DestinationValidation>();
  const [targetPath, setTargetPath] = useState("");
  const [plan, setPlan] = useState<DataRootPlan>();
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [migrating, setMigrating] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  useEffect(() => {
    let active = true;
    storageApi
      .settings()
      .then((result) => {
        if (active) setSettings(result);
      })
      .catch((reason) => {
        if (active) setError(storageError(reason));
      });
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    if (!migrating && settings?.migration?.status !== "running") return;
    let active = true;
    const timer = window.setInterval(() => {
      storageApi
        .settings()
        .then((result) => {
          if (active) setSettings(result);
        })
        .catch(() => undefined);
    }, 1500);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [migrating, settings?.migration?.status]);
  const perform = async (action: () => Promise<void>) => {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await action();
    } catch (reason) {
      setError(storageError(reason));
    } finally {
      setBusy(false);
    }
  };
  const register = () =>
    perform(async () => {
      if (!validated?.writable) throw new Error("请先验证目录可写");
      await storageApi.registerDestination(
        destinationName.trim(),
        validated.resolved_path,
      );
      setSettings(await storageApi.settings());
      setMessage("输出目标已注册，可在任务配置档中按产物选择。");
      onChanged?.();
    });
  const execute = () =>
    perform(async () => {
      if (!plan || !confirmed) return;
      setMigrating(true);
      try {
        setSettings(await storageApi.executeDataRoot(plan.plan_id));
        setMessage(
          "数据已复制并校验，待切换配置已保存。重启工作台后采用新目录。",
        );
        setPlan(undefined);
        setConfirmed(false);
        onChanged?.();
      } finally {
        setMigrating(false);
      }
    });
  return (
    <section className="workflow-form panel">
      <header className="workflow-heading">
        <div>
          <h2>存储位置</h2>
          <p>设置 JSON、标注预览图与工作流产物的保存目录，也可迁移整个工作台数据。</p>
        </div>
        <button
          className="button"
          disabled={busy}
          onClick={() =>
            void perform(async () => setSettings(await storageApi.settings()))
          }
        >
          刷新状态
        </button>
      </header>
      {error && (
        <p className="validation invalid" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="validation" role="status">
          {message}
        </p>
      )}
      <ExportLocationsPanel />
      <h3>当前工作台</h3>
      <p>
        数据根 <code>{settings?.current.data_dir ?? "正在读取"}</code>
      </p>
      <p>
        数据库 <code>{settings?.current.database_path ?? "正在读取"}</code>
      </p>
      <p>来源 {settings?.current.source ?? "待确认"}</p>
      {(settings?.environment_overrides.data_dir ||
        settings?.environment_overrides.database_url) && (
        <p className="validation">
          启动环境变量正在指定数据或数据库位置。重启前需核对并更新相应覆盖项。
        </p>
      )}
      {settings?.pending && (
        <div className="validation">
          <strong>待切换目录</strong>
          <p>{settings.pending.data_dir}</p>
          <p>{settings.pending.database_path}</p>
          <p>
            {settings.restart_required
              ? "需要重启工作台使新目录生效。"
              : "等待状态更新。"}
            {settings.write_locked ? "当前写入已锁定。" : ""}
          </p>
          <button
            className="button"
            disabled={busy}
            onClick={() =>
              void perform(async () => {
                setSettings(await storageApi.cancelDataRoot());
                setMessage(
                  "待切换配置已撤销，当前目录恢复可写，复制目录保留。",
                );
                onChanged?.();
              })
            }
          >
            撤销待切换配置
          </button>
        </div>
      )}
      {settings?.migration && (
        <p role="status">
          迁移 {settings.migration.status} · {settings.migration.phase}
          {settings.migration.error
            ? ` · ${settings.migration.error}`
            : ""}{" "}
          {JSON.stringify(settings.migration.counts ?? {})}
        </p>
      )}
      <fieldset
        disabled={busy || !!settings?.write_locked}
        className="workflow-endpoint"
      >
        <legend>注册工作流输出目标</legend>
        <p>输入绝对本地目录，验证后注册。Agent 和工作流使用注册目标 ID。</p>
        <div className="workflow-fields">
          <label>
            目标名称
            <input
              value={destinationName}
              onChange={(event) => setDestinationName(event.target.value)}
              placeholder="例如 项目结果盘"
            />
          </label>
          <label>
            输出目录绝对路径
            <input
              value={destinationPath}
              onChange={(event) => {
                setDestinationPath(event.target.value);
                setValidated(undefined);
              }}
              placeholder="D:\项目结果"
            />
          </label>
        </div>
        <div className="workflow-actions">
          <button
            className="button"
            disabled={!destinationPath.trim()}
            onClick={() =>
              void perform(async () =>
                setValidated(
                  await storageApi.validateDestination(destinationPath.trim()),
                ),
              )
            }
          >
            验证目录
          </button>
          <button
            className="button primary"
            disabled={!destinationName.trim() || !validated?.writable}
            onClick={() => void register()}
          >
            注册输出目标
          </button>
        </div>
        {validated && (
          <p>
            {validated.resolved_path} · {validated.writable ? "可写" : "不可写"}{" "}
            · 可用 {formatBytes(validated.free_bytes)}
          </p>
        )}
        <ul>
          {settings?.destinations.map((destination) => (
            <li key={destination.id}>
              <strong>{destination.name}</strong> · {destination.path} ·{" "}
              <code>{destination.id}</code>
            </li>
          ))}
        </ul>
      </fieldset>
      <fieldset
        disabled={busy || !!settings?.pending}
        className="workflow-endpoint"
      >
        <legend>复制并切换整个工作台数据根</legend>
        <p>
          包括数据库、图片、模型、设置与运行产物。复制保留原目录，成功后需重启。
        </p>
        <label>
          新的数据根绝对路径
          <input
            value={targetPath}
            onChange={(event) => {
              setTargetPath(event.target.value);
              setPlan(undefined);
              setConfirmed(false);
            }}
            placeholder="E:\YoloWorkbenchData"
          />
        </label>
        <button
          className="button"
          disabled={!targetPath.trim() || !settings}
          onClick={() =>
            void perform(async () => {
              setPlan(await storageApi.prepareDataRoot(targetPath.trim()));
              setConfirmed(false);
            })
          }
        >
          预检数据根迁移
        </button>
        {plan && (
          <div className="workflow-migration-plan">
            <p>当前目录 {plan.source_dir}</p>
            <p>复制目标 {plan.target_dir}</p>
            <p>数据库 {plan.database_path}</p>
            <p>
              {plan.file_count} 个文件 · 需要 {formatBytes(plan.required_bytes)}{" "}
              · 可用 {formatBytes(plan.free_bytes)}
            </p>
            <label className="workflow-check">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(event) => setConfirmed(event.target.checked)}
              />
              已核对复制范围与目标目录，复制后重启采用新目录，原目录保留
            </label>
            <button
              className="button primary"
              disabled={!confirmed || busy}
              onClick={() => void execute()}
            >
              {migrating ? "正在复制与校验…" : "执行复制迁移"}
            </button>
          </div>
        )}
      </fieldset>
    </section>
  );
}
