import { useEffect, useState, type FormEvent } from "react";
import { workflowApi } from "../api/workflows";
import type { WorkflowProject } from "../types/workflows";

const lastProjectKey = "vlaa.workflow.last-project";

function rememberProject(id: string) {
  try {
    if (id) window.localStorage.setItem(lastProjectKey, id);
    else window.localStorage.removeItem(lastProjectKey);
  } catch {
    // Project registration remains available if the browser disables local storage.
  }
}

function rememberedProject() {
  try {
    return window.localStorage.getItem(lastProjectKey);
  } catch {
    return null;
  }
}

export function ProjectFolderPicker({
  value,
  onChange,
  onReadyChange,
}: {
  value?: WorkflowProject;
  onChange: (project: WorkflowProject | undefined) => void;
  onReadyChange: (ready: boolean) => void;
}) {
  const [projects, setProjects] = useState<WorkflowProject[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("");
  const [rootPath, setRootPath] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setLoadFailed(false);
    setError("");
    void workflowApi
      .projects()
      .then(
        (nextProjects) => {
          if (!active) return;
          setProjects(nextProjects);
          const previous = nextProjects.find(
            (project) => project.id === rememberedProject(),
          );
          if (previous) onChange(previous);
        },
        (failure) => {
          if (active) {
            setLoadFailed(true);
            setError(
              failure instanceof Error ? failure.message : "项目目录读取失败",
            );
          }
        },
      )
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [onChange, loadAttempt]);

  useEffect(() => {
    onReadyChange(!loading && !loadFailed && !creating && !busy);
  }, [loading, loadFailed, creating, busy, onReadyChange]);

  const select = (id: string) => {
    onChange(projects.find((project) => project.id === id));
    rememberProject(id);
  };

  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const project = await workflowApi.saveProject({
        name: name.trim(),
        root_path: rootPath.trim(),
      });
      setProjects((current) => [
        ...current.filter((item) => item.id !== project.id),
        project,
      ]);
      onChange(project);
      rememberProject(project.id);
      setCreating(false);
      setName("");
      setRootPath("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "项目目录保存失败");
    } finally {
      setBusy(false);
    }
  };

  const separator = value?.root_path.includes("\\") ? "\\" : "/";
  const outputRoot = value
    ? value.root_path.replace(/[\\/]+$/, "") + separator + "workflow-<运行编号>"
    : "";

  return (
    <section
      className="panel workflow-form workflow-setup-section"
      aria-labelledby="workflow-project-title"
    >
      <div className="workflow-section-heading">
        <span className="workflow-section-number">4</span>
        <div>
          <h2 id="workflow-project-title">项目与保存位置</h2>
          <p>
            为项目选择一个根目录。以后选择同一项目，即可沿用这里的保存位置。
          </p>
        </div>
      </div>
      <label>
        本次使用的项目
        <select
          value={value?.id ?? ""}
          disabled={loading || loadFailed || busy}
          onChange={(event) => select(event.target.value)}
        >
          <option value="">使用工作台默认位置</option>
          {projects.map((project) => (
            <option key={project.id} value={project.id}>
              {project.name} · {project.root_path}
            </option>
          ))}
        </select>
      </label>
      {loading && <p>正在读取已保存的项目…</p>}
      {value && (
        <div className="workflow-choice-summary">
          <p>
            项目根目录 <code>{value.root_path}</code>
          </p>
          <p>
            本次
            JSON、标注图片、模型和评估结果统一保存到项目内，自动按运行编号分开。
          </p>
          <ul>
            <li>
              LabelMe JSON{" "}
              <code>
                {outputRoot + separator + "annotations" + separator + "json"}
              </code>
            </li>
            <li>
              标注预览 PNG{" "}
              <code>
                {outputRoot + separator + "annotations" + separator + "images"}
              </code>
            </li>
            <li>
              模型与评估结果 <code>{outputRoot + separator + "models"}</code>、
              <code>{outputRoot + separator + "evaluations"}</code>
            </li>
          </ul>
          <p className="hint">
            本次导出优先使用项目目录，类别与模型方案中另设的输出路径由项目目录接管。
          </p>
        </div>
      )}
      {creating ? (
        <form onSubmit={(event) => void save(event)}>
          <div className="workflow-fields">
            <label>
              项目名称
              <input
                value={name}
                onChange={(event) => setName(event.target.value)}
                disabled={busy}
                placeholder="例如 塔体构件标注"
                required
              />
            </label>
            <label>
              项目根目录
              <input
                value={rootPath}
                onChange={(event) => setRootPath(event.target.value)}
                disabled={busy}
                placeholder={"例如 E:\\视觉标注项目\\塔体构件"}
                required
              />
            </label>
          </div>
          <p className="hint">
            填写电脑上的完整文件夹路径。保存项目后，系统会自动安排各类结果的子目录。
          </p>
          <div className="workflow-actions">
            <button
              className="button primary"
              disabled={busy || !name.trim() || !rootPath.trim()}
            >
              {busy ? "正在保存项目…" : "保存项目并使用"}
            </button>
            <button
              className="button"
              type="button"
              disabled={busy}
              onClick={() => setCreating(false)}
            >
              取消新建
            </button>
          </div>
        </form>
      ) : (
        <button
          className="button"
          disabled={loading || loadFailed}
          onClick={() => {
            setCreating(true);
            setError("");
          }}
        >
          新建项目并设置根目录
        </button>
      )}
      {error && (
        <div className="validation invalid" role="alert">
          <p>{error}</p>
          {loadFailed && (
            <button
              className="button"
              onClick={() => setLoadAttempt((attempt) => attempt + 1)}
            >
              重新读取项目
            </button>
          )}
        </div>
      )}
    </section>
  );
}
