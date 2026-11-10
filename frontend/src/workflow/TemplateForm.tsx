import { useEffect, useRef, useState } from "react";
import { workflowApi } from "../api/workflows";
import type { WorkflowNode, WorkflowTemplate } from "../types/workflows";
import {
  nodeDescription,
  nodeLabel,
  scopeLabel,
  templateTitle,
} from "./presentation";

interface RegisteredStep {
  type: string;
  scope: "item" | "batch";
  version: number;
}

export function TemplateForm({
  source,
  onSaved,
  onClose,
}: {
  source: WorkflowTemplate;
  onSaved: (template: WorkflowTemplate) => void;
  onClose: () => void;
}) {
  const [name, setName] = useState(() => templateTitle(source));
  const [key, setKey] = useState(source.key);
  const [nodes, setNodes] = useState(JSON.stringify(source.nodes, null, 2));
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [registry, setRegistry] = useState<RegisteredStep[]>([]);
  const [addingType, setAddingType] = useState("");
  const headingRef = useRef<HTMLHeadingElement>(null);
  useEffect(() => {
    headingRef.current?.focus();
    let active = true;
    workflowApi
      .nodes()
      .then((contracts) => {
        if (!active) return;
        const available = contracts.flatMap((item): RegisteredStep[] =>
          typeof item.type === "string" &&
          (item.scope === "item" || item.scope === "batch") &&
          typeof item.version === "number"
            ? [{ type: item.type, scope: item.scope, version: item.version }]
            : [],
        );
        setRegistry(available);
        setAddingType(available[0]?.type ?? "");
      })
      .catch((reason) => {
        if (active)
          setError(
            reason instanceof Error
              ? reason.message
              : "暂时无法读取可添加的步骤",
          );
      });
    return () => {
      active = false;
    };
  }, []);

  let parsedNodes: WorkflowNode[] = [];
  let invalidSteps = false;
  try {
    const value: unknown = JSON.parse(nodes);
    if (
      !Array.isArray(value) ||
      value.some(
        (node) =>
          !node ||
          typeof node !== "object" ||
          typeof node.id !== "string" ||
          typeof node.type !== "string",
      )
    )
      invalidSteps = true;
    else parsedNodes = value as WorkflowNode[];
  } catch {
    invalidSteps = true;
  }
  const setStepList = (steps: WorkflowNode[]) =>
    setNodes(JSON.stringify(steps, null, 2));
  const updateNode = (index: number, fields: Partial<WorkflowNode>) =>
    setStepList(
      parsedNodes.map((node, position) =>
        position === index ? { ...node, ...fields } : node,
      ),
    );
  const moveNode = (index: number, offset: number) => {
    const ordered = [...parsedNodes];
    const [node] = ordered.splice(index, 1);
    ordered.splice(index + offset, 0, node);
    setStepList(ordered);
  };
  const save = async () => {
    setBusy(true);
    setError("");
    try {
      if (invalidSteps)
        throw new Error("高级配置中的步骤格式有误，请修正后再保存。");
      if (!parsedNodes.length) throw new Error("请至少保留一个处理步骤。");
      const result = await workflowApi.saveTemplate({
        key: key.trim(),
        name: name.trim(),
        nodes: parsedNodes,
      });
      onSaved(result);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "流程模板保存失败");
    } finally {
      setBusy(false);
    }
  };
  const independentCopy = key !== source.key;
  return (
    <section className="panel workflow-form workflow-template-editor">
      <header className="workflow-heading">
        <div>
          <h2 ref={headingRef} tabIndex={-1}>
            编辑流程模板
          </h2>
          <p>
            流程模板决定先做什么、后做什么。类别、模型和处理参数保存在“类别与模型方案”中。
          </p>
          <p>
            {independentCopy
              ? "本次保存为独立模板的第 1 版。"
              : "正在编辑第 " +
                source.version +
                " 版，保存后会生成新版本，已有版本保留。"}
          </p>
        </div>
        <button type="button" className="button" onClick={onClose}>
          关闭
        </button>
      </header>
      {error && (
        <p role="alert" className="validation invalid">
          {error}
        </p>
      )}
      <div className="workflow-fields">
        <label>
          流程模板名称
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
          />
        </label>
      </div>
      <div className="workflow-actions">
        <button
          type="button"
          className="button"
          disabled={busy || independentCopy}
          onClick={() => {
            setKey("workflow-" + crypto.randomUUID());
            setName(name + "（副本）");
          }}
        >
          复制为独立模板
        </button>
      </div>
      <section className="workflow-form-section">
        <h3>按顺序执行的步骤</h3>
        <p>
          每一步都列出用途。需要时可以调整顺序、移除或添加步骤，保存时会检查组合是否可执行。
        </p>
        {invalidSteps && (
          <p className="validation invalid">
            高级配置中的步骤需要修正，当前编辑内容已保留。
          </p>
        )}
        <ol className="workflow-step-list">
          {parsedNodes.map((node, index) => (
            <li key={index}>
              <strong>
                {index + 1}. {nodeLabel(node.type)}
              </strong>
              <p>{nodeDescription(node.type)}</p>
              <div className="workflow-actions">
                <button
                  type="button"
                  className="button"
                  disabled={busy || index === 0}
                  onClick={() => moveNode(index, -1)}
                >
                  上移
                </button>
                <button
                  type="button"
                  className="button"
                  disabled={busy || index === parsedNodes.length - 1}
                  onClick={() => moveNode(index, 1)}
                >
                  下移
                </button>
                <button
                  type="button"
                  className="button"
                  disabled={busy}
                  onClick={() =>
                    setStepList(
                      parsedNodes.filter(
                        (_item, position) => position !== index,
                      ),
                    )
                  }
                >
                  移除
                </button>
              </div>
            </li>
          ))}
        </ol>
        <div className="workflow-actions">
          <label>
            选择要添加的步骤
            <select
              value={addingType}
              disabled={busy || invalidSteps || !registry.length}
              onChange={(event) => setAddingType(event.target.value)}
            >
              {registry.map((item) => (
                <option key={item.type} value={item.type}>
                  {nodeLabel(item.type)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="button"
            disabled={busy || invalidSteps || !addingType}
            onClick={() => {
              const contract = registry.find(
                (item) => item.type === addingType,
              );
              if (contract)
                setStepList([
                  ...parsedNodes,
                  {
                    id: "step-" + crypto.randomUUID(),
                    type: contract.type,
                    scope: contract.scope,
                    version: contract.version,
                    params: {},
                    inputs: {},
                    routes: {},
                  },
                ]);
            }}
          >
            添加步骤
          </button>
        </div>
        {addingType && <p className="hint">{nodeDescription(addingType)}</p>}
      </section>
      <details className="workflow-advanced">
        <summary>高级配置</summary>
        <p>
          内部标识、步骤类型和数据连接由系统使用。已有设置会完整保留，可在这里按需调整。
        </p>
        <label>
          内部标识（高级）
          <input
            value={key}
            onChange={(event) => setKey(event.target.value)}
            maxLength={64}
            pattern="[a-zA-Z0-9_-]+"
          />
        </label>
        {parsedNodes.map((node, index) => (
          <fieldset className="workflow-endpoint" key={index}>
            <legend>
              第 {index + 1} 步 · {nodeLabel(node.type)}
            </legend>
            <div className="workflow-fields">
              <label>
                步骤内部标识
                <input
                  value={node.id}
                  onChange={(event) =>
                    updateNode(index, { id: event.target.value })
                  }
                />
              </label>
              <label>
                步骤类型
                <select
                  value={node.type}
                  onChange={(event) => {
                    const contract = registry.find(
                      (item) => item.type === event.target.value,
                    );
                    if (contract)
                      updateNode(index, {
                        type: contract.type,
                        scope: contract.scope,
                        version: contract.version,
                      });
                  }}
                >
                  {!registry.some((item) => item.type === node.type) && (
                    <option value={node.type}>
                      {nodeLabel(node.type)}（{node.type}）
                    </option>
                  )}
                  {registry.map((item) => (
                    <option key={item.type} value={item.type}>
                      {nodeLabel(item.type)}（{item.type}）
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p>
              {scopeLabel(node.scope)} · 执行器第 {node.version} 版 ·{" "}
              <code>{node.scope}</code>
            </p>
          </fieldset>
        ))}
        <label>
          步骤参数与数据连接（JSON）
          <textarea
            aria-label="模板步骤 JSON"
            rows={20}
            value={nodes}
            onChange={(event) => setNodes(event.target.value)}
            spellCheck={false}
          />
        </label>
      </details>
      <button
        type="button"
        className="button primary"
        disabled={busy || !name.trim() || !key.trim()}
        onClick={() => void save()}
      >
        {busy ? "正在保存…" : "保存流程模板"}
      </button>
    </section>
  );
}
