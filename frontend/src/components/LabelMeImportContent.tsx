import { useEffect, useRef, useState } from "react";
import { labelmeApi, type LabelMePreview } from "../api/labelme";

export function LabelMeImportContent({
  onImported,
  onClose,
}: {
  onImported: () => void;
  onClose: () => void;
}) {
  const [file, setFile] = useState<File>();
  const [name, setName] = useState("");
  const [task, setTask] = useState<"detect" | "segment">("segment");
  const [split, setSplit] = useState("train");
  const [preview, setPreview] = useState<LabelMePreview>();
  const [mapping, setMapping] = useState<Record<string, string | null>>({});
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const mounted = useRef(true);
  useEffect(
    () => () => {
      mounted.current = false;
    },
    [],
  );
  const inspect = async () => {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const result = await labelmeApi.preview(file, task, split);
      if (!mounted.current) return;
      setPreview(result);
      setMapping(
        Object.fromEntries(
          result.labels.map((label) => [label.name, label.name]),
        ),
      );
      setConfirmed(false);
    } catch (failure) {
      if (mounted.current)
        setError(failure instanceof Error ? failure.message : "预览失败");
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  const importDataset = async () => {
    if (!file || !preview || !confirmed) return;
    setBusy(true);
    setError("");
    try {
      await labelmeApi.import(file, task, split, name.trim(), mapping);
      onImported();
      onClose();
    } catch (failure) {
      if (mounted.current)
        setError(failure instanceof Error ? failure.message : "导入失败");
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  return (
    <section className="workflow-form">
      <h2>LabelMe 导入预览与类别映射</h2>
      <p>
        上传包含 JSON 与原图的 ZIP。导入后标签保持待审状态，母场景和 split
        关系会保留。
      </p>
      {error && (
        <p role="alert" className="validation invalid">
          {error}
        </p>
      )}
      <fieldset disabled={busy}>
        <div className="workflow-fields">
          <label>
            数据集名称
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label>
            任务
            <select
              value={task}
              onChange={(event) => {
                setTask(event.target.value as "detect" | "segment");
                setPreview(undefined);
              }}
            >
              <option value="segment">实例分割</option>
              <option value="detect">目标检测</option>
            </select>
          </label>
          <label>
            默认 split
            <select
              value={split}
              onChange={(event) => {
                setSplit(event.target.value);
                setPreview(undefined);
              }}
            >
              <option value="train">train</option>
              <option value="val">val</option>
              <option value="test">test</option>
            </select>
          </label>
          <label>
            LabelMe ZIP
            <input
              type="file"
              accept=".zip,application/zip"
              onChange={(event) => {
                setFile(event.target.files?.[0]);
                setPreview(undefined);
              }}
            />
          </label>
        </div>
        <button
          className="button"
          disabled={!file || busy}
          onClick={() => void inspect()}
        >
          预览图片、格式和类别
        </button>
        {preview && (
          <>
            <p>
              {preview.image_count} 张图片 · {preview.shape_count} 个源形状
            </p>
            {preview.issues.map((issue, index) => (
              <p
                key={index}
                className={
                  issue.level === "error" ? "validation invalid" : "hint"
                }
              >
                {issue.message ?? JSON.stringify(issue)}
              </p>
            ))}
            {preview.labels.map((label) => (
              <div className="workflow-rule" key={label.name}>
                <strong>
                  {label.name} · {label.count}
                </strong>
                <label className="workflow-check">
                  <input
                    type="checkbox"
                    checked={mapping[label.name] === null}
                    onChange={(event) => {
                      setMapping({
                        ...mapping,
                        [label.name]: event.target.checked ? null : label.name,
                      });
                      setConfirmed(false);
                    }}
                  />
                  忽略此源类别
                </label>
                {mapping[label.name] !== null && (
                  <label>
                    目标类别名称
                    <input
                      value={mapping[label.name] ?? ""}
                      onChange={(event) => {
                        setMapping({
                          ...mapping,
                          [label.name]: event.target.value,
                        });
                        setConfirmed(false);
                      }}
                    />
                  </label>
                )}
              </div>
            ))}
            <details>
              <summary>图片与源场景预览</summary>
              <div className="workflow-image-picker">
                {preview.images.map((image) => (
                  <p key={image.json_file}>
                    {image.image_file} · {image.width} × {image.height} ·{" "}
                    {image.split} · group {image.source_group_id}
                  </p>
                ))}
              </div>
            </details>
            <label className="workflow-check">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(event) => setConfirmed(event.target.checked)}
              />
              已核对所有源类别的映射或忽略决定
            </label>
            <button
              className="button primary"
              disabled={
                busy ||
                !name.trim() ||
                !confirmed ||
                Object.values(mapping).some(
                  (value) => value !== null && !value.trim(),
                ) ||
                preview.issues.some((issue) => issue.level === "error")
              }
              onClick={() => void importDataset()}
            >
              按已确认映射导入待审数据
            </button>
          </>
        )}
      </fieldset>
    </section>
  );
}
