import { useEffect, useState, type FormEvent } from "react";
import { storageApi } from "../api/storage";

export function ExportLocationsPanel({ onSaved }: { onSaved?: () => void }) {
  const [jsonPath, setJsonPath] = useState("");
  const [imagesPath, setImagesPath] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const readPaths = async (active = () => true) => {
    setLoading(true);
    setError("");
    try {
      const paths = await storageApi.manualExports();
      if (!active()) return;
      setJsonPath((current) => current || paths.json_path);
      setImagesPath((current) => current || paths.images_path);
    } catch (failure) {
      if (active())
        setError(
          failure instanceof Error ? failure.message : "导出位置读取失败",
        );
    } finally {
      if (active()) setLoading(false);
    }
  };

  useEffect(() => {
    let active = true;
    void readPaths(() => active);
    return () => {
      active = false;
    };
  }, []);

  const save = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const paths = await storageApi.saveManualExports({
        json_path: jsonPath.trim(),
        images_path: imagesPath.trim(),
      });
      setJsonPath(paths.json_path);
      setImagesPath(paths.images_path);
      setMessage("导出位置已保存，下次导出会写入对应文件夹。");
      onSaved?.();
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "导出位置保存失败，请检查文件夹路径。",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="workflow-form-section">
      <h3>JSON 与标注图片的保存位置</h3>
      <p>分别填写电脑上的完整文件夹路径。已有导出仍可在原位置访问。</p>
      <form onSubmit={(event) => void save(event)}>
        <div className="workflow-fields">
          <label>
            JSON 保存文件夹
            <input
              value={jsonPath}
              disabled={loading || busy}
              onChange={(event) => setJsonPath(event.target.value)}
              placeholder={"例如 D:\\标注结果\\JSON"}
            />
          </label>
          <label>
            标注预览图 PNG 保存文件夹
            <input
              value={imagesPath}
              disabled={loading || busy}
              onChange={(event) => setImagesPath(event.target.value)}
              placeholder={"例如 D:\\标注结果\\图像"}
            />
          </label>
        </div>
        <button
          className="button"
          disabled={loading || busy || !jsonPath.trim() || !imagesPath.trim()}
        >
          {loading ? "读取导出位置…" : busy ? "正在保存位置…" : "保存导出位置"}
        </button>
      </form>
      {error && (
        <div className="validation invalid" role="alert">
          <p>{error}</p>
          <button
            className="button"
            disabled={loading || busy}
            onClick={() => void readPaths()}
          >
            重新读取位置
          </button>
        </div>
      )}
      {message && (
        <p className="validation" role="status">
          {message}
        </p>
      )}
    </section>
  );
}
