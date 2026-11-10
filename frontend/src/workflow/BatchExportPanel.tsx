import { useState } from "react";
import { apiUrl } from "../api/client";
import { workflowApi, type BatchAnnotationExport } from "../api/workflows";

export function BatchExportPanel({
  runId,
  readyImages,
}: {
  runId: string;
  readyImages: number;
}) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<BatchAnnotationExport>();
  const [error, setError] = useState("");

  const save = async () => {
    setBusy(true);
    setError("");
    try {
      setResult(await workflowApi.exportBatch(runId));
    } catch (failure) {
      setError(
        failure instanceof Error ? failure.message : "批量导出失败，请重试。",
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <section
      className="panel workflow-form"
      aria-labelledby="workflow-batch-export-title"
    >
      <h2 id="workflow-batch-export-title">批量保存标注结果</h2>
      <p>
        一次保存本批已完成的 LabelMe JSON 和标注预览 PNG，并生成可下载的 ZIP
        文件。
      </p>
      <p className="hint">
        导出本批已保存结果，尚未处理完成的图片可稍后再次导出。复核页的修改需先保存草稿，草稿会保留待复核状态。
      </p>
      <button
        className="button primary"
        disabled={busy || readyImages === 0}
        onClick={() => void save()}
      >
        {busy
          ? "正在逐图生成 JSON 与 PNG…"
          : result
            ? "重新批量保存 JSON 与标注图片"
            : "批量保存 JSON 与标注图片"}
      </button>
      {!readyImages && <p>图片完成自动标注与检查后，即可在这里批量保存。</p>}
      {busy && (
        <p role="status">系统正在逐张处理原尺寸图片，请等待本次导出完成。</p>
      )}
      {error && (
        <p className="validation invalid" role="alert">
          {error}
        </p>
      )}
      {result && (
        <div className="workflow-choice-summary" role="status">
          <p>
            本批 {result.total_images} 张，已保存 {result.exported_images}{" "}
            张，其中草稿 {result.pending_images} 张。失败 {result.failed_images}{" "}
            张，暂未导出 {result.skipped_images} 张。
          </p>
          <p>
            导出包保存位置 <code>{result.saved_path}</code>
          </p>
          <a
            className="button"
            href={apiUrl(result.download_url)}
            target="_blank"
            rel="noopener noreferrer"
            download={result.file_name}
          >
            下载本批 JSON 与标注图片 ZIP
          </a>
          <details>
            <summary>查看逐图保存位置与未导出原因</summary>
            <ul>
              {result.results.map((item) => (
                <li key={item.image_id}>
                  <strong>{item.image_id}</strong>
                  {item.status === "exported" ? (
                    <>
                      <span>
                        {" "}
                        ·{" "}
                        {item.export_state === "draft"
                          ? "已保存草稿"
                          : "已保存已确认标注"}
                      </span>
                      {item.json_path && (
                        <p>
                          JSON <code>{item.json_path}</code>
                        </p>
                      )}
                      {item.image_path && (
                        <p>
                          PNG <code>{item.image_path}</code>
                        </p>
                      )}
                    </>
                  ) : (
                    <span>
                      {" "}
                      · {item.status === "failed"
                        ? "保存失败"
                        : "暂未导出"} ·{" "}
                      {item.error ??
                        item.reason ??
                        "图片尚未准备好，请稍后再次导出。"}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          </details>
        </div>
      )}
    </section>
  );
}
