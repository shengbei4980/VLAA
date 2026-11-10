import { useState } from "react";
import { reviewApi, type DatasetReviewStatus } from "../api/reviews";

export function DatasetReviewStatusPanel({
  datasetId,
  onOpenImage,
}: {
  datasetId: string;
  onOpenImage: (id: string) => void;
}) {
  const [status, setStatus] = useState<DatasetReviewStatus>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const refresh = async () => {
    setBusy(true);
    setError("");
    try {
      setStatus(await reviewApi.status(datasetId));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "审核汇总读取失败");
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="workflow-dataset-review">
      <button className="button" disabled={busy} onClick={() => void refresh()}>
        {busy ? "正在读取…" : status ? "刷新人工审核状态" : "查看人工审核状态"}
      </button>
      {error && (
        <p className="validation invalid" role="alert">
          {error}
        </p>
      )}
      {status && (
        <>
          <p>
            已确认 {status.confirmed} · 待审 {status.pending} · 确认无目标{" "}
            {status.confirmed_empty} · 总计 {status.total}
          </p>
          <div className="workflow-image-picker">
            {status.items.map((item) => (
              <article className="workflow-item" key={item.image_id}>
                <div>
                  <strong>{item.image_id}</strong>
                  <small>
                    {item.review_status} · {item.annotation_count} 个标签
                    {item.contains_auto ? " · 含自动来源" : ""}
                  </small>
                </div>
                <button
                  className="button"
                  onClick={() => onOpenImage(item.image_id)}
                >
                  打开整图复核
                </button>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
