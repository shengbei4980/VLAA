import { useEffect, useRef, useState, type FormEvent } from "react";
import { buildHybridPayload, hybridApi, type HybridResult, type HybridStatus } from "../api/hybrid";
import type { AppLocale } from "../locale";
import "../hybrid.css";

type Translate = (zh: string, en: string) => string;
type Role = "gpt" | "deepseek";
type Mode = "drafts" | "online";

function FileField({ name, label, accept, required = true, help }: {
  name: string; label: string; accept: string; required?: boolean; help?: string;
}) {
  return <label className="hybrid-field">
    {label}
    <input type="file" name={name} accept={accept} required={required} />
    {help && <small>{help}</small>}
  </label>;
}

function EndpointFields({ role, t }: { role: Role; t: Translate }) {
  const [source, setSource] = useState("custom");
  return <fieldset className="hybrid-fieldset hybrid-role hybrid-stack">
    <legend>{role === "gpt" ? t("GPT · 多边形", "GPT · polygons") : t("DeepSeek · 候选框", "DeepSeek · candidate boxes")}</legend>
    <label className="hybrid-field">
      {t("接入方式", "Endpoint configuration")}
      <select name={role + "_source"} value={source} onChange={event => setSource(event.target.value)}>
        <option value="custom">{t("本次填写", "Enter for this run")}</option>
        <option value="proposer">{t("使用本机已保存的候选提出者", "Use saved proposer")}</option>
        <option value="reviewer">{t("使用本机已保存的独立视觉检查者", "Use saved reviewer")}</option>
      </select>
    </label>
    {source === "custom" && <>
      <label className="hybrid-field">API Base URL
        <input name={role + "_url"} type="url" placeholder="https://…/v1" required autoComplete="off" />
      </label>
      <label className="hybrid-field">{t("视觉模型名称", "Vision model name")}
        <input name={role + "_model"} type="text" required autoComplete="off" spellCheck={false} />
      </label>
      <label className="hybrid-field">{t("API Key（如需要）", "API key (if required)")}
        <input name={role + "_key"} type="password" autoComplete="off" spellCheck={false} />
      </label>
    </>}
    <label className="hybrid-field">{t("请求协议", "Request protocol")}
      <select name={role + "_protocol"} defaultValue={role === "gpt" ? "responses" : "chat_completions"}>
        <option value="responses">Responses</option><option value="chat_completions">Chat Completions</option>
      </select>
    </label>
    {source === "custom" && <label className="hybrid-field">{t("模型输出坐标", "Model output coordinates")}
      <select name={role + "_coords"} defaultValue="pixel">
        <option value="pixel">{t("图像像素", "Image pixels")}</option>
        <option value="normalized_1">0–1</option><option value="normalized_1000">0–1000</option>
      </select>
    </label>}
  </fieldset>;
}

const classes = [
  ["水体", "Water", "#2563eb"], ["裸地", "Bare ground", "#b45309"],
  ["草本", "Grass", "#84cc16"], ["木本", "Woody vegetation", "#166534"],
  ["耕地", "Cropland", "#eab308"], ["建筑及硬化地表", "Built surfaces", "#ef4444"],
] as const;
const sources = [
  ["GPT", "GPT", "#2563eb", "gpt_pixels"], ["SAM", "SAM", "#f59e0b", "sam_pixels"],
  ["RF", "RF", "#16a34a", "rf_pixels"], ["待裁决冲突", "Conflicts", "#d946ef", "conflict_pixels"],
] as const;

function useBlobUrl(blob: Blob | undefined) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!blob) { setUrl(""); return; }
    const value = URL.createObjectURL(blob);
    setUrl(value);
    return () => URL.revokeObjectURL(value);
  }, [blob]);
  return url;
}

function DraftResult({ result, rgb, t }: { result: HybridResult; rgb: File; t: Translate }) {
  const [layer, setLayer] = useState<"class" | "source">("class");
  const [blob] = useState(() => new Blob([JSON.stringify(result.labelme, null, 2)], { type: "application/json" }));
  const download = useBlobUrl(blob);
  const original = useBlobUrl(rgb);
  const jsonName = rgb.name.replace(/\.[^.]+$/, "") + ".json";
  return <section className="hybrid-card hybrid-stack" aria-label={t("草稿结果", "Draft result")}>
    <div className="hybrid-tools">
      <h2>{t("待校对草稿", "Draft for review")}</h2>
      <a className="hybrid-primary" href={download} download={jsonName}>{t("下载 LabelMe JSON", "Download LabelMe JSON")}</a>
    </div>
    <div className="hybrid-tools">
      <label className="hybrid-field">{t("预览图层", "Preview layer")}
        <select value={layer} onChange={event => setLayer(event.target.value as typeof layer)}>
          <option value="class">{t("地表类别", "Land cover")}</option>
          <option value="source">{t("标注来源", "Label source")}</option>
        </select>
      </label>
      <span>{t("有效范围", "Valid area")} · {result.stats.eligible_pixels.toLocaleString()} px</span>
    </div>
    <div className="hybrid-figures">
      <figure><img src={original || undefined} alt={t("输入真彩色图", "Input RGB tile")} /><figcaption>{rgb.name}</figcaption></figure>
      <figure>
        <img src={"data:image/png;base64," + (layer === "class" ? result.preview_png : result.source_png)} alt={layer === "class" ? t("类别预览", "Land-cover preview") : t("来源预览", "Source preview")} />
        <figcaption>{layer === "class" ? t("类别叠加", "Land cover overlay") : t("来源叠加", "Label source overlay")}</figcaption>
      </figure>
    </div>
    <div className="hybrid-legend">
      {layer === "class"
        ? classes.map(([zh, en, color]) => <span key={en}><i style={{ background: color }} />{t(zh, en)}</span>)
        : sources.map(([zh, en, color, key]) => <span key={key}><i style={{ background: color }} />{t(zh, en)} · {result.stats[key].toLocaleString()} px</span>)}
    </div>
    <small>{t("将同名 JSON 和原图一起导入数据集进行人工校对。优先查看 RF 新增区、草本边界和冲突像素。", "Import the JSON with its matching image for human review. Check RF additions, grass boundaries and conflicts first.")}</small>
    {result.warnings.length > 0 && <ul>{result.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul>}
  </section>;
}

export function HybridView({ locale }: { locale: AppLocale }) {
  const t: Translate = (zh, en) => locale === "zh" ? zh : en;
  const [mode, setMode] = useState<Mode>("drafts");
  const [status, setStatus] = useState<HybridStatus>();
  const [statusError, setStatusError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const [completed, setCompleted] = useState<{ result: HybridResult; rgb: File }>();
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);
  useEffect(() => {
    let active = true;
    setStatus(undefined);
    setStatusError("");
    hybridApi.status().then(data => { if (active) setStatus(data); })
      .catch(reason => { if (active) setStatusError(String(reason instanceof Error ? reason.message : reason)); });
    return () => { active = false; };
  }, [refresh]);

  async function run(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    if (inFlight.current || !form.reportValidity()) return;
    const payload = buildHybridPayload(new FormData(form), mode);
    inFlight.current = true;
    setBusy(true);
    setError("");
    setCompleted(undefined);
    try {
      const result = await hybridApi.run(payload);
      if (mounted.current) setCompleted({ result, rgb: payload.get("rgb") as File });
    } catch (reason) {
      if (mounted.current) setError(String(reason instanceof Error ? reason.message : reason));
    } finally {
      inFlight.current = false;
      if (mounted.current) setBusy(false);
    }
  }

  return <main className="hybrid-page">
    <header><h1>{t("遥感联合标注", "Hybrid land-cover annotation")}</h1>
      <p>{t("GPT 绘制多边形，DeepSeek 提框并由 SAM 分割，再用 Sentinel-2 十波段随机森林填补剩余空白。", "GPT draws polygons, DeepSeek proposes boxes for SAM, and a ten-band Sentinel-2 random forest fills the remaining gaps.")}</p>
    </header>
    <form className="hybrid-form" onSubmit={run} onChange={() => { setCompleted(undefined); setError(""); }}>
      <fieldset className="hybrid-fieldset hybrid-stack" disabled={busy}>
        <section className="hybrid-card hybrid-stack">
          <h2>{t("输入图块", "Input tile")}</h2>
          <div className="hybrid-grid">
            <FileField name="rgb" label={t("真彩色图块（PNG / JPEG）", "RGB tile (PNG / JPEG)")} accept="image/png,image/jpeg" help={t("保留原尺寸，单张最多 4,194,304 个像素。", "Original dimensions, up to 4,194,304 pixels per tile.")} />
            <FileField name="raster" label={t("十波段地表反射率 GeoTIFF", "Ten-band surface reflectance GeoTIFF")} accept=".tif,.tiff" help="B02 · B03 · B04 · B05 · B06 · B07 · B08 · B8A · B11 · B12 (≤512 MiB)" />
            <FileField name="roi" label={t("有效区域掩膜（可选 PNG）", "ROI mask (optional PNG)")} accept="image/png" required={false} help={t("与 RGB 图块同尺寸，非零像素有效。", "Match the RGB tile dimensions. Nonzero pixels are eligible.")} />
          </div>
          <div className="hybrid-grid">
            <label className="hybrid-field">{t("左上角列偏移 X", "Left column offset X")}<input type="number" name="x" min="0" step="1" defaultValue="0" required /></label>
            <label className="hybrid-field">{t("左上角行偏移 Y", "Top row offset Y")}<input type="number" name="y" min="0" step="1" defaultValue="0" required /></label>
          </div>
          <small>{t("使用整景 TIFF 时填写图块在整景中的像素坐标。使用已对齐的小块 TIFF 时，两项均填 0。", "For a full-scene TIFF, enter the RGB tile’s pixel offset in that scene. For an aligned tile TIFF, use 0 for both.")}</small>
        </section>
        <section className="hybrid-card hybrid-stack">
          <h2>{t("生成方式", "Draft generation")}</h2>
          <div className="hybrid-choice">
            {(["drafts", "online"] as const).map(value => <label key={value}><input type="radio" name="mode" checked={value === mode} onChange={() => setMode(value)} />
              {value === "drafts" ? t("导入两路已有草稿", "Import two existing drafts") : t("从原图调用视觉模型", "Generate with vision endpoints")}
            </label>)}
          </div>
          {mode === "drafts" ? <div className="hybrid-grid">
            <FileField name="gpt_draft" label={t("GPT 多边形 JSON", "GPT polygon JSON")} accept=".json" />
            <FileField name="sam_draft" label={t("DeepSeek＋SAM 掩膜 JSON", "DeepSeek + SAM mask JSON")} accept=".json" />
          </div> : <>
            <div className="hybrid-grid"><EndpointFields role="gpt" t={t} /><EndpointFields role="deepseek" t={t} /></div>
            <label><input type="checkbox" name="allow_remote" required /> {t("将本张 RGB 图发送到上述两个视觉端点。", "Send this RGB tile to both vision endpoints.")}</label>
          </>}
          <div className="hybrid-status" aria-live="polite">
            {status ? <>
              <span title={status.rf_detail}><i className={status.rf_ready ? "ready" : ""} />{t("十波段 RF", "Ten-band RF")} · {status.rf_ready ? t("可用", "Ready") : status.rf_detail}</span>
              <span title={status.sam_detail}><i className={status.sam_ready ? "ready" : ""} />SAM · {status.sam_ready ? t("可用", "Ready") : status.sam_detail}</span>
            </> : <span>{statusError || t("正在检查本地模型…", "Checking local models…")}</span>}
            <button type="button" className="hybrid-link" onClick={() => setRefresh(n => n + 1)}>{t("刷新状态", "Refresh status")}</button>
          </div>
        </section>
        <div className="hybrid-tools"><button className="hybrid-primary" type="submit" disabled={!status?.rf_ready || (mode === "online" && !status.sam_ready)}>
          {busy ? t("正在生成…", "Generating…") : t("生成待校对草稿", "Generate review draft")}
        </button>
        {busy && <span role="status">{t("请保持此页面打开。", "Keep this page open.")}</span>}</div>
      </fieldset>
      {error && <div className="hybrid-alert" role="alert">{error}</div>}
    </form>
    {completed && <DraftResult result={completed.result} rgb={completed.rgb} t={t} />}
  </main>;
}
