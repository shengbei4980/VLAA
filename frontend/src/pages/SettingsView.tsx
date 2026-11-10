import { useEffect, useMemo, useState, type FormEvent, type ReactNode } from "react";
import { api } from "../api/client";
import type { AppLocale } from "../locale";
import { saveLocale } from "../locale";
import type { LLMAuthScheme, LLMSettings, SamSettings, TrainingDevice } from "../types";
import { VisionSettingsPanel } from "./VisionSettingsPanel";
import { StorageSettingsPanel } from "./StorageSettingsPanel";

export type SettingsTab = "llm" | "sam" | "vision" | "storage" | "language";

type LlmFormState = LLMSettings & { api_key: string };

const LLM_AUTH_SCHEMES: LLMAuthScheme[] = ["bearer", "header", "raw_authorization"];

const copy = {
  zh: {
    eyebrow: "WORKSPACE SETTINGS",
    title: "设置",
    subtitle: "按用途连接任务助手、视觉标注与轮廓分割模型，并设置保存位置。首次自动标注请先配置“视觉模型接入”。",
    local: "本地",
    llmNav: "Agent LLM",
    llmNavDescription: "任务对话、数据查询与流程安排",
    samNav: "SAM 辅助",
    samNavDescription: "根据框选或点提示生成对象轮廓",
    languageNav: "语言",
    languageNavDescription: "工作台显示偏好",
    llmTitle: "任务助手（Agent LLM）",
    llmSubtitle: "用于智能体页面的任务对话、查询数据与安排工作流程。图片中的对象识别和标注检查请在“视觉模型接入”中配置。",
    enabled: "启用大模型任务助手",
    enabledHint: "开启后通过所填接口调用语言模型。关闭时使用本地规则处理支持的任务指令。",
    provider: "服务类型（Provider）",
    model: "模型",
    apiBase: "API Base URL",
    apiKey: "API Key",
    apiKeyKeep: "已配置密钥：留空则保留原值",
    apiKeyOptional: "可选：本地服务可不填",
    authScheme: "鉴权方式",
    authHeaderName: "自定义 Header 名",
    authHeaderHint: "选择自定义 Header 鉴权时填写，例如 X-Api-Key。",
    temperature: "回答随机性（Temperature）",
    timeout: "请求超时（秒）",
    advanced: "高级",
    plannerTitle: "回答方式",
    plannerSubtitle: "预设会调整回答随机性（Temperature），保存后生效。",
    cost: "更稳妥",
    balanced: "均衡",
    quality: "更灵活",
    test: "测试连接",
    testing: "测试中…",
    save: "保存 LLM 设置",
    saved: "LLM 设置已保存",
    llmReady: "已启用",
    llmLocal: "本地规则",
    llmFooter: "测试连接使用当前表单，保存后在智能体页面生效。已有 API Key 可留空保留。",
    samTitle: "SAM 辅助标注",
    samSubtitle: "根据框选区域或正负点生成对象轮廓，辅助修边，并为分割工作流生成掩膜。生成后请检查边界再保存。",
    samEnabled: "启用 SAM 辅助",
    modelSam: "模型权重或模型 ID",
    modelHint: "可填写本地 .pt 路径或 Ultralytics 可识别的模型名，例如 sam_b.pt。",
    device: "推理设备",
    imageSize: "推理尺寸",
    fallback: "未配置模型时",
    box: "使用矩形区域建议（待人工修边）",
    disabled: "要求先配置模型",
    saveSam: "保存 SAM 设置",
    savedSam: "SAM 设置已保存",
    ready: "已配置真实模型",
    fallbackReady: "使用框形建议",
    disabledStatus: "已关闭",
    samFooter: "用于分割任务（segment）的轮廓标注，生成结果在人工确认后应用。",
    languageTitle: "语言设置",
    languageSubtitle: "切换工作台界面语言。偏好只保存在本机浏览器中。",
    displayLanguage: "显示语言",
    chinese: "简体中文",
    english: "English",
    applied: "已立即应用",
    loading: "正在读取设置…",
    error: "设置读取失败",
    availableDevices: "当前设备",
    noDevice: "没有额外 GPU，仍可手动填写设备标识。",
    cuda: "CUDA",
    mps: "Apple GPU (MPS)",
    cpu: "CPU",
    auto: "自动选择",
    show: "显示",
    hide: "隐藏",
    testMissingApiBase: "请填写 API Base URL（例如 https://api.openai.com/v1）",
    testSkippedNoKey: "未配置 API Key（可选），已跳过远端连接测试。需要调用 LLM 时再填写密钥即可。",
    testConnectError: "无法连接到服务器，请检查 API Base URL。",
    testTimeout: "请求超时，请检查网络、代理或增大 Request Timeout 后重试",
    testRequestError: "请求失败，请检查 API Base URL 和网络连接。",
    testOk: "连接成功，Chat Completions 可正常使用。",
    testHttpError: "API 返回 HTTP {status}",
  },
  en: {
    eyebrow: "WORKSPACE SETTINGS",
    title: "Settings",
    subtitle: "Connect models for task assistance, visual annotation and object outlines, then choose where to save results. Start with Vision model access for automatic annotation.",
    local: "Local",
    llmNav: "Agent LLM",
    llmNavDescription: "Task chat, data queries and workflow planning",
    samNav: "SAM assist",
    samNavDescription: "Create outlines from boxes or point prompts",
    languageNav: "Language",
    languageNavDescription: "Workspace display preference",
    llmTitle: "Task assistant (Agent LLM)",
    llmSubtitle: "Powers task conversations, data queries and workflow planning on the Agent page. Configure image recognition and annotation checks under Vision model access.",
    enabled: "Enable LLM provider",
    enabledHint: "When enabled, the assistant calls your model endpoint. When off, local rules handle supported task instructions.",
    provider: "Provider",
    model: "Model",
    apiBase: "API Base URL",
    apiKey: "API Key",
    apiKeyKeep: "Key configured — leave blank to keep it",
    apiKeyOptional: "Optional for local endpoints",
    authScheme: "Auth scheme",
    authHeaderName: "Custom header name",
    authHeaderHint: "Required when auth_scheme is header, e.g. X-Api-Key",
    temperature: "Temperature",
    timeout: "Request timeout (seconds)",
    advanced: "Advanced",
    plannerTitle: "Response style",
    plannerSubtitle: "Presets adjust response randomness (Temperature). Save to apply.",
    cost: "More consistent",
    balanced: "Balanced",
    quality: "More flexible",
    test: "Test connection",
    testing: "Testing…",
    save: "Save LLM settings",
    saved: "LLM settings saved",
    llmReady: "Enabled",
    llmLocal: "Local rules",
    llmFooter: "Connection tests use this form. Save to apply it on the Agent page. Leave an existing API key blank to keep it.",
    samTitle: "SAM assisted annotation",
    samSubtitle: "Generate object outlines from boxes or positive and negative points for editing and segmentation workflows. Review the boundaries before saving.",
    samEnabled: "Enable SAM assist",
    modelSam: "Model checkpoint or model id",
    modelHint: "Use a local .pt path or an Ultralytics model name such as sam_b.pt.",
    device: "Inference device",
    imageSize: "Inference size",
    fallback: "When no model is configured",
    box: "Use rectangular regions for manual refinement",
    disabled: "Require a configured model",
    saveSam: "Save SAM settings",
    savedSam: "SAM settings saved",
    ready: "Real model configured",
    fallbackReady: "Box proposals",
    disabledStatus: "Disabled",
    samFooter: "Used for object outlines in segmentation tasks. Proposals are applied after human confirmation.",
    languageTitle: "Language settings",
    languageSubtitle: "Switch the workspace language. The preference is stored in this browser only.",
    displayLanguage: "Display language",
    chinese: "简体中文",
    english: "English",
    applied: "Applied immediately",
    loading: "Loading settings…",
    error: "Could not load settings",
    availableDevices: "Available devices",
    noDevice: "No extra GPU was detected; you can still enter a device id manually.",
    cuda: "CUDA",
    mps: "Apple GPU (MPS)",
    cpu: "CPU",
    auto: "Auto",
    show: "Show",
    hide: "Hide",
    testMissingApiBase: "Enter an API Base URL (for example https://api.openai.com/v1).",
    testSkippedNoKey: "No API key configured (optional). Remote probe skipped; add a key before calling the LLM.",
    testConnectError: "Could not reach the server. Check the API Base URL.",
    testTimeout: "Request timed out. Check network/proxy or increase Request Timeout.",
    testRequestError: "Request failed. Check the API Base URL and network.",
    testOk: "Connection succeeded. Chat Completions is usable.",
    testHttpError: "API returned HTTP {status}",
  },
} as const;

type SettingsText = (typeof copy)[AppLocale];

function localizeLlmTestMessage(
  text: SettingsText,
  result: { message: string; code?: string; http_status?: number | null },
): string {
  switch (result.code) {
    case "missing_api_base":
      return text.testMissingApiBase;
    case "skipped_no_key":
      return text.testSkippedNoKey;
    case "connect_error":
      return text.testConnectError;
    case "timeout":
      return text.testTimeout;
    case "request_error":
      return text.testRequestError;
    case "ok":
      return text.testOk;
    case "http_error":
      return text.testHttpError.replace("{status}", String(result.http_status ?? ""));
    default:
      return result.message;
  }
}

function emptyLlmForm(): LlmFormState {
  return {
    enabled: false,
    provider: "openai-compatible",
    api_base: "",
    model: "gpt-4o-mini",
    temperature: 0.2,
    timeout_seconds: 60,
    api_key_configured: false,
    auth_scheme: "bearer",
    auth_header_name: "",
    api_key: "",
  };
}

export function SettingsView({
  tab,
  locale,
  onLocaleChange,
  onSamSettingsChange,
  onTabChange,
  onStorageSettingsChange,
}: {
  tab: SettingsTab;
  locale: AppLocale;
  onLocaleChange: (locale: AppLocale) => void;
  onSamSettingsChange?: () => void;
  onTabChange: (tab: SettingsTab) => void;
  onStorageSettingsChange?: () => void;
}) {
  const text = copy[locale];
  return (
    <main className="settings-layout">
      <header className="settings-page-head">
        <div>
          <span className="eyebrow">{text.eyebrow}</span>
          <h1>{text.title}</h1>
          <p>{text.subtitle}</p>
        </div>
        <span className="settings-page-badge">{text.local}</span>
      </header>
      <div className="settings-section-layout">
        <nav className="settings-section-nav" aria-label={text.title}>
          <SettingsNavigationItem active={tab === "storage"} icon={<SamIcon />} title={locale === "zh" ? "存储位置" : "Storage locations"} description={locale === "zh" ? "数据集、模型与导出结果的保存位置" : "Save locations for datasets, models and exports"} onClick={() => onTabChange("storage")} />
          <SettingsNavigationItem active={tab === "vision"} icon={<SamIcon />} title={locale === "zh" ? "视觉模型接入" : "Vision model access"} description={locale === "zh" ? "自动找对象、提出标注与独立检查" : "Find objects, propose labels and check annotations"} onClick={() => onTabChange("vision")} />
          <SettingsNavigationItem
            active={tab === "llm"}
            icon={<LlmIcon />}
            title={text.llmNav}
            description={text.llmNavDescription}
            badge={text.local}
            onClick={() => onTabChange("llm")}
          />
          <SettingsNavigationItem
            active={tab === "sam"}
            icon={<SamIcon />}
            title={text.samNav}
            description={text.samNavDescription}
            badge={text.local}
            onClick={() => onTabChange("sam")}
          />
          <SettingsNavigationItem
            active={tab === "language"}
            icon={<LanguageIcon />}
            title={text.languageNav}
            description={text.languageNavDescription}
            onClick={() => onTabChange("language")}
          />
        </nav>
        <div className="settings-section-content">
          {tab === "storage" ? <StorageSettingsPanel onChanged={onStorageSettingsChange} /> : tab === "vision" ? <VisionSettingsPanel /> : tab === "llm" ? (
            <LlmSettingsPanel locale={locale} text={text} />
          ) : tab === "sam" ? (
            <SamSettingsPanel locale={locale} text={text} onSamSettingsChange={onSamSettingsChange} />
          ) : (
            <LanguageSettingsPanel locale={locale} text={text} onLocaleChange={onLocaleChange} />
          )}
        </div>
      </div>
    </main>
  );
}

function SettingsNavigationItem({
  active,
  icon,
  title,
  description,
  badge,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  title: string;
  description: string;
  badge?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`settings-section-nav-item ${active ? "active" : ""}`}
      aria-current={active ? "page" : undefined}
      onClick={onClick}
    >
      <span className="settings-nav-icon">{icon}</span>
      <span className="settings-section-nav-copy">
        <strong>{title}</strong>
        <small>{description}</small>
      </span>
      {badge ? <span className="settings-nav-badge">{badge}</span> : null}
    </button>
  );
}

function LlmSettingsPanel({ text }: { locale: AppLocale; text: SettingsText }) {
  const [form, setForm] = useState<LlmFormState>(emptyLlmForm);
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [preset, setPreset] = useState<"cost" | "balanced" | "quality">("balanced");

  useEffect(() => {
    void api
      .getLlmSettings()
      .then((settings) => setForm({ ...settings, api_key: "" }))
      .catch((reason) => setError(reason instanceof Error ? reason.message : text.error))
      .finally(() => setBusy(false));
  }, [text.error]);

  const status = form.enabled ? text.llmReady : text.llmLocal;

  const save = async (event?: FormEvent) => {
    event?.preventDefault();
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const saved = await api.updateLlmSettings({
        enabled: form.enabled,
        provider: form.provider,
        api_base: form.api_base,
        api_key: form.api_key,
        model: form.model,
        temperature: form.temperature,
        timeout_seconds: form.timeout_seconds,
        auth_scheme: form.auth_scheme,
        auth_header_name: form.auth_scheme === "header" ? form.auth_header_name : "",
      });
      setForm({ ...saved, api_key: "" });
      setMessage(text.saved);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setSaving(false);
    }
  };

  const test = async () => {
    setTesting(true);
    setError("");
    setMessage("");
    try {
      const result = await api.testLlmConnection({
        api_base: form.api_base,
        model: form.model,
        api_key: form.api_key,
        timeout_seconds: form.timeout_seconds,
        auth_scheme: form.auth_scheme,
        auth_header_name: form.auth_scheme === "header" ? form.auth_header_name : "",
      });
      const localized = localizeLlmTestMessage(text, result);
      setMessage(localized);
      if (!result.ok) setError(localized);
      else setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setTesting(false);
    }
  };

  const applyPreset = (next: "cost" | "balanced" | "quality") => {
    setPreset(next);
    const temperature = next === "cost" ? 0.1 : next === "quality" ? 0.9 : 0.4;
    setForm((current) => ({ ...current, temperature }));
  };

  if (busy) return <section className="panel settings-card"><p className="muted settings-loading">{text.loading}</p></section>;

  return (
    <form className="panel settings-card" onSubmit={(event) => void save(event)}>
      <header className="settings-card-head">
        <div className="settings-card-icon"><LlmIcon /></div>
        <div>
          <h2>{text.llmTitle}</h2>
          <p>{text.llmSubtitle}</p>
        </div>
        <span className={`settings-status ${form.enabled ? "ready" : "disabled"}`}>{status}</span>
      </header>
      {error ? <div className="validation invalid"><span>{error}</span></div> : null}
      {message && !error ? <div className="validation valid"><span>{message}</span></div> : null}

      <label className="settings-toggle">
        <input
          type="checkbox"
          checked={form.enabled}
          onChange={(event) => setForm({ ...form, enabled: event.target.checked })}
        />
        <span>
          <strong>{text.enabled}</strong>
          <small>{text.enabledHint}</small>
        </span>
      </label>

      <div className="settings-form-grid">
        <label>
          <span>{text.provider}</span>
          <select value={form.provider} onChange={(event) => setForm({ ...form, provider: event.target.value })}>
            <option value="openai-compatible">openai-compatible</option>
            <option value="openai">openai</option>
            <option value="local">local</option>
          </select>
        </label>
        <label>
          <span>{text.model}</span>
          <input value={form.model} onChange={(event) => setForm({ ...form, model: event.target.value })} />
        </label>
        <label className="settings-span-2">
          <span>{text.apiBase}</span>
          <input
            value={form.api_base}
            placeholder="https://api.openai.com/v1"
            onChange={(event) => setForm({ ...form, api_base: event.target.value })}
          />
        </label>
        <label className="settings-span-2">
          <span>{text.apiKey}</span>
          <div className="settings-key-line">
            <input
              type={showKey ? "text" : "password"}
              value={form.api_key}
              placeholder={form.api_key_configured ? text.apiKeyKeep : text.apiKeyOptional}
              onChange={(event) => setForm({ ...form, api_key: event.target.value })}
              autoComplete="off"
            />
            <button type="button" className="button" onClick={() => setShowKey((value) => !value)}>
              {showKey ? text.hide : text.show}
            </button>
          </div>
        </label>
        <label>
          <span>{text.authScheme}</span>
          <select
            value={form.auth_scheme}
            onChange={(event) => {
              const auth_scheme = event.target.value as LLMAuthScheme;
              setForm({
                ...form,
                auth_scheme,
                auth_header_name: auth_scheme === "header" ? form.auth_header_name : "",
              });
            }}
          >
            {LLM_AUTH_SCHEMES.map((scheme) => (
              <option key={scheme} value={scheme}>{scheme}</option>
            ))}
          </select>
        </label>
        {form.auth_scheme === "header" ? (
          <label>
            <span>{text.authHeaderName}</span>
            <input
              value={form.auth_header_name}
              placeholder="X-Api-Key"
              onChange={(event) => setForm({ ...form, auth_header_name: event.target.value })}
            />
            <small>{text.authHeaderHint}</small>
          </label>
        ) : (
          <div />
        )}
      </div>

      <div className="settings-advanced-title">{text.advanced}</div>
      <div className="settings-form-grid">
        <label>
          <span>{text.temperature} · {form.temperature.toFixed(1)}</span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={form.temperature}
            onChange={(event) => setForm({ ...form, temperature: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>{text.timeout}</span>
          <input
            type="number"
            min={1}
            max={7200}
            value={form.timeout_seconds}
            onChange={(event) => {
              const n = Number(event.target.value);
              setForm({
                ...form,
                timeout_seconds: Number.isFinite(n) ? Math.min(7200, Math.max(1, Math.round(n))) : form.timeout_seconds,
              });
            }}
          />
        </label>
      </div>

      <section className="settings-planner-card">
        <strong>{text.plannerTitle}</strong>
        <p>{text.plannerSubtitle}</p>
        <div className="settings-preset-row">
          {(["cost", "balanced", "quality"] as const).map((item) => (
            <button
              key={item}
              type="button"
              className={`button ${preset === item ? "primary" : ""}`}
              onClick={() => applyPreset(item)}
            >
              {text[item]}
            </button>
          ))}
        </div>
      </section>

      <footer className="settings-footer">
        <span>{text.llmFooter}</span>
        <div className="settings-footer-actions">
          <button type="button" className="button" disabled={testing || saving} onClick={() => void test()}>
            {testing ? text.testing : text.test}
          </button>
          <button type="submit" className="button primary" disabled={saving || testing}>
            {saving ? "…" : text.save}
          </button>
        </div>
      </footer>
    </form>
  );
}

function SamSettingsPanel({ locale, text, onSamSettingsChange }: { locale: AppLocale; text: SettingsText; onSamSettingsChange?: () => void }) {
  const [settings, setSettings] = useState<SamSettings>();
  const [devices, setDevices] = useState<TrainingDevice[]>([]);
  const [busy, setBusy] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    void Promise.all([api.getSamSettings(), api.listTrainingDevices()])
      .then(([nextSettings, nextDevices]) => {
        setSettings(nextSettings);
        setDevices(nextDevices.items);
      })
      .catch((reason) => setError(reason instanceof Error ? reason.message : text.error))
      .finally(() => setBusy(false));
  }, [text.error]);

  const deviceOptions = useMemo(
    () => [
      { value: "auto", label: text.auto },
      { value: "cpu", label: text.cpu },
      ...devices.filter((device) => device.type === "mps").map(() => ({ value: "mps", label: text.mps })),
      ...(devices.some((device) => device.type === "cuda") ? [{ value: "cuda", label: text.cuda }] : []),
      ...devices
        .filter((device) => device.type === "cuda")
        .map((device) => ({ value: String(device.index ?? 0), label: `${text.cuda}:${device.index ?? 0} · ${device.name}` })),
    ],
    [devices, text],
  );

  const save = async () => {
    if (!settings) return;
    setSaving(true);
    setError("");
    setMessage("");
    try {
      setSettings(
        await api.updateSamSettings({
          enabled: settings.enabled,
          model: settings.model,
          device: settings.device,
          img_size: settings.img_size,
          fallback_mode: settings.fallback_mode,
        }),
      );
      onSamSettingsChange?.();
      setMessage(text.savedSam);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : text.error);
    } finally {
      setSaving(false);
    }
  };

  if (busy) return <section className="panel settings-card"><p className="muted settings-loading">{text.loading}</p></section>;
  if (!settings) return <section className="panel settings-card"><div className="validation invalid">{error || text.error}</div></section>;
  const status = settings.enabled ? (settings.model_configured ? text.ready : text.fallbackReady) : text.disabledStatus;
  return (
    <section className="panel settings-card">
      <header className="settings-card-head">
        <div className="settings-card-icon"><SamIcon /></div>
        <div>
          <h2>{text.samTitle}</h2>
          <p>{text.samSubtitle}</p>
        </div>
        <span className={`settings-status ${settings.enabled ? "ready" : "disabled"}`}>{status}</span>
      </header>
      {error ? <div className="validation invalid"><span>{error}</span></div> : null}
      {message ? <div className="validation valid"><span>{message}</span></div> : null}
      <div className="settings-form-grid">
        <label className="settings-toggle">
          <input
            type="checkbox"
            checked={settings.enabled}
            onChange={(event) => setSettings({ ...settings, enabled: event.target.checked })}
          />
          <span>
            <strong>{text.samEnabled}</strong>
            <small>{settings.model_configured ? settings.model : text.modelHint}</small>
          </span>
        </label>
        <label>
          <span>{text.modelSam}</span>
          <input value={settings.model} placeholder="sam_b.pt" onChange={(event) => setSettings({ ...settings, model: event.target.value })} />
          <small>{text.modelHint}</small>
        </label>
        <label>
          <span>{text.device}</span>
          <select value={settings.device} onChange={(event) => setSettings({ ...settings, device: event.target.value })}>
            {!deviceOptions.some((option) => option.value === settings.device) ? (
              <option value={settings.device}>{settings.device}</option>
            ) : null}
            {deviceOptions.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <label>
          <span>{text.imageSize}</span>
          <input
            type="number"
            min={64}
            max={4096}
            step={64}
            value={settings.img_size}
            onChange={(event) => setSettings({ ...settings, img_size: Number(event.target.value) })}
          />
        </label>
        <label>
          <span>{text.fallback}</span>
          <select
            value={settings.fallback_mode}
            onChange={(event) => setSettings({ ...settings, fallback_mode: event.target.value as SamSettings["fallback_mode"] })}
          >
            <option value="box">{text.box}</option>
            <option value="disabled">{text.disabled}</option>
          </select>
        </label>
      </div>
      {!devices.some((device) => device.type !== "cpu") ? <p className="settings-hint">{text.noDevice}</p> : null}
      <footer className="settings-footer">
        <span>{text.samFooter}</span>
        <button className="button primary" disabled={saving} onClick={() => void save()}>
          {saving ? "…" : text.saveSam}
        </button>
      </footer>
    </section>
  );
}

function LanguageSettingsPanel({
  locale,
  text,
  onLocaleChange,
}: {
  locale: AppLocale;
  text: SettingsText;
  onLocaleChange: (locale: AppLocale) => void;
}) {
  const change = (next: AppLocale) => {
    saveLocale(next);
    onLocaleChange(next);
  };
  return (
    <section className="panel settings-card language-card">
      <header className="settings-card-head">
        <div className="settings-card-icon"><LanguageIcon /></div>
        <div>
          <h2>{text.languageTitle}</h2>
          <p>{text.languageSubtitle}</p>
        </div>
        <span className="settings-status ready">{text.applied}</span>
      </header>
      <div className="language-card-body">
        <strong>{text.displayLanguage}</strong>
        <div className="language-segment" role="group" aria-label={text.displayLanguage}>
          <button type="button" className={locale === "zh" ? "active" : ""} onClick={() => change("zh")}>{text.chinese}</button>
          <button type="button" className={locale === "en" ? "active" : ""} onClick={() => change("en")}>{text.english}</button>
        </div>
      </div>
      <footer className="settings-footer"><span>{text.applied}</span></footer>
    </section>
  );
}

function LlmIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7h16v10H4zM8 11h8M8 14h5" />
    </svg>
  );
}

function SamIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5.5h16v13H4zM8 9l2.5 2.5L16 8m-8 7h8" />
    </svg>
  );
}

function LanguageIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8" />
      <path d="M4 12h16M12 4c2.2 2.1 3.3 4.8 3.3 8S14.2 17.9 12 20c-2.2-2.1-3.3-4.8-3.3-8S9.8 6.1 12 4" />
    </svg>
  );
}
