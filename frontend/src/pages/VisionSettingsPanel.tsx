import { useEffect, useRef, useState } from "react";
import {
  visionApi,
  type VisionEndpoint,
  type VisionSettings,
} from "../api/vision";
import "../workflow.css";

function emptyEndpoint(): VisionEndpoint {
  return {
    enabled: false,
    mode: "local",
    api_base: "",
    model: "",
    auth_scheme: "bearer",
    auth_header_name: "",
    coordinate_format: "pixel",
    max_image_side: 1024,
    max_output_tokens: 4096,
    timeout_seconds: 60,
    multi_image: false,
    structured_output: "json_object",
    api_key_configured: false,
  };
}

function settingsError(reason: unknown): string {
  const message = reason instanceof Error ? reason.message : "视觉设置请求失败";
  return /404|not found/i.test(message)
    ? "后端视觉接口尚未加载，请重启新版后端。基本配置可继续填写，保存失败时输入会保留。"
    : message;
}

export function VisionSettingsPanel() {
  const [settings, setSettings] = useState<VisionSettings>(() => ({
    proposer: emptyEndpoint(),
    reviewer: emptyEndpoint(),
  }));
  const [keys, setKeys] = useState({ proposer: "", reviewer: "" });
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);
  const edits = useRef<
    Record<"proposer" | "reviewer", Partial<VisionEndpoint>>
  >({ proposer: {}, reviewer: {} });
  const settingsGeneration = useRef(0);
  const mergeEdits = (saved: VisionSettings): VisionSettings => ({
    proposer: { ...saved.proposer, ...edits.current.proposer },
    reviewer: { ...saved.reviewer, ...edits.current.reviewer },
  });
  useEffect(() => {
    let active = true;
    const generation = ++settingsGeneration.current;
    visionApi
      .settings()
      .then((saved) => {
        if (active && generation === settingsGeneration.current)
          setSettings(mergeEdits(saved));
      })
      .catch((reason) => {
        if (active && generation === settingsGeneration.current)
          setError(settingsError(reason));
      })
      .finally(() => {
        if (active && generation === settingsGeneration.current)
          setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);
  const update = (
    role: "proposer" | "reviewer",
    value: Partial<VisionEndpoint>,
  ) => {
    if (!settings) return;
    edits.current[role] = { ...edits.current[role], ...value };
    setSettings((current) => ({
      ...current,
      [role]: { ...current[role], ...value },
    }));
    setDirty(true);
  };
  const save = async () => {
    if (!settings) return;
    setBusy(true);
    setError("");
    settingsGeneration.current++;
    try {
      const latest = mergeEdits(await visionApi.settings());
      const saved = await visionApi.saveSettings({
        proposer: { ...latest.proposer, api_key: keys.proposer || undefined },
        reviewer: { ...latest.reviewer, api_key: keys.reviewer || undefined },
      });
      setSettings(saved);
      setKeys({ proposer: "", reviewer: "" });
      setDirty(false);
      edits.current = { proposer: {}, reviewer: {} };
      setMessage(
        "视觉设置已保存，可以使用下方测试按钮检查图片输入与坐标返回。",
      );
    } catch (reason) {
      setError(settingsError(reason));
    } finally {
      setBusy(false);
      setLoading(false);
    }
  };
  const probe = async (role: "proposer" | "reviewer") => {
    setBusy(true);
    setError("");
    try {
      const result = await visionApi.probe(role);
      setMessage(
        `${result.model} · ${result.message} · ${result.coordinate_verified ? "图像与坐标测试通过" : "坐标尚未验证"}。请再用自己的图片检查实际标注效果。`,
      );
      if (!result.ok) setError(result.message);
    } catch (reason) {
      setError(settingsError(reason));
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="panel workflow-form">
      <h2>视觉模型接入</h2>
      <p>
        用于通用图像工作流中的视觉 LLM
        标注与检查。候选提出者先找出对象并提出标注，
        独立检查者再检查错类、漏标和边界。两个角色均支持远端 API 或本地模型 HTTP
        服务。
      </p>
      <p className="settings-hint">
        首次使用，请为两个角色填写支持图片输入的模型名称、API Base URL 和所需的
        API Key，
        启用角色后保存。随后可测试图片与坐标，再到工作流中选择图片开始标注。已配置的密钥留空可保留。
      </p>
      {error && (
        <p role="alert" className="validation invalid">
          {error}
        </p>
      )}
      {message && <p role="status">{message}</p>}
      {loading && <p>正在读取已保存配置，基本表单可先填写…</p>}
      {settings && (
        <>
          {(["proposer", "reviewer"] as const).map((role) => {
            const endpoint = settings[role];
            return (
              <fieldset
                key={role}
                disabled={busy}
                className="workflow-endpoint"
              >
                <legend>
                  {role === "proposer"
                    ? "候选提出者 · 自动找对象"
                    : "独立视觉检查者 · 检查标注"}
                </legend>
                <p className="settings-hint">
                  {role === "proposer"
                    ? "根据图片、任务说明和类别定义，提出对象类别与位置，供后续分割和人工复核使用。"
                    : "独立查看图片并对照候选标注，给出疑似漏标、错类和边界问题，供修正与人工核对。"}
                </p>
                <label className="workflow-check">
                  <input
                    type="checkbox"
                    checked={endpoint.enabled}
                    onChange={(event) =>
                      update(role, { enabled: event.target.checked })
                    }
                  />
                  启用此角色
                </label>
                <div className="workflow-fields">
                  <label>
                    运行位置
                    <select
                      value={endpoint.mode}
                      onChange={(event) =>
                        update(role, {
                          mode: event.target.value as "remote" | "local",
                        })
                      }
                    >
                      <option value="remote">远端 API</option>
                      <option value="local">本地模型 HTTP 服务</option>
                    </select>
                  </label>
                  <label>
                    模型名称
                    <input
                      value={endpoint.model}
                      onChange={(event) =>
                        update(role, { model: event.target.value })
                      }
                    />
                    <small>
                      填写服务商提供的模型名称，模型需支持图片输入。
                    </small>
                  </label>
                  <label>
                    API Base URL
                    <input
                      type="url"
                      value={endpoint.api_base}
                      onChange={(event) =>
                        update(role, { api_base: event.target.value })
                      }
                      placeholder="http://127.0.0.1:8001/v1"
                    />
                    <small>填写服务的接口根地址，通常以 /v1 结尾。</small>
                  </label>
                  <label>
                    API Key
                    <input
                      type="password"
                      autoComplete="new-password"
                      value={keys[role]}
                      placeholder={
                        endpoint.api_key_configured
                          ? "已配置，留空保留"
                          : "本地服务可留空"
                      }
                      onChange={(event) => {
                        setKeys({ ...keys, [role]: event.target.value });
                        setDirty(true);
                      }}
                    />
                  </label>
                  <label>
                    鉴权方式
                    <select
                      value={endpoint.auth_scheme}
                      onChange={(event) =>
                        update(role, {
                          auth_scheme: event.target
                            .value as VisionEndpoint["auth_scheme"],
                        })
                      }
                    >
                      <option value="bearer">Bearer</option>
                      <option value="header">自定义 Header</option>
                      <option value="raw_authorization">
                        Raw Authorization
                      </option>
                    </select>
                  </label>
                  {endpoint.auth_scheme === "header" && (
                    <label>
                      Header 名
                      <input
                        value={endpoint.auth_header_name}
                        onChange={(event) =>
                          update(role, { auth_header_name: event.target.value })
                        }
                      />
                    </label>
                  )}
                  <label>
                    模型原生坐标
                    <select
                      value={endpoint.coordinate_format}
                      onChange={(event) =>
                        update(role, {
                          coordinate_format: event.target
                            .value as VisionEndpoint["coordinate_format"],
                        })
                      }
                    >
                      <option value="pixel">像素</option>
                      <option value="normalized_1">归一化 0–1</option>
                      <option value="normalized_1000">归一化 0–1000</option>
                    </select>
                    <small>
                      用于解释模型返回的位置框，请与模型服务的坐标格式保持一致。
                    </small>
                  </label>
                  <label>
                    最大图像边长
                    <input
                      type="number"
                      min="64"
                      max="4096"
                      value={endpoint.max_image_side}
                      onChange={(event) =>
                        update(role, {
                          max_image_side: Number(event.target.value),
                        })
                      }
                    />
                    <small>
                      仅缩小超过此边长的发送副本。2301×1955 的图片设为 2301
                      或以上即可按原尺寸发送。两个角色分别设置。
                    </small>
                  </label>
                  <label>
                    最大输出 tokens
                    <input
                      type="number"
                      min="128"
                      max="32768"
                      value={endpoint.max_output_tokens}
                      onChange={(event) =>
                        update(role, {
                          max_output_tokens: Number(event.target.value),
                        })
                      }
                    />
                  </label>
                  <label>
                    超时秒数
                    <input
                      type="number"
                      min="1"
                      max="7200"
                      value={endpoint.timeout_seconds}
                      onChange={(event) =>
                        update(role, {
                          timeout_seconds: Number(event.target.value),
                        })
                      }
                    />
                  </label>
                  <label>
                    结构化输出
                    <select
                      value={endpoint.structured_output}
                      onChange={(event) =>
                        update(role, {
                          structured_output: event.target
                            .value as VisionEndpoint["structured_output"],
                        })
                      }
                    >
                      <option value="json_schema">JSON Schema</option>
                      <option value="json_object">JSON Object</option>
                    </select>
                  </label>
                  <label className="workflow-check">
                    <input
                      type="checkbox"
                      checked={endpoint.multi_image}
                      onChange={(event) =>
                        update(role, { multi_image: event.target.checked })
                      }
                    />
                    服务支持多图输入
                  </label>
                </div>
                <p>
                  测试会向当前服务发送程序生成的小图，检查图片输入和位置框的返回是否可用。
                  请先保存配置，再进行测试。
                </p>
                <button
                  className="button"
                  disabled={busy || dirty || !endpoint.enabled}
                  onClick={() => void probe(role)}
                >
                  测试图像与坐标
                </button>
              </fieldset>
            );
          })}
          <button
            className="button primary"
            disabled={busy || !dirty}
            onClick={() => void save()}
          >
            {busy ? "正在处理…" : "保存视觉设置"}
          </button>
        </>
      )}
    </section>
  );
}
