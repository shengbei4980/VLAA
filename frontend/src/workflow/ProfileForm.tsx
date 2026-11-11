import { useEffect, useRef, useState, type FormEvent } from "react";
import { api } from "../api/client";
import { workflowApi } from "../api/workflows";
import { storageApi, type OutputDestination } from "../api/storage";
import type { ClassLabel, Dataset, ModelVersion, TaskType } from "../types";
import type {
  ProfileClassRule,
  SegmentationSource,
  TaskProfile,
  TaskProfileConfig,
  WorkflowTemplate,
} from "../types/workflows";
import { defaultModelFor, taskTypeLabel } from "../training/helpers";

export const geometryFor = (task: TaskType): TaskProfileConfig["geometry"] =>
  (
    ({
      detect: "bbox",
      segment: "mask",
      obb: "obb",
      classify: "classify",
    }) as const
  )[task];

function defaultConfig(task: TaskType): TaskProfileConfig {
  return {
    task_type: task,
    geometry: geometryFor(task),
    class_rules: [],
    models: {
      proposer: "proposer",
      reviewer: "reviewer",
      segmenter: task === "segment" ? "vision_llm" : "sam",
      training: defaultModelFor(task),
    },
    image_strategy: {
      tile_size: null,
      overlap: 0.2,
      include_overview: true,
      merge_iou: 0.6,
      max_views: 64,
    },
    training: {
      epochs: 30,
      img_size: 640,
      batch_size: 4,
      device: "auto",
      workers: 0,
    },
    quality_conditions: {},
    repair_limit: 0,
    max_requests: 50,
    export_format: "yolo",
  };
}

function objectFromJson(text: string, label: string): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(label + "的格式有误，请在高级配置中检查 JSON。");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(label + "应填写为 JSON 对象。");
  }
  return value as Record<string, unknown>;
}

function formValues(text: string): Record<string, unknown> {
  try {
    return objectFromJson(text, "配置");
  } catch {
    return {};
  }
}

function inputValue(value: unknown): string | number {
  return typeof value === "string" || typeof value === "number" ? value : "";
}

export function ProfileForm({
  source,
  datasets,
  initialDatasetId,
  template,
  onSaved,
  onClose,
}: {
  source?: TaskProfile;
  datasets: Dataset[];
  initialDatasetId?: string;
  template?: WorkflowTemplate;
  onSaved: (profile: TaskProfile) => void;
  onClose: () => void;
}) {
  const initialDataset =
    datasets.find((dataset) => dataset.id === initialDatasetId) ?? datasets[0];
  const [name, setName] = useState(source?.name ?? "新的类别与模型方案");
  const [key, setKey] = useState(
    () => source?.key ?? "profile-" + crypto.randomUUID(),
  );
  const [config, setConfig] = useState<TaskProfileConfig>(() =>
    structuredClone(
      source?.config ?? defaultConfig(initialDataset?.task_type ?? "detect"),
    ),
  );
  const [datasetId, setDatasetId] = useState(initialDataset?.id ?? "");
  const [models, setModels] = useState(JSON.stringify(config.models, null, 2));
  const [strategy, setStrategy] = useState(
    JSON.stringify(config.image_strategy, null, 2),
  );
  const [training, setTraining] = useState(
    JSON.stringify(config.training, null, 2),
  );
  const [quality, setQuality] = useState(
    JSON.stringify(config.quality_conditions, null, 2),
  );
  const [rules, setRules] = useState(
    JSON.stringify(
      config.rules ?? { existing_annotations: "preserve", version: "v1" },
      null,
      2,
    ),
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [destinations, setDestinations] = useState<OutputDestination[]>([]);
  const [registeredModels, setRegisteredModels] = useState<ModelVersion[]>([]);
  const [storageError, setStorageError] = useState("");
  const [modelError, setModelError] = useState("");
  const [newClassName, setNewClassName] = useState("");
  const [classError, setClassError] = useState("");
  const [classNotice, setClassNotice] = useState("");
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    headingRef.current?.focus();
    let active = true;
    Promise.allSettled([
      storageApi.destinations(),
      api.listModels(undefined, false),
    ] as const).then(([storage, models]) => {
      if (!active) return;
      if (storage.status === "fulfilled") setDestinations(storage.value);
      else
        setStorageError(
          "暂时无法读取输出目录，已保存的选择会保留。新目录可在设置的存储位置中登记。",
        );
      if (models.status === "fulfilled")
        setRegisteredModels(models.value.items);
      else setModelError("暂时无法读取已有模型，已保存的选择会保留。");
    });
    return () => {
      active = false;
    };
  }, []);

  const updateJson = (
    text: string,
    setter: (value: string) => void,
    field: string,
    value: unknown,
    label: string,
  ) => {
    try {
      const updated = { ...objectFromJson(text, label) };
      if (value === undefined) delete updated[field];
      else updated[field] = value;
      setter(JSON.stringify(updated, null, 2));
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "请先修正高级配置。");
      setAdvancedOpen(true);
    }
  };
  const updateClass = (index: number, fields: Partial<ProfileClassRule>) => {
    setConfig((previous) => ({
      ...previous,
      class_rules: previous.class_rules.map((rule, position) =>
        position === index ? { ...rule, ...fields } : rule,
      ),
    }));
  };
  const removeClass = (rule: ProfileClassRule) => {
    setConfig((previous) => ({
      ...previous,
      class_rules: previous.class_rules.filter(
        (item) => item.class_id !== rule.class_id,
      ),
    }));
    setClassError("");
    setClassNotice(`已从本方案移除“${rule.name}”，请保存方案。`);
  };
  const deleteDatasetClass = async (rule: ProfileClassRule) => {
    const datasetName =
      datasets.find((item) => item.id === datasetId)?.name ?? "所选数据集";
    if (
      busy ||
      !datasetId ||
      !window.confirm(
        `确认删除数据集“${datasetName}”中的类别“${rule.name}”？\n删除会立即生效，并使已有整图确认转为待审核。已有标注引用的类别无法删除。`,
      )
    )
      return;
    setBusy(true);
    setClassError("");
    setClassNotice("");
    try {
      await api.deleteClass(datasetId, rule.class_id);
      removeClass(rule);
      setClassNotice(`已从数据集删除“${rule.name}”，请保存当前方案。`);
    } catch (reason) {
      const message =
        reason instanceof Error ? reason.message : "类别删除失败，请重试。";
      if (message === "Class label was not found.") {
        removeClass(rule);
        setClassNotice(
          `所选数据集中已没有类别“${rule.name}”，已从本方案移除，请保存方案。`,
        );
      } else {
        setClassError(
          message === "Delete annotations using this class before deleting it."
            ? "该类别已有标注使用。可以点击“从本方案移除”，调整本次识别范围。"
            : message,
        );
      }
    } finally {
      setBusy(false);
    }
  };
  const setDatasetClasses = (classes: ClassLabel[]) =>
    setConfig((previous) => ({
      ...previous,
      class_rules: classes.map((item) => ({
        ...previous.class_rules.find((rule) => rule.class_id === item.id),
        class_id: item.id,
        name: item.name,
        color: item.color,
      })),
    }));
  const loadClasses = async () => {
    setBusy(true);
    setClassError("");
    setClassNotice("");
    try {
      const classes = await api.listClasses(datasetId);
      setDatasetClasses(classes);
    } catch (reason) {
      setClassError(reason instanceof Error ? reason.message : "类别读取失败");
    } finally {
      setBusy(false);
    }
  };
  const addClass = async () => {
    const name = newClassName.trim();
    if (!datasetId || !name || busy) return;
    setBusy(true);
    setClassError("");
    setClassNotice("");
    try {
      const classes = await api.listClasses(datasetId);
      const existing = classes.find((item) => item.name === name);
      const added = existing ?? (await api.createClass(datasetId, name));
      const availableIds = new Set(classes.map((item) => item.id));
      setConfig((previous) => {
        const selected = previous.class_rules.filter((rule) =>
          availableIds.has(rule.class_id),
        );
        return {
          ...previous,
          class_rules: selected.some((rule) => rule.class_id === added.id)
            ? selected
            : [
                ...selected,
                { class_id: added.id, name: added.name, color: added.color },
              ],
        };
      });
      setNewClassName("");
      setClassNotice(
        existing
          ? `已选用数据集中的类别“${added.name}”，可以在下方补充识别说明。`
          : `已将类别“${added.name}”保存到所选数据集，可以在下方补充识别说明。`,
      );
    } catch (reason) {
      setClassError(
        reason instanceof Error ? reason.message : "类别添加失败，请重试。",
      );
    } finally {
      setBusy(false);
    }
  };
  const changeTask = (task: TaskType) => {
    try {
      const updatedModels = { ...objectFromJson(models, "模型配置") };
      if (updatedModels.training === defaultModelFor(config.task_type))
        updatedModels.training = defaultModelFor(task);
      if (!source && task === "segment" && config.task_type !== "segment")
        updatedModels.segmenter = "vision_llm";
      setModels(JSON.stringify(updatedModels, null, 2));
      setConfig({
        ...config,
        task_type: task,
        geometry: geometryFor(task),
        export_format: ["obb", "classify"].includes(task)
          ? "yolo"
          : config.export_format,
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "请先修正模型配置。");
      setAdvancedOpen(true);
    }
  };
  const save = async (event: FormEvent) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const modelConfig = objectFromJson(models, "模型配置");
      const imageStrategy = objectFromJson(strategy, "图片处理配置");
      const trainingConfig = objectFromJson(training, "训练参数");
      const conditions = objectFromJson(quality, "质量要求");
      const taskRules = objectFromJson(rules, "标注规则");
      if (
        !config.class_rules.length ||
        config.class_rules.some(
          (rule) => !rule.class_id.trim() || !rule.name.trim(),
        )
      ) {
        throw new Error(
          "请先手动添加类别或读取数据集类别，并保留有效的类别名称。",
        );
      }
      if (
        Object.values(conditions).some(
          (value) => typeof value !== "number" || !Number.isFinite(value),
        )
      ) {
        throw new Error("质量指标的最低值应填写为有限数字。");
      }
      const saved = await workflowApi.saveProfile({
        key: key.trim(),
        name: name.trim(),
        config: {
          ...config,
          models: modelConfig as TaskProfileConfig["models"],
          rules: taskRules,
          image_strategy: imageStrategy,
          training: trainingConfig,
          quality_conditions: conditions as Record<string, number>,
        },
      });
      onSaved(saved);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "方案保存失败");
      setAdvancedOpen(true);
    } finally {
      setBusy(false);
    }
  };

  const modelValues = formValues(models);
  const segmentationSource: SegmentationSource = modelValues.segmenter === "vision_llm" ? "vision_llm" : "sam";
  const trainingValues = formValues(training);
  const strategyValues = formValues(strategy);
  const ruleValues = formValues(rules);
  const inferenceId =
    typeof modelValues.inference === "string" ? modelValues.inference : "";
  const availableModels = registeredModels.filter(
    (model) =>
      model.task_type === config.task_type &&
      model.format === "pt" &&
      model.status === "active",
  );
  const tiled =
    typeof strategyValues.tile_size === "number" &&
    strategyValues.tile_size > 0;
  const independentCopy = Boolean(source && key !== source.key);
  const usesVisionLlm = template?.nodes.some(
    (node) =>
      node.type === "vision_propose" && node.params?.source === "vision",
  );
  const showsInferenceModel =
    !template ||
    template.nodes.some(
      (node) =>
        node.type === "managed_inference" ||
        (node.type === "vision_propose" && node.params?.source !== "vision"),
    );
  const showsTraining =
    !template || template.nodes.some((node) => node.type === "train");
  const usesVisionService =
    !template ||
    template.nodes.some((node) =>
      ["vision_propose", "vision_review", "repair_proposals"].includes(
        node.type,
      ),
    );

  return (
    <form
      className="panel workflow-form workflow-profile-editor"
      onSubmit={(event) => void save(event)}
    >
      <header className="workflow-heading">
        <div>
          <h2 ref={headingRef} tabIndex={-1}>
            {source ? "编辑类别与模型方案" : "新建类别与模型方案"}
          </h2>
          <p>
            把要识别的类别、使用的模型和处理参数保存为方案，之后可以重复使用。
          </p>
          <p>
            {source && !independentCopy
              ? "正在编辑第 " +
                source.version +
                " 版，保存后会生成新版本，已有版本保留。"
              : "本次保存为新方案的第 1 版。"}
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
      <section className="workflow-form-section">
        <h3>基本信息</h3>
        <div className="workflow-fields">
          <label>
            方案名称
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              required
            />
          </label>
          <label>
            想识别什么
            <select
              value={config.task_type}
              onChange={(event) => changeTask(event.target.value as TaskType)}
            >
              <option value="detect">找出目标位置（检测框）</option>
              <option value="segment">勾出目标轮廓（分割）</option>
              <option value="obb">标出倾斜目标（旋转框）</option>
              <option value="classify">判断整张图片的类别（分类）</option>
            </select>
          </label>
          <label>
            标注导出格式
            <select
              value={config.export_format}
              onChange={(event) =>
                setConfig({
                  ...config,
                  export_format: event.target
                    .value as TaskProfileConfig["export_format"],
                })
              }
            >
              <option value="yolo">YOLO · 用于 YOLO 训练</option>
              {["detect", "segment"].includes(config.task_type) && (
                <>
                  <option value="coco">COCO · 通用检测与分割数据</option>
                  <option value="labelme">
                    LabelMe · 每张图片一个标注文件
                  </option>
                </>
              )}
            </select>
          </label>
        </div>
        <p className="hint">
          导出格式决定标注文件怎样保存，视觉 LLM
          与已有模型生成的草稿都可以在人工确认后导出。
          {config.export_format === "labelme" &&
            "LabelMe 导出包含图片与 JSON。分割掩膜需使用支持原生 mask 的 LabelMe 5.8 及以上版本，区域与忽略规则作为工作台扩展保存。"}
          {config.export_format === "coco" &&
            "COCO 分割导出使用 RLE 保存掩膜，可保留孔洞与多个独立区域。"}
        </p>
        {source && (
          <button
            type="button"
            className="button"
            disabled={busy || independentCopy}
            onClick={() => {
              setKey("profile-" + crypto.randomUUID());
              setName(name + "（副本）");
            }}
          >
            复制为独立方案
          </button>
        )}
      </section>
      {usesVisionService && (
        <section className="workflow-form-section">
          <h3>LLM 全局任务说明</h3>
          <p>
            对本方案的全部图片和类别生效，标注建议、视觉检查和按需修正都会读取。每个类别的识别细节可在下方补充。
            实际识别范围以下方类别清单为准，说明中提到的其他类别不会自动加入本次任务。
          </p>
          <label>
            任务内容（选填）
            <textarea
              rows={6}
              maxLength={20000}
              value={
                typeof ruleValues.task_instructions === "string"
                  ? ruleValues.task_instructions
                  : ""
              }
              placeholder={
                "说明图片类型和本次标注目标\n规定需要标注的区域、对象与排除内容\n补充遮挡、模糊、小目标等情况的处理要求\n例如只标注建筑本体与构件，排除文字、水印和装饰边框，对不确定的区域提出复核问题。"
              }
              onChange={(event) =>
                updateJson(
                  rules,
                  setRules,
                  "task_instructions",
                  event.target.value || undefined,
                  "标注规则",
                )
              }
            />
          </label>
          <p className="hint">
            最多 20000 个字符，随方案保存并用于后续新建运行。
          </p>
        </section>
      )}
      <section className="workflow-form-section">
        <h3>要识别的类别</h3>
        <p>
          先选择数据集，可以手动添加想识别的类别，也可以读取已有类别。添加类别后即可填写识别说明，尚未标注的图片也可以使用。
        </p>
        <div className="workflow-actions">
          <label>
            类别来源数据集
            <select
              value={datasetId}
              disabled={busy}
              onChange={(event) => {
                setDatasetId(event.target.value);
                setClassError("");
                setClassNotice("");
              }}
            >
              {!datasets.length && <option value="">暂无数据集</option>}
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} · {taskTypeLabel(dataset.task_type)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="button"
            disabled={!datasetId || busy}
            onClick={() => void loadClasses()}
          >
            读取数据集类别
          </button>
        </div>
        <div className="workflow-actions">
          <label>
            新类别名称
            <input
              value={newClassName}
              placeholder="例如 塔体、窗户、道路"
              maxLength={255}
              disabled={!datasetId || busy}
              onChange={(event) => setNewClassName(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  void addClass();
                }
              }}
            />
          </label>
          <button
            type="button"
            className="button"
            disabled={!datasetId || !newClassName.trim() || busy}
            onClick={() => void addClass()}
          >
            添加类别
          </button>
        </div>
        <p className="hint">
          点击“添加类别”后，类别名称会立即保存到所选数据集。新增类别会使已有的整图确认转为待审核。
        </p>
        <p className="hint">
          “从本方案移除”在保存方案后生效。“删除数据集类别”会在确认后立即删除未被标注使用的类别。
        </p>
        {classError && (
          <p role="alert" className="validation invalid">
            {classError}
          </p>
        )}
        {classNotice && (
          <p role="status" className="validation valid">
            {classNotice}
          </p>
        )}
        {!config.class_rules.length && (
          <p className="hint">
            还没有类别。输入名称并点击“添加类别”，或读取数据集已有类别。
          </p>
        )}
        {config.class_rules.map((rule, index) => (
          <div className="workflow-rule" key={rule.class_id + "-" + index}>
            <div className="workflow-rule-heading">
              <strong>{rule.name}</strong>
              <div className="workflow-actions">
                <button
                  type="button"
                  className="button"
                  disabled={busy}
                  aria-label={`从本方案移除 ${rule.name}`}
                  onClick={() => removeClass(rule)}
                >
                  从本方案移除
                </button>
                <button
                  type="button"
                  className="button danger"
                  disabled={busy || !datasetId}
                  aria-label={`删除数据集类别 ${rule.name}`}
                  onClick={() => void deleteDatasetClass(rule)}
                >
                  删除数据集类别
                </button>
              </div>
            </div>
            <label>
              怎样识别这个类别（选填）
              <textarea
                value={rule.definition ?? ""}
                placeholder="描述外观特征，以及容易混淆的情况。"
                onChange={(event) =>
                  updateClass(index, { definition: event.target.value })
                }
              />
            </label>
            <details className="workflow-advanced">
              <summary>补充例子和别名（选填）</summary>
              {(["positive_examples", "negative_examples"] as const).map(
                (field) => (
                  <label key={field}>
                    {field === "positive_examples"
                      ? "属于这个类别的例子（每行一个）"
                      : "容易混淆但不属于的例子（每行一个）"}
                    <textarea
                      value={rule[field]?.join("\n") ?? ""}
                      onChange={(event) =>
                        updateClass(index, {
                          [field]: event.target.value
                            .split("\n")
                            .filter(Boolean),
                        })
                      }
                    />
                  </label>
                ),
              )}
              <label>
                同义词或别名（用逗号分开）
                <input
                  value={rule.aliases?.join(", ") ?? ""}
                  onChange={(event) =>
                    updateClass(index, {
                      aliases: event.target.value
                        .split(/[,，]/)
                        .map((alias) => alias.trim())
                        .filter(Boolean),
                    })
                  }
                />
              </label>
            </details>
          </div>
        ))}
      </section>
      <section className="workflow-form-section">
        <h3>使用的模型</h3>
        {usesVisionLlm && (
          <p className="workflow-empty-note">
            本流程由视觉 LLM
            直接看图，按上方类别及说明生成标注草稿，再由独立视觉检查者检查错类与漏标。请在“视觉模型设置”中配置这两个服务，可选择远端
            API 或本地 HTTP 服务。
          </p>
        )}
        {showsInferenceModel && (
          <>
            <label>
              已有的预标注模型（按流程需要选择）
              <select
                value={inferenceId}
                onChange={(event) =>
                  updateJson(
                    models,
                    setModels,
                    "inference",
                    event.target.value || undefined,
                    "模型配置",
                  )
                }
              >
                <option value="">暂不选择</option>
                {inferenceId &&
                  !availableModels.some(
                    (model) => model.id === inferenceId,
                  ) && (
                    <option value={inferenceId}>
                      已保存的模型（等待核验）
                    </option>
                  )}
                {availableModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} · {model.version}
                  </option>
                ))}
              </select>
            </label>
            {modelError && <p className="hint">{modelError}</p>}
            <p className="hint">
              “用已有模型标注并检查”需要已登记的识别模型。“标注、训练与持续改进”也可以留空，使用视觉
              LLM 生成草稿。
            </p>
          </>
        )}
        {showsTraining && (
          <>
            <label>
              训练时使用的基础模型
              <input
                value={inputValue(modelValues.training)}
                placeholder={defaultModelFor(config.task_type)}
                onChange={(event) =>
                  updateJson(
                    models,
                    setModels,
                    "training",
                    event.target.value || undefined,
                    "模型配置",
                  )
                }
              />
            </label>
            <p className="hint">
              仅在流程包含训练时使用。可填写模型名称，或训练设置中已登记的权重路径。
            </p>
          </>
        )}
        {usesVisionService && !usesVisionLlm && (
          <p>
            标注建议服务和视觉复核服务使用“设置 →
            视觉模型设置”中已启用的服务。启动前会显示本次实际使用的端点和模型。
          </p>
        )}
        {config.task_type === "segment" &&
          (!template ||
            template.nodes.some(
              (node) => node.type === "segment_proposals",
            )) && (
            <>
              <label>
                分割轮廓来源
                <select
                  value={segmentationSource}
                  onChange={(event) => updateJson(models, setModels, "segmenter", event.target.value, "模型配置")}
                >
                  <option value="vision_llm">视觉 LLM 直接画轮廓</option>
                  <option value="sam">视觉 LLM 定位 + SAM 分割</option>
                </select>
              </label>
              <p className="hint">
                {segmentationSource === "vision_llm"
                  ? "视觉 LLM 直接返回多边形轮廓，不调用 SAM。所选视觉服务需支持输出多边形坐标，生成后可在复核页补画、调整顶点和修边。"
                  : "视觉 LLM 提出目标框，再由“设置 → SAM 辅助”中配置的模型生成轮廓。生成后仍需检查边界并补齐遗漏。"}
              </p>
              <p className="hint">此选择用于视觉 LLM 生成的分割候选。已有分割模型输出的轮廓会保留，直接画轮廓也需要人工复核。</p>
            </>
          )}
      </section>
      {showsTraining && (
        <section className="workflow-form-section">
          <h3>训练参数</h3>
          <p>流程包含训练时生效。已有方案的其他参数会保留在高级配置中。</p>
          <div className="workflow-fields">
            {(
              [
                {
                  field: "epochs",
                  label: "训练轮数",
                  placeholder: "50",
                  min: 1,
                },
                {
                  field: "img_size",
                  label: "训练图片尺寸",
                  placeholder: "640",
                  min: 32,
                },
                {
                  field: "batch_size",
                  label: "每批训练图片数",
                  placeholder: "16",
                  min: 1,
                },
              ] as const
            ).map(({ field, label, placeholder, min }) => (
              <label key={field}>
                {label}
                <input
                  type="number"
                  min={min}
                  value={inputValue(trainingValues[field])}
                  placeholder={placeholder}
                  onChange={(event) =>
                    updateJson(
                      training,
                      setTraining,
                      field,
                      event.target.value === ""
                        ? undefined
                        : Number(event.target.value),
                      "训练参数",
                    )
                  }
                />
              </label>
            ))}
            <label>
              训练设备
              <input
                value={inputValue(trainingValues.device)}
                placeholder="auto"
                onChange={(event) =>
                  updateJson(
                    training,
                    setTraining,
                    "device",
                    event.target.value || undefined,
                    "训练参数",
                  )
                }
              />
              <small>auto 为自动选择，也可填写 cpu 或设备编号。</small>
            </label>
          </div>
        </section>
      )}
      <section className="workflow-form-section">
        <h3>图片处理与用量</h3>
        <p>视觉服务处理大图时，可以查看整图，或把图片切成小块处理。</p>
        <div className="workflow-fields">
          <label>
            视觉服务查看图片的方式
            <select
              value={tiled ? "tiles" : "whole"}
              onChange={(event) =>
                updateJson(
                  strategy,
                  setStrategy,
                  "tile_size",
                  event.target.value === "tiles" ? 1024 : null,
                  "图片处理配置",
                )
              }
            >
              <option value="whole">查看整张图片</option>
              <option value="tiles">切成小块查看</option>
            </select>
          </label>
          {tiled && (
            <>
              <label>
                每块图片的边长（像素）
                <input
                  type="number"
                  min="32"
                  value={inputValue(strategyValues.tile_size)}
                  onChange={(event) =>
                    updateJson(
                      strategy,
                      setStrategy,
                      "tile_size",
                      Number(event.target.value),
                      "图片处理配置",
                    )
                  }
                />
              </label>
              <label>
                相邻图片块的重叠比例
                <input
                  type="number"
                  min="0"
                  max="0.9"
                  step="0.05"
                  placeholder="0.2"
                  value={inputValue(strategyValues.overlap)}
                  onChange={(event) =>
                    updateJson(
                      strategy,
                      setStrategy,
                      "overlap",
                      event.target.value === ""
                        ? undefined
                        : Number(event.target.value),
                      "图片处理配置",
                    )
                  }
                />
              </label>
            </>
          )}
          <label>
            每次运行最多调用视觉服务的次数
            <input
              type="number"
              min="1"
              value={config.max_requests}
              onChange={(event) =>
                setConfig({
                  ...config,
                  max_requests: Number(event.target.value),
                })
              }
              required
            />
          </label>
        </div>
      </section>
      <fieldset className="workflow-endpoint workflow-form-section">
        <legend>结果保存位置</legend>
        <p>
          选择已登记的本地目录。新目录可在“设置 →
          存储位置”中登记，启动前会显示实际保存路径。
        </p>
        {storageError && <p className="hint">{storageError}</p>}
        <label>
          默认保存目录
          <select
            value={config.outputs?.destination_id ?? ""}
            onChange={(event) =>
              setConfig({
                ...config,
                outputs: {
                  ...config.outputs,
                  destination_id: event.target.value || undefined,
                },
              })
            }
          >
            <option value="">工作台默认目录</option>
            {config.outputs?.destination_id &&
              !destinations.some(
                (destination) =>
                  destination.id === config.outputs?.destination_id,
              ) && (
                <option value={config.outputs.destination_id}>
                  已保存的目录（等待核验）
                </option>
              )}
            {destinations.map((destination) => (
              <option key={destination.id} value={destination.id}>
                {destination.name} · {destination.path}
              </option>
            ))}
          </select>
        </label>
        <label className="workflow-check">
          <input
            type="checkbox"
            checked={config.outputs?.include_images ?? true}
            onChange={(event) =>
              setConfig({
                ...config,
                outputs: {
                  ...config.outputs,
                  include_images: event.target.checked,
                },
              })
            }
          />
          标注结果中同时保存原图
        </label>
        <details className="workflow-advanced">
          <summary>分别设置标注、模型和评估目录</summary>
          <div className="workflow-fields">
            {(["annotations", "models", "evaluations"] as const).map((kind) => {
              const field = (kind + "_destination_id") as
                | "annotations_destination_id"
                | "models_destination_id"
                | "evaluations_destination_id";
              const value = config.outputs?.[field] ?? "";
              const label = {
                annotations: "标注结果",
                models: "模型副本",
                evaluations: "评估结果",
              }[kind];
              return (
                <div key={kind}>
                  <label className="workflow-check">
                    <input
                      type="checkbox"
                      checked={config.outputs?.[kind] ?? true}
                      onChange={(event) =>
                        setConfig({
                          ...config,
                          outputs: {
                            ...config.outputs,
                            [kind]: event.target.checked,
                          },
                        })
                      }
                    />
                    保存{label}
                  </label>
                  <label>
                    {label}目录
                    <select
                      value={value}
                      disabled={config.outputs?.[kind] === false}
                      onChange={(event) =>
                        setConfig({
                          ...config,
                          outputs: {
                            ...config.outputs,
                            [field]: event.target.value || undefined,
                          },
                        })
                      }
                    >
                      <option value="">沿用默认保存目录</option>
                      {value &&
                        !destinations.some(
                          (destination) => destination.id === value,
                        ) && (
                          <option value={value}>
                            已保存的目录（等待核验）
                          </option>
                        )}
                      {destinations.map((destination) => (
                        <option key={destination.id} value={destination.id}>
                          {destination.name} · {destination.path}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              );
            })}
          </div>
        </details>
      </fieldset>
      <details
        className="workflow-advanced"
        open={advancedOpen}
        onToggle={(event) => setAdvancedOpen(event.currentTarget.open)}
      >
        <summary>高级配置</summary>
        <p>
          内部标识由系统生成。完整规则、模型引用和处理参数保存在下方，已有方案的字段会完整保留。
        </p>
        <div className="workflow-fields">
          <label>
            内部标识（高级）
            <input
              value={key}
              onChange={(event) => setKey(event.target.value)}
              required
              pattern="[a-zA-Z0-9_-]+"
            />
          </label>
          <label>
            发现问题后最多自动修正几次
            <input
              type="number"
              min="0"
              max="3"
              value={config.repair_limit}
              onChange={(event) =>
                setConfig({
                  ...config,
                  repair_limit: Number(event.target.value),
                })
              }
            />
            <small>仅在流程包含“按需修正标注”时生效。</small>
          </label>
        </div>
        <p>类别内部标识</p>
        <ul>
          {config.class_rules.map((rule) => (
            <li key={rule.class_id}>
              {rule.name} · <code>{rule.class_id}</code>
            </li>
          ))}
        </ul>
        <div className="workflow-fields">
          <label>
            模型引用（JSON）
            <textarea
              aria-label="模型引用 JSON"
              rows={8}
              value={models}
              onChange={(event) => setModels(event.target.value)}
              spellCheck={false}
            />
            <small>
              proposer 和 reviewer
              是保留的内部引用。实际服务由视觉模型设置决定。
            </small>
          </label>
          <label>
            标注与区域规则（JSON）
            <textarea
              aria-label="标注规则 JSON"
              rows={8}
              value={rules}
              onChange={(event) => setRules(event.target.value)}
              spellCheck={false}
            />
          </label>
          <label>
            完整图片处理参数（JSON）
            <textarea
              aria-label="图片处理 JSON"
              rows={8}
              value={strategy}
              onChange={(event) => setStrategy(event.target.value)}
              spellCheck={false}
            />
          </label>
          <label>
            完整训练参数（JSON）
            <textarea
              aria-label="训练参数 JSON"
              rows={8}
              value={training}
              onChange={(event) => setTraining(event.target.value)}
              spellCheck={false}
            />
          </label>
          <label>
            质量指标最低值（JSON）
            <textarea
              aria-label="质量要求 JSON"
              rows={8}
              value={quality}
              onChange={(event) => setQuality(event.target.value)}
              placeholder={'{"map50": 0.5}'}
              spellCheck={false}
            />
            <small>
              留空对象表示不设置最低值。继续下一批前还会核对批次条件。
            </small>
          </label>
        </div>
      </details>
      <button
        className="button primary"
        disabled={busy || !name.trim() || !key.trim()}
        type="submit"
      >
        {busy ? "正在保存…" : "保存类别与模型方案"}
      </button>
    </form>
  );
}
