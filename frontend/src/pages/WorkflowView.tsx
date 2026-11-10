import { useEffect, useState } from "react";
import { api, apiUrl } from "../api/client";
import { workflowApi, type PreparedWorkflowRun } from "../api/workflows";
import { ProfileForm } from "../workflow/ProfileForm";
import { TemplateForm } from "../workflow/TemplateForm";
import { ProjectFolderPicker } from "../workflow/ProjectFolderPicker";
import { WorkflowRunHistory } from "../workflow/WorkflowRunHistory";
import { WorkflowRunView } from "./WorkflowRunView";
import type { Dataset, ImageItem } from "../types";
import type {
  TaskProfile,
  WorkflowTemplate,
  WorkflowProject,
} from "../types/workflows";
import { taskTypeLabel } from "../training/helpers";
import {
  nodeDescription,
  nodeLabel,
  splitLabel,
  templateGuide,
  templateTitle,
  workflowError,
} from "../workflow/presentation";
import "../workflow.css";

export function WorkflowView({
  datasets,
  onSettings,
  onOpenTraining,
  onOpenModel,
  onDirtyChange,
  onStorageSettings,
  initialDatasetId,
  onDataChanged,
}: {
  datasets: Dataset[];
  onSettings: () => void;
  onOpenTraining: (datasetId: string | undefined, taskId: string) => void;
  onOpenModel: (datasetId: string | undefined, modelId: string) => void;
  onDirtyChange?: (dirty: boolean) => void;
  onStorageSettings?: () => void;
  initialDatasetId?: string;
  onDataChanged?: () => void;
}) {
  const [profiles, setProfiles] = useState<TaskProfile[]>([]);
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [profileId, setProfileId] = useState("");
  const [project, setProject] = useState<WorkflowProject>();
  const [projectReady, setProjectReady] = useState(false);
  const [templateId, setTemplateId] = useState("");
  const [datasetId, setDatasetId] = useState(initialDatasetId ?? datasets[0]?.id ?? "");
  const [images, setImages] = useState<ImageItem[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [entryRoute] = useState(readWorkflowRoute);
  const [runId, setRunId] = useState(entryRoute.runId);
  const [initialReviewId, setInitialReviewId] = useState(entryRoute.reviewId);
  const [editingProfile, setEditingProfile] = useState<
    TaskProfile | null | undefined
  >();
  const [editingTemplate, setEditingTemplate] = useState<WorkflowTemplate>();
  const [launching, setLaunching] = useState(false);
  const [approved, setApproved] = useState(false);
  const [snapshotId, setSnapshotId] = useState("");
  const [nextBatches, setNextBatches] = useState("");
  const [prepared, setPrepared] = useState<PreparedWorkflowRun>();
  const [busy, setBusy] = useState(true);
  const [imagesLoading, setImagesLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [idempotencyKey, setIdempotencyKey] = useState(() =>
    crypto.randomUUID(),
  );

  const refresh = async () => {
    const [nextProfiles, nextTemplates] = await Promise.all([
      workflowApi.profiles(),
      workflowApi.templates(),
    ]);
    setProfiles(nextProfiles);
    setTemplates(nextTemplates);
    const targetDataset = datasets.find((item) => item.id === initialDatasetId);
    const matchingProfile = targetDataset
      ? nextProfiles.find((item) => item.config.task_type === targetDataset.task_type)
      : nextProfiles[0];
    const visualTemplate = initialDatasetId
      ? nextTemplates.filter((item) => item.nodes.some((node) => node.type === "vision_propose" && node.params?.source === "vision"))
        .sort((left, right) => right.version - left.version)[0]
      : undefined;
    setProfileId((current) => current || matchingProfile?.id || "");
    setTemplateId((current) => current || visualTemplate?.id || nextTemplates[0]?.id || "");
  };
  useEffect(() => {
    void refresh()
      .catch((reason) => setError(String(reason)))
      .finally(() => setBusy(false));
  }, []);
  useEffect(() => {
    if (!runId) window.history.replaceState(null, "", "#workflow");
  }, [runId]);
  useEffect(() => {
    if (!datasetId && datasets[0]) setDatasetId(datasets[0].id);
  }, [datasets, datasetId]);
  useEffect(() => {
    let active = true;
    setImages([]);
    setSelectedIds([]);
    if (!datasetId) return;
    setImagesLoading(true);
    api
      .listImages(datasetId)
      .then((page) => {
        if (!active) return;
        setImages(page.items);
        setSelectedIds(
          page.items
            .filter((image) => image.split !== "test")
            .map((image) => image.id),
        );
      })
      .catch((reason) => {
        if (active) setError(String(reason));
      })
      .finally(() => {
        if (active) setImagesLoading(false);
      });
    return () => {
      active = false;
    };
  }, [datasetId]);
  useEffect(() => {
    setApproved(false);
    setPrepared(undefined);
    setLaunching(false);
    setIdempotencyKey(crypto.randomUUID());
  }, [
    datasetId,
    profileId,
    templateId,
    selectedIds,
    snapshotId,
    nextBatches,
    project,
  ]);

  const profile = profiles.find((item) => item.id === profileId);
  const template = templates.find((item) => item.id === templateId);
  const dataset = datasets.find((item) => item.id === datasetId);
  const choices = templates.filter((candidate) =>
    templates.every(
      (other) =>
        other.key !== candidate.key || other.version <= candidate.version,
    ),
  );
  const guide = template ? templateGuide(template) : undefined;
  const visibleSteps =
    template?.nodes.filter(
      (node) =>
        !node.task_types?.length ||
        node.task_types.includes(
          profile?.config.task_type ?? dataset?.task_type ?? "detect",
        ),
    ) ?? [];
  const requiresReview = template?.nodes.some(
    (node) => node.type === "human_review",
  );
  const usesVisionLlm = template?.nodes.some(
    (node) =>
      node.type === "vision_propose" && node.params?.source === "vision",
  );
  const beginPreview = async () => {
    if (!profile || !template || !dataset) return;
    setBusy(true);
    setError("");
    try {
      const available = new Set(images.map((image) => image.id));
      const allIds = new Set(selectedIds);
      const batches = nextBatches
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const imageIds = line
            .split(",")
            .map((id) => id.trim())
            .filter(Boolean);
          for (const id of imageIds) {
            if (!available.has(id))
              throw new Error(`下一批图片不属于所选数据集 ${id}`);
            if (allIds.has(id)) throw new Error(`图片重复纳入批次 ${id}`);
            allIds.add(id);
          }
          return { dataset_id: datasetId, image_ids: imageIds };
        });
      const preview = await workflowApi.prepareRun({
        idempotency_key: idempotencyKey,
        template_id: templateId,
        profile_id: profileId,
        dataset_id: datasetId,
        project_id: project?.id,
        image_ids: selectedIds,
        snapshot_id: snapshotId.trim() || undefined,
        next_batches: batches,
        inputs: usesVisionLlm
          ? {}
          : { model_id: profile.config.models.inference || undefined },
      });
      setPrepared(preview);
      setLaunching(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "运行预检失败");
    } finally {
      setBusy(false);
    }
  };
  const create = async () => {
    if (!prepared || !approved) return;
    setBusy(true);
    setError("");
    try {
      const run = await workflowApi.createRun(prepared.request);
      setLaunching(false);
      setRunId(run.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "创建运行失败");
    } finally {
      setBusy(false);
    }
  };

  if (runId)
    return (
      <WorkflowRunView
        key={runId}
        onDirtyChange={onDirtyChange}
        onDataChanged={onDataChanged}
        initialReviewId={initialReviewId}
        runId={runId}
        onOpenRun={(id) => { setInitialReviewId(""); setRunId(id); }}
        onBack={() => {
          setRunId("");
          setInitialReviewId("");
          void refresh().catch((reason) => setError(String(reason)));
        }}
        onOpenTraining={onOpenTraining}
        onOpenModel={onOpenModel}
      />
    );
  if (editingProfile !== undefined)
    return (
      <main className="workflow-page">
        <ProfileForm
          key={editingProfile?.id ?? "new"}
          source={editingProfile ?? undefined}
          datasets={datasets}
          initialDatasetId={datasetId}
          template={template}
          onClose={() => setEditingProfile(undefined)}
          onSaved={(saved) => {
            setEditingProfile(undefined);
            setProfileId(saved.id);
            setNotice(`已保存 ${saved.name}，第 ${saved.version} 版`);
            void refresh().catch((reason) => setError(String(reason)));
          }}
        />
      </main>
    );
  if (editingTemplate)
    return (
      <main className="workflow-page">
        <TemplateForm
          key={editingTemplate.id}
          source={editingTemplate}
          onClose={() => setEditingTemplate(undefined)}
          onSaved={(saved) => {
            setEditingTemplate(undefined);
            setTemplateId(saved.id);
            void refresh().catch((reason) => setError(String(reason)));
          }}
        />
      </main>
    );
  return (
    <main className="workflow-page">
      <header className="workflow-heading">
        <div>
          <span className="eyebrow">图像任务助手</span>
          <h1>通用图像工作流</h1>
          <p>先选择要做的工作，再选图片与模型。每个步骤都有用途说明。</p>
          {initialDatasetId && dataset && <p className="hint">已选择数据集“{dataset.name}”。请核对类别与模型方案，再预览本次任务。</p>}
        </div>
        <div className="workflow-actions">
          {onStorageSettings && (
            <button className="button" onClick={onStorageSettings}>
              存储位置
            </button>
          )}
          <button className="button" onClick={onSettings}>
            视觉模型设置
          </button>
        </div>
      </header>
      {error && (
        <div className="validation invalid" role="alert">
          {workflowError(error)}
          {workflowError(error) !== error && (
            <details>
              <summary>查看技术详情</summary>
              <p>{error}</p>
            </details>
          )}
        </div>
      )}
      {notice && (
        <p role="status" className="validation valid">
          {notice}
        </p>
      )}
      <nav className="workflow-setup-guide" aria-label="任务准备步骤">
        {[
          ["workflow-choose-title", "选工作"],
          ["workflow-images-title", "选图片"],
          ["workflow-profile-title", "设类别与模型"],
          ["workflow-project-title", "选项目目录"],
          ["workflow-start-title", "核对并启动"],
        ].map(([target, label], index) => (
          <button
            key={target}
            type="button"
            onClick={() =>
              document
                .getElementById(target)
                ?.scrollIntoView({ block: "start" })
            }
          >
            <span>{index + 1}</span> {label}
          </button>
        ))}
      </nav>
      <section
        className="panel workflow-form workflow-setup-section"
        aria-labelledby="workflow-choose-title"
      >
        <div className="workflow-section-heading">
          <span className="workflow-section-number">1</span>
          <div>
            <h2 id="workflow-choose-title">选择你要完成的工作</h2>
            <p>根据你已经准备好的材料，选择一种流程。</p>
          </div>
        </div>
        <div
          className="workflow-template-grid"
          role="group"
          aria-label="选择工作内容"
        >
          {choices.map((choice) => {
            const choiceGuide = templateGuide(choice);
            const selected = template?.key === choice.key;
            return (
              <button
                key={choice.id}
                type="button"
                className="workflow-template-card"
                aria-pressed={selected}
                aria-label={"选择" + templateTitle(choice)}
                disabled={busy}
                onClick={() => setTemplateId(choice.id)}
              >
                <span className="workflow-card-top">
                  <span>{selected ? "当前选择" : "选择此流程"}</span>
                  <small>
                    第 {selected ? template!.version : choice.version} 版
                  </small>
                </span>
                <strong>{templateTitle(choice)}</strong>
                <span className="workflow-card-purpose">
                  {choiceGuide.purpose}
                </span>
                <span className="workflow-card-result">
                  {choiceGuide.result}
                </span>
              </button>
            );
          })}
        </div>
        {!busy && !choices.length && (
          <p className="hint">暂时没有可用模板，请确认后端服务已启动。</p>
        )}
        {template && guide && (
          <div className="workflow-choice-summary">
            <p>
              <strong>开始前准备</strong> {guide.preparation}
            </p>
            <p>
              <strong>主要步骤</strong>
            </p>
            <ol
              className="workflow-flow-preview"
              aria-label="所选流程的中文步骤"
            >
              {visibleSteps
                .filter(
                  (node) =>
                    !node.params?.only_if_repaired &&
                    !["validate_proposals", "repair_proposals"].includes(
                      node.type,
                    ),
                )
                .map((node) => (
                  <li key={node.id}>{nodeLabel(node.type)}</li>
                ))}
            </ol>
            <details className="workflow-advanced">
              <summary>查看全部步骤与模板版本</summary>
              <ol className="workflow-step-list">
                {visibleSteps.map((node, index) => (
                  <li key={node.id}>
                    <strong>
                      {index + 1}.{" "}
                      {node.params?.only_if_repaired
                        ? "复查修正后的标注"
                        : nodeLabel(node.type)}
                    </strong>
                    <span>{nodeDescription(node.type)}</span>
                  </li>
                ))}
              </ol>
              <label>
                选择模板版本
                <select
                  value={templateId}
                  onChange={(event) => setTemplateId(event.target.value)}
                >
                  {templates.map((item) => (
                    <option key={item.id} value={item.id}>
                      {templateTitle(item)} · 第 {item.version} 版
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="button"
                onClick={() => setEditingTemplate(template)}
              >
                调整或复制流程步骤
              </button>
            </details>
          </div>
        )}
      </section>
      <section
        className="panel workflow-form workflow-setup-section"
        aria-labelledby="workflow-images-title"
      >
        <div className="workflow-section-heading">
          <span className="workflow-section-number">2</span>
          <div>
            <h2 id="workflow-images-title">选择本次处理的图片</h2>
            <p>
              训练图片用于学习，验证图片用于检查效果，测试图片默认留作独立检验。
            </p>
          </div>
        </div>
        <label className="workflow-dataset-select">
          图片所在的数据集
          <select
            value={datasetId}
            onChange={(event) => setDatasetId(event.target.value)}
          >
            {!datasets.length && (
              <option value="">请先在“数据集”页面导入图片</option>
            )}
            {datasets.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} · {taskTypeLabel(item.task_type)}
              </option>
            ))}
          </select>
        </label>
        <div className="workflow-actions">
          <strong>
            已选 {selectedIds.length} / {images.length} 张
          </strong>
          <button
            className="button"
            onClick={() =>
              setSelectedIds(
                images
                  .filter((image) => image.split !== "test")
                  .map((image) => image.id),
              )
            }
          >
            选择训练与验证图片
          </button>
          <button className="button" onClick={() => setSelectedIds([])}>
            清空选择
          </button>
        </div>
        <div className="workflow-image-picker">
          {imagesLoading ? (
            <p>正在读取图片…</p>
          ) : !images.length ? (
            <p className="workflow-empty-note">
              这个数据集还没有图片。导入图片后即可在这里选择。
            </p>
          ) : (
            images.map((image) => (
              <label key={image.id}>
                <input
                  type="checkbox"
                  checked={selectedIds.includes(image.id)}
                  onChange={(event) =>
                    setSelectedIds((items) =>
                      event.target.checked
                        ? [...items, image.id]
                        : items.filter((id) => id !== image.id),
                    )
                  }
                />
                <img
                  className="workflow-image-thumb"
                  src={apiUrl(image.file_url)}
                  alt=""
                  loading="lazy"
                />
                <span>
                  {image.file_name}
                  <small>
                    {splitLabel(image.split)} · {image.width} × {image.height}
                  </small>
                </span>
              </label>
            ))
          )}
        </div>
      </section>
      <section
        className="panel workflow-form workflow-setup-section"
        aria-labelledby="workflow-profile-title"
      >
        <div className="workflow-section-heading">
          <span className="workflow-section-number">3</span>
          <div>
            <h2 id="workflow-profile-title">选择类别与模型方案</h2>
            <p>
              方案保存要识别的类别、使用的模型和参数，同类项目可以重复使用。
            </p>
          </div>
        </div>
        <div className="workflow-fields">
          <label>
            类别与模型方案
            <select
              value={profileId}
              onChange={(event) => setProfileId(event.target.value)}
            >
              {!profiles.length && <option value="">请先新建一个方案</option>}
              {profiles.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name} · 第 {item.version} 版 ·{" "}
                  {taskTypeLabel(item.config.task_type)}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="workflow-actions">
          <button
            className="button primary"
            onClick={() => setEditingProfile(null)}
          >
            新建类别与模型方案
          </button>
          <button
            className="button"
            disabled={!profile}
            onClick={() => setEditingProfile(profile)}
          >
            查看或调整方案
          </button>
        </div>
        {!profiles.length && (
          <p className="workflow-empty-note">
            {usesVisionLlm
              ? "第一次使用时，点“新建类别与模型方案”，自行添加或读取数据集类别，并补充类别说明。视觉服务在“视觉模型设置”中配置。"
              : "第一次使用时，点“新建类别与模型方案”，自行添加或读取数据集类别，并选择模型。"}
          </p>
        )}
        {profile && (
          <p className="hint">
            当前方案用于{taskTypeLabel(profile.config.task_type)}，
            {profile.config.class_rules.length
              ? "包含 " + profile.config.class_rules.length + " 个类别"
              : "运行时读取数据集类别"}
            。
          </p>
        )}
        {profile &&
          dataset &&
          profile.config.task_type !== dataset.task_type && (
            <p role="alert" className="validation invalid">
              所选方案的任务类型与数据集不同。请选择匹配的方案，或新建一个方案。
            </p>
          )}
        <details className="workflow-advanced">
          <summary>更多设置 · 历史数据版本与下一批</summary>
          <p className="hint">首次使用可以保留为空，先完成当前批次。</p>
          <label>
            使用已有的已确认数据版本（可选）
            <input
              value={snapshotId}
              placeholder="已有数据快照编号"
              onChange={(event) => setSnapshotId(event.target.value)}
            />
          </label>
          <label>
            下一批图片编号（可选，每行一批，用英文逗号分隔）
            <textarea
              rows={3}
              value={nextBatches}
              onChange={(event) => setNextBatches(event.target.value)}
            />
          </label>
          <details>
            <summary>查看图片编号</summary>
            <ul>
              {images.map((image) => (
                <li key={image.id}>
                  {image.file_name} · <code>{image.id}</code>
                </li>
              ))}
            </ul>
          </details>
        </details>
      </section>
      <ProjectFolderPicker
        value={project}
        onChange={setProject}
        onReadyChange={setProjectReady}
      />
      <section
        className="panel workflow-form workflow-setup-section workflow-start-section"
        aria-labelledby="workflow-start-title"
      >
        <div className="workflow-section-heading">
          <span className="workflow-section-number">5</span>
          <div>
            <h2 id="workflow-start-title">核对后开始</h2>
            <p>先预览图片范围、模型服务、保存目录和后续步骤，再确认启动。</p>
          </div>
        </div>
        <dl className="workflow-launch-summary">
          <div>
            <dt>本次工作</dt>
            <dd>{template ? templateTitle(template) : "待选择流程"}</dd>
          </div>
          <div>
            <dt>图片范围</dt>
            <dd>
              {dataset?.name ?? "待选择数据集"} · {selectedIds.length} 张
            </dd>
          </div>
          <div>
            <dt>类别与模型</dt>
            <dd>{profile?.name ?? "待选择方案"}</dd>
          </div>
          <div>
            <dt>项目目录</dt>
            <dd>
              {project
                ? `${project.name} · ${project.root_path}`
                : "工作台默认位置"}
            </dd>
          </div>
        </dl>
        <p>
          {requiresReview
            ? "生成的标注需要你逐图检查并确认，流程会在人工检查步骤等待。"
            : "训练将使用已完成整图确认的标注，请先在标注页面完成检查。"}
        </p>
        {(!profile || !selectedIds.length) && (
          <p className="hint">
            {!selectedIds.length
              ? "请先选择要处理的图片。"
              : "请先新建或选择类别与模型方案。"}
          </p>
        )}
        <button
          className="button primary"
          disabled={
            busy ||
            !projectReady ||
            imagesLoading ||
            !profile ||
            !template ||
            !dataset ||
            !selectedIds.length ||
            profile.config.task_type !== dataset.task_type
          }
          onClick={() => void beginPreview()}
        >
          预览本次任务
        </button>
      </section>
      <WorkflowRunHistory profiles={profiles} templates={templates} onOpenRun={(id) => { setInitialReviewId(""); setRunId(id); }} />
      {launching && profile && template && (
        <div className="modal-backdrop">
          <section
            className="dataset-dialog workflow-launch"
            role="dialog"
            aria-modal="true"
            aria-labelledby="workflow-launch-title"
          >
            <h2 id="workflow-launch-title">确认本次运行</h2>
            <p>
              {templateTitle(template)} · 第 {template.version} 版 ·{" "}
              {profile.name} · 第 {profile.version} 版
            </p>
            <p>
              {dataset?.name} · 当前批 {selectedIds.length} 张 · 请求上限{" "}
              {profile.config.max_requests} · 修复上限{" "}
              {profile.config.repair_limit}
            </p>
            <p>
              服务器已核对图片 hash
              和模型能力。所选图片及其缩放、切片和对象裁剪可发送到以下端点。
            </p>
            <ul>
              {prepared &&
                Object.entries(prepared.vision_approvals).map(
                  ([role, endpoint]) => (
                    <li key={role}>
                      {role === "proposer" ? "标注建议服务" : "视觉检查服务"} ·{" "}
                      {endpoint.mode === "local" ? "本地服务" : "远端 API"} ·{" "}
                      {endpoint.model} · {endpoint.api_base} · 请求上限{" "}
                      {endpoint.max_requests}
                    </li>
                  ),
                )}
            </ul>
            <p>
              总纳入{" "}
              {prepared?.scope.reduce(
                (count, batch) => count + batch.image_ids.length,
                0,
              ) ?? 0}{" "}
              张，含预选下一批。
              {prepared && !Object.keys(prepared.vision_approvals).length
                ? "本模板不调用外部视觉角色。"
                : ""}
            </p>
            <p>
              {requiresReview
                ? "流程会在人工检查步骤等待。请逐图确认本批标注，再继续模板中的后续步骤。"
                : "本流程核验已确认的数据后，按所选步骤继续。"}
            </p>
            {template.nodes.some((node) => node.type === "next_batch") && (
              <p>
                如果预选了下一批，将在满足继续条件后用新模型生成下一批标注建议，并再次等待人工检查。
              </p>
            )}
            {prepared?.outputs && (
              <>
                <h3>本次产物目录</h3>
                {prepared.outputs.project && (
                  <p>
                    项目 {prepared.outputs.project.name} · 根目录{" "}
                    <code>{prepared.outputs.project.root_path}</code>
                  </p>
                )}
                {prepared.outputs.json_path && (
                  <p>
                    LabelMe JSON <code>{prepared.outputs.json_path}</code>
                  </p>
                )}
                {prepared.outputs.images_path && (
                  <p>
                    标注预览 PNG <code>{prepared.outputs.images_path}</code>
                  </p>
                )}
                <ul>
                  {(["annotations", "models", "evaluations"] as const).map(
                    (kind) => {
                      const output = prepared.outputs![kind];
                      return (
                        <li key={kind}>
                          {
                            {
                              annotations: "标注",
                              models: "模型副本",
                              evaluations: "评估结果",
                            }[kind]
                          }{" "}
                          ·{" "}
                          {output.enabled
                            ? output.path
                              ? `${output.path}/workflow-<run-id>/${kind}`
                              : "工作台受管目录"
                            : "本次不额外输出"}
                        </li>
                      );
                    },
                  )}
                </ul>
                <p>
                  {prepared.outputs.include_images
                    ? "标注输出包含原图。"
                    : "标注输出仅保存标签与清单。"}
                </p>
              </>
            )}
            <label className="workflow-check">
              <input
                type="checkbox"
                checked={approved}
                onChange={(event) => setApproved(event.target.checked)}
              />
              我已核对图片、目的端点、预算和后续步骤
            </label>
            {error && (
              <p role="alert" className="validation invalid">
                {error}
              </p>
            )}
            <footer>
              <button
                className="button"
                disabled={busy}
                onClick={() => setLaunching(false)}
              >
                返回修改
              </button>
              <button
                className="button primary"
                disabled={!approved || busy}
                onClick={() => void create()}
              >
                {busy ? "正在提交…" : "启动运行"}
              </button>
            </footer>
          </section>
        </div>
      )}
    </main>
  );
}

function readWorkflowRoute(): { runId: string; reviewId: string } {
  const match = window.location.hash.match(/^#workflow\/([^/]+)(?:\/review\/([^/]+))?$/);
  try {
    return {
      runId: match ? decodeURIComponent(match[1]) : "",
      reviewId: match?.[2] ? decodeURIComponent(match[2]) : "",
    };
  } catch {
    return { runId: "", reviewId: "" };
  }
}
