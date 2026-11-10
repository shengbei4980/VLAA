import type { WorkflowTemplate } from "../types/workflows";

const nodeText: Record<string, { title: string; description: string }> = {
  vision_propose: {
    title: "生成类别与位置建议",
    description:
      "视觉 LLM 查看图片，按类别说明提出对象和位置。配置为已有模型时使用模型推理。",
  },
  managed_inference: {
    title: "用已有模型预标注",
    description: "使用已登记的模型生成标注草稿。",
  },
  segment_proposals: {
    title: "生成目标轮廓",
    description: "分割模型根据提示生成每个对象的轮廓。",
  },
  validate_proposals: {
    title: "检查标注格式",
    description: "检查类别、坐标和轮廓，标出需要处理的问题。",
  },
  vision_review: {
    title: "检查错类与漏标",
    description: "视觉检查者核对对象类别、遗漏和边界。",
  },
  repair_proposals: {
    title: "按需修正标注",
    description: "在配置的次数与预算内处理可定位的问题。",
  },
  human_review: {
    title: "人工检查并确认",
    description: "你可以改类、补漏和修边，检查整张图片后确认。",
  },
  snapshot: {
    title: "保存本次数据版本",
    description: "记录本次已确认的图片和标注，供后续步骤使用。",
  },
  train: {
    title: "训练识别模型",
    description: "使用已确认的数据训练模型并保存权重。",
  },
  evaluate: {
    title: "评估模型效果",
    description: "在独立的数据上计算指标，帮助你判断模型效果。",
  },
  export: {
    title: "导出标注结果",
    description: "将已确认标注导出为选定格式。",
  },
  next_batch: {
    title: "继续处理下一批",
    description: "满足继续条件时，处理预先选定的下一批图片。",
  },
};

export const nodeLabel = (type: string) =>
  nodeText[type]?.title ?? "扩展处理步骤";
export const nodeDescription = (type: string) =>
  nodeText[type]?.description ?? "使用模板中登记的处理能力。";
export const scopeLabel = (scope: string) =>
  scope === "item"
    ? "逐张图片处理"
    : scope === "batch"
      ? "整批处理"
      : "处理范围";

const guides: Record<
  string,
  {
    title: string;
    legacyNames: string[];
    purpose: string;
    preparation: string;
    result: string;
  }
> = {
  model_review_export_v1: {
    title: "用已有模型标注并检查",
    legacyNames: ["模型推理、人工复核与导出", "给图片自动标注并检查"],
    purpose: "我已有识别模型，想给一批图片做标注。",
    preparation: "准备图片、类别和已登记的识别模型。",
    result: "得到经过人工检查、可以导出的标注。",
  },
  vision_review_export_v1: {
    title: "用视觉 LLM 标注并检查",
    legacyNames: [],
    purpose: "我想让看图大模型按类别说明，直接给图片生成标注草稿。",
    preparation:
      "准备检测或分割图片与类别，在“视觉模型设置”中配置标注建议者和独立检查者。分割任务还需配置 SAM 生成轮廓。",
    result: "得到经过视觉检查、可供人工修改确认并导出的标注。",
  },
  confirmed_train_evaluate_v1: {
    title: "用已标注图片训练模型",
    legacyNames: ["已确认快照训练与评估"],
    purpose: "我已有标注，想训练模型并查看效果。",
    preparation: "准备已完成整图确认的训练图片与验证图片。",
    result: "得到模型权重和评估报告。",
  },
  visual_annotation_train_cycle_v1: {
    title: "标注、训练与持续改进",
    legacyNames: ["视觉标注、训练与迭代"],
    purpose: "我想从预标注开始，连续完成检查、训练和下一批处理。",
    preparation:
      "适用于检测和分割。先配置视觉检查服务，可用已有模型或视觉服务生成建议，分割任务还需分割模型。",
    result: "得到已确认标注、训练模型和评估结果。",
  },
};

export const templateTitle = (
  template: Pick<WorkflowTemplate, "key" | "name">,
) => {
  const guide = guides[template.key];
  return guide &&
    (guide.legacyNames.includes(template.name) || template.name === guide.title)
    ? guide.title
    : template.name;
};

export const templateGuide = (template: WorkflowTemplate) =>
  guides[template.key] ?? {
    title: template.name,
    purpose: "使用已保存的自定义流程处理图片。",
    preparation: "按下方步骤核对所需图片、模型和标注。",
    result: "按该模板的实际步骤生成结果。",
  };

const statusText: Record<string, string> = {
  pending: "等待开始",
  running: "正在处理",
  waiting_task: "等待后续步骤完成",
  awaiting_review: "等待你检查标注",
  needs_attention: "需要你处理",
  completed: "已完成",
  cancelling: "正在取消",
  cancelled: "已取消",
  approved: "已确认",
  background: "已确认无目标",
  excluded: "已排除",
  rejected: "已退回修改",
  failed: "执行失败",
  interrupted: "运行已中断",
};
export const workflowStatus = (status: string) =>
  statusText[status] ?? "待查看";
const splitText: Record<string, string> = {
  train: "训练图片",
  val: "验证图片",
  test: "测试图片",
};
export const splitLabel = (split: string) => splitText[split] ?? "图片";

export function workflowError(message: string): string {
  if (
    /Vision bounding box does not fit the declared sent-image coordinates|vision_box_out_of_bounds/i.test(
      message,
    )
  )
    return "视觉模型返回的框超出当前图像范围。请核对视觉模型设置中的坐标格式，再重试失败项。";
  if (/Some selected items failed or were rejected/i.test(message))
    return "部分图片处理失败或被退回。请查看下方原因，处理后重试失败项。";
  if (/Configure and enable the (proposer|reviewer)/i.test(message))
    return "请先在“视觉模型设置”中启用标注建议服务和视觉检查服务，再回来预览。";
  if (/All selected images require current human confirmation/i.test(message))
    return "这批图片还有标注尚未确认。请在标注页面逐图检查并确认后，再开始训练。";
  if (/Profile classes must belong/i.test(message))
    return "方案中的类别与当前数据集不匹配。请编辑方案，重新读取当前数据集的类别。";
  if (/Automatic visual labeling currently supports/i.test(message))
    return "自动标注流程目前适用于检测和分割数据集。请核对数据集的任务类型。";
  if (
    /Select an active managed PT model|model requires|model_id is unavailable/i.test(
      message,
    )
  )
    return "请在类别与模型方案中选择已登记的预标注模型，再预览本次任务。";
  return message;
}
