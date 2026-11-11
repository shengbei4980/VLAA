import { describe, expect, it } from "vitest";
import {
  nodeLabel,
  nodeDescription,
  templateGuide,
  templateTitle,
  workflowError,
  workflowStatus,
} from "./presentation";
import type { WorkflowTemplate } from "../types/workflows";

describe("Chinese workflow presentation", () => {
  it("explains image coordinate errors and the batch failure that follows them", () => {
    expect(
      workflowError(
        "Vision bounding box does not fit the declared sent-image coordinates.",
      ),
    ).toBe(
      "视觉模型返回的框超出当前图像范围。请核对视觉模型设置中的坐标格式，再重试失败项。",
    );
    expect(
      workflowError("vision_box_out_of_bounds: sent_size=[1024,870]"),
    ).toContain("坐标格式");
    expect(
      workflowError(
        "Some selected items failed or were rejected. Review or explicitly exclude them.",
      ),
    ).toContain("查看下方原因");
  });

  it("translates existing built-in names while preserving user names", () => {
    expect(
      templateTitle({
        key: "confirmed_train_evaluate_v1",
        name: "已确认快照训练与评估",
      }),
    ).toBe("用已标注图片训练模型");
    expect(
      templateTitle({
        key: "confirmed_train_evaluate_v1",
        name: "塔体训练流程",
      }),
    ).toBe("塔体训练流程");
    expect(templateTitle({ key: "custom", name: "卫星图复核" })).toBe(
      "卫星图复核",
    );
  });

  it("explains common steps and statuses without showing internal keys", () => {
    expect(nodeLabel("vision_review")).toBe("检查错类与漏标");
    expect(nodeLabel("human_review")).toBe("人工检查并确认");
    expect(workflowStatus("awaiting_review")).toBe("等待你检查标注");
    expect(nodeLabel("future_node")).toBe("扩展处理步骤");
    expect(workflowStatus("future_status")).toBe("待查看");
    expect(nodeLabel("segment_proposals")).toBe("生成或整理目标轮廓");
    expect(nodeDescription("segment_proposals")).not.toContain("分割模型根据提示");
  });

  it("keeps preparation instructions specific to the selected task", () => {
    const guide = (key: string) =>
      templateGuide({
        key,
        name: "模板",
        id: key,
        version: 1,
        nodes: [],
      } as WorkflowTemplate);
    expect(guide("model_review_export_v1").preparation).toContain(
      "已登记的识别模型",
    );
    expect(guide("confirmed_train_evaluate_v1").preparation).toContain(
      "训练图片与验证图片",
    );
    expect(guide("visual_annotation_train_cycle_v1").preparation).toContain(
      "视觉检查服务",
    );
    expect(
      workflowError("All selected images require current human confirmation"),
    ).toContain("逐图检查并确认");
    expect(workflowError("custom error")).toBe("custom error");
  });
});
