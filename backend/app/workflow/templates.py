from app.workflow.schemas import NodeDefinition, TemplateCreate


def builtin_templates() -> list[TemplateCreate]:
    return [
        TemplateCreate(key="vision_review_export_v1", name="用视觉 LLM 标注并检查", nodes=[
            NodeDefinition(id="propose", type="vision_propose", params={"source": "vision"}),
            NodeDefinition(id="segment", type="segment_proposals", task_types=["segment"], inputs={"proposals": "propose"}),
            NodeDefinition(id="validate", type="validate_proposals"),
            NodeDefinition(id="review", type="vision_review"),
            NodeDefinition(id="repair", type="repair_proposals"),
            NodeDefinition(id="validate_repair", type="validate_proposals"),
            NodeDefinition(id="recheck", type="vision_review", params={"only_if_repaired": True}),
            NodeDefinition(id="confirm", type="human_review"),
            NodeDefinition(id="snapshot", type="snapshot", scope="batch"),
            NodeDefinition(id="export", type="export", scope="batch"),
        ]),
        TemplateCreate(key="visual_annotation_train_cycle_v1", name="视觉标注、训练与迭代", nodes=[
            NodeDefinition(id="propose", type="vision_propose"),
            NodeDefinition(id="segment", type="segment_proposals", task_types=["segment"], inputs={"proposals": "propose"}),
            NodeDefinition(id="validate", type="validate_proposals"),
            NodeDefinition(id="review", type="vision_review"),
            NodeDefinition(id="repair", type="repair_proposals"),
            NodeDefinition(id="validate_repair", type="validate_proposals"),
            NodeDefinition(id="recheck", type="vision_review", params={"only_if_repaired": True}),
            NodeDefinition(id="confirm", type="human_review"),
            NodeDefinition(id="snapshot", type="snapshot", scope="batch"),
            NodeDefinition(id="train", type="train", scope="batch", inputs={"snapshot": "snapshot"}),
            NodeDefinition(id="evaluate", type="evaluate", scope="batch", inputs={"model": "train", "snapshot": "snapshot"}),
            NodeDefinition(id="next", type="next_batch", scope="batch", inputs={"evaluation": "evaluate"}),
        ]),
        TemplateCreate(key="confirmed_train_evaluate_v1", name="已确认快照训练与评估", nodes=[
            NodeDefinition(id="snapshot", type="snapshot", scope="batch"),
            NodeDefinition(id="train", type="train", scope="batch", inputs={"snapshot": "snapshot"}),
            NodeDefinition(id="evaluate", type="evaluate", scope="batch", inputs={"model": "train", "snapshot": "snapshot"}),
        ]),
        TemplateCreate(key="model_review_export_v1", name="模型推理、人工复核与导出", nodes=[
            NodeDefinition(id="infer", type="managed_inference"),
            NodeDefinition(id="validate", type="validate_proposals"),
            NodeDefinition(id="confirm", type="human_review"),
            NodeDefinition(id="snapshot", type="snapshot", scope="batch"),
            NodeDefinition(id="export", type="export", scope="batch"),
        ]),
    ]
