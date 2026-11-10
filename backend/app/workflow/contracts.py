from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.core.errors import ValidationError
from app.workflow.schemas import NodeDefinition


@dataclass(frozen=True)
class NodeContract:
    type: str
    scope: str
    output_type: str
    input_types: tuple[str, ...] = ()
    version: int = 1
    resource: str = "cpu"
    side_effect: str = "artifact"


CONTRACTS = {
    item.type: item for item in (
        NodeContract("vision_propose", "item", "proposals", resource="vision"),
        NodeContract("managed_inference", "item", "proposals", resource="gpu"),
        NodeContract("segment_proposals", "item", "proposals", ("proposals",), resource="gpu"),
        NodeContract("validate_proposals", "item", "proposals", ("proposals",)),
        NodeContract("vision_review", "item", "proposals", ("proposals",), resource="vision"),
        NodeContract("repair_proposals", "item", "proposals", ("proposals",), resource="vision"),
        NodeContract("human_review", "item", "confirmed_annotations", ("proposals",), side_effect="human"),
        NodeContract("snapshot", "batch", "dataset_snapshot", side_effect="snapshot"),
        NodeContract("train", "batch", "model", ("dataset_snapshot",), resource="gpu", side_effect="task"),
        NodeContract("evaluate", "batch", "evaluation", ("model", "dataset_snapshot"), resource="gpu", side_effect="task"),
        NodeContract("export", "batch", "export", side_effect="export"),
        NodeContract("next_batch", "batch", "workflow_run", ("evaluation",), side_effect="workflow"),
    )
}


def validate_nodes(nodes: list[NodeDefinition]) -> None:
    ids = [node.id for node in nodes]
    if len(ids) != len(set(ids)):
        raise ValidationError("workflow_duplicate_node", "Template node IDs must be unique.")
    batch_started = False
    by_id = {node.id: node for node in nodes}
    for index, node in enumerate(nodes):
        contract = CONTRACTS.get(node.type)
        if contract is None or node.version != contract.version or node.scope != contract.scope:
            raise ValidationError("workflow_node_unavailable", f"Unsupported node contract: {node.type} v{node.version}.")
        if node.scope == "batch":
            batch_started = True
        elif batch_started:
            raise ValidationError("workflow_invalid_scope_order", "Item nodes must precede the batch barrier.")
        for upstream in node.inputs.values():
            if upstream not in by_id or ids.index(upstream) >= index:
                raise ValidationError("workflow_invalid_port", "Inputs must bind an earlier node.")
            source_type = CONTRACTS[by_id[upstream].type].output_type
            if contract.input_types and source_type not in contract.input_types:
                raise ValidationError("workflow_incompatible_port", f"{source_type} cannot feed {node.type}.")
        for outcome, target in node.routes.items():
            if outcome not in {"completed", "valid", "invalid", "approved", "rejected", "passed", "failed"}:
                raise ValidationError("workflow_invalid_route", "Routing requires a declared result enum.")
            if target not in by_id or ids.index(target) <= index or by_id[target].scope != node.scope:
                raise ValidationError("workflow_invalid_route", "Routes must point forward within the same scope.")


Handler = Callable[[dict], dict]
