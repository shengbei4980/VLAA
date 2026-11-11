import type { BBox, OBB, TaskType } from "./index";

export interface MaskRle {
  size: [number, number];
  counts: number[] | string;
}

export interface WorkflowProject {
  id: string;
  name: string;
  root_path: string;
  destination_id: string;
  created_at: string;
}

export interface WorkflowOutputDestinations {
  project?: WorkflowProject;
  json_path?: string;
  images_path?: string;
  include_images: boolean;
  annotations: {
    destination_id: string | null;
    path: string | null;
    run_path?: string | null;
    enabled: boolean;
  };
  models: {
    destination_id: string | null;
    path: string | null;
    run_path?: string | null;
    enabled: boolean;
  };
  evaluations: {
    destination_id: string | null;
    path: string | null;
    run_path?: string | null;
    enabled: boolean;
  };
}

export interface ProfileClassRule {
  class_id: string;
  name: string;
  definition?: string;
  aliases?: string[];
  positive_examples?: string[];
  negative_examples?: string[];
  color?: string;
}

export type SegmentationSource = "vision_llm" | "sam";

export interface TaskProfileConfig {
  task_type: TaskType;
  geometry: "bbox" | "mask" | "obb" | "classify";
  class_rules: ProfileClassRule[];
  rules?: Record<string, unknown>;
  models: {
    proposer?: string;
    reviewer?: string;
    segmenter?: string;
    training?: string;
    inference?: string;
  };
  image_strategy: Record<string, unknown>;
  training: Record<string, unknown>;
  quality_conditions: Record<string, number>;
  repair_limit: number;
  max_requests: number;
  export_format: "yolo" | "coco" | "labelme";
  outputs?: {
    destination_id?: string;
    annotations_destination_id?: string;
    models_destination_id?: string;
    evaluations_destination_id?: string;
    annotations?: boolean;
    models?: boolean;
    evaluations?: boolean;
    include_images?: boolean;
  };
}

export interface TaskProfile {
  id: string;
  key: string;
  name: string;
  version: number;
  config: TaskProfileConfig;
  created_at?: string;
}

export interface WorkflowNode {
  id: string;
  type: string;
  version: number;
  scope: "item" | "batch";
  params: Record<string, unknown>;
  inputs?: Record<string, string>;
  routes?: Record<string, string>;
  task_types?: TaskType[];
}

export interface WorkflowTemplate {
  id: string;
  key: string;
  name: string;
  version: number;
  nodes: WorkflowNode[];
}

export interface ReviewObject {
  object_id: string;
  class_id: string | null;
  type: "bbox" | "polygon" | "obb" | "classify";
  bbox?: BBox;
  polygon?: [number, number][];
  obb?: OBB;
  mask?: MaskRle;
  mask_ref?: string;
  source?: "manual" | "imported" | "sam" | "auto";
  decision?: "accepted" | "rejected";
  provenance?: Record<string, unknown>;
  provenance_json?: Record<string, unknown>;
}

export interface ReviewIssue {
  id?: string;
  type?: string;
  code?: string;
  message: string;
  evidence?: string;
  coordinate_details?: Record<string, unknown>;
  bbox?: BBox;
  object_id?: string | null;
  region?: BBox;
  severity?: string;
  blocking?: boolean;
  resolution?: string;
  reason?: string;
}

export interface ReviewPayload {
  image_id: string;
  dataset_id: string;
  image_url: string;
  width: number;
  height: number;
  base_revision: number;
  objects: ReviewObject[];
  issues: ReviewIssue[];
  roi?: MaskRle | null;
  ignore?: MaskRle | null;
  [key: string]: unknown;
}

export interface WorkflowReview {
  id: string;
  run_id: string;
  item_id: string;
  status: string;
  revision: number;
  base_revision: number;
  payload: ReviewPayload;
  decision?: string | null;
  reason?: string | null;
  project?: WorkflowProject;
  export_locations?: { json_path: string; images_path: string };
}

export interface WorkflowItem {
  id: string;
  status: string;
  file_name?: string | null;
  cursor?: number;
  subject_ref?: string;
  image_id?: string;
  error_message?: string | null;
  [key: string]: unknown;
}

export interface WorkflowAttempt {
  id: string;
  node_id: string;
  status: string;
  created_at?: string;
  finished_at?: string | null;
  item_id?: string | null;
  error_message?: string | null;
  [key: string]: unknown;
}

export interface WorkflowArtifact {
  id: string;
  type?: string;
  kind?: string;
  [key: string]: unknown;
}

export interface WorkflowProgress {
  measured_at: string;
  automatic_applicable: boolean;
  total_images: number;
  ready_images: number;
  waiting_review_images: number;
  failed_images: number;
  excluded_images: number;
  confirmed_images: number;
  completed_steps: number;
  total_steps: number;
  percent: number | null;
  processing_seconds: number;
  sample_count: number;
  average_image_seconds: number | null;
  eta_seconds: number | null;
  eta_state: "not_applicable" | "cancelled" | "complete" | "blocked" | "waiting_review" | "learning" | "overdue" | "estimated";
  current: { node_id: string; node_type: string; item_id: string | null; image_index: number | null; file_name: string | null; elapsed_seconds: number } | null;
}

export interface WorkflowRun {
  id: string;
  template_id?: string;
  profile_id?: string;
  template_name?: string;
  profile_name?: string;
  status: string;
  deleted_at?: string | null;
  can_delete?: boolean;
  error_message?: string | null;
  profile_snapshot: TaskProfileConfig | TaskProfile;
  template_snapshot: WorkflowTemplate;
  context: Record<string, unknown> & {
    output_destinations?: WorkflowOutputDestinations;
  };
  items: WorkflowItem[];
  attempts: WorkflowAttempt[];
  artifacts: WorkflowArtifact[];
  reviews: Array<Pick<WorkflowReview, "id" | "item_id" | "status" | "revision">>;
  progress?: WorkflowProgress;
  events: Array<Record<string, unknown>>;
  batch_counts: Record<string, number>;
  created_at?: string;
  request_count?: number;
  reserved_requests?: number;
  parent_run_id?: string | null;
}

export function runProfile(run: WorkflowRun): TaskProfileConfig {
  return "config" in run.profile_snapshot
    ? run.profile_snapshot.config
    : run.profile_snapshot;
}
