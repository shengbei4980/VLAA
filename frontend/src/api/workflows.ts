import { apiUrl, request } from "./client";
import type {
  ReviewPayload,
  TaskProfile,
  TaskProfileConfig,
  WorkflowReview,
  WorkflowRun,
  WorkflowTemplate,
  WorkflowNode,
  WorkflowProject,
  WorkflowOutputDestinations,
} from "../types/workflows";

export interface WorkflowRunRequest {
  idempotency_key: string;
  template_id: string;
  profile_id: string;
  dataset_id: string;
  project_id?: string;
  image_ids: string[];
  snapshot_id?: string;
  next_batches: Array<{ dataset_id: string; image_ids: string[] }>;
  inputs?: Record<string, unknown>;
}

export interface SavedWorkflowPreview {
  file_name: string;
  saved_path: string;
  download_url: string;
  width: number;
  height: number;
  review_revision: number;
  preview_state: "draft" | "confirmed";
}

export interface SavedLabelMeExport {
  file_name: string;
  saved_path: string;
  download_url: string;
  review_revision: number;
  export_state: "draft" | "confirmed";
  exported_objects: number;
  pending_objects: number;
}

export interface BatchAnnotationExport {
  file_name: string;
  saved_path: string;
  download_url: string;
  total_images: number;
  exported_images: number;
  pending_images: number;
  failed_images: number;
  skipped_images: number;
  results: Array<{
    image_id: string;
    review_id?: string;
    status: "exported" | "failed" | "skipped";
    export_state?: "draft" | "confirmed";
    json_path?: string;
    image_path?: string;
    error?: string;
    reason?: string;
  }>;
}
export interface PreparedWorkflowRun {
  request: WorkflowRunRequest;
  vision_approvals: Record<
    string,
    { mode: string; api_base: string; model: string; max_requests: number }
  >;
  scope: Array<{
    dataset_id: string;
    image_ids: string[];
    image_hashes: string[];
  }>;
  profile_snapshot: TaskProfileConfig;
  sam_config?: { model: string; device?: string; enabled?: boolean; weights_available?: boolean } | null;
  template_snapshot: WorkflowTemplate;
  outputs?: WorkflowOutputDestinations;
}

export async function workflowRequest<T>(
  path: string,
  body?: unknown,
  method = "POST",
): Promise<T> {
  return request<T>(
    `/api/workflows${path}`,
    body === undefined
      ? undefined
      : {
          method,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
}

export const workflowApi = {
  projects: () => workflowRequest<WorkflowProject[]>("/projects"),
  saveProject: (body: { name: string; root_path: string }) =>
    workflowRequest<WorkflowProject>("/projects", body),
  exportUrl: (id: string) =>
    apiUrl(`/api/workflows/runs/${encodeURIComponent(id)}/export`),
  profiles: () => workflowRequest<TaskProfile[]>("/profiles"),
  saveProfile: (body: {
    key: string;
    name: string;
    config: TaskProfileConfig;
  }) => workflowRequest<TaskProfile>("/profiles", body),
  templates: () => workflowRequest<WorkflowTemplate[]>("/templates"),
  saveTemplate: (body: { key: string; name: string; nodes: WorkflowNode[] }) =>
    workflowRequest<WorkflowTemplate>("/templates", body),
  nodes: () => workflowRequest<Array<Record<string, unknown>>>("/nodes"),
  runs: (deleted = false) =>
    workflowRequest<WorkflowRun[]>(deleted ? "/runs?deleted=true" : "/runs"),
  deleteRun: (id: string) =>
    workflowRequest<WorkflowRun>(`/runs/${encodeURIComponent(id)}`, {}, "DELETE"),
  restoreRun: (id: string) =>
    workflowRequest<WorkflowRun>(`/runs/${encodeURIComponent(id)}/restore`, {}),
  run: (id: string) =>
    workflowRequest<WorkflowRun>(`/runs/${encodeURIComponent(id)}?summary=true`),
  prepareRun: (body: WorkflowRunRequest) =>
    workflowRequest<PreparedWorkflowRun>("/runs/prepare", body),
  createRun: (body: WorkflowRunRequest) =>
    workflowRequest<WorkflowRun>("/runs", body),
  exportBatch: (id: string, includePreviews = true) =>
    workflowRequest<BatchAnnotationExport>(
      `/runs/${encodeURIComponent(id)}/annotation-batch`,
      { include_previews: includePreviews },
    ),
  controlRun: (
    id: string,
    action: "cancel" | "resume" | "retry",
    itemIds?: string[],
  ) =>
    workflowRequest<WorkflowRun>(
      `/runs/${encodeURIComponent(id)}/${action}`,
      itemIds ? { item_ids: itemIds } : {},
    ),
  review: (id: string) =>
    workflowRequest<WorkflowReview>(`/reviews/${encodeURIComponent(id)}`),
  continueEditing: (id: string, revision: number) =>
    workflowRequest<WorkflowReview>(
      `/reviews/${encodeURIComponent(id)}/continue-editing`,
      { expected_revision: revision },
    ),
  exportLabelMe: (id: string, revision: number, payload: ReviewPayload) =>
    workflowRequest<SavedLabelMeExport>(
      `/reviews/${encodeURIComponent(id)}/labelme`,
      {
        expected_revision: revision,
        payload,
      },
    ),
  savePreview: (
    id: string,
    revision: number,
    image: Blob,
    hasUnsavedChanges: boolean,
  ) => {
    const form = new FormData();
    form.append("file", image, "annotation-preview.png");
    form.append("expected_revision", String(revision));
    form.append("has_unsaved_changes", String(hasUnsavedChanges));
    return request<SavedWorkflowPreview>(
      `/api/workflows/reviews/${encodeURIComponent(id)}/preview`,
      {
        method: "POST",
        body: form,
      },
    );
  },
  saveDraft: (id: string, revision: number, payload: ReviewPayload) =>
    workflowRequest<WorkflowReview>(
      `/reviews/${encodeURIComponent(id)}/draft`,
      { expected_revision: revision, payload },
      "PUT",
    ),
  decide: (
    id: string,
    revision: number,
    decision: "approved" | "background" | "rejected" | "excluded",
    reason: string,
    semanticResolutions: Record<string, unknown> = {},
  ) =>
    workflowRequest<WorkflowReview>(
      `/reviews/${encodeURIComponent(id)}/decide`,
      {
        expected_revision: revision,
        decision,
        reason,
        semantic_resolutions: semanticResolutions,
        actor: "local-user",
      },
    ),
};
