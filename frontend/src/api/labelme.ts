import { request } from "./client";
export interface LabelMePreview {
  image_count: number;
  shape_count: number;
  labels: Array<{ name: string; count: number }>;
  images: Array<{
    json_file: string;
    image_file: string;
    width: number;
    height: number;
    split: string;
    source_group_id: string;
    shape_count: number;
  }>;
  issues: Array<{ message?: string; level?: string; [key: string]: unknown }>;
  mapping_required: boolean;
}
function form(file: File, task: "detect" | "segment", split: string) {
  const payload = new FormData();
  payload.append("file", file);
  payload.append("task_type", task);
  payload.append("default_split", split);
  return payload;
}
export const labelmeApi = {
  preview: (file: File, task: "detect" | "segment", split: string) =>
    request<LabelMePreview>("/api/datasets/import/labelme/preview", {
      method: "POST",
      body: form(file, task, split),
    }),
  import: (
    file: File,
    task: "detect" | "segment",
    split: string,
    name: string,
    mapping: Record<string, string | null>,
  ) => {
    const payload = form(file, task, split);
    payload.append("name", name);
    payload.append("class_mapping", JSON.stringify(mapping));
    return request<{
      dataset_id: string;
      imported_images: number;
      imported_annotations: number;
      warnings: unknown[];
    }>("/api/datasets/import/labelme", { method: "POST", body: payload });
  },
};
