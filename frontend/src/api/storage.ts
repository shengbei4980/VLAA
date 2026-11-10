import { request } from "./client";

export interface OutputDestination {
  id: string;
  name: string;
  path: string;
  created_at?: string;
}
export interface StorageSettings {
  current: { data_dir: string; database_path: string; source: string };
  pending: { data_dir: string; database_path: string } | null;
  restart_required: boolean;
  write_locked: boolean;
  environment_overrides: { data_dir: boolean; database_url: boolean };
  destinations: OutputDestination[];
  migration: {
    status: string;
    phase: string;
    error?: string | null;
    counts?: Record<string, number>;
  } | null;
}
export interface DataRootPlan {
  plan_id: string;
  source_dir: string;
  target_dir: string;
  database_path: string;
  required_bytes: number;
  free_bytes: number;
  file_count: number;
  restart_required: boolean;
}
export interface DestinationValidation {
  resolved_path: string;
  writable: boolean;
  free_bytes: number;
}
export interface ManualExportLocations {
  json_path: string;
  images_path: string;
}
const post = <T>(path: string, payload: unknown) =>
  request<T>(`/api/storage${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const storageApi = {
  manualExports: () =>
    request<ManualExportLocations>("/api/storage/manual-exports"),
  saveManualExports: (paths: ManualExportLocations) =>
    request<ManualExportLocations>("/api/storage/manual-exports", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(paths),
    }),
  settings: () => request<StorageSettings>("/api/storage/settings"),
  destinations: () => request<OutputDestination[]>("/api/storage/destinations"),
  validateDestination: (path: string) =>
    post<DestinationValidation>("/destinations/validate", { path }),
  registerDestination: (name: string, path: string) =>
    post<OutputDestination>("/destinations", { name, path }),
  prepareDataRoot: (path: string) =>
    post<DataRootPlan>("/data-root/prepare", { path }),
  executeDataRoot: (planId: string) =>
    post<StorageSettings>("/data-root/execute", { plan_id: planId }),
  cancelDataRoot: () => post<StorageSettings>("/data-root/cancel", {}),
};
