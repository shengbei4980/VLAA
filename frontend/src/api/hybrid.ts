import { request } from "./client";

export interface HybridStatus {
  rf_ready: boolean;
  rf_detail: string;
  sam_ready: boolean;
  sam_detail: string;
}

export interface HybridResult {
  labelme: { imagePath: string; shapes: unknown[]; [key: string]: unknown };
  preview_png: string;
  source_png: string;
  stats: Record<"gpt_pixels" | "sam_pixels" | "rf_pixels" | "conflict_pixels" | "eligible_pixels", number>;
  warnings: string[];
}

export const hybridApi = {
  status: () => request<HybridStatus>("/api/hybrid/status"),
  run: (body: FormData) => request<HybridResult>("/api/hybrid/run", {
    method: "POST", body,
    headers: body.get("mode") === "online" ? { "X-VLAA-Hybrid-Run": "1" } : undefined,
  }),
};

export function buildHybridPayload(inputs: FormData, mode: "drafts" | "online") {
  const payload = new FormData();
  for (const name of ["rgb", "raster", "roi", "x", "y"]) {
    const value = inputs.get(name);
    if (value !== null && !(value instanceof File && !value.size)) payload.append(name, value);
  }
  payload.set("mode", mode);
  if (mode === "drafts") {
    for (const key of ["gpt_draft", "sam_draft"]) payload.set(key, inputs.get(key)!);
    return payload;
  }
  payload.set("allow_remote", String(inputs.get("allow_remote") === "on"));
  for (const role of ["gpt", "deepseek"]) {
    const field = (name: string) => String(inputs.get(role + "_" + name) ?? "").trim();
    const source = field("source");
    const config = source === "custom" ? {
      api_base: field("url"), model: field("model"), api_key: field("key"),
      coordinate_format: field("coords"), protocol: field("protocol"),
    } : { saved_role: source, protocol: field("protocol") };
    payload.set(role + "_endpoint", JSON.stringify(config));
  }
  return payload;
}
